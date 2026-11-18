// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

#pragma once

#include "bf16_plan.hpp"
#include "bf16_tiling.hpp"

#include <acl/acl.h>
#include <ATen/MemoryOverlap.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <limits>

#include <tiling/platform/platform_ascendc.h>

#include "shmem.h"

#include "check.hpp"
#include "bf16_launcher.hpp"
#include "kernels/mega_moe/timer/ascend_timer.hpp"
#include <vector>
#include "kernels/mega_moe/common/mega_moe_bf16_tiling.hpp"
#include "kernels/mega_moe/common/mega_moe_bf16_layout.hpp"
#include "kernels/mega_moe/common/mega_moe_workspace_backward.hpp"
#include "kernels/mega_moe/common/mega_moe_bf16_saved_meta.hpp"
#include "kernels/mega_moe/common/mega_moe_bf16_wave_plan.hpp"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace {

constexpr std::size_t kDeviceLargePageBytes = 2 * 1024 * 1024;
constexpr int64_t kMegaMoeContextBytes = static_cast<int64_t>(sizeof(uint32_t) * 2 + sizeof(uint64_t) * 1025);

constexpr std::size_t kBackwardTopkPartialUbBytes = 128ULL * 1024ULL;
constexpr std::size_t kA5Fp32PerVectorRegister = 64ULL;

void CheckNpuContiguous(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

uint64_t ForwardTransientWorkspaceBytes(const MegaMoeBf16TilingData& tiling)
{
    using namespace MegaMoeBF16Training;
    const ForwardTransientWorkspaceLayout transient(tiling.n, tiling.expertPerRank, tiling.maxOutputSize);
    const auto saved = tiling.savedGateUpMode;
    const Wave::ForwardWorkspaceLayout layout(transient.totalBytes, tiling.maxOutputSize, tiling.expertPerRank,
                                              tiling.k, tiling.n / 2U, saved == kSaveBf16);
    const uint64_t bytes = layout.totalBytes;
    TORCH_CHECK(bytes <= uint64_t(std::numeric_limits<int64_t>::max()), "forward workspace exceeds int64 capacity");
    return bytes;
}

torch::Tensor AllocateForwardTransientWorkspace(const torch::Tensor& like, const MegaMoeBf16TilingData& tiling)
{
    return torch::empty({static_cast<int64_t>(ForwardTransientWorkspaceBytes(tiling))},
                        like.options().dtype(torch::kUInt8));
}

struct Bf16WeightLayoutInfo {
    int64_t n;
    bool transW;
};

Bf16WeightLayoutInfo ResolveBf16WeightLayout(const torch::Tensor& weight1, const torch::Tensor& weight2, int64_t k)
{
    const int64_t expertPerRank = weight1.size(0);
    const bool canonical = weight1.size(2) == k && weight2.size(0) == expertPerRank && weight2.size(1) == k &&
                           weight1.size(1) % 2 == 0 && weight2.size(2) == weight1.size(1) / 2;
    if (canonical) {
        return {weight1.size(1), true};
    }

    const bool legacy = weight1.size(1) == k && weight2.size(0) == expertPerRank && weight2.size(2) == k &&
                        weight1.size(2) % 2 == 0 && weight2.size(1) == weight1.size(2) / 2;
    TORCH_CHECK(legacy,
                "MegaMoe BF16 weights must use either canonical layout "
                "weight1 [E, N, K], weight2 [E, K, N / 2], or legacy Megatron "
                "layout weight1 [E, K, N], weight2 [E, N / 2, K]");
    return {weight1.size(2), false};
}

torch::Tensor AllocateTrainingSavedState(const MegaMoeBF16Training::TrainingSavedStateLayout& savedStateLayout,
                                         const torch::Tensor& like)
{
    // The compact ordinary-BF16 layout begins at cumsumOffset=0. Embedded
    // layouts still skip their ptrA payload here. Keep only
    // the metadata/control portion zero-initialized; gate/up planes are written
    // by every valid forward tile.
    const uint64_t initOffset = savedStateLayout.cumsumOffset;
    const uint64_t initBytes = savedStateLayout.legacyBytes - initOffset;
    TORCH_CHECK(savedStateLayout.ptrAOffset <= initOffset && initOffset <= savedStateLayout.legacyBytes &&
                    savedStateLayout.legacyBytes <= savedStateLayout.totalBytes &&
                    savedStateLayout.totalBytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "invalid MegaMoe BF16 training saved-state initialization range");

    auto savedState =
        torch::empty({static_cast<int64_t>(savedStateLayout.totalBytes)}, like.options().dtype(torch::kUInt8));
    savedState.narrow(0, static_cast<int64_t>(initOffset), static_cast<int64_t>(initBytes)).zero_();
    return savedState;
}

std::size_t MegaMoeWindowControlRegionBytes(int64_t worldSize, int64_t expertPerRank, bool compactTokenCountExchange)
{
    // Clear the optional compact inbox and the unchanged dense token table.
    // Forward/backward payload starts at payloadOffset and is never cleared.
    const MegaMoeBF16Training::CompactTokenCountExchangeLayout countLayout(
        static_cast<uint64_t>(worldSize), static_cast<uint64_t>(expertPerRank), compactTokenCountExchange);
    return static_cast<std::size_t>(
        MegaMoeBF16Training::AlignUp(countLayout.payloadOffset, MegaMoeBF16Training::kAlignment512));
}

void ClearMegaMoeWindowControlRegion(void* symmetricBuffer, std::size_t symmetricBufferBytes, int64_t worldSize,
                                     int64_t expertPerRank, bool compactTokenCountExchange)
{
    const std::size_t controlBytes =
        MegaMoeWindowControlRegionBytes(worldSize, expertPerRank, compactTokenCountExchange);
    const std::size_t clearOffset = static_cast<std::size_t>(MegaMoeBF16Training::PeerDataOffset(worldSize));
    TORCH_CHECK(controlBytes <= symmetricBufferBytes,
                "MegaMoe BF16 SHMEM control region exceeds the symmetric buffer: need ", controlBytes, " bytes, got ",
                symmetricBufferBytes);
    TORCH_CHECK(clearOffset <= controlBytes, "invalid MegaMoe BF16 SHMEM control clear range");
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    ASCEND_DEEPEP_CHECK_ACL(aclrtMemsetAsync(static_cast<uint8_t*>(symmetricBuffer) + clearOffset,
                                             symmetricBufferBytes - clearOffset, 0, controlBytes - clearOffset,
                                             stream));
}

constexpr std::size_t Bf16Align(std::size_t value, std::size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

using MegaMoeBF16BackwardWorkspaceLayout = MegaMoeBF16Training::BackwardWorkspaceLayout;

torch::Tensor AllocateBackwardWorkspace(const MegaMoeBF16BackwardWorkspaceLayout& layout, const torch::Tensor& like)
{
    // dO, recomputed gate/up, dZ, Hp and dP partials are payload regions.
    // Their valid packed rows are fully overwritten before any consumer runs.
    // Atomic dispatch/ready/done counters and the per-producer progress suffix require zero.
    TORCH_CHECK(layout.controlOffset <= layout.totalBytes &&
                    layout.totalBytes <= static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
                "invalid MegaMoe BF16 backward workspace initialization range");
    auto workspace = torch::empty({static_cast<int64_t>(layout.totalBytes)}, like.options().dtype(torch::kUInt8));
    workspace
        .narrow(0, static_cast<int64_t>(layout.controlOffset),
                static_cast<int64_t>(layout.totalBytes - layout.controlOffset))
        .zero_();
    return workspace;
}

torch::Tensor AllocateTrainingTimer(const torch::Tensor& like, const torch::Tensor& weights,
                                    const torch::Tensor& weight1, const torch::Tensor& weight2, int64_t maxRecv,
                                    int phase, const std::string& mode)
{
#ifndef DEEPEP_MEGAMOE_TIMER
    TORCH_CHECK(false, "Rebuild with DEEPEP_MEGAMOE_TIMER=ON before capturing training timers");
#endif
    using namespace MegaMoeTrainTimer;
    const int64_t h = like.size(1), e = weight1.size(0);
    const auto w = ResolveBf16WeightLayout(weight1, weight2, h);
    const int64_t f = w.n / 2;
    const int64_t groups = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    TORCH_CHECK(groups > 0 && groups <= kMaxCoreGroups, "unsupported Timer AIC count");
    auto ceil = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
    const int64_t rows = phase == 0 ? 192 : 128;
    const int64_t tiles = ceil(maxRecv + e * (rows - 1), rows) * ceil(std::max(h, 2 * f), 256);
    const int64_t dwTiles = e * std::max(ceil(2 * f, 128) * ceil(h, 256), ceil(h, 128) * ceil(f, 256));
    const int64_t bound = ceil(std::max(tiles, phase == 0 ? int64_t(0) : dwTiles), groups) + 4 * e + 32;
    const int capacity = static_cast<int>(ceil(std::max(bound, int64_t(128)), 64) * 64);
    TORCH_CHECK(capacity <= kMaxCapacity, "training Timer task capacity exceeds supported bound");
    const Layout layout{static_cast<int>(groups * 3), capacity};
    TORCH_CHECK(layout.Elements() * 8 <= 256LL * 1024 * 1024, "training Timer exceeds 256 MiB per phase");
    auto timer = torch::zeros({layout.Elements()}, like.options().dtype(torch::kInt64));
    const int64_t saved = mode == "none" ? 0 : (mode == "bf16" ? 1 : 2);
    const int64_t flags = saved | (MegaMoeBF16SwigluDecoupled(h, saved) ? 256 : 0) | (w.transW ? 1024 : 0);
    std::vector<int64_t> values{kMagic,
                                kVersion,
                                groups * 3,
                                capacity,
                                phase,
                                FIXED_TIMING_COUNT,
                                DYNAMIC_TYPE_COUNT,
                                kMetadataWords,
                                maxRecv,
                                like.size(0),
                                h,
                                f,
                                e,
                                weights.size(1),
                                weights.scalar_type() == torch::kFloat32 ? 1 : 0,
                                flags};
    auto header = torch::tensor(values, torch::TensorOptions().dtype(torch::kInt64));
    timer.narrow(0, 0, kHeaderWords).copy_(header);
    return timer;
}

}  // namespace
