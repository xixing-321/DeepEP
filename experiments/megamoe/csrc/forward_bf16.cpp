// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

#include "forward.hpp"
#include "kernel_launch.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>

#include <tiling/platform/platform_ascendc.h>
#include "kernels/mega_moe/common/mega_moe_bf16_layout.hpp"
#include "kernels/mega_moe/common/mega_moe_bf16_saved_meta.hpp"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include <ATen/MemoryOverlap.h>
#include <vector>

#include "check.hpp"
#include "kernels/mega_moe/timer/ascend_timer.hpp"
#include "kernels/mega_moe/common/mega_moe_bf16_wave_plan.hpp"

namespace {

template <typename T, typename U>
constexpr auto HostCeilDiv(T value, U divisor)
{
    return (value + divisor - 1) / divisor;
}

template <typename T, typename U>
constexpr auto HostCeilAlign(T value, U alignment)
{
    return HostCeilDiv(value, alignment) * alignment;
}

constexpr int64_t ALIGN_128 = 128;
constexpr int64_t ALIGN_256 = 256;
constexpr int64_t ALIGN_512 = 512;
constexpr int64_t INT_CACHELINE = 16;
constexpr int64_t SWIGLU_N_HALF = 2;
constexpr int64_t SIZE_BF_16 = 2;
constexpr int64_t SIZE_INT_32 = 4;

using namespace AscendC;

namespace optiling {

enum WorkspaceIdx {
    PTR_A2 = 0,
    PTR_A2_SCALE,
    COMM0,
    COMM1,
    COMM2,
    COMM3,
    COMM4,
    EXPANDED_ROW_IDX,
    EXPANDED_ROW_IDX_GATHER,
    FLAG_SWIGLU_TO_GMM2,

    WORKSPACE_NUM
};

struct WorkspaceInfo {
    size_t totalSize = 0;
    size_t offset[WORKSPACE_NUM] = {0};
};

WorkspaceInfo CalcWorkspaceSize(const MegaMoeBf16TilingData& td)
{
    WorkspaceInfo ws;
    ws.totalSize += HostCeilAlign(SIZE_BF_16 * static_cast<int64_t>(td.maxOutputSize) * td.k, ALIGN_512);
    ws.totalSize +=
        HostCeilAlign(SIZE_BF_16 * static_cast<int64_t>(td.maxOutputSize) * td.n / SWIGLU_N_HALF, ALIGN_512);
    ws.totalSize += HostCeilAlign(static_cast<int64_t>(td.epWorldSize) * td.expertPerRank * SIZE_INT_32, ALIGN_512);
    ws.totalSize += SIZE_INT_32 * HostCeilAlign(static_cast<int64_t>(td.m), ALIGN_256) * td.topK * 2;
    ws.totalSize += SIZE_INT_32 * HostCeilAlign(static_cast<int64_t>(td.epWorldSize) * td.expertPerRank, ALIGN_128);
    ws.totalSize += HostCeilAlign(static_cast<int64_t>(SIZE_INT_32) * td.expertPerRank * INT_CACHELINE, ALIGN_128);
    return ws;
}

inline size_t CalcBf16EarlyTopkWeightWorkspaceSize(const MegaMoeBf16TilingData& td)
{
    return HostCeilAlign(
        static_cast<int64_t>(td.maxOutputSize) * MegaMoeBF16Training::TopkWeightBytes(td.topkWeightType), ALIGN_512);
}

}  // namespace optiling

namespace {

bool ResolveCompactTokenCountExchange(int64_t worldSize, int64_t expertPerRank)
{
    const char* rawMode = std::getenv("MEGAMOE_BF16_TOKEN_COUNT_EXCHANGE");
    const std::string mode = rawMode == nullptr || rawMode[0] == '\0' ? "auto" : rawMode;
    if (mode == "auto") {
        return MegaMoeBF16Training::UseCompactTokenCountExchangeByDefault(static_cast<uint64_t>(worldSize),
                                                                          static_cast<uint64_t>(expertPerRank));
    }
    if (mode == "dense") {
        return false;
    }
    TORCH_CHECK(mode == "compact", "MEGAMOE_BF16_TOKEN_COUNT_EXCHANGE must be auto, dense or compact; got '", mode,
                "'");
    TORCH_CHECK(MegaMoeBF16Training::CanUseCompactTokenCountExchange(static_cast<uint64_t>(worldSize),
                                                                     static_cast<uint64_t>(expertPerRank)),
                "forced compact BF16 token-count exchange requires 2..1024 EP ranks, "
                "1..8 local experts");
    return true;
}

}  // namespace

struct BufferSizes {
    uint64_t workspace_bytes;
    uint64_t peer_bytes;
};

uint32_t SaveMode(const std::string& mode)
{
    using namespace MegaMoeBF16Training;
    if (mode == "none") return kSaveNone;
    if (mode == "mxfp8") return kSaveMxFp8;
    TORCH_CHECK(mode == "bf16", "gmm1_save_type must be none, bf16 or mxfp8");
    return kSaveBf16;
}

float ClampLimit(double limit)
{
    return limit > 0.0 ? static_cast<float>(limit) : kMegaMoeSwigluLimitDisabled;
}

BufferSizes MakeTiling(MegaMoeBf16TilingData& tiling, int64_t m, int64_t k, int64_t n, int64_t topk, int64_t world_size,
                       int64_t expert_per_rank, int64_t max_recv_tokens, int rank, double swiglu_limit,
                       const std::string& save_gmm1_output, uint32_t topk_weight_type)
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t aic_num = platform->GetCoreNumAic();
    const uint32_t aiv_num = platform->GetCoreNumAiv();
    uint64_t ub_size = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    TORCH_CHECK(aic_num > 0 && aiv_num > 0 && ub_size > 0, "failed to query AscendC platform resources");
    {
        const MegaMoeBF16Training::ForwardDispatchUbLayout dispatchUb(world_size, expert_per_rank, k, 2);
        TORCH_CHECK(dispatchUb.totalBytes <= ub_size, "MegaMoe BF16 Dispatch requires ", dispatchUb.totalBytes,
                    " UB bytes, but the device has ", ub_size);
        TORCH_CHECK(MegaMoeBF16Training::ForwardDispatchUbLayout::kCountResetChunkBytes <= ub_size,
                    "MegaMoe BF16 count reset chunk exceeds device UB capacity");
    }

    const float normalizedSwigluLimit = ClampLimit(swiglu_limit);
    using namespace MegaMoeBF16Training;
    const uint32_t saveMode = SaveMode(save_gmm1_output);
    if (saveMode == kSaveMxFp8) {
        const uint64_t quantUbBytes = SaveGateUpClipMasks(normalizedSwigluLimit, saveMode)
                                          ? SavedGateUpQuantUbLayout::totalBytesWithClipMasks
                                          : SavedGateUpQuantUbLayout::totalBytes;
        const uint64_t ubBudget = std::min<uint64_t>(ub_size, 248ULL * 1024ULL);
        TORCH_CHECK(quantUbBytes <= ubBudget, "MegaMoe BF16 saved MXFP8 epilogue requires ", quantUbBytes,
                    " UB bytes, have ", ubBudget);
        TORCH_CHECK((n / 2) % 32 == 0, "saved MXFP8 requires F aligned to 32");
    }

    const uint32_t blocks_per_ep = std::max<uint32_t>(1, aic_num / world_size);
    tiling.m = m;
    tiling.k = k;
    tiling.n = n;
    tiling.topK = topk;
    tiling.epWorldSize = world_size;
    tiling.expertPerRank = expert_per_rank;
    tiling.blockNumPerEP = blocks_per_ep;
    tiling.maxOutputSize = max_recv_tokens;
    tiling.rankId = rank;
    tiling.dispatchRows = 2;
    tiling.groupListType = 1;
    tiling.transW = tiling.transW2 = true;
    tiling.swigluLimit = normalizedSwigluLimit;
    tiling.savedGateUpMode = saveMode;
    TORCH_CHECK(topk_weight_type <= kTopkWeightFp32, "unsupported TopK weight type");
    tiling.topkWeightType = topk_weight_type;
    tiling.mGroupsPerWave = Wave::GroupsPerWave(aic_num, static_cast<uint32_t>(n / 2), static_cast<uint32_t>(k));
    TORCH_CHECK(aic_num <= Wave::kMaxCores, "BF16 forward supports at most ", Wave::kMaxCores, " AIC groups");
    TORCH_CHECK(Wave::ScratchFits(static_cast<uint32_t>(k), static_cast<uint32_t>(world_size),
                                  static_cast<uint32_t>(expert_per_rank)),
                "BF16 forward Dispatch/Combine buffers and routing metadata exceed the AIV1 UB budget");
    tiling.compactTokenCountExchange = ResolveCompactTokenCountExchange(world_size, expert_per_rank) ? 1U : 0U;
    const auto workspace_info = optiling::CalcWorkspaceSize(tiling);
    const CompactTokenCountExchangeLayout counts(world_size, expert_per_rank, tiling.compactTokenCountExchange != 0);
    const PeerPayloadLayout payload(counts.payloadOffset, m, k, topk, TopkWeightBytes(topk_weight_type));
    return {workspace_info.totalSize + optiling::CalcBf16EarlyTopkWeightWorkspaceSize(tiling), payload.totalBytes};
}

constexpr std::size_t kDeviceLargePageBytes = 2 * 1024 * 1024;

struct MegaMoeContext {
    uint32_t ep_rank_id;
    uint32_t rank_size_per_server;
    uint64_t kfc_context_addr;
    uint64_t ep_buffer[1024];
};
static_assert(sizeof(MegaMoeContext) == 8208, "MegaMoe context ABI changed");

void InitializeMegaMoeBF16Context(torch::Tensor context, void* symmetric_buffer, int rank, int world_size)
{
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");
    const std::size_t context_bytes = static_cast<std::size_t>(context.numel()) * context.element_size();
    TORCH_CHECK(context_bytes >= sizeof(MegaMoeContext), "MegaMoe BF16 context tensor is too small");
    TORCH_CHECK(world_size <= 1024, "MegaMoe BF16 supports at most 1024 EP ranks");
    MegaMoeContext host_context{};
    host_context.ep_rank_id = static_cast<uint32_t>(rank);
    host_context.rank_size_per_server = static_cast<uint32_t>(world_size);
    deepep::megamoe::FillPeerAddresses(host_context.ep_buffer, symmetric_buffer, world_size);
    const auto copy_status =
        aclrtMemcpy(context.data_ptr(), context_bytes, &host_context, sizeof(host_context), ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(copy_status == ACL_SUCCESS, "failed to initialize MegaMoe BF16 communication context, acl error ",
                copy_status);
}

uint64_t ForwardTransientWorkspaceBytes(const MegaMoeBf16TilingData& tiling)
{
    using namespace MegaMoeBF16Training;
    const Wave::ForwardScratch scratch(tiling.maxOutputSize, tiling.mGroupsPerWave, tiling.forwardSchedule);
    const ForwardTransientWorkspaceLayout transient(tiling.n, tiling.expertPerRank, scratch.hpRows);
    const auto expertWeightBytes = tiling.routeWeightBeforeGmm2 ? TopkWeightBytes(tiling.topkWeightType) : 0U;
    const Wave::ForwardWorkspaceLayout layout(transient.totalBytes, tiling.maxOutputSize, tiling.expertPerRank,
                                              tiling.k, expertWeightBytes, tiling.m, tiling.mGroupsPerWave,
                                              scratch.rows);
    const uint64_t bytes = layout.totalBytes;
    TORCH_CHECK(bytes <= uint64_t(std::numeric_limits<int64_t>::max()), "forward workspace exceeds int64 capacity");
    return bytes;
}

torch::Tensor AllocateForwardTransientWorkspace(const torch::Tensor& like, const MegaMoeBf16TilingData& tiling)
{
    return torch::empty({static_cast<int64_t>(ForwardTransientWorkspaceBytes(tiling))},
                        like.options().dtype(torch::kUInt8));
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

void LaunchMegaMoeBF16ForwardKernel(torch::Tensor context, torch::Tensor x, torch::Tensor topk_ids,
                                    torch::Tensor topk_weights, torch::Tensor weight1, torch::Tensor weight2,
                                    torch::Tensor y, torch::Tensor expert_token_nums,
                                    torch::Tensor training_saved_state, torch::Tensor dispatch_output,
                                    torch::Tensor forward_transient_workspace, const MegaMoeBf16TilingData& tiling,
                                    uint32_t block_dim, uint64_t ffts_addr, torch::Tensor timer)
{
    auto stream = c10_npu::getCurrentNPUStream().stream();
    auto ptr = [](const torch::Tensor& t) { return static_cast<uint8_t*>(t.data_ptr()); };
    if (timer.defined()) {
        auto kernel = &deepep_train_mega_moe_bf16_forward_timer_kernel_do;
        kernel(block_dim, stream, ptr(context), ptr(x), ptr(topk_ids), ptr(topk_weights), ptr(weight1), ptr(weight2),
               ptr(y), ptr(expert_token_nums), ptr(training_saved_state),
               dispatch_output.defined() ? ptr(dispatch_output) : nullptr, ptr(forward_transient_workspace),
               reinterpret_cast<const uint8_t*>(&tiling), ffts_addr, ptr(timer));
    } else {
        auto kernel = &deepep_train_mega_moe_bf16_forward_kernel_do;
        kernel(block_dim, stream, ptr(context), ptr(x), ptr(topk_ids), ptr(topk_weights), ptr(weight1), ptr(weight2),
               ptr(y), ptr(expert_token_nums), ptr(training_saved_state),
               dispatch_output.defined() ? ptr(dispatch_output) : nullptr, ptr(forward_transient_workspace),
               reinterpret_cast<const uint8_t*>(&tiling), ffts_addr);
    }
}

}  // namespace

namespace deepep::megamoe::bf16 {

BF16Plan::WeightLayout BF16Plan::ResolveWeightLayout(const torch::Tensor& weight1, const torch::Tensor& weight2,
                                                     int64_t k)
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

void BF16Plan::ClearWindowControl() const
{
    const auto worldSize = tiling_.epWorldSize;
    const auto expertPerRank = tiling_.expertPerRank;
    const bool compactTokenCountExchange = tiling_.compactTokenCountExchange != 0;
    const auto symmetricBufferBytes = SymmetricBufferBytes();
    const std::size_t controlBytes =
        MegaMoeWindowControlRegionBytes(worldSize, expertPerRank, compactTokenCountExchange);
    const std::size_t clearOffset = static_cast<std::size_t>(MegaMoeBF16Training::PeerDataOffset(worldSize));
    TORCH_CHECK(controlBytes <= symmetricBufferBytes,
                "MegaMoe BF16 SHMEM control region exceeds the symmetric buffer: need ", controlBytes, " bytes, got ",
                symmetricBufferBytes);
    TORCH_CHECK(clearOffset <= controlBytes, "invalid MegaMoe BF16 SHMEM control clear range");
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    ASCEND_DEEPEP_CHECK_ACL(aclrtMemsetAsync(static_cast<uint8_t*>(SymmetricBuffer()) + clearOffset,
                                             symmetricBufferBytes - clearOffset, 0, controlBytes - clearOffset,
                                             stream));
}

torch::Tensor BF16Plan::AllocateTimer(const torch::Tensor& like, const torch::Tensor& weights,
                                      const torch::Tensor& weight1, const torch::Tensor& weight2, int64_t maxRecv,
                                      int phase, const std::string& mode)
{
#ifndef DEEPEP_MEGAMOE_TIMER
    TORCH_CHECK(false, "Rebuild with DEEPEP_MEGAMOE_TIMER=ON before capturing training timers");
#endif
    using namespace MegaMoeTrainTimer;
    const int64_t h = like.size(1), e = weight1.size(0);
    const auto w = ResolveWeightLayout(weight1, weight2, h);
    const int64_t f = w.n / 2;
    const int64_t groups = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    TORCH_CHECK(groups > 0 && groups <= kMaxCoreGroups, "unsupported Timer AIC count");
    auto ceil = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
    const int64_t rows = phase == 0 ? 192 : 128;
    const int64_t tiles = ceil(maxRecv + e * (rows - 1), rows) * ceil(std::max(h, 2 * f), 256);
    const int64_t dwTiles = e * std::max(ceil(2 * f, 128) * ceil(h, 256), ceil(h, 128) * ceil(f, 256));
    const int64_t gemmTasks = ceil(std::max(tiles, phase == 0 ? int64_t(0) : dwTiles), groups);
    // Short dP sends from every peer/expert pair can all be owned by the first AIV.
    const int64_t sendTasks = phase == 0 ? 0 : int64_t(world_size_) * e;
    const int64_t bound = std::max(gemmTasks, sendTasks) + 4 * e + 32;
    const int capacity = static_cast<int>(ceil(std::max(bound, int64_t(128)), 64) * 64);
    TORCH_CHECK(capacity <= kMaxCapacity, "training Timer task capacity exceeds supported bound");
    const Layout layout{static_cast<int>(groups * 3), capacity};
    TORCH_CHECK(layout.Elements() * 8 <= 256LL * 1024 * 1024, "training Timer exceeds 256 MiB per phase");
    auto timer = torch::zeros({layout.Elements()}, like.options().dtype(torch::kInt64));
    const int64_t saved = mode == "none" ? 0 : (mode == "bf16" ? 1 : 2);
    const int64_t flags = saved | (deepep_train_mega_moe_bf16_swiglu_decoupled_for_saved_mode(h, saved) ? 256 : 0) |
                          (w.transW ? 1024 : 0);
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

BF16Plan::BF16Plan(uintptr_t address, int64_t bytes, int rank, int world_size, torch::Tensor like,
                   bool weight_before_gmm2)
    : address_(address), bytes_(bytes), rank_(rank), world_size_(world_size), weight_before_gmm2_(weight_before_gmm2)
{
    TORCH_CHECK(address != 0 && bytes > 0 && bytes % kDeviceLargePageBytes == 0, "invalid borrowed training window");
    TORCH_CHECK(rank >= 0 && rank < world_size && world_size <= 1024, "invalid training rank/world size");
    TORCH_CHECK(aclshmem_my_pe() == rank && aclshmem_n_pes() == world_size,
                "training identity disagrees with the root Session");
    ffts_addr_ = shmemx_get_ffts_config();
    (void)Context(like);
}

int64_t buffer_size(int64_t max_tokens, int64_t hidden, int64_t intermediate, int64_t topk, int64_t world,
                    int64_t local_experts, int64_t max_recv)
{
    TORCH_CHECK(max_tokens > 0 && max_tokens <= 16384 && max_recv > 0, "invalid BF16 token capacity");
    TORCH_CHECK(hidden > 0 && hidden % 256 == 0 && intermediate > 0 && intermediate % 256 == 0,
                "BF16 dimensions must be positive multiples of 256");
    TORCH_CHECK(world >= 2 && world <= 1024 && local_experts > 0 && local_experts <= 32 &&
                    world * local_experts <= 2048 && topk > 0 && topk <= 32 && topk <= world * local_experts,
                "invalid BF16 expert configuration");
    MegaMoeBf16TilingData tiling{};
    const auto host = MakeTiling(tiling, max_tokens, hidden, 2 * intermediate, topk, world, local_experts, max_recv, 0,
                                 0.0, "bf16", MegaMoeBF16Training::kTopkWeightFp32);
    const uint64_t total = host.peer_bytes;
    TORCH_CHECK(total <= uint64_t(std::numeric_limits<int64_t>::max()) - kDeviceLargePageBytes,
                "BF16 symmetric capacity overflow");
    return static_cast<int64_t>((total + kDeviceLargePageBytes - 1) / kDeviceLargePageBytes * kDeviceLargePageBytes);
}

void BF16Plan::prepare(int64_t m, int64_t h, int64_t f, int64_t topk, int64_t local_experts, int64_t max_recv,
                       double limit, const std::string& save_type, bool fp32_p)
{
    TORCH_CHECK(address_ != 0, "BF16 Plan has been closed");
    MegaMoeBf16TilingData tiling{};
    static_assert(DEEPEP_MEGAMOE_BF16_FORWARD_SCHEDULE == 0 || DEEPEP_MEGAMOE_BF16_FORWARD_SCHEDULE == 1,
                  "BF16 training forward schedule must be 0 (interleaved) or 1 (staged)");
    tiling.routeWeightBeforeGmm2 = weight_before_gmm2_ ? 1U : 0U;
    tiling.forwardSchedule = weight_before_gmm2_ ? DEEPEP_MEGAMOE_BF16_FORWARD_SCHEDULE : 0U;
    const auto host = MakeTiling(tiling, m, h, 2 * f, topk, world_size_, local_experts, max_recv, rank_, limit,
                                 weight_before_gmm2_ ? save_type : "none",
                                 fp32_p ? MegaMoeBF16Training::kTopkWeightFp32 : MegaMoeBF16Training::kTopkWeightBf16);
    TORCH_CHECK(host.peer_bytes <= SymmetricBufferBytes(), "BF16 symmetric window is too small");
    auto workspace = AllocateForwardTransientWorkspace(context_, tiling);
    tiling_ = tiling;
    layout_workspace_bytes_ = host.workspace_bytes;
    workspace_ = std::move(workspace);
}

MegaMoeBf16TilingData BF16Plan::TilingForCall(int64_t m, int64_t k, int64_t n, int64_t topk, int64_t experts,
                                              int64_t max_recv, double limit, const std::string& mode,
                                              uint32_t weight_type, bool trans_w)
{
    if (!workspace_.defined() || tiling_.m != m || tiling_.k != k || tiling_.n != n || tiling_.topK != topk ||
        tiling_.expertPerRank != experts || tiling_.maxOutputSize != max_recv ||
        tiling_.topkWeightType != weight_type || tiling_.savedGateUpMode != SaveMode(mode) ||
        tiling_.swigluLimit != ClampLimit(limit)) {
        prepare(m, k, n / 2, topk, experts, max_recv, limit, mode, weight_type == MegaMoeBF16Training::kTopkWeightFp32);
    }
    auto tiling = tiling_;
    tiling.transW = tiling.transW2 = trans_w;
    return tiling;
}

void BF16Plan::close()
{
    address_ = 0;
    context_ = torch::Tensor();
    workspace_ = torch::Tensor();
}

ops::ForwardResult BF16Plan::forward_out(torch::Tensor out, torch::Tensor x, torch::Tensor topk_ids,
                                         torch::Tensor topk_weights, torch::Tensor weight1, torch::Tensor weight2,
                                         int64_t max_recv_tokens, double swiglu_limit,
                                         const std::string& save_gmm1_output, bool saveDispatchedX)
{
    TORCH_CHECK(address_ != 0, "training operator has been closed");
    TORCH_CHECK(out.defined(), "y must be a defined output tensor");
    return Forward(x, topk_ids, topk_weights, weight1, weight2, max_recv_tokens, swiglu_limit, save_gmm1_output, out,
                   {}, saveDispatchedX);
}

ops::ForwardResult BF16Plan::forward_with_timer(torch::Tensor x, torch::Tensor ids, torch::Tensor weights,
                                                torch::Tensor w1, torch::Tensor w2, int64_t maxRecv, double limit,
                                                const std::string& mode, c10::optional<torch::Tensor> out,
                                                bool saveDispatchedX)
{
    TORCH_CHECK(address_ != 0, "training operator has been closed");
    auto timer = AllocateTimer(x, weights, w1, w2, maxRecv, 0, mode);
    return Forward(x, ids, weights, w1, w2, maxRecv, limit, mode, out.value_or(torch::Tensor()), timer,
                   saveDispatchedX);
}

torch::Tensor BF16Plan::Context(const torch::Tensor& like)
{
    TORCH_CHECK(like.defined(), "MegaMoe BF16 context reference tensor must be defined");
    TORCH_CHECK(like.device().type() == c10::DeviceType::PrivateUse1,
                "MegaMoe BF16 context reference tensor must be an NPU tensor");
    TORCH_CHECK(like.is_contiguous(), "MegaMoe BF16 context reference tensor must be contiguous");
    TORCH_CHECK(like.scalar_type() == torch::kBFloat16, "MegaMoe BF16 context reference tensor must be bfloat16");
    const int rank = rank_;
    const int world_size = world_size_;
    if (!context_.defined()) {
        auto new_context =
            torch::empty({static_cast<int64_t>(sizeof(MegaMoeContext))}, like.options().dtype(torch::kUInt8));
        InitializeMegaMoeBF16Context(new_context, SymmetricBuffer(), rank, world_size);
        // Publish the persistent state only after initialization succeeds.
        // A failed first call must remain retryable instead of leaving a
        // defined but partially initialized context behind.
        context_ = new_context;
        context_rank_ = rank;
        context_world_size_ = world_size;
    } else {
        TORCH_CHECK(context_.device() == like.device(), "MegaMoe BF16 context device changed across calls");
        TORCH_CHECK(context_rank_ == rank && context_world_size_ == world_size,
                    "MegaMoe BF16 SHMEM rank/world size changed across calls");
    }
    return context_;
}

void* BF16Plan::SymmetricBuffer() const
{
    return reinterpret_cast<void*>(address_);
}

std::size_t BF16Plan::SymmetricBufferBytes() const
{
    return static_cast<std::size_t>(bytes_);
}

ops::ForwardResult BF16Plan::Forward(torch::Tensor x, torch::Tensor topkIds, torch::Tensor topkWeights,
                                     torch::Tensor weight1, torch::Tensor weight2, int64_t maxRecvTokens,
                                     double swigluLimit, const std::string& save_gmm1_output, torch::Tensor out,
                                     torch::Tensor timer, bool saveDispatchedX)
{
    auto context = Context(x);
    const int worldSize = world_size_;
    CheckNpuContiguous(x, "x");
    CheckNpuContiguous(topkIds, "topk_ids");
    CheckNpuContiguous(topkWeights, "topk_weights");
    CheckNpuContiguous(weight1, "weight1");
    CheckNpuContiguous(weight2, "weight2");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(topkIds.scalar_type() == torch::kInt32, "topk_ids must be int32");
    TORCH_CHECK(topkWeights.scalar_type() == torch::kBFloat16 || topkWeights.scalar_type() == torch::kFloat32,
                "topk_weights must be bfloat16 or float32");
    const uint32_t topkWeightType = topkWeights.scalar_type() == torch::kFloat32 ? MegaMoeBF16Training::kTopkWeightFp32
                                                                                 : MegaMoeBF16Training::kTopkWeightBf16;
    const auto topkWeightBytes = MegaMoeBF16Training::TopkWeightBytes(topkWeightType);
    TORCH_CHECK(weight1.scalar_type() == torch::kBFloat16, "weight1 must be bfloat16");
    TORCH_CHECK(weight2.scalar_type() == torch::kBFloat16, "weight2 must be bfloat16");
    TORCH_CHECK(x.dim() == 2 && topkIds.dim() == 2, "x and topk_ids must be rank-2 tensors");
    TORCH_CHECK(topkWeights.sizes() == topkIds.sizes(), "topk_weights shape must match topk_ids");
    TORCH_CHECK(x.size(0) == topkIds.size(0), "x and topk_ids must have the same token count");
    TORCH_CHECK(weight1.dim() == 3 && weight2.dim() == 3, "weight1 and weight2 must be rank-3 tensors");

    if (out.defined()) {
        CheckNpuContiguous(out, "y");
        TORCH_CHECK(out.scalar_type() == x.scalar_type() && out.device() == x.device() && out.sizes() == x.sizes(),
                    "y must have the same BF16 shape and NPU device as x");
        at::assert_no_internal_overlap(out);
        for (const auto* input : {&context, &x, &topkIds, &topkWeights, &weight1, &weight2}) {
            at::assert_no_overlap(out, *input);
        }
    }

    const int64_t m = x.size(0);
    const int64_t k = x.size(1);
    const int64_t topk = topkIds.size(1);
    const int64_t expertPerRank = weight1.size(0);
    const auto weightLayout = ResolveWeightLayout(weight1, weight2, k);
    const int64_t n = weightLayout.n;
    TORCH_CHECK(m > 0 && k > 0 && topk > 0 && expertPerRank > 0 && n > 0,
                "all MegaMoe BF16 dimensions must be positive");
    TORCH_CHECK(worldSize >= 2 && worldSize <= static_cast<int64_t>(MegaMoeBF16Training::kMaxEpRanks),
                "MegaMoe BF16 requires 2..1024 EP ranks");
    TORCH_CHECK(m <= 16384 && expertPerRank <= static_cast<int64_t>(MegaMoeBF16Training::kMaxLocalExperts) &&
                    expertPerRank * worldSize <= static_cast<int64_t>(MegaMoeBF16Training::kMaxTotalExperts),
                "MegaMoe BF16 forward shape exceeds supported routing limits "
                "(32 local, 2048 total experts)");
    TORCH_CHECK(topk <= expertPerRank * worldSize, "MegaMoe BF16 topk must not exceed the total expert count");
    TORCH_CHECK(maxRecvTokens > 0, "max_recv_tokens must be positive");
    TORCH_CHECK(k % 256 == 0 && n % 512 == 0, "MegaMoe BF16 requires k % 256 == 0 and n % 512 == 0");
    const auto tiling =
        TilingForCall(m, k, n, topk, expertPerRank, maxRecvTokens, swigluLimit,
                      weight_before_gmm2_ ? save_gmm1_output : "none", topkWeightType, weightLayout.transW);

    const MegaMoeBF16Training::TrainingSavedStateLayout savedStateLayout(
        m, k, topk, worldSize, expertPerRank, maxRecvTokens, n,
        tiling.savedGateUpMode == MegaMoeBF16Training::kSaveMxFp8,
        tiling.savedGateUpMode != MegaMoeBF16Training::kSaveNone,
        MegaMoeBF16Training::SaveGateUpClipMasks(tiling.swigluLimit, tiling.savedGateUpMode), false, topkWeightBytes);
    const MegaMoeBF16Training::DispatchOutputLayout dispatchOutputLayout(k, maxRecvTokens, topkWeightBytes);
    const MegaMoeBF16Training::ForwardTransientWorkspaceLayout transientWorkspaceLayout(n, expertPerRank,
                                                                                        maxRecvTokens);
    const uint64_t perRankTokenCountStride =
        MegaMoeBF16Training::AlignUp(worldSize * expertPerRank, MegaMoeBF16Training::kAlignment128);
    const uint64_t legacyTrainingTailBytes = MegaMoeBF16Training::AlignUp(
        worldSize * perRankTokenCountStride * MegaMoeBF16Training::kInt32Bytes, MegaMoeBF16Training::kAlignment512);
    TORCH_CHECK(savedStateLayout.legacyBytes + dispatchOutputLayout.totalBytes + transientWorkspaceLayout.totalBytes ==
                    layout_workspace_bytes_ + legacyTrainingTailBytes,
                "MegaMoe BF16 split workspace layout does not match the legacy "
                "forward workspace capacity");

    // Both unpermute implementations overwrite every output token row once.
    // ExpertTokenNumCopyOut likewise writes all local-expert entries, so these
    // returned tensors do not need a device-wide pre-clear.
    auto y = out.defined() ? out : torch::empty({m, k}, x.options());
    auto expertTokenNums = torch::empty({expertPerRank}, topkIds.options().dtype(torch::kInt32));
    // Inference uses only the working route indices and prefixes up to this offset.
    auto routingState = weight_before_gmm2_
                            ? AllocateTrainingSavedState(savedStateLayout, x)
                            : torch::zeros({static_cast<int64_t>(savedStateLayout.savedTokenCountsOffset)},
                                           x.options().dtype(torch::kUInt8));
    torch::Tensor dispatchOutput;
    if (weight_before_gmm2_ && saveDispatchedX) {
        dispatchOutput =
            torch::empty({static_cast<int64_t>(dispatchOutputLayout.totalBytes)}, x.options().dtype(torch::kUInt8));
    }
    ClearWindowControl();

    LaunchMegaMoeBF16ForwardKernel(
        context, x, topkIds, topkWeights, weight1, weight2, y, expertTokenNums, routingState, dispatchOutput,
        workspace_, tiling,
        static_cast<uint32_t>(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic()), ffts_addr_,
        timer);
    const auto stream = c10_npu::getCurrentNPUStream();
    for (const auto* tensor : {&context, &x, &topkIds, &topkWeights, &weight1, &weight2, &y, &expertTokenNums,
                               &routingState, &dispatchOutput, &workspace_}) {
        if (tensor->defined()) c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    }
    if (timer.defined()) c10_npu::NPUCachingAllocator::recordStream(timer.storage().data_ptr(), stream);
    return {y, expertTokenNums, weight_before_gmm2_ ? routingState : torch::Tensor(), dispatchOutput, timer};
}

}  // namespace deepep::megamoe::bf16
