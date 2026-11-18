// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

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

#include "kernels/mega_moe/common/mega_moe_workspace_backward.hpp"

namespace {

constexpr std::size_t kBackwardTopkPartialUbBytes = 128ULL * 1024ULL;
constexpr std::size_t kA5Fp32PerVectorRegister = 64ULL;

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

void LaunchMegaMoeBF16BackwardKernel(torch::Tensor context, torch::Tensor grad_y, torch::Tensor topk_weights,
                                     torch::Tensor weight1, torch::Tensor weight2, torch::Tensor grad_x,
                                     torch::Tensor grad_topk_weights, torch::Tensor grad_weight1,
                                     torch::Tensor grad_weight2, torch::Tensor training_saved_state,
                                     torch::Tensor dispatch_output, torch::Tensor backward_workspace,
                                     const MegaMoeBf16TilingData& tiling, uint32_t splitExperts, uint32_t block_dim,
                                     uint64_t ffts_addr, torch::Tensor timer)
{
    auto stream = c10_npu::getCurrentNPUStream().stream();
    auto ptr = [](const torch::Tensor& t) { return static_cast<uint8_t*>(t.data_ptr()); };
    if (timer.defined()) {
        auto kernel = &deepep_train_mega_moe_bf16_backward_timer_kernel_do;
        kernel(block_dim, stream, ptr(context), ptr(grad_y), ptr(topk_weights), ptr(weight1), ptr(weight2), ptr(grad_x),
               ptr(grad_topk_weights), ptr(grad_weight1), ptr(grad_weight2), ptr(training_saved_state),
               ptr(dispatch_output), ptr(backward_workspace), reinterpret_cast<const uint8_t*>(&tiling), ffts_addr,
               splitExperts, ptr(timer));
    } else {
        auto kernel = &deepep_train_mega_moe_bf16_backward_ready_k_kernel_do;
        kernel(block_dim, stream, ptr(context), ptr(grad_y), ptr(topk_weights), ptr(weight1), ptr(weight2), ptr(grad_x),
               ptr(grad_topk_weights), ptr(grad_weight1), ptr(grad_weight2), ptr(training_saved_state),
               ptr(dispatch_output), ptr(backward_workspace), reinterpret_cast<const uint8_t*>(&tiling), ffts_addr,
               splitExperts);
    }
}

}  // namespace

namespace deepep::megamoe::bf16 {

ops::BackwardResult BF16Plan::Backward(torch::Tensor gradY, torch::Tensor topkWeights, torch::Tensor weight1,
                                       torch::Tensor weight2, torch::Tensor expertTokenNums,
                                       torch::Tensor trainingSavedState, torch::Tensor dispatchOutput,
                                       int64_t maxRecvTokens, double swigluLimit, const std::string& save_gmm1_output,
                                       torch::Tensor timer)
{
    auto context = Context(gradY);
    const int worldSize = world_size_;
    const int64_t m = gradY.size(0);
    const int64_t k = gradY.size(1);
    TORCH_CHECK(weight1.dim() == 3 && weight2.dim() == 3, "weight1 and weight2 must be rank-3 tensors");
    const int64_t expertPerRank = weight1.size(0);
    const auto weightLayout = ResolveWeightLayout(weight1, weight2, k);
    const int64_t n = weightLayout.n;
    TORCH_CHECK(topkWeights.dim() == 2 && topkWeights.size(0) == m, "topk_weights must have shape [m, topk]");
    const int64_t topk = topkWeights.size(1);
    TORCH_CHECK(expertTokenNums.numel() == expertPerRank, "expert_token_nums must have one entry per local expert");
    // Only persistent saved-state is consumed by backward.
    TORCH_CHECK(trainingSavedState.scalar_type() == torch::kUInt8 &&
                    trainingSavedState.device().type() == c10::DeviceType::PrivateUse1,
                "training_saved_state must be an NPU UInt8 tensor");
    TORCH_CHECK(m > 0 && k > 0 && topk > 0 && expertPerRank > 0 && n > 0,
                "all MegaMoe BF16 backward dimensions must be positive");
    TORCH_CHECK(worldSize >= 2 && worldSize <= static_cast<int64_t>(MegaMoeBF16Training::kMaxEpRanks),
                "MegaMoe BF16 backward requires 2..1024 EP ranks");
    TORCH_CHECK(m <= 16384 && expertPerRank <= static_cast<int64_t>(MegaMoeBF16Training::kMaxLocalExperts) &&
                    expertPerRank * worldSize <= static_cast<int64_t>(MegaMoeBF16Training::kMaxTotalExperts),
                "MegaMoe BF16 backward shape exceeds supported routing limits "
                "(32 local, 2048 total experts)");
    TORCH_CHECK(topk <= expertPerRank * worldSize, "MegaMoe BF16 backward topk must not exceed total expert count");
    TORCH_CHECK(maxRecvTokens > 0, "MegaMoe BF16 backward max_recv_tokens must be positive");
    TORCH_CHECK(maxRecvTokens <= std::numeric_limits<int32_t>::max(),
                "MegaMoe BF16 backward max_recv_tokens exceeds int32 routing capacity");
    TORCH_CHECK(k % 256 == 0 && n % 512 == 0, "MegaMoe BF16 backward requires k % 256 == 0 and n % 512 == 0");
    TORCH_CHECK(gradY.scalar_type() == torch::kBFloat16, "grad_y must be bfloat16");
    TORCH_CHECK(topkWeights.scalar_type() == torch::kBFloat16 || topkWeights.scalar_type() == torch::kFloat32,
                "topk_weights must be bfloat16 or float32");
    const uint32_t topkWeightType = topkWeights.scalar_type() == torch::kFloat32 ? MegaMoeBF16Training::kTopkWeightFp32
                                                                                 : MegaMoeBF16Training::kTopkWeightBf16;
    const auto topkWeightBytes = MegaMoeBF16Training::TopkWeightBytes(topkWeightType);
    TORCH_CHECK(weight1.scalar_type() == torch::kBFloat16, "weight1 must be bfloat16");
    TORCH_CHECK(weight2.scalar_type() == torch::kBFloat16, "weight2 must be bfloat16");
    TORCH_CHECK(expertTokenNums.scalar_type() == torch::kInt32, "expert_token_nums must be int32");
    CheckNpuContiguous(gradY, "grad_y");
    CheckNpuContiguous(topkWeights, "topk_weights");
    CheckNpuContiguous(weight1, "weight1");
    CheckNpuContiguous(weight2, "weight2");
    CheckNpuContiguous(trainingSavedState, "training_saved_state");
    CheckNpuContiguous(dispatchOutput, "dispatch_output");
    const MegaMoeBF16Training::DispatchOutputLayout dispatchOutputLayout(k, maxRecvTokens, topkWeightBytes);
    TORCH_CHECK(dispatchOutput.scalar_type() == torch::kUInt8 &&
                    static_cast<uint64_t>(dispatchOutput.numel()) >= dispatchOutputLayout.totalBytes,
                "dispatch_output must be an NPU UInt8 tensor with at least ", dispatchOutputLayout.totalBytes,
                " bytes");

    const auto tiling = TilingForCall(m, k, n, topk, expertPerRank, maxRecvTokens, swigluLimit, save_gmm1_output,
                                      topkWeightType, weightLayout.transW);

    constexpr std::size_t kTileRows = 128ULL;
    constexpr std::size_t kTileColumns = 256ULL;
    const std::size_t ubTileBytes = 3ULL * kTileRows * kTileColumns * sizeof(uint16_t);
    // MXFP8 decoding uses a data chunk, compact E8M0 scales, two expanded
    // scale registers stored in UB and three full-tile clipping masks.
    constexpr std::size_t kMxDecodeRows = 16ULL;
    constexpr std::size_t kMxScaleRowBytes = 32ULL;
    constexpr std::size_t kMxScaleExpansionBytes = 2ULL * 512ULL;
    const std::size_t ubMxFp8TileBytes = ubTileBytes + kMxDecodeRows * (kTileColumns + kMxScaleRowBytes) +
                                         kMxScaleExpansionBytes + 3ULL * kTileRows * kMxScaleRowBytes;
    const std::size_t ubRestoreBytes = Bf16Align(static_cast<std::size_t>(worldSize) * expertPerRank, 128ULL) * 4ULL;
    const std::size_t dispatchElements =
        Bf16Align(static_cast<std::size_t>(tiling.dispatchRows) * static_cast<std::size_t>(k), 32ULL);
    const std::size_t ubP0ScratchBytes = 4ULL * dispatchElements +
                                         Bf16Align(2ULL * static_cast<std::size_t>(k) * sizeof(uint16_t), 32ULL) +
                                         ubRestoreBytes;
    const std::size_t ubUnpermuteBytes = 3ULL * Bf16Align(6ULL * static_cast<std::size_t>(k), 32ULL);
    const std::size_t dpPartialTiles = MegaMoeBF16Training::BackwardTopkPartialCount(tiling.savedGateUpMode, n / 2);
    const std::size_t alignedDpPartialTiles = Bf16Align(dpPartialTiles * sizeof(float), 32ULL) / sizeof(float);
    TORCH_CHECK((alignedDpPartialTiles + kA5Fp32PerVectorRegister - 1) * sizeof(float) <= kBackwardTopkPartialUbBytes,
                "MegaMoe BF16 backward n=", n, " exceeds the P6 partial-reduction UB capacity");
    const std::size_t ubTopkBytes =
        kBackwardTopkPartialUbBytes + Bf16Align(1024ULL * topkWeightBytes, 32ULL) +
        Bf16Align(static_cast<std::size_t>(worldSize) * expertPerRank * sizeof(int32_t), 32ULL);
    const std::size_t ubResetBytes = MegaMoeBF16Training::ForwardDispatchUbLayout::kCountResetChunkBytes;
    // These phases own UB at different times. Include the post-P6 count
    // reset and the P6 route and prefix buffers in the UB budget.
    const uint32_t savedMode = tiling.savedGateUpMode;
    const std::size_t p1UbBytes = savedMode == MegaMoeBF16Training::kSaveMxFp8 ? ubMxFp8TileBytes : ubTileBytes;
    const std::size_t ubFootprint =
        std::max({p1UbBytes, ubP0ScratchBytes, ubUnpermuteBytes, ubTopkBytes, ubResetBytes});
    uint64_t deviceUbBytes = 0;
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(platform_ascendc::CoreMemType::UB,
                                                                            deviceUbBytes);
    const uint64_t ubBudget = std::min<uint64_t>(deviceUbBytes, 248ULL * 1024ULL);
    TORCH_CHECK(ubFootprint <= ubBudget, "MegaMoe BF16 backward k=", k, " exceeds the UB budget (kernel needs ",
                ubFootprint, " bytes, have ", ubBudget, ")");

    const MegaMoeBF16Training::TrainingSavedStateLayout savedStateLayout(
        m, k, topk, worldSize, expertPerRank, maxRecvTokens, n,
        tiling.savedGateUpMode == MegaMoeBF16Training::kSaveMxFp8,
        tiling.savedGateUpMode != MegaMoeBF16Training::kSaveNone,
        MegaMoeBF16Training::SaveGateUpClipMasks(tiling.swigluLimit, tiling.savedGateUpMode), false, topkWeightBytes);
    TORCH_CHECK(static_cast<std::size_t>(trainingSavedState.numel()) * trainingSavedState.element_size() >=
                    savedStateLayout.totalBytes,
                "MegaMoe BF16 training_saved_state is too small: need ", savedStateLayout.totalBytes, " bytes");
    uint32_t splitExperts = MegaMoeBF16Training::DefaultBackwardDispatchSplitExperts(
        tiling.savedGateUpMode, static_cast<uint32_t>(expertPerRank));
    // This diagnostic override selects how many experts use row readiness on the GM path.
    const char* splitText = std::getenv("MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS");
    if (splitText != nullptr) {
        if (std::string(splitText) == "all") {
            splitExperts = static_cast<uint32_t>(expertPerRank);
        } else {
            char* end = nullptr;
            const long value = std::strtol(splitText, &end, 10);
            TORCH_CHECK(end != splitText && *end == '\0' && value >= 0 && value <= expertPerRank,
                        "MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS must be 0..localE or all");
            splitExperts = static_cast<uint32_t>(value);
        }
    }
    if (!MegaMoeBF16Training::UseGmForBackwardDataGrad(tiling.savedGateUpMode)) splitExperts = 0;
    const MegaMoeBF16BackwardWorkspaceLayout backwardLayout(tiling, splitExperts);
    auto backwardWorkspace = AllocateBackwardWorkspace(backwardLayout, gradY);

    // P5 unpermutes and sums packed dX into every gradX token row; P6 writes
    // every TopK route once. The kernel overwrites every non-empty expert's
    // weight-gradient plane and clears only the experts whose effective row
    // count is zero, so none of these outputs needs a full pre-clear here.
    auto gradX = torch::empty({m, k}, gradY.options());
    auto gradTopkWeights = torch::empty_like(topkWeights);
    auto gradWeight1 = torch::empty_like(weight1);
    auto gradWeight2 = torch::empty_like(weight2);
    ClearWindowControl();

    LaunchMegaMoeBF16BackwardKernel(
        context, gradY, topkWeights, weight1, weight2, gradX, gradTopkWeights, gradWeight1, gradWeight2,
        trainingSavedState, dispatchOutput, backwardWorkspace, tiling, splitExperts,
        static_cast<uint32_t>(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic()), ffts_addr_,
        timer);
    const auto stream = c10_npu::getCurrentNPUStream();
    for (const auto* tensor :
         {&context, &gradY, &topkWeights, &weight1, &weight2, &expertTokenNums, &trainingSavedState, &dispatchOutput,
          &gradX, &gradTopkWeights, &gradWeight1, &gradWeight2, &backwardWorkspace}) {
        c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    }
    if (timer.defined()) c10_npu::NPUCachingAllocator::recordStream(timer.storage().data_ptr(), stream);
    return {gradX, gradTopkWeights, gradWeight1, gradWeight2, timer};
}

ops::BackwardResult BF16Plan::backward(torch::Tensor grad_y, torch::Tensor topk_weights, torch::Tensor weight1,
                                       torch::Tensor weight2, torch::Tensor counts, torch::Tensor saved_state,
                                       torch::Tensor dispatch_output, int64_t max_recv_tokens, double swiglu_limit,
                                       const std::string& save_gmm1_output)
{
    TORCH_CHECK(address_ != 0, "training operator has been closed");
    TORCH_CHECK(weight_before_gmm2_, "BF16 backward requires a need_grad=True Plan");
    return Backward(grad_y, topk_weights, weight1, weight2, counts, saved_state, dispatch_output, max_recv_tokens,
                    swiglu_limit, save_gmm1_output, {});
}

ops::BackwardResult BF16Plan::backward_with_timer(torch::Tensor dy, torch::Tensor weights, torch::Tensor w1,
                                                  torch::Tensor w2, torch::Tensor counts, torch::Tensor saved,
                                                  torch::Tensor dispatch, int64_t maxRecv, double limit,
                                                  const std::string& mode)
{
    TORCH_CHECK(address_ != 0, "training operator has been closed");
    TORCH_CHECK(weight_before_gmm2_, "BF16 backward requires a need_grad=True Plan");
    auto timer = AllocateTimer(dy, weights, w1, w2, maxRecv, 1, mode);
    return Backward(dy, weights, w1, w2, counts, saved, dispatch, maxRecv, limit, mode, timer);
}

}  // namespace deepep::megamoe::bf16
