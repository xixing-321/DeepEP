// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

#include "bf16_host_utils.hpp"

namespace deepep::megamoe::bf16::ops {
BackwardResult MegaMoeBF16Backward(torch::Tensor context, torch::Tensor gradY, torch::Tensor topkWeights,
                                   torch::Tensor weight1, torch::Tensor weight2, torch::Tensor expertTokenNums,
                                   torch::Tensor trainingSavedState, torch::Tensor dispatchOutput,
                                   int64_t maxRecvTokens, int rank, int worldSize, std::size_t symmetricBufferBytes,
                                   uint64_t ffts_addr, void* symmetricBuffer, double swigluLimit,
                                   const std::string& save_gmm1_output, torch::Tensor timer)
{
    const int64_t m = gradY.size(0);
    const int64_t k = gradY.size(1);
    TORCH_CHECK(weight1.dim() == 3 && weight2.dim() == 3, "weight1 and weight2 must be rank-3 tensors");
    const int64_t expertPerRank = weight1.size(0);
    const auto weightLayout = ResolveBf16WeightLayout(weight1, weight2, k);
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

    MegaMoeBf16TilingData tiling{};
    const auto hostTiling =
        detail::MakeMegaMoeHostTiling(tiling, m, k, n, topk, worldSize, expertPerRank, maxRecvTokens, rank, true,
                                      swigluLimit, true, save_gmm1_output, topkWeightType);
    tiling.transW = weightLayout.transW;
    tiling.transW2 = weightLayout.transW;
    const auto earlyWeightBuffers = detail::GetBf16EarlyTopkWeightBufferBytes(tiling);
    const uint64_t peerBytes = hostTiling.peer_bytes + earlyWeightBuffers.peer_bytes;
    TORCH_CHECK(peerBytes <= symmetricBufferBytes, "MegaMoe BF16 backward local_mem_size is too small: need ",
                peerBytes, " bytes, got ", symmetricBufferBytes);

    const std::size_t ubTileBytes = 3ULL * 128ULL * 256ULL * 2ULL;
    // MXFP8 backward decodes 16 rows at a time after the three persistent
    // BF16 work tiles: one FP8 data chunk, compact/padded E8M0 scales,
    // expanded BF16/FP32 scales, a temporary FP32 decoded chunk and three
    // optional clipping-mask chunks.
    constexpr std::size_t kMxDecodeRows = 16ULL;
    constexpr std::size_t kMxTileColumns = 256ULL;
    const std::size_t ubMxFp8TileBytes =
        ubTileBytes + kMxDecodeRows * kMxTileColumns + kMxDecodeRows * 32ULL +
        kMxDecodeRows * (kMxTileColumns / 32ULL) * (2ULL * sizeof(uint16_t) + 4ULL * sizeof(float)) +
        kMxDecodeRows * kMxTileColumns * sizeof(float) + 3ULL * kMxDecodeRows * 32ULL;
    const std::size_t ubRestoreBytes = Bf16Align(static_cast<std::size_t>(worldSize) * expertPerRank, 128ULL) * 4ULL;
    const std::size_t dispatchElements =
        Bf16Align(static_cast<std::size_t>(tiling.dispatchRows) * static_cast<std::size_t>(k), 32ULL);
    const std::size_t ubP0ScratchBytes = 4ULL * dispatchElements +
                                         Bf16Align(2ULL * static_cast<std::size_t>(k) * sizeof(uint16_t), 32ULL) +
                                         ubRestoreBytes;
    const std::size_t ubUnpermuteBytes = 3ULL * Bf16Align(6ULL * static_cast<std::size_t>(k), 32ULL);
    const std::size_t dpPartialTiles = MegaMoeBF16Training::BackwardTopkPartialCount(
        MegaMoeBF16Training::ResolveSavedGateUpMode(tiling.savedGateUpMode, tiling.swigluLimit), n / 2);
    const std::size_t alignedDpPartialTiles = Bf16Align(dpPartialTiles * sizeof(float), 32ULL) / sizeof(float);
    TORCH_CHECK((alignedDpPartialTiles + kA5Fp32PerVectorRegister - 1) * sizeof(float) <= kBackwardTopkPartialUbBytes,
                "MegaMoe BF16 backward n=", n, " exceeds the P6 partial-reduction UB capacity");
    const std::size_t ubTopkBytes = kBackwardTopkPartialUbBytes + Bf16Align(1024ULL * topkWeightBytes, 32ULL);
    const std::size_t ubResetBytes = MegaMoeBF16Training::ForwardDispatchUbLayout::kCountResetChunkBytes;
    // These phases own UB at different times. Include the post-P6 count
    // reset and the P6 route buffer, not just the P0 and SwiGLU allocations.
    const uint32_t savedMode = MegaMoeBF16Training::ResolveSavedGateUpMode(tiling.savedGateUpMode, tiling.swigluLimit);
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
        MegaMoeBF16Training::SaveGateUpAsMxFp8(tiling.swigluLimit, tiling.savedGateUpMode),
        tiling.savedGateUpMode != MegaMoeBF16Training::kSaveNone,
        MegaMoeBF16Training::SaveGateUpClipMasks(tiling.swigluLimit, tiling.savedGateUpMode), false, topkWeightBytes);
    TORCH_CHECK(static_cast<std::size_t>(trainingSavedState.numel()) * trainingSavedState.element_size() >=
                    savedStateLayout.totalBytes,
                "MegaMoe BF16 training_saved_state is too small: need ", savedStateLayout.totalBytes, " bytes");
#if DEEPEP_BF16_FORCE_EXPERT_READY
    // This build policy takes precedence over the diagnostic split override.
    const uint32_t splitExperts = 0;
#else
    uint32_t splitExperts = MegaMoeBF16Training::DefaultBackwardDispatchSplitExperts(
        tiling.savedGateUpMode, static_cast<uint32_t>(expertPerRank));
    // Explicit k is a diagnostic override; none still forces k=0 below.
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
#endif
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
    ClearMegaMoeWindowControlRegion(symmetricBuffer, symmetricBufferBytes, worldSize, expertPerRank,
                                    tiling.compactTokenCountExchange != 0);

    LaunchMegaMoeBF16BackwardKernel(
        context, gradY, topkWeights, weight1, weight2, gradX, gradTopkWeights, gradWeight1, gradWeight2,
        trainingSavedState, dispatchOutput, backwardWorkspace, tiling, splitExperts,
        static_cast<uint32_t>(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic()),
        symmetricBuffer, symmetricBufferBytes, rank, worldSize, ffts_addr, timer);
    const auto stream = c10_npu::getCurrentNPUStream();
    for (const auto* tensor :
         {&context, &gradY, &topkWeights, &weight1, &weight2, &expertTokenNums, &trainingSavedState, &dispatchOutput,
          &gradX, &gradTopkWeights, &gradWeight1, &gradWeight2, &backwardWorkspace}) {
        c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    }
    if (timer.defined()) c10_npu::NPUCachingAllocator::recordStream(timer.storage().data_ptr(), stream);
    return {gradX, gradTopkWeights, gradWeight1, gradWeight2, timer};
}

}  // namespace deepep::megamoe::bf16::ops

namespace deepep::megamoe::bf16 {
ops::BackwardResult BF16Plan::backward(torch::Tensor grad_y, torch::Tensor topk_weights, torch::Tensor weight1,
                                       torch::Tensor weight2, torch::Tensor counts, torch::Tensor saved_state,
                                       torch::Tensor dispatch_output, int64_t max_recv_tokens, double swiglu_limit,
                                       const std::string& save_gmm1_output)
{
    TORCH_CHECK(address_ != 0, "training operator has been closed");
    TORCH_CHECK(weight_before_gmm2_, "BF16 backward requires a need_grad=True Plan");
    return ops::MegaMoeBF16Backward(Context(grad_y), grad_y, topk_weights, weight1, weight2, counts, saved_state,
                                    dispatch_output, max_recv_tokens, rank_, world_size_, SymmetricBufferBytes(),
                                    ffts_addr_, SymmetricBuffer(), swiglu_limit, save_gmm1_output);
}

ops::BackwardResult BF16Plan::backward_with_timer(torch::Tensor dy, torch::Tensor weights, torch::Tensor w1,
                                                  torch::Tensor w2, torch::Tensor counts, torch::Tensor saved,
                                                  torch::Tensor dispatch, int64_t maxRecv, double limit,
                                                  const std::string& mode)
{
    TORCH_CHECK(address_ != 0, "training operator has been closed");
    TORCH_CHECK(weight_before_gmm2_, "BF16 backward requires a need_grad=True Plan");
    auto timer = AllocateTrainingTimer(dy, weights, w1, w2, maxRecv, 1, limit, mode);
    return ops::MegaMoeBF16Backward(Context(dy), dy, weights, w1, w2, counts, saved, dispatch, maxRecv, rank_,
                                    world_size_, SymmetricBufferBytes(), ffts_addr_, SymmetricBuffer(), limit, mode,
                                    timer);
}

}  // namespace deepep::megamoe::bf16
