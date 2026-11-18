// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

#include "bf16_host_utils.hpp"

namespace deepep::megamoe::bf16::ops {

ForwardResult MegaMoeBF16Forward(torch::Tensor context, torch::Tensor x, torch::Tensor topkIds,
                                 torch::Tensor topkWeights, torch::Tensor weight1, torch::Tensor weight2,
                                 int64_t maxRecvTokens, int rank, int worldSize, std::size_t symmetricBufferBytes,
                                 uint64_t ffts_addr, void* symmetricBuffer, double swigluLimit,
                                 const std::string& save_gmm1_output, torch::Tensor out, torch::Tensor timer,
                                 torch::Tensor workspace, bool weight_before_gmm2)
{
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
    const auto weightLayout = ResolveBf16WeightLayout(weight1, weight2, k);
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
    MegaMoeBf16TilingData tiling{};
    const auto hostTiling =
        detail::MakeMegaMoeHostTiling(tiling, m, k, n, topk, worldSize, expertPerRank, maxRecvTokens, rank, true,
                                      swigluLimit, true, save_gmm1_output, topkWeightType);
    tiling.routeWeightBeforeGmm2 = weight_before_gmm2 ? 1U : 0U;
    tiling.transW = weightLayout.transW;
    tiling.transW2 = weightLayout.transW;
    const auto earlyWeightBuffers = detail::GetBf16EarlyTopkWeightBufferBytes(tiling);
    const uint64_t peerBytes = hostTiling.peer_bytes + earlyWeightBuffers.peer_bytes;
    TORCH_CHECK(peerBytes <= symmetricBufferBytes, "MegaMoe BF16 local_mem_size is too small: need ", peerBytes,
                " bytes, got ", symmetricBufferBytes);

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
                    hostTiling.workspace_bytes + earlyWeightBuffers.workspace_bytes + legacyTrainingTailBytes,
                "MegaMoe BF16 split workspace layout does not match the legacy "
                "forward workspace capacity");

    // Both unpermute implementations overwrite every output token row once.
    // ExpertTokenNumCopyOut likewise writes all local-expert entries, so these
    // returned tensors do not need a device-wide pre-clear.
    auto y = out.defined() ? out : torch::empty({m, k}, x.options());
    auto expertTokenNums = torch::empty({expertPerRank}, topkIds.options().dtype(torch::kInt32));
    auto trainingSavedState = AllocateTrainingSavedState(savedStateLayout, x);
    auto dispatchOutput =
        torch::empty({static_cast<int64_t>(dispatchOutputLayout.totalBytes)}, x.options().dtype(torch::kUInt8));
    if (workspace.defined()) {
        TORCH_CHECK(workspace.device() == x.device() && workspace.scalar_type() == torch::kUInt8 &&
                        workspace.is_contiguous() &&
                        static_cast<uint64_t>(workspace.numel()) >= ForwardTransientWorkspaceBytes(tiling),
                    "BF16 prepared workspace does not match this invocation");
    }
    auto forwardTransientWorkspace = workspace.defined() ? workspace : AllocateForwardTransientWorkspace(x, tiling);
    ClearMegaMoeWindowControlRegion(symmetricBuffer, symmetricBufferBytes, worldSize, expertPerRank,
                                    tiling.compactTokenCountExchange != 0);

    LaunchMegaMoeBF16ForwardKernel(
        context, x, topkIds, topkWeights, weight1, weight2, y, expertTokenNums, trainingSavedState, dispatchOutput,
        forwardTransientWorkspace, tiling,
        static_cast<uint32_t>(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic()),
        symmetricBuffer, symmetricBufferBytes, rank, worldSize, ffts_addr, timer);
    const auto stream = c10_npu::getCurrentNPUStream();
    for (const auto* tensor : {&context, &x, &topkIds, &topkWeights, &weight1, &weight2, &y, &expertTokenNums,
                               &trainingSavedState, &dispatchOutput, &forwardTransientWorkspace}) {
        c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    }
    if (timer.defined()) c10_npu::NPUCachingAllocator::recordStream(timer.storage().data_ptr(), stream);
    return {y, expertTokenNums, trainingSavedState, dispatchOutput, timer};
}

}  // namespace deepep::megamoe::bf16::ops

namespace deepep::megamoe::bf16 {

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
    const auto host =
        ops::detail::MakeMegaMoeHostTiling(tiling, max_tokens, hidden, 2 * intermediate, topk, world, local_experts,
                                           max_recv, 0, true, 0.0, true, "bf16", MegaMoeBF16Training::kTopkWeightFp32);
    const auto weights = ops::detail::GetBf16EarlyTopkWeightBufferBytes(tiling);
    const uint64_t total = host.peer_bytes + weights.peer_bytes;
    TORCH_CHECK(total <= uint64_t(std::numeric_limits<int64_t>::max()) - kDeviceLargePageBytes,
                "BF16 symmetric capacity overflow");
    return static_cast<int64_t>((total + kDeviceLargePageBytes - 1) / kDeviceLargePageBytes * kDeviceLargePageBytes);
}

void BF16Plan::prepare(int64_t m, int64_t h, int64_t f, int64_t topk, int64_t local_experts, int64_t max_recv,
                       double limit, const std::string& save_type, bool fp32_p)
{
    TORCH_CHECK(address_ != 0, "BF16 Plan has been closed");
    MegaMoeBf16TilingData tiling{};
    const auto host = ops::detail::MakeMegaMoeHostTiling(
        tiling, m, h, 2 * f, topk, world_size_, local_experts, max_recv, rank_, true, limit, true, save_type,
        fp32_p ? MegaMoeBF16Training::kTopkWeightFp32 : MegaMoeBF16Training::kTopkWeightBf16);
    const auto weights = ops::detail::GetBf16EarlyTopkWeightBufferBytes(tiling);
    TORCH_CHECK(host.peer_bytes + weights.peer_bytes <= SymmetricBufferBytes(), "BF16 symmetric window is too small");
    workspace_ = AllocateForwardTransientWorkspace(context_, tiling);
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
                                         const std::string& save_gmm1_output)
{
    TORCH_CHECK(address_ != 0, "training operator has been closed");
    TORCH_CHECK(out.defined(), "y must be a defined output tensor");
    return ops::MegaMoeBF16Forward(Context(x), x, topk_ids, topk_weights, weight1, weight2, max_recv_tokens, rank_,
                                   world_size_, SymmetricBufferBytes(), ffts_addr_, SymmetricBuffer(), swiglu_limit,
                                   save_gmm1_output, out, {}, workspace_, weight_before_gmm2_);
}

ops::ForwardResult BF16Plan::forward_with_timer(torch::Tensor x, torch::Tensor ids, torch::Tensor weights,
                                                torch::Tensor w1, torch::Tensor w2, int64_t maxRecv, double limit,
                                                const std::string& mode, c10::optional<torch::Tensor> out)
{
    TORCH_CHECK(address_ != 0, "training operator has been closed");
    auto timer = AllocateTrainingTimer(x, weights, w1, w2, maxRecv, 0, mode);
    return ops::MegaMoeBF16Forward(Context(x), x, ids, weights, w1, w2, maxRecv, rank_, world_size_,
                                   SymmetricBufferBytes(), ffts_addr_, SymmetricBuffer(), limit, mode,
                                   out.value_or(torch::Tensor()), timer, workspace_, weight_before_gmm2_);
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
        auto new_context = torch::empty({kMegaMoeContextBytes}, like.options().dtype(torch::kUInt8));
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

}  // namespace deepep::megamoe::bf16
