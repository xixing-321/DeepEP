// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

#include "plan.hpp"
#include "check.hpp"
#include "shmem.h"
#include "kernels/mega_moe/common/mega_moe_w8a8_backward_params.hpp"
#include "kernels/mega_moe/common/mega_moe_w8a8_saved_layout.hpp"
#include "host_utils.hpp"
#include "kernel_launch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "kernels/mega_moe/common/mega_moe_bf16_layout.hpp"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "kernels/mega_moe/common/mega_moe_workspace_backward.hpp"

namespace {

constexpr std::size_t kBackwardTopkPartialUbBytes = 128ULL * 1024ULL;
constexpr std::size_t kA5Fp32PerVectorRegister = 64ULL;

using MegaMoeBF16BackwardWorkspaceLayout = MegaMoeBF16Impl::BackwardWorkspaceLayout;

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

}  // namespace

namespace deepep::megamoe::bf16 {

BackwardResult BF16Plan::backward(torch::Tensor gradY, torch::Tensor topkWeights, torch::Tensor weight1,
                                  torch::Tensor weight2, torch::Tensor expertTokenNums,
                                  torch::Tensor trainingSavedState, torch::Tensor dispatchOutput, int64_t maxRecvTokens,
                                  double swigluLimit, const std::string& save_gmm1_output, bool captureTimer)
{
    TORCH_CHECK(address_ != 0, "training operator has been closed");
    TORCH_CHECK(weight_before_gmm2_, "BF16 backward requires a need_grad=True Plan");
    const torch::Tensor timer =
        captureTimer ? AllocateTimer(gradY, topkWeights, weight1, weight2, maxRecvTokens, 1, save_gmm1_output)
                     : torch::Tensor();
    auto context = Context(gradY);
    const int worldSize = world_size_;
    CheckTensor(gradY, 2, "grad_y");
    CheckTensor(topkWeights, 2, "topk_weights");
    CheckTensor(weight1, 3, "weight1");
    CheckTensor(weight2, 3, "weight2");
    CheckTensor(expertTokenNums, 1, "expert_token_nums");
    CheckTensor(trainingSavedState, 1, "training_saved_state");
    CheckTensor(dispatchOutput, 1, "dispatch_output");
    const int64_t m = gradY.size(0);
    const int64_t k = gradY.size(1);
    const int64_t expertPerRank = weight1.size(0);
    const auto weightLayout = ResolveWeightLayout(weight1, weight2, k);
    const int64_t n = weightLayout.n;
    TORCH_CHECK(topkWeights.size(0) == m, "topk_weights must have shape [m, topk]");
    const int64_t topk = topkWeights.size(1);
    TORCH_CHECK(expertTokenNums.numel() == expertPerRank, "expert_token_nums must have one entry per local expert");
    // Only persistent saved-state is consumed by backward; NPU/contiguous/dims via CheckTensor.
    TORCH_CHECK(trainingSavedState.scalar_type() == torch::kUInt8, "training_saved_state must be UInt8");
    TORCH_CHECK(gradY.scalar_type() == torch::kBFloat16, "grad_y must be bfloat16");
    const auto topkWeightType = TopkWeightType(topkWeights.scalar_type());
    const auto topkWeightBytes = MegaMoeBF16Impl::TopkWeightBytes(topkWeightType);
    TORCH_CHECK(weight1.scalar_type() == torch::kBFloat16, "weight1 must be bfloat16");
    TORCH_CHECK(weight2.scalar_type() == torch::kBFloat16, "weight2 must be bfloat16");
    TORCH_CHECK(expertTokenNums.scalar_type() == torch::kInt32, "expert_token_nums must be int32");
    const auto tiling = TilingForCall(m, k, n, topk, expertPerRank, maxRecvTokens, swigluLimit, save_gmm1_output,
                                      topkWeightType, weightLayout.transW);
    const MegaMoeBF16Impl::DispatchOutputLayout dispatchOutputLayout(k, maxRecvTokens, topkWeightBytes);
    TORCH_CHECK(dispatchOutput.scalar_type() == torch::kUInt8 &&
                    static_cast<uint64_t>(dispatchOutput.numel()) >= dispatchOutputLayout.totalBytes,
                "dispatch_output must be UInt8 with at least ", dispatchOutputLayout.totalBytes, " bytes");

    using SwigluUb = MegaMoeBF16Impl::BackwardSwigluUbLayout;
    const std::size_t ubRestoreBytes = AlignUp(static_cast<std::size_t>(worldSize) * expertPerRank, 128ULL) * 4ULL;
    const std::size_t dispatchElements =
        AlignUp(static_cast<std::size_t>(tiling.dispatchRows) * static_cast<std::size_t>(k), 32ULL);
    const std::size_t ubP0ScratchBytes = 4ULL * dispatchElements +
                                         AlignUp(2ULL * static_cast<std::size_t>(k) * sizeof(uint16_t), 32ULL) +
                                         ubRestoreBytes;
    const std::size_t ubUnpermuteBytes = 3ULL * AlignUp(6ULL * static_cast<std::size_t>(k), 32ULL);
    const std::size_t dpPartialTiles = MegaMoeBF16Impl::BackwardTopkPartialCount(tiling.savedGateUpMode, n / 2);
    const std::size_t alignedDpPartialTiles = AlignUp(dpPartialTiles * sizeof(float), 32ULL) / sizeof(float);
    TORCH_CHECK((alignedDpPartialTiles + kA5Fp32PerVectorRegister - 1) * sizeof(float) <= kBackwardTopkPartialUbBytes,
                "MegaMoe BF16 backward n=", n, " exceeds the P6 partial-reduction UB capacity");
    const std::size_t ubTopkBytes =
        kBackwardTopkPartialUbBytes + AlignUp(1024ULL * topkWeightBytes, 32ULL) +
        AlignUp(static_cast<std::size_t>(worldSize) * expertPerRank * sizeof(int32_t), 32ULL);
    const std::size_t ubResetBytes = MegaMoeBF16Impl::ForwardDispatchUbLayout::kCountResetChunkBytes;
    // These phases own UB at different times. Include the post-P6 count
    // reset and the P6 route and prefix buffers in the UB budget.
    const uint32_t savedMode = tiling.savedGateUpMode;
    const std::size_t p1UbBytes =
        savedMode == MegaMoeBF16Impl::kSaveMxFp8 ? SwigluUb::kMxScratchEnd : SwigluUb::kWorkTileBytes;
    const std::size_t ubFootprint =
        std::max({p1UbBytes, ubP0ScratchBytes, ubUnpermuteBytes, ubTopkBytes, ubResetBytes});
    const uint64_t ubBudget = std::min<uint64_t>(platform_.ub, 248ULL * 1024ULL);
    TORCH_CHECK(ubFootprint <= ubBudget, "MegaMoe BF16 backward k=", k, " exceeds the UB budget (kernel needs ",
                ubFootprint, " bytes, have ", ubBudget, ")");

    const auto savedStateLayout = MakeSavedStateLayout(tiling, k, n, maxRecvTokens);
    TORCH_CHECK(static_cast<std::size_t>(trainingSavedState.numel()) * trainingSavedState.element_size() >=
                    savedStateLayout.totalBytes,
                "MegaMoe BF16 training_saved_state is too small: need ", savedStateLayout.totalBytes, " bytes");
    const uint32_t splitExperts = MegaMoeBF16Impl::BackwardDispatchSplitExperts(tiling.savedGateUpMode, tiling.n / 2,
                                                                                static_cast<uint32_t>(expertPerRank));
    const MegaMoeBF16BackwardWorkspaceLayout backwardLayout(tiling, splitExperts, platform_.aic * 2);
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

    const auto stream = c10_npu::getCurrentNPUStream();
    auto ptr = [](const torch::Tensor& t) { return static_cast<uint8_t*>(t.data_ptr()); };
    const uint32_t block_dim = platform_.aic;
    if (timer.defined()) {
        mega_moe_bf16_backward_timer_kernel_do(
            block_dim, stream.stream(), ptr(context), ptr(gradY), ptr(topkWeights), ptr(weight1), ptr(weight2),
            ptr(gradX), ptr(gradTopkWeights), ptr(gradWeight1), ptr(gradWeight2), ptr(trainingSavedState),
            ptr(dispatchOutput), ptr(backwardWorkspace), reinterpret_cast<const uint8_t*>(&tiling), ffts_addr_,
            splitExperts, ptr(timer));
    } else {
        mega_moe_bf16_backward_kernel_do(block_dim, stream.stream(), ptr(context), ptr(gradY), ptr(topkWeights),
                                         ptr(weight1), ptr(weight2), ptr(gradX), ptr(gradTopkWeights), ptr(gradWeight1),
                                         ptr(gradWeight2), ptr(trainingSavedState), ptr(dispatchOutput),
                                         ptr(backwardWorkspace), reinterpret_cast<const uint8_t*>(&tiling), ffts_addr_,
                                         splitExperts);
    }
    for (const auto* tensor : {&gradY, &topkWeights, &weight1, &weight2}) {
        c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    }
    // Rebuilt dispatch and hook-restored storage can belong to a different stream.
    allocations_.RecordIfForeign(context);
    allocations_.RecordIfForeign(trainingSavedState);
    allocations_.RecordIfForeign(dispatchOutput);
    return {gradX, gradTopkWeights, gradWeight1, gradWeight2, timer};
}

}  // namespace deepep::megamoe::bf16

namespace deepep::megamoe {

int64_t w8a8_backward_buffer_size(int64_t world, int64_t experts, int64_t tokens, int64_t topk, int64_t hidden)
{
    // Reuse the forward dimension validation and reserve space for FP32 dP.
    (void)buffer_size(world, experts, tokens, topk, hidden, true);
    return MegaMoeTrainW8A8::PeerLayout(world, tokens, hidden, topk, experts, 1).total;
}

W8A8BackwardPlan::W8A8BackwardPlan(uintptr_t address, int64_t bytes, int64_t max_tokens, int64_t hidden, int64_t ffn,
                                   int64_t topk, int64_t experts, int64_t world, const torch::Tensor& like,
                                   uint64_t ffts)
    : platform_(GetPlatformInfo()),
      rank_(aclshmem_my_pe()),
      world_(world),
      max_tokens_(max_tokens),
      hidden_(hidden),
      ffn_(ffn),
      topk_(topk),
      experts_(experts),
      ffts_(ffts)
{
    const int64_t required = w8a8_backward_buffer_size(world, experts, max_tokens, topk, hidden);
    TORCH_CHECK(rank_ < world_ && address && bytes >= required && ffn >= 256 && ffn <= 4096 && ffn % 128 == 0,
                "Invalid W8A8 backward plan");
    CheckTensor(like, 2, "W8A8 input");
    auto* window = reinterpret_cast<void*>(address);
    const auto control = MegaMoeTrainW8A8::PeerLayout(world_, max_tokens_, hidden_, topk_, experts_, 1).xq;
    ASCEND_DEEPEP_CHECK_ACL(aclrtMemset(window, required, 0, control));
    std::vector<uint64_t> addresses(world_);
    FillPeerAddresses(addresses.data(), window, world_);
    // Order initialization after prior uses of a recycled allocation on the plan's stream.
    peers_ = torch::from_blob(addresses.data(), {world}, torch::kInt64).to(like.device(), torch::kInt64);
    allocations_.RememberAllocation(peers_);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> W8A8BackwardPlan::backward(
    torch::Tensor dy, torch::Tensor packed_p, torch::Tensor dispatch, torch::Tensor scales, torch::Tensor maps,
    torch::Tensor offsets, const std::vector<torch::Tensor>& compressed, const std::vector<torch::Tensor>& weight_pairs,
    double limit, bool capture_timer, const StreamAllocations& inputs)
{
    using namespace MegaMoeTrainW8A8;
    const int64_t e = experts_, h = hidden_, f = ffn_, k = topk_;
    const int64_t n = dy.size(0), r = dispatch.size(0);
    TORCH_CHECK(n > 0 && n <= max_tokens_ && r <= int64_t(world_) * max_tokens_ * std::min(experts_, topk_) &&
                    std::isfinite(limit) && limit >= 0 && limit <= std::numeric_limits<float>::max(),
                "Invalid W8A8 backward shape or activation limit");
    TORCH_CHECK(generation_ < (1U << 31) - 1U, "W8A8 backward generation exhausted");
    const auto p_type = packed_p.scalar_type();
    TORCH_CHECK(p_type == torch::kFloat32 || p_type == torch::kBFloat16, "Invalid W8A8 route weight dtype");
    auto check = [&](const torch::Tensor& t, torch::IntArrayRef dimensions, torch::ScalarType dtype, const char* name) {
        CheckTensor(t, dimensions.size(), name);
        TORCH_CHECK(t.device() == peers_.device() && t.sizes() == dimensions && t.scalar_type() == dtype, name,
                    " must match the W8A8 backward shape");
    };
    constexpr auto fp8 = c10::ScalarType::Float8_e4m3fn;
    constexpr auto e8m0 = c10::ScalarType::Float8_e8m0fnu;
    check(dy, {n, h}, torch::kBFloat16, "dY");
    check(packed_p, {r}, p_type, "packed route weights");
    check(dispatch, {r, h}, fp8, "dispatched X");
    check(scales, {r, h / 32}, e8m0, "dispatched X scales");
    check(maps, {r, 3}, torch::kInt32, "route metadata");
    check(offsets, {e + 1}, torch::kInt32, "expert offsets");

    const bool recompute = compressed.empty();
    TORCH_CHECK(weight_pairs.size() == (recompute ? 6 : 4), "Wrong weight directions for W8A8 backward");
    check(weight_pairs[0], {e, h, 2 * f}, fp8, "W1 dgrad");
    check(weight_pairs[1], {e, h, f / 32, 2}, e8m0, "W1 dgrad scales");
    check(weight_pairs[2], {e, f, h}, fp8, "W2 dgrad");
    check(weight_pairs[3], {e, f, h / 64, 2}, e8m0, "W2 dgrad scales");
    if (recompute) {
        check(weight_pairs[4], {e, 2 * f, h}, fp8, "W1 recompute");
        check(weight_pairs[5], {e, 2 * f, h / 64, 2}, e8m0, "W1 recompute scales");
    } else {
        TORCH_CHECK(compressed.size() == (limit > 0 ? 7 : 4), "Incomplete saved W8A8 Gate/Up");
        for (size_t i = 0; i < 4; ++i) {
            check(compressed[i], {r, i % 2 == 0 ? f : f / 32}, i % 2 == 0 ? fp8 : e8m0, "saved Gate/Up");
        }
        for (size_t i = 4; i < compressed.size(); ++i) {
            check(compressed[i], {r, f / 8}, torch::kUInt8, "saved clipping mask");
        }
    }

    const auto stream = c10_npu::getCurrentNPUStream();
    auto address = [](const torch::Tensor& tensor) { return reinterpret_cast<uint64_t>(tensor.data_ptr()); };
    auto inputAddress = [&](const torch::Tensor& tensor) {
        inputs.RecordIfForeign(tensor);
        return address(tensor);
    };
    // Scratch and outputs are allocated and last used here on the launch stream.
    std::vector<torch::Tensor> temporaries;
    auto scratch = [&](torch::IntArrayRef shape, torch::ScalarType dtype) {
        const auto& tensor = temporaries.emplace_back(torch::empty(shape, dy.options().dtype(dtype)));
        return address(tensor);
    };

    auto dx = torch::empty_like(dy);
    auto dp = torch::empty({n, k}, dy.options().dtype(p_type));
    auto dw1 = torch::empty({e, 2 * f, h}, dy.options());
    auto dw2 = torch::empty({e, h, f}, dy.options());
    TrainParams a{};
    a.rank = rank_;
    a.world = world_;
    a.tokens = n;
    a.maxTokens = max_tokens_;
    a.hidden = h;
    a.ffn = f;
    a.topk = k;
    a.experts = e;
    a.capacity = r;
    a.saveMode = recompute ? kSaveNone : kSaveMxFp8;
    a.generation = ++generation_;
    a.limit = static_cast<float>(limit);
    a.topkWeightType = p_type == torch::kFloat32;
    allocations_.RecordIfForeign(peers_);
    a.peers = address(peers_);
    a.x = inputAddress(dy);
    a.packedP = inputAddress(packed_p);
    a.xq = inputAddress(dispatch);
    a.sx = inputAddress(scales);
    a.maps = inputAddress(maps);
    a.offsets = inputAddress(offsets);
    a.w1b = inputAddress(weight_pairs[0]);
    a.s1b = inputAddress(weight_pairs[1]);
    a.w2b = inputAddress(weight_pairs[2]);
    a.s2b = inputAddress(weight_pairs[3]);
    if (recompute) {
        a.w1 = inputAddress(weight_pairs[4]);
        a.s1 = inputAddress(weight_pairs[5]);
        a.gate = scratch({r, f}, torch::kBFloat16);
        a.up = scratch({r, f}, torch::kBFloat16);
    } else {
        a.gateq = inputAddress(compressed[0]);
        a.sqg = inputAddress(compressed[1]);
        a.upq = inputAddress(compressed[2]);
        a.squ = inputAddress(compressed[3]);
        if (limit > 0) {
            a.gateHigh = inputAddress(compressed[4]);
            a.upLow = inputAddress(compressed[5]);
            a.upHigh = inputAddress(compressed[6]);
        }
    }
    a.output = address(dx);
    a.dpOutput = address(dp);
    a.dw1 = address(dw1);
    a.dw2 = address(dw2);
    // Receive BF16 dY for token-direction quantization independently of dHp.
    a.deff = scratch({r, h}, torch::kBFloat16);
    // MXFP8 reuses each consumed dHp tile for half of its FP32 dP products.
    // Recompute mode retains dHp and forms the products during row reduction.
    a.db = scratch({r, f}, torch::kBFloat16);
    a.hp = recompute ? a.db : scratch({r, f}, torch::kBFloat16);
    // Expert-local feature-format dY is overwritten by dX after dW2 drains.
    a.packedY = scratch({r, h}, torch::kBFloat16);
    a.dzq = scratch({r, 2 * f}, torch::kUInt8);
    a.sz = scratch({r, 2 * f / 32}, torch::kUInt8);
    const int64_t padded = AlignUp(r + e * 63, int64_t{64});
    a.xt = scratch({padded * h}, torch::kUInt8);
    a.sxt = scratch({padded * h / 32}, torch::kUInt8);
    a.dt = scratch({padded * h}, torch::kUInt8);
    a.sdt = scratch({padded * h / 32}, torch::kUInt8);
    a.zt = scratch({padded * 2 * f}, torch::kUInt8);
    a.szt = scratch({padded * 2 * f / 32}, torch::kUInt8);
    a.bt = scratch({padded * f}, torch::kUInt8);
    a.sbt = scratch({padded * f / 32}, torch::kUInt8);
    const int64_t dz_stripes = !recompute && f % 256 == 0 ? f / 256 : 0;
    const int64_t flags = e * (CeilDiv(r, int64_t{256}) + 11 * 2 * platform_.aic + dz_stripes);
    a.pipeline = scratch({flags * 16}, torch::kInt32);
    auto timer = capture_timer
                     ? AllocateW8A8Timer(dy, platform_.aic, r, f, e, k, p_type == torch::kFloat32, true, recompute)
                     : torch::Tensor{};
    auto* timer_addr = timer.defined() ? reinterpret_cast<uint8_t*>(address(timer)) : nullptr;
    mega_moe_w8a8_backward_do(platform_.block_dim, platform_.ub, stream.stream(), &a, ffts_, timer_addr);
    return {dx, dp, dw1, dw2, timer};
}

}  // namespace deepep::megamoe
