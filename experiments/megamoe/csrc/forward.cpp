// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include "csrc/kernels/common/mega_moe_wave_math.hpp"
#include "plan.hpp"
#include "host_utils.hpp"
#include "kernel_launch.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

#include <ATen/MemoryOverlap.h>
#include <acl/acl.h>
#include "shmem.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "tiling/platform/platform_ascendc.h"

#include "check.hpp"
#include "kernels/mega_moe/common/ascend_timer_v2.hpp"
#include "kernels/mega_moe/common/mega_moe_bf16_layout.hpp"
#include "kernels/mega_moe/common/mega_moe_bf16_saved_meta.hpp"
#include "kernels/mega_moe/common/mega_moe_bf16_wave_plan.hpp"
#include "kernels/mega_moe/common/mega_moe_constants.hpp"
#include "kernels/mega_moe/common/mega_moe_workspace.hpp"
#include "kernels/mega_moe/mc2_common/mc2_moe_context.hpp"
#include "kernels/mega_moe/timer/ascend_timer.hpp"

namespace deepep::megamoe {
namespace {

// Raw launches allocate the kernel-visible WorkspaceInfo regions,
// the AscendC system workspace, and a 50 MiB reserve.
constexpr int64_t kReservedWorkspaceBytes = 50LL * 1024LL * 1024LL;

uint32_t DispatchCopyBytes(const MegaMoeTilingData& t)
{
    const uint32_t token = AlignUp(t.h, uint32_t{256});
    const uint32_t scale = AlignUp(CeilDiv(t.h, uint32_t{32}), uint32_t{32});
    return token + scale;
}

MegaMoeDispatchBufferConfig DispatchConfig(const MegaMoeTilingData& t, uint32_t ub)
{
    MegaMoeDispatchBufferConfig c{};
    // Compact-route receive stores at most one local-expert index per source
    // token.  The old mask route used numMaxTokens * topK here.
    const uint64_t total = static_cast<uint64_t>(t.numMaxTokensPerRank);
    const uint64_t aligned = AlignUp(total, uint64_t{256});
    c.copyBufferBytes = DispatchCopyBytes(t);
    const uint32_t fixed =
        AlignUp(t.epWorldSize * t.moeExpertPerRank * 4U, 32U) + AlignUp(t.moeExpertPerRank * 4U, 32U);
    c.routeItemsPerBatch = static_cast<int32_t>(std::min<uint64_t>(aligned, BASE_RECV_ROUTE_ITEMS_PER_BATCH));
    c.routeBatchCount = static_cast<int32_t>(CeilDiv(total, static_cast<uint64_t>(c.routeItemsPerBatch)));
    const uint32_t route = static_cast<uint32_t>(c.routeItemsPerBatch);
    const uint32_t without_slots = fixed + route * sizeof(int32_t);
    const uint32_t budget = ub > without_slots ? ub - without_slots : 0U;
    c.bufferCount = std::clamp(static_cast<int32_t>(budget / (c.copyBufferBytes + 32U)), MIN_DISPATCH_BUFFER_COUNT,
                               MAX_DISPATCH_BUFFER_COUNT);
    if (static_cast<uint64_t>(c.routeItemsPerBatch) < total) {
        const uint32_t fixed_with_slots = fixed + static_cast<uint32_t>(c.bufferCount) * (c.copyBufferBytes + 32U);
        const uint32_t route_budget = ub > fixed_with_slots ? ub - fixed_with_slots : 0U;
        uint32_t expanded = route_budget / sizeof(int32_t);
        expanded = expanded / 256U * 256U;
        expanded = static_cast<uint32_t>(std::min<uint64_t>(expanded, aligned));
        if (expanded > static_cast<uint32_t>(c.routeItemsPerBatch)) {
            c.routeItemsPerBatch = static_cast<int32_t>(expanded);
            c.routeBatchCount = static_cast<int32_t>(CeilDiv(total, static_cast<uint64_t>(expanded)));
        }
    }
    return c;
}

MegaMoeSendMaskBufferConfig SendMaskConfig(const MegaMoeTilingData& t, uint32_t fixed, uint32_t owned_experts,
                                           uint32_t ub)
{
    MegaMoeSendMaskBufferConfig c{};
    const uint64_t total = static_cast<uint64_t>(t.numMaxTokensPerRank) * t.topK;
    const uint64_t aligned = AlignUp(total, uint64_t{256});
    c.routeItemsPerBatch = static_cast<int32_t>(std::min<uint64_t>(aligned, BASE_SEND_ROUTE_ITEMS_PER_BATCH));
    c.routeBatchCount = static_cast<int32_t>(CeilDiv(total, static_cast<uint64_t>(c.routeItemsPerBatch)));
    auto route_slot_bytes = [&](uint32_t route_items) {
        // A batch can begin in the middle of a token's top-k segment.  Reserve
        // one matched index per covered token, including both boundary tokens.
        const uint64_t matched = CeilDiv<uint64_t>(static_cast<uint64_t>(route_items) + t.topK - 1U, t.topK);
        const uint64_t valid_index_bytes = AlignUp<uint64_t>(matched * sizeof(int32_t), 32U);
        return static_cast<uint32_t>(route_items / 8U + valid_index_bytes);
    };
    c.bufferBytes = route_slot_bytes(static_cast<uint32_t>(c.routeItemsPerBatch));
    const uint32_t route_bytes = static_cast<uint32_t>(c.routeItemsPerBatch) * 4U;
    const uint32_t without_masks = fixed + 2U * route_bytes;
    const uint32_t budget = ub > without_masks ? ub - without_masks : 0U;
    c.bufferCount = std::min(static_cast<int32_t>(budget / c.bufferBytes), MAX_SEND_MASK_BUFFER_COUNT);
    const uint64_t pushes = static_cast<uint64_t>(c.routeBatchCount) * owned_experts;
    if (pushes > 0 && static_cast<uint64_t>(c.bufferCount) > pushes) {
        c.bufferCount = static_cast<int32_t>(pushes);
    }
    c.bufferCount = std::max(c.bufferCount, MIN_SEND_MASK_BUFFER_COUNT);
    if (static_cast<uint64_t>(c.routeItemsPerBatch) < total) {
        const uint64_t fixed_with_padding =
            static_cast<uint64_t>(fixed) + static_cast<uint64_t>(c.bufferCount) * (32U + 2U * sizeof(int32_t));
        const uint64_t route_budget = ub > fixed_with_padding ? ub - fixed_with_padding : 0U;
        const uint64_t ring_bits =
            static_cast<uint64_t>(c.bufferCount) +
            CeilDiv<uint64_t>(static_cast<uint64_t>(c.bufferCount) * sizeof(int32_t) * 8U, t.topK);
        uint64_t expanded = route_budget * 8U / (2U * sizeof(int32_t) * 8U + ring_bits);
        expanded = expanded / 256U * 256U;
        expanded = std::min<uint64_t>(expanded, aligned);
        if (expanded > static_cast<uint64_t>(c.routeItemsPerBatch)) {
            c.routeItemsPerBatch = static_cast<int32_t>(expanded);
            c.routeBatchCount = static_cast<int32_t>(CeilDiv(total, static_cast<uint64_t>(expanded)));
            c.bufferBytes = route_slot_bytes(static_cast<uint32_t>(expanded));
        }
    }
    return c;
}

uint64_t HostFlagElements(const MegaMoeTilingData& t)
{
    const uint64_t waves = CeilDiv<uint64_t>(t.maxOutputSize, 256U);
    const uint64_t wave_slots = waves * MegaMoeImpl::INT_CACHELINE;
    uint64_t count = static_cast<uint64_t>(t.moeExpertPerRank) *
                     (wave_slots + wave_slots + static_cast<uint64_t>(MegaMoeImpl::INT_CACHELINE) * t.aicNum);
    const bool is_w4 = t.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4 ||
                       t.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A4W4 ||
                       t.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A4W4_NZ;
    if (is_w4 || (t.topoType == MegaMoeImpl::TOPO_TYPE_MTE && t.combineQuantMode == MegaMoeImpl::COMBINE_NO_QUANT)) {
        // Pairwise GMM2/combine publishes one sequence per AIC.
        count += static_cast<uint64_t>(t.aicNum) * MegaMoeImpl::INT_CACHELINE;
    }
    if (t.topoType == MegaMoeImpl::TOPO_TYPE_MTE && (t.combineQuantMode != MegaMoeImpl::COMBINE_NO_QUANT ||
                                                     t.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4)) {
        // Match WorkspaceInfo: the atomic A8W4 path owns one counter per M-group,
        // while quantized Combine retains one ready slot per AIC.
        const uint64_t ready_slots_per_expert = t.combineQuantMode == MegaMoeImpl::COMBINE_NO_QUANT
                                                    ? wave_slots
                                                    : static_cast<uint64_t>(t.aicNum) * MegaMoeImpl::INT_CACHELINE;
        count += static_cast<uint64_t>(t.moeExpertPerRank) * ready_slots_per_expert;
    }
    if (t.topoType == MegaMoeImpl::TOPO_TYPE_MTE && t.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4 &&
        t.combineQuantMode == MegaMoeImpl::COMBINE_NO_QUANT && t.sharedExpertNum == 0U) {
        count += static_cast<uint64_t>(t.aicNum) * MegaMoeImpl::INT_CACHELINE;
    }
    if (t.sharedExpertNum > 0U) {
        count += CeilDiv<uint64_t>(t.bs, 256U) * t.sharedExpertNum * MegaMoeImpl::INT_CACHELINE;
        if (t.topoType == MegaMoeImpl::TOPO_TYPE_MTE && t.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4) {
            count += 4ULL * t.aicNum * MegaMoeImpl::INT_CACHELINE;
        }
    }
    return count + uint64_t(t.scratchWaveSlots) * MegaMoeImpl::INT_CACHELINE;
}

void SetSendMaskConfigs(MegaMoeTilingData& t, uint32_t ub)
{
    const uint32_t reset_per_core =
        static_cast<uint32_t>(CeilDiv(HostFlagElements(t), static_cast<uint64_t>(t.blockAivNum)));
    const uint32_t reset_count = std::min(reset_per_core, static_cast<uint32_t>(DISPATCH_RESET_BATCH));
    const uint32_t reset_bytes = AlignUp(reset_count, uint32_t{MegaMoeImpl::INT32_PER_256B}) * 4U;
    const uint32_t quant_out = AlignUp(t.h, 256U) + AlignUp(CeilDiv(t.h, 32U), 32U);
    const uint32_t quant_in = AlignUp(t.h, 128U) * 2U;
    const auto sharedPolicy = MegaMoeImpl::SelectSharedExpertPolicy(
        t.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4, t.sharedExpertNum, t.bs);
    const uint32_t route_cores = sharedPolicy.routeAiv1Only ? t.aicNum : t.blockAivNum;
    const uint32_t max_experts = MegaMoeImpl::CalcSendMaskExpertCapacity(&t, route_cores);
    const uint32_t accumulator = AlignUp(max_experts * 4U, 32U);
    const uint32_t fixed = reset_bytes + 2048U + 2U * quant_out + 2U * quant_in + accumulator;
    const uint32_t experts = t.epWorldSize * t.moeExpertPerRank;
    const uint32_t base = experts / route_cores;
    t.sendMaskCoreCountWithExtraExpert = experts % route_cores;
    const bool rail = MegaMoeImpl::UseDecodeRailRouting(&t);
    t.sendMaskConfigForCoreWithExtraExpert = SendMaskConfig(t, fixed, rail ? max_experts : base + 1U, ub);
    t.sendMaskConfigForCoreWithoutExtraExpert = SendMaskConfig(t, fixed, rail ? max_experts : base, ub);
}

MegaMoeUnpermuteBufferConfig UnpermuteConfig(const MegaMoeTilingData& t, uint32_t core_tokens, uint32_t ub)
{
    MegaMoeUnpermuteBufferConfig c{};
    if (core_tokens == 0) return c;
    const uint32_t bf16_bytes = AlignUp(t.h * 2U, 32U);
    const uint32_t fp32_bytes = AlignUp(t.h * 4U, 32U);
    const uint32_t slot_bytes = bf16_bytes + fp32_bytes;
    c.bf16SlotElementCount = bf16_bytes / 2U;
    c.fp32SlotElementCount = fp32_bytes / 4U;
    const uint32_t base_tokens = UNPERMUTE_WEIGHT_ITEMS_PER_BATCH / t.topK;
    c.tokensPerBatch = static_cast<int32_t>(std::min(base_tokens, core_tokens));
    auto set_weight_bytes = [&]() {
        c.topKWeightsBufferBytes = AlignUp(static_cast<uint32_t>(c.tokensPerBatch) * t.topK * 4U, 32U);
        c.topKWeightsConversionBufferBytes = 0U;
    };
    set_weight_bytes();
    const uint32_t before_inputs = c.topKWeightsBufferBytes + slot_bytes;
    const uint32_t input_budget = ub > before_inputs ? ub - before_inputs : 0U;
    c.inputBufferCount = std::clamp(static_cast<int32_t>(input_budget / slot_bytes), MIN_UNPERMUTE_INPUT_BUFFER_COUNT,
                                    MAX_UNPERMUTE_INPUT_BUFFER_COUNT);
    c.inputBufferCount = std::min(c.inputBufferCount, c.tokensPerBatch * static_cast<int32_t>(t.topK));
    c.inputBufferCount = std::max(c.inputBufferCount, MIN_UNPERMUTE_INPUT_BUFFER_COUNT);
    if (base_tokens < core_tokens) {
        const uint32_t fixed = (static_cast<uint32_t>(c.inputBufferCount) + 1U) * slot_bytes;
        const uint32_t weight_budget = ub > fixed + UNPERMUTE_WEIGHT_ALIGNMENT_RESERVE_BYTES
                                           ? ub - fixed - UNPERMUTE_WEIGHT_ALIGNMENT_RESERVE_BYTES
                                           : 0U;
        const uint32_t expanded = std::min(weight_budget / (t.topK * 4U), core_tokens);
        if (expanded > static_cast<uint32_t>(c.tokensPerBatch)) {
            c.tokensPerBatch = static_cast<int32_t>(expanded);
            set_weight_bytes();
        }
    }
    return c;
}

void SetUnpermuteConfigs(MegaMoeTilingData& t, uint32_t ub)
{
    const uint32_t full = CeilDiv(t.bs, t.blockAivNum);
    const uint32_t active = CeilDiv(t.bs, full);
    const uint32_t tail = t.bs - (active - 1U) * full;
    const bool tail_full = tail == full;
    t.unpermuteFullTokenChunkCoreCount = tail_full ? active : active - 1U;
    t.unpermuteConfigForFullTokenChunk = UnpermuteConfig(t, full, ub);
    t.unpermuteConfigForTailTokenChunk = tail_full ? MegaMoeUnpermuteBufferConfig{} : UnpermuteConfig(t, tail, ub);
}

MegaMoeTilingData MakeTiling(int64_t bs, int64_t h, int64_t hidden_dim, int64_t topk, int64_t local_experts,
                             int64_t world, int64_t max_tokens, int64_t max_recv, bool is_a8w4, int64_t shared_experts,
                             const PlatformInfo& p)
{
    MegaMoeTilingData t{};
    t.moeExpertPerRank = static_cast<uint32_t>(local_experts);
    t.bs = static_cast<uint32_t>(bs);
    t.h = static_cast<uint32_t>(h);
    t.hiddenDim = static_cast<uint32_t>(hidden_dim);
    t.epWorldSize = static_cast<uint32_t>(world);
    t.blockNumPerEP = std::max(1U, p.aic / t.epWorldSize);
    t.numMaxTokensPerRank = static_cast<uint32_t>(max_tokens == 0 ? bs : max_tokens);
    t.maxOutputSize = static_cast<uint32_t>(
        max_recv == 0 ? t.numMaxTokensPerRank * t.epWorldSize * std::min<uint32_t>(topk, local_experts) : max_recv);
    t.topK = static_cast<uint32_t>(topk);
    t.aicNum = p.aic;
    t.blockAivNum = p.aiv;
    t.combineQuantMode = MegaMoeImpl::COMBINE_NO_QUANT;
    t.clampLimit = std::numeric_limits<float>::max();
    t.groupedMatmulMode = is_a8w4 ? MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4 : MegaMoeImpl::GROUPED_MATMUL_MODE_GENERAL;
    t.topoType = MegaMoeImpl::TOPO_TYPE_MTE;
    t.sharedExpertNum = static_cast<uint32_t>(shared_experts);
    t.combineSyncSlotCountPerExpert = 0;
    t.topkWeightsPrefetch = 0;
    t.maxTilesPerExpert = 0;
    t.actMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SWIGLU);
    t.actSubMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActSubMode::DEFAULT);
    t.activationAlpha = 1.0F;
    t.activationBeta = 1.0F;
    // 按本次每卡 BS 选择，不用预分配容量；小 batch 缩短 wave，较早释放 GMM2。
    if (is_a8w4 && t.bs <= MegaMoeImpl::W4A8_SMALL_BATCH_MAX_TOKENS) {
        const uint64_t gmm1_tasks_per_m_group =
            CeilDiv<uint64_t>(t.hiddenDim / MegaMoeImpl::ACTIVATION_N_HALF, MegaMoeImpl::GMM1_FUSED_TILE_N);
        t.mGroupsPerWave = MegaMoeImpl::WaveGroups(p.aic, gmm1_tasks_per_m_group, 1U, 2U, false);
    } else {
        // 大 batch 和 A8W8 的预算包括 GMM2 的最低任务量约束。
        const uint64_t gmm1_tiles = CeilDiv<uint64_t>(t.hiddenDim, 256U);
        const uint64_t gmm2_tiles = CeilDiv<uint64_t>(t.h, 256U);
        t.mGroupsPerWave = MegaMoeImpl::WaveGroups(p.aic, gmm1_tiles, gmm2_tiles, 4U);
    }
    // Routed MTE Combine returns the slot after all send lanes finish the Wave.
    if (is_a8w4 && shared_experts == 0 &&
        MegaMoeImpl::WaveScratchRows(t.maxOutputSize, t.mGroupsPerWave * MegaMoeImpl::L1_TILE_M_256) <
            t.maxOutputSize) {
        t.scratchWaveSlots = MegaMoeImpl::kWaveScratchSlots;
    }
    t.isPerExpertWeightTensor = false;
    t.dispatchBufferConfig = DispatchConfig(t, p.ub);
    SetSendMaskConfigs(t, p.ub);
    SetUnpermuteConfigs(t, p.ub);
    return t;
}

}  // namespace

void FillPeerAddresses(uint64_t* addresses, void* window, int64_t world)
{
    for (int64_t peer = 0; peer < world; ++peer) {
        void* address = aclshmem_ptr(window, static_cast<int32_t>(peer));
        TORCH_CHECK(address != nullptr, "MegaMoE cannot resolve SHMEM peer ", peer);
        addresses[peer] = reinterpret_cast<uint64_t>(address);
    }
}

Plan::Plan(torch::Tensor example, int64_t intermediate, int64_t topk, int64_t experts, int64_t world,
           int64_t max_tokens, bool is_a8w4, int64_t shared_experts, int64_t ranks_per_node, uint64_t ffts,
           uint64_t symmetric_address, int64_t symmetric_bytes)
    : ffts_(ffts)
{
    CheckTensor(example, 2, "plan input");
    TORCH_CHECK(example.size(0) > 0 && example.size(0) <= max_tokens && max_tokens <= 65536,
                "Plan BS must be in [1, max_tokens], max_tokens <= 65536");
    TORCH_CHECK(intermediate >= 256 && intermediate <= 4096 && intermediate % 128 == 0,
                "Plan intermediate_hidden must be divisible by 128, in [256,4096]");
    TORCH_CHECK(shared_experts >= 0 && shared_experts <= 8 && (is_a8w4 || shared_experts == 0),
                "Plan shared experts require W4A8, with count in [0,8]");
    TORCH_CHECK(ranks_per_node == 0 || (ranks_per_node > 0 && ranks_per_node <= world && world % ranks_per_node == 0),
                "Plan ranks_per_node must divide EP");
    const auto required = buffer_size(world, experts, max_tokens, topk, example.size(1));
    TORCH_CHECK(symmetric_address != 0 && symmetric_bytes >= required, "MegaMoE symmetric buffer needs ", required,
                " bytes, got ", symmetric_bytes);
    platform_ = GetPlatformInfo();
    tiling_ = MakeTiling(example.size(0), example.size(1), intermediate * 2, topk, experts, world, max_tokens, 0,
                         is_a8w4, shared_experts, platform_);
    tiling_.dispatchRanksPerNode = static_cast<uint32_t>(ranks_per_node);
    static_assert(sizeof(MegaMoeTilingData) == 224, "MegaMoE tiling ABI changed");
    const MegaMoeImpl::WorkspaceInfo info(nullptr, &tiling_);
    const int64_t bytes =
        info.workspaceSize + static_cast<int64_t>(platform_.system_workspace) + kReservedWorkspaceBytes;
    TORCH_CHECK(bytes > info.workspaceSize, "MegaMoE workspace size overflow");
    workspace_ = torch::empty({bytes}, example.options().dtype(torch::kUInt8));
    context_ = torch::empty({static_cast<int64_t>(sizeof(Mc2Aclnn::Mc2MoeContext))}, workspace_.options());

    // peer 地址和 context 只在准备阶段解析，不在每次 forward 中重新分配、同步拷贝。
    Mc2Aclnn::Mc2MoeContext context{};
    const int rank = aclshmem_my_pe();
    TORCH_CHECK(rank >= 0 && rank < world && world <= Mc2Aclnn::HCCL_MAX_RANK_SIZE, "Invalid SHMEM rank/world: ", rank,
                "/", world);
    context.epRankId = static_cast<uint32_t>(rank);
    context.rankSizePerServer = static_cast<uint32_t>(world);
    FillPeerAddresses(context.epHcclBuffer_, reinterpret_cast<void*>(symmetric_address), world);
    ASCEND_DEEPEP_CHECK_ACL(
        aclrtMemcpy(context_.data_ptr(), sizeof(context), &context, sizeof(context), ACL_MEMCPY_HOST_TO_DEVICE));
}

std::tuple<torch::Tensor, torch::Tensor> Plan::forward(
    torch::Tensor x, c10::optional<torch::Tensor> x_scales, torch::Tensor topk_ids, torch::Tensor topk_weights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor weight_scales1, torch::Tensor weight_scales2,
    c10::optional<torch::Tensor> timer, c10::optional<torch::Tensor> shared_weight1,
    c10::optional<torch::Tensor> shared_weight2, c10::optional<torch::Tensor> shared_weight_scales1,
    c10::optional<torch::Tensor> shared_weight_scales2, c10::optional<torch::Tensor> out)
{
    CheckTensor(x, 2, "x");
    CheckTensor(topk_ids, 2, "topk_ids");
    CheckTensor(topk_weights, 2, "topk_weights");
    CheckTensor(weight1, 3, "weight1");
    CheckTensor(weight2, 3, "weight2");
    CheckTensor(weight_scales1, 4, "weight_scales1");
    CheckTensor(weight_scales2, 4, "weight_scales2");
    const bool prequantized = x_scales.has_value();
    TORCH_CHECK(x.scalar_type() == (prequantized ? c10::ScalarType::Float8_e4m3fn : torch::kBFloat16),
                "x must be BF16, or FP8 E4M3 with E8M0 scales");
    if (prequantized) {
        CheckTensor(*x_scales, 2, "x_scales");
        TORCH_CHECK(x_scales->scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                        x_scales->sizes() == torch::IntArrayRef({x.size(0), CeilDiv(x.size(1), int64_t{32})}),
                    "x_scales must be E8M0 [BS, H/32]");
    }
    TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32, "topk_ids must be int32");
    TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32, "topk_weights must be float32");
    const bool is_a8w4 = weight1.scalar_type() == torch::kUInt8;
    if (timer.has_value()) {
#ifndef DEEPEP_MEGAMOE_TIMER
        TORCH_CHECK(false, "Rebuild with DEEPEP_MEGAMOE_TIMER=ON before passing timer");
#endif
        TORCH_CHECK(is_a8w4, "MegaMoE timer is supported only by the A8W4 wave kernel");
        CheckTensor(*timer, 1, "timer");
        TORCH_CHECK(timer->scalar_type() == torch::kInt64, "timer must be int64");
        TORCH_CHECK(timer->numel() == AscendTimer::TOTAL_BUFFER_SIZE, "timer must contain exactly ",
                    AscendTimer::TOTAL_BUFFER_SIZE, " int64 values");
    }
    TORCH_CHECK(weight2.scalar_type() == weight1.scalar_type(), "weight1 and weight2 must use the same dtype");
    TORCH_CHECK(is_a8w4 || weight1.scalar_type() == c10::ScalarType::Float8_e4m3fn,
                "weight1/weight2 must be packed E2M1 uint8 (A8W4) or FP8 E4M3 (A8W8)");
    TORCH_CHECK(weight_scales1.scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                    weight_scales2.scalar_type() == c10::ScalarType::Float8_e8m0fnu,
                "weight scales must be FP8 E8M0 tensors");
    for (const auto* tensor : {&topk_ids, &topk_weights, &weight1, &weight2, &weight_scales1, &weight_scales2}) {
        TORCH_CHECK(tensor->device() == x.device(), "MegaMoE tensors must be on the same NPU");
    }
    for (const auto* tensor : {&x_scales, &timer, &out}) {
        TORCH_CHECK(!tensor->has_value() || tensor->value().device() == x.device(),
                    "optional MegaMoE tensors must be on the same NPU");
    }
    const int64_t bs = x.size(0), h = x.size(1), topk = topk_ids.size(1);
    const int64_t experts = weight1.size(0), hidden = weight1.size(1);
    TORCH_CHECK(bs > 0 && h >= 1024 && h <= 8192 && h % 32 == 0, "invalid x shape for A5 MegaMoE");
    TORCH_CHECK(topk >= 1 && topk <= 32 && topk_ids.sizes() == topk_weights.sizes() && topk_ids.size(0) == bs,
                "top-k tensors must have matching [bs, topk] shapes");
    TORCH_CHECK(hidden >= 512 && hidden <= 8192 && hidden % 256 == 0, "invalid weight1 hiddenDim");
    const int64_t weight1_last = is_a8w4 ? h / 2 : h;
    const int64_t weight2_last = is_a8w4 ? hidden / 4 : hidden / 2;
    TORCH_CHECK(weight1.size(2) == weight1_last,
                is_a8w4 ? "packed A8W4 weight1 shape must be [local_experts, hiddenDim, h/2]"
                        : "A8W8 weight1 shape must be [local_experts, hiddenDim, h]");
    TORCH_CHECK(weight2.sizes() == torch::IntArrayRef({experts, h, weight2_last}),
                is_a8w4 ? "packed A8W4 weight2 shape must be [local_experts, h, hiddenDim/4]"
                        : "A8W8 weight2 shape must be [local_experts, h, hiddenDim/2]");
    TORCH_CHECK(weight_scales1.sizes() == torch::IntArrayRef({experts, hidden, CeilDiv(h, int64_t{64}), 2}),
                "weight_scales1 shape must be [local_experts, hiddenDim, ceil(h/64), 2]");
    TORCH_CHECK(weight_scales2.sizes() == torch::IntArrayRef({experts, h, CeilDiv(hidden / 2, int64_t{64}), 2}),
                "weight_scales2 shape must be [local_experts, h, ceil(hiddenDim/2/64), 2]");
    TORCH_CHECK(x.device() == workspace_.device() && bs == tiling_.bs && h == tiling_.h &&
                    hidden == tiling_.hiddenDim && topk == tiling_.topK && experts == tiling_.moeExpertPerRank &&
                    is_a8w4 == (tiling_.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4),
                "Input/device/weights differ from the prepared plan");

    const bool has_shared = shared_weight1.has_value();
    TORCH_CHECK(shared_weight2.has_value() == has_shared && shared_weight_scales1.has_value() == has_shared &&
                    shared_weight_scales2.has_value() == has_shared,
                "all four shared weight and scale tensors must be supplied together");
    int64_t shared_experts = 0;
    if (has_shared) {
        TORCH_CHECK(is_a8w4, "shared experts currently require A8W4 weights");
        CheckTensor(*shared_weight1, 3, "shared_weight1");
        CheckTensor(*shared_weight2, 3, "shared_weight2");
        CheckTensor(*shared_weight_scales1, 4, "shared_weight_scales1");
        CheckTensor(*shared_weight_scales2, 4, "shared_weight_scales2");
        shared_experts = shared_weight1->size(0);
        TORCH_CHECK(shared_experts > 0 && shared_experts <= std::numeric_limits<uint32_t>::max(),
                    "shared expert count must be positive and fit uint32");
        for (const auto* tensor :
             {&*shared_weight1, &*shared_weight2, &*shared_weight_scales1, &*shared_weight_scales2}) {
            TORCH_CHECK(tensor->device() == x.device(), "shared tensors must be on the same device as x");
        }
        TORCH_CHECK(shared_weight1->scalar_type() == torch::kUInt8 && shared_weight2->scalar_type() == torch::kUInt8,
                    "shared weights must use packed E2M1 uint8 NZ storage");
        TORCH_CHECK(shared_weight_scales1->scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                        shared_weight_scales2->scalar_type() == c10::ScalarType::Float8_e8m0fnu,
                    "shared weight scales must be FP8 E8M0");
        TORCH_CHECK(shared_weight1->sizes() == torch::IntArrayRef({shared_experts, hidden, h / 2}) &&
                        shared_weight2->sizes() == torch::IntArrayRef({shared_experts, h, hidden / 4}),
                    "shared weights must match routed H and hiddenDim: [S, hiddenDim, H/2], [S, H, hiddenDim/4]");
        TORCH_CHECK(shared_weight_scales1->sizes() ==
                            torch::IntArrayRef({shared_experts, hidden, CeilDiv(h, int64_t{64}), 2}) &&
                        shared_weight_scales2->sizes() ==
                            torch::IntArrayRef({shared_experts, h, CeilDiv(hidden / 2, int64_t{64}), 2}),
                    "shared scales must match [S, N, ceil(K/64), 2] for each shared weight");
    }

    TORCH_CHECK(shared_experts == tiling_.sharedExpertNum, "Shared experts differ from the prepared plan");
    torch::Tensor y = out.has_value() ? *out : torch::empty(x.sizes(), x.options().dtype(torch::kBFloat16));
    CheckTensor(y, 2, "out");
    TORCH_CHECK(y.scalar_type() == torch::kBFloat16 && y.sizes() == x.sizes(), "out must be BF16 [BS, H]");
    torch::Tensor expert_tokens = torch::empty({experts}, x.options().dtype(torch::kInt32));

    const auto stream = c10_npu::getCurrentNPUStream();
    auto ptr = [](const torch::Tensor& tensor) { return static_cast<uint8_t*>(tensor.data_ptr()); };
    auto opt = [&](const c10::optional<torch::Tensor>& tensor) { return tensor ? ptr(*tensor) : nullptr; };
    mega_moe_kernel_do(platform_.block_dim, platform_.ub, stream.stream(), is_a8w4, ptr(context_), ptr(x),
                       opt(x_scales), ptr(topk_ids), ptr(topk_weights), ptr(weight1), ptr(weight2), ptr(weight_scales1),
                       ptr(weight_scales2), opt(shared_weight1), opt(shared_weight2), opt(shared_weight_scales1),
                       opt(shared_weight_scales2), ptr(y), ptr(expert_tokens), ptr(workspace_), opt(timer),
                       reinterpret_cast<const uint8_t*>(&tiling_), ffts_);
    // The direct kernel launch bypasses the PyTorch dispatcher; record that
    // every argument remains live until the current NPU stream reaches it.
    for (const auto* tensor : {&x, &topk_ids, &topk_weights, &weight1, &weight2, &weight_scales1, &weight_scales2,
                               &workspace_, &context_, &y, &expert_tokens}) {
        c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    }
    if (timer.has_value()) {
        c10_npu::NPUCachingAllocator::recordStream(timer->storage().data_ptr(), stream);
    }
    if (has_shared) {
        for (const auto* tensor :
             {&*shared_weight1, &*shared_weight2, &*shared_weight_scales1, &*shared_weight_scales2}) {
            c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
        }
    }
    if (x_scales) {
        c10_npu::NPUCachingAllocator::recordStream(x_scales->storage().data_ptr(), stream);
    }
    return {y, expert_tokens};
}

int64_t buffer_size(int64_t world, int64_t experts, int64_t tokens, int64_t topk, int64_t hidden)
{
    TORCH_CHECK(world >= 2 && world <= 128 && experts >= 1 && experts <= 128 && tokens >= 1 && tokens <= 65536 &&
                    topk >= 1 && topk <= 32 && topk <= world * experts && hidden >= 1024 && hidden <= 8192 &&
                    hidden % 64 == 0,
                "Invalid buffer shape");
    // The buffer outlives BS/weight-mode changes. Reserve the decode rail/count
    // layout as well; prefill continues to bind its original device offsets.
    const MegaMoeImpl::PeermemSizeParams params{
        tokens, topk, hidden, experts, world, 2, 1, false, false, MegaMoeImpl::TOPO_TYPE_MTE, true};
    return AlignUp<int64_t>(MegaMoeImpl::EXCEPTION_DUMP_REGION_SIZE + MegaMoeImpl::CalcPeermemLeastSize(params),
                            2LL << 20);
}

}  // namespace deepep::megamoe

namespace {

using deepep::megamoe::AlignUp;
using deepep::megamoe::CeilDiv;

constexpr int64_t ALIGN_128 = 128;
constexpr int64_t ALIGN_256 = 256;
constexpr int64_t ALIGN_512 = 512;
constexpr int64_t PEERMEM_DATA_OFFSET = 60 * 1024;
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
    ws.totalSize += AlignUp(SIZE_BF_16 * static_cast<int64_t>(td.maxOutputSize) * td.k, ALIGN_512);
    ws.totalSize += AlignUp(SIZE_BF_16 * static_cast<int64_t>(td.maxOutputSize) * td.n / SWIGLU_N_HALF, ALIGN_512);
    ws.totalSize += AlignUp(static_cast<int64_t>(td.epWorldSize) * td.expertPerRank * SIZE_INT_32, ALIGN_512);
    ws.totalSize += SIZE_INT_32 * AlignUp(static_cast<int64_t>(td.m), ALIGN_256) * td.topK * 2;
    ws.totalSize += SIZE_INT_32 * AlignUp(static_cast<int64_t>(td.epWorldSize) * td.expertPerRank, ALIGN_128);
    ws.totalSize += AlignUp(static_cast<int64_t>(SIZE_INT_32) * td.expertPerRank * INT_CACHELINE, ALIGN_128);
    return ws;
}

inline size_t CalcBf16EarlyTopkWeightWorkspaceSize(const MegaMoeBf16TilingData& td)
{
    return AlignUp(
        static_cast<int64_t>(td.maxOutputSize) * MegaMoeBF16Training::TopkWeightBytes(td.topkWeightType), ALIGN_512);
}

enum PeerMemIdx {
    PEER_TOKEN_PER_EXPERT = 0,
    PEER_COMM_FLAG,
    PEER_PTR_A0,
    PEER_PTR_A1,
    PEER_PTR_D,

    PEER_MEM_NUM
};

struct PeerMemInfo {
    size_t totalSize = 0;
    size_t offset[PEER_MEM_NUM] = {0};
};

PeerMemInfo CalcPeermemSize(const MegaMoeBf16TilingData& td)
{
    PeerMemInfo pm;
    // Keep this host calculation consistent with device-side PeermemInfo.
    pm.totalSize = PEERMEM_DATA_OFFSET;
    pm.totalSize += td.epWorldSize * AlignUp(static_cast<int64_t>(td.epWorldSize) * td.expertPerRank, ALIGN_128) *
                    SIZE_INT_32;
    // Source-side packed X and returned expert outputs each hold one BF16 row per route.
    pm.totalSize += 2 * static_cast<int64_t>(td.m) * td.topK * td.k * SIZE_BF_16;

    return pm;
}

inline size_t CalcBf16EarlyTopkWeightPeermemSize(const MegaMoeBf16TilingData& td)
{
    return AlignUp(static_cast<int64_t>(td.m) * td.topK * MegaMoeBF16Training::TopkWeightBytes(td.topkWeightType),
                         ALIGN_512);
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

BufferSizes MakeBf16Tiling(MegaMoeBf16TilingData& tiling, int64_t m, int64_t k, int64_t n, int64_t topk, int64_t world_size,
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
    const auto peer_info = optiling::CalcPeermemSize(tiling);
    uint64_t peerBytes = static_cast<uint64_t>(peer_info.totalSize);
    {
        // CalcPeermemSize retains the 60 KiB legacy header used by the
        // non-BF16 and experimental variants. Ordinary BF16 uses its 72 KiB
        // synchronization header and, at large EP, an aligned compact-count
        // inbox before the unchanged dense token-count table.
        const MegaMoeBF16Training::CompactTokenCountExchangeLayout countLayout(static_cast<uint64_t>(world_size),
                                                                               static_cast<uint64_t>(expert_per_rank),
                                                                               tiling.compactTokenCountExchange != 0);
        peerBytes += countLayout.tokenCountsOffset - MegaMoeBF16Training::kPeerBaseDataOffset;
    }
    return {workspace_info.totalSize + optiling::CalcBf16EarlyTopkWeightWorkspaceSize(tiling),
            peerBytes + optiling::CalcBf16EarlyTopkWeightPeermemSize(tiling)};
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
    const int64_t rows = phase == 0 ? 192 : 128;
    const int64_t tiles =
        CeilDiv(maxRecv + e * (rows - 1), rows) * CeilDiv(std::max(h, 2 * f), int64_t{256});
    const int64_t dwTiles =
        e * std::max(CeilDiv(2 * f, int64_t{128}) * CeilDiv(h, int64_t{256}), CeilDiv(h, int64_t{128}) * CeilDiv(f, int64_t{256}));
    const int64_t gemmTasks = CeilDiv(std::max(tiles, phase == 0 ? int64_t(0) : dwTiles), groups);
    // Short dP sends from every peer/expert pair can all be owned by the first AIV.
    const int64_t sendTasks = phase == 0 ? 0 : int64_t(world_size_) * e;
    const int64_t bound = std::max(gemmTasks, sendTasks) + 4 * e + 32;
    const int capacity = static_cast<int>(CeilDiv(std::max(bound, int64_t(128)), int64_t{64}) * 64);
    TORCH_CHECK(capacity <= kMaxCapacity, "training Timer task capacity exceeds supported bound");
    const Layout layout{static_cast<int>(groups * 3), capacity};
    TORCH_CHECK(layout.Elements() * 8 <= 256LL * 1024 * 1024, "training Timer exceeds 256 MiB per phase");
    auto timer = torch::zeros({layout.Elements()}, like.options().dtype(torch::kInt64));
    const int64_t saved = mode == "none" ? 0 : (mode == "bf16" ? 1 : 2);
    const int64_t flags = saved | (mega_moe_bf16_swiglu_decoupled_for_saved_mode(h, saved) ? 256 : 0) |
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
    const auto host = MakeBf16Tiling(tiling, max_tokens, hidden, 2 * intermediate, topk, world, local_experts, max_recv, 0,
                                 0.0, "bf16", MegaMoeBF16Training::kTopkWeightFp32);
    const uint64_t total = host.peer_bytes;
    TORCH_CHECK(total <= uint64_t(std::numeric_limits<int64_t>::max()) - kDeviceLargePageBytes,
                "BF16 symmetric capacity overflow");
    return static_cast<int64_t>(deepep::megamoe::AlignUp(total, uint64_t{kDeviceLargePageBytes}));
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
    const auto host = MakeBf16Tiling(tiling, m, h, 2 * f, topk, world_size_, local_experts, max_recv, rank_, limit,
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

ForwardResult BF16Plan::forward(torch::Tensor out, torch::Tensor x, torch::Tensor topkIds, torch::Tensor topkWeights,
                                torch::Tensor weight1, torch::Tensor weight2, int64_t maxRecvTokens,
                                double swigluLimit, const std::string& save_gmm1_output, bool captureTimer,
                                bool saveDispatchedX)
{
    TORCH_CHECK(address_ != 0, "training operator has been closed");
    TORCH_CHECK(out.defined(), "y must be a defined output tensor");
    const torch::Tensor timer =
        captureTimer ? AllocateTimer(x, topkWeights, weight1, weight2, maxRecvTokens, 0, save_gmm1_output)
                     : torch::Tensor();
    auto context = Context(x);
    const int worldSize = world_size_;
    CheckTensor(x, 2, "x");
    CheckTensor(topkIds, 2, "topk_ids");
    CheckTensor(topkWeights, 2, "topk_weights");
    CheckTensor(weight1, 3, "weight1");
    CheckTensor(weight2, 3, "weight2");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(topkIds.scalar_type() == torch::kInt32, "topk_ids must be int32");
    TORCH_CHECK(topkWeights.scalar_type() == torch::kBFloat16 || topkWeights.scalar_type() == torch::kFloat32,
                "topk_weights must be bfloat16 or float32");
    const uint32_t topkWeightType = topkWeights.scalar_type() == torch::kFloat32 ? MegaMoeBF16Training::kTopkWeightFp32
                                                                                 : MegaMoeBF16Training::kTopkWeightBf16;
    const auto topkWeightBytes = MegaMoeBF16Training::TopkWeightBytes(topkWeightType);
    TORCH_CHECK(weight1.scalar_type() == torch::kBFloat16, "weight1 must be bfloat16");
    TORCH_CHECK(weight2.scalar_type() == torch::kBFloat16, "weight2 must be bfloat16");
    TORCH_CHECK(topkWeights.sizes() == topkIds.sizes(), "topk_weights shape must match topk_ids");
    TORCH_CHECK(x.size(0) == topkIds.size(0), "x and topk_ids must have the same token count");

    if (out.defined()) {
        CheckTensor(out, 2, "y");
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
    const auto tiling = TilingForCall(m, k, n, topk, expertPerRank, maxRecvTokens, swigluLimit,
                                      weight_before_gmm2_ ? save_gmm1_output : "none", topkWeightType,
                                      weightLayout.transW);

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

    const auto stream = c10_npu::getCurrentNPUStream();
    auto ptr = [](const torch::Tensor& t) { return static_cast<uint8_t*>(t.data_ptr()); };
    const uint32_t block_dim =
        static_cast<uint32_t>(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic());
    if (timer.defined()) {
        mega_moe_bf16_forward_timer_kernel_do(
            block_dim, stream.stream(), ptr(context), ptr(x), ptr(topkIds), ptr(topkWeights), ptr(weight1),
            ptr(weight2), ptr(y), ptr(expertTokenNums), ptr(routingState),
            dispatchOutput.defined() ? ptr(dispatchOutput) : nullptr, ptr(workspace_),
            reinterpret_cast<const uint8_t*>(&tiling), ffts_addr_, ptr(timer));
    } else {
        mega_moe_bf16_forward_kernel_do(
            block_dim, stream.stream(), ptr(context), ptr(x), ptr(topkIds), ptr(topkWeights), ptr(weight1),
            ptr(weight2), ptr(y), ptr(expertTokenNums), ptr(routingState),
            dispatchOutput.defined() ? ptr(dispatchOutput) : nullptr, ptr(workspace_),
            reinterpret_cast<const uint8_t*>(&tiling), ffts_addr_);
    }
    for (const auto* tensor : {&context, &x, &topkIds, &topkWeights, &weight1, &weight2, &y, &expertTokenNums,
                               &routingState, &dispatchOutput, &workspace_}) {
        if (tensor->defined()) c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    }
    if (timer.defined()) c10_npu::NPUCachingAllocator::recordStream(timer.storage().data_ptr(), stream);
    return {y, expertTokenNums, weight_before_gmm2_ ? routingState : torch::Tensor(), dispatchOutput, timer};
}

}  // namespace deepep::megamoe::bf16
