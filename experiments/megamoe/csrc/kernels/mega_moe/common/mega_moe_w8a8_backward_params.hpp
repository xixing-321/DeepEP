// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once
#include <cstdint>

namespace MegaMoeTrainW8A8 {
// Explicit uint64 address ABI, populated by the W8A8 C++ Host. Forward and
// backward have separate entry points. The framework supplies loss/dY.
struct TrainParams {
    uint32_t rank, world, tokens, maxTokens, hidden, ffn, topk, experts;
    uint32_t capacity, saveMode, generation, topkWeightType;  // 0=BF16, 1=FP32 p/dp.
    float limit;
    uint32_t padding;
    uint64_t peers, x, ids, p, output, counts, offsets, maps, packedP;
    uint64_t w1, s1, w2, s2, w1b, s1b, w2b, s2b;
    uint64_t xq, sx, gate, up, gateq, sqg, upq, squ, gateHigh, upLow, upHigh;
    uint64_t bq, sb, packedY, dq, sd, db, dz, hp, dzq, sz, xeff, deff;
    uint64_t dpOutput, dw1, dw2;
    uint64_t xt, sxt, dt, sdt, zt, szt, bt, sbt;
    uint64_t reservedProgress;  // Unused ABI slot; no diagnostic buffer.
    uint64_t routingScratch;
    uint64_t pipeline;
};
static_assert(sizeof(TrainParams) == 488, "Host/device training ABI size mismatch");

struct PeerLayout {
    uint64_t signals, numTokens, xq, result, dp, total;
#if defined(__NPU_DEVICE__) || defined(__ASC_NPU_HOST__)
    __host_aicore__
#elif defined(__CCE__)
    __host__ __aicore__
#endif
    PeerLayout(uint32_t world, uint32_t maxTokens, uint32_t h, uint32_t topk, uint32_t, uint32_t topkWeightType = 0)
    {
        // Dedicated phase/rank cache lines. Monotonic generation comparisons
        // tolerate a faster rank advancing without losing a previous signal.
        signals = 0;
        numTokens = 3ULL * world * 64;
        xq = (numTokens + 512 + 511) / 512 * 512;
        result = xq + uint64_t(maxTokens) * h * sizeof(uint16_t);
        dp = result + uint64_t(maxTokens) * topk * h * sizeof(uint16_t);
        total = dp + uint64_t(maxTokens) * topk * (topkWeightType ? 4U : 2U);
        total = (total + 2097151) / 2097152 * 2097152;
    }
};
}  // namespace MegaMoeTrainW8A8
