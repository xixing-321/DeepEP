// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026
// Modified by guofanfeng 2026

#pragma once

#include <type_traits>

#include "kernel_operator.h"
#include <adv_api/reduce/reduce.h>
#include "common/mega_moe_bf16_layout.hpp"

namespace MegaMoeImpl {

using TopkGradTile = AscendC::LocalTensor<float>;

// With a partial final stripe, CANN pads missing columns with the sum identity.
template <class Tensor>
__aicore__ inline void LoadReductionStripe(TopkGradTile dst, Tensor source, uint32_t col, uint32_t f, uint32_t width)
{
    const uint32_t count = f - col < width ? f - col : width;
    if (count < width) {
        AscendC::Duplicate(dst, 0.0F, width);
        AscendC::PipeBarrier<PIPE_V>();
    }
    if constexpr (std::is_same_v<Tensor, TopkGradTile>) {
        AscendC::Muls(dst, source[col], 1.0F, count);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        AscendC::SyncFunc<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopyPad(dst, source[col], AscendC::DataCopyExtParams{1U, count * 4U, 0U, 0U, 0U}, {});
        AscendC::SyncFunc<AscendC::HardEvent::MTE2_V>();
    }
}

// Match the contiguous AR kernel: each logical core reduces consecutive
// stripes, then the RA pass sums those group results for the same route.
template <class Tensor>
__aicore__ inline TopkGradTile ReduceTopkGradGroupedRow(Tensor products, uint32_t f,
                                                        MegaMoeBF16Impl::TopkGradReductionPlan plan,
                                                        uint32_t scratchOffset)
{
    const uint32_t width = plan.width, stripes = (f + width - 1U) / width;
    const uint32_t aligned = (width + 7U) / 8U * 8U;
    TopkGradTile left{AscendC::TPosition::VECCALC, scratchOffset, aligned};
    TopkGradTile right{AscendC::TPosition::VECCALC, scratchOffset + 4U * aligned, aligned};
    TopkGradTile sums{AscendC::TPosition::VECCALC, scratchOffset + 8U * aligned, 33 * 8};
    const uint32_t count = (stripes + plan.stripesPerGroup - 1U) / plan.stripesPerGroup;
    TopkGradTile groupSums{AscendC::TPosition::VECCALC, scratchOffset + 8U * aligned + 33U * 32U, count * 8U};
    auto scratch = right.template ReinterpretCast<uint8_t>();
    uint32_t arShape[2] = {1U, width};
    for (uint32_t group = 0; group < count; ++group) {
        const uint32_t begin = group * plan.stripesPerGroup;
        const uint32_t size = stripes - begin < plan.stripesPerGroup ? stripes - begin : plan.stripesPerGroup;
        uint32_t main = 1;
        while (2U * main < size) main *= 2U;
        for (uint32_t i = 0; i < main; ++i) {
            LoadReductionStripe(left, products, (begin + i) * width, f, width);
            if (i + main < size) {
                LoadReductionStripe(right, products, (begin + i + main) * width, f, width);
                AscendC::Add(left, left, right, width);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(sums, left, scratch, arShape, false);
            AscendC::PipeBarrier<PIPE_V>();
            uint32_t level = 0;
            for (uint32_t carry = i; carry & 1U; carry >>= 1U) {
                AscendC::Add(sums, sums[(level + 1U) * 8U], sums, 1);
                AscendC::PipeBarrier<PIPE_V>();
                ++level;
            }
            if (i + 1U < main) {
                AscendC::Muls(sums[(level + 1U) * 8U], sums, 1.0F, 1);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        if (count > 1U) {
            AscendC::Muls(groupSums[group * 8U], sums, 1.0F, 1);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    if (count > 1U) {
        uint32_t raShape[2] = {count, 8U};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sums, groupSums, scratch, raShape, false);
    }
    return sums;
}

}  // namespace MegaMoeImpl
