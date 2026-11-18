// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

#pragma once
#include <cstdint>
#if defined(__NPU_DEVICE__) || defined(__ASC_NPU_HOST__)
#include "kernel_operator.h"
#define MEGA_MOE_WAVE_MATH_HD __host_aicore__
#else
#define MEGA_MOE_WAVE_MATH_HD
#endif
namespace MegaMoeImpl {
// Budget is measured in tasks per AIC. Callers provide payload-specific tile
// counts (paired Gate/Up or quantized output), not a fixed 256-column assumption.
MEGA_MOE_WAVE_MATH_HD constexpr uint32_t WaveGroups(uint32_t cores, uint64_t gmm1Tiles, uint64_t gmm2Tiles,
                                                    uint32_t gmm1Budget, bool constrainGmm2 = true)
{
    if (!cores || !gmm1Tiles || (constrainGmm2 && !gmm2Tiles)) return 1U;
    const uint64_t first = (uint64_t(cores) * gmm1Budget + gmm1Tiles - 1U) / gmm1Tiles;
    const uint64_t second = constrainGmm2 ? (uint64_t(cores) + gmm2Tiles - 1U) / gmm2Tiles : 1U;
    return static_cast<uint32_t>(first > second ? first : second);
}
}  // namespace MegaMoeImpl
#undef MEGA_MOE_WAVE_MATH_HD
