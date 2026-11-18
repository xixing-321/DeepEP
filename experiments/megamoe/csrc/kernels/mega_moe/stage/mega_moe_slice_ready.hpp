// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

#pragma once
#include "kernel_operator.h"

namespace MegaMoeImpl {
// One cacheline per expert-local M tile. Each completed N tile contributes
// one arrival; consumers wait for all columns of every row tile they read.
// Packed expert bases are not necessarily tile-aligned, so reserve one gap
// per expert instead of rounding the complete receive buffer down.
struct RowSliceReady {
    __gm__ int32_t* slots{nullptr};
    uint64_t first{0};
    uint32_t tileRows{128};
    __aicore__ inline RowSliceReady() = default;
    __aicore__ inline RowSliceReady(GM_ADDR base, uint64_t expertBase, uint32_t expert, uint32_t m)
        : slots(reinterpret_cast<__gm__ int32_t*>(base)), first(expertBase / m + expert), tileRows(m)
    {
    }
    __aicore__ inline __gm__ int32_t* Slot(uint32_t row) const
    {
        return slots + (first + row / tileRows) * 16U;
    }
    __aicore__ inline void Publish(uint32_t row) const
    {
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
        AscendC::AtomicAdd(Slot(row), int32_t(1));
    }
    __aicore__ inline void Wait(uint32_t row, uint32_t rows, uint32_t columns) const
    {
        if (slots == nullptr || rows == 0) return;
        const uint32_t end = (row + rows - 1U) / tileRows;
        for (uint32_t tile = row / tileRows; tile <= end; ++tile)
            while (AscendC::ReadGmByPassDCache(slots + (first + tile) * 16U) < static_cast<int32_t>(columns)) {
            }
    }
};
}  // namespace MegaMoeImpl
