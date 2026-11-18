// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026
// Modified by guofanfeng 2026

#pragma once

#include <type_traits>
#include "kernel_operator.h"

namespace MegaMoeImpl::Bf16Route {
using namespace AscendC;

struct PackTile {
    uint32_t columns, stride, rows;

    __aicore__ PackTile(uint32_t hidden, uint32_t slotElements)
    {
        columns = hidden < slotElements ? hidden : slotElements;
        stride = (columns + 15U) / 16U * 16U;  // BF16 elements per 32-byte DMA block
        rows = stride <= slotElements / 4 ? 4 : (stride <= slotElements / 2 ? 2 : 1);
    }
};

__aicore__ inline void BeginPackRows()
{
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

__aicore__ inline void DrainPackRows()
{
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// rowMap and packed start at the same routed row. The caller owns the two UB slots.
template <class RowMap>
__aicore__ inline void PackRows(RowMap rowMap, GlobalTensor<bfloat16_t> input, GlobalTensor<bfloat16_t> packed,
                                LocalTensor<bfloat16_t> slots, const PackTile& tile, uint32_t count, uint32_t hidden,
                                uint32_t topK, uint32_t& ping)
{
    for (uint32_t first = 0; first < count; first += tile.rows) {
        const uint32_t n = count - first < tile.rows ? count - first : tile.rows;
        uint64_t source[4];
        for (uint32_t r = 0; r < n; ++r) {
            source[r] = static_cast<uint64_t>(rowMap.GetValue(first + r) / topK) * hidden;
        }
        for (uint32_t col = 0; col < hidden; col += tile.columns) {
            const uint32_t width = hidden - col < tile.columns ? hidden - col : tile.columns;
            const auto event = ping == 0 ? EVENT_ID0 : EVENT_ID1;
            auto local = slots[ping * tile.rows * tile.stride];
            WaitFlag<HardEvent::MTE3_MTE2>(event);
            if constexpr (std::is_same_v<RowMap, GlobalTensor<int32_t>>) {
                // Backward obtains source offsets through scalar GM loads.
                if (tile.rows > 1) {
                    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
                }
            }
            for (uint32_t r = 0; r < n; ++r) {
                DataCopyPad(local[r * tile.stride], input[source[r] + col], {1U, width * 2U, 0U, 0U, 0U},
                            DataCopyPadExtParams<bfloat16_t>{false, 0U, 0U, 0U});
            }
            SetFlag<HardEvent::MTE2_MTE3>(event);
            WaitFlag<HardEvent::MTE2_MTE3>(event);
            if (n == 1 || tile.stride == hidden) {
                DataCopyPad(packed[static_cast<uint64_t>(first) * hidden + col], local,
                            {1U, n * width * 2U, 0U, 0U, 0U});
            } else {
                for (uint32_t r = 0; r < n; ++r) {
                    DataCopyPad(packed[static_cast<uint64_t>(first + r) * hidden + col], local[r * tile.stride],
                                {1U, width * 2U, 0U, 0U, 0U});
                }
            }
            SetFlag<HardEvent::MTE3_MTE2>(event);
            ping ^= 1U;
        }
    }
}
}  // namespace MegaMoeImpl::Bf16Route
