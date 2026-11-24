// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Per-token transport, with independent data/scale addresses and two UB slots.
#pragma once

__aicore__ inline uint32_t FusedTrain::SourceRow(uint32_t e, uint32_t source) const
{
    uint32_t lo = 0, hi = Rows(e);
    const uint32_t begin = Begin(e);
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (uint32_t(I32(a.maps, uint64_t(begin + mid) * 3)) < source)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

__aicore__ inline void FusedTrain::DispatchDy(uint32_t e, uint32_t first, uint32_t rows, uint32_t& ping)
{
    const uint32_t begin = Begin(e);
    const uint64_t raw = a.deff + uint64_t(begin + first) * a.hidden * 2;
    const uint64_t q = DqData(e);
    const uint64_t s = DqScale(e);
    // The row map occupies [64,67) KiB between the primary input and the MXFP8 prefetch slot.
    LocalTensor<int32_t> maps{TPosition::VECIN, 65536, 768};
    GlobalTensor<int32_t> mg;
    mg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.maps));
    mg.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    uint32_t lastSource = 0xffffffffU;
    uint64_t remoteBase = 0;
    for (uint32_t mapBegin = 0; mapBegin < rows; mapBegin += 256) {
        const uint32_t count = rows - mapBegin < 256 ? rows - mapBegin : 256;
        SyncFunc<HardEvent::S_MTE2>();
        DataCopyPad(maps, mg[uint64_t(begin + first + mapBegin) * 3], DataCopyExtParams{1, count * 12, 0, 0, 0},
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SyncFunc<HardEvent::MTE2_S>();
        // Feature groups are contiguous within each route. Quantize the
        // same UB rows that are written to raw dY, including a short last batch.
        if (a.hidden % 256 == 0 && a.hidden <= 32768) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            // Both input slots end before the quantization scratch at 128 KiB.
            constexpr uint32_t secondSlot = 68 * 1024;
            constexpr uint32_t slotElements = (128 * 1024 - secondSlot) / sizeof(bfloat16_t);
            const bool prefetch = a.saveMode == kSaveMxFp8 && a.hidden <= slotElements;
            const uint32_t batchLimit = (prefetch ? slotElements : 32768) / a.hidden;
            const auto loadRows = [&](LocalTensor<bfloat16_t> input, uint32_t row, uint32_t batch) __aicore__ {
                for (uint32_t r = 0; r < batch; ++r) {
                    const uint32_t source = maps.GetValue((row + r) * 3);
                    const uint32_t token = maps.GetValue((row + r) * 3 + 1);
                    if (source != lastSource) {
                        remoteBase = reinterpret_cast<uint64_t>(Peer(source, peer.xq));
                        lastSource = source;
                    }
                    GlobalTensor<bfloat16_t> src;
                    src.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(remoteBase) + uint64_t(token) * a.hidden);
                    src.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                    DataCopyPad(input[r * a.hidden], src, DataCopyExtParams{1, a.hidden * 2, 0, 0, 0},
                                DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
                }
            };
            uint32_t slot = 0;
            LocalTensor<bfloat16_t> input{TPosition::VECIN, 0, 32768};
            loadRows(input, 0, count < batchLimit ? count : batchLimit);
            for (uint32_t row = 0; row < count; row += batchLimit) {
                const uint32_t batch = count - row < batchLimit ? count - row : batchLimit;
                SyncFunc<HardEvent::MTE2_MTE3>();
                SyncFunc<HardEvent::MTE2_V>();
                const uint32_t next = row + batch;
                const uint32_t nextCount = count - next < batchLimit ? count - next : batchLimit;
                LocalTensor<bfloat16_t> other{TPosition::VECIN, slot ? 0U : secondSlot, slotElements};
                if (prefetch && nextCount) loadRows(other, next, nextCount);
                GlobalTensor<bfloat16_t> dst;
                dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(raw) + uint64_t(mapBegin + row) * a.hidden);
                dst.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                DataCopyPad(dst, input, DataCopyExtParams{1, batch * a.hidden * 2, 0, 0, 0});
                quant.QuantizeTile(input, Addr(q), Addr(s), batch * a.hidden / 256, 256, 256,
                                   uint64_t(first + mapBegin + row) * a.hidden, true);
                if (nextCount) {
                    if (prefetch) {
                        input = other;
                        slot ^= 1;
                    } else {
                        loadRows(input, next, nextCount);
                    }
                }
            }
            SyncFunc<HardEvent::MTE3_S>();
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            continue;
        }
        for (uint32_t row = 0; row < count; ++row) {
            const uint32_t source = maps.GetValue(row * 3), token = maps.GetValue(row * 3 + 1);
            if (source != lastSource) {
                remoteBase = reinterpret_cast<uint64_t>(Peer(source, peer.xq));
                lastSource = source;
            }
            for (uint32_t col = 0; col < a.hidden * 2; col += 16384) {
                const uint32_t size = a.hidden * 2 - col < 16384 ? a.hidden * 2 - col : 16384;
                const uint64_t src = remoteBase + uint64_t(token) * a.hidden * 2 + col;
                const uint64_t dst = raw + uint64_t(mapBegin + row) * a.hidden * 2 + col;
                const uint32_t slot = ping++ % 2;
                const auto id = slot ? EVENT_ID1 : EVENT_ID0;
                LocalTensor<uint8_t> buffer{TPosition::VECIN, slot * 32768, 16384};
                GlobalTensor<uint8_t> gs, gd;
                gs.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(src));
                gd.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(dst));
                gs.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                gd.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                WaitFlag<HardEvent::MTE3_MTE2>(id);
                DataCopyPad(buffer, gs, DataCopyExtParams{1, size, 0, 0, 0},
                            DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_MTE3>(id);
                WaitFlag<HardEvent::MTE2_MTE3>(id);
                DataCopyPad(gd, buffer, DataCopyExtParams{1, size, 0, 0, 0});
                SetFlag<HardEvent::MTE3_MTE2>(id);
            }
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        // Feature quantization of exactly this producer's received fragment.
        // Publish ready only after raw BF16 and feature data/scales are final.
        LocalTensor<bfloat16_t> input{TPosition::VECIN, 0, 32768};
        for (uint32_t row = 0; row < count; row += 128) {
            const uint32_t batch = count - row < 128 ? count - row : 128;
            for (uint32_t col = 0; col < a.hidden; col += 256) {
                const uint32_t columns = a.hidden - col < 256 ? a.hidden - col : 256;
                LoadBFMatrix(input, raw + (uint64_t(mapBegin + row) * a.hidden + col) * 2, batch, columns, a.hidden);
                const uint64_t offset = uint64_t(first + mapBegin + row) * a.hidden + col;
                quant.QuantizeTile(input, Addr(q), Addr(s), batch, columns, a.hidden, offset, true);
            }
        }
        SyncFunc<HardEvent::MTE3_S>();
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}
