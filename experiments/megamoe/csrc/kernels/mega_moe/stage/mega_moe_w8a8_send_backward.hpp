// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Included inside MegaMoeTrainW8A8 after FusedTrain is declared. The producer has already published
// the completed GM output tile; no Cube acknowledgement is required because
// packedY retains all tiles until the invocation's communication drain.
#pragma once

__aicore__ inline void FusedTrain::SendTile(uint32_t e, uint32_t localRow, uint32_t col, uint32_t rows, uint32_t cols)
{
    if (rows == 0 || cols == 0) return;

    // A 256 x 256 BF16 tile occupies 128 KiB. Routing metadata lives above
    // the payload, so all remote stores can remain in flight together. This
    // follows ordinary CombineTokens' stable-UB, enqueue-many-stores pattern,
    // while preserving this training operator's per-token route addresses.
    // The caller supplies M/N tiles <= 256. Pad UB rows for short N tails.
    const uint32_t ubStride = (cols + 15U) / 16U * 16U;
    const uint32_t start = Begin(e) + localRow;
    LocalTensor<bfloat16_t> tile{TPosition::VECIN, 0, 65536};
    LocalTensor<int32_t> maps{TPosition::VECIN, 131072, 768};
    GlobalTensor<bfloat16_t> source;
    GlobalTensor<int32_t> mapSource;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a.packedY) + uint64_t(start) * a.hidden + col);
    mapSource.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.maps) + uint64_t(start) * 3U);
    source.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    mapSource.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

    // These UB regions are reused sequentially by activation, quantization
    // and send helpers on this AIV; protect prior readers before replacing it.
    SyncFunc<HardEvent::MTE3_MTE2>();
    SyncFunc<HardEvent::V_MTE2>();
    SyncFunc<HardEvent::S_MTE2>();
    DataCopyPad(tile, source, DataCopyExtParams{uint16_t(rows), cols * 2U, (a.hidden - cols) * 2U, 0, 0},
                DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
    DataCopyPad(maps, mapSource, DataCopyExtParams{1, rows * 3U * 4U, 0, 0, 0},
                DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    SyncFunc<HardEvent::MTE2_S>();
    SyncFunc<HardEvent::MTE2_MTE3>();

    uint32_t lastRank = 0xffffffffU;
    uint64_t remoteBase = 0;
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t rank = maps.GetValue(row * 3U);
        if (rank != lastRank) {
            remoteBase = reinterpret_cast<uint64_t>(Peer(rank, 0));
            lastRank = rank;
        }
        const uint32_t token = maps.GetValue(row * 3U + 1U);
        const uint32_t slot = maps.GetValue(row * 3U + 2U);
        const uint64_t route = uint64_t(token) * a.topk + slot;
        GlobalTensor<bfloat16_t> target;
        target.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(remoteBase + peer.result + (route * a.hidden + col) * 2U));
        target.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        DataCopyPad(target, tile[row * ubStride], DataCopyExtParams{1, cols * 2U, 0, 0, 0});
    }
    // One completion per tile, never the old read/write/wait per token.
    SyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void FusedTrain::SendDpRows(uint32_t start, uint32_t rows, LocalTensor<float> sums)
{
    LocalTensor<int32_t> maps{TPosition::VECIN, 131072, 24};
    LocalTensor<bfloat16_t> compact{TPosition::VECOUT, 135168, 32};
    LocalTensor<uint16_t> padded{TPosition::VECOUT, 135680, 128};
    GlobalTensor<int32_t> mapSource;
    mapSource.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.maps) + uint64_t(start) * 3U);
    mapSource.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    SyncFunc<HardEvent::S_MTE2>();
    DataCopyPad(maps, mapSource, DataCopyExtParams{1, rows * 3U * 4U, 0, 0, 0},
                DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    SyncFunc<HardEvent::MTE2_S>();

    if (a.topkWeightType) {
        // Each short DMA still consumes one padded 32-byte source row.
        // Preserve the FP32 sum; never narrow dp through the BF16 buffer.
        LocalTensor<float> paddedFp32{TPosition::VECOUT, 135680, 64};
        for (uint32_t row = 0; row < rows; ++row) paddedFp32.SetValue(row * 8U, sums.GetValue(row));
        SyncFunc<HardEvent::S_MTE3>();
        uint32_t lastRank = 0xffffffffU;
        uint64_t remoteBase = 0;
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t rank = maps.GetValue(row * 3U);
            if (rank != lastRank) {
                remoteBase = reinterpret_cast<uint64_t>(Peer(rank, 0));
                lastRank = rank;
            }
            const uint64_t route = uint64_t(maps.GetValue(row * 3U + 1U)) * a.topk + maps.GetValue(row * 3U + 2U);
            GlobalTensor<float> target;
            target.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(remoteBase + peer.dp + route * 4U));
            target.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            DataCopyPad(target, paddedFp32[row * 8U], DataCopyExtParams{1, 4, 0, 0, 0});
        }
        SyncFunc<HardEvent::MTE3_S>();
        return;
    }
    SyncFunc<HardEvent::S_V>();
    Cast(compact, sums, RoundMode::CAST_RINT, rows);
    SyncFunc<HardEvent::V_S>();
    auto raw = compact.ReinterpretCast<uint16_t>();
    for (uint32_t row = 0; row < rows; ++row) padded.SetValue(row * 16U, raw.GetValue(row));
    SyncFunc<HardEvent::S_MTE3>();

    uint32_t lastRank = 0xffffffffU;
    uint64_t remoteBase = 0;
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t rank = maps.GetValue(row * 3U);
        if (rank != lastRank) {
            remoteBase = reinterpret_cast<uint64_t>(Peer(rank, 0));
            lastRank = rank;
        }
        const uint32_t token = maps.GetValue(row * 3U + 1U);
        const uint32_t slot = maps.GetValue(row * 3U + 2U);
        const uint64_t route = uint64_t(token) * a.topk + slot;
        GlobalTensor<uint16_t> target;
        target.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(remoteBase + peer.dp + route * 2U));
        target.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        DataCopyPad(target, padded[row * 16U], DataCopyExtParams{1, 2, 0, 0, 0});
    }
    SyncFunc<HardEvent::MTE3_S>();
}
