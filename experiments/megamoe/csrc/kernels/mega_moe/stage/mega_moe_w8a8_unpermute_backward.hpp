// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// BF16-style two input/two output DMA slots; preserve FP32 TopK accumulation.
#pragma once

__aicore__ inline void FusedTrain::CombinePipelined(bool backward)
{
    const uint32_t width = 8192, tiles = (a.hidden + width - 1) / width;
    LocalTensor<float> temp{TPosition::VECCALC, 32768, 8192};
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    uint32_t outputSlot = 0;
    for (uint64_t task = VCore(); task < uint64_t(a.tokens) * tiles; task += VCores(), outputSlot ^= 1) {
        const uint32_t token = task / tiles, col = (task % tiles) * width,
                       n = a.hidden - col < width ? a.hidden - col : width;
        LocalTensor<float> sum{TPosition::VECCALC, 65536 + outputSlot * 32768, 8192};
        LocalTensor<bfloat16_t> result{TPosition::VECOUT, 131072 + outputSlot * 16384, 8192};
        const auto outId = outputSlot ? EVENT_ID1 : EVENT_ID0;
        for (uint32_t k = 0; k < (a.topk < 2 ? a.topk : 2); ++k) {
            const auto id = k ? EVENT_ID1 : EVENT_ID0;
            LocalTensor<bfloat16_t> input{TPosition::VECIN, k * 16384, 8192};
            GlobalTensor<bfloat16_t> source;
            source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(Peer(a.rank, peer.result)) +
                                   (uint64_t(token) * a.topk + k) * a.hidden + col);
            source.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            WaitFlag<HardEvent::V_MTE2>(id);
            DataCopyPad(input, source, DataCopyExtParams{1, n * 2, 0, 0, 0},
                        DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(id);
        }
        WaitFlag<HardEvent::MTE3_V>(outId);
        for (uint32_t k = 0; k < a.topk; ++k) {
            const uint32_t slot = k % 2;
            const auto id = slot ? EVENT_ID1 : EVENT_ID0;
            LocalTensor<bfloat16_t> input{TPosition::VECIN, slot * 16384, 8192};
            WaitFlag<HardEvent::MTE2_V>(id);
            const bool first = k == 0;
            MegaMoeBF16Routing::AccumulateUnpermuteRoute(sum, temp, input, n, first);
            // dX includes positive zero even when every route contributes negative zero.
            if (first) {
                Adds(sum, sum, 0.0f, n);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE2>(id);
            if (k + 2 < a.topk) {
                GlobalTensor<bfloat16_t> source;
                source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(Peer(a.rank, peer.result)) +
                                       (uint64_t(token) * a.topk + k + 2) * a.hidden + col);
                source.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                WaitFlag<HardEvent::V_MTE2>(id);
                DataCopyPad(input, source, DataCopyExtParams{1, n * 2, 0, 0, 0},
                            DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>(id);
            }
        }
        Cast(result, sum, RoundMode::CAST_RINT, n);
        SetFlag<HardEvent::V_MTE3>(outId);
        WaitFlag<HardEvent::V_MTE3>(outId);
        GlobalTensor<bfloat16_t> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a.output) + uint64_t(token) * a.hidden + col);
        DataCopyPad(output, result, DataCopyExtParams{1, n * 2, 0, 0, 0});
        SetFlag<HardEvent::MTE3_V>(outId);
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    PipeBarrier<PIPE_ALL>();
    if (backward) CopyTopkGradient();
}

__aicore__ inline void FusedTrain::CopyTopkGradient()
{
    const uint64_t bytes = uint64_t(a.tokens) * a.topk * WeightBytes();
    const uint64_t stripe = ((bytes + VCores() * 32 - 1) / (VCores() * 32)) * 32;
    const uint64_t offset = uint64_t(VCore()) * stripe;
    if (offset >= bytes) return;
    Copy(a.dpOutput + offset, reinterpret_cast<uint64_t>(Peer(a.rank, peer.dp)) + offset,
         bytes - offset < stripe ? bytes - offset : stripe);
}
