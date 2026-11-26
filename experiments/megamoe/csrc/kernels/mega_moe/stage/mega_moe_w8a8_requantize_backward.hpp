// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Received feature-oriented FP8 -> BF16 UB -> token-oriented FP8.
#pragma once

__aicore__ inline void FusedTrain::RequantizeToken(uint64_t input, uint64_t inputScale, uint64_t output,
                                                   uint64_t outputScale, uint32_t width, uint32_t worker,
                                                   uint32_t workers, uint32_t expert, bool gradient)
{
    if (!workers || worker >= workers) return;
    constexpr uint32_t tileRows = 128, tileColumns = 128;
    constexpr uint32_t tileElements = tileRows * tileColumns;
    static_assert(65536 + tileElements * sizeof(bfloat16_t) <= MegaMoeTrainW8A8::SavedGateUpQuantUbLayout::gateData);
    LocalTensor<int8_t> data{TPosition::VECIN, 0, tileElements};
    // Short per-row scale DMA occupies one 32-byte UB block per row.
    LocalTensor<int8_t> scales{TPosition::VECIN, 32768, tileRows * 32};
    LocalTensor<bfloat16_t> bf{TPosition::VECCALC, 65536, tileElements};
    // This aliases the dead FP8 input and scale tiles, only after BF16 cast.
    LocalTensor<uint16_t> transposed{TPosition::VECIN, 0, tileElements};
    for (uint32_t e = 0; e < a.experts; ++e) {
        if (expert != 0xffffffffU && expert != e) continue;
        const uint32_t rows = Rows(e), pad = (rows + 63) / 64 * 64, before = PadBegin(e);
        // Counts/offsets are immutable for this invocation. Resolve reused
        // expert bands once, not via scalar GM reads in every transpose tile.
        const uint64_t gradientData = gradient ? DqData(e) : 0;
        const uint64_t gradientScale = gradient ? DqScale(e) : 0;
        const uint32_t bands = (width + 127) / 128;
        const uint32_t firstRow = gradient ? 0 : Begin(e);
        for (uint32_t task = worker; task < bands * ((pad + tileRows - 1) / tileRows); task += workers) {
            const uint32_t feature = (task % bands) * 128, token = (task / bands) * tileRows;
            const uint32_t depth = pad - token < tileRows ? pad - token : tileRows;
            const uint32_t columns = width - feature < 128 ? width - feature : 128;
            const uint32_t valid = rows - token < depth ? rows - token : depth;
            // Padding is a real zero operand, never a read across experts.
            SyncFunc<HardEvent::MTE3_V>();
            if (valid < depth || columns < 128) {
                Duplicate(data.ReinterpretCast<uint8_t>(), uint8_t(0), tileElements);
                Duplicate(scales.ReinterpretCast<uint8_t>(), uint8_t(127), tileRows * 32);
            }
            SyncFunc<HardEvent::V_MTE2>();
            const uint64_t off = uint64_t(firstRow + token) * width + feature;
            const uint64_t dataAddress = (gradient ? gradientData : input) + off;
            const uint64_t scaleAddress = (gradient ? gradientScale : inputScale) + off / 32;
            GlobalTensor<int8_t> gd, gs;
            gd.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(dataAddress));
            gs.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(scaleAddress));
            gd.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            gs.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            DataCopyPad(data, gd, DataCopyExtParams{uint16_t(valid), columns, width - columns, (128 - columns) / 32, 0},
                        DataCopyPadExtParams<int8_t>{false, 0, 0, 0});
            DataCopyPad(scales, gs, DataCopyExtParams{uint16_t(valid), columns / 32, (width - columns) / 32, 0, 0},
                        DataCopyPadExtParams<int8_t>{false, 0, 0, 0});
            SyncFunc<HardEvent::MTE2_V>();
            MegaMoeTrainW8A8Vec::DecodeMxFp8Rows(data, scales, bf, depth, 128);
            PipeBarrier<PIPE_V>();
            uint64_t dstList[16], srcList[16];
            const uint64_t srcBase = reinterpret_cast<uint64_t>(bf.GetPhyAddr());
            const uint64_t dstBase = reinterpret_cast<uint64_t>(transposed.GetPhyAddr());
            for (uint32_t block = 0; block < depth / 16; ++block) {
                for (uint32_t i = 0; i < 16; ++i) {
                    srcList[i] = srcBase + (block * 16 + i) * 128 * 2;
                    dstList[i] = dstBase + (i * depth + block * 16) * 2;
                }
                TransDataTo5HD<uint16_t>(dstList, srcList,
                                         TransDataTo5HDParams{false, false, uint8_t(columns / 16), uint16_t(depth), 1});
            }
            PipeBarrier<PIPE_V>();
            quant.QuantizeTile(transposed.ReinterpretCast<bfloat16_t>(), Addr(output + uint64_t(before) * width),
                               Addr(outputScale + uint64_t(before / 32) * width), columns, depth, pad,
                               uint64_t(feature) * pad + token, true);
        }
    }
}
