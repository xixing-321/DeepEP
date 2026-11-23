// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Received feature-oriented FP8 -> BF16 UB -> token-oriented FP8.
#pragma once

__aicore__ inline void FusedTrain::RequantizeToken(uint64_t input, uint64_t inputScale, uint64_t output,
                                                   uint64_t outputScale, uint32_t width, uint32_t worker,
                                                   uint32_t workers, uint32_t expert, bool gradient)
{
    if (!workers || worker >= workers) return;
    constexpr uint32_t kScaleBfStride = 16;
    static_assert(kScaleBfStride * sizeof(bfloat16_t) % 32 == 0);
    static_assert(12288 + 64 * kScaleBfStride * sizeof(bfloat16_t) <= 16384);
    static_assert(65536 + 8192 * sizeof(bfloat16_t) <= MegaMoeTrainW8A8::SavedGateUpQuantUbLayout::gateData);
    LocalTensor<int8_t> data{TPosition::VECIN, 0, 8192};
    // Short per-row scale DMA occupies one 32-byte UB block per row.
    LocalTensor<int8_t> scales{TPosition::VECIN, 8192, 2048};
    // Each row starts on a 32-byte VEC address, even though only eight BF16
    // values are live. A compact 16-byte row stride triggers device error 340.
    LocalTensor<bfloat16_t> scaleBf{TPosition::VECCALC, 12288, 1024};
    LocalTensor<float> scaleFp{TPosition::VECCALC, 16384, 1024};
    LocalTensor<float> decoded{TPosition::VECCALC, 32768, 8192};
    LocalTensor<bfloat16_t> bf{TPosition::VECCALC, 65536, 8192};
    // This aliases the dead FP8 input and scale tiles, only after BF16 cast.
    LocalTensor<uint16_t> transposed{TPosition::VECIN, 0, 8192};
    for (uint32_t e = 0; e < a.experts; ++e) {
        if (expert != 0xffffffffU && expert != e) continue;
        const uint32_t rows = Rows(e), pad = (rows + 63) / 64 * 64, before = PadBegin(e);
        // Counts/offsets are immutable for this invocation. Resolve reused
        // expert bands once, not via scalar GM reads in every transpose tile.
        const uint64_t gradientData = gradient ? DqData(e) : 0;
        const uint64_t gradientScale = gradient ? DqScale(e) : 0;
        const uint32_t bands = (width + 127) / 128;
        for (uint32_t task = worker; task < bands * (pad / 64); task += workers) {
            const uint32_t feature = (task % bands) * 128, token = (task / bands) * 64;
            const uint32_t columns = width - feature < 128 ? width - feature : 128;
            const uint32_t valid = rows - token < 64 ? rows - token : 64;
            // Padding is a real zero operand, never a read across experts.
            SyncFunc<HardEvent::MTE3_V>();
            Duplicate(data.ReinterpretCast<uint8_t>(), uint8_t(0), 8192);
            Duplicate(scales.ReinterpretCast<uint8_t>(), uint8_t(127), 2048);
            SyncFunc<HardEvent::V_MTE2>();
            const uint64_t off = uint64_t((gradient ? 0 : Begin(e)) + token) * width + feature;
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
            auto q = reinterpret_cast<__ubuf__ fp8_e4m3fn_t*>(data.GetPhyAddr());
            auto sq = reinterpret_cast<__ubuf__ fp8_e8m0_t*>(scales.GetPhyAddr());
            auto sbf = reinterpret_cast<__ubuf__ bfloat16_t*>(scaleBf.GetPhyAddr());
            auto sfp = reinterpret_cast<__ubuf__ float*>(scaleFp.GetPhyAddr());
            auto dst = reinterpret_cast<__ubuf__ float*>(decoded.GetPhyAddr());
            __VEC_SCOPE__
            {
                static constexpr Reg::CastTrait cast = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
                // Expand four feature-group scales per 128-wide row. The
                // padded scale bytes are ignored by the output feature mask.
                for (uint16_t row = 0; row < 64; ++row) {
                    uint32_t rem = 4;
                    auto mask = Reg::UpdateMask<bfloat16_t>(rem);
                    Reg::RegTensor<fp8_e8m0_t> se;
                    Reg::RegTensor<bfloat16_t> sw, sr;
                    Reg::RegTensor<float> sf;
                    Reg::DataCopy<fp8_e8m0_t, Reg::LoadDist::DIST_UNPACK_B8>(se, sq + row * 32);
                    Reg::Cast<bfloat16_t, fp8_e8m0_t, cast>(sw, se, mask);
                    Reg::DataCopy<bfloat16_t, Reg::StoreDist::DIST_INTLV_B16>(sbf + row * kScaleBfStride, sw, sw, mask);
                    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                    rem = 8;
                    auto fmask = Reg::UpdateMask<float>(rem);
                    Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(sr, sbf + row * kScaleBfStride);
                    Reg::Cast<float, bfloat16_t, cast>(sf, sr, fmask);
                    Reg::DataCopy<float, Reg::StoreDist::DIST_INTLV_B32>(sfp + row * 16, sf, sf, fmask);
                }
                Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                uint32_t remaining = 8192;
                for (uint16_t i = 0; i < 128; ++i) {
                    auto mask = Reg::UpdateMask<float>(remaining);
                    Reg::RegTensor<float> s, x, result;
                    Reg::RegTensor<fp8_e4m3fn_t> v;
                    Reg::DataCopy<float, Reg::LoadDist::DIST_E2B_B32>(s, sfp + i * 8);
                    Reg::DataCopy<fp8_e4m3fn_t, Reg::LoadDist::DIST_UNPACK4_B8>(v, q + i * 64);
                    Reg::Cast<float, fp8_e4m3fn_t, cast>(x, v, mask);
                    Reg::Mul(result, s, x, mask);
                    Reg::DataCopy(dst + i * 64, result, mask);
                }
            }
            PipeBarrier<PIPE_V>();
            // Preserve the old GM-intermediate BF16 rounding boundary.
            Cast(bf, decoded, RoundMode::CAST_RINT, 8192);
            PipeBarrier<PIPE_V>();
            uint64_t dstList[16], srcList[16];
            const uint64_t srcBase = reinterpret_cast<uint64_t>(bf.GetPhyAddr());
            const uint64_t dstBase = reinterpret_cast<uint64_t>(transposed.GetPhyAddr());
            for (uint32_t block = 0; block < 4; ++block) {
                for (uint32_t i = 0; i < 16; ++i) {
                    srcList[i] = srcBase + (block * 16 + i) * 128 * 2;
                    dstList[i] = dstBase + (i * 64 + block * 16) * 2;
                }
                TransDataTo5HD<uint16_t>(dstList, srcList,
                                         TransDataTo5HDParams{false, false, uint8_t(columns / 16), 64, 1});
            }
            PipeBarrier<PIPE_V>();
            quant.QuantizeTile(transposed.ReinterpretCast<bfloat16_t>(), Addr(output + uint64_t(before) * width),
                               Addr(outputScale + uint64_t(before / 32) * width), columns, 64, pad,
                               uint64_t(feature) * pad + token, true);
        }
    }
}
