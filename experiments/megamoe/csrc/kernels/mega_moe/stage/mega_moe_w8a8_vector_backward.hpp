// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Tile consumers for the ordinary MXFP8 training pipeline.
// Include inside namespace MegaMoeTrainW8A8 after the FusedTrain declaration.
#pragma once

__aicore__ inline void FusedTrain::GradientTile(uint32_t e, uint32_t localRow, uint32_t col, uint32_t rows,
                                                uint32_t cols)
{
    if (!rows || !cols) return;
    BwdSwi swi;
    BwdSwi::Arguments args;
    const bool compressed = a.saveMode == MegaMoeTrainW8A8::kSaveMxFp8;
    args.gatePlaneGmAddr = Addr(compressed ? a.gateq : a.gate);
    args.upPlaneGmAddr = Addr(compressed ? a.upq : a.up);
    args.gateScaleGmAddr = Addr(a.sqg);
    args.upScaleGmAddr = Addr(a.squ);
    args.gateHighGmAddr = a.limit > 0 ? Addr(a.gateHigh) : nullptr;
    args.upLowGmAddr = a.limit > 0 ? Addr(a.upLow) : nullptr;
    args.upHighGmAddr = a.limit > 0 ? Addr(a.upHigh) : nullptr;
    args.gradZGmAddr = Addr(a.dz);
    args.hpGmAddr = Addr(a.hp);
    args.dpLow = compressed ? Addr(a.db) : nullptr;
    args.packedTopkWeightsGmAddr = Addr(a.packedP);
    args.baseM = 128;
    args.baseN = 256;
    args.swigluLimit = a.limit > 0 ? a.limit : kMegaMoeSwigluLimitDisabled;
    args.savedGateUpMode = a.saveMode;
    args.topkWeightType = a.topkWeightType;
    swi.Init(args);
    swi.UpdateGlobalAddr(0);
    swi.UpdateNextProblem({0, a.ffn * 2, a.hidden, 0});

    LocalTensor<bfloat16_t> first{TPosition::VECIN, 0, 32768};
    LocalTensor<bfloat16_t> second{TPosition::VECIN, 65536, 32768};
    for (uint32_t base = 0; base < rows; base += 128) {
        const uint32_t batch = rows - base < 128 ? rows - base : 128;
        const uint32_t row = Begin(e) + localRow + base;
        const uint64_t offset = uint64_t(row) * a.ffn + col;
        LoadBFMatrix(first, a.db + offset * 2, batch, cols, a.ffn);
        swi.template SwiGLUGrad<false>({batch, cols, 0, 0}, {int64_t(offset), 0, 0, 0, 0, 0},
                                       HpTokenConsumer{this, e, localRow + base, col});

        // SwiGLUGrad leaves dGate/dUp in UB 0/65536.
        // Feature-quant scratch starts at UB 131072; both BF16 outputs remain
        // intact through the two quantizers. Keep these feature-direction
        // representations independent of the token-direction wgrad work.
        const uint64_t dzOffset = uint64_t(row) * (2 * a.ffn) + col;
        quant.QuantizeTile(first, Addr(a.dzq), Addr(a.sz), batch, cols, 2 * a.ffn, dzOffset, true);
        quant.QuantizeTile(second, Addr(a.dzq), Addr(a.sz), batch, cols, 2 * a.ffn, dzOffset + a.ffn, true);
    }
}

__aicore__ inline void FusedTrain::ComputeDp(uint32_t worker, uint32_t workers)
{
    const uint32_t rows = Total();
    if (rows == 0) {
        return;
    }
    const uint32_t f = a.ffn;
    const auto plan = MegaMoeBF16Impl::MakeTopkGradReductionPlan(rows, f, VCores());

    // Up to eight rows of dHp occupy [0,4096). The current batch of FP32
    // products fits before the Gate tile at 64 KiB. Restore the original column
    // order before reduction for both cached and recomputed products.
    constexpr uint32_t kProductsOffset = 4096;
    constexpr uint32_t kReductionScratchOffset = 65536;
    constexpr uint32_t kProductsBytes = kReductionScratchOffset - kProductsOffset;
    const uint32_t maxRows = kProductsBytes / (f * sizeof(float));
    const uint32_t batch = maxRows < 8 ? maxRows : 8;
    LocalTensor<float> products{TPosition::VECCALC, kProductsOffset, kProductsBytes / sizeof(float)};
    LocalTensor<float> sums{TPosition::VECCALC, 134144, 8};
    BwdSwi swi;
    BwdSwi::Arguments args;
    const bool compressed = a.saveMode == kSaveMxFp8;
    args.gatePlaneGmAddr = Addr(compressed ? a.gateq : a.gate);
    args.upPlaneGmAddr = Addr(compressed ? a.upq : a.up);
    args.gateScaleGmAddr = Addr(a.sqg);
    args.upScaleGmAddr = Addr(a.squ);
    args.gateHighGmAddr = Addr(a.gateHigh);
    args.upLowGmAddr = Addr(a.upLow);
    args.upHighGmAddr = Addr(a.upHigh);
    args.swigluLimit = a.limit > 0 ? a.limit : kMegaMoeSwigluLimitDisabled;
    args.savedGateUpMode = a.saveMode;
    swi.Init(args);
    swi.UpdateNextProblem({0, 2 * f, a.hidden, 0});
    swi.SetDpInput(Addr(a.db));

    for (uint32_t row = worker * batch; row < rows; row += workers * batch) {
        const uint32_t take = rows - row < batch ? rows - row : batch;
        for (uint32_t col = 0; col < f; col += 256) {
            const uint32_t cols = f - col < 256 ? f - col : 256;
            if (compressed) {
                GlobalTensor<float> lo, hi;
                lo.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a.db));
                hi.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a.hp));
                lo.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                hi.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                const uint32_t half = cols / 2;
                const uint64_t off = (uint64_t(row) * f + col) / 2;
                DataCopyExtParams p{uint16_t(take), half * 4, (f / 2 - half) * 4, (f - half) / 8, 0};
                DataCopyPadExtParams<float> pad{false, 0, 0, 0};
                SyncFunc<HardEvent::V_MTE2>();
                DataCopyPad(products[col], lo[off], p, pad);
                DataCopyPad(products[col + half], hi[off], p, pad);
                SyncFunc<HardEvent::MTE2_V>();
            } else {
                swi.LoadDpProducts(row, col, take, cols, products);
            }
        }
        PipeBarrier<PIPE_V>();
        if (plan.width == f) {
            uint32_t shape[2] = {take, f};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(
                sums, products, products.ReinterpretCast<uint8_t>(), shape, false);
        } else {
            for (uint32_t i = 0; i < take; ++i) {
                auto sum = MegaMoeImpl::ReduceTopkGradGroupedRow(products[i * f], f, plan, kReductionScratchOffset);
                SyncFunc<HardEvent::V_S>();
                sums.SetValue(i, sum.GetValue(0));
                SyncFunc<HardEvent::S_V>();
            }
        }
        SyncFunc<HardEvent::V_S>();
        SendDpRows(row, take, sums);
    }
}

// Quantize Hp from the third UB tile before it is reused by SwiGLUGrad.
template <bool PreserveDUp>
__aicore__ inline void FusedTrain::QuantizeHpUb(LocalTensor<bfloat16_t> hp, uint32_t stride, uint32_t e, uint32_t row,
                                                uint32_t column, uint32_t rows, uint32_t columns)
{
    using Layout = std::conditional_t<PreserveDUp, HpTokenQuantPreserveGradUbLayout, HpTokenQuantUbLayout>;
    static_assert(Layout::totalBytesWithClipMasks <= Catlass::Arch::Ascend950::UB_SIZE);
    LocalTensor<uint16_t> transposed{TPosition::VECCALC, Layout::transpose, 64 * 128};
    MegaMoeTrainW8A8Vec::TrainMxQuantWithLayout<Layout> hpQuant;
    const uint32_t pad = (Rows(e) + 63) / 64 * 64, before = PadBegin(e);
    for (uint32_t token = 0; token < rows; token += 64) {
        const uint32_t valid = rows - token < 64 ? rows - token : 64;
        for (uint32_t band = 0; band < columns; band += 128) {
            const uint32_t cols = columns - band < 128 ? columns - band : 128;
            SyncFunc<HardEvent::MTE3_V>();
            PipeBarrier<PIPE_V>();
            const uint64_t srcBase = reinterpret_cast<uint64_t>(hp.GetPhyAddr()) + uint64_t(token * stride + band) * 2;
            const uint64_t dstBase = reinterpret_cast<uint64_t>(transposed.GetPhyAddr());
            uint64_t dstList[16], srcList[16];
            for (uint32_t block = 0; block < 4; ++block) {
                for (uint32_t i = 0; i < 16; ++i) {
                    srcList[i] = srcBase + uint64_t(block * 16 + i) * stride * 2;
                    dstList[i] = dstBase + (i * 64 + block * 16) * 2;
                }
                TransDataTo5HD<uint16_t>(dstList, srcList,
                                         TransDataTo5HDParams{false, false, uint8_t(cols / 16), 64, 1});
            }
            PipeBarrier<PIPE_V>();
            if (valid < 64) {
                SyncFunc<HardEvent::V_S>();
                for (uint32_t c = 0; c < cols; ++c)
                    for (uint32_t r = valid; r < 64; ++r) transposed.SetValue(c * 64 + r, uint16_t(0));
                SyncFunc<HardEvent::S_V>();
            }
            hpQuant.QuantizeTile(transposed.ReinterpretCast<bfloat16_t>(), Addr(a.bt + uint64_t(before) * a.ffn),
                                 Addr(a.sbt + uint64_t(before / 32) * a.ffn), cols, 64, pad,
                                 uint64_t(column + band) * pad + row + token, true);
        }
    }
    // Retire the quantized writes before the caller reuses the UB tiles.
    SyncFunc<HardEvent::MTE3_MTE2>();
}

__aicore__ inline void FusedTrain::TokenQuantUb(LocalTensor<bfloat16_t> source, uint32_t stride, uint64_t q,
                                                uint64_t scale, uint32_t width, uint32_t e, uint32_t localRow,
                                                uint32_t feature, uint32_t rows, uint32_t cols)
{
    constexpr uint32_t bytes = 64 * 128 * sizeof(bfloat16_t);
    constexpr uint32_t scratch = Catlass::Arch::Ascend950::UB_SIZE - bytes;
    static_assert(scratch >= MegaMoeTrainW8A8::SavedGateUpQuantUbLayout::totalBytesWithClipMasks,
                  "Token transpose must not overlap quant scratch or derivative sources");
    static_assert(scratch % 32 == 0, "Token transpose must be DMA aligned");
    LocalTensor<uint16_t> transposed{TPosition::VECCALC, scratch, 64 * 128};
    const uint32_t pad = (Rows(e) + 63) / 64 * 64, before = PadBegin(e);
    for (uint32_t token = 0; token < rows; token += 64) {
        const uint32_t validRows = rows - token < 64 ? rows - token : 64;
        for (uint32_t band = 0; band < cols; band += 128) {
            const uint32_t columns = cols - band < 128 ? cols - band : 128;
            PipeBarrier<PIPE_V>();
            const uint64_t srcBase =
                reinterpret_cast<uint64_t>(source.GetPhyAddr()) + uint64_t(token * stride + band) * 2;
            const uint64_t dstBase = reinterpret_cast<uint64_t>(transposed.GetPhyAddr());
            uint64_t dstList[16], srcList[16];
            for (uint32_t block = 0; block < 4; ++block) {
                for (uint32_t i = 0; i < 16; ++i) {
                    srcList[i] = srcBase + uint64_t(block * 16 + i) * stride * 2;
                    dstList[i] = dstBase + (i * 64 + block * 16) * 2;
                }
                TransDataTo5HD<uint16_t>(dstList, srcList,
                                         TransDataTo5HDParams{false, false, uint8_t(columns / 16), 64, 1});
            }
            PipeBarrier<PIPE_V>();
            // Transpose may copy stale values from the allocated but invalid
            // tail rows. Clear them before quantizing the expert's final group.
            if (validRows < 64) {
                // A vector Duplicate at +validRows need not be 32B aligned.
                // The infrequent expert tail uses 16-bit scalar stores
                // with explicit V/S ordering; complete groups stay vector-only.
                SyncFunc<HardEvent::V_S>();
                for (uint32_t c = 0; c < columns; ++c)
                    for (uint32_t r = validRows; r < 64; ++r) transposed.SetValue(c * 64 + r, uint16_t(0));
                SyncFunc<HardEvent::S_V>();
            }
            quant.QuantizeTile(transposed.ReinterpretCast<bfloat16_t>(), Addr(q + uint64_t(before) * width),
                               Addr(scale + uint64_t(before / 32) * width), columns, 64, pad,
                               uint64_t(feature + band) * pad + localRow + token, true);
        }
    }
}

__aicore__ inline void FusedTrain::TokenQuantRange(uint64_t source, uint64_t q, uint64_t scale, uint32_t width,
                                                   uint32_t worker, uint32_t workers, uint32_t expert)
{
    if (!workers || worker >= workers) return;
    // A scheduler-selected subset of AIVs prepares wgrad independently of
    // dX and Combine. Each expert's token axis has separate 64-row padding.
    // [64 tokens,128 features] -> [128 features,64 tokens], using the same
    // 16x16 transpose and MX quantizer as the validated ordinary path.
    LocalTensor<uint16_t> input{TPosition::VECIN, 0, 8192};
    LocalTensor<uint16_t> matrix{TPosition::VECIN, 32768, 8192};
    for (uint32_t e = 0; e < a.experts; ++e) {
        if (expert != 0xffffffffU && e != expert) continue;
        const uint32_t rows = Rows(e);
        const uint32_t pad = (rows + 63) / 64 * 64;
        const uint32_t paddedBefore = PadBegin(e);
        const uint32_t bands = (width + 127) / 128;
        const uint32_t tokenTiles = pad / 64;
        for (uint32_t task = worker; task < bands * tokenTiles; task += workers) {
            const uint32_t band = task % bands;
            const uint32_t token = (task / bands) * 64;
            const uint32_t feature = band * 128;
            const uint32_t columns = width - feature < 128 ? width - feature : 128;
            SyncFunc<HardEvent::MTE2_V>();
            Duplicate(matrix, uint16_t(0), 8192);
            SyncFunc<HardEvent::V_MTE2>();
            const uint32_t validRows = token < rows ? (rows - token < 64 ? rows - token : 64) : 0;
            GlobalTensor<uint16_t> src;
            src.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(source) + uint64_t(Begin(e) + token) * width +
                                feature);
            src.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            if (validRows) {
                DataCopyPad(matrix, src,
                            DataCopyExtParams{uint16_t(validRows), columns * 2, (width - columns) * 2,
                                              (128 - columns) * 2 / 32, 0},
                            DataCopyPadExtParams<uint16_t>{false, 0, 0, 0});
            }
            SyncFunc<HardEvent::MTE2_V>();
            uint64_t dstList[16], srcList[16];
            const uint64_t srcBase = reinterpret_cast<uint64_t>(matrix.GetPhyAddr());
            const uint64_t dstBase = reinterpret_cast<uint64_t>(input.GetPhyAddr());
            for (uint32_t tokenBlock = 0; tokenBlock < 4; ++tokenBlock) {
                for (uint32_t i = 0; i < 16; ++i) {
                    srcList[i] = srcBase + (tokenBlock * 16 + i) * 128 * 2;
                    dstList[i] = dstBase + (i * 64 + tokenBlock * 16) * 2;
                }
                TransDataTo5HD<uint16_t>(dstList, srcList,
                                         TransDataTo5HDParams{false, false, uint8_t(columns / 16), 64, 1});
            }
            PipeBarrier<PIPE_V>();
            quant.QuantizeTile(input.ReinterpretCast<bfloat16_t>(), Addr(q + uint64_t(paddedBefore) * width),
                               Addr(scale + uint64_t(paddedBefore / 32) * width), columns, 64, pad,
                               uint64_t(feature) * pad + token, true);
        }
    }
}

__aicore__ inline void FusedTrain::DecodeRange(uint64_t q, uint64_t scale, uint64_t out, uint32_t width,
                                               uint32_t worker, uint32_t workers, uint32_t rows)
{
    if (!workers || worker >= workers) return;
    // Identical decode and BF16 rounding to Decode(), with an explicit AIV
    // partition. The caller joins all decoder workers before consuming out.
    LocalTensor<int8_t> data{TPosition::VECIN, 0, 8192};
    LocalTensor<int8_t> scales{TPosition::VECIN, 8192, 256};
    LocalTensor<bfloat16_t> scaleBf{TPosition::VECCALC, 12288, 512};
    LocalTensor<float> scaleFp{TPosition::VECCALC, 16384, 1024};
    LocalTensor<float> decoded{TPosition::VECCALC, 32768, 8192};
    LocalTensor<bfloat16_t> bf{TPosition::VECOUT, 65536, 8192};
    const uint64_t total = uint64_t(rows == 0xffffffffU ? Total() : rows) * width;
    for (uint64_t off = uint64_t(worker) * 8192; off < total; off += uint64_t(workers) * 8192) {
        const uint32_t n = total - off < 8192 ? total - off : 8192;
        const uint32_t groups = n / 32;
        GlobalTensor<int8_t> gd, gs;
        gd.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(q) + off);
        gs.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(scale) + off / 32);
        gd.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        gs.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        SyncFunc<HardEvent::MTE3_MTE2>();
        SyncFunc<HardEvent::V_MTE2>();
        DataCopyPad(data, gd, DataCopyExtParams{1, n, 0, 0, 0}, DataCopyPadExtParams<int8_t>{false, 0, 0, 0});
        DataCopyPad(scales, gs, DataCopyExtParams{1, groups, 0, 0, 0}, DataCopyPadExtParams<int8_t>{false, 0, 0, 0});
        SyncFunc<HardEvent::MTE2_V>();
        auto source = reinterpret_cast<__ubuf__ fp8_e4m3fn_t*>(data.GetPhyAddr());
        auto scaleSource = reinterpret_cast<__ubuf__ fp8_e8m0_t*>(scales.GetPhyAddr());
        auto sbf = reinterpret_cast<__ubuf__ bfloat16_t*>(scaleBf.GetPhyAddr());
        auto sfp = reinterpret_cast<__ubuf__ float*>(scaleFp.GetPhyAddr());
        auto dst = reinterpret_cast<__ubuf__ float*>(decoded.GetPhyAddr());
        __VEC_SCOPE__
        {
            static constexpr Reg::CastTrait cast = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                    Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            for (uint16_t begin = 0; begin < groups; begin += 32) {
                const uint32_t count = groups - begin < 32 ? groups - begin : 32;
                uint32_t rem = count;
                auto mask = Reg::UpdateMask<bfloat16_t>(rem);
                Reg::RegTensor<fp8_e8m0_t> se;
                Reg::RegTensor<bfloat16_t> sw, sr;
                Reg::RegTensor<float> sf;
                Reg::DataCopy<fp8_e8m0_t, Reg::LoadDist::DIST_UNPACK_B8>(se, scaleSource + begin);
                Reg::Cast<bfloat16_t, fp8_e8m0_t, cast>(sw, se, mask);
                Reg::DataCopy<bfloat16_t, Reg::StoreDist::DIST_INTLV_B16>(sbf + begin * 2, sw, sw, mask);
                Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                rem = count * 2;
                auto fmask = Reg::UpdateMask<float>(rem);
                Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(sr, sbf + begin * 2);
                Reg::Cast<float, bfloat16_t, cast>(sf, sr, fmask);
                Reg::DataCopy<float, Reg::StoreDist::DIST_INTLV_B32>(sfp + begin * 4, sf, sf, fmask);
            }
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            uint32_t remaining = n;
            for (uint16_t i = 0; i < (n + 63) / 64; ++i) {
                auto mask = Reg::UpdateMask<float>(remaining);
                Reg::RegTensor<float> s, x, result;
                Reg::RegTensor<fp8_e4m3fn_t> qv;
                Reg::DataCopy<float, Reg::LoadDist::DIST_E2B_B32>(s, sfp + i * 8);
                Reg::DataCopy<fp8_e4m3fn_t, Reg::LoadDist::DIST_UNPACK4_B8>(qv, source + i * 64);
                Reg::Cast<float, fp8_e4m3fn_t, cast>(x, qv, mask);
                Reg::Mul(result, s, x, mask);
                Reg::DataCopy(dst + i * 64, result, mask);
            }
        }
        PipeBarrier<PIPE_V>();
        Cast(bf, decoded, RoundMode::CAST_RINT, n);
        SyncFunc<HardEvent::V_MTE3>();
        GlobalTensor<bfloat16_t> go;
        go.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out) + off);
        go.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        DataCopyPad(go, bf, DataCopyExtParams{1, n * 2, 0, 0, 0});
        SyncFunc<HardEvent::MTE3_V>();
    }
}

__aicore__ inline void FusedTrain::TokenQuantTile(uint64_t source, uint64_t q, uint64_t scale, uint32_t width,
                                                  uint32_t e, uint32_t localRow, uint32_t feature, uint32_t rows,
                                                  uint32_t cols)
{
    const uint32_t expertRows = Rows(e);
    if (!rows || !cols || localRow >= expertRows || feature >= width) return;
    // localRow is 64-aligned. rows spans complete 64-token groups, except
    // the true expert tail; all valid rows of those groups must be ready.
    // feature is 64-aligned and cols <= 256. Do not quantize a partial live
    // token group: its scale must include every token belonging to it.
    const uint32_t endRow = rows < expertRows - localRow ? localRow + rows : expertRows;
    const uint32_t endFeature = cols < width - feature ? feature + cols : width;
    const uint32_t pad = (expertRows + 63) / 64 * 64;
    const uint32_t paddedBefore = PadBegin(e);
    LocalTensor<uint16_t> input{TPosition::VECIN, 0, 8192};
    LocalTensor<uint16_t> matrix{TPosition::VECIN, 32768, 8192};
    for (uint32_t token = localRow; token < endRow; token += 64) {
        const uint32_t validRows = endRow - token < 64 ? endRow - token : 64;
        for (uint32_t band = feature; band < endFeature; band += 128) {
            const uint32_t columns = endFeature - band < 128 ? endFeature - band : 128;
            SyncFunc<HardEvent::MTE2_V>();
            Duplicate(matrix, uint16_t(0), 8192);
            SyncFunc<HardEvent::V_MTE2>();
            GlobalTensor<uint16_t> src;
            src.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(source) + uint64_t(Begin(e) + token) * width + band);
            src.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            DataCopyPad(
                matrix, src,
                DataCopyExtParams{uint16_t(validRows), columns * 2, (width - columns) * 2, (128 - columns) * 2 / 32, 0},
                DataCopyPadExtParams<uint16_t>{false, 0, 0, 0});
            SyncFunc<HardEvent::MTE2_V>();
            uint64_t dstList[16], srcList[16];
            const uint64_t srcBase = reinterpret_cast<uint64_t>(matrix.GetPhyAddr());
            const uint64_t dstBase = reinterpret_cast<uint64_t>(input.GetPhyAddr());
            for (uint32_t tokenBlock = 0; tokenBlock < 4; ++tokenBlock) {
                for (uint32_t i = 0; i < 16; ++i) {
                    srcList[i] = srcBase + (tokenBlock * 16 + i) * 128 * 2;
                    dstList[i] = dstBase + (i * 64 + tokenBlock * 16) * 2;
                }
                TransDataTo5HD<uint16_t>(dstList, srcList,
                                         TransDataTo5HDParams{false, false, uint8_t(columns / 16), 64, 1});
            }
            PipeBarrier<PIPE_V>();
            quant.QuantizeTile(input.ReinterpretCast<bfloat16_t>(), Addr(q + uint64_t(paddedBefore) * width),
                               Addr(scale + uint64_t(paddedBefore / 32) * width), columns, 64, pad,
                               uint64_t(band) * pad + token, true);
        }
    }
}
