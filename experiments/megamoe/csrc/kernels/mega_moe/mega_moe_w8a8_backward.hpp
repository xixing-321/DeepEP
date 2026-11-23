// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Native W8A8 backward consuming the forward Wave's canonical expert rows.
// Cube uses MXFP8 throughout; BF16 vector arithmetic follows BF16 train.
#pragma once

#include <type_traits>

#include "kernel_operator.h"
#include "mc2_common/mc2_kernel_utils.hpp"
#include "catlass_gmm/mega_moe_w8a8_catlass_backward.hpp"
#include "common/mega_moe_w8a8_backward_params.hpp"
#include "epilogue/block_epilogue_mx_quant_backward.hpp"
#include "epilogue/block_epilogue_swiglu_w8a8_backward.hpp"
#include "epilogue/topk_grad_reduce.hpp"
#include "timer/ascend_timer_device.hpp"

namespace MegaMoeTrainW8A8 {
using namespace AscendC;
using VecShape = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<256>>;
using BwdSwi = MegaMoeTrainW8A8Vec::TrainSwigluBackwardBF16<VecShape>;
using Quant = MegaMoeTrainW8A8Vec::TrainMxQuant;

// Flag 6 fences routing/activation stages; 8/10 transfer BF16 matrix ownership.
// Avoid CANN's reserved SyncAll flags 11..14. Wait on MTE2, then bridge to S
// so both data reads and scalar metadata reads observe the producer.
// Paired AIV1 uses the high flag bank, exactly as in BF16 train.
__aicore__ inline void VectorPublish(uint16_t flag = 6)
{
    SyncFunc<HardEvent::MTE3_S>();
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    CrossCoreSetFlag<4, PIPE_MTE3>(flag);
}
__aicore__ inline void CubeWait(uint16_t flag = 6)
{
    CrossCoreWaitFlag<4, PIPE_MTE2>(flag);
    CrossCoreWaitFlag<4, PIPE_MTE2>(flag + 16);
    SetFlag<HardEvent::MTE2_S>(7);
    WaitFlag<HardEvent::MTE2_S>(7);
}
__aicore__ inline void CubePublish(uint16_t flag = 4)
{
    PipeBarrier<PIPE_ALL>();
    CrossCoreSetFlag<4, PIPE_FIX>(flag);
    CrossCoreSetFlag<4, PIPE_FIX>(flag + 16);
}
__aicore__ inline void VectorWait(uint16_t flag = 4)
{
    CrossCoreWaitFlag<4, PIPE_MTE2>(flag);
    SetFlag<HardEvent::MTE2_S>(7);
    WaitFlag<HardEvent::MTE2_S>(7);
    SyncAll<true>();
}

class FusedTrain {
public:
    __aicore__ inline FusedTrain(const TrainParams& args, MegaMoeTrainTimerDevice* timer)
        : a(args),
          timer_(timer),
          peer(args.world, args.maxTokens, args.hidden, args.topk, args.experts, args.topkWeightType)
    {
    }
    __aicore__ inline void Backward();

private:
    TrainParams a;
    MegaMoeTrainTimerDevice* timer_;
    PeerLayout peer;
    KernelGemm cubeGemm;
    Quant quant;
    struct HpTokenConsumer {
        static constexpr bool enabled = true;
        FusedTrain* owner;
        uint32_t expert, row, column;
        template <bool PreserveDUp = false>
        __aicore__ inline void operator()(LocalTensor<bfloat16_t> hp, uint32_t rows, uint32_t columns,
                                          uint32_t stride) const
        {
            owner->QuantizeHpUb<PreserveDUp>(hp, stride, expert, row, column, rows, columns);
        }
    };
    template <bool PreserveDUp>
    __aicore__ inline void QuantizeHpUb(LocalTensor<bfloat16_t> hp, uint32_t stride, uint32_t e, uint32_t row,
                                        uint32_t column, uint32_t rows, uint32_t columns);
    // Per-Cube cursor, initialized at backward entry before any use.
    uint32_t dw1Expert_, dw1Job_, dw1GradReady_, dw1XReady_;
    __aicore__ inline bool CanRunReadyDw1Tile();
    __aicore__ inline void RunReadyDw1Tile();
    __aicore__ inline void WaitDw2InputsWithDw1(uint32_t e);
    __aicore__ inline uint32_t WeightBytes() const
    {
        return a.topkWeightType ? 4U : 2U;
    }
    __aicore__ inline uint32_t VCore() const
    {
        return GetBlockIdx();
    }
    __aicore__ inline uint32_t VCores() const
    {
        return GetBlockNum() * 2;
    }
    __aicore__ inline GM_ADDR Addr(uint64_t p) const
    {
        return reinterpret_cast<GM_ADDR>(p);
    }
    __aicore__ inline GM_ADDR Peer(uint32_t rank, uint64_t offset) const
    {
        auto peers = reinterpret_cast<__gm__ uint64_t*>(a.peers);
        return reinterpret_cast<GM_ADDR>(ReadGmByPassDCache(peers + rank) + offset);
    }
    __aicore__ inline int32_t I32(uint64_t p, uint64_t i) const
    {
        return ReadGmByPassDCache(reinterpret_cast<__gm__ int32_t*>(p) + i);
    }
    __aicore__ inline uint32_t Begin(uint32_t e) const
    {
        return I32(a.offsets, e);
    }
    __aicore__ inline uint32_t Rows(uint32_t e) const
    {
        return Begin(e + 1) - Begin(e);
    }
    __aicore__ inline uint32_t Total() const
    {
        return Begin(a.experts);
    }
    __aicore__ inline uint64_t DqData(uint32_t e) const
    {
        return a.packedY + uint64_t(Begin(e)) * a.hidden * 2;
    }
    __aicore__ inline uint64_t DqScale(uint32_t e) const
    {
        return DqData(e) + uint64_t(Rows(e)) * a.hidden;
    }
    __aicore__ inline uint32_t PadBegin(uint32_t e) const
    {
        uint32_t sum = 0;
        for (uint32_t i = 0; i < e; ++i) sum += (Rows(i) + 63) / 64 * 64;
        return sum;
    }

    __aicore__ inline void RankFence(uint32_t phase);
    __aicore__ inline void Copy(uint64_t dst, uint64_t src, uint64_t bytes, bool cacheData = false);
    __aicore__ inline void Zero(uint64_t dst, uint64_t bytes);
    __aicore__ inline void LoadBF(LocalTensor<bfloat16_t> dst, uint64_t src, uint32_t count);
    __aicore__ inline void LoadBFMatrix(LocalTensor<bfloat16_t> dst, uint64_t src, uint32_t rows, uint32_t columns,
                                        uint32_t width);
    __aicore__ inline void PublishDy();
    __aicore__ inline void DispatchDy(uint32_t e, uint32_t first, uint32_t rows, uint32_t& ping);
    __aicore__ inline uint32_t SourceRow(uint32_t e, uint32_t source) const;
    __aicore__ inline void Combine(bool backward);
    __aicore__ inline void CombinePipelined(bool backward);
    __aicore__ inline void GradientTile(uint32_t e, uint32_t row, uint32_t col, uint32_t rows, uint32_t cols);
    __aicore__ inline void TokenQuantRange(uint64_t source, uint64_t q, uint64_t scale, uint32_t width, uint32_t worker,
                                           uint32_t workers, uint32_t expert = 0xffffffffU);
    __aicore__ inline void DecodeRange(uint64_t q, uint64_t scale, uint64_t out, uint32_t width, uint32_t worker,
                                       uint32_t workers, uint32_t rows = 0xffffffffU);
    __aicore__ inline void RequantizeToken(uint64_t input, uint64_t inputScale, uint64_t output, uint64_t outputScale,
                                           uint32_t width, uint32_t worker, uint32_t workers,
                                           uint32_t expert = 0xffffffffU, bool gradient = false);
    __aicore__ inline void PrepareXExpert(uint32_t e, bool forward);
    __aicore__ inline void TokenQuantTile(uint64_t source, uint64_t q, uint64_t scale, uint32_t width, uint32_t e,
                                          uint32_t row, uint32_t col, uint32_t rows, uint32_t cols);
    __aicore__ inline void TokenQuantUb(LocalTensor<bfloat16_t> source, uint32_t stride, uint64_t q, uint64_t scale,
                                        uint32_t width, uint32_t e, uint32_t row, uint32_t col, uint32_t rows,
                                        uint32_t cols);
    __aicore__ inline void SendTile(uint32_t e, uint32_t row, uint32_t col, uint32_t rows, uint32_t cols);
    __aicore__ inline void ComputeDp();
    __aicore__ inline void SendDpRows(uint32_t start, uint32_t rows, LocalTensor<float> sums);
    __aicore__ inline uint32_t RowFlagCount() const
    {
        return a.experts * ((a.capacity + 255) / 256);
    }
    __aicore__ inline uint64_t FlagAddress(uint32_t section, uint32_t e, uint32_t worker) const;
    __aicore__ inline void PublishFlag(uint32_t section, uint32_t e, uint32_t worker, uint32_t value);
    __aicore__ inline void WaitFlagValue(uint32_t section, uint32_t e, uint32_t worker, uint32_t value);
    __aicore__ inline void WaitWorkers(uint32_t section, uint32_t e, uint32_t count);

    __aicore__ inline void DispatchTiles();
    __aicore__ inline void RecomputeGateUp();
    __aicore__ inline void ComputeOutputTiles(bool backward);
    __aicore__ inline void ComputeOutputExpert(bool backward, uint32_t e);
    __aicore__ inline void ComputeEarlyDw2();
    __aicore__ inline void PrepareDExpert(uint32_t e);
    __aicore__ inline bool SendNextDxTile(uint32_t& expert, uint32_t& sequence, bool wait);
    __aicore__ inline void ComputeDBTiles();
    __aicore__ inline void ConsumeGradientTiles(uint32_t& combineExpert, uint32_t& combineSequence);
    __aicore__ inline void ComputeWgrad();
    __aicore__ inline void BackwardPipeline();
};

__aicore__ inline void FusedTrain::Copy(uint64_t dst, uint64_t src, uint64_t bytes, bool cacheData)
{
    GlobalTensor<uint8_t> gs, gd;
    gs.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(src));
    gd.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(dst));
    gs.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    gd.SetL2CacheHint(cacheData ? AscendC::CacheMode::CACHE_MODE_NORMAL : AscendC::CacheMode::CACHE_MODE_DISABLE);
    LocalTensor<uint8_t> tmp{TPosition::VECIN, 0, 16384};
    // UB 0 may still contain scalar route counters or vector quantization
    // input. MTE3_MTE2 alone only retires DMA readers; it does not retire S/V
    // writers from the previous stage before this MTE2 fill.
    SyncFunc<HardEvent::S_MTE2>();
    SyncFunc<HardEvent::V_MTE2>();
    for (uint64_t off = 0; off < bytes; off += 16384) {
        const uint32_t n = bytes - off < 16384 ? bytes - off : 16384;
        SyncFunc<HardEvent::MTE3_MTE2>();
        DataCopyPad(tmp, gs[off], DataCopyExtParams{1, n, 0, 0, 0}, DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
        SyncFunc<HardEvent::MTE2_MTE3>();
        DataCopyPad(gd[off], tmp, DataCopyExtParams{1, n, 0, 0, 0});
    }
    SyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void FusedTrain::Zero(uint64_t dst, uint64_t bytes)
{
    GlobalTensor<uint8_t> gd;
    gd.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(dst));
    gd.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    LocalTensor<uint8_t> tmp{TPosition::VECOUT, 0, 16384};
    SyncFunc<HardEvent::MTE3_V>();
    Duplicate(tmp, uint8_t(0), 16384);
    SyncFunc<HardEvent::V_MTE3>();
    for (uint64_t off = uint64_t(VCore()) * 16384; off < bytes; off += uint64_t(VCores()) * 16384) {
        uint32_t n = bytes - off < 16384 ? bytes - off : 16384;
        DataCopyPad(gd[off], tmp, DataCopyExtParams{1, n, 0, 0, 0});
    }
    SyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void FusedTrain::RankFence(uint32_t phase)
{
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    {
        for (uint32_t rank = VCore(); rank < a.world; rank += VCores()) {
            auto signal = reinterpret_cast<__gm__ int32_t*>(Peer(rank, (uint64_t(phase) * a.world + a.rank) * 64));
            WriteGmByPassDCache(signal, int32_t(a.generation));
        }
        for (uint32_t rank = VCore(); rank < a.world; rank += VCores()) {
            auto signal = reinterpret_cast<__gm__ int32_t*>(Peer(a.rank, (uint64_t(phase) * a.world + rank) * 64));
            while (ReadGmByPassDCache(signal) < int32_t(a.generation)) {
            }
        }
    }
    SyncAll<true>();
}

__aicore__ inline void FusedTrain::LoadBF(LocalTensor<bfloat16_t> dst, uint64_t src, uint32_t count)
{
    GlobalTensor<bfloat16_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(src));
    gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    SyncFunc<HardEvent::MTE3_MTE2>();
    DataCopyPad(dst, gm, DataCopyExtParams{1, count * 2, 0, 0, 0}, DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
    SyncFunc<HardEvent::MTE2_V>();
}

__aicore__ inline void FusedTrain::LoadBFMatrix(LocalTensor<bfloat16_t> dst, uint64_t src, uint32_t rows,
                                                uint32_t columns, uint32_t width)
{
    GlobalTensor<bfloat16_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(src));
    gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    SyncFunc<HardEvent::MTE3_MTE2>();
    DataCopyPad(dst, gm, DataCopyExtParams{uint16_t(rows), columns * 2, (width - columns) * 2, 0, 0},
                DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
    SyncFunc<HardEvent::MTE2_V>();
}

// Source-side grouping: scan local bulk-loaded IDs in parallel by global
// expert, publish stable route lists/counts, never scalar-scan remote IDs.
__aicore__ inline void FusedTrain::PublishDy()
{
    const uint64_t bytes = uint64_t(a.tokens) * a.hidden * sizeof(bfloat16_t);
    for (uint64_t off = uint64_t(VCore()) * 16384; off < bytes; off += uint64_t(VCores()) * 16384) {
        const uint32_t count = bytes - off < 16384 ? bytes - off : 16384;
        Copy(reinterpret_cast<uint64_t>(Peer(a.rank, peer.xq)) + off, a.x + off, count);
    }
    RankFence(0);
}

// FP32 source p uses bounded raw-bit windows. Keep the 128 KiB source
// budget of BF16 routing, reusing it for each 32768-element window. Route
// batches may contain arbitrary source indices: do not assume sorted maps.
// Large BF16 p tables reuse the same 128 KiB source UB in raw-bit windows.
// Preserve route order and maps; no arithmetic or new GM allocation is added.
__aicore__ inline void FusedTrain::Combine(bool backward)
{
    CombinePipelined(backward);
    return;
    LocalTensor<bfloat16_t> bf{TPosition::VECIN, 0, 32768};
    LocalTensor<float> sum{TPosition::VECCALC, 65536, 16384};
    LocalTensor<float> fp{TPosition::VECCALC, 131072, 16384};
    const uint32_t tiles = (a.hidden + 16383) / 16384;
    for (uint64_t task = VCore(); task < uint64_t(a.tokens) * tiles; task += VCores()) {
        uint32_t token = task / tiles, col = (task % tiles) * 16384;
        uint32_t n = a.hidden - col < 16384 ? a.hidden - col : 16384;
        SyncFunc<HardEvent::MTE3_V>();
        Duplicate(sum, 0.0f, n);
        for (uint32_t slot = 0; slot < a.topk; ++slot) {
            uint64_t off = (uint64_t(token) * a.topk + slot) * a.hidden + col;
            LoadBF(bf, reinterpret_cast<uint64_t>(Peer(a.rank, peer.result + off * 2)), n);
            Cast(fp, bf, RoundMode::CAST_NONE, n);
            PipeBarrier<PIPE_V>();
            Add(sum, sum, fp, n);
            SyncFunc<HardEvent::V_MTE2>();
        }
        PipeBarrier<PIPE_V>();
        Cast(bf, sum, RoundMode::CAST_RINT, n);
        SyncFunc<HardEvent::V_MTE3>();
        GlobalTensor<bfloat16_t> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a.output) + uint64_t(token) * a.hidden + col);
        DataCopyPad(output, bf, DataCopyExtParams{1, n * 2, 0, 0, 0});
        SyncFunc<HardEvent::MTE3_V>();
    }
    if (backward) {
        for (uint32_t t = VCore(); t < a.tokens; t += VCores()) {
            Copy(a.dpOutput + uint64_t(t) * a.topk * WeightBytes(),
                 reinterpret_cast<uint64_t>(Peer(a.rank, peer.dp + uint64_t(t) * a.topk * WeightBytes())),
                 a.topk * WeightBytes());
        }
    }
}

// Ordinary W8A8 keeps the entire K reduction in L0C. A completed matrix is
// published once, instead of exporting FP32 partials and syncing every strip.
__aicore__ inline void FusedTrain::Backward()
{
    BackwardPipeline();
}
#include "stage/mega_moe_w8a8_unpermute_backward.hpp"
#include "stage/mega_moe_w8a8_vector_backward.hpp"
#include "stage/mega_moe_w8a8_requantize_backward.hpp"
#include "stage/mega_moe_w8a8_send_backward.hpp"
#include "stage/mega_moe_w8a8_schedule_backward.hpp"
}  // namespace MegaMoeTrainW8A8
