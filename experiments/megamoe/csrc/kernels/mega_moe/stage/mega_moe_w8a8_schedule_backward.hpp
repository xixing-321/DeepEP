// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Ordinary full-K tile pipeline. Include inside MegaMoeTrainW8A8.
#pragma once

// Per-producer cache lines. GM tile outputs are disjoint and remain valid
// until the invocation drains, so a monotonic sequence needs no tile ACK.
enum PipelineFlag : uint32_t {
    G1_SEQUENCE = 0,
    ACTIVATION_DONE = 1,
    G2_SEQUENCE = 2,
    DB_SEQUENCE = 3,
    DZ_DONE = 4,
    DX_SEQUENCE = 5,
    WGRAD_READY = 6,
    DW_DONE = 7,
    DECODE_DONE = 8,
    DISPATCH_ROW_READY = 9,
    DISPATCH_EXPERT_DONE = 10,
    D_WGRAD_READY = 11
};

__aicore__ inline uint64_t FusedTrain::FlagAddress(uint32_t section, uint32_t e, uint32_t worker) const
{
    const uint64_t index = section == DISPATCH_ROW_READY ? uint64_t(e) * ((a.capacity + 255) / 256) + worker
                                                         : uint64_t(RowFlagCount()) +
                                                               (uint64_t(section == DISPATCH_EXPERT_DONE ? 9
                                                                         : section == D_WGRAD_READY      ? 10
                                                                                                         : section) *
                                                                    a.experts +
                                                                e) *
                                                                   VCores() +
                                                               worker;
    return a.pipeline + index * 64;
}

__aicore__ inline void FusedTrain::PublishFlag(uint32_t section, uint32_t e, uint32_t worker, uint32_t value)
{
    if constexpr (g_coreType == AIC) {
        SetFlag<HardEvent::FIX_S>(7);
        WaitFlag<HardEvent::FIX_S>(7);
    } else {
        SyncFunc<HardEvent::MTE3_S>();
    }
    WriteGmByPassDCache(reinterpret_cast<__gm__ int32_t*>(FlagAddress(section, e, worker)), int32_t(value));
}

__aicore__ inline void FusedTrain::WaitFlagValue(uint32_t section, uint32_t e, uint32_t worker, uint32_t value)
{
    auto ptr = reinterpret_cast<__gm__ int32_t*>(FlagAddress(section, e, worker));
    while (ReadGmByPassDCache(ptr) < int32_t(value)) {
        const uint64_t begin = GetSystemCycle();
        while (GetSystemCycle() - begin < 64) {
        }
    }
}

__aicore__ inline void FusedTrain::WaitWorkers(uint32_t section, uint32_t e, uint32_t count)
{
    for (uint32_t worker = 0; worker < count; ++worker) WaitFlagValue(section, e, worker, 1);
}

#include "stage/mega_moe_w8a8_dispatch_backward.hpp"

__aicore__ inline void FusedTrain::DispatchTiles()
{
    const uint32_t worker = VCore() / 2;
    const uint32_t workers = GetBlockNum();
    const uint32_t parts = (workers + a.world - 1U) / a.world;
    uint32_t ping = 0;
    SyncFunc<HardEvent::MTE3_MTE2>();
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t e = 0; e < a.experts; ++e) {
        MegaMoeTrainTimerPassiveScope timer(timer_, MegaMoeTrainTimer::AIV_BW_DISPATCH, e, 0, Rows(e));
        for (uint32_t task = worker; task < a.world * parts; task += workers) {
            const uint32_t source = task / parts;
            const uint32_t begin = SourceRow(e, source);
            const uint32_t end = SourceRow(e, source + 1U);
            const uint32_t share = (end - begin + parts - 1U) / parts;
            const uint32_t first = begin + task % parts * share;
            const uint32_t last = first + share < end ? first + share : end;
            if (a.saveMode == kSaveNone) {
                if (first < last) DispatchDy(e, first, last - first, ping);
                continue;
            }
            for (uint32_t row = first; row < last;) {
                const uint32_t block = row / 256;
                const uint32_t boundary = (block + 1) * 256;
                const uint32_t next = last < boundary ? last : boundary;
                DispatchDy(e, row, next - row, ping);
                // DispatchDy has retired the raw dY and both feature-quant planes.
                AscendC::AtomicAdd(reinterpret_cast<__gm__ int32_t*>(FlagAddress(DISPATCH_ROW_READY, e, block)),
                                   static_cast<int32_t>(next - row));
                row = next;
            }
        }
        if (a.saveMode == kSaveNone) PublishFlag(DISPATCH_EXPERT_DONE, e, worker, 1);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

__aicore__ inline void FusedTrain::RecomputeGateUp()
{
    for (uint32_t e = 0; e < a.experts; ++e) {
        const uint32_t start = Begin(e), m = Rows(e);
        uint32_t sequence = 0;
        MegaMoeTrainTimerPassiveScope timer(timer_, MegaMoeTrainTimer::AIC_BW_RECOMPUTE, e, 0, m);
        Scheduler scheduler(Catlass::GemmCoord{m, a.ffn, a.hidden}, Catlass::MatrixCoord{256U, 256U});
        for (uint32_t job = GetBlockIdx(); job < scheduler.GetCoreLoops(); job += GetBlockNum()) {
            const auto coord = scheduler.GetBlockCoord(job);

            for (uint32_t half = 0; half < 2; ++half) {
                cubeGemm.Run(MakeParams(Addr(a.xq + uint64_t(start) * a.hidden),
                                        Addr(a.w1 + (uint64_t(e) * 2 * a.ffn + half * a.ffn) * a.hidden),
                                        Addr(a.sx + uint64_t(start) * (a.hidden / 32)),
                                        Addr(a.s1 + (uint64_t(e) * 2 * a.ffn + half * a.ffn) * (a.hidden / 32)),
                                        Addr((half ? a.up : a.gate) + uint64_t(start) * a.ffn * 2), m, a.ffn, a.hidden),
                             job, coord.m(), coord.n());
            }
            PublishFlag(G1_SEQUENCE, e, GetBlockIdx(), ++sequence);
        }
    }
}

__aicore__ inline void FusedTrain::ComputeOutputTiles(bool backward)
{
    for (uint32_t e = 0; e < a.experts; ++e) ComputeOutputExpert(backward, e);
}

__aicore__ inline void FusedTrain::ComputeOutputExpert(bool backward, uint32_t e)
{
    WaitWorkers(backward ? DZ_DONE : ACTIVATION_DONE, e, backward ? VCores() : GetBlockNum());
    const uint32_t m = Rows(e), start = Begin(e), k = backward ? 2 * a.ffn : a.ffn;
    MegaMoeTrainTimerPassiveScope timer(timer_, MegaMoeTrainTimer::AIC_BW_DX, e, 0, m);
    const uint64_t qa = (backward ? a.dzq : a.bq) + uint64_t(start) * k;
    const uint64_t sa = (backward ? a.sz : a.sb) + uint64_t(start) * (k / 32);
    const uint64_t qb = (backward ? a.w1b : a.w2) + uint64_t(e) * a.hidden * k;
    const uint64_t sb = (backward ? a.s1b : a.s2) + uint64_t(e) * a.hidden * (k / 32);
    Scheduler scheduler(Catlass::GemmCoord{m, a.hidden, k}, Catlass::MatrixCoord{256U, 256U});
    uint32_t sequence = 0;
    for (uint32_t job = GetBlockIdx(); job < scheduler.GetCoreLoops(); job += GetBlockNum()) {
        cubeGemm.Run(MakeParams(Addr(qa), Addr(qb), Addr(sa), Addr(sb),
                                Addr(a.packedY + uint64_t(start) * a.hidden * 2), m, a.hidden, k),
                     job);
        PublishFlag(backward ? DX_SEQUENCE : G2_SEQUENCE, e, GetBlockIdx(), ++sequence);
    }
}

__aicore__ inline bool FusedTrain::SendNextDxTile(uint32_t& expert, uint32_t& sequence, bool wait)
{
    const uint32_t worker = VCore() / 2;
    while (expert < a.experts) {
        Scheduler scheduler(Catlass::GemmCoord{Rows(expert), a.hidden, 2 * a.ffn}, Catlass::MatrixCoord{256U, 256U});
        const uint32_t job = worker + sequence * GetBlockNum();
        if (job >= scheduler.GetCoreLoops()) {
            ++expert;
            sequence = 0;
            continue;
        }
        if (wait) {
            WaitFlagValue(DX_SEQUENCE, expert, worker, sequence + 1);
        } else if (ReadGmByPassDCache(reinterpret_cast<__gm__ int32_t*>(FlagAddress(DX_SEQUENCE, expert, worker))) <=
                   static_cast<int32_t>(sequence)) {
            return false;
        }
        const auto coord = scheduler.GetBlockCoord(job);
        const auto shape = scheduler.GetActualBlockShape(coord);
        MegaMoeTrainTimerPassiveScope timer(timer_, MegaMoeTrainTimer::AIV_BW_COMBINE, expert, coord.m() * 256,
                                            shape.m(), coord.n() * 256);
        SendTile(expert, coord.m() * 256, coord.n() * 256, shape.m(), shape.n());
        ++sequence;
        return true;
    }
    return false;
}

__aicore__ inline void FusedTrain::ComputeDBTiles()
{
    const uint32_t workers = GetBlockNum();
    uint32_t firstOwner = 0;
    for (uint32_t e = 0; e < a.experts; ++e) {
        const uint32_t m = Rows(e), start = Begin(e);
        uint32_t sequence = 0;
        if (m != 0 && a.saveMode == kSaveNone) {
            MegaMoeTrainTimerPassiveScope wait(timer_, MegaMoeTrainTimer::AIC_BW_DHP_WAIT, e, 0, m);
            WaitWorkers(DISPATCH_EXPERT_DONE, e, GetBlockNum());
        }
        MegaMoeTrainTimerPassiveScope timer(timer_, MegaMoeTrainTimer::AIC_BW_DHP, e, 0, m);
        Scheduler scheduler(Catlass::GemmCoord{m, a.ffn, a.hidden}, Catlass::MatrixCoord{256U, 256U});
        uint32_t readyBlock = UINT32_MAX;
        for (uint32_t job = (GetBlockIdx() + workers - firstOwner) % workers; job < scheduler.GetCoreLoops();
             job += workers) {
            if (a.saveMode != kSaveNone) {
                const uint32_t block = scheduler.GetBlockCoord(job).m();
                if (block != readyBlock) {
                    const uint32_t remaining = m - block * 256;
                    const uint32_t rows = remaining < 256 ? remaining : 256;
                    MegaMoeTrainTimerPassiveScope wait(timer_, MegaMoeTrainTimer::AIC_BW_DHP_WAIT, e, block * 256,
                                                       rows);
                    WaitFlagValue(DISPATCH_ROW_READY, e, block, rows);
                    readyBlock = block;
                }
            }
            cubeGemm.Run(MakeParams(Addr(DqData(e)), Addr(a.w2b + uint64_t(e) * a.ffn * a.hidden), Addr(DqScale(e)),
                                    Addr(a.s2b + uint64_t(e) * a.ffn * (a.hidden / 32)),
                                    Addr(a.db + uint64_t(start) * a.ffn * 2), m, a.ffn, a.hidden),
                         job);
            PublishFlag(DB_SEQUENCE, e, GetBlockIdx(), ++sequence);
        }
        if (a.saveMode != kSaveNone) firstOwner = (firstOwner + scheduler.GetCoreLoops()) % workers;
    }
}

__aicore__ inline void FusedTrain::ConsumeGradientTiles(uint32_t& combineExpert, uint32_t& combineSequence)
{
    const uint32_t worker = VCore() / 2, sub = VCore() % 2, workers = GetBlockNum();
    uint32_t firstOwner = 0;
    for (uint32_t e = 0; e < a.experts; ++e) {
        Scheduler scheduler(Catlass::GemmCoord{Rows(e), a.ffn, a.hidden}, Catlass::MatrixCoord{256U, 256U});
        uint32_t sequence = 0;
        for (uint32_t job = (worker + workers - firstOwner) % workers; job < scheduler.GetCoreLoops(); job += workers) {
            const auto c = scheduler.GetBlockCoord(job), shape = scheduler.GetActualBlockShape(c);
            WaitFlagValue(DB_SEQUENCE, e, worker, ++sequence);
            const uint32_t rows =
                shape.m() > sub * 128 ? (shape.m() - sub * 128 < 128 ? shape.m() - sub * 128 : 128) : 0;
            MegaMoeTrainTimerPassiveScope timer(timer_, MegaMoeTrainTimer::AIV_BW_SWIGLU_GRAD, e,
                                                c.m() * 256 + sub * 128, rows, c.n() * 256, 128);
            GradientTile(e, c.m() * 256 + sub * 128, c.n() * 256, rows, shape.n());
            SyncFunc<HardEvent::MTE3_S>();
            // This lane owns whole 64-token groups. Prepare its wgrad tiles
            // here, using both AIV lanes rather than a later half-group pass.

            const uint32_t row = c.m() * 256 + sub * 128, col = c.n() * 256;
            // GradientTile owns at most 128 rows and leaves the exact BF16
            // derivatives in UB. Quant scratch and the token transpose do not
            // alias either plane. Keep this work at the original token stage.
            LocalTensor<bfloat16_t> dGate{TPosition::VECIN, 0, 32768};
            LocalTensor<bfloat16_t> dUp{TPosition::VECIN, 65536, 32768};
            TokenQuantUb(dGate, shape.n(), a.zt, a.szt, 2 * a.ffn, e, row, col, rows, shape.n());
            TokenQuantUb(dUp, shape.n(), a.zt, a.szt, 2 * a.ffn, e, row, col + a.ffn, rows, shape.n());
            SyncFunc<HardEvent::MTE3_S>();
        }
        PublishFlag(DZ_DONE, e, VCore(), 1);
        // X is immutable saved FP8: it has no dB/dZ/dY dependency. Pair this
        // lane's X work with AIV1's per-expert dY preparation, not its tail.
        if (sub == 0) {
            PrepareXExpert(e, false);
            // Try one completed dX tile before taking on the next expert.
            // The blocking tail resumes at the same cursor.
            SendNextDxTile(combineExpert, combineSequence, false);
        } else {
            PrepareDExpert(e);
        }
        if (a.saveMode != kSaveNone) firstOwner = (firstOwner + scheduler.GetCoreLoops()) % workers;
    }
}

__aicore__ inline void FusedTrain::PrepareDExpert(uint32_t e)
{
    const uint32_t worker = VCore() / 2, workers = GetBlockNum(), start = Begin(e), m = Rows(e);
    WaitWorkers(DZ_DONE, e, VCores());

    TokenQuantRange(a.deff, a.dt, a.sdt, a.hidden, worker, workers, e);
    PublishFlag(D_WGRAD_READY, e, worker, 1);
}

// Cursor state is private to each Cube. Ready flags are immutable until the
// invocation drains; remember validated producers rather than rescanning them.
__aicore__ inline bool FusedTrain::CanRunReadyDw1Tile()
{
    while (dw1Expert_ < a.experts) {
        const uint32_t rows = Rows(dw1Expert_);
        if (rows) {
            Scheduler scheduler(Catlass::GemmCoord{2 * a.ffn, a.hidden, (rows + 63) / 64 * 64},
                                Catlass::MatrixCoord{256U, 256U});
            if (dw1Job_ < scheduler.GetCoreLoops()) {
                while (dw1GradReady_ < VCores()) {
                    if (ReadGmByPassDCache(
                            reinterpret_cast<__gm__ int32_t*>(FlagAddress(DZ_DONE, dw1Expert_, dw1GradReady_))) < 1)
                        return false;
                    ++dw1GradReady_;
                }
                while (dw1XReady_ < GetBlockNum()) {
                    if (ReadGmByPassDCache(
                            reinterpret_cast<__gm__ int32_t*>(FlagAddress(WGRAD_READY, dw1Expert_, dw1XReady_))) < 1)
                        return false;
                    ++dw1XReady_;
                }
                return true;
            }
        }
        ++dw1Expert_;
        dw1Job_ = GetBlockIdx();
        dw1GradReady_ = 0;
        dw1XReady_ = 0;
    }
    return false;
}

__aicore__ inline void FusedTrain::RunReadyDw1Tile()
{
    const uint32_t e = dw1Expert_, k = (Rows(e) + 63) / 64 * 64, before = PadBegin(e);
    uint32_t row = 0, column = 0, rows = 0, columns = 0;
    if (timer_ != nullptr) {
        Scheduler scheduler(Catlass::GemmCoord{2 * a.ffn, a.hidden, k}, Catlass::MatrixCoord{256U, 256U});
        const auto coord = scheduler.GetBlockCoord(dw1Job_);
        const auto shape = scheduler.GetActualBlockShape(coord);
        row = coord.m() * 256;
        column = coord.n() * 256;
        rows = shape.m();
        columns = shape.n();
    }
    MegaMoeTrainTimerPassiveScope timer(timer_, MegaMoeTrainTimer::AIC_BW_DW1, e, row, rows, column, 256, 256, k,
                                        columns);

    cubeGemm.Run(
        MakeParams(Addr(a.zt + uint64_t(before) * 2 * a.ffn), Addr(a.xt + uint64_t(before) * a.hidden),
                   Addr(a.szt + uint64_t(before / 32) * 2 * a.ffn), Addr(a.sxt + uint64_t(before / 32) * a.hidden),
                   Addr(a.dw1 + uint64_t(e) * 2 * a.ffn * a.hidden * 2), 2 * a.ffn, a.hidden, k),
        dw1Job_);
    SetFlag<HardEvent::FIX_S>(7);
    WaitFlag<HardEvent::FIX_S>(7);

    dw1Job_ += GetBlockNum();
}

__aicore__ inline void FusedTrain::WaitDw2InputsWithDw1(uint32_t e)
{
    for (uint32_t worker = 0; worker < GetBlockNum(); ++worker) {
        auto ptr = reinterpret_cast<__gm__ int32_t*>(FlagAddress(D_WGRAD_READY, e, worker));
        while (ReadGmByPassDCache(ptr) < 1) {
            if (CanRunReadyDw1Tile()) {
                RunReadyDw1Tile();

            } else {
                const uint64_t begin = GetSystemCycle();
                while (GetSystemCycle() - begin < 64) {
                }
            }
        }
    }
}

__aicore__ inline void FusedTrain::ComputeEarlyDw2()
{
    // Every Cube observes all D_WGRAD_READY producers before writing dX.
    // Those producers retire all dB readers and dq/sd -> dt/sdt transfers.
    // dW2 reads separate dt/sdt: no extra all-Cube retirement fence is needed
    // when this expert's dq/sd band becomes BF16 packedY. Other bands persist.
    for (uint32_t e = 0; e < a.experts; ++e) {
        const uint32_t rows = Rows(e);
        if (rows) {
            WaitDw2InputsWithDw1(e);
            const uint32_t k = (rows + 63) / 64 * 64, before = PadBegin(e);
            MegaMoeTrainTimerPassiveScope timer(timer_, MegaMoeTrainTimer::AIC_BW_DW2, e, 0, a.hidden,
                                                static_cast<uint32_t>(-1), 256, 256, k, a.ffn);

            cubeGemm.Run(MakeParams(Addr(a.dt + uint64_t(before) * a.hidden), Addr(a.bt + uint64_t(before) * a.ffn),
                                    Addr(a.sdt + uint64_t(before / 32) * a.hidden),
                                    Addr(a.sbt + uint64_t(before / 32) * a.ffn),
                                    Addr(a.dw2 + uint64_t(e) * a.hidden * a.ffn * 2), a.hidden, a.ffn, k));
            SetFlag<HardEvent::FIX_S>(7);
            WaitFlag<HardEvent::FIX_S>(7);
        }
        ComputeOutputExpert(true, e);
    }
}

__aicore__ inline void FusedTrain::PrepareXExpert(uint32_t e, bool forward)
{
    const uint32_t worker = VCore() / 2, workers = GetBlockNum();
    if (forward) {
        WaitWorkers(DISPATCH_EXPERT_DONE, e, workers);
    }

    RequantizeToken(a.xq, a.sx, a.xt, a.sxt, a.hidden, worker, workers, e);
    // Includes MTE3 completion, including empty/idle workers. Existing flag
    // space is per expert; backward entry clears forward's control state.
    PublishFlag(WGRAD_READY, e, worker, 1);
}

__aicore__ inline void FusedTrain::ComputeWgrad()
{
    while (dw1Expert_ < a.experts) {
        if (CanRunReadyDw1Tile()) {
            RunReadyDw1Tile();
        } else if (dw1Expert_ < a.experts) {
            // The final expert's X preparation may still be in flight after
            // dX. Preserve both producer fences before draining its tiles.
            WaitWorkers(DZ_DONE, dw1Expert_, VCores());
            WaitWorkers(WGRAD_READY, dw1Expert_, GetBlockNum());
        }
    }
    PublishFlag(DW_DONE, 0, GetBlockIdx(), 1);
}

__aicore__ inline void FusedTrain::BackwardPipeline()
{
    if constexpr (g_coreType == AIV) {
        Zero(a.pipeline, uint64_t(RowFlagCount() + 11 * a.experts * VCores()) * 64);
        {
            MegaMoeTrainTimerFixedScope timer(timer_, MegaMoeTrainTimer::AIV_BW_GATHER);
            PublishDy();
        }
        VectorPublish();
        if (VCore() % 2) DispatchTiles();
        uint32_t combineExpert = 0, combineSequence = 0;
        ConsumeGradientTiles(combineExpert, combineSequence);
        if (VCore() % 2 == 0) {
            while (SendNextDxTile(combineExpert, combineSequence, true)) {
            }
        }
        // Each Vector lane publishes DZ_DONE in expert order. The final
        // publication proves every dHp tile is available for the existing
        // reduction, while a different lane can still be sending dX.
        WaitWorkers(DZ_DONE, a.experts - 1, VCores());
        {
            MegaMoeTrainTimerFixedScope timer(timer_, MegaMoeTrainTimer::AIV_BW_DP_REDUCE);
            ComputeDp();
        }

        for (uint32_t e = 0; e < a.experts; ++e)
            if (!Rows(e)) {
                Zero(a.dw1 + uint64_t(e) * 2 * a.ffn * a.hidden * 2, uint64_t(2 * a.ffn) * a.hidden * 2);
                Zero(a.dw2 + uint64_t(e) * a.hidden * a.ffn * 2, uint64_t(a.hidden) * a.ffn * 2);
            }

        RankFence(1);
        {
            MegaMoeTrainTimerFixedScope timer(timer_, MegaMoeTrainTimer::AIV_BW_UNPERMUTE);
            Combine(true);
        }
        RankFence(2);
        WaitWorkers(DW_DONE, 0, GetBlockNum());
        VectorPublish();
    } else {
        CubeWait();
        // Same scheduler/owner for recompute and dB; the dB sequence proves
        // the matching saved Gate/Up tile is already complete for save=none.
        if (a.saveMode == MegaMoeTrainW8A8::kSaveNone) RecomputeGateUp();
        dw1Expert_ = 0;
        dw1Job_ = GetBlockIdx();
        dw1GradReady_ = 0;
        dw1XReady_ = 0;
        ComputeDBTiles();
        ComputeEarlyDw2();
        ComputeWgrad();
        CubeWait();
    }
}
