// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

#pragma once

#include "catlass/mega_moe_bf16_forward_catlass_impl.hpp"
#include "schedule/forward_wave_plan.hpp"
#include "schedule/forward_wave_completion.hpp"

namespace MegaMoeBF16ForwardCatlassImpl {

using WaveBlockScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

// Both projections write BF16 to GM. One persistent MMAD retains its L1/L0
// buffers and event sequence when switching between W1 and W2.
template <bool TransW>
class WaveMatmul {
public:
    static constexpr uint32_t kRows = MegaMoeBF16Training::Wave::kGmm1Rows;
    static constexpr uint32_t kDownRows = MegaMoeBF16Training::Wave::kGmm2Rows;
    using WeightLayout = std::conditional_t<TransW, Catlass::layout::ColumnMajor, Catlass::layout::RowMajor>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, Gmm1L1TileShape, Gmm1L0TileShape, bfloat16_t,
                                                         bfloat16_t, bfloat16_t, void, ForwardGmTileCopy<WeightLayout>>;
    Catlass::Arch::Resource<ArchTag> resource;
    BlockMmad blockMmad;

    __aicore__ inline WaveMatmul() : blockMmad(resource) {}

    __aicore__ inline void GateUp(const Params& params, const MegaMoeBF16Training::Wave::Slice& slice,
                                  __gm__ int32_t* dispatchReady, uint32_t& startBlock, uint32_t& sequence)
    {
        using namespace MegaMoeBF16Training::Wave;
        const uint32_t h = params.tilingData->k, f = params.tilingData->n / 2U;
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t core = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        AscendC::GlobalTensor<bfloat16_t> a, b, gate, up;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.workspaceInfo.ptrA) + slice.packed * h);
        b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.bGmAddr) +
                          static_cast<uint64_t>(slice.expert) * 2U * f * h);
        gate.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.workspaceInfo.ptrSavedGate) +
                             slice.packed * f);
        up.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.workspaceInfo.ptrSavedUp) + slice.packed * f);
        gate.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        up.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        auto ta = tla::MakeTensor(a, tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(slice.rows, h),
                                  Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(b, tla::MakeLayout<bfloat16_t, WeightLayout>(h, 2U * f), Catlass::Arch::PositionGM{});
        auto tg = tla::MakeTensor(gate, tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(slice.rows, f),
                                  Catlass::Arch::PositionGM{});
        auto tu = tla::MakeTensor(up, tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(slice.rows, f),
                                  Catlass::Arch::PositionGM{});
        WaveBlockScheduler scheduler({slice.rows, f, h}, Catlass::MakeCoord(kRows, L1_TILE_N));
        const uint32_t tasks = scheduler.GetCoreLoops();
        const uint32_t first = (core < startBlock ? core + cores : core) - startBlock;
        bool ready = false;
        for (uint32_t task = first; task < tasks; task += cores) {
            const auto coord = scheduler.GetBlockCoord(task);
            const auto shape = scheduler.GetActualBlockShape(coord);
            const uint32_t m = coord.m() * kRows, n = coord.n() * L1_TILE_N;
            if (!ready) {
                while (AscendC::ReadGmByPassDCache(dispatchReady) < cores) {
                }
                AscendC::SyncFunc<AscendC::HardEvent::S_MTE2>();
                ready = true;
            }
            GateUpCompletion::Acquire(sequence);
            auto tileA = tla::GetTile(ta, tla::MakeCoord(m, 0U), tla::MakeShape(shape.m(), shape.k()));
            auto tileGate = tla::GetTile(tg, tla::MakeCoord(m, n), tla::MakeShape(shape.m(), shape.n()));
            auto tileUp = tla::GetTile(tu, tla::MakeCoord(m, n), tla::MakeShape(shape.m(), shape.n()));
            for (uint32_t part = 0; part < 2U; ++part) {
                auto tileB = tla::GetTile(tb, tla::MakeCoord(0U, n + part * f), tla::MakeShape(shape.k(), shape.n()));
                auto& output = part ? tileUp : tileGate;
                blockMmad(tileA, tileB, output, shape);
            }
            GateUpCompletion::Publish(sequence++);
        }
        startBlock = (startBlock + tasks) % cores;
    }

    __aicore__ inline bool Down(const Params& params, const MegaMoeBF16Training::Wave::Slice& slice, GM_ADDR output,
                                __gm__ int32_t* hpFlags, uint32_t& startBlock)
    {
        const uint32_t h = params.tilingData->k, f = params.tilingData->n / 2U;
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t core = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        AscendC::GlobalTensor<bfloat16_t> a, b, c;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.workspaceInfo.ptrA2) + slice.packed * f);
        b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.b2GmAddr) +
                          static_cast<uint64_t>(slice.expert) * f * h);
        c.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output));
        c.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        auto ta = tla::MakeTensor(a, tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(slice.rows, f),
                                  Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(b, tla::MakeLayout<bfloat16_t, WeightLayout>(f, h), Catlass::Arch::PositionGM{});
        auto tc = tla::MakeTensor(c, tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(slice.rows, h),
                                  Catlass::Arch::PositionGM{});
        WaveBlockScheduler scheduler({slice.rows, h, f}, Catlass::MakeCoord(kDownRows, L1_TILE_N));
        const uint32_t tasks = scheduler.GetCoreLoops();
        const uint32_t first = (core < startBlock ? core + cores : core) - startBlock;
        const bool owns = first < tasks;
        uint32_t readyM = 0xffffffffU;
        for (uint32_t task = first; task < tasks; task += cores) {
            const auto coord = scheduler.GetBlockCoord(task);
            const auto shape = scheduler.GetActualBlockShape(coord);
            const uint32_t m = coord.m() * kDownRows, n = coord.n() * L1_TILE_N;
            const uint32_t rows = shape.m(), columns = shape.n();
            if (m != readyM) {
                while (AscendC::ReadGmByPassDCache(hpFlags + m / kRows) < Ops::Base::CeilDiv(f, L1_TILE_N)) {
                }
                AscendC::SyncFunc<AscendC::HardEvent::S_MTE2>();
                readyM = m;
            }
            auto tileA = tla::GetTile(ta, tla::MakeCoord(m, 0U), tla::MakeShape(rows, f));
            auto tileB = tla::GetTile(tb, tla::MakeCoord(0U, n), tla::MakeShape(f, columns));
            auto tileC = tla::GetTile(tc, tla::MakeCoord(m, n), tla::MakeShape(rows, columns));
            blockMmad(tileA, tileB, tileC, Catlass::GemmCoord{rows, columns, f});
        }
        startBlock = (startBlock + tasks) % cores;
        return owns;
    }
};

}  // namespace MegaMoeBF16ForwardCatlassImpl
