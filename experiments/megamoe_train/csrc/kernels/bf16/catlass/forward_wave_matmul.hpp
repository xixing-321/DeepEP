// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

#pragma once

#include "catlass/mega_moe_bf16_forward_catlass_impl.hpp"
#include "schedule/forward_wave_plan.hpp"

namespace MegaMoeBF16ForwardCatlassImpl {

using WaveGmm1UbLayout = decltype(tla::MakeLayout(
    tla::MakeShape(128U, 256U), tla::MakeStride(tla::Int<256>{}, tla::Int<1>{}), tla::MakeShape(128U, 256U)));
__aicore__ inline WaveGmm1UbLayout MakeWaveGmm1UbLayout()
{
    return tla::MakeLayout(tla::MakeShape(128U, 256U), tla::MakeStride(tla::Int<256>{}, tla::Int<1>{}),
                           tla::MakeShape(128U, 256U));
}
using WaveGmm2UbLayout = decltype(tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(uint32_t{}, uint32_t{}));
template <class Layout>
using WaveLeadingStride = std::decay_t<decltype(tla::stride<0>(Layout{}))>;
template <class Layout>
constexpr uint32_t kWaveTargetAiv = tla::is_static<WaveLeadingStride<Layout>>::value ? 0U : 1U;
using WaveGmm1TileLayout = decltype(tla::GetTileLayout(WaveGmm1UbLayout{}, tla::MakeShape(uint32_t{}, uint32_t{}),
                                                       tla::MakeCoord(uint32_t{}, uint32_t{})));
using WaveGmm2TileLayout = decltype(tla::GetTileLayout(WaveGmm2UbLayout{}, tla::MakeShape(uint32_t{}, uint32_t{}),
                                                       tla::MakeCoord(uint32_t{}, uint32_t{})));
static_assert(kWaveTargetAiv<WaveGmm1TileLayout> == 0U);
static_assert(kWaveTargetAiv<WaveGmm2TileLayout> == 1U);

// Both output layouts have the same 128x256 physical strides. Their distinct
// leading-stride types select AIV0 (Gate/Up) or AIV1 (Combine) at compile time.
// GetTile retains that stride type when it crops the runtime shape, allowing one
// MMAD and one set of L1/L0 events to live across all waves and both GEMMs.
template <class WeightLayout>
struct WaveTileCopy
    : Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, bfloat16_t, Catlass::layout::RowMajor, bfloat16_t, WeightLayout,
                                             bfloat16_t, Catlass::layout::RowMajor, void> {
    using Base = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, bfloat16_t, Catlass::layout::RowMajor, bfloat16_t,
                                                        WeightLayout, bfloat16_t, Catlass::layout::RowMajor, void>;
    using TensorL0C = typename Base::TensorL0C;
    template <class TensorC>
    struct CopyL0CToDst {
        static constexpr bool gmm1 = kWaveTargetAiv<typename TensorC::Layout> == 0U;
        __aicore__ inline void operator()(const TensorC& dst, const TensorL0C& src, uint8_t unitFlag = 0)
        {
            Catlass::Gemm::Tile::CopyL0CToTargetUBTla<
                ArchTag, TensorL0C, TensorC, Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT,
                Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT, gmm1 ? 0U : 1U, false>
                copy;
            copy(dst, src, unitFlag);
        }
    };
};

template <bool TransW>
class WaveMatmul {
public:
    static constexpr uint32_t kRows = MegaMoeBF16Training::Wave::kGroupRows;
    static constexpr uint32_t kTileElements = kRows * L1_TILE_N;
    using WeightLayout = std::conditional_t<TransW, Catlass::layout::ColumnMajor, Catlass::layout::RowMajor>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, bfloat16_t,
                                                         bfloat16_t, bfloat16_t, void, WaveTileCopy<WeightLayout>>;
    using LayoutB = decltype(tla::MakeLayout<bfloat16_t, WeightLayout>(uint32_t{}, uint32_t{}));

    Catlass::Arch::Resource<ArchTag> resource;
    BlockMmad blockMmad;

    __aicore__ inline WaveMatmul() : blockMmad(resource) {}

    struct DownProjection {
        BlockMmad& blockMmad;
        AscendC::LocalTensor<bfloat16_t> l0cOutUbFirst;
        AscendC::LocalTensor<bfloat16_t> l0cOutUbSecond;
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t blockIdx = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        uint32_t n;
        uint32_t k;
        LayoutB layoutB;
        WaveGmm2UbLayout layoutC;

        __aicore__ inline DownProjection(WaveMatmul& owner, uint32_t h, uint32_t f)
            : blockMmad(owner.blockMmad),
              l0cOutUbFirst(owner.resource.ubBuf.template GetBufferByByte<bfloat16_t>(0)),
              l0cOutUbSecond(owner.resource.ubBuf.template GetBufferByByte<bfloat16_t>(kTileElements * 2U)),
              n(h),
              k(f),
              layoutB(tla::MakeLayout<bfloat16_t, WeightLayout>(f, h)),
              layoutC(tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(kRows, L1_TILE_N))
        {
        }
    };

    template <class EpilogueOp>
    __aicore__ inline void GateUp(const Params& params, const MegaMoeBF16Training::Wave::Slice& slice,
                                  EpilogueOp& epilogue, uint32_t& startBlock, int32_t& pending)
    {
        const uint32_t h = params.tilingData->k, f = params.tilingData->n / 2U;
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t core = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        AscendC::GlobalTensor<bfloat16_t> a, b;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.workspaceInfo.ptrA) + slice.packed * h);
        b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.bGmAddr) +
                          static_cast<uint64_t>(slice.expert) * 2U * f * h);
        if (slice.row == 0 && slice.rows == slice.fullRows && slice.rows <= kRows)
            b.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        if (2U * f <= L1_TILE_N) a.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        auto ta = tla::MakeTensor(a, tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(slice.rows, h),
                                  Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(b, tla::MakeLayout<bfloat16_t, WeightLayout>(h, 2U * f), Catlass::Arch::PositionGM{});
        auto gate = tla::MakeTensor(resource.ubBuf.template GetBufferByByte<bfloat16_t>(0), MakeWaveGmm1UbLayout(),
                                    Catlass::Arch::PositionUB{});
        auto up = tla::MakeTensor(resource.ubBuf.template GetBufferByByte<bfloat16_t>(kTileElements * 2U),
                                  MakeWaveGmm1UbLayout(), Catlass::Arch::PositionUB{});
        BlockScheduler scheduler({slice.rows, f, h}, Catlass::MakeCoord(kRows, L1_TILE_N));
        const uint32_t tasks = scheduler.GetCoreLoops();
        const uint32_t first = (core < startBlock ? core + cores : core) - startBlock;
        const uint32_t width = SelectGmm1NStripeWidth<kRows>(slice.rows, f, cores);
        bool ready = false;
        for (uint32_t task = first; task < tasks; task += cores) {
            const auto coord =
                width ? GetGmm1AdaptiveNStripeCoord<kRows>(task, slice.rows, f, width) : scheduler.GetBlockCoord(task);
            const auto shape = scheduler.GetActualBlockShape(coord);
            const uint32_t m = coord.m() * kRows, n = coord.n() * L1_TILE_N;
            if constexpr (g_coreType == AscendC::AIC) {
                if (!ready) {
                    const uint32_t target = slice.ordinal * cores;
                    auto* flag = reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.ptrFlagDispatchToGmm1) +
                                 slice.expert * INT_CACHELINE;
                    while (AscendC::ReadGmByPassDCache(flag) < target) {
                    }
                    ready = true;
                }
                if (pending) WaitForVector();
                auto tileA = tla::GetTile(ta, tla::MakeCoord(m, 0U), tla::MakeShape(shape.m(), shape.k()));
                auto tileGate = tla::GetTile(gate, tla::MakeCoord(0U, 0U), tla::MakeShape(shape.m(), shape.n()));
                auto tileUp = tla::GetTile(up, tla::MakeCoord(0U, 0U), tla::MakeShape(shape.m(), shape.n()));
                for (uint32_t part = 0; part < 2U; ++part) {
                    auto tileB =
                        tla::GetTile(tb, tla::MakeCoord(0U, n + part * f), tla::MakeShape(shape.k(), shape.n()));
                    auto& output = part ? tileUp : tileGate;
                    blockMmad(tileA, tileB, output, shape);
                }
                NotifyVector();
                pending = 1;
            } else if constexpr (g_coreType == AscendC::AIV) {
                WaitForCube();
                epilogue.template Run<MegaMoeBF16Training::kSaveBf16>({shape.m(), shape.n(), 0, 0},
                                                                      {static_cast<int64_t>(m) * f + n, 0, 0, 0, 0, 0});
                NotifyCube();
            }
        }
        startBlock = (startBlock + tasks) % cores;
    }
};

}  // namespace MegaMoeBF16ForwardCatlassImpl
