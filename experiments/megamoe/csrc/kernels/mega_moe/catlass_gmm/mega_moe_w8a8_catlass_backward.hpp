// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Ordinary W8A8 tile policy: no BF16 adaptive/slow-group specialization.
#pragma once
#include "kernel_operator.h"
#include "mc2_common/mc2_kernel_utils.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/kernel/mx_matmul_tla.hpp"
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_HSTU_QK_SPLIT_ROW_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_HSTU_PV_SPLIT_ROW_TLA_HPP
#include "catlass/gemm/block/block_mmad.hpp"
#undef CATLASS_GEMM_BLOCK_BLOCK_MMAD_HSTU_PV_SPLIT_ROW_TLA_HPP
#undef CATLASS_GEMM_BLOCK_BLOCK_MMAD_HSTU_QK_SPLIT_ROW_TLA_HPP
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"

namespace MegaMoeTrainW8A8 {
using Arch = Catlass::Arch::Ascend950;
using Data = float8_e4m3_t;
using Scale = float8_e8m0_t;
using Output = bfloat16_t;
using Row = Catlass::layout::RowMajor;
using Col = Catlass::layout::ColumnMajor;
using ScaleA = decltype(tla::MakeMxScaleLayout<Scale, Row, false>(uint32_t{}, uint32_t{}));
using ScaleB = decltype(tla::MakeMxScaleLayout<Scale, Col, true>(uint32_t{}, uint32_t{}));
using L1 = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>;
using L0 = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<128>>;
using Policy = Catlass::Gemm::MmadMx<Arch, true>;
using BaseCopy =
    Catlass::Gemm::Tile::PackedMxTileCopyTla<Arch, Data, Row, Data, Col, Scale, ScaleA, Scale, ScaleB, Output, Row>;
// Scale writers/readers bypass L2, including reinterpreted E8M0 pairs.
// Locally received/quantized FP8 data use NORMAL on both MTE3 and MTE2;
// repeated A/B tile reads can reuse L2 without retaining stale generations.
template <class Delegate>
struct TrainingGmRead {
    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(const TensorDst& dst, const TensorSrc& src)
    {
        auto fresh = src;
        fresh.data().SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        Delegate copy;
        copy(dst, fresh);
    }
};
template <class Delegate>
struct TrainingGmWrite {
    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(const TensorDst& dst, const TensorSrc& src, uint8_t flag = 0)
    {
        auto fresh = dst;
        fresh.data().SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        Delegate copy;
        copy(fresh, src, flag);
    }
};
struct Copy : BaseCopy {
    template <class T>
    using CopyGmToL1MxScaleA = TrainingGmRead<typename BaseCopy::template CopyGmToL1MxScaleA<T>>;
    template <class T>
    using CopyGmToL1MxScaleB = TrainingGmRead<typename BaseCopy::template CopyGmToL1MxScaleB<T>>;
    template <class T>
    using CopyL0CToDst = TrainingGmWrite<typename BaseCopy::template CopyL0CToDst<T>>;
};
using Mmad = Catlass::Gemm::Block::BlockMmadTla<Policy, L1, L0, Data, Data, Output, void, Copy>;
using Scheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
using Gemm = Catlass::Gemm::Kernel::MxMatmulTla<Mmad, void, Scheduler>;

// One resource/TPipe and MMAD instance for the lifetime of the MIX kernel.
// Constructing/destroying a TPipe after cross-core publication resets shared
// execution state; ordinary W8A8 constructs its resources before routing.
class FullKGemm {
    Catlass::Arch::Resource<Arch> resource;
    Mmad mm;

public:
    __aicore__ inline FullKGemm() : mm(resource) {}
    __aicore__ inline void Run(const Gemm::Params& p, uint32_t singleJob = 0xffffffffU,
                               uint32_t explicitM = 0xffffffffU, uint32_t explicitN = 0)
    {
        AscendC::GlobalTensor<Data> ga, gb;
        AscendC::GlobalTensor<Scale> sa, sb;
        AscendC::GlobalTensor<Output> gc;
        ga.SetGlobalBuffer((__gm__ Data*)p.ptrA);
        gb.SetGlobalBuffer((__gm__ Data*)p.ptrB);
        sa.SetGlobalBuffer((__gm__ Scale*)p.ptrMxScaleA);
        sb.SetGlobalBuffer((__gm__ Scale*)p.ptrMxScaleB);
        gc.SetGlobalBuffer((__gm__ Output*)p.ptrC);
        // Match ordinary W8A8's reuse policy. Local dynamic data producers
        // update L2 as well; communication source heaps and scales bypass it.
        ga.SetL2CacheHint(p.problemShape.n() > 256 ? AscendC::CacheMode::CACHE_MODE_NORMAL
                                                   : AscendC::CacheMode::CACHE_MODE_DISABLE);
        gb.SetL2CacheHint(p.problemShape.m() > 256 ? AscendC::CacheMode::CACHE_MODE_NORMAL
                                                   : AscendC::CacheMode::CACHE_MODE_DISABLE);
        auto a = tla::MakeTensor(ga, p.layoutA, Catlass::Arch::PositionGM{});
        auto b = tla::MakeTensor(gb, p.layoutB, Catlass::Arch::PositionGM{});
        auto as = tla::MakeTensor(sa, p.layoutMxScaleA, Catlass::Arch::PositionGM{});
        auto bs = tla::MakeTensor(sb, p.layoutMxScaleB, Catlass::Arch::PositionGM{});
        auto c = tla::MakeTensor(gc, p.layoutC, Catlass::Arch::PositionGM{});
        Scheduler scheduler(p.problemShape, Catlass::MatrixCoord{256U, 256U});
        // A single persistent MMAD owns these L1/L0 rings throughout the
        // fused kernel. Tile calls keep its natural buffer progression:
        // resetting MX-scale ring indices between Gate and Up would discard
        // that state while the preceding tile may still be in flight.
        // The whole-matrix compatibility path retains its phase reset.
        if (singleJob == 0xffffffffU) mm.RestoreStatus();
        const uint32_t first = singleJob == 0xffffffffU ? AscendC::GetBlockIdx() : singleJob;
        const uint32_t end = singleJob == 0xffffffffU ? scheduler.GetCoreLoops() : singleJob + 1;
        for (uint32_t job = first; job < end; job += AscendC::GetBlockNum()) {
            auto coord =
                explicitM == 0xffffffffU ? scheduler.GetBlockCoord(job) : Catlass::GemmCoord{explicitM, explicitN, 0};
            auto shape = scheduler.GetActualBlockShape(coord);
            auto av = tla::GetTile(a, tla::MakeCoord(coord.m() * 256, 0U), tla::MakeShape(shape.m(), shape.k()));
            auto bv = tla::GetTile(b, tla::MakeCoord(0U, coord.n() * 256), tla::MakeShape(shape.k(), shape.n()));
            auto avs = tla::GetTile(as, tla::MakeCoord(coord.m() * 256, 0U), tla::MakeShape(shape.m(), shape.k() / 32));
            auto bvs = tla::GetTile(bs, tla::MakeCoord(0U, coord.n() * 256), tla::MakeShape(shape.k() / 32, shape.n()));
            auto cv =
                tla::GetTile(c, tla::MakeCoord(coord.m() * 256, coord.n() * 256), tla::MakeShape(shape.m(), shape.n()));
            mm(av, bv, cv, shape, avs, bvs);
        }
        // Tile readiness is published by the scheduler after a FIX->S
        // completion fence. Do not drain every pipe between paired Gate/Up
        // calls: BlockMmad already protects its L1/L0 reuse with events and
        // UNIT_FLAG. Whole-matrix users retain the original completion fence.
        if (singleJob == 0xffffffffU) AscendC::PipeBarrier<PIPE_ALL>();
    }
};
struct VectorNoGemm {
    __aicore__ inline void Run(const Gemm::Params&, uint32_t = 0xffffffffU, uint32_t = 0xffffffffU, uint32_t = 0) {}
};
using KernelGemm = std::conditional_t<g_coreType == AscendC::AIC, FullKGemm, VectorNoGemm>;

// B is physically [N,K] row-major (logically [K,N] column-major).
// Both scale arrays are physically [..., ceil(K/64), 2].
#if defined(__NPU_DEVICE__) || defined(__ASC_NPU_HOST__)
__host_aicore__
#elif defined(__CCE__)
__host__ __aicore__
#endif
    inline Gemm::Params
    MakeParams(GM_ADDR a, GM_ADDR b, GM_ADDR sa, GM_ADDR sb, GM_ADDR c, uint32_t m, uint32_t n, uint32_t k)
{
    const uint32_t sk = (k + 63) / 64 * 2;
    return {Catlass::GemmCoord(m, n, k),
            a,
            tla::MakeLayout<Data, Row>(m, k),
            b,
            tla::MakeLayout<Data, Col>(k, n),
            sa,
            tla::MakeMxScaleLayout<Scale, Row, false>(m, sk),
            sb,
            tla::MakeMxScaleLayout<Scale, Col, true>(sk, n),
            c,
            tla::MakeLayout<Output, Row>(m, n)};
}
}  // namespace MegaMoeTrainW8A8
