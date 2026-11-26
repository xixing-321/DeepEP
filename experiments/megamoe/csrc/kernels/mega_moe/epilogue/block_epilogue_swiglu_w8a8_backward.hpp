/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * W8A8 SwiGLU backward consumes saved or recomputed Gate/Up.
 * It reconstructs H in FP32 and rounds Hp and dZ to BF16 for quantization.
 * The route gradient reduces complete FP32 dHp*H rows in UB.
 *
 * The register instruction organisation (Exp / Div / Mul / Cast in
 * __VEC_SCOPE__) mirrors ComputeSwiglu() of the forward epilogue.
 *
 * Tile addressing convention (matches the forward GMM1 epilogue): the caller
 * passes a single flattened block offset `mLoc * F + nLoc` (rows * plane
 * width + tile column base) in Get<0>(blockCoord).  Row strides are handled
 * by DataCopyPad.
 */
#ifndef DEEPEP_BLOCK_SWIGLU_BACKWARD_TRAIN_W8A8_H
#define DEEPEP_BLOCK_SWIGLU_BACKWARD_TRAIN_W8A8_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "catlass/arch/arch.hpp"
#include "tla/layout.hpp"
#include "common/mega_moe_utils.hpp"
#include "common/mega_moe_w8a8_saved_layout.hpp"

namespace MegaMoeTrainW8A8Vec {

using AscendC::SyncFunc;
using MegaMoeImpl::Get;
using MegaMoeTrainW8A8::BACKWARD_DP_PARTIAL_TILE_N;
using MegaMoeTrainW8A8::kMegaMoeSwigluLimitDisabled;
using MegaMoeTrainW8A8::SWIGLU_N_HALF;

static constexpr AscendC::Reg::DivSpecificMode kBackwardDivMode = {AscendC::Reg::MaskMergeMode::ZEROING, true};
static constexpr AscendC::Reg::CastTrait kBackwardCastBF16ToFP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait kBackwardCastFP32ToBF16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

// Amplifier for the clamp range indicator: |raw - clamped| is exactly zero
// for unclamped elements, so multiplying by this scale saturates any nonzero
// difference before min(1, .) turns it into a clean 0/1 mask.
static constexpr float kClampMaskScale = 1.0e30F;

// Decode rows of E4M3FN with one E8M0 scale per 32 features. Short GM
// scale rows are padded to the 32-byte DMA alignment in UB.
__aicore__ inline void DecodeMxFp8Rows(AscendC::LocalTensor<int8_t> dataTile, AscendC::LocalTensor<int8_t> scaleTile,
                                      AscendC::LocalTensor<bfloat16_t> dst, uint16_t rows, uint32_t columns)
{
    constexpr uint32_t fp32PerReg = 256U / sizeof(float);
    constexpr uint32_t kMxScaleRowBytes = 32;
    const uint32_t groups = columns / 32;
    auto data = (__ubuf__ fp8_e4m3fn_t*)dataTile.GetPhyAddr();
    auto scales = (__ubuf__ fp8_e8m0_t*)scaleTile.GetPhyAddr();
    auto decoded = (__ubuf__ bfloat16_t*)dst.GetPhyAddr();
    __VEC_SCOPE__
    {
        static constexpr AscendC::Reg::CastTrait fp8ToWide = {
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};
        namespace V = AscendC::Reg;
        auto all = V::CreateMask<uint16_t, V::MaskPattern::ALL>();
        V::RegTensor<int16_t> lanes;
        V::RegTensor<uint16_t> groupInPair;
        V::Arange(lanes, int16_t(0));
        V::ShiftRights(groupInPair, reinterpret_cast<V::RegTensor<uint16_t>&>(lanes), int16_t(6), all);
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t groupCount = groups;
            auto validGroups = V::UpdateMask<bfloat16_t>(groupCount);
            V::RegTensor<fp8_e8m0_t> encoded;
            V::RegTensor<bfloat16_t> compact;
            V::DataCopy<fp8_e8m0_t, V::LoadDist::DIST_UNPACK_B8>(encoded, scales + row * kMxScaleRowBytes);
            V::Cast<bfloat16_t, fp8_e8m0_t, fp8ToWide>(compact, encoded, validGroups);
            uint32_t dataRemaining = columns;
            const uint16_t dataRepeats = static_cast<uint16_t>(Ops::Base::CeilDiv(columns, fp32PerReg));
            for (uint16_t i = 0; i < dataRepeats; ++i) {
                auto mask = AscendC::Reg::UpdateMask<float>(dataRemaining);
                AscendC::Reg::RegTensor<float> scale;
                AscendC::Reg::RegTensor<fp8_e4m3fn_t> src;
                AscendC::Reg::RegTensor<float> value;
                AscendC::Reg::RegTensor<float> result;
                AscendC::Reg::RegTensor<bfloat16_t> rounded;
                // BF16 -> FP32 selects even b16 lanes. Repeat each MX scale in 64 b16 lanes
                // so the FP32 product uses one scale for each consecutive group of 32 values.
                V::RegTensor<uint16_t> group;
                V::RegTensor<bfloat16_t> expanded;
                V::Adds(group, groupInPair, uint16_t(2U * i), all);
                V::Gather(expanded, compact, group);
                V::Cast<float, bfloat16_t, fp8ToWide>(scale, expanded, mask);
                AscendC::Reg::DataCopy<fp8_e4m3fn_t, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(
                    src, data + row * columns + i * fp32PerReg);
                AscendC::Reg::Cast<float, fp8_e4m3fn_t, fp8ToWide>(value, src, mask);
                AscendC::Reg::Mul(result, scale, value, mask);
                AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(rounded, result, mask);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                    decoded + row * columns + i * fp32PerReg, rounded, mask);
            }
        }
    }
}

template <typename L0TileShape_>
class TrainSwigluBackwardBF16 {
public:
    struct StoreHpInGm {
        static constexpr bool enabled = false;
    };
    struct Arguments {
        GM_ADDR gatePlaneGmAddr{nullptr};
        GM_ADDR upPlaneGmAddr{nullptr};
        GM_ADDR gateScaleGmAddr{nullptr};
        GM_ADDR upScaleGmAddr{nullptr};
        GM_ADDR gateHighGmAddr{nullptr};
        GM_ADDR upLowGmAddr{nullptr};
        GM_ADDR upHighGmAddr{nullptr};
        GM_ADDR gradZGmAddr{nullptr};
        GM_ADDR hpGmAddr{nullptr};
        GM_ADDR dpLow{nullptr};
        GM_ADDR packedTopkWeightsGmAddr{nullptr};
        uint32_t baseM{0};
        uint32_t baseN{0};
        // dsv4 swiglu_limit clamp; kMegaMoeSwigluLimitDisabled disables it.
        float swigluLimit{kMegaMoeSwigluLimitDisabled};
        uint32_t savedGateUpMode{MegaMoeTrainW8A8::kSaveNone};
        uint32_t topkWeightType{0};
    };

    using Params = Arguments;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    __aicore__ inline void Init(Params const& params)
    {
        if constexpr (g_coreType == AscendC::AIV) {
            // Store the whole parameter set: ComputeForwardAux /
            // ComputeSwigluGrad / ComputeSwigluRecompute read
            // params_.swigluLimit from it.  Without this copy the limit
            // stays at the default (disabled) value and the clamp is
            // silently skipped in the backward pass.
            params_ = params;
            gatePlaneGm_.SetGlobalBuffer((__gm__ bfloat16_t*)params.gatePlaneGmAddr);
            upPlaneGm_.SetGlobalBuffer((__gm__ bfloat16_t*)params.upPlaneGmAddr);
            gateMxFp8Gm_.SetGlobalBuffer((__gm__ int8_t*)params.gatePlaneGmAddr);
            upMxFp8Gm_.SetGlobalBuffer((__gm__ int8_t*)params.upPlaneGmAddr);
            gateScaleGm_.SetGlobalBuffer((__gm__ int8_t*)params.gateScaleGmAddr);
            upScaleGm_.SetGlobalBuffer((__gm__ int8_t*)params.upScaleGmAddr);
            gateHighGm_.SetGlobalBuffer((__gm__ int8_t*)params.gateHighGmAddr);
            upLowGm_.SetGlobalBuffer((__gm__ int8_t*)params.upLowGmAddr);
            upHighGm_.SetGlobalBuffer((__gm__ int8_t*)params.upHighGmAddr);
            gradZGm_.SetGlobalBuffer((__gm__ bfloat16_t*)params.gradZGmAddr);
            hpGm_.SetGlobalBuffer((__gm__ bfloat16_t*)params.hpGmAddr);
            packedTopkWeightsGm_.SetGlobalBuffer((__gm__ bfloat16_t*)params.packedTopkWeightsGmAddr);
            packedTopkWeightsFp32Gm_.SetGlobalBuffer((__gm__ float*)params.packedTopkWeightsGmAddr);
            gatePlaneGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            upPlaneGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            gateMxFp8Gm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            upMxFp8Gm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            gateScaleGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            upScaleGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            gateHighGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            upLowGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            upHighGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            gradZGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            hpGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            packedTopkWeightsGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            packedTopkWeightsFp32Gm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
    }

    // baseRows: cumulative expert row offset (M_OFFSET equivalent). All four
    // planes are addressed from this single row base: gate/up/H planes have
    // row width F, gradZ has row width N.
    __aicore__ inline void UpdateGlobalAddr(int64_t baseRows)
    {
        if constexpr (g_coreType == AscendC::AIV) {
            rowBase_ = baseRows;
        }
    }

    // problemShape = {0, N, K, 0}; F = N / 2.
    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape)
    {
        outputN_ = Get<1>(problemShape);
        outputF_ = outputN_ / SWIGLU_N_HALF;
    }

    __aicore__ inline void LoadDpProducts(uint32_t row, uint32_t col, uint16_t rows, uint16_t columns,
                                          AscendC::LocalTensor<float> products)
    {
        const auto index = static_cast<uint64_t>(row) * outputF_ + col;
        CopyPlaneTileToUb(dhpGm_, index, rows, columns, outputF_, l0cOutUbFirst_);
        LoadGateUpTiles(row, col, rows, columns, l0cOutUbSecond_, extraA_);
        ComputeForwardAux<true>(rows, columns, columns, products[col]);
        SyncFunc<AscendC::HardEvent::V_MTE2>();
    }

    __aicore__ inline void SetDpInput(GM_ADDR dhp)
    {
        dhpGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(dhp));
        dhpGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }

    // ---- P1: early-topk backward epilogue. dHp is the GMM2 Grad fixpipe
    // output at UB offset 0. First produce Hp; either store Hp or pass
    // its exact BF16 tile to the token quantizer before reloading gate/up.
    // Then apply dH=p*dHp and produce dZ. Reloading is intentional:
    // three A5 UB tiles cannot simultaneously retain dHp/gate/up/Hp. ----
    template <bool StoreGradZ = true, class HpConsumer = StoreHpInGm>
    __aicore__ inline void SwiGLUGrad(const BlockShape& blockShape, const BlockCoord& blockCoord,
                                      HpConsumer consumeHp = {})
    {
        if constexpr (g_coreType != AscendC::AIV) {
            return;
        }
        const uint32_t singleM = Get<0>(blockShape);
        const uint32_t singleN = Get<1>(blockShape);
        if (singleM == 0 || singleN == 0) {
            return;
        }
        const int64_t flatOffset = Get<0>(blockCoord);
        const uint32_t mLoc = static_cast<uint32_t>(flatOffset / outputF_);
        const uint32_t nLoc = static_cast<uint32_t>(flatOffset % outputF_);
        const uint32_t rowStrideInPlane = Ops::Base::CeilAlign(
            static_cast<uint32_t>(singleN), static_cast<uint32_t>(AscendC::ONE_BLK_SIZE / sizeof(bfloat16_t)));

        LoadPackedWeights(singleM, mLoc);

        // The caller's WaitForCube runs on PIPE_V and protects the P1 dHp
        // Fixpipe tile. Bridge PIPE_V to MTE2 before loading gate/up into the
        // other two work tiles. In save=none they come from current-backward
        // recompute GM; saved modes read forward-persistent GM instead.
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        // Load gate/up while dHp remains in the Fixpipe-produced first tile.
        LoadGateUpTiles(rowBase_ + mLoc, nLoc, singleM, singleN, l0cOutUbSecond_, extraA_);

        if constexpr (HpConsumer::enabled && !StoreGradZ) {
            const bool fusedDp = params_.dpLow != nullptr && params_.savedGateUpMode == MegaMoeTrainW8A8::kSaveMxFp8 &&
                                 params_.swigluLimit != kMegaMoeSwigluLimitDisabled && rowStrideInPlane == singleN &&
                                 singleN % 64U == 0;
            if (fusedDp) {
                SaveDpProducts<true>(rowBase_ + mLoc, nLoc, singleM, singleN, rowStrideInPlane);
                AscendC::PipeBarrier<PIPE_V>();
                ApplySavedClipGradientMask(singleM, singleN, rowStrideInPlane);
                SyncFunc<AscendC::HardEvent::V_MTE2>();
                SyncFunc<AscendC::HardEvent::V_MTE3>();
                consumeHp.template operator()<true>(extraA_, singleM, singleN, rowStrideInPlane);
                return;
            }
            if (params_.dpLow != nullptr) {
                SaveDpProducts(rowBase_ + mLoc, nLoc, singleM, singleN, rowStrideInPlane);
            }

            if (params_.savedGateUpMode == MegaMoeTrainW8A8::kSaveMxFp8 && rowStrideInPlane == singleN &&
                singleN % 64U == 0) {
                // Keep Hp, dGate and dUp in separate UB planes. Consume the clip masks before Hp reuses their scratch.
                if (params_.swigluLimit == kMegaMoeSwigluLimitDisabled) {
                    ComputeContiguousGrad<true>(singleM, singleN);
                } else {
                    ComputeSwigluGradImpl<true, true>(singleM, singleN, rowStrideInPlane);
                }
                SyncFunc<AscendC::HardEvent::V_MTE2>();
                SyncFunc<AscendC::HardEvent::V_MTE3>();
                consumeHp.template operator()<true>(extraA_, singleM, singleN, rowStrideInPlane);
                return;
            }
        }

        ComputeForwardAux(singleM, singleN, rowStrideInPlane);
        AscendC::PipeBarrier<PIPE_V>();

        // Tile2 holds Hp until the token quantizer has read it.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        if constexpr (HpConsumer::enabled) {
            consumeHp(extraA_, singleM, singleN, rowStrideInPlane);
        } else {
            AscendC::DataCopyExtParams auxCopyParams{
                static_cast<uint16_t>(singleM), static_cast<uint32_t>(singleN * sizeof(bfloat16_t)), 0,
                static_cast<uint32_t>((outputF_ - singleN) * sizeof(bfloat16_t)), 0};
            AscendC::DataCopyPad(hpGm_[rowBase_ * outputF_ + flatOffset], extraA_, auxCopyParams);
        }

        // Both buffers are MTE3 sources and will be overwritten by MTE2.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        LoadGateUpTiles(rowBase_ + mLoc, nLoc, singleM, singleN, l0cOutUbSecond_, extraA_);

        // dB remains raw BF16 in tile0; multiply by p in FP32 inside the
        // derivative, without materializing an intermediate BF16 dH.
        ComputeSwigluGrad(singleM, singleN, rowStrideInPlane);

        // extraA_ holds the up tile.  PIPE_V reads it while computing dGate,
        // and the next loop iteration reuses the same UB through PIPE_MTE2.
        // Keep that loop-carried WAR dependency explicit; MTE2_V above only
        // orders the current load before V and does not protect the next load.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);

        // dGate/dUp are produced by PIPE_V and consumed by PIPE_MTE3.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        if constexpr (StoreGradZ) {
            // dGate tile (UB offset 0) -> ptrGradZ rows at col nLoc.
            // dUp tile (UB offset kTileBytes) -> ptrGradZ rows at col F + nLoc.
            AscendC::DataCopyExtParams gradCopyParams{
                static_cast<uint16_t>(singleM), static_cast<uint32_t>(singleN * sizeof(bfloat16_t)), 0,
                static_cast<uint32_t>((outputN_ - singleN) * sizeof(bfloat16_t)), 0};
            AscendC::DataCopyPad(gradZGm_[rowBase_ * outputN_ + mLoc * outputN_ + nLoc], l0cOutUbFirst_,
                                 gradCopyParams);
            AscendC::DataCopyPad(gradZGm_[rowBase_ * outputN_ + mLoc * outputN_ + outputF_ + nLoc], l0cOutUbSecond_,
                                 gradCopyParams);
            // Both MTE3 sources become MTE2 destinations in the next tile.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        }
    }

private:
    static constexpr uint32_t kTileRows = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t kTileColumns = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t kTileElements = kTileRows * kTileColumns;
    static constexpr uint32_t kTileBytes = kTileElements * sizeof(bfloat16_t);
    static constexpr uint32_t kDecodeRows = 32;
    static constexpr uint32_t kMxDataElements = kDecodeRows * kTileColumns;
    static constexpr uint32_t kMxScaleRowBytes = 32;
    static constexpr uint32_t kMxScaleBytes = kDecodeRows * kMxScaleRowBytes;
    static constexpr uint32_t kMxDecodedElements = kDecodeRows * kTileColumns;
    static constexpr uint32_t kMxDecodedBytes = kMxDecodedElements * sizeof(float);
    static constexpr uint32_t kMxDataOffset = 3U * kTileBytes;
    static constexpr uint32_t kMxScaleOffset = kMxDataOffset + kMxDataElements;
    static constexpr uint32_t kMxDecodedOffset = kMxScaleOffset + kMxScaleBytes;
    // Retain every decoded row until the whole-tile gradient mask consumes it.
    static constexpr uint32_t kClipMaskBytes = kTileRows * kMxScaleRowBytes;
    static constexpr uint32_t kGateHighOffset = kMxDecodedOffset + kMxDecodedBytes;
    static constexpr uint32_t kUpLowOffset = kGateHighOffset + kClipMaskBytes;
    static constexpr uint32_t kUpHighOffset = kUpLowOffset + kClipMaskBytes;
    static constexpr uint32_t kMxScratchEnd = kUpHighOffset + kClipMaskBytes;
    static_assert(kMxScratchEnd <= Catlass::Arch::Ascend950::UB_SIZE,
                  "The backward SwiGLU work tiles and MXFP8 decoder must fit in A5 UB");

    static constexpr uint32_t kWeightBytes = kTileRows * sizeof(float);
    static constexpr uint32_t kWeightOffset = Catlass::Arch::Ascend950::UB_SIZE - kWeightBytes;
    static_assert(kWeightOffset >= kMxScratchEnd, "Route weights must not overlap decoder stores or clip-mask readers");
    __aicore__ inline void LoadPackedWeights(uint32_t rows, uint32_t mLoc)
    {
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        if (params_.topkWeightType) {
            AscendC::DataCopyPad(packedWeightFp32Ub_, packedTopkWeightsFp32Gm_[rowBase_ + mLoc],
                                 AscendC::DataCopyExtParams{1, rows * 4, 0, 0, 0},
                                 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        } else {
            AscendC::DataCopyPad(packedWeightBf16Ub_, packedTopkWeightsGm_[rowBase_ + mLoc],
                                 AscendC::DataCopyExtParams{1, rows * 2, 0, 0, 0},
                                 AscendC::DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
        }
        SyncFunc<AscendC::HardEvent::MTE2_V>();
    }

    __aicore__ inline void LoadGateUpTiles(uint32_t rowStart, uint32_t nLoc, uint32_t singleM, uint32_t singleN,
                                           AscendC::LocalTensor<bfloat16_t>& gateDst,
                                           AscendC::LocalTensor<bfloat16_t>& upDst)
    {
        if (params_.savedGateUpMode == MegaMoeTrainW8A8::kSaveMxFp8) {
            DecodeMxFp8Plane(gateMxFp8Gm_, gateScaleGm_, gateHighGm_, gateHighGm_, rowStart, nLoc, singleM, singleN,
                             gateDst, true);
            DecodeMxFp8Plane(upMxFp8Gm_, upScaleGm_, upLowGm_, upHighGm_, rowStart, nLoc, singleM, singleN, upDst,
                             false);
            return;
        }
        const int64_t index = static_cast<int64_t>(rowStart) * outputF_ + nLoc;
        CopyPlaneTileToUb(gatePlaneGm_, index, singleM, singleN, outputF_, gateDst);
        CopyPlaneTileToUb(upPlaneGm_, index, singleM, singleN, outputF_, upDst);
    }

    __aicore__ inline void CopyMxRowsToUb(const AscendC::GlobalTensor<int8_t>& src, int64_t index, uint16_t rows,
                                          uint32_t rowBytes, uint32_t srcRowBytes, AscendC::LocalTensor<int8_t> dst)
    {
        AscendC::DataCopyExtParams copyParams{rows, rowBytes, srcRowBytes - rowBytes, 0, 0};
        AscendC::DataCopyPadExtParams<int8_t> padParams{false, 0U, 0U, 0U};
        AscendC::DataCopyPad(dst, src[index], copyParams, padParams);
    }

    // Decode one E4M3FN plane in small row chunks. The E8M0 conversion and
    // group broadcast mirror the production MXFP8 combine path; no full
    // decompressed GM plane is materialized.
    __aicore__ inline void DecodeMxFp8Plane(const AscendC::GlobalTensor<int8_t>& dataGm,
                                            const AscendC::GlobalTensor<int8_t>& scaleGm,
                                            const AscendC::GlobalTensor<int8_t>& lowMaskGm,
                                            const AscendC::GlobalTensor<int8_t>& highMaskGm, uint32_t rowStart,
                                            uint32_t nLoc, uint32_t singleM, uint32_t singleN,
                                            AscendC::LocalTensor<bfloat16_t>& dst, bool gatePlane)
    {
        const uint32_t scaleWidth = outputF_ / 32;
        const uint32_t groups = singleN / 32;
        const bool hasLowMask = params_.gateHighGmAddr != nullptr;
        // Mask rows occupy dedicated UB for the lifetime of this tile.
        if (hasLowMask) {
            const uint32_t maskWidthBytes = scaleWidth * sizeof(uint32_t);
            const int64_t index = static_cast<int64_t>(rowStart) * maskWidthBytes + (nLoc / 32) * sizeof(uint32_t);
            AscendC::SyncFunc<AscendC::HardEvent::V_MTE2>();
            CopyMxRowsToUb(lowMaskGm, index, singleM, groups * sizeof(uint32_t), maskWidthBytes,
                           gatePlane ? gateHighMask_ : upLowMask_);
            if (!gatePlane) {
                CopyMxRowsToUb(highMaskGm, index, singleM, groups * sizeof(uint32_t), maskWidthBytes, upHighMask_);
            }
        }
        for (uint32_t chunkRow = 0; chunkRow < singleM; chunkRow += kDecodeRows) {
            const uint16_t rows =
                static_cast<uint16_t>(singleM - chunkRow > kDecodeRows ? kDecodeRows : singleM - chunkRow);
            const uint32_t globalRow = rowStart + chunkRow;
            const int64_t dataIndex = static_cast<int64_t>(globalRow) * outputF_ + nLoc;
            const int64_t scaleIndex = static_cast<int64_t>(globalRow) * scaleWidth + nLoc / 32;
            // The scratch regions are overwritten by MTE2 in the next chunk;
            // wait for all vector consumers from this chunk first.
            AscendC::SyncFunc<AscendC::HardEvent::V_MTE2>();
            CopyMxRowsToUb(dataGm, dataIndex, rows, singleN, outputF_, mxData_);
            CopyMxRowsToUb(scaleGm, scaleIndex, rows, groups, scaleWidth, mxScale_);
            AscendC::SyncFunc<AscendC::HardEvent::MTE2_V>();
            DecodeMxFp8Rows(mxData_, mxScale_, dst[chunkRow * singleN], rows, singleN);
        }
        if (hasLowMask) {
            AscendC::PipeBarrier<PIPE_V>();
            ApplyClipSentinels(dst, singleM, singleN, gatePlane);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }



    __aicore__ inline void SetClippedValues(AscendC::LocalTensor<bfloat16_t> dst, AscendC::LocalTensor<int8_t> clipMask,
                                            uint16_t rows, uint32_t columns, uint32_t rowStride, uint16_t replacement)
    {
        using namespace AscendC::Reg;
        constexpr uint32_t lanes = 256U / sizeof(uint16_t);
        const uint16_t repeats = static_cast<uint16_t>(Ops::Base::CeilDiv(columns, lanes));
        auto* dstBits = reinterpret_cast<__ubuf__ uint16_t*>(dst.GetPhyAddr());
        auto* maskBits = reinterpret_cast<__ubuf__ uint8_t*>(clipMask.GetPhyAddr());
        __VEC_SCOPE__
        {
            auto all = CreateMask<uint16_t, MaskPattern::ALL>();
            RegTensor<uint16_t> value;
            Duplicate(value, replacement, all);
            for (uint16_t row = 0; row < rows; ++row) {
                uint32_t remaining = columns;
                for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
                    const uint32_t column = repeat * lanes;
                    MaskReg clipped;
                    // Each packed bit selects one BF16 element (two predicate bits).
                    DataCopy<uint8_t, MaskDist::DIST_US>(clipped, maskBits + row * kMxScaleRowBytes + column / 8U);
                    auto valid = UpdateMask<uint16_t>(remaining);
                    And(clipped, clipped, valid, all);
                    DataCopy(dstBits + row * rowStride + column, value, clipped);
                }
            }
        }
    }

    __aicore__ inline void ApplyClipSentinels(AscendC::LocalTensor<bfloat16_t> dst, uint16_t rows, uint32_t columns,
                                              bool gatePlane)
    {
        constexpr uint16_t positiveInfinityBf16 = 0x7f80U;
        constexpr uint16_t negativeInfinityBf16 = 0xff80U;
        if (gatePlane) {
            SetClippedValues(dst, gateHighMask_, rows, columns, columns, positiveInfinityBf16);
        } else {
            SetClippedValues(dst, upHighMask_, rows, columns, columns, positiveInfinityBf16);
            AscendC::PipeBarrier<PIPE_V>();
            SetClippedValues(dst, upLowMask_, rows, columns, columns, negativeInfinityBf16);
        }
    }

    __aicore__ inline void CopyPlaneTileToUb(const AscendC::GlobalTensor<bfloat16_t>& srcPlane, int64_t index,
                                             uint32_t singleM, uint32_t singleN, uint32_t planeWidth,
                                             AscendC::LocalTensor<bfloat16_t>& dstUb)
    {
        AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(singleM),
                                              static_cast<uint32_t>(singleN * sizeof(bfloat16_t)),
                                              static_cast<uint32_t>((planeWidth - singleN) * sizeof(bfloat16_t)), 0, 0};
        AscendC::DataCopyPadExtParams<bfloat16_t> padParams{false, 0U, 0U, 0U};
        AscendC::DataCopyPad(dstUb, srcPlane[index], copyParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
    }

    template <bool WithGrad = false>
    __aicore__ inline void SaveDpProducts(uint32_t rowStart, uint32_t nLoc, uint32_t singleM, uint32_t singleN,
                                          uint32_t rowStrideInPlane)
    {
        AscendC::GlobalTensor<float> low, high;
        low.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params_.dpLow));
        high.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params_.hpGmAddr));
        low.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        high.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        for (uint32_t r = 0; r < singleM; r += kDecodeRows) {
            const uint16_t take = singleM - r < kDecodeRows ? singleM - r : kDecodeRows;
            // dHp is resident in UB. Reuse its GM tile for the low half of the FP32 products.
            if constexpr (WithGrad) {
                ComputeContiguousGrad<true, true>(take, singleN, r);
            } else {
                ComputeForwardAux<true>(take, singleN, rowStrideInPlane, mxDecoded_, r, singleN);
            }
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            const uint32_t half = singleN / 2;
            AscendC::DataCopyExtParams p{take, half * 4, half / 8, (outputF_ / 2 - half) * 4, 0};
            const uint64_t off = ((uint64_t(rowStart + r) * outputF_) + nLoc) / 2;
            AscendC::DataCopyPad(low[off], mxDecoded_, p);
            AscendC::DataCopyPad(high[off], mxDecoded_[half], p);
            SyncFunc<AscendC::HardEvent::MTE3_V>();
        }
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }

    template <bool ForDp = false>
    __aicore__ inline void ComputeForwardAux(uint16_t mSize, uint16_t nSize, uint32_t rowStride,
                                             AscendC::LocalTensor<float> products = {}, uint32_t inputRow = 0,
                                             uint32_t productStride = 0)
    {
        constexpr uint16_t lanes = 256U / sizeof(float);
        const float limit = params_.swigluLimit;
        const bool limitEnabled = limit != kMegaMoeSwigluLimitDisabled;
        auto* dHp = (__ubuf__ bfloat16_t*)l0cOutUbFirst_.GetPhyAddr();
        auto* gate = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();
        auto* up = (__ubuf__ bfloat16_t*)extraA_.GetPhyAddr();
        auto* weights16 = (__ubuf__ bfloat16_t*)packedWeightBf16Ub_.GetPhyAddr();
        auto* weights32 = (__ubuf__ float*)packedWeightFp32Ub_.GetPhyAddr();
        auto* productPtr = (__ubuf__ float*)products.GetPhyAddr();
        const uint32_t ffn = productStride ? productStride : outputF_;

        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < mSize; ++row) {
                auto all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
                AscendC::Reg::RegTensor<float> pFp, limitFp, negLimitFp;
                AscendC::Reg::RegTensor<bfloat16_t> pBf;
                if constexpr (!ForDp) {
                    if (!params_.topkWeightType) {
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_BRC_B16>(pBf, weights16 + row);
                        AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(pFp, pBf, all);
                    } else {
                        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(pFp, weights32 + row);
                    }
                }
                if (limitEnabled) {
                    AscendC::Reg::Duplicate(limitFp, limit, all);
                    AscendC::Reg::Duplicate(negLimitFp, -limit, all);
                }
                uint32_t remaining = nSize;
                for (uint16_t col = 0; col < nSize; col += lanes) {
                    auto mask = AscendC::Reg::UpdateMask<float>(remaining);
                    const uint32_t offset = (inputRow + row) * rowStride + col;
                    AscendC::Reg::RegTensor<bfloat16_t> gateBf, upBf, dHpBf, hpBf;
                    AscendC::Reg::RegTensor<float> gateFp, upFp, dHpFp, neg, exp, denom, silu;
                    AscendC::Reg::RegTensor<float> hFp, hpFp, productFp;
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(gateBf, gate + offset);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(upBf, up + offset);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gateFp, gateBf, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(upFp, upBf, mask);
                    if (limitEnabled) {
                        AscendC::Reg::Min(gateFp, gateFp, limitFp, mask);
                        AscendC::Reg::Max(upFp, upFp, negLimitFp, mask);
                        AscendC::Reg::Min(upFp, upFp, limitFp, mask);
                    }
                    AscendC::Reg::Muls(neg, gateFp, -1.0F, mask);
                    AscendC::Reg::Exp(exp, neg, mask);
                    AscendC::Reg::Adds(denom, exp, 1.0F, mask);
                    AscendC::Reg::Div<float, &kBackwardDivMode>(silu, gateFp, denom, mask);
                    AscendC::Reg::Mul(hFp, silu, upFp, mask);
                    if constexpr (ForDp) {
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(dHpBf,
                                                                                                    dHp + offset);
                        AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(dHpFp, dHpBf, mask);
                        AscendC::Reg::Mul(productFp, dHpFp, hFp, mask);
                        AscendC::Reg::DataCopy(productPtr + static_cast<uint32_t>(row) * ffn + col, productFp, mask);
                    } else {
                        AscendC::Reg::Mul(hpFp, hFp, pFp, mask);
                        AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(hpBf, hpFp, mask);
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(up + offset, hpBf,
                                                                                                   mask);
                    }
                }
            }
        }
    }

    // Each register covers 64 consecutive columns. Clipped saved tiles also
    // emit dP products in their original column order, before multiplying by p.
    template <bool ComputeHp = false, bool SavedClippedDp = false>
    __aicore__ inline void ComputeContiguousGrad(uint16_t mSize, uint16_t nSize, uint32_t firstRow = 0)
    {
        constexpr float kOne = 1.0F;
        __ubuf__ bfloat16_t* dH = (__ubuf__ bfloat16_t*)l0cOutUbFirst_.GetPhyAddr();
        __ubuf__ bfloat16_t* gate = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();
        __ubuf__ bfloat16_t* up = (__ubuf__ bfloat16_t*)extraA_.GetPhyAddr();
        __ubuf__ bfloat16_t* dGate = dH;
        __ubuf__ bfloat16_t* dUp = gate;
        auto* weights16 = (__ubuf__ bfloat16_t*)packedWeightBf16Ub_.GetPhyAddr();
        auto* weights32 = (__ubuf__ float*)packedWeightFp32Ub_.GetPhyAddr();
        auto* products = (__ubuf__ float*)mxDecoded_.GetPhyAddr();
        const float limit = params_.swigluLimit;
        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < mSize; ++row) {
                auto pMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
                AscendC::Reg::RegTensor<float> pFp;
                if (params_.topkWeightType) {
                    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(pFp, weights32 + firstRow + row);
                } else {
                    AscendC::Reg::RegTensor<bfloat16_t> pBf;
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_BRC_B16>(pBf, weights16 + firstRow + row);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(pFp, pBf, pMask);
                }
                for (uint16_t column = 0; column < nSize; column += 64U) {
                    const uint32_t offset = (firstRow + row) * nSize + column;
                    auto mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
                    AscendC::Reg::RegTensor<bfloat16_t> dhBf, gateBf, upBf;
                    AscendC::Reg::RegTensor<float> dHpRaw, dHFp, gateFp, upFp;
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(dhBf, dH + offset);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(gateBf, gate + offset);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(upBf, up + offset);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(dHpRaw, dhBf, mask);
                    AscendC::Reg::Mul(dHFp, dHpRaw, pFp, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gateFp, gateBf, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(upFp, upBf, mask);
                    AscendC::Reg::RegTensor<float> oneFp, neg, exp, denom, sig, silu;
                    AscendC::Reg::RegTensor<float> siluRounded, gradSwish, gradSwishRounded;
                    AscendC::Reg::RegTensor<float> oneMinusS, inner, term;
                    AscendC::Reg::RegTensor<float> dUpFp, dGateFp;
                    AscendC::Reg::RegTensor<float> gateC, upC;
                    AscendC::Reg::RegTensor<bfloat16_t> dUpOut, dGateOut;
                    // sigmoid(gate) = 1 / (1 + exp(-gate)).  Reg::Div only
                    // takes tensor operands, so materialize a constant 1.
                    AscendC::Reg::Duplicate(oneFp, kOne, mask);
                    if constexpr (SavedClippedDp) {
                        AscendC::Reg::RegTensor<float> upper, lower;
                        AscendC::Reg::Duplicate(upper, limit, mask);
                        AscendC::Reg::Duplicate(lower, -limit, mask);
                        AscendC::Reg::Min(gateC, gateFp, upper, mask);
                        AscendC::Reg::Max(upC, upFp, lower, mask);
                        AscendC::Reg::Min(upC, upC, upper, mask);
                    } else {
                        gateC = gateFp;
                        upC = upFp;
                    }
                    // Use the same sigmoid and fused FP32 SiLU derivative.
                    AscendC::Reg::Muls(neg, gateC, -kOne, mask);
                    AscendC::Reg::Exp(exp, neg, mask);
                    AscendC::Reg::Adds(denom, exp, kOne, mask);
                    AscendC::Reg::Div<float, &kBackwardDivMode>(sig, oneFp, denom, mask);
                    // Differentiate the same fused FP32 SiLU/Up recipe.
                    AscendC::Reg::Div<float, &kBackwardDivMode>(silu, gateC, denom, mask);
                    if constexpr (ComputeHp) {
                        AscendC::Reg::RegTensor<float> h, hp;
                        AscendC::Reg::RegTensor<bfloat16_t> result;
                        AscendC::Reg::Mul(h, silu, upC, mask);
                        if constexpr (SavedClippedDp) {
                            AscendC::Reg::RegTensor<float> product;
                            AscendC::Reg::Mul(product, dHpRaw, h, mask);
                            AscendC::Reg::DataCopy(products + uint32_t(row) * nSize + column, product, mask);
                        }
                        AscendC::Reg::Mul(hp, h, pFp, mask);
                        AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(result, hp, mask);
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(up + offset, result,
                                                                                                   mask);
                    }
                    siluRounded = silu;  // Same FP32 fused activation as forward.
                    AscendC::Reg::Mul(dUpFp, dHFp, siluRounded, mask);

                    // Keep dH*Up in FP32 within the fused derivative.
                    AscendC::Reg::Mul(gradSwish, dHFp, upC, mask);
                    gradSwishRounded = gradSwish;  // No extra BF16 materialization inside fused derivative.
                    // dGate = dS * sig * (1 + gate * (1 - sig)).
                    AscendC::Reg::Muls(oneMinusS, sig, -kOne, mask);
                    AscendC::Reg::Adds(oneMinusS, oneMinusS, kOne, mask);
                    AscendC::Reg::Mul(inner, gateC, oneMinusS, mask);
                    AscendC::Reg::Adds(inner, inner, kOne, mask);
                    AscendC::Reg::Mul(term, sig, inner, mask);
                    AscendC::Reg::Mul(dGateFp, gradSwishRounded, term, mask);

                    AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(dUpOut, dUpFp, mask);
                    AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(dGateOut, dGateFp, mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(dUp + offset, dUpOut,
                                                                                               mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(dGate + offset, dGateOut,
                                                                                               mask);
                }
            }
        }
    }

    // Per-row / per-repeat computation of:
    //   s     = sigmoid(gate)
    //   dUp   = dH * gate * s
    //   dGate = dH * up * s * (1 + gate * (1 - s))
    // dGate accumulates into l0cOutUbFirst_ (overwriting dH), dUp into
    // l0cOutUbSecond_ (overwriting gate). gate/up row-chunks are read before
    // the same offset is overwritten.
    __aicore__ inline void ComputeSwigluGrad(uint16_t mSize, uint16_t nSize, uint32_t rowStride)
    {
        if (params_.swigluLimit == kMegaMoeSwigluLimitDisabled && rowStride == nSize && nSize % 64U == 0) {
            ComputeContiguousGrad(mSize, nSize);
            return;
        }
        if (params_.savedGateUpMode == MegaMoeTrainW8A8::kSaveMxFp8 && params_.gateHighGmAddr != nullptr) {
            // MXFP8 can move an untruncated value across the clipping
            // boundary. Clamp the reconstructed value, but derive the
            // derivative switch exclusively from the forward bit planes.
            ComputeSwigluGradImpl<true>(mSize, nSize, rowStride);
        } else {
            ComputeSwigluGradImpl<false>(mSize, nSize, rowStride);
        }
    }

    template <bool UseSavedClipMask, bool ComputeHp = false>
    __aicore__ inline void ComputeSwigluGradImpl(uint16_t mSize, uint16_t nSize, uint32_t rowStride)
    {
        constexpr uint16_t kElementsPerRepeat = 256U / sizeof(bfloat16_t);
        const uint16_t repeatsPerRow = static_cast<uint16_t>(
            Ops::Base::CeilDiv(static_cast<uint64_t>(nSize), static_cast<uint64_t>(kElementsPerRepeat)));
        constexpr float kOne = 1.0F;
        const float limit = params_.swigluLimit;
        // The disabled limit (FLT_MAX) makes the clamps numerical no-ops;
        // skip them (and the range indicators) entirely so the unlimited
        // path pays nothing extra.
        const bool limitEnabled = limit != kMegaMoeSwigluLimitDisabled;
        __ubuf__ bfloat16_t* dH = (__ubuf__ bfloat16_t*)l0cOutUbFirst_.GetPhyAddr();
        __ubuf__ bfloat16_t* gate = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();
        __ubuf__ bfloat16_t* up = (__ubuf__ bfloat16_t*)extraA_.GetPhyAddr();
        __ubuf__ bfloat16_t* dGate = dH;  // reuse dH tile
        __ubuf__ bfloat16_t* dUp = gate;  // reuse gate tile
        auto* weights16 = (__ubuf__ bfloat16_t*)packedWeightBf16Ub_.GetPhyAddr();
        auto* weights32 = (__ubuf__ float*)packedWeightFp32Ub_.GetPhyAddr();

        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < mSize; ++row) {
                AscendC::Reg::RegTensor<float> pFp;
                if (params_.topkWeightType) {
                    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(pFp,
                                                                                        weights32 + row);
                } else {
                    AscendC::Reg::RegTensor<bfloat16_t> pBf;
                    auto all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_BRC_B16>(
                        pBf, weights16 + row);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(pFp, pBf, all);
                }
                uint32_t remaining = nSize;
                for (uint16_t repeat = 0; repeat < repeatsPerRow; ++repeat) {
                    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
                    AscendC::Reg::RegTensor<bfloat16_t> dH0, dH1, gate0, gate1, up0, up1;
                    AscendC::Reg::RegTensor<float> dHFp0, dHFp1, gateFp0, gateFp1, upFp0, upFp1;
                    const uint32_t offset = (row) * rowStride + repeat * kElementsPerRepeat;
                    AscendC::Reg::DataCopy(dH0, dH + offset);
                    AscendC::Reg::DataCopy(gate0, gate + offset);
                    AscendC::Reg::DataCopy(up0, up + offset);
                    AscendC::Reg::Interleave(dH0, dH1, dH0, dH1);
                    AscendC::Reg::Interleave(gate0, gate1, gate0, gate1);
                    AscendC::Reg::Interleave(up0, up1, up0, up1);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(dHFp0, dH0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(dHFp1, dH1, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gateFp0, gate0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gateFp1, gate1, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(upFp0, up0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(upFp1, up1, mask);

                    constexpr uint16_t kParts = sizeof(float) / sizeof(bfloat16_t);
                    constexpr uint16_t kPartSize = kElementsPerRepeat / kParts;
                    for (uint16_t part = 0; part < kParts; ++part) {
                        // Float lanes, not BF16 byte lanes, govern each 64-value part.
                        const uint32_t consumed = repeat * kElementsPerRepeat + part * kPartSize;
                        uint32_t partRemaining = consumed < nSize ? nSize - consumed : 0;
                        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(partRemaining);
                        auto dHpRaw = part == 0 ? dHFp0 : dHFp1;
                        AscendC::Reg::RegTensor<float> dHFp;
                        AscendC::Reg::Mul(dHFp, dHpRaw, pFp, mask);
                        auto gateFp = part == 0 ? gateFp0 : gateFp1;
                        auto upFp = part == 0 ? upFp0 : upFp1;
                        AscendC::Reg::RegTensor<float> oneFp, neg, exp, denom, sig, silu;
                        AscendC::Reg::RegTensor<float> siluRounded, gradSwish, gradSwishRounded;
                        AscendC::Reg::RegTensor<float> oneMinusS, inner, term;
                        AscendC::Reg::RegTensor<float> dUpFp, dGateFp;
                        AscendC::Reg::RegTensor<bfloat16_t> siluBf16, gradSwishBf16;
                        AscendC::Reg::RegTensor<float> gateC, upC, gateInd, upInd, diff;
                        AscendC::Reg::RegTensor<float> limitReg, negLimitReg;
                        AscendC::Reg::RegTensor<bfloat16_t> dUpOut, dGateOut;
                        // sigmoid(gate) = 1 / (1 + exp(-gate)).  Reg::Div only
                        // takes tensor operands, so materialize a constant 1.
                        AscendC::Reg::Duplicate(oneFp, kOne, mask);
                        // dsv4 swiglu_limit clamp, mirroring the eager path:
                        // gateC = min(gate, L); upC = clamp(up, -L, L).
                        if (limitEnabled) {
                            AscendC::Reg::Muls(limitReg, oneFp, limit, mask);
                            AscendC::Reg::Muls(negLimitReg, oneFp, -limit, mask);
                            AscendC::Reg::Min(gateC, gateFp, limitReg, mask);
                            AscendC::Reg::Max(upC, upFp, negLimitReg, mask);
                            AscendC::Reg::Min(upC, upC, limitReg, mask);
                            if constexpr (UseSavedClipMask) {
                                // The scalar post-pass below applies the exact
                                // forward mask. Keep all lanes enabled here so
                                // an unmasked decoded value crossing L cannot
                                // invent a new zero-gradient branch.
                                gateInd = oneFp;
                                upInd = oneFp;
                            } else {
                                // Range indicators (1 = inside, 0 = clamped):
                                // BF16/none use the exact raw GMM1 value.
                                AscendC::Reg::Sub(diff, gateFp, gateC, mask);
                                AscendC::Reg::Abs(diff, diff, mask);
                                AscendC::Reg::Muls(diff, diff, kClampMaskScale, mask);
                                AscendC::Reg::Min(diff, diff, oneFp, mask);
                                AscendC::Reg::Sub(gateInd, oneFp, diff, mask);
                                AscendC::Reg::Sub(diff, upFp, upC, mask);
                                AscendC::Reg::Abs(diff, diff, mask);
                                AscendC::Reg::Muls(diff, diff, kClampMaskScale, mask);
                                AscendC::Reg::Min(diff, diff, oneFp, mask);
                                AscendC::Reg::Sub(upInd, oneFp, diff, mask);
                            }
                        } else {
                            // No limit: the clamp is the identity, so alias
                            // the raw registers (no instruction) and let the
                            // gradients flow through unmodified.
                            gateC = gateFp;
                            upC = upFp;
                        }
                        // sigmoid / silu / silu' use the CLAMPED gate.
                        AscendC::Reg::Muls(neg, gateC, -kOne, mask);
                        AscendC::Reg::Exp(exp, neg, mask);
                        AscendC::Reg::Adds(denom, exp, kOne, mask);
                        AscendC::Reg::Div<float, &kBackwardDivMode>(sig, oneFp, denom, mask);
                        // Differentiate the same fused FP32 SiLU/Up recipe.
                        AscendC::Reg::Div<float, &kBackwardDivMode>(silu, gateC, denom, mask);
                        if constexpr (ComputeHp) {
                            AscendC::Reg::RegTensor<float> hp;
                            AscendC::Reg::RegTensor<bfloat16_t> hpBf;
                            AscendC::Reg::Mul(hp, silu, upC, mask);
                            AscendC::Reg::Mul(hp, hp, pFp, mask);
                            AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(hpBf, hp, mask);
                            AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                                up + offset + part * kPartSize, hpBf, mask);
                        }
                        siluRounded = silu;  // Same FP32 fused activation as forward.
                        AscendC::Reg::Mul(dUpFp, dHFp, siluRounded, mask);
                        if (limitEnabled) {
                            AscendC::Reg::Mul(dUpFp, dUpFp, upInd, mask);
                        }
                        // Keep dH*Up in FP32 within the fused derivative.
                        AscendC::Reg::Mul(gradSwish, dHFp, upC, mask);
                        gradSwishRounded = gradSwish;  // No extra BF16 materialization inside fused derivative.
                        // dGate = dS * sig * (1 + gateC * (1 - sig)),
                        // zeroed where gate was clamped.
                        AscendC::Reg::Muls(oneMinusS, sig, -kOne, mask);
                        AscendC::Reg::Adds(oneMinusS, oneMinusS, kOne, mask);
                        AscendC::Reg::Mul(inner, gateC, oneMinusS, mask);
                        AscendC::Reg::Adds(inner, inner, kOne, mask);
                        AscendC::Reg::Mul(term, sig, inner, mask);
                        AscendC::Reg::Mul(dGateFp, gradSwishRounded, term, mask);
                        if (limitEnabled) {
                            AscendC::Reg::Mul(dGateFp, dGateFp, gateInd, mask);
                        }
                        AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(dUpOut, dUpFp, mask);
                        AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(dGateOut, dGateFp, mask);
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            dUp + offset + part * kPartSize, dUpOut, mask);
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            dGate + offset + part * kPartSize, dGateOut, mask);
                    }
                }
            }
        }
        if constexpr (UseSavedClipMask) {
            AscendC::PipeBarrier<PIPE_V>();
            ApplySavedClipGradientMask(mSize, nSize, rowStride);
        }
    }

    __aicore__ inline void ApplySavedClipGradientMask(uint16_t rows, uint32_t columns, uint32_t rowStride)
    {
        SetClippedValues(l0cOutUbFirst_, gateHighMask_, rows, columns, rowStride, 0U);
        SetClippedValues(l0cOutUbSecond_, upHighMask_, rows, columns, rowStride, 0U);
        AscendC::PipeBarrier<PIPE_V>();
        SetClippedValues(l0cOutUbSecond_, upLowMask_, rows, columns, rowStride, 0U);
    }

    // H = SiLU(gate) * up, written to extraA_ (UB offset 2*kTileBytes).
    __aicore__ inline void ComputeSwigluRecompute(uint16_t mSize, uint16_t nSize, uint32_t rowStride)
    {
        constexpr uint16_t kElementsPerRepeat = 256U / sizeof(bfloat16_t);
        const uint16_t repeatsPerRow = static_cast<uint16_t>(
            Ops::Base::CeilDiv(static_cast<uint64_t>(nSize), static_cast<uint64_t>(kElementsPerRepeat)));
        constexpr float kOne = 1.0F;
        const float limit = params_.swigluLimit;
        // The disabled limit (FLT_MAX) makes the clamps numerical no-ops;
        // skip them entirely so the unlimited path pays nothing extra.
        const bool limitEnabled = limit != kMegaMoeSwigluLimitDisabled;
        __ubuf__ bfloat16_t* gate = (__ubuf__ bfloat16_t*)l0cOutUbFirst_.GetPhyAddr();
        __ubuf__ bfloat16_t* up = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();
        __ubuf__ bfloat16_t* h = (__ubuf__ bfloat16_t*)extraA_.GetPhyAddr();

        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < mSize; ++row) {
                uint32_t remaining = nSize;
                for (uint16_t repeat = 0; repeat < repeatsPerRow; ++repeat) {
                    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
                    AscendC::Reg::RegTensor<float> oneReg, limitReg, negLimitReg;
                    if (limitEnabled) {
                        AscendC::Reg::Duplicate(oneReg, kOne, mask);
                        AscendC::Reg::Muls(limitReg, oneReg, limit, mask);
                        AscendC::Reg::Muls(negLimitReg, oneReg, -limit, mask);
                    }
                    AscendC::Reg::RegTensor<bfloat16_t> gate0, gate1, up0, up1;
                    AscendC::Reg::RegTensor<float> gateFp0, gateFp1, upFp0, upFp1;
                    const uint32_t offset = row * rowStride + repeat * kElementsPerRepeat;
                    AscendC::Reg::DataCopy(gate0, gate + offset);
                    AscendC::Reg::DataCopy(up0, up + offset);
                    AscendC::Reg::Interleave(gate0, gate1, gate0, gate1);
                    AscendC::Reg::Interleave(up0, up1, up0, up1);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gateFp0, gate0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(gateFp1, gate1, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(upFp0, up0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kBackwardCastBF16ToFP32>(upFp1, up1, mask);

                    constexpr uint16_t kParts = sizeof(float) / sizeof(bfloat16_t);
                    constexpr uint16_t kPartSize = kElementsPerRepeat / kParts;
                    for (uint16_t part = 0; part < kParts; ++part) {
                        // Float lanes, not BF16 byte lanes, govern each 64-value part.
                        const uint32_t consumed = repeat * kElementsPerRepeat + part * kPartSize;
                        uint32_t partRemaining = consumed < nSize ? nSize - consumed : 0;
                        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(partRemaining);
                        auto gateFp = part == 0 ? gateFp0 : gateFp1;
                        auto upFp = part == 0 ? upFp0 : upFp1;
                        AscendC::Reg::RegTensor<float> oneFp, neg, exp, denom, sig, silu;
                        AscendC::Reg::RegTensor<float> siluRounded, result;
                        AscendC::Reg::RegTensor<bfloat16_t> siluBf16, output;
                        // H = SiLU(gate) * up; SiLU(gate) = gate * sigma(gate).
                        // Apply the dsv4 swiglu_limit clamp first so the
                        // recomputed H matches the forward epilogue.
                        // Skipped entirely when the limit is disabled.
                        if (limitEnabled) {
                            AscendC::Reg::Min(gateFp, gateFp, limitReg, mask);
                            AscendC::Reg::Max(upFp, upFp, negLimitReg, mask);
                            AscendC::Reg::Min(upFp, upFp, limitReg, mask);
                        }
                        AscendC::Reg::Duplicate(oneFp, kOne, mask);
                        AscendC::Reg::Muls(neg, gateFp, -kOne, mask);
                        AscendC::Reg::Exp(exp, neg, mask);
                        AscendC::Reg::Adds(denom, exp, kOne, mask);
                        AscendC::Reg::Div<float, &kBackwardDivMode>(sig, oneFp, denom, mask);
                        AscendC::Reg::Mul(silu, sig, gateFp, mask);
                        siluRounded = silu;  // Same FP32 fused activation as forward.
                        AscendC::Reg::Mul(result, siluRounded, upFp, mask);
                        AscendC::Reg::Cast<bfloat16_t, float, kBackwardCastFP32ToBF16>(output, result, mask);
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            h + offset + part * kPartSize, output, mask);
                    }
                }
            }
        }
    }

    AscendC::LocalTensor<bfloat16_t> l0cOutUbFirst_{AscendC::TPosition::VECIN, 0, kTileElements};
    AscendC::LocalTensor<bfloat16_t> l0cOutUbSecond_{AscendC::TPosition::VECIN, kTileBytes, kTileElements};
    AscendC::LocalTensor<bfloat16_t> extraA_{AscendC::TPosition::VECIN, 2 * kTileBytes, kTileElements};
    AscendC::LocalTensor<int8_t> mxData_{AscendC::TPosition::VECIN, kMxDataOffset, kMxDataElements};
    AscendC::LocalTensor<int8_t> mxScale_{AscendC::TPosition::VECIN, kMxScaleOffset, kMxScaleBytes};
    AscendC::LocalTensor<float> mxDecoded_{AscendC::TPosition::VECCALC, kMxDecodedOffset, kMxDecodedElements};
    AscendC::LocalTensor<int8_t> gateHighMask_{AscendC::TPosition::VECIN, kGateHighOffset, kClipMaskBytes};
    AscendC::LocalTensor<int8_t> upLowMask_{AscendC::TPosition::VECIN, kUpLowOffset, kClipMaskBytes};
    AscendC::LocalTensor<int8_t> upHighMask_{AscendC::TPosition::VECIN, kUpHighOffset, kClipMaskBytes};

    AscendC::LocalTensor<bfloat16_t> packedWeightBf16Ub_{AscendC::TPosition::VECIN, kWeightOffset, kTileRows};
    AscendC::LocalTensor<float> packedWeightFp32Ub_{AscendC::TPosition::VECIN, kWeightOffset, kTileRows};
    AscendC::GlobalTensor<bfloat16_t> gatePlaneGm_;
    AscendC::GlobalTensor<bfloat16_t> upPlaneGm_;
    AscendC::GlobalTensor<int8_t> gateMxFp8Gm_;
    AscendC::GlobalTensor<int8_t> upMxFp8Gm_;
    AscendC::GlobalTensor<int8_t> gateScaleGm_;
    AscendC::GlobalTensor<int8_t> upScaleGm_;
    AscendC::GlobalTensor<int8_t> gateHighGm_;
    AscendC::GlobalTensor<int8_t> upLowGm_;
    AscendC::GlobalTensor<int8_t> upHighGm_;
    AscendC::GlobalTensor<bfloat16_t> gradZGm_;
    AscendC::GlobalTensor<bfloat16_t> hpGm_;
    AscendC::GlobalTensor<bfloat16_t> dhpGm_;
    AscendC::GlobalTensor<bfloat16_t> packedTopkWeightsGm_;
    AscendC::GlobalTensor<float> packedTopkWeightsFp32Gm_;
    int64_t rowBase_{0};
    Params params_{};
    int64_t outputN_{0};
    int64_t outputF_{0};
};

}  // namespace MegaMoeTrainW8A8Vec

#endif  // DEEPEP_BLOCK_SWIGLU_BACKWARD_TRAIN_W8A8_H
