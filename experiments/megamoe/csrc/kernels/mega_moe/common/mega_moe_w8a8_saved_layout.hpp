#pragma once
#include <cstdint>

namespace MegaMoeTrainW8A8 {
// Frozen W8A8 saved-state and UB contract. No BF16 operator dependency.
constexpr uint32_t kSaveNone = 1, kSaveMxFp8 = 3;
constexpr uint32_t BACKWARD_DP_PARTIAL_TILE_N = 256U;
constexpr uint32_t SWIGLU_N_HALF = 2U;
constexpr float kMegaMoeSwigluLimitDisabled = 3.402823466e+38F;
constexpr uint32_t kBf16Bytes = 2;
template <uint32_t RawRows, uint32_t ChunkRows = RawRows>
struct SavedGateUpQuantUbLayoutFor {
    static constexpr uint32_t kRows = ChunkRows;
    static constexpr uint32_t kColumns = 256;
    static constexpr uint32_t kElements = kRows * kColumns;
    static constexpr uint32_t kGroups = kElements / 32;
    static constexpr uint32_t gateData = 2 * RawRows * kColumns * kBf16Bytes;
    static constexpr uint32_t upData = gateData + kElements;
    static constexpr uint32_t compactScale = upData + kElements;
    static constexpr uint32_t gateScale = compactScale + kGroups * 2;
    static constexpr uint32_t upScale = gateScale + kRows * 32;
    static constexpr uint32_t maxExp = upScale + kRows * 32;
    static constexpr uint32_t halfScale = maxExp + kGroups * 2;
    static constexpr uint32_t totalBytes = halfScale + kGroups * 2;
    static constexpr uint32_t maskPlaneBytes = kRows * 32;
    static constexpr uint32_t gateHigh = totalBytes;
    static constexpr uint32_t upLow = gateHigh + maskPlaneBytes;
    static constexpr uint32_t upHigh = upLow + maskPlaneBytes;
    static constexpr uint32_t totalBytesWithClipMasks = upHigh + maskPlaneBytes;
};

using SavedGateUpQuantUbLayout = SavedGateUpQuantUbLayoutFor<128>;

// Hp occupies the third 64 KiB UB tile. The fused gradient path also keeps
// dUp in the second tile, so it transposes Hp above all three work tiles.
template <uint32_t TransposeOffset = 65536>
struct HpTokenQuantUbLayoutFor {
    static constexpr uint32_t transpose = TransposeOffset;
    static constexpr uint32_t kRows = 128, kColumns = 64;
    static constexpr uint32_t kElements = kRows * kColumns;
    static constexpr uint32_t kGroups = kElements / 32;
    static constexpr uint32_t gateData = transpose + kElements * 2;
    static constexpr uint32_t upData = gateData + kElements;
    static constexpr uint32_t compactScale = upData + kElements;
    static constexpr uint32_t gateScale = compactScale + kGroups * 2;
    static constexpr uint32_t upScale = gateScale + kRows * 32;
    static constexpr uint32_t maxExp = upScale + kRows * 32;
    static constexpr uint32_t halfScale = maxExp + kGroups * 2;
    static constexpr uint32_t totalBytes = halfScale + kGroups * 2;
    static constexpr uint32_t maskPlaneBytes = kRows * 32;
    static constexpr uint32_t gateHigh = totalBytes;
    static constexpr uint32_t upLow = gateHigh + maskPlaneBytes;
    static constexpr uint32_t upHigh = upLow + maskPlaneBytes;
    static constexpr uint32_t totalBytesWithClipMasks = upHigh + maskPlaneBytes;
};
using HpTokenQuantUbLayout = HpTokenQuantUbLayoutFor<>;
static_assert(HpTokenQuantUbLayout::totalBytesWithClipMasks <= 131072,
              "Hp transpose/quant scratch must not overwrite live dHp or Hp");
using HpTokenQuantPreserveGradUbLayout = HpTokenQuantUbLayoutFor<196608>;

}  // namespace MegaMoeTrainW8A8
