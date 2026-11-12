// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

#pragma once

#include <cstdint>
#if defined(__NPU_DEVICE__) || defined(__ASC_NPU_HOST__)
#include "kernel_operator.h"
#define MEGA_MOE_WAVE_HOST_DEVICE __host_aicore__
#else
#define MEGA_MOE_WAVE_HOST_DEVICE
#endif

namespace MegaMoeBF16Training::Wave {

constexpr uint32_t kGroupRows = 128;
constexpr uint32_t kDispatchUbBase = 128 * 1024;
constexpr uint32_t kWeightUbBytes = 4096;
constexpr uint32_t kMetadataUbBytes = 4096;
constexpr uint32_t kMetadataUbBase = 248 * 1024 - kMetadataUbBytes;

MEGA_MOE_WAVE_HOST_DEVICE constexpr uint32_t Gcd(uint32_t a, uint32_t b)
{
    while (b) {
        const uint32_t next = a % b;
        a = b;
        b = next;
    }
    return a;
}

MEGA_MOE_WAVE_HOST_DEVICE constexpr uint32_t GroupsPerWave(uint32_t cores, uint32_t f, uint32_t h)
{
    const uint32_t gateUpPairs = (f + 255U) / 256U;
    const uint32_t downTasks = (h + 255U) / 256U;
    if (!cores || !gateUpPairs || !downTasks) return 1;
    // A Gate/Up pair is one scheduler job containing two MMAD calls.
    // Round the work budget to complete core rounds in both projections.
    const uint32_t first = (2U * cores + gateUpPairs - 1U) / gateUpPairs;
    const uint32_t second = (cores + downTasks - 1U) / downTasks;
    const uint32_t minimum = first > second ? first : second;
    const uint32_t quantum = cores / Gcd(Gcd(cores, gateUpPairs), downTasks);
    return (minimum + quantum - 1U) / quantum * quantum;
}

MEGA_MOE_WAVE_HOST_DEVICE constexpr bool ScratchFits(uint32_t hidden)
{
    return kDispatchUbBase + 4ULL * ((hidden + 31U) / 32U * 32U) + kWeightUbBytes + kMetadataUbBytes <=
           248ULL * 1024ULL;
}

MEGA_MOE_WAVE_HOST_DEVICE constexpr bool MetadataFits(uint32_t world, uint32_t experts)
{
    return 2ULL * ((static_cast<uint64_t>(world) * experts * 4U + 31U) / 32U * 32U) <= kMetadataUbBytes;
}

MEGA_MOE_WAVE_HOST_DEVICE constexpr uint64_t ReadySlot(uint64_t expertBase, uint32_t expert, uint32_t row)
{
    return expertBase / kGroupRows + expert + row / kGroupRows;
}

struct Cursor {
    uint32_t expert = 0;
    uint32_t row = 0;
    uint32_t slice = 0;
    uint64_t packed = 0;
};

struct Slice {
    uint32_t expert;
    uint32_t row;
    uint32_t rows;
    uint32_t ordinal;
    uint32_t fullRows;
    uint64_t packed;
};

struct Range {
    Cursor begin;
    Cursor end;
    bool valid = false;
};

// Counts returns each expert's non-negative row count. Capacity clipping uses
// the same packed expert order as the ordinary forward and saved-state layout.
template <class Counts>
MEGA_MOE_WAVE_HOST_DEVICE inline bool Next(Cursor& cursor, uint32_t& groups, uint32_t experts, uint64_t capacity,
                                           const Counts& counts, Slice& slice)
{
    while (groups && cursor.expert < experts) {
        const uint64_t base = cursor.packed - cursor.row;
        const uint64_t available = base < capacity ? capacity - base : 0;
        const uint64_t raw = counts(cursor.expert);
        const uint32_t rows = static_cast<uint32_t>(raw < available ? raw : available);
        if (cursor.row >= rows) {
            ++cursor.expert;
            cursor.row = cursor.slice = 0;
            continue;
        }
        const uint64_t budget = static_cast<uint64_t>(groups) * kGroupRows;
        const uint32_t left = rows - cursor.row;
        const uint32_t take = left < budget ? left : static_cast<uint32_t>(budget);
        slice = {cursor.expert, cursor.row, take, ++cursor.slice, rows, cursor.packed};
        cursor.row += take;
        cursor.packed += take;
        groups -= (take + kGroupRows - 1U) / kGroupRows;
        if (cursor.row == rows) {
            ++cursor.expert;
            cursor.row = cursor.slice = 0;
        }
        return true;
    }
    return false;
}

template <class Counts>
MEGA_MOE_WAVE_HOST_DEVICE inline Range Plan(const Cursor& begin, uint32_t groups, uint32_t experts, uint64_t capacity,
                                            const Counts& counts)
{
    Range range{begin, begin, false};
    Slice slice{};
    while (Next(range.end, groups, experts, capacity, counts, slice)) range.valid = true;
    return range;
}

}  // namespace MegaMoeBF16Training::Wave

#undef MEGA_MOE_WAVE_HOST_DEVICE
