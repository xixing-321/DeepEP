// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

#pragma once
#include "csrc/kernels/common/mega_moe_wave_math.hpp"

#if defined(__NPU_DEVICE__) || defined(__ASC_NPU_HOST__)
#define MEGAMOE_BWD_WAVE_HD __host_aicore__
#else
#define MEGAMOE_BWD_WAVE_HD
#endif

namespace MegaMoeBF16Training {
// Split a canonical expert-local interval at source-rank boundaries. The
// destination is the original row index even when a wave visits this tile early.
template <class Prefix, class Copy>
MEGAMOE_BWD_WAVE_HD inline void VisitCanonicalPeerRows(const Prefix& prefix, uint32_t sources, uint32_t experts,
                                                       uint32_t expert, uint32_t row, uint32_t end, Copy copy)
{
    uint32_t lo = 0, hi = sources;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (prefix.Get(mid * experts + expert) <= row)
            lo = mid + 1;
        else
            hi = mid;
    }
    uint32_t begin = lo ? prefix.Get((lo - 1) * experts + expert) : 0;
    for (uint32_t peer = lo; row < end && peer < sources; ++peer) {
        const uint32_t next = prefix.Get(peer * experts + expert);
        const uint32_t stop = next < end ? next : end;
        if (row < stop) {
            copy(peer, row - begin, row, stop - row);
            row = stop;
        }
        begin = next;
    }
}

// Waves count complete expert-local M tiles. A short expert can finish in
// the same logical wave in which the next expert starts, with separate flags.
struct BackwardWavePlan {
    static constexpr uint32_t tileM = 128;
    uint32_t rows, tiles, groups, pivot, firstTile, cores;

    MEGAMOE_BWD_WAVE_HD constexpr BackwardWavePlan(uint32_t m, uint32_t n, uint32_t workers, uint32_t sourceBegin,
                                                   uint32_t precedingTiles)
        : rows(m),
          tiles(m / tileM + (m % tileM != 0)),
          groups(MegaMoeImpl::WaveGroups(workers, (uint64_t(n) + 255U) / 256U, 0, 2, false)),
          pivot(sourceBegin < m ? sourceBegin / tileM : 0),
          firstTile(precedingTiles),
          cores(workers)
    {
    }

    MEGAMOE_BWD_WAVE_HD inline uint32_t SliceTiles(uint32_t begin) const
    {
        const uint32_t budget = groups - (firstTile + begin) % groups;
        return tiles - begin < budget ? tiles - begin : budget;
    }

    // Dispatch, dHp and SwiGLUGrad share the rotated visit order. The row
    // argument is the start of the canonical tile, before AIV subdivision.
    MEGAMOE_BWD_WAVE_HD inline uint32_t WaveId(uint32_t row) const
    {
        const uint32_t tile = row / tileM;
        const uint32_t ordinal = tile >= pivot ? tile - pivot : tiles - pivot + tile;
        return (firstTile + ordinal) / groups;
    }

    // Dispatch and SwiGLUGrad partition the canonical rows between AIVs.
    template <class Visit>
    MEGAMOE_BWD_WAVE_HD inline void ForEach(uint32_t lane, Visit visit) const
    {
        for (uint32_t begin = 0; begin < tiles;) {
            const uint32_t count = SliceTiles(begin);
            const uint32_t teams = count < cores ? count : cores;
            const uint32_t team = lane % teams;
            const uint32_t owners = 1U + (cores - 1U - team) / teams;
            for (uint32_t slot = team; slot < count; slot += teams) {
                const uint32_t tile = (pivot + begin + slot) % tiles;
                const uint32_t row = tile * tileM;
                visit(row, rows - row < tileM ? rows - row : tileM, lane / teams, owners);
            }
            begin += count;
        }
    }

    // AICs rotate across all M/N tiles as in forward. Visit each owned M row
    // block once so Fixpipe completion can be published once per writer.
    template <class Visit>
    MEGAMOE_BWD_WAVE_HD inline void ForEachDhpRow(uint32_t lane, uint32_t columns, uint32_t startBlock,
                                                  Visit visit) const
    {
        for (uint32_t begin = 0; begin < tiles;) {
            const uint32_t count = SliceTiles(begin);
            const uint32_t teams = count < cores ? count : cores;
            const uint32_t tasks = count * columns;
            const uint32_t dmaBase = cores / teams, dmaExtra = cores % teams;
            uint32_t task = MegaMoeImpl::GetGmmFirstOwnedTile(lane, cores, startBlock);
            while (task < tasks) {
                const uint32_t slot = task / columns;
                const uint32_t firstCol = task % columns;
                const uint32_t row = ((pivot + begin + slot) % tiles) * tileM;
                const uint32_t extent = rows - row < tileM ? rows - row : tileM;
                const uint32_t dmaOwners = dmaBase + (slot < dmaExtra);
                visit(row, extent, firstCol, extent < dmaOwners ? extent : dmaOwners);
                do {
                    task += cores;
                } while (task < (slot + 1U) * columns);
            }
            startBlock = MegaMoeImpl::GetGmmNextStartBlock(startBlock, tasks, cores);
            begin += count;
        }
    }
};
}  // namespace MegaMoeBF16Training
#undef MEGAMOE_BWD_WAVE_HD
