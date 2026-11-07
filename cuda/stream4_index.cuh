#pragma once

#include "../src/types.hpp"

#include <cstdint>

#ifndef BEAM_EXPERIMENTAL_INDEX_SORT
#define BEAM_EXPERIMENTAL_INDEX_SORT 0
#endif

#if BEAM_EXPERIMENTAL_INDEX_SORT
#include "radix128.cuh"
#endif

namespace beam {

// The metadata stays immutable until the gather has completed on the same
// stream. CUB may also evaluate a reducer on inactive tile lanes, whose
// payloads are not guaranteed to equal our UINT32_MAX padding sentinel.
struct Stream4BestIndex {
    const CandidateMeta* metadata;
    const std::uint32_t* compact_count;
    std::uint32_t sort_items;

    __host__ __device__ std::uint32_t operator()(std::uint32_t a, std::uint32_t b) const {
        const std::uint32_t count = compact_count == nullptr ? 0U : *compact_count;
        const std::uint32_t limit = count < sort_items ? count : sort_items;
        if (a >= limit) return b < limit ? b : UINT32_MAX;
        if (b >= limit) return a;
        const CandidateMeta x = metadata[a];
        const CandidateMeta y = metadata[b];
        if (x.score_key != y.score_key) return x.score_key < y.score_key ? a : b;
        if (x.parent_idx != y.parent_idx) return x.parent_idx < y.parent_idx ? a : b;
        return x.route_packed <= y.route_packed ? a : b;
    }
};

} // namespace beam
