#pragma once

#include "../src/types.hpp"

#include <cstdint>

namespace beam::detail {

struct Stream4HashKeys {
    const Hash128* keys;
    BEAM_HOST_DEVICE Hash128 operator()(std::uint32_t i) const { return keys[i]; }
};

struct Stream4CandidateKeys {
    const CandidateMeta* candidates;
    BEAM_HOST_DEVICE Hash128 operator()(std::uint32_t i) const { return candidates[i].hash; }
};

BEAM_HOST_DEVICE inline bool stream4_hash_less(Hash128 a, Hash128 b) {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}

// Number of elements from the first range in the stable merged prefix of
// length diagonal. Equal full 128-bit hashes from the first range come first.
// The caller supplies sorted ranges and diagonal <= left_count + right_count.
template <typename Left, typename Right>
BEAM_HOST_DEVICE std::uint32_t stream4_merge_partition(
    Left a, std::uint32_t left_count, Right b, std::uint32_t right_count,
    std::uint32_t diagonal) {
    std::uint32_t lower = diagonal > right_count ? diagonal - right_count : 0U;
    std::uint32_t upper = diagonal < left_count ? diagonal : left_count;
    while (lower < upper) {
        const std::uint32_t left = lower + (upper - lower) / 2U;
        const std::uint32_t right = diagonal - left;
        if (right > 0U && left < left_count && !stream4_hash_less(b(right - 1U), a(left)))
            lower = left + 1U;
        else
            upper = left;
    }
    return lower;
}

} // namespace beam::detail
