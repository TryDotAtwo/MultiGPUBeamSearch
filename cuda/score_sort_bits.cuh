#pragma once

#include "config.hpp"

#include <cstdint>

namespace beam::detail {

// Number of low bits needed to represent every value through max_value.
// Zero still uses one bit, and UINT32_MAX never causes an overflowing shift.
constexpr std::uint32_t score_key_bits(std::uint32_t max_value) {
    std::uint32_t bits = 1;
    while (max_value >>= 1U) ++bits;
    return bits;
}

// Valid histogram scores are [0, SCORE_BIN_COUNT); SCORE_BIN_COUNT is the
// sorted sentinel and must remain distinct from the maximum valid score.
inline constexpr std::uint32_t SCORE_HISTOGRAM_KEY_BITS = score_key_bits(SCORE_BIN_COUNT);
static_assert((std::uint64_t{1} << SCORE_HISTOGRAM_KEY_BITS) > SCORE_BIN_COUNT);

#if BEAM_EXPERIMENTAL_SCORE_BITS
inline constexpr int SCORE_HISTOGRAM_SORT_END_BIT = static_cast<int>(SCORE_HISTOGRAM_KEY_BITS);
#else
inline constexpr int SCORE_HISTOGRAM_SORT_END_BIT = 32;
#endif

} // namespace beam::detail
