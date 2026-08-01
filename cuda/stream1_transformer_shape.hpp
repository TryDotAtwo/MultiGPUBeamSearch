#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace beam {

struct Stream1TransformerSequencePlan {
    std::uint32_t logical_seq_len;
    std::uint32_t padded_seq_len;
    std::uint32_t alignment;
};

inline Stream1TransformerSequencePlan make_stream1_transformer_sequence_plan(
    std::uint32_t logical_seq_len,
    std::uint32_t alignment) {
    if (logical_seq_len == 0U) {
        throw std::invalid_argument("logical_seq_len must be positive");
    }
    if (alignment == 0U) {
        throw std::invalid_argument("sequence alignment must be positive");
    }
    const std::uint64_t padded =
        ((static_cast<std::uint64_t>(logical_seq_len) + alignment - 1ULL) / alignment) * alignment;
    if (padded > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("padded_seq_len overflow");
    }
    return {logical_seq_len, static_cast<std::uint32_t>(padded), alignment};
}

} // namespace beam
