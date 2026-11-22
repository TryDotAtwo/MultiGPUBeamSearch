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

struct Stream1TransformerPaddingTailPlan {
    std::uint32_t row_count;
    std::uint32_t tail_tokens_per_row;
    std::uint64_t tail_elements;
    std::uint64_t row_stride_elements;
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

inline bool stream1_transformer_supports_generic_final_cls_only(
    std::uint32_t logical_seq_len,
    std::uint32_t padded_seq_len,
    std::uint32_t d_model,
    std::uint32_t nhead,
    std::uint32_t head_dim,
    std::uint32_t transformer_layers,
    std::uint32_t ff_dim,
    std::uint32_t output_dim) {
    return logical_seq_len > 0U &&
        logical_seq_len <= padded_seq_len &&
        padded_seq_len <= 64U &&
        d_model == 256U &&
        nhead == 8U &&
        head_dim == 32U &&
        d_model == nhead * head_dim &&
        transformer_layers == 4U &&
        ff_dim == 1024U &&
        output_dim == 24U;
}

inline Stream1TransformerPaddingTailPlan make_stream1_transformer_padding_tail_plan(
    std::uint32_t row_count,
    std::uint32_t logical_seq_len,
    std::uint32_t padded_seq_len,
    std::uint32_t d_model) {
    if (logical_seq_len > padded_seq_len) {
        throw std::invalid_argument("logical_seq_len must not exceed padded_seq_len");
    }
    if (d_model == 0U) {
        throw std::invalid_argument("d_model must be positive");
    }
    const std::uint32_t tail_tokens = padded_seq_len - logical_seq_len;
    const std::uint64_t row_stride = static_cast<std::uint64_t>(padded_seq_len) * d_model;
    const std::uint64_t tail_elements =
        static_cast<std::uint64_t>(row_count) * tail_tokens * d_model;
    return {row_count, tail_tokens, tail_elements, row_stride};
}

} // namespace beam
