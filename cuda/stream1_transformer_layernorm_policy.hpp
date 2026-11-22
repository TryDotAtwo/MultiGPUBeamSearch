#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifndef STREAM1_TRANSFORMER_LN256_SPLIT_SLOTS
#define STREAM1_TRANSFORMER_LN256_SPLIT_SLOTS 0
#endif

namespace beam {

enum class Stream1TransformerLayerNormRowsPolicy { RowPerBlock, TwoRowsPerBlock, PersistentRows };
inline Stream1TransformerLayerNormRowsPolicy parse_stream1_transformer_layernorm_rows_policy(const char* value) {
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "row") == 0) {
        return Stream1TransformerLayerNormRowsPolicy::RowPerBlock;
    }

    if (std::strcmp(value, "block2") == 0) {
        return Stream1TransformerLayerNormRowsPolicy::TwoRowsPerBlock;
    }
    if (std::strcmp(value, "persistent") == 0) {
        return Stream1TransformerLayerNormRowsPolicy::PersistentRows;
    }
    throw std::invalid_argument("LayerNorm rows policy must be row, block2, or persistent");
}

inline bool stream1_transformer_layernorm_copy_policy_supported(
    Stream1TransformerLayerNormRowsPolicy policy) {
    return policy == Stream1TransformerLayerNormRowsPolicy::RowPerBlock ||
        policy == Stream1TransformerLayerNormRowsPolicy::PersistentRows;
}
inline int stream1_transformer_layernorm_persistent_blocks_per_sm(
    const char* value,
    int maximum_blocks_per_sm) {
    if (maximum_blocks_per_sm <= 0) {
        throw std::invalid_argument("LayerNorm maximum persistent blocks per SM must be positive");
    }
    if (value == nullptr || value[0] == '\0') {
        return maximum_blocks_per_sm;
    }
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < 1L || parsed > maximum_blocks_per_sm) {
        throw std::invalid_argument("LayerNorm persistent blocks per SM must be within the occupancy limit");
    }
    return static_cast<int>(parsed);
}

inline constexpr std::uint32_t STREAM1_TRANSFORMER_LN256_WARP_SLOTS = 4U;
inline constexpr bool STREAM1_TRANSFORMER_LN256_SPLIT_SLOTS_ENABLED =
    STREAM1_TRANSFORMER_LN256_SPLIT_SLOTS != 0;
#if STREAM1_TRANSFORMER_LN256_SPLIT_SLOTS
inline constexpr std::uint32_t STREAM1_TRANSFORMER_LN256_MEAN_SLOT = 4U;
inline constexpr std::uint32_t STREAM1_TRANSFORMER_LN256_INV_STD_SLOT = 5U;
inline constexpr std::uint32_t STREAM1_TRANSFORMER_LN256_SHARED_FLOATS = 6U;
#else
inline constexpr std::uint32_t STREAM1_TRANSFORMER_LN256_MEAN_SLOT = 0U;
inline constexpr std::uint32_t STREAM1_TRANSFORMER_LN256_INV_STD_SLOT = 0U;
inline constexpr std::uint32_t STREAM1_TRANSFORMER_LN256_SHARED_FLOATS = 4U;
#endif

} // namespace beam
