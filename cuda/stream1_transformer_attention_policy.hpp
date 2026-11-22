#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace beam {

enum class Stream1TransformerAttentionTilePolicy : std::uint32_t { Q64K64, Q32K64, Q64K64V4 };

enum class Stream1TransformerAttentionMaxKPolicy : std::uint32_t { Padded64, Exact32 };

enum class Stream1TransformerClsAttentionPolicy : std::uint32_t { Cutlass, Q32K64 };

struct Stream1TransformerAttentionTileDesc {
    std::uint32_t queries;
    std::uint32_t keys;
    std::uint32_t warps;
    std::uint32_t alignment_elements;
};

inline Stream1TransformerAttentionTilePolicy parse_stream1_transformer_attention_tile_policy(const char* value) {
    if (value == nullptr || value[0] == '\0' || std::string(value) == "q64k64") {
        return Stream1TransformerAttentionTilePolicy::Q64K64;
    }
    if (std::string(value) == "q32k64") {
        return Stream1TransformerAttentionTilePolicy::Q32K64;
    }
    if (std::string(value) == "q64k64v4") {
        return Stream1TransformerAttentionTilePolicy::Q64K64V4;
    }
    throw std::invalid_argument("attention tile policy must be q64k64, q32k64, or q64k64v4");
}

inline Stream1TransformerAttentionTileDesc stream1_transformer_attention_tile_desc(
    Stream1TransformerAttentionTilePolicy policy) {
    switch (policy) {
    case Stream1TransformerAttentionTilePolicy::Q64K64: return {64U, 64U, 4U, 8U};
    case Stream1TransformerAttentionTilePolicy::Q32K64: return {32U, 64U, 2U, 8U};
    case Stream1TransformerAttentionTilePolicy::Q64K64V4: return {64U, 64U, 4U, 4U};
    }
    throw std::invalid_argument("unknown attention tile policy enum");
}

inline Stream1TransformerAttentionMaxKPolicy parse_stream1_transformer_attention_max_k_policy(const char* value) {
    if (value == nullptr || value[0] == '\0' || std::string(value) == "padded64") {
        return Stream1TransformerAttentionMaxKPolicy::Padded64;
    }
    if (std::string(value) == "exact32") {
        return Stream1TransformerAttentionMaxKPolicy::Exact32;
    }
    throw std::invalid_argument("attention max-k policy must be padded64 or exact32");
}

inline std::uint32_t stream1_transformer_attention_max_k(Stream1TransformerAttentionMaxKPolicy policy) {
    switch (policy) {
    case Stream1TransformerAttentionMaxKPolicy::Padded64: return 64U;
    case Stream1TransformerAttentionMaxKPolicy::Exact32: return 32U;
    }
    throw std::invalid_argument("unknown attention max-k policy enum");
}

inline Stream1TransformerClsAttentionPolicy parse_stream1_transformer_cls_attention_policy(const char* value) {
    if (value == nullptr || value[0] == '\0' || std::string(value) == "cutlass") {
        return Stream1TransformerClsAttentionPolicy::Cutlass;
    }
    if (std::string(value) == "q32k64") {
        return Stream1TransformerClsAttentionPolicy::Q32K64;
    }

    throw std::invalid_argument("CLS attention policy must be cutlass or q32k64");
}
} // namespace beam