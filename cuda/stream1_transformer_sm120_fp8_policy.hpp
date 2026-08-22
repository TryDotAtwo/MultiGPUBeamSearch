#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

inline constexpr std::uint32_t stream1_sm120_fp8_qkv_bit(std::uint32_t layer) {
    return 1U << (layer * 4U);
}

inline constexpr std::uint32_t stream1_sm120_fp8_ff1_bit(std::uint32_t layer) {
    return 1U << (layer * 4U + 2U);
}

inline std::uint32_t parse_stream1_sm120_fp8_operator_policy(const char* raw) {
    if (raw == nullptr || *raw == '\0') {
        return 0U;
    }
    std::uint32_t mask = 0U;
    std::string_view remaining(raw);
    while (!remaining.empty()) {
        const std::size_t comma = remaining.find(',');
        const std::string_view token = remaining.substr(0, comma);
        bool matched = false;
        for (std::uint32_t layer = 0; layer < 4U; ++layer) {
            const std::string prefix = "blocks." + std::to_string(layer) + ".";
            if (token == prefix + "attn.in_proj_weight") {
                mask |= stream1_sm120_fp8_qkv_bit(layer);
                matched = true;
            } else if (token == prefix + "ff.0.weight") {
                mask |= stream1_sm120_fp8_ff1_bit(layer);
                matched = true;
            }
        }
        if (!matched) {
            throw std::invalid_argument(
                "BEAM_STREAM1_SM120_FP8_OPERATORS contains unsupported operator: " +
                std::string(token));
        }
        if (comma == std::string_view::npos) {
            break;
        }
        remaining.remove_prefix(comma + 1U);
        if (remaining.empty()) {
            throw std::invalid_argument("BEAM_STREAM1_SM120_FP8_OPERATORS has an empty token");
        }
    }
    return mask;
}
