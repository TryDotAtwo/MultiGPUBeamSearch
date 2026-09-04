#pragma once

#include <cstring>
#include <stdexcept>

namespace beam {

enum class Stream1TransformerHopperMode {
    Off,
    Fp16Tma,
    Fp8E4m3,
};

inline Stream1TransformerHopperMode parse_stream1_transformer_hopper_mode(const char* value) {
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "off") == 0) {
        return Stream1TransformerHopperMode::Off;
    }
    if (std::strcmp(value, "fp16_tma") == 0) {
        return Stream1TransformerHopperMode::Fp16Tma;
    }
    if (std::strcmp(value, "fp8_e4m3") == 0) {
        return Stream1TransformerHopperMode::Fp8E4m3;
    }
    throw std::invalid_argument(
        "BEAM_STREAM1_TRANSFORMER_HOPPER must be off, fp16_tma, or fp8_e4m3");
}

inline Stream1TransformerHopperMode select_stream1_transformer_hopper_mode(
    const char* global_value,
    const char* family_override) {
    return parse_stream1_transformer_hopper_mode(
        family_override != nullptr && family_override[0] != '\0' ? family_override : global_value);
}

inline bool stream1_transformer_hopper_large_gemm_allowed(
    Stream1TransformerHopperMode mode,
    int sm,
    unsigned rows) {
    // Hopper warp-specialized TMA/GMMA is architecture-specific.  Small CLS
    // GEMMs are deliberately excluded because launch overhead dominates them.
    return mode != Stream1TransformerHopperMode::Off && sm == 90 && rows >= 4096U;
}

inline bool stream1_transformer_hopper_uses_packed_full_token_weight(
    Stream1TransformerHopperMode mode,
    unsigned layer,
    unsigned layer_count) {
    // The last block is intentionally kept in the legacy layout: the
    // final-CLS path executes only 1/51 of its rows and does not use GMMA.
    return mode != Stream1TransformerHopperMode::Off &&
        layer_count > 0U && layer + 1U < layer_count;
}

}  // namespace beam
