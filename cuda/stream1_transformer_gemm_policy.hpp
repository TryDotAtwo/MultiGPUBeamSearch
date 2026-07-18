#pragma once

#include <cstring>
#include <stdexcept>

namespace beam {

enum class Stream1TransformerFf1Policy {
    Baseline,
    M64N128,
    M64N64,
};

inline Stream1TransformerFf1Policy parse_stream1_transformer_ff1_policy(const char* value) {
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "baseline") == 0) {
        return Stream1TransformerFf1Policy::Baseline;
    }
    if (std::strcmp(value, "m64n128") == 0) {
        return Stream1TransformerFf1Policy::M64N128;
    }
    if (std::strcmp(value, "m64n64") == 0) {
        return Stream1TransformerFf1Policy::M64N64;
    }
    throw std::invalid_argument("unknown BEAM_STREAM1_TRANSFORMER_FF1_POLICY");
}

inline const char* stream1_transformer_ff1_policy_name(Stream1TransformerFf1Policy policy) {
    switch (policy) {
        case Stream1TransformerFf1Policy::Baseline:
            return "baseline";
        case Stream1TransformerFf1Policy::M64N128:
            return "m64n128";
        case Stream1TransformerFf1Policy::M64N64:
            return "m64n64";
    }
    throw std::invalid_argument("invalid Stream1TransformerFf1Policy value");
}

} // namespace beam
