#pragma once

#include <cstring>
#include <stdexcept>

namespace beam {

enum class Stream1TransformerGemmFamily { Qkv, AttentionOut, Ff1, Ff2, Cls };
enum class Stream1TransformerGemmPolicy { Baseline, M64N128, M128N128, M128N128W64N32, M256N128, M64N64, M32N64 };
enum class Stream1TransformerGemmStagePolicy { Stages3, Stages2 };
enum class Stream1TransformerGemmSwizzlePolicy { Identity1, Identity2, Identity4, Identity8 };
enum class Stream1TransformerResidualEpiloguePolicy { Separate, FusedBiasRound };

inline Stream1TransformerResidualEpiloguePolicy parse_stream1_transformer_residual_epilogue_policy(
    const char* value) {
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "separate") == 0) {
        return Stream1TransformerResidualEpiloguePolicy::Separate;
    }
    if (std::strcmp(value, "fused") == 0) {
        return Stream1TransformerResidualEpiloguePolicy::FusedBiasRound;
    }
    throw std::invalid_argument("residual epilogue policy must be separate or fused");
}

struct Stream1TransformerGemmPolicyDesc {
    int threadblock_m;
    int threadblock_n;
    int threadblock_k;
    int warp_m;
    int warp_n;
    int warp_k;
    int stages;
};

inline Stream1TransformerGemmPolicyDesc stream1_transformer_gemm_policy_desc(Stream1TransformerGemmPolicy policy) {
    switch (policy) {
        case Stream1TransformerGemmPolicy::Baseline: return {128, 64, 32, 64, 32, 32, 3};
        case Stream1TransformerGemmPolicy::M64N128: return {64, 128, 32, 32, 64, 32, 3};
        case Stream1TransformerGemmPolicy::M128N128: return {128, 128, 32, 64, 64, 32, 3};
        case Stream1TransformerGemmPolicy::M128N128W64N32: return {128, 128, 32, 64, 32, 32, 3};
        case Stream1TransformerGemmPolicy::M256N128: return {256, 128, 32, 64, 64, 32, 3};
        case Stream1TransformerGemmPolicy::M64N64: return {64, 64, 32, 32, 32, 32, 3};
        case Stream1TransformerGemmPolicy::M32N64: return {32, 64, 32, 32, 32, 32, 3};
    }
    throw std::invalid_argument("invalid Stream1TransformerGemmPolicy value");
}

inline bool stream1_transformer_gemm_policy_allowed(Stream1TransformerGemmFamily family, Stream1TransformerGemmPolicy policy) {
    if (policy == Stream1TransformerGemmPolicy::Baseline) return true;
    switch (family) {
        case Stream1TransformerGemmFamily::Qkv:
            return policy == Stream1TransformerGemmPolicy::M128N128 ||
                policy == Stream1TransformerGemmPolicy::M256N128 ||
                policy == Stream1TransformerGemmPolicy::M64N128;
        case Stream1TransformerGemmFamily::AttentionOut:
            return policy == Stream1TransformerGemmPolicy::M128N128 ||
                policy == Stream1TransformerGemmPolicy::M64N64;
        case Stream1TransformerGemmFamily::Ff1:
            return policy == Stream1TransformerGemmPolicy::M128N128 ||
                policy == Stream1TransformerGemmPolicy::M128N128W64N32 ||
                policy == Stream1TransformerGemmPolicy::M64N128;
        case Stream1TransformerGemmFamily::Ff2:
            return policy == Stream1TransformerGemmPolicy::M128N128 ||
                policy == Stream1TransformerGemmPolicy::M64N64;
        case Stream1TransformerGemmFamily::Cls:
            return policy == Stream1TransformerGemmPolicy::M64N64 || policy == Stream1TransformerGemmPolicy::M32N64;
    }
    return false;
}


inline bool stream1_transformer_gemm_policy_supported_on_sm(
    Stream1TransformerGemmFamily family,
    Stream1TransformerGemmPolicy policy,
    int sm) {
    return sm >= 75 && stream1_transformer_gemm_policy_allowed(family, policy);
}
inline bool stream1_transformer_gemm_stage_supported_on_sm(
    Stream1TransformerGemmFamily family,
    Stream1TransformerGemmStagePolicy stage_policy,
    int sm) {
    if (sm < 75) return false;
    if (sm == 75 && family == Stream1TransformerGemmFamily::Ff1) {
        return stage_policy == Stream1TransformerGemmStagePolicy::Stages2;
    }
    return true;
}
inline Stream1TransformerGemmPolicy parse_stream1_transformer_gemm_policy(Stream1TransformerGemmFamily family, const char* value) {
    Stream1TransformerGemmPolicy policy = Stream1TransformerGemmPolicy::Baseline;
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "baseline") == 0) policy = Stream1TransformerGemmPolicy::Baseline;
    else if (std::strcmp(value, "m64n128") == 0) policy = Stream1TransformerGemmPolicy::M64N128;
    else if (std::strcmp(value, "m128n128") == 0) policy = Stream1TransformerGemmPolicy::M128N128;
    else if (std::strcmp(value, "m128n128w64n32") == 0) policy = Stream1TransformerGemmPolicy::M128N128W64N32;
    else if (std::strcmp(value, "m256n128") == 0) policy = Stream1TransformerGemmPolicy::M256N128;
    else if (std::strcmp(value, "m64n64") == 0) policy = Stream1TransformerGemmPolicy::M64N64;
    else if (std::strcmp(value, "m32n64") == 0) policy = Stream1TransformerGemmPolicy::M32N64;
    else throw std::invalid_argument("unknown Stream1 transformer GEMM policy");
    if (!stream1_transformer_gemm_policy_allowed(family, policy)) {
        throw std::invalid_argument("Stream1 transformer GEMM policy is not compiled for this family");
    }
    return policy;
}

inline Stream1TransformerGemmSwizzlePolicy parse_stream1_transformer_gemm_swizzle_policy(const char* value) {
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "1") == 0) {
        return Stream1TransformerGemmSwizzlePolicy::Identity1;
    }
    if (std::strcmp(value, "2") == 0) {
        return Stream1TransformerGemmSwizzlePolicy::Identity2;
    }
    if (std::strcmp(value, "4") == 0) {
        return Stream1TransformerGemmSwizzlePolicy::Identity4;
    }
    if (std::strcmp(value, "8") == 0) {
        return Stream1TransformerGemmSwizzlePolicy::Identity8;
    }
    throw std::invalid_argument("GEMM swizzle policy must be 1, 2, 4, or 8");
}
inline bool stream1_transformer_gemm_swizzle_allowed(
    Stream1TransformerGemmFamily family,
    Stream1TransformerGemmPolicy gemm_policy,
    Stream1TransformerGemmStagePolicy stage_policy,
    Stream1TransformerGemmSwizzlePolicy swizzle_policy) {

    if (swizzle_policy == Stream1TransformerGemmSwizzlePolicy::Identity1) {
        return true;
    }

    if (gemm_policy != Stream1TransformerGemmPolicy::M128N128 ||
        stage_policy != Stream1TransformerGemmStagePolicy::Stages3) {
        return false;
    }
    if (swizzle_policy == Stream1TransformerGemmSwizzlePolicy::Identity8 ||
        swizzle_policy == Stream1TransformerGemmSwizzlePolicy::Identity4) {
        return family == Stream1TransformerGemmFamily::Qkv ||
            family == Stream1TransformerGemmFamily::Ff1;
    }
    if (swizzle_policy == Stream1TransformerGemmSwizzlePolicy::Identity2) {
        return family == Stream1TransformerGemmFamily::AttentionOut ||
            family == Stream1TransformerGemmFamily::Ff2;
    }
    return false;
}
inline Stream1TransformerGemmStagePolicy parse_stream1_transformer_gemm_stage_policy(const char* value) {
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "3") == 0) {
        return Stream1TransformerGemmStagePolicy::Stages3;
    }
    if (std::strcmp(value, "2") == 0) {
        return Stream1TransformerGemmStagePolicy::Stages2;
    }
    throw std::invalid_argument("GEMM stage policy must be 2 or 3");
}

inline int stream1_transformer_gemm_stages(Stream1TransformerGemmStagePolicy policy) {
    switch (policy) {
    case Stream1TransformerGemmStagePolicy::Stages3: return 3;
    case Stream1TransformerGemmStagePolicy::Stages2: return 2;
    }
    throw std::invalid_argument("invalid Stream1TransformerGemmStagePolicy value");
}


enum class Stream1TransformerFf1Policy { Baseline, M64N128 };
inline Stream1TransformerFf1Policy parse_stream1_transformer_ff1_policy(const char* value) {
    const auto policy = parse_stream1_transformer_gemm_policy(Stream1TransformerGemmFamily::Ff1, value);
    if (policy == Stream1TransformerGemmPolicy::Baseline) return Stream1TransformerFf1Policy::Baseline;
    if (policy == Stream1TransformerGemmPolicy::M64N128) return Stream1TransformerFf1Policy::M64N128;
    throw std::invalid_argument("legacy FF1 parser only supports baseline and m64n128");
}
inline const char* stream1_transformer_ff1_policy_name(Stream1TransformerFf1Policy policy) {
    switch (policy) {
        case Stream1TransformerFf1Policy::Baseline: return "baseline";
        case Stream1TransformerFf1Policy::M64N128: return "m64n128";
    }
    throw std::invalid_argument("invalid Stream1TransformerFf1Policy value");
}
inline int stream1_transformer_ff1_threadblock_m(Stream1TransformerFf1Policy policy) {
    return policy == Stream1TransformerFf1Policy::M64N128 ? 64 : 128;
}
inline int stream1_transformer_ff1_threadblock_n(Stream1TransformerFf1Policy policy) {
    return policy == Stream1TransformerFf1Policy::M64N128 ? 128 : 64;
}

} // namespace beam
