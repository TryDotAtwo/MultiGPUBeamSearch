#include "stream1_transformer_gemm_policy.hpp"

#include <iostream>
#include <stdexcept>
#include <string_view>

using namespace beam;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_rejects(Stream1TransformerGemmFamily family, const char* value) {
    bool rejected = false;
    try {
        static_cast<void>(parse_stream1_transformer_gemm_policy(family, value));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "unknown family policy must fail closed");
}
} // namespace

int main() {
    using Family = Stream1TransformerGemmFamily;
    using Policy = Stream1TransformerGemmPolicy;

    require(parse_stream1_transformer_gemm_swizzle_policy(nullptr) == Stream1TransformerGemmSwizzlePolicy::Identity1,
            "unset GEMM swizzle policy must select identity1");
    require(parse_stream1_transformer_gemm_swizzle_policy("") == Stream1TransformerGemmSwizzlePolicy::Identity1,
            "empty GEMM swizzle policy must select identity1");
    require(parse_stream1_transformer_gemm_swizzle_policy("1") == Stream1TransformerGemmSwizzlePolicy::Identity1,
            "GEMM identity1 swizzle parse failed");
    require(parse_stream1_transformer_gemm_swizzle_policy("2") == Stream1TransformerGemmSwizzlePolicy::Identity2,
            "GEMM identity2 swizzle parse failed");
    require(parse_stream1_transformer_gemm_swizzle_policy("4") == Stream1TransformerGemmSwizzlePolicy::Identity4,
            "GEMM identity4 swizzle parse failed");
    require(parse_stream1_transformer_gemm_swizzle_policy("8") == Stream1TransformerGemmSwizzlePolicy::Identity8,
            "GEMM identity8 swizzle parse failed");
    bool swizzle_rejected = false;
    try { static_cast<void>(parse_stream1_transformer_gemm_swizzle_policy("16")); }
    catch (const std::invalid_argument&) { swizzle_rejected = true; }
    require(swizzle_rejected, "unknown GEMM swizzle policy must fail closed");
    require(stream1_transformer_gemm_swizzle_allowed(
                Family::Qkv, Policy::M128N128, Stream1TransformerGemmStagePolicy::Stages3,
                Stream1TransformerGemmSwizzlePolicy::Identity8),
            "QKV m128n128 stages3 must allow identity8 swizzle");
    require(stream1_transformer_gemm_swizzle_allowed(
                Family::Qkv, Policy::M128N128, Stream1TransformerGemmStagePolicy::Stages3,
                Stream1TransformerGemmSwizzlePolicy::Identity4),
            "QKV m128n128 stages3 must allow identity4 swizzle");
    require(stream1_transformer_gemm_swizzle_allowed(
                Family::Ff1, Policy::M128N128, Stream1TransformerGemmStagePolicy::Stages3,
                Stream1TransformerGemmSwizzlePolicy::Identity4),
            "FF1 m128n128 stages3 must allow identity4 swizzle");
    require(stream1_transformer_gemm_swizzle_allowed(
                Family::Ff1, Policy::M128N128, Stream1TransformerGemmStagePolicy::Stages3,
                Stream1TransformerGemmSwizzlePolicy::Identity8),
            "FF1 m128n128 stages3 must allow identity8 swizzle");


    require(stream1_transformer_gemm_swizzle_allowed(
                Family::AttentionOut, Policy::M128N128, Stream1TransformerGemmStagePolicy::Stages3,
                Stream1TransformerGemmSwizzlePolicy::Identity2),
            "attention-out m128n128 stages3 must allow identity2 swizzle");
    require(stream1_transformer_gemm_swizzle_allowed(
                Family::Ff2, Policy::M128N128, Stream1TransformerGemmStagePolicy::Stages3,
                Stream1TransformerGemmSwizzlePolicy::Identity2),
            "FF2 m128n128 stages3 must allow identity2 swizzle");

    require(!stream1_transformer_gemm_swizzle_allowed(
                Family::AttentionOut, Policy::M128N128, Stream1TransformerGemmStagePolicy::Stages3,
                Stream1TransformerGemmSwizzlePolicy::Identity8),
            "attention-out must reject uncompiled identity8 swizzle");
    require(!stream1_transformer_gemm_swizzle_allowed(
                Family::Ff1, Policy::M128N128, Stream1TransformerGemmStagePolicy::Stages2,
                Stream1TransformerGemmSwizzlePolicy::Identity8),
            "FF1 stages2 must reject uncompiled identity8 swizzle");
    require(stream1_transformer_gemm_swizzle_allowed(
                Family::AttentionOut, Policy::Baseline, Stream1TransformerGemmStagePolicy::Stages3,
                Stream1TransformerGemmSwizzlePolicy::Identity1),
            "identity1 swizzle must remain universally available");
    require(parse_stream1_transformer_gemm_stage_policy(nullptr) == Stream1TransformerGemmStagePolicy::Stages3,
            "unset GEMM stage policy must select stages3");
    require(parse_stream1_transformer_gemm_stage_policy("") == Stream1TransformerGemmStagePolicy::Stages3,
            "empty GEMM stage policy must select stages3");
    require(parse_stream1_transformer_gemm_stage_policy("3") == Stream1TransformerGemmStagePolicy::Stages3,
            "GEMM stages3 parse failed");
    require(parse_stream1_transformer_gemm_stage_policy("2") == Stream1TransformerGemmStagePolicy::Stages2,
            "GEMM stages2 parse failed");

    require(stream1_transformer_gemm_stages(Stream1TransformerGemmStagePolicy::Stages3) == 3,
            "GEMM stages3 descriptor failed");
    require(stream1_transformer_gemm_stages(Stream1TransformerGemmStagePolicy::Stages2) == 2,
            "GEMM stages2 descriptor failed");

    bool stage_rejected = false;
    try { static_cast<void>(parse_stream1_transformer_gemm_stage_policy("unknown")); }
    catch (const std::invalid_argument&) { stage_rejected = true; }
    require(stage_rejected, "unknown GEMM stage policy must fail closed");

    require(parse_stream1_transformer_residual_epilogue_policy(nullptr) ==
                Stream1TransformerResidualEpiloguePolicy::Separate,
            "unset residual epilogue policy must select separate");
    require(parse_stream1_transformer_residual_epilogue_policy("fused") ==
                Stream1TransformerResidualEpiloguePolicy::FusedBiasRound,
            "fused residual epilogue policy parse failed");
    bool residual_epilogue_rejected = false;
    try { static_cast<void>(parse_stream1_transformer_residual_epilogue_policy("unknown")); }
    catch (const std::invalid_argument&) { residual_epilogue_rejected = true; }
    require(residual_epilogue_rejected, "unknown residual epilogue policy must fail closed");

    require(parse_stream1_transformer_gemm_policy(Family::Qkv, nullptr) == Policy::Baseline,
            "unset QKV policy must select baseline");
    require(parse_stream1_transformer_gemm_policy(Family::Qkv, "m128n128") == Policy::M128N128,
            "QKV m128n128 parse failed");
    require(parse_stream1_transformer_gemm_policy(Family::Qkv, "m64n128") == Policy::M64N128,
            "QKV m64n128 parse failed");
    require(parse_stream1_transformer_gemm_policy(Family::Qkv, "m256n128") == Policy::M256N128,
            "QKV m256n128 parse failed");
    require_rejects(Family::Qkv, "m64n64");

    require(parse_stream1_transformer_gemm_policy(Family::AttentionOut, "m64n64") == Policy::M64N64,
            "attention-out m64n64 parse failed");
    require(parse_stream1_transformer_gemm_policy(Family::AttentionOut, "m128n128") == Policy::M128N128,
            "attention-out m128n128 parse failed");
    require_rejects(Family::AttentionOut, "m64n128");

    require(parse_stream1_transformer_gemm_policy(Family::Ff1, "m64n128") == Policy::M64N128,
            "FF1 m64n128 parse failed");
    require(parse_stream1_transformer_gemm_policy(Family::Ff1, "m128n128") == Policy::M128N128,
            "FF1 m128n128 parse failed");
    require(parse_stream1_transformer_gemm_policy(Family::Ff1, "m128n128w64n32") == Policy::M128N128W64N32,
            "FF1 m128n128w64n32 parse failed");

    require(parse_stream1_transformer_gemm_policy(Family::Ff2, "m64n64") == Policy::M64N64,
            "FF2 m64n64 parse failed");
    require(parse_stream1_transformer_gemm_policy(Family::Ff2, "m128n128") == Policy::M128N128,
            "FF2 m128n128 parse failed");

    require(parse_stream1_transformer_gemm_policy(Family::Cls, "m32n64") == Policy::M32N64,
            "CLS m32n64 parse failed");
    require(parse_stream1_transformer_gemm_policy(Family::Cls, "m64n64") == Policy::M64N64,
            "CLS m64n64 parse failed");

    const Stream1TransformerGemmPolicyDesc baseline =
        stream1_transformer_gemm_policy_desc(Policy::Baseline);
    require(baseline.threadblock_m == 128 && baseline.threadblock_n == 64 &&
                baseline.warp_m == 64 && baseline.warp_n == 32 && baseline.stages == 3,
            "baseline descriptor mismatch");
    const Stream1TransformerGemmPolicyDesc m64n128 =
        stream1_transformer_gemm_policy_desc(Policy::M64N128);
    require(m64n128.threadblock_m == 64 && m64n128.threadblock_n == 128 &&
                m64n128.warp_m == 32 && m64n128.warp_n == 64,
            "m64n128 descriptor mismatch");
    const Stream1TransformerGemmPolicyDesc m128n128w64n32 =
        stream1_transformer_gemm_policy_desc(Policy::M128N128W64N32);
    require(m128n128w64n32.threadblock_m == 128 && m128n128w64n32.threadblock_n == 128 &&
                m128n128w64n32.warp_m == 64 && m128n128w64n32.warp_n == 32,
            "m128n128w64n32 descriptor mismatch");
    const Stream1TransformerGemmPolicyDesc m256n128 =
        stream1_transformer_gemm_policy_desc(Policy::M256N128);
    require(m256n128.threadblock_m == 256 && m256n128.threadblock_n == 128 &&
                m256n128.warp_m == 64 && m256n128.warp_n == 64,
            "m256n128 descriptor mismatch");

    const Stream1TransformerGemmPolicyDesc m32n64 =
        stream1_transformer_gemm_policy_desc(Policy::M32N64);
    require(m32n64.threadblock_m == 32 && m32n64.threadblock_n == 64,
            "m32n64 descriptor mismatch");

    require(parse_stream1_transformer_ff1_policy(nullptr) == Stream1TransformerFf1Policy::Baseline,
            "legacy unset FF1 policy must select baseline");
    require(parse_stream1_transformer_ff1_policy("m64n128") == Stream1TransformerFf1Policy::M64N128,
            "legacy FF1 m64n128 parse failed");
    require_rejects(Family::Ff1, "invalid-policy");

    require(stream1_transformer_gemm_policy_supported_on_sm(
                Family::Qkv, Policy::M128N128, 75),
            "SM75 must expose opt-in QKV m128n128 for hardware tuning");
    require(stream1_transformer_gemm_policy_supported_on_sm(
                Family::Ff1, Policy::M64N128, 75),
            "SM75 must expose opt-in FF1 m64n128 for hardware tuning");
    require(stream1_transformer_gemm_policy_supported_on_sm(
                Family::AttentionOut, Policy::M128N128, 75),
            "SM75 must expose opt-in attention-out m128n128 for hardware tuning");
    require(stream1_transformer_gemm_policy_supported_on_sm(
                Family::Ff2, Policy::M64N64, 75),
            "SM75 must expose opt-in FF2 m64n64 for hardware tuning");
    require(!stream1_transformer_gemm_policy_supported_on_sm(
                Family::Cls, Policy::M128N128, 75),
            "SM75 must keep uncompiled CLS m128n128 fail-closed");

    require(stream1_transformer_gemm_stage_supported_on_sm(
                Family::Ff1, Stream1TransformerGemmStagePolicy::Stages2, 75),
            "SM75 FF1 must compile with two pipeline stages");
    require(!stream1_transformer_gemm_stage_supported_on_sm(
                Family::Ff1, Stream1TransformerGemmStagePolicy::Stages3, 75),
            "SM75 FF1 must reject unsupported three-stage fused epilogue");
    require(stream1_transformer_gemm_stage_supported_on_sm(
                Family::Ff1, Stream1TransformerGemmStagePolicy::Stages3, 80),
            "SM80 FF1 must retain three-stage support");
    std::cout << "stream1_transformer_gemm_policy_tests=pass\n";
    return 0;
}
