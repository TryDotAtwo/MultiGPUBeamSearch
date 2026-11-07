#pragma once
#include "stream1_transformer_gemm_policy.hpp"
#include <array>
namespace beam {
struct Stream1ResolvedGemm {
    Stream1TransformerGemmPolicy policy = Stream1TransformerGemmPolicy::Baseline;
    Stream1TransformerGemmStagePolicy stage = Stream1TransformerGemmStagePolicy::Stages3;
    Stream1TransformerGemmSwizzlePolicy swizzle = Stream1TransformerGemmSwizzlePolicy::Identity1;
    Stream1TransformerResidualEpiloguePolicy epilogue = Stream1TransformerResidualEpiloguePolicy::Separate;
};
struct Stream1ResolvedGemmSet {
    int sm=0;
    bool fp16=true;
    std::array<Stream1ResolvedGemm,4> families{};
};
inline thread_local const Stream1ResolvedGemmSet* active_stream1_resolved_gemm = nullptr;
class Stream1ResolvedGemmScope {
public:
    explicit Stream1ResolvedGemmScope(const Stream1ResolvedGemmSet& choices)
        : previous_(active_stream1_resolved_gemm) { active_stream1_resolved_gemm=&choices; }
    Stream1ResolvedGemmScope(Stream1ResolvedGemmSet&&)=delete;
    Stream1ResolvedGemmScope(const Stream1ResolvedGemmScope&)=delete;
    ~Stream1ResolvedGemmScope() { active_stream1_resolved_gemm=previous_; }
private:
    const Stream1ResolvedGemmSet* previous_;
};
// Mirrors unpacked GEMM launch branches. Packed Hopper/FP8, tensor shape and
// compiled-device admission are separate parts of the execution contract.
inline Stream1ResolvedGemm resolve_stream1_gemm_uncached(Stream1TransformerGemmFamily family,
    int sm, bool fp16, const char* policy_text, const char* stage_text,
    const char* swizzle_text, const char* epilogue_text) {
    using F = Stream1TransformerGemmFamily;
    using P = Stream1TransformerGemmPolicy;
    using S = Stream1TransformerGemmStagePolicy;
    using W = Stream1TransformerGemmSwizzlePolicy;
    using E = Stream1TransformerResidualEpiloguePolicy;
    if (family == F::Cls)
        throw std::invalid_argument("actual output head uses QKV linear-bias launcher, not Cls policy family");
    if (sm < 75 || (!fp16 && sm < 80))
        throw std::invalid_argument("GEMM execution requires FP16 SM75+ or BF16 SM80+");
    const bool residual = family == F::Ff2 || family == F::AttentionOut;
    Stream1ResolvedGemm result;
    result.epilogue = parse_stream1_transformer_residual_epilogue_policy(epilogue_text);
    if (!residual && result.epilogue != E::Separate)
        throw std::invalid_argument("residual epilogue selector on non-residual GEMM");
    // Validate supplied syntax even for a fixed BF16 route; report only actual
    // effective choices below, never substitute requested FP16 tiles for BF16.
    const auto requested_policy = parse_stream1_transformer_gemm_policy(family, policy_text);
    const auto requested_stage = parse_stream1_transformer_gemm_stage_policy(stage_text);
    const auto requested_swizzle = parse_stream1_transformer_gemm_swizzle_policy(swizzle_text);
    if (result.epilogue == E::FusedBiasRound) {
        if (!fp16 || requested_policy != P::M128N128)
            throw std::invalid_argument("fused residual requires FP16 and explicit m128n128");
        result.policy = requested_policy;
        result.swizzle = requested_swizzle;
        if (!stream1_transformer_gemm_swizzle_allowed(family, result.policy, result.stage, result.swizzle))
            throw std::invalid_argument("fused residual swizzle is not compiled");
        return result;
    }
    if (residual && requested_swizzle != W::Identity1)
        throw std::invalid_argument("residual swizzle greater than1 requires fused epilogue");
    if (!fp16) return result; // Actual BF16 GEMM branches use fixed baseline/stages3/swizzle1.
    result.policy = sm < 80 ? requested_policy : select_stream1_transformer_gemm_policy(family, policy_text, sm);
    if (residual) {
        // Separate residual uses Gemm's pinned DefaultGemmConfiguration;
        // fused residual above explicitly instantiates three stages.
        result.stage = sm < 80 ? S::Stages2 : S::Stages3;
    } else if (family == F::Qkv) {
        result.stage = sm < 80 ? S::Stages2 : S::Stages3;
    } else if (family == F::Ff1) {
        result.stage = sm < 80 && (!stage_text || !*stage_text) ? S::Stages2 : requested_stage;
    }
    result.swizzle = sm < 80 || residual ? requested_swizzle :
        select_stream1_transformer_gemm_swizzle_policy(family, result.policy, result.stage, swizzle_text, sm);
    if (!stream1_transformer_gemm_policy_supported_on_sm(family, result.policy, sm) ||
        !stream1_transformer_gemm_stage_supported_on_sm(family, result.stage, sm) ||
        !stream1_transformer_gemm_swizzle_allowed(family, result.policy, result.stage, result.swizzle) ||
        (sm < 80 && (family == F::Qkv || family == F::Ff1) && result.swizzle != W::Identity1))
        throw std::invalid_argument("GEMM policy/stage/swizzle combination is not compiled");
    return result;
}
inline Stream1ResolvedGemm resolve_stream1_gemm(Stream1TransformerGemmFamily family,
    int sm, bool fp16, const char* policy_text, const char* stage_text,
    const char* swizzle_text, const char* epilogue_text) {
    if (active_stream1_resolved_gemm) {
        const auto& choices=*active_stream1_resolved_gemm;
        const auto index=static_cast<unsigned>(family);
        if (choices.sm!=sm || choices.fp16!=fp16 || index>=choices.families.size())
            throw std::invalid_argument("GEMM launch disagrees with owned execution contract");
        return choices.families[index];
    }
    return resolve_stream1_gemm_uncached(family,sm,fp16,policy_text,stage_text,swizzle_text,epilogue_text);
}
} // namespace beam
