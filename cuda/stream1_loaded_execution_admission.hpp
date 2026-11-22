#pragma once
#include "stream1.hpp"
#include "stream1_policy_snapshot.hpp"
#include "stream1_resolved_gemm.hpp"
#include "stream1_transformer_shape.hpp"
#include "stream1_transformer_hopper_policy.hpp"
#include "stream1_transformer_attention_policy.hpp"
#include "stream1_transformer_layernorm_policy.hpp"
#include <cmath>
#include <cstring>
#include <limits>
namespace beam {
// Structural compatibility only. Does not attest tensor content or numerical
// quality, and must not be emitted as a production-score admission receipt.
inline void validate_stream1_cube4_loaded_execution(
    const Stream1TransformerNetworkView& view, int sm, bool compiled_hopper,
    std::uint32_t outer, std::uint32_t inner, std::uint32_t lanes,
    const Stream1PolicySnapshot& policies) {
    const auto& d=view.dims;
    if(sm<75 || (d.dtype==STREAM1_DTYPE_BF16 && sm<80) ||
       (d.dtype!=STREAM1_DTYPE_FP16 && d.dtype!=STREAM1_DTYPE_BF16))
        throw std::invalid_argument("Cube4 execution requires FP16 SM75+ or BF16 SM80+");
    if(d.state_len!=96 || d.num_classes!=6 || d.num_pieces!=56 || d.max_piece_size!=3 ||
       d.seq_len!=57 || d.d_model!=256 || d.nhead!=8 || d.head_dim!=32 ||
       d.transformer_layers!=4 || d.ff_dim!=1024 || d.output_dim!=24 ||
       d.activation!=STREAM1_ACTIVATION_RELU)
        throw std::invalid_argument("loaded Cube4 transformer shape/activation is unsupported");
    if(d.sequence_alignment!=1 && d.sequence_alignment!=16)
        throw std::invalid_argument("unsupported loaded Cube4 sequence alignment");
    const auto seq=make_stream1_transformer_sequence_plan(d.seq_len,d.sequence_alignment);
    if(d.padded_seq_len!=seq.padded_seq_len)
        throw std::invalid_argument("loaded sequence padding differs from physical plan");
    if(!outer || !inner || inner>outer || !lanes ||
       static_cast<std::uint64_t>(inner)*d.padded_seq_len>
           static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid Cube4 batch/lane or GEMM row extent");
    if(!view.blocks) throw std::invalid_argument("missing loaded Cube4 block views");
    const auto enabled=[&](const char* key) {
        const char* value=policies.get(key);
        if(!value || !*value || std::strcmp(value,"0")==0) return false;
        if(std::strcmp(value,"1")==0) return true;
        throw std::invalid_argument("Cube4 execution flag must be unset,0 or1");
    };
    const bool cls=enabled("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY");
    const bool attn=enabled("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION");
    const bool split=enabled("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV");
    const bool dual=enabled("BEAM_STREAM1_TRANSFORMER_DUAL_INPUT_LN");
    (void)enabled("BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM");
    (void)parse_stream1_transformer_attention_tile_policy(
        policies.get("BEAM_STREAM1_TRANSFORMER_ATTENTION_TILE_POLICY"));
    (void)parse_stream1_transformer_attention_max_k_policy(
        policies.get("BEAM_STREAM1_TRANSFORMER_ATTENTION_MAX_K_POLICY"));
    (void)parse_stream1_transformer_cls_attention_policy(
        policies.get("BEAM_STREAM1_TRANSFORMER_CLS_ATTENTION_POLICY"));
    const auto ln_policy=parse_stream1_transformer_layernorm_rows_policy(
        policies.get("BEAM_STREAM1_TRANSFORMER_LAYERNORM_ROWS_POLICY"));
    // Cube4 always reaches the 256-column copy path in its block/output LNs.
    // That actual launch implements row/persistent, not block2.
    if(!stream1_transformer_layernorm_copy_policy_supported(ln_policy))
        throw std::invalid_argument("Cube4 LayerNorm copy requires row or persistent policy");
    if((attn && !cls) || (split && (!cls || !attn)))
        throw std::invalid_argument("inconsistent final CLS execution flags");
    const bool fp16=d.dtype==STREAM1_DTYPE_FP16;
    // The ordinary output head uses the QKV launcher even if earlier blocks
    // are physically packed; it must always have an admitted unpacked route.
    const auto qkv=[&] { return resolve_stream1_gemm_uncached(Stream1TransformerGemmFamily::Qkv,sm,fp16,
        policies.get("BEAM_STREAM1_TRANSFORMER_QKV_POLICY"),nullptr,
        policies.get("BEAM_STREAM1_TRANSFORMER_QKV_SWIZZLE"),nullptr); };
    (void)qkv();
    for(unsigned i=0;i<d.transformer_layers;++i) {
        const auto& b=view.blocks[i];
        const float scales[]={b.qkv_e4m3_scale,b.ff1_e4m3_scale,b.ff2_e4m3_scale};
        const bool packed[]={b.qkv_hopper_fp16,b.ff1_hopper_fp16,b.ff2_hopper_fp16};
        for(unsigned k=0;k<3;++k) {
            if(!std::isfinite(scales[k]) || scales[k]<0 || (packed[k] && scales[k]>0))
                throw std::invalid_argument("invalid or conflicting loaded weight layouts");
            (void)resolve_stream1_packed_fp16_hopper(packed[k],fp16,sm,compiled_hopper);
            if(packed[k] && (!cls || !attn || !split || i+1==d.transformer_layers))
                throw std::invalid_argument("packed Cube4 blocks require final CLS routes and unpacked final block");
            if(scales[k]>0 && (!fp16 || sm!=90 || !compiled_hopper ||
                d.padded_seq_len!=57 || !cls || dual))
                throw std::invalid_argument("loaded E4M3 execution is incompatible with device/shape/flags");
        }
        if(b.ff2_hopper_fp16 && d.padded_seq_len!=57)
            throw std::invalid_argument("packed Hopper FF2 requires compact57");
        if(!b.ff1_hopper_fp16 && b.ff1_e4m3_scale==0)
            (void)resolve_stream1_gemm_uncached(Stream1TransformerGemmFamily::Ff1,sm,fp16,
                policies.get("BEAM_STREAM1_TRANSFORMER_FF1_POLICY"),
                policies.get("BEAM_STREAM1_TRANSFORMER_FF1_STAGES"),
                policies.get("BEAM_STREAM1_TRANSFORMER_FF1_SWIZZLE"),nullptr);
        (void)resolve_stream1_gemm_uncached(Stream1TransformerGemmFamily::AttentionOut,sm,fp16,
            policies.get("BEAM_STREAM1_TRANSFORMER_ATTN_OUT_POLICY"),nullptr,
            policies.get("BEAM_STREAM1_TRANSFORMER_ATTN_OUT_SWIZZLE"),
            policies.get("BEAM_STREAM1_TRANSFORMER_ATTN_OUT_EPILOGUE"));
        if(!b.ff2_hopper_fp16 && b.ff2_e4m3_scale==0)
            (void)resolve_stream1_gemm_uncached(Stream1TransformerGemmFamily::Ff2,sm,fp16,
                policies.get("BEAM_STREAM1_TRANSFORMER_FF2_POLICY"),nullptr,
                policies.get("BEAM_STREAM1_TRANSFORMER_FF2_SWIZZLE"),
                policies.get("BEAM_STREAM1_TRANSFORMER_FF2_EPILOGUE"));
    }
}
} // namespace beam
