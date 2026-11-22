#pragma once
#include "stream1_policy_snapshot.hpp"
#include "stream1_transformer_attention_policy.hpp"
#include "stream1_transformer_layernorm_policy.hpp"
#include "stream1_transformer_hopper_policy.hpp"
#include "stream1_layernorm_execution.hpp"
#include <array>
#include <cerrno>
namespace beam {
inline constexpr std::array<const char*,8> stream1_launch_flag_keys={
    "BEAM_STREAM1_TRANSFORMER_LEGACY_PADDING_ZERO","BEAM_STREAM1_TRANSFORMER_BLOCK51",
    "BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY","BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION",
    "BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV","BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM",
    "BEAM_STREAM1_TRANSFORMER_DUAL_INPUT_LN","BEAM_STREAM1_TRANSFORMER_STAGE_PROFILE"};
inline bool parse_stream1_launch_flag(const char* text) {
    if(!text || !*text || std::strcmp(text,"0")==0) return false;
    if(std::strcmp(text,"1")==0) return true;
    throw std::invalid_argument("Stream1 execution flag must be unset,0 or1");
}
struct Stream1ResolvedLaunch {
    Stream1TransformerAttentionTilePolicy attention_tile=Stream1TransformerAttentionTilePolicy::Q64K64;
    Stream1TransformerAttentionMaxKPolicy attention_max_k=Stream1TransformerAttentionMaxKPolicy::Padded64;
    Stream1TransformerClsAttentionPolicy cls_attention=Stream1TransformerClsAttentionPolicy::Cutlass;
    Stream1TransformerLayerNormRowsPolicy layernorm_rows=Stream1TransformerLayerNormRowsPolicy::RowPerBlock;
    bool hopper_ff1_epilogue_128x64=false;
    Stream1LayerNormResidentPlan layernorm_resident{};
    std::array<bool,8> flags{};
    std::uint32_t stage_profile_skip_calls=1;
};
inline thread_local const Stream1ResolvedLaunch* active_stream1_resolved_launch=nullptr;
class Stream1ResolvedLaunchScope {
public:
    explicit Stream1ResolvedLaunchScope(const Stream1ResolvedLaunch& choices)
        : previous_(active_stream1_resolved_launch) { active_stream1_resolved_launch=&choices; }
    Stream1ResolvedLaunchScope(Stream1ResolvedLaunch&&)=delete;
    Stream1ResolvedLaunchScope(const Stream1ResolvedLaunchScope&)=delete;
    Stream1ResolvedLaunchScope& operator=(const Stream1ResolvedLaunchScope&)=delete;
    ~Stream1ResolvedLaunchScope() { active_stream1_resolved_launch=previous_; }
private:
    const Stream1ResolvedLaunch* previous_;
};
inline Stream1TransformerAttentionTilePolicy stream1_resolved_attention_tile(const char* text) {
    if(active_stream1_resolved_launch) return active_stream1_resolved_launch->attention_tile;
    return parse_stream1_transformer_attention_tile_policy(text);
}
inline Stream1TransformerAttentionMaxKPolicy stream1_resolved_attention_max_k(const char* text) {
    if(active_stream1_resolved_launch) return active_stream1_resolved_launch->attention_max_k;
    return parse_stream1_transformer_attention_max_k_policy(text);
}
inline Stream1TransformerClsAttentionPolicy stream1_resolved_cls_attention(const char* text) {
    if(active_stream1_resolved_launch) return active_stream1_resolved_launch->cls_attention;
    return parse_stream1_transformer_cls_attention_policy(text);
}
inline Stream1TransformerLayerNormRowsPolicy stream1_resolved_layernorm_rows(const char* text) {
    if(active_stream1_resolved_launch) return active_stream1_resolved_launch->layernorm_rows;
    return parse_stream1_transformer_layernorm_rows_policy(text);
}
inline bool stream1_resolved_hopper_ff1_epilogue(const char* text) {
    if(active_stream1_resolved_launch) return active_stream1_resolved_launch->hopper_ff1_epilogue_128x64;
    return stream1_transformer_hopper_epilogue_128x64(text);
}
inline bool stream1_resolved_flag(const char* name,const char* text) {
    if(active_stream1_resolved_launch) {
        if(name) for(std::size_t i=0;i<stream1_launch_flag_keys.size();++i)
            if(std::strcmp(name,stream1_launch_flag_keys[i])==0)
                return active_stream1_resolved_launch->flags[i];
        throw std::invalid_argument("unbound resolved Stream1 execution flag");
    }
    return parse_stream1_launch_flag(text);
}
inline std::uint32_t parse_stream1_stage_profile_skip(const char* text) {
    if(!text || !*text) return 1;
    errno=0;
    char* end=nullptr;
    const auto value=std::strtoul(text,&end,10);
    if(errno || end==text || *end || value>std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Stream1 stage-profile skip calls must be uint32");
    return static_cast<std::uint32_t>(value);
}
inline std::uint32_t stream1_resolved_stage_profile_skip(const char* text) {
    return active_stream1_resolved_launch ? active_stream1_resolved_launch->stage_profile_skip_calls
                                         : parse_stream1_stage_profile_skip(text);
}
}
