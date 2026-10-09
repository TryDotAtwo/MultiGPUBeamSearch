#pragma once
#include "stream1_loaded_execution_admission.hpp"
#include "stream1_resolved_launch.hpp"
#include <array>
#include <algorithm>

namespace beam {
// Interface foundation: not yet a complete resolved-kernel admission contract.
class Stream1ExecutionContract {
public:
    Stream1ExecutionContract(const Stream1TransformerNetworkView& view, int sm,
        bool hopper, std::uint32_t outer, std::uint32_t inner, std::uint32_t lanes,
        const Stream1PolicySnapshot& policies,
        const Stream1LayerNormOccupancy& occupancy={},bool graph_executor=false)
        : view_(view), policies_(policies), sm_(sm), outer_(outer), inner_(inner), lanes_(lanes), graph_executor_(graph_executor) {
        validate_stream1_cube4_loaded_execution(view_,sm,hopper,outer,inner,lanes,policies_);
        std::copy_n(view.blocks,blocks_.size(),blocks_.begin());
        view_.blocks=blocks_.data();
        for(std::size_t i=0;i<stream1_launch_flag_keys.size();++i)
            launch_.flags[i]=parse_stream1_launch_flag(policies_.get(stream1_launch_flag_keys[i]));
        launch_.stage_profile_skip_calls=parse_stream1_stage_profile_skip(policies_.get("BEAM_STREAM1_TRANSFORMER_STAGE_PROFILE_SKIP_CALLS"));
        if(graph_executor_ && launch_.flags[7])
            throw std::invalid_argument("stage profiling requires eager executor, not graph capture");
        launch_.attention_tile=parse_stream1_transformer_attention_tile_policy(policies_.get("BEAM_STREAM1_TRANSFORMER_ATTENTION_TILE_POLICY"));
        launch_.attention_max_k=parse_stream1_transformer_attention_max_k_policy(policies_.get("BEAM_STREAM1_TRANSFORMER_ATTENTION_MAX_K_POLICY"));
        launch_.cls_attention=parse_stream1_transformer_cls_attention_policy(policies_.get("BEAM_STREAM1_TRANSFORMER_CLS_ATTENTION_POLICY"));
        launch_.layernorm_rows=parse_stream1_transformer_layernorm_rows_policy(policies_.get("BEAM_STREAM1_TRANSFORMER_LAYERNORM_ROWS_POLICY"));
        if(launch_.layernorm_rows==Stream1TransformerLayerNormRowsPolicy::PersistentRows)
            launch_.layernorm_resident=resolve_stream1_layernorm_resident_plan(occupancy,sm,
                policies_.get("BEAM_STREAM1_TRANSFORMER_LAYERNORM_PERSISTENT_BLOCKS_PER_SM"));
        launch_.hopper_ff1_epilogue_128x64=stream1_transformer_hopper_epilogue_128x64(policies_.get("BEAM_STREAM1_TRANSFORMER_HOPPER_FF1_EPILOGUE"));
        gemm_.sm=sm;
        gemm_.fp16=view_.dims.dtype==STREAM1_DTYPE_FP16;
        using F=Stream1TransformerGemmFamily;
        auto resolve=[&](F family,const char* p,const char* s,const char* w,const char* e) {
            gemm_.families[static_cast<unsigned>(family)]=resolve_stream1_gemm_uncached(
                family,sm,gemm_.fp16,policies_.get(p),s?policies_.get(s):nullptr,
                policies_.get(w),e?policies_.get(e):nullptr);
        };
        resolve(F::Qkv,"BEAM_STREAM1_TRANSFORMER_QKV_POLICY",nullptr,"BEAM_STREAM1_TRANSFORMER_QKV_SWIZZLE",nullptr);
        resolve(F::AttentionOut,"BEAM_STREAM1_TRANSFORMER_ATTN_OUT_POLICY",nullptr,"BEAM_STREAM1_TRANSFORMER_ATTN_OUT_SWIZZLE","BEAM_STREAM1_TRANSFORMER_ATTN_OUT_EPILOGUE");
        resolve(F::Ff1,"BEAM_STREAM1_TRANSFORMER_FF1_POLICY","BEAM_STREAM1_TRANSFORMER_FF1_STAGES","BEAM_STREAM1_TRANSFORMER_FF1_SWIZZLE",nullptr);
        resolve(F::Ff2,"BEAM_STREAM1_TRANSFORMER_FF2_POLICY",nullptr,"BEAM_STREAM1_TRANSFORMER_FF2_SWIZZLE","BEAM_STREAM1_TRANSFORMER_FF2_EPILOGUE");
    }
    Stream1ExecutionContract(const Stream1ExecutionContract&)=delete;
    Stream1ExecutionContract& operator=(const Stream1ExecutionContract&)=delete;
    const Stream1TransformerNetworkView& network() const noexcept { return view_; }
    const Stream1PolicySnapshot& policies() const noexcept { return policies_; }
    const Stream1ResolvedGemmSet& gemm_choices() const noexcept { return gemm_; }
    const Stream1ResolvedLaunch& launch_choices() const noexcept { return launch_; }
    int sm() const noexcept { return sm_; }
    std::uint32_t outer_microbatch() const noexcept { return outer_; }
    std::uint32_t transformer_microbatch() const noexcept { return inner_; }
    std::uint32_t lanes() const noexcept { return lanes_; }
    bool graph_executor() const noexcept { return graph_executor_; }
private:
    Stream1ResolvedGemmSet gemm_{};
    Stream1ResolvedLaunch launch_{};
    std::array<Stream1TransformerBlockView,4> blocks_{};
    Stream1TransformerNetworkView view_;
    Stream1PolicySnapshot policies_;
    int sm_;
    std::uint32_t outer_,inner_,lanes_;
    bool graph_executor_;
};
}
