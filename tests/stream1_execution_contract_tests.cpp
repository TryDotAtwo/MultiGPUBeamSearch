#include "../cuda/stream1_execution_contract.hpp"
#include "../cuda/stream1_transformer_policy_snapshot.hpp"
#include "../cuda/stream1_layernorm_execution.hpp"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <type_traits>
using namespace beam;
int main() {
#ifndef _WIN32
    setenv("BEAM_STREAM1_TRANSFORMER_FF1_POLICY","m128n128",1);
    setenv("BEAM_STREAM1_TRANSFORMER_FF1_STAGES","2",1);
    setenv("BEAM_STREAM1_TRANSFORMER_FF1_SWIZZLE","1",1);
#endif
    Stream1TransformerBlockView blocks[4]{};
    Stream1TransformerNetworkView view{};
    view.blocks=blocks;
    view.dims={96,6,56,3,57,64,16,256,8,32,4,1024,24,STREAM1_DTYPE_FP16,STREAM1_ACTIVATION_RELU};
    const auto requested=capture_stream1_transformer_launch_policies();
    const Stream1ExecutionContract contract(view,86,false,32,13,2,requested);
    blocks[0].ff1_hopper_fp16=true;
    view.dims.activation=STREAM1_ACTIVATION_SILU;
    if(contract.network().blocks[0].ff1_hopper_fp16 ||
       contract.network().dims.activation!=STREAM1_ACTIVATION_RELU)
        throw std::runtime_error("contract aliases mutable loaded view");
    if(contract.sm()!=86 || contract.outer_microbatch()!=32 ||
       contract.transformer_microbatch()!=13 || contract.lanes()!=2)
        throw std::runtime_error("contract lost actual device/batch/lane shape");
    static_assert(!std::is_copy_constructible_v<Stream1ExecutionContract>);
    static_assert(!std::is_move_constructible_v<Stream1ExecutionContract>);
    {
        const Stream1ResolvedGemmScope scope(contract.gemm_choices());
        const auto actual=resolve_stream1_gemm(Stream1TransformerGemmFamily::Ff1,
            86,true,"changed-invalid-policy","changed-invalid-stage","changed-invalid-swizzle",nullptr);
        if(actual.policy!=Stream1TransformerGemmPolicy::M128N128 ||
           actual.stage!=Stream1TransformerGemmStagePolicy::Stages2 ||
           actual.swizzle!=Stream1TransformerGemmSwizzlePolicy::Identity1)
            throw std::runtime_error("launch re-resolved rather than using frozen GEMM choice");
        bool rejected=false;
        try { (void)resolve_stream1_gemm(Stream1TransformerGemmFamily::Ff1,90,true,nullptr,nullptr,nullptr,nullptr); }
        catch(const std::invalid_argument&) { rejected=true; }
        if(!rejected) throw std::runtime_error("contract accepted another device SM");
        rejected=false;
        try { (void)resolve_stream1_gemm(Stream1TransformerGemmFamily::Ff1,86,false,nullptr,nullptr,nullptr,nullptr); }
        catch(const std::invalid_argument&) { rejected=true; }
        if(!rejected) throw std::runtime_error("contract accepted another dtype");
#ifndef _WIN32
        unsetenv("BEAM_STREAM1_TRANSFORMER_FF1_POLICY");
        unsetenv("BEAM_STREAM1_TRANSFORMER_FF1_STAGES");
        unsetenv("BEAM_STREAM1_TRANSFORMER_FF1_SWIZZLE");
#endif
        const auto other_requested=capture_stream1_transformer_launch_policies();
        blocks[0].ff1_hopper_fp16=false;
        view.dims.activation=STREAM1_ACTIVATION_RELU;
        const Stream1ExecutionContract nested(view,86,false,32,13,2,other_requested);
        {
            const Stream1ResolvedGemmScope nested_scope(nested.gemm_choices());
            const auto nested_choice=resolve_stream1_gemm(Stream1TransformerGemmFamily::Ff1,86,true,nullptr,nullptr,nullptr,nullptr);
            if(nested_choice.policy!=Stream1TransformerGemmPolicy::Baseline)
                throw std::runtime_error("nested construction inherited outer choices");
        }
        if(active_stream1_resolved_gemm!=&contract.gemm_choices())
            throw std::runtime_error("nested GEMM scope did not restore previous contract");
    }
    if(active_stream1_resolved_gemm) throw std::runtime_error("GEMM scope leaked contract");
    {
        const Stream1ResolvedLaunchScope scope(contract.launch_choices());
        if(stream1_resolved_attention_tile("changed-invalid")!=Stream1TransformerAttentionTilePolicy::Q64K64 ||
           stream1_resolved_attention_max_k("changed-invalid")!=Stream1TransformerAttentionMaxKPolicy::Padded64 ||
           stream1_resolved_cls_attention("changed-invalid")!=Stream1TransformerClsAttentionPolicy::Cutlass ||
           stream1_resolved_layernorm_rows("changed-invalid")!=Stream1TransformerLayerNormRowsPolicy::RowPerBlock ||
           stream1_resolved_hopper_ff1_epilogue("changed-invalid"))
            throw std::runtime_error("non-GEMM launch reparsed rather than consumed frozen choice");
        Stream1ResolvedLaunch alternative;
        alternative.attention_tile=Stream1TransformerAttentionTilePolicy::Q32K64;
        alternative.attention_max_k=Stream1TransformerAttentionMaxKPolicy::Exact32;
        alternative.cls_attention=Stream1TransformerClsAttentionPolicy::Q32K64;
        alternative.layernorm_rows=Stream1TransformerLayerNormRowsPolicy::PersistentRows;
        alternative.hopper_ff1_epilogue_128x64=true;
        {
            const Stream1ResolvedLaunchScope nested(alternative);
            if(stream1_resolved_attention_tile(nullptr)!=alternative.attention_tile ||
               stream1_resolved_attention_max_k(nullptr)!=alternative.attention_max_k ||
               stream1_resolved_cls_attention(nullptr)!=alternative.cls_attention ||
               stream1_resolved_layernorm_rows(nullptr)!=alternative.layernorm_rows ||
               !stream1_resolved_hopper_ff1_epilogue(nullptr))
                throw std::runtime_error("nested non-GEMM choices ignored");
        }
        if(active_stream1_resolved_launch!=&contract.launch_choices())
            throw std::runtime_error("nested non-GEMM scope did not restore contract");
    }
    if(active_stream1_resolved_launch) throw std::runtime_error("non-GEMM scope leaked contract");
    {
        const Stream1ResolvedLaunchScope scope(contract.launch_choices());
        if(stream1_resolved_flag("BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM","changed-invalid"))
            throw std::runtime_error("execution flags ignored frozen launch contract");
        if(stream1_resolved_stage_profile_skip("changed-invalid")!=1)
            throw std::runtime_error("profiling skip ignored frozen launch contract");
        bool rejected=false;
        try { (void)stream1_resolved_flag("unknown_flag","0"); }
        catch(const std::invalid_argument&) { rejected=true; }
        if(!rejected) throw std::runtime_error("unbound execution flag accepted");
    }
    {
        const Stream1LayerNormOccupancy observed{86,84,8,6};
        const auto plan=resolve_stream1_layernorm_resident_plan(observed,86,"4");
        if(plan.copy_grid!=336 || plan.bias_round_grid!=504)
            throw std::runtime_error("resident LN grid ignores actual occupancy or requested copy limit");
        const auto defaults=resolve_stream1_layernorm_resident_plan(observed,86,nullptr);
        if(defaults.copy_grid!=672 || defaults.bias_round_grid!=504)
            throw std::runtime_error("resident LN default is not actual maximum occupancy");
        for(const char* invalid:{"0","9","bad"}) {
            bool rejected=false;
            try { (void)resolve_stream1_layernorm_resident_plan(observed,86,invalid); }
            catch(const std::invalid_argument&) { rejected=true; }
            if(!rejected) throw std::runtime_error("invalid resident copy limit accepted");
        }
        bool rejected=false;
        try { (void)resolve_stream1_layernorm_resident_plan(observed,90,nullptr); }
        catch(const std::invalid_argument&) { rejected=true; }
        if(!rejected) throw std::runtime_error("occupancy from another SM accepted");
#ifndef _WIN32
        setenv("BEAM_STREAM1_TRANSFORMER_LAYERNORM_ROWS_POLICY","persistent",1);
        setenv("BEAM_STREAM1_TRANSFORMER_LAYERNORM_PERSISTENT_BLOCKS_PER_SM","4",1);
#endif
        const auto persistent_policies=capture_stream1_transformer_launch_policies();
        const Stream1ExecutionContract persistent(view,86,false,32,13,2,persistent_policies,observed);
        if(persistent.launch_choices().layernorm_resident.copy_grid!=336 ||
           persistent.launch_choices().layernorm_resident.bias_round_grid!=504)
            throw std::runtime_error("contract lost independently observed resident grids");
        rejected=false;
        try { const Stream1ExecutionContract missing(view,86,false,32,13,2,persistent_policies); }
        catch(const std::invalid_argument&) { rejected=true; }
        if(!rejected) throw std::runtime_error("persistent contract accepted missing occupancy");
#ifndef _WIN32
        setenv("BEAM_STREAM1_TRANSFORMER_STAGE_PROFILE","1",1);
        setenv("BEAM_STREAM1_TRANSFORMER_STAGE_PROFILE_SKIP_CALLS","3",1);
#endif
        const auto profiling_policies=capture_stream1_transformer_launch_policies();
        const Stream1ExecutionContract eager_profile(view,86,false,32,13,2,profiling_policies,observed,false);
        {
            const Stream1ResolvedLaunchScope scope(eager_profile.launch_choices());
            if(!stream1_resolved_flag("BEAM_STREAM1_TRANSFORMER_STAGE_PROFILE","invalid") ||
               stream1_resolved_stage_profile_skip("invalid")!=3)
                throw std::runtime_error("profiling choices not frozen");
        }
        rejected=false;
        try { const Stream1ExecutionContract graph_profile(view,86,false,32,13,2,profiling_policies,observed,true); }
        catch(const std::invalid_argument&) { rejected=true; }
        if(!rejected) throw std::runtime_error("graph profiling conflict admitted");
    }
    std::cout<<"execution_contract_owned_view=pass\n";
}
