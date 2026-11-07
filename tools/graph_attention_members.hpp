#pragma once
#include "graph_pointer_registry.hpp"
#include "../cuda/cuda_check.hpp"
#include "../third_party/nlohmann/json.hpp"
#include <cstring>
#include <type_traits>
#if BEAM_HAS_CUTLASS_FMHA
#include <cutlass/numeric_types.h>
#include "kernel_forward.h"
#endif

namespace beam::score_mode {
#if BEAM_HAS_CUTLASS_FMHA
template<class Arch,int Q,int K,bool Aligned>
bool extract_attention_members(const cudaKernelNodeParams& node,
    const std::string& symbol,const GraphPointerRegistry& registry,nlohmann::json& result) {
    using A=AttentionKernel<cutlass::half_t,Arch,Aligned,Q,64,K,false,false>;
    const char* compiled_name=nullptr;
    BEAM_CUDA_CHECK(cudaFuncGetName(&compiled_name,attention_kernel_batched_impl<A>));
    if(!compiled_name || symbol!=compiled_name)return false;
    using P=typename A::Params;
    static_assert(std::is_trivially_copyable_v<P>);
    std::size_t offset=0,size=0;
    BEAM_CUDA_CHECK(cudaFuncGetParamInfo(node.func,0,&offset,&size));
    if(offset || size!=sizeof(P) || node.extra || !node.kernelParams || !node.kernelParams[0])
        throw std::runtime_error("unsupported captured attention Params storage");
    P p{};std::memcpy(&p,node.kernelParams[0],sizeof(p));
    auto pointers=nlohmann::json::object();
    auto add=[&](const char* name,const void* pointer) {
        if(!pointer) {pointers[name]={{"role","null"},{"offset",0},{"available_bytes",0}};return;}
        const auto endpoint=registry.resolve(pointer,1);
        pointers[name]={{"role",endpoint.role},{"offset",endpoint.offset},
            {"available_bytes",endpoint.available_bytes}};
    };
    add("query_ptr",p.query_ptr);add("key_ptr",p.key_ptr);
    add("value_ptr",p.value_ptr);add("output_ptr",p.output_ptr);
    add("output_accum_ptr",p.output_accum_ptr);add("logsumexp_ptr",p.logsumexp_ptr);
    result["pointers"]=std::move(pointers);
    result["scalars"]={{"scale",p.scale},{"num_heads",p.num_heads},
        {"num_batches",p.num_batches},{"head_dim",p.head_dim},{"head_dim_value",p.head_dim_value},
        {"num_queries",p.num_queries},{"num_keys",p.num_keys},{"custom_mask_type",int(p.custom_mask_type)},
        {"q_strideH",p.q_strideH},{"k_strideH",p.k_strideH},{"v_strideH",p.v_strideH},
        {"q_strideM",p.q_strideM},{"k_strideM",p.k_strideM},{"v_strideM",p.v_strideM},
        {"q_strideB",p.q_strideB},{"k_strideB",p.k_strideB},{"v_strideB",p.v_strideB},
        {"o_strideM",p.o_strideM}};
    pointers=nlohmann::json::object();
    add("attn_bias_ptr",p.attn_bias_ptr);add("seqstart_q_ptr",p.seqstart_q_ptr);
    add("seqstart_k_ptr",p.seqstart_k_ptr);add("seqlen_k_ptr",p.seqlen_k_ptr);
    result["optional_pointers"]=std::move(pointers);
    result["optional_scalars"]={{"causal_diagonal_offset",p.causal_diagonal_offset},
        {"num_keys_absolute",p.num_keys_absolute},{"bias_strideM",p.bias_strideM},
        {"bias_strideH",p.bias_strideH},{"bias_strideB",p.bias_strideB},
        {"use_dropout",p.use_dropout},{"dropout_batch_head_rng_offset",p.dropout_batch_head_rng_offset},
        {"dropout_prob",p.dropout_prob}};
#ifdef HAS_PYTORCH
    result["pytorch_rng_state_present"]=true;
#else
    result["pytorch_rng_state_present"]=false;
#endif
    return true;
}
template<class Arch>
bool extract_attention_arch(const cudaKernelNodeParams& node,const std::string& symbol,
    const GraphPointerRegistry& registry,nlohmann::json& result) {
    return extract_attention_members<Arch,64,64,true>(node,symbol,registry,result) ||
        extract_attention_members<Arch,32,64,true>(node,symbol,registry,result) ||
        extract_attention_members<Arch,64,32,true>(node,symbol,registry,result) ||
        extract_attention_members<Arch,32,32,true>(node,symbol,registry,result) ||
        extract_attention_members<Arch,64,64,false>(node,symbol,registry,result) ||
        extract_attention_members<Arch,64,32,false>(node,symbol,registry,result);
}
#endif
inline nlohmann::json captured_attention_members(const cudaKernelNodeParams& node,
    const std::string& symbol,const GraphPointerRegistry* registry,std::size_t index,std::uint32_t fid) {
    nlohmann::json result={{"kernel_index",index},{"function_id",fid},
        {"attention_signature_known",false},{"parameter_index",nullptr},
        {"pointers",nlohmann::json::object()},{"scalars",nlohmann::json::object()},
        {"optional_pointers",nlohmann::json::object()},{"optional_scalars",nlohmann::json::object()},
        {"pytorch_rng_state_present",false}};
    if(symbol.rfind("_Z29attention_kernel_batched_impl",0)!=0 || !registry)return result;
#if BEAM_HAS_CUTLASS_FMHA
    if(extract_attention_arch<cutlass::arch::Sm75>(node,symbol,*registry,result) ||
       extract_attention_arch<cutlass::arch::Sm80>(node,symbol,*registry,result)) {
        result["attention_signature_known"]=true;result["parameter_index"]=0;return result;
    }
#endif
    throw std::runtime_error("unknown captured attention Params specialization");
}
}
