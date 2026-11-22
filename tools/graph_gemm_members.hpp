#pragma once
#include "graph_pointer_registry.hpp"
#include "../cuda/cuda_check.hpp"
#include "../third_party/nlohmann/json.hpp"
#include <cstring>
#include <type_traits>
#if BEAM_HAS_CUTLASS
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/epilogue/thread/activation.h>
#include "mainloop_iterator_observation.hpp"
#endif
namespace beam::score_mode {
inline void gemm_member_endpoint(nlohmann::json& out,const char* name,const void* pointer,
    const GraphPointerRegistry& registry) {
    if(!pointer) {out[name]={{"role","null"},{"offset",0},{"available_bytes",0}};return;}
    const auto endpoint=registry.resolve(pointer,1);
    out[name]={{"role",endpoint.role},{"offset",endpoint.offset},{"available_bytes",endpoint.available_bytes}};
}
#if BEAM_HAS_CUTLASS
using DiagnosticGemmElement=cutlass::half_t;
using DiagnosticGemmLayout=cutlass::layout::RowMajor;
template<class P>nlohmann::json observed_epilogue_iterator(const P& p) {
    return {{"stride",p.stride},{"increment_row",p.increment_row},
        {"increment_group",p.increment_group},{"increment_cluster",p.increment_cluster},
        {"advance_row",p.advance_row},{"advance_group",p.advance_group},
        {"advance_cluster",p.advance_cluster},{"advance_tile",p.advance_tile}};
}
template<class G,bool Broadcast>
bool extract_gemm_members(const cudaKernelNodeParams& node,const std::string& symbol,
    const GraphPointerRegistry& registry,nlohmann::json& result) {
    using K=typename G::GemmKernel;using P=typename K::Params;
    const char* name=nullptr;
    if constexpr(Broadcast)BEAM_CUDA_CHECK(cudaFuncGetName(&name,cutlass::Kernel2<K>));
    else BEAM_CUDA_CHECK(cudaFuncGetName(&name,cutlass::Kernel<K>));
    if(!name || symbol!=name)return false;
    static_assert(std::is_trivially_copyable_v<P>);
    std::size_t offset=0,size=0;BEAM_CUDA_CHECK(cudaFuncGetParamInfo(node.func,0,&offset,&size));
    if(offset || size!=sizeof(P) || node.extra || !node.kernelParams || !node.kernelParams[0])
        throw std::runtime_error("unsupported captured GEMM Params representation");
    P p{};std::memcpy(&p,node.kernelParams[0],sizeof(p));
    nlohmann::json launch_values={{"grid_tiled_shape",{{"m",p.grid_tiled_shape.m()},
        {"n",p.grid_tiled_shape.n()},{"k",p.grid_tiled_shape.k()}}},
        {"swizzle_log_tile",p.swizzle_log_tile},{"gemm_k_size",p.gemm_k_size},
        {"semaphore_null",p.semaphore==nullptr}};
    if constexpr(Broadcast) {
        launch_values["mode"]=static_cast<int>(p.mode);
        launch_values["batch_count"]=p.batch_count;
        launch_values["batch_stride_D"]=p.batch_stride_D;
    }
    result["launch_scalars"]=std::move(launch_values);
    auto mainloop=[](const MainloopBase& q) {
        return nlohmann::json{{"stride",q.stride_},{"increment_strided",q.inc_strided_},
            {"increment_next",q.inc_next_},{"advance",q.inc_advance_}};
    };
    result["mainloop_iterators"]={
        {"A",mainloop(observe_mainloop_base<typename K::Mma::IteratorA>(p.params_A))},
        {"B",mainloop(observe_mainloop_base<typename K::Mma::IteratorB>(p.params_B))}};
    result["epilogue_iterators"]={{"C",observed_epilogue_iterator(p.params_C)},
        {"D",observed_epilogue_iterator(p.params_D)}};
    result["tensor_iterator"]=nlohmann::json::object();
    if constexpr(Broadcast)result["tensor_iterator"]=observed_epilogue_iterator(p.params_Tensor);
    auto epilogue_pointers=nlohmann::json::object();
    gemm_member_endpoint(epilogue_pointers,"alpha",p.output_op.alpha_ptr,registry);
    gemm_member_endpoint(epilogue_pointers,"beta",p.output_op.beta_ptr,registry);
    if constexpr(!Broadcast) {
        gemm_member_endpoint(epilogue_pointers,"alpha_array",p.output_op.alpha_ptr_array,registry);
        gemm_member_endpoint(epilogue_pointers,"beta_array",p.output_op.beta_ptr_array,registry);
    }
    result["epilogue_pointers"]=std::move(epilogue_pointers);
    result["problem"]={{"m",p.problem_size.m()},{"n",p.problem_size.n()},{"k",p.problem_size.k()}};
    auto pointers=nlohmann::json::object();
    auto add=[&](const char* key,const void* pointer){gemm_member_endpoint(pointers,key,pointer,registry);};
    nlohmann::json scalars={{"alpha",p.output_op.alpha},{"beta",p.output_op.beta}};
    if constexpr(Broadcast) {
        add("A",p.ptr_A);add("B",p.ptr_B);add("C",p.ptr_C);
        add("D",p.ptr_D);add("Vector",p.ptr_Vector);add("Tensor",p.ptr_Tensor);
        scalars["batch_stride_A"]=p.batch_stride_A;scalars["batch_stride_B"]=p.batch_stride_B;
        scalars["batch_stride_C"]=p.batch_stride_C;
        scalars["batch_stride_Vector"]=p.batch_stride_Vector;scalars["batch_stride_Tensor"]=p.batch_stride_Tensor;
        scalars["ldr"]=p.ldr;
    } else {
        add("A",p.ref_A.data());add("B",p.ref_B.data());add("C",p.ref_C.data());add("D",p.ref_D.data());
        add("semaphore",p.semaphore);add("gather_A",p.gather_A_indices);
        add("gather_B",p.gather_B_indices);add("scatter_D",p.scatter_D_indices);
        scalars["lda"]=p.ref_A.stride(0);scalars["ldb"]=p.ref_B.stride(0);
        scalars["ldc"]=p.ref_C.stride(0);scalars["ldd"]=p.ref_D.stride(0);
        scalars["gemm_k_size"]=p.gemm_k_size;
    }
    result["pointers"]=std::move(pointers);result["scalars"]=std::move(scalars);
    result["gemm_signature_known"]=true;result["parameter_index"]=0;
    return true;
}
template<class Arch,class Inst,int Stages,class Shape,class Warp,class Activation>
using DiagnosticBroadcast=cutlass::gemm::device::GemmUniversalWithBroadcast<
    DiagnosticGemmElement,DiagnosticGemmLayout,DiagnosticGemmElement,DiagnosticGemmLayout,
    DiagnosticGemmElement,DiagnosticGemmLayout,float,cutlass::arch::OpClassTensorOp,
    Arch,Shape,Warp,Inst,cutlass::epilogue::thread::LinearCombinationBiasElementwise<
        DiagnosticGemmElement,float,float,DiagnosticGemmElement,DiagnosticGemmElement,8,
        Activation,cutlass::plus<float>,false,DiagnosticGemmElement>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,Stages>;
template<class Arch,class Inst,int Stages>
bool extract_gemm_baseline(const cudaKernelNodeParams& node,const std::string& symbol,
    const GraphPointerRegistry& registry,nlohmann::json& result) {
    using S=cutlass::gemm::GemmShape<128,64,32>;using W=cutlass::gemm::GemmShape<64,32,32>;
    using E=DiagnosticGemmElement;using L=DiagnosticGemmLayout;
    using Regular=cutlass::gemm::device::Gemm<E,L,E,L,E,L,float,cutlass::arch::OpClassTensorOp,Arch,S,W,Inst>;
    using Bias=DiagnosticBroadcast<Arch,Inst,Stages,S,W,cutlass::epilogue::thread::Identity<float>>;
    using Relu=DiagnosticBroadcast<Arch,Inst,Stages,S,W,cutlass::epilogue::thread::ReLu<float>>;
    return extract_gemm_members<Regular,false>(node,symbol,registry,result) ||
        extract_gemm_members<Bias,true>(node,symbol,registry,result) ||
        extract_gemm_members<Relu,true>(node,symbol,registry,result);
}
#endif
inline nlohmann::json captured_gemm_members(const cudaKernelNodeParams& node,const std::string& symbol,
    const GraphPointerRegistry* registry,std::size_t index,std::uint32_t fid) {
    nlohmann::json result={{"kernel_index",index},{"function_id",fid},{"gemm_signature_known",false},
        {"parameter_index",nullptr},{"problem",nlohmann::json::object()},
        {"pointers",nlohmann::json::object()},{"scalars",nlohmann::json::object()},
        {"launch_scalars",nlohmann::json::object()}};
    if(!registry || (symbol.rfind("_ZN7cutlass6KernelI",0)!=0 && symbol.rfind("_ZN7cutlass7Kernel2I",0)!=0))return result;
#if BEAM_HAS_CUTLASS
    if(extract_gemm_baseline<cutlass::arch::Sm75,cutlass::gemm::GemmShape<16,8,8>,2>(node,symbol,*registry,result) ||
       extract_gemm_baseline<cutlass::arch::Sm80,cutlass::gemm::GemmShape<16,8,16>,3>(node,symbol,*registry,result))return result;
    using Split=DiagnosticBroadcast<cutlass::arch::Sm80,cutlass::gemm::GemmShape<16,8,16>,3,
        cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::epilogue::thread::Identity<float>>;
    extract_gemm_members<Split,true>(node,symbol,*registry,result);
#endif
    return result; // Unknown specializations stay explicitly unknown.
}
}
