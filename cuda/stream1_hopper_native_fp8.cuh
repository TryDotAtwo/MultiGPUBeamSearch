#pragma once

// Experimental SM90 boundary. Weights are encoded before inference; activation
// inverse_scale is explicit, finite and positive. No hidden allocator or cache.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/float8.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cmath>
#include <stdexcept>

namespace beam {
namespace hopper_fp8_detail {
__device__ inline float warp_sum(float x) {
    for(int d=16;d;d/=2)x+=__shfl_down_sync(0xffffffff,x,d);
    return __shfl_sync(0xffffffff,x,0);
}
// One warp owns a row. Eight elements/thread remain in registers across both
// reductions; neighboring lanes access neighboring elements. No shared memory.
template<class Output>
__global__ void ln256(const half* input,const half* gamma,const half* beta,
                      Output* output,unsigned rows,float inverse_scale,
                      const half* input_bias=nullptr,half* residual=nullptr) {
    unsigned row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
    if(row>=rows)return; // Entire warp exits together.
    float x[8],sum=0;
    #pragma unroll
    for(int i=0;i<8;++i){auto idx=size_t(row)*256+lane+i*32;
        x[i]=__half2float(input[idx]);
        if(input_bias)x[i]+=__half2float(input_bias[lane+i*32]);
        if(residual)residual[idx]=__float2half(x[i]);
        sum+=x[i];}
    float mean=warp_sum(sum)/256.f,var=0;
    #pragma unroll
    for(int i=0;i<8;++i){x[i]-=mean;var+=x[i]*x[i];}
    float invstd=rsqrtf(warp_sum(var)/256.f+1.e-5f);
    #pragma unroll
    for(int i=0;i<8;++i){unsigned c=lane+i*32;
        float y=x[i]*invstd*__half2float(gamma[c])+__half2float(beta[c]);
        output[size_t(row)*256+c]=Output(y*inverse_scale);
    }
}
}

inline void hopper_ln256_fp8(const half* input,const half* gamma,const half* beta,
    cutlass::float_e4m3_t* output,unsigned rows,float inverse_scale,cudaStream_t stream,
    const half* input_bias=nullptr,half* residual=nullptr) {
    if(!std::isfinite(inverse_scale)||inverse_scale<=0)throw std::invalid_argument("Invalid FP8 activation scale");
    if(!rows)return;
    hopper_fp8_detail::ln256<<<(rows+3)/4,128,0,stream>>>(input,gamma,beta,output,rows,inverse_scale,input_bias,residual);
}

inline void hopper_fp8_linear(const cutlass::float_e4m3_t* input,
    const cutlass::float_e4m3_t* weight,const half* bias,half* output,
    unsigned rows,unsigned k,unsigned n,float scale_product,cudaStream_t stream) {
    using namespace cute;
    using F8=cutlass::float_e4m3_t;using H=cutlass::half_t;
    using Tile=Shape<_128,_128,_128>;using Cluster=Shape<_1,_2,_1>;
    using LA=cutlass::layout::RowMajor;using LB=cutlass::layout::ColumnMajor;
    using Fusion=cutlass::epilogue::fusion::LinCombPerColBiasEltAct<
        cutlass::epilogue::thread::Identity,H,float,H,void,float>;
    using Epi=typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,
        Shape<_128,_64>,float,float,void,LA,1,H,LA,8,
        cutlass::epilogue::TmaWarpSpecializedCooperative,Fusion>::CollectiveOp;
    using Main=typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,F8,LA,16,F8,LB,16,
        float,Tile,Cluster,cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
    using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Main,Epi>;
    using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    if(!rows)return;
    if(!k||!n||k%16||n%8||!std::isfinite(scale_product)||scale_product<=0)
        throw std::invalid_argument("Invalid Hopper FP8 shape/scale");
    auto sa=cutlass::make_cute_packed_stride(typename Kernel::StrideA{},make_shape(int(rows),int(k),1));
    auto sb=cutlass::make_cute_packed_stride(typename Kernel::StrideB{},make_shape(int(n),int(k),1));
    auto sd=cutlass::make_cute_packed_stride(typename Kernel::StrideD{},make_shape(int(rows),int(n),1));
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
        make_shape(int(rows),int(n),int(k),1),{input,sa,weight,sb},
        {{},nullptr,sd,reinterpret_cast<H*>(output),sd}};
    args.epilogue.thread.alpha=scale_product;args.epilogue.thread.beta=0;
    args.epilogue.thread.bias_ptr=reinterpret_cast<const H*>(bias);
    if(Gemm::get_workspace_size(args))throw std::runtime_error("Unexpected FP8 GEMM workspace");
    Gemm gemm;
    if(gemm(args,nullptr,stream)!=cutlass::Status::kSuccess)throw std::runtime_error("Native Hopper FP8 GEMM failed");
}
}
