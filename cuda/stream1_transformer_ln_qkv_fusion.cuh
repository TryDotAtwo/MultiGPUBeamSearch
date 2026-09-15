#pragma once
// Experimental only: numerically checked on H200, but ~14x slower than the
// existing two-kernel LN+Hopper-QKV stage. Not used by production inference.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cutlass/numeric_types.h>

namespace beam {
namespace ln_qkv_fusion {
using E = cutlass::half_t;
using ALayout = decltype(cute::tile_to_shape(cute::GMMA::Layout_K_SW128_Atom<E>{}, cute::Shape<cute::_64,cute::_256>{}));
using BLayout = decltype(cute::tile_to_shape(cute::GMMA::Layout_K_SW128_Atom<E>{}, cute::Shape<cute::_128,cute::_256>{}));
struct Shared {
    alignas(128) E a[cute::cosize_v<ALayout>];
    alignas(128) E b[cute::cosize_v<BLayout>];
};
__device__ __forceinline__ float warp_sum(float x) {
    for(int offset=16;offset>0;offset/=2)x+=__shfl_down_sync(0xffffffff,x,offset);
    return __shfl_sync(0xffffffff,x,0);
}
// Match the incumbent's 128-thread/two-values-per-thread reduction tree.
// Four logical warps are evaluated by one physical warp without CTA barriers.
__device__ __forceinline__ float sum256(const float (&x)[8]) {
    float s0=warp_sum(x[0]+x[4]),s1=warp_sum(x[1]+x[5]);
    float s2=warp_sum(x[2]+x[6]),s3=warp_sum(x[3]+x[7]);
    return (s0+s2)+(s1+s3);
}
__global__ __launch_bounds__(128) void kernel(
    const half* input,const half* gamma,const half* beta,const half* weights,
    const half* bias,half* output,std::uint32_t rows) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using namespace cute;
    extern __shared__ __align__(128) unsigned char storage[];
    auto& smem=*reinterpret_cast<Shared*>(storage);
    auto a=make_tensor(make_smem_ptr(smem.a),ALayout{});
    auto b=make_tensor(make_smem_ptr(smem.b),BLayout{});
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    for(int r=warp;r<64;r+=4) {
        const unsigned row=blockIdx.x*64+r;
        float x[8];
        #pragma unroll
        for(int j=0;j<8;++j)x[j]=row<rows?__half2float(input[std::size_t(row)*256+lane+32*j]):0.0f;
        const float mean=sum256(x)*(1.0f/256.0f);
        float c[8];
        #pragma unroll
        for(int j=0;j<8;++j)c[j]=x[j]-mean;
        float v0=warp_sum(c[0]*c[0]+c[4]*c[4]);
        float v1=warp_sum(c[1]*c[1]+c[5]*c[5]);
        float v2=warp_sum(c[2]*c[2]+c[6]*c[6]);
        float v3=warp_sum(c[3]*c[3]+c[7]*c[7]);
        const float inv=rsqrtf(((v0+v2)+(v1+v3))*(1.0f/256.0f)+1.0e-5f);
        #pragma unroll
        for(int j=0;j<8;++j) {
            const int k=lane+32*j;
            a(r,k)=E(c[j]*inv*__half2float(gamma[k])+__half2float(beta[k]));
        }
    }
    for(int i=threadIdx.x;i<128*256;i+=128) {
        const int n=i/256,k=i%256;
        b(n,k)=reinterpret_cast<const E*>(weights)[(blockIdx.y*128+n)*256+k];
    }
    __syncthreads();
    // Publish ordinary shared stores to WGMMA's asynchronous proxy.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    auto mma=make_tiled_mma(SM90_64x128x16_F32F16F16_SS<GMMA::Major::K,GMMA::Major::K>{});
    auto thr=mma.get_slice(threadIdx.x);
    auto sa=thr.partition_A(a);
    auto sb=thr.partition_B(b);
    auto fa=thr.make_fragment_A(sa);
    auto fb=thr.make_fragment_B(sb);
    auto coords=thr.partition_C(make_identity_tensor(Shape<_64,_128>{}));
    auto accum=thr.make_fragment_C(coords);
    clear(accum);
    warpgroup_arrive();
    gemm(mma,fa,fb,accum);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    #pragma unroll
    for(int i=0;i<size(accum);++i) {
        const unsigned r=blockIdx.x*64+get<0>(coords(i));
        const unsigned n=blockIdx.y*128+get<1>(coords(i));
        if(r<rows)output[std::size_t(r)*768+n]=__float2half(float(accum(i))+__half2float(bias[n]));
    }
#else
    asm volatile("trap;");
#endif
}
}

// Fixed FP16 D256 -> QKV768, packed [N,K] weights. Input is immutable:
// several N tiles read each row, so updating residuals here would be a race.
inline void stream1_transformer_ln_qkv_fused(
    const half* input,const half* gamma,const half* beta,const half* weights,
    const half* bias,half* output,std::uint32_t rows,cudaStream_t stream) {
    if(rows==0)return;
    int device=0;cudaGetDevice(&device);
    static thread_local int prepared=-1;
    if(prepared!=device) {
        cudaDeviceProp prop{};cudaGetDeviceProperties(&prop,device);
        if(prop.major!=9)throw std::invalid_argument("LN-QKV fusion requires Hopper SM90");
        auto status=cudaFuncSetAttribute(ln_qkv_fusion::kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(ln_qkv_fusion::Shared));
        if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
        prepared=device;
    }
    ln_qkv_fusion::kernel<<<dim3((rows+63)/64,6),128,sizeof(ln_qkv_fusion::Shared),stream>>>(input,gamma,beta,weights,bias,output,rows);
    auto status=cudaGetLastError();
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
}
