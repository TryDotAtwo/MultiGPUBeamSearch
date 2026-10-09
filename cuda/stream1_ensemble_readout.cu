#include "stream1_blend_readout.hpp"
#include "config.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>
#include <climits>
#if BEAM_HAS_CUTLASS
#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/layout/matrix.h>

namespace beam {
namespace {
struct EnsembleReadout : cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    float,float,float,float,float,1,cutlass::epilogue::thread::Identity<float>,
    cutlass::plus<float>,false,float> {
    using Base=cutlass::epilogue::thread::LinearCombinationBiasElementwise<
        float,float,float,float,float,1,cutlass::epilogue::thread::Identity<float>,
        cutlass::plus<float>,false,float>;
    struct Params : Base::Params {
        float coefficient;
        bool initialize,finalize;
        CUTLASS_HOST_DEVICE Params(float a=1,bool init=true,bool last=true)
            : Base::Params(a,init?0.f:1.f),coefficient(a),initialize(init),finalize(last) {}
    };
    float coefficient;
    bool initialize,finalize;
    CUTLASS_HOST_DEVICE explicit EnsembleReadout(Params const& p)
        : Base(p),coefficient(p.coefficient),initialize(p.initialize),finalize(p.finalize) {}
    CUTLASS_HOST_DEVICE bool is_source_needed() const {return !initialize;}
    CUTLASS_HOST_DEVICE void set_k_partition(int,int) {}
    CUTLASS_DEVICE void operator()(FragmentZ& z,FragmentT&,FragmentAccumulator const& ab,
                                  FragmentC const& c,FragmentCompute const& bias) const {
        float head=__fadd_rn(ab[0],bias[0]);
        float q=__fadd_rn(__fmul_rn(coefficient,head),initialize?0.f:c[0]);
        if(finalize) {
            z[0]=(!isfinite(head)||!isfinite(q))?__uint_as_float(0xffffffffU):
                __uint_as_float(__float2uint_rn(__fmul_rn(
                    fminf(fmaxf(q,0.f),SCORE_MAX_Q),float(SCORE_SCALE))));
        } else {
            z[0]=(!isfinite(head)||!isfinite(q))?__int_as_float(0x7fffffff):q;
        }
    }
    CUTLASS_DEVICE void operator()(FragmentZ& z,FragmentT& t,FragmentAccumulator const& ab,
                                  FragmentCompute const& bias) const {
        FragmentC zero;zero.clear();operator()(z,t,ab,zero,bias);
    }
};
__global__ void check_ensemble(const void* output,std::uint64_t count,bool final,std::uint32_t* error) {
    auto i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count && (final?static_cast<const std::uint32_t*>(output)[i]==0xffffffffU:
                        !isfinite(static_cast<const float*>(output)[i]))) atomicExch(error,1U);
}
}
void stream1_ensemble_head_fp16_cuda(const __half* a,const __half* w,const float* bias,
    float* accumulator,std::uint32_t* keys,std::uint32_t rows,std::uint32_t hidden,
    std::uint32_t outputs,float coefficient,bool initialize,bool finalize,
    std::uint32_t* error,cudaStream_t stream) {
    if(!rows) return;
    if(!a||!w||!bias||!accumulator||!error||(finalize&&!keys)||!hidden||!outputs||
       hidden%8||rows>std::uint32_t(INT_MAX)||hidden>std::uint32_t(INT_MAX)||
       outputs>std::uint32_t(INT_MAX)||!std::isfinite(coefficient))
        throw std::invalid_argument("invalid aligned FP16 ensemble readout contract");
    using GemmAligned=cutlass::gemm::device::GemmUniversalWithBroadcast<
        cutlass::half_t,cutlass::layout::RowMajor,cutlass::half_t,cutlass::layout::RowMajor,
        float,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,cutlass::gemm::GemmShape<128,32,32>,
        cutlass::gemm::GemmShape<64,32,32>,cutlass::gemm::GemmShape<16,8,16>,EnsembleReadout,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,8,8,
        cutlass::arch::OpMultiplyAdd>;
    using GemmScalar=cutlass::gemm::device::GemmUniversalWithBroadcast<
        cutlass::half_t,cutlass::layout::RowMajor,cutlass::half_t,cutlass::layout::RowMajor,
        float,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,cutlass::gemm::GemmShape<128,32,32>,
        cutlass::gemm::GemmShape<64,32,32>,cutlass::gemm::GemmShape<16,8,16>,EnsembleReadout,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,8,1,
        cutlass::arch::OpMultiplyAdd>;
    auto destination=finalize?static_cast<void*>(keys):static_cast<void*>(accumulator);
    auto launch=[&](auto gemm) {
    using Gemm=decltype(gemm);
    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm,
        {int(rows),int(outputs),int(hidden)},1,EnsembleReadout::Params(coefficient,initialize,finalize),
        reinterpret_cast<const cutlass::half_t*>(a),reinterpret_cast<const cutlass::half_t*>(w),
        accumulator,destination,const_cast<float*>(bias),nullptr,
        std::int64_t(rows)*hidden,std::int64_t(hidden)*outputs,
        std::int64_t(rows)*outputs,std::int64_t(rows)*outputs,0,0,
        hidden,outputs,outputs,outputs,0,outputs);
    if(gemm.can_implement(args)!=cutlass::Status::kSuccess||
       gemm(args,nullptr,stream)!=cutlass::Status::kSuccess)
        throw std::runtime_error("CUTLASS ensemble readout launch failed");
    };
    if(outputs%8==0) launch(GemmAligned{});else launch(GemmScalar{});
    auto count=std::uint64_t(rows)*outputs;
    check_ensemble<<<unsigned((count+255)/256),256,0,stream>>>(destination,count,finalize,error);
    if(cudaGetLastError()!=cudaSuccess) throw std::runtime_error("ensemble numeric guard launch failed");
}
}
#else
namespace beam {
void stream1_ensemble_head_fp16_cuda(const __half*,const __half*,const float*,float*,
    std::uint32_t*,std::uint32_t,std::uint32_t,std::uint32_t,float,bool,bool,
    std::uint32_t*,cudaStream_t) {
    throw std::runtime_error("ensemble readout requires CUTLASS");
}
}
#endif
