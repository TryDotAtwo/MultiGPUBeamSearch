#include "stream1_blend_readout.hpp"
#include "config.hpp"
#if BEAM_HAS_CUTLASS
#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/layout/matrix.h>
#include <cmath>
#include <stdexcept>

namespace beam {
namespace {
// Broadcast epilogue carries FP32 MLP as C and the Transformer bias as V;
// Z stores uint32 key bits through a float iterator (identical 32-bit stores).
// Bitcast, never numeric conversion. Do not quantize either head first.
struct BlendKeys : cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    float, float, float, float, float, 1,
    cutlass::epilogue::thread::Identity<float>, cutlass::plus<float>, false, float> {
    using Base = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
        float,float,float,float,float,1,
        cutlass::epilogue::thread::Identity<float>,cutlass::plus<float>,false,float>;
    struct Params : Base::Params {
        bool transformer_only;
        CUTLASS_HOST_DEVICE explicit Params(bool only=false)
            : Base::Params(only?1.f:0.6f,only?0.f:0.4f), transformer_only(only) {}
    };
    float tr_weight,mlp_weight;
    CUTLASS_HOST_DEVICE explicit BlendKeys(Params const& p) : Base(p), tr_weight(p.transformer_only?1.f:0.6f),mlp_weight(p.transformer_only?0.f:0.4f) {}
    CUTLASS_HOST_DEVICE bool is_source_needed() const { return mlp_weight!=0.f; }
    CUTLASS_HOST_DEVICE void set_k_partition(int, int) {} // Only split_k=1 is launched.
    CUTLASS_DEVICE void operator()(FragmentZ& z, FragmentT&, FragmentAccumulator const& ab,
                                  FragmentC const& c, FragmentCompute const& bias) const {
        float tr = __fadd_rn(ab[0], bias[0]);
        float q = __fadd_rn(__fmul_rn(tr_weight,tr), __fmul_rn(mlp_weight,c[0]));
        if (!isfinite(tr) || !isfinite(c[0]) || !isfinite(q)) {
            z[0] = __uint_as_float(0xffffffffU);
        } else {
            q = fminf(fmaxf(q,0.f),SCORE_MAX_Q);
            z[0] = __uint_as_float(__float2uint_rn(__fmul_rn(q,static_cast<float>(SCORE_SCALE))));
        }
    }
    CUTLASS_DEVICE void operator()(FragmentZ& z, FragmentT& t, FragmentAccumulator const& ab,
                                  FragmentCompute const& bias) const {
        FragmentC zero; zero.clear(); operator()(z,t,ab,zero,bias);
    }
};
// CUTLASS invokes output functors for masked tile lanes too. Only stored,
// logical outputs may raise the sticky flag; padded lanes are not candidates.
__global__ void check_blend_keys(const std::uint32_t* keys,std::uint64_t count,std::uint32_t* error) {
    auto i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count && keys[i]==0xffffffffU) atomicExch(error,1U);
}
}
void stream1_blend_readout_cuda(const float* cls,const float* weight,const float* bias,
    const float* mlp_q,std::uint32_t* keys,std::uint32_t rows,
    std::uint32_t* numeric_error,cudaStream_t stream,bool transformer_only) {
    if (!rows) return;
    if (!cls || !weight || !bias || (!mlp_q && !transformer_only) || !keys || !numeric_error)
        throw std::invalid_argument("blend readout requires all buffers and sticky numeric flag");
    // Ampere three-pass TF32 readout; admission compares FP32 logits and keys.
    using Gemm = cutlass::gemm::device::GemmUniversalWithBroadcast<
        float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,
        float,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,cutlass::gemm::GemmShape<128,32,16>,
        cutlass::gemm::GemmShape<64,32,16>,cutlass::gemm::GemmShape<16,8,8>,BlendKeys,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,4,4,
        cutlass::arch::OpMultiplyAddFastF32>;
    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm,
        {static_cast<int>(rows),24,256},1,BlendKeys::Params(transformer_only),
        cls,weight,mlp_q,keys,const_cast<float*>(bias),nullptr,
        std::int64_t(rows)*256,256*24,std::int64_t(rows)*24,std::int64_t(rows)*24,0,0,
        256,24,24,24,0,24);
    Gemm gemm;
    if (gemm.can_implement(args)!=cutlass::Status::kSuccess ||
        gemm(args,nullptr,stream)!=cutlass::Status::kSuccess)
        throw std::runtime_error("CUTLASS fused blend readout launch failed");
    auto count=std::uint64_t(rows)*24;
    check_blend_keys<<<static_cast<unsigned>((count+255)/256),256,0,stream>>>(keys,count,numeric_error);
    if(cudaGetLastError()!=cudaSuccess) throw std::runtime_error("blend numeric guard launch failed");
}
void stream1_blend_readout_fp16_cuda(const __half* cls,const __half* weight,const float* bias,
    const float* mlp_q,std::uint32_t* keys,std::uint32_t rows,
    std::uint32_t* numeric_error,cudaStream_t stream,bool transformer_only) {
    if (!rows) return;
    if (!cls || !weight || !bias || (!mlp_q && !transformer_only) || !keys || !numeric_error)
        throw std::invalid_argument("blend readout requires all buffers and sticky numeric flag");
    // FP16 Tensor Core operands, FP32 accumulation and blend.
    using Gemm = cutlass::gemm::device::GemmUniversalWithBroadcast<
        cutlass::half_t,cutlass::layout::RowMajor,cutlass::half_t,cutlass::layout::RowMajor,
        float,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,cutlass::gemm::GemmShape<128,32,32>,
        cutlass::gemm::GemmShape<64,32,32>,cutlass::gemm::GemmShape<16,8,16>,BlendKeys,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,8,8,
        cutlass::arch::OpMultiplyAdd>;
    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm,
        {static_cast<int>(rows),24,256},1,BlendKeys::Params(transformer_only),
        reinterpret_cast<const cutlass::half_t*>(cls),reinterpret_cast<const cutlass::half_t*>(weight),mlp_q,keys,const_cast<float*>(bias),nullptr,
        std::int64_t(rows)*256,256*24,std::int64_t(rows)*24,std::int64_t(rows)*24,0,0,
        256,24,24,24,0,24);
    Gemm gemm;
    if (gemm.can_implement(args)!=cutlass::Status::kSuccess ||
        gemm(args,nullptr,stream)!=cutlass::Status::kSuccess)
        throw std::runtime_error("CUTLASS fused blend readout launch failed");
    auto count=std::uint64_t(rows)*24;
    check_blend_keys<<<static_cast<unsigned>((count+255)/256),256,0,stream>>>(keys,count,numeric_error);
    if(cudaGetLastError()!=cudaSuccess) throw std::runtime_error("blend numeric guard launch failed");
}
}
#else
#include <stdexcept>
namespace beam {
void stream1_blend_readout_cuda(const float*,const float*,const float*,const float*,
    std::uint32_t*,std::uint32_t,std::uint32_t*,cudaStream_t,bool) {
    throw std::runtime_error("fused blend readout requires CUTLASS-enabled build");
}
void stream1_blend_readout_fp16_cuda(const __half*,const __half*,const float*,const float*,
    std::uint32_t*,std::uint32_t,std::uint32_t*,cudaStream_t,bool) {
    throw std::runtime_error("fused blend readout requires CUTLASS-enabled build");
}

}
#endif
