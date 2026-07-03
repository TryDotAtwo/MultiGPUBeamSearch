#include "stream1_transformer_fmha.hpp"

#include "config.hpp"
#include "cuda_check.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if BEAM_HAS_CUTLASS
#include <cutlass/bfloat16.h>
#include <cutlass/numeric_types.h>
#if BEAM_HAS_CUTLASS_FMHA
#include "kernel_forward.h"
#endif
#endif

#include <stdexcept>

namespace beam {

constexpr std::uint32_t FMHA_SEQ51 = 51U;
constexpr std::uint32_t FMHA_DMODEL256 = 256U;
constexpr std::uint32_t FMHA_HEAD_DIM32 = 32U;
constexpr std::uint32_t FMHA_NHEAD8 = 8U;
constexpr std::uint32_t FMHA_QKV_STRIDE51 = 3U * FMHA_DMODEL256;

__device__ float stream1_fmha_load_scalar_device(const half* ptr, std::uint64_t idx, std::uint32_t dtype) {
    if (dtype == STREAM1_DTYPE_BF16) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[idx]);
    }
    return __half2float(ptr[idx]);
}

__device__ void stream1_fmha_store_scalar_device(half* ptr, std::uint64_t idx, float value, std::uint32_t dtype) {
    if (dtype == STREAM1_DTYPE_BF16) {
        reinterpret_cast<__nv_bfloat16*>(ptr)[idx] = __float2bfloat16(value);
        return;
    }
    ptr[idx] = __float2half(value);
}

__global__ void stream1_transformer_fmha_pack_qkv_bmhk51_kernel(
    const half* __restrict__ qkv,
    half* __restrict__ packed_qkv,
    std::uint32_t dtype,
    std::uint32_t b_micro) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    constexpr std::uint64_t matrix_values = static_cast<std::uint64_t>(FMHA_SEQ51) * FMHA_NHEAD8 * FMHA_HEAD_DIM32;
    const std::uint64_t total = static_cast<std::uint64_t>(b_micro) * 3ULL * matrix_values;
    if (i >= total) {
        return;
    }
    const std::uint64_t per_kind_values = static_cast<std::uint64_t>(b_micro) * matrix_values;
    const std::uint32_t kind = static_cast<std::uint32_t>(i / per_kind_values);
    const std::uint64_t local = i - static_cast<std::uint64_t>(kind) * per_kind_values;
    const std::uint32_t dim = static_cast<std::uint32_t>(local % FMHA_HEAD_DIM32);
    const std::uint32_t head = static_cast<std::uint32_t>((local / FMHA_HEAD_DIM32) % FMHA_NHEAD8);
    const std::uint32_t token = static_cast<std::uint32_t>((local / (FMHA_HEAD_DIM32 * FMHA_NHEAD8)) % FMHA_SEQ51);
    const std::uint32_t row = static_cast<std::uint32_t>(local / matrix_values);
    const std::uint64_t qkv_col = static_cast<std::uint64_t>(kind) * FMHA_DMODEL256 +
        static_cast<std::uint64_t>(head) * FMHA_HEAD_DIM32 + dim;
    const std::uint64_t src_idx = static_cast<std::uint64_t>(row) * FMHA_SEQ51 * FMHA_QKV_STRIDE51 +
        static_cast<std::uint64_t>(token) * FMHA_QKV_STRIDE51 + qkv_col;
    const float value = stream1_fmha_load_scalar_device(qkv, src_idx, dtype);
    stream1_fmha_store_scalar_device(packed_qkv, i, value, dtype);
}

#if BEAM_HAS_CUTLASS && BEAM_HAS_CUTLASS_FMHA
template <typename Element, typename ArchTag>
void stream1_transformer_fmha_launch_typed(
    half* packed_qkv,
    half* context,
    std::uint32_t b_micro,
    cudaStream_t stream) {
    using Attention = AttentionKernel<
        Element,
        ArchTag,
        true,
        64,
        64,
        64,
        false,
        false>;
    constexpr std::uint64_t matrix_values = static_cast<std::uint64_t>(FMHA_SEQ51) * FMHA_NHEAD8 * FMHA_HEAD_DIM32;
    typename Attention::Params params{};
    params.query_ptr = reinterpret_cast<Element*>(packed_qkv);
    params.key_ptr = reinterpret_cast<Element*>(packed_qkv + matrix_values * b_micro);
    params.value_ptr = reinterpret_cast<Element*>(packed_qkv + 2ULL * matrix_values * b_micro);
    params.output_ptr = reinterpret_cast<Element*>(context);
    params.output_accum_ptr = nullptr;
    params.logsumexp_ptr = nullptr;
    params.scale = 0.1767766952966369f;
    params.num_heads = static_cast<int32_t>(FMHA_NHEAD8);
    params.num_batches = static_cast<int32_t>(b_micro);
    params.head_dim = static_cast<int32_t>(FMHA_HEAD_DIM32);
    params.head_dim_value = static_cast<int32_t>(FMHA_HEAD_DIM32);
    params.num_queries = static_cast<int32_t>(FMHA_SEQ51);
    params.num_keys = static_cast<int32_t>(FMHA_SEQ51);
    params.custom_mask_type = Attention::NoCustomMask;
    params.q_strideH = static_cast<int32_t>(FMHA_HEAD_DIM32);
    params.k_strideH = static_cast<int32_t>(FMHA_HEAD_DIM32);
    params.v_strideH = static_cast<int32_t>(FMHA_HEAD_DIM32);
    params.q_strideM = static_cast<int32_t>(FMHA_DMODEL256);
    params.k_strideM = static_cast<int32_t>(FMHA_DMODEL256);
    params.v_strideM = static_cast<int32_t>(FMHA_DMODEL256);
    params.q_strideB = static_cast<int64_t>(FMHA_SEQ51 * FMHA_DMODEL256);
    params.k_strideB = static_cast<int64_t>(FMHA_SEQ51 * FMHA_DMODEL256);
    params.v_strideB = static_cast<int64_t>(FMHA_SEQ51 * FMHA_DMODEL256);
    params.o_strideM = static_cast<int32_t>(FMHA_DMODEL256);
    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    const int smem_bytes = static_cast<int>(sizeof(typename Attention::SharedStorage));
    if (smem_bytes > 0xc000) {
        BEAM_CUDA_CHECK(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }
    if (!Attention::check_supported(params)) {
        throw std::runtime_error("Stream1 piece_transformer CUTLASS FMHA does not support this tensor layout");
    }
    kernel_fn<<<params.getBlocksGrid(), params.getThreadsGrid(), smem_bytes, stream>>>(params);
}
#endif

void stream1_transformer_fmha_attention_cuda(
    half* qkv,
    half* packed_qkv,
    half* context,
    Stream1TransformerDims dims,
    std::uint32_t b_micro,
    cudaStream_t stream) {
    if (dims.seq_len != FMHA_SEQ51 || dims.d_model != FMHA_DMODEL256 ||
        dims.nhead != FMHA_NHEAD8 || dims.head_dim != FMHA_HEAD_DIM32) {
        throw std::invalid_argument("Stream1 piece_transformer CUTLASS FMHA requires seq_len=51 d_model=256 nhead=8 head_dim=32");
    }
    if (qkv == nullptr || packed_qkv == nullptr || context == nullptr) {
        throw std::invalid_argument("Stream1 piece_transformer CUTLASS FMHA requires qkv, packed_qkv, and context");
    }
#if BEAM_HAS_CUTLASS && BEAM_HAS_CUTLASS_FMHA
    const std::uint64_t qkv_total = static_cast<std::uint64_t>(b_micro) * FMHA_SEQ51 * FMHA_QKV_STRIDE51;
    stream1_transformer_fmha_pack_qkv_bmhk51_kernel<<<
        static_cast<unsigned>((qkv_total + 255ULL) / 256ULL),
        256,
        0,
        stream>>>(qkv, packed_qkv, dims.dtype, b_micro);
    if (dims.dtype == STREAM1_DTYPE_BF16) {
        stream1_transformer_fmha_launch_typed<cutlass::bfloat16_t, cutlass::arch::Sm80>(packed_qkv, context, b_micro, stream);
        return;
    }
    if (dims.dtype == STREAM1_DTYPE_FP16) {
        stream1_transformer_fmha_launch_typed<cutlass::half_t, cutlass::arch::Sm80>(packed_qkv, context, b_micro, stream);
        return;
    }
    throw std::invalid_argument("Stream1 piece_transformer CUTLASS FMHA dtype must be fp16 or bf16");
#else
    (void)qkv;
    (void)packed_qkv;
    (void)context;
    (void)dims;
    (void)b_micro;
    (void)stream;
    throw std::invalid_argument("Stream1 piece_transformer CUTLASS FMHA requires CUTLASS example 41 headers");
#endif
}

} // namespace beam