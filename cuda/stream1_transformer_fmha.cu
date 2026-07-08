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

#if BEAM_HAS_CUTLASS && BEAM_HAS_CUTLASS_FMHA
template <typename Element, typename ArchTag>
void stream1_transformer_fmha_launch_typed(
    half* qkv,
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
    typename Attention::Params params{};
    params.query_ptr = reinterpret_cast<Element*>(qkv);
    params.key_ptr = reinterpret_cast<Element*>(qkv + FMHA_DMODEL256);
    params.value_ptr = reinterpret_cast<Element*>(qkv + 2ULL * FMHA_DMODEL256);
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
    params.q_strideM = static_cast<int32_t>(FMHA_QKV_STRIDE51);
    params.k_strideM = static_cast<int32_t>(FMHA_QKV_STRIDE51);
    params.v_strideM = static_cast<int32_t>(FMHA_QKV_STRIDE51);
    params.q_strideB = static_cast<int64_t>(FMHA_SEQ51 * FMHA_QKV_STRIDE51);
    params.k_strideB = static_cast<int64_t>(FMHA_SEQ51 * FMHA_QKV_STRIDE51);
    params.v_strideB = static_cast<int64_t>(FMHA_SEQ51 * FMHA_QKV_STRIDE51);
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

template <typename Element, typename ArchTag>
void stream1_transformer_fmha_cls_launch_typed(
    half* qkv,
    half* cls_context,
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
    typename Attention::Params params{};
    params.query_ptr = reinterpret_cast<Element*>(qkv);
    params.key_ptr = reinterpret_cast<Element*>(qkv + FMHA_DMODEL256);
    params.value_ptr = reinterpret_cast<Element*>(qkv + 2ULL * FMHA_DMODEL256);
    params.output_ptr = reinterpret_cast<Element*>(cls_context);
    params.output_accum_ptr = nullptr;
    params.logsumexp_ptr = nullptr;
    params.scale = 0.1767766952966369f;
    params.num_heads = static_cast<int32_t>(FMHA_NHEAD8);
    params.num_batches = static_cast<int32_t>(b_micro);
    params.head_dim = static_cast<int32_t>(FMHA_HEAD_DIM32);
    params.head_dim_value = static_cast<int32_t>(FMHA_HEAD_DIM32);
    params.num_queries = 1;
    params.num_keys = static_cast<int32_t>(FMHA_SEQ51);
    params.custom_mask_type = Attention::NoCustomMask;
    params.q_strideH = static_cast<int32_t>(FMHA_HEAD_DIM32);
    params.k_strideH = static_cast<int32_t>(FMHA_HEAD_DIM32);
    params.v_strideH = static_cast<int32_t>(FMHA_HEAD_DIM32);
    params.q_strideM = static_cast<int32_t>(FMHA_QKV_STRIDE51);
    params.k_strideM = static_cast<int32_t>(FMHA_QKV_STRIDE51);
    params.v_strideM = static_cast<int32_t>(FMHA_QKV_STRIDE51);
    params.q_strideB = static_cast<int64_t>(FMHA_SEQ51 * FMHA_QKV_STRIDE51);
    params.k_strideB = static_cast<int64_t>(FMHA_SEQ51 * FMHA_QKV_STRIDE51);
    params.v_strideB = static_cast<int64_t>(FMHA_SEQ51 * FMHA_QKV_STRIDE51);
    params.o_strideM = static_cast<int32_t>(FMHA_DMODEL256);
    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    const int smem_bytes = static_cast<int>(sizeof(typename Attention::SharedStorage));
    if (smem_bytes > 0xc000) {
        BEAM_CUDA_CHECK(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }
    if (!Attention::check_supported(params)) {
        throw std::runtime_error("Stream1 piece_transformer CUTLASS CLS FMHA does not support this tensor layout");
    }
    kernel_fn<<<params.getBlocksGrid(), params.getThreadsGrid(), smem_bytes, stream>>>(params);
}
#endif

void stream1_transformer_fmha_attention_cuda(
    half* qkv,
    half* packed_qkv,
    half* context,
    Stream1TransformerDims dims,
    bool sm75_fp16,
    std::uint32_t b_micro,
    cudaStream_t stream) {
    if (dims.seq_len != FMHA_SEQ51 || dims.d_model != FMHA_DMODEL256 ||
        dims.nhead != FMHA_NHEAD8 || dims.head_dim != FMHA_HEAD_DIM32) {
        throw std::invalid_argument("Stream1 piece_transformer CUTLASS FMHA requires seq_len=51 d_model=256 nhead=8 head_dim=32");
    }
    if (qkv == nullptr || context == nullptr) {
        throw std::invalid_argument("Stream1 piece_transformer CUTLASS FMHA requires qkv and context");
    }
#if BEAM_HAS_CUTLASS && BEAM_HAS_CUTLASS_FMHA
    (void)packed_qkv;
    if (sm75_fp16) {
        if (dims.dtype != STREAM1_DTYPE_FP16) {
            throw std::invalid_argument("Stream1 piece_transformer SM75 CUTLASS FMHA requires fp16");
        }
        stream1_transformer_fmha_launch_typed<cutlass::half_t, cutlass::arch::Sm75>(qkv, context, b_micro, stream);
        return;
    }
    if (dims.dtype == STREAM1_DTYPE_BF16) {
        stream1_transformer_fmha_launch_typed<cutlass::bfloat16_t, cutlass::arch::Sm80>(qkv, context, b_micro, stream);
        return;
    }
    if (dims.dtype == STREAM1_DTYPE_FP16) {
        stream1_transformer_fmha_launch_typed<cutlass::half_t, cutlass::arch::Sm80>(qkv, context, b_micro, stream);
        return;
    }
    throw std::invalid_argument("Stream1 piece_transformer CUTLASS FMHA dtype must be fp16 or bf16");
#else
    (void)qkv;
    (void)packed_qkv;
    (void)context;
    (void)dims;
    (void)sm75_fp16;
    (void)b_micro;
    (void)stream;
    throw std::invalid_argument("Stream1 piece_transformer CUTLASS FMHA requires CUTLASS example 41 headers");
#endif
}


void stream1_transformer_fmha_cls_attention_cuda(
    half* qkv,
    half* cls_context,
    Stream1TransformerDims dims,
    bool sm75_fp16,
    std::uint32_t b_micro,
    cudaStream_t stream) {
    if (dims.seq_len != FMHA_SEQ51 || dims.d_model != FMHA_DMODEL256 ||
        dims.nhead != FMHA_NHEAD8 || dims.head_dim != FMHA_HEAD_DIM32) {
        throw std::invalid_argument("Stream1 piece_transformer CUTLASS CLS FMHA requires seq_len=51 d_model=256 nhead=8 head_dim=32");
    }
    if (qkv == nullptr || cls_context == nullptr) {
        throw std::invalid_argument("Stream1 piece_transformer CUTLASS CLS FMHA requires qkv and cls_context");
    }
#if BEAM_HAS_CUTLASS && BEAM_HAS_CUTLASS_FMHA
    if (sm75_fp16) {
        if (dims.dtype != STREAM1_DTYPE_FP16) {
            throw std::invalid_argument("Stream1 piece_transformer SM75 CUTLASS CLS FMHA requires fp16");
        }
        stream1_transformer_fmha_cls_launch_typed<cutlass::half_t, cutlass::arch::Sm75>(qkv, cls_context, b_micro, stream);
        return;
    }
    if (dims.dtype == STREAM1_DTYPE_BF16) {
        stream1_transformer_fmha_cls_launch_typed<cutlass::bfloat16_t, cutlass::arch::Sm80>(qkv, cls_context, b_micro, stream);
        return;
    }
    if (dims.dtype == STREAM1_DTYPE_FP16) {
        stream1_transformer_fmha_cls_launch_typed<cutlass::half_t, cutlass::arch::Sm80>(qkv, cls_context, b_micro, stream);
        return;
    }
    throw std::invalid_argument("Stream1 piece_transformer CUTLASS CLS FMHA dtype must be fp16 or bf16");
#else
    (void)qkv;
    (void)cls_context;
    (void)dims;
    (void)sm75_fp16;
    (void)b_micro;
    (void)stream;
    throw std::invalid_argument("Stream1 piece_transformer CUTLASS CLS FMHA requires CUTLASS example 41 headers");
#endif
}
} // namespace beam
