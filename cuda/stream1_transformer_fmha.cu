#include "stream1_transformer_fmha.hpp"
#include "stream1_transformer_attention_policy.hpp"

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

#include <cstdlib>
#include <stdexcept>

namespace beam {

#if BEAM_HAS_CUTLASS && BEAM_HAS_CUTLASS_FMHA
template <typename Element, typename ArchTag, int QueriesPerBlock, int KeysPerBlock, int MaxK, bool IsAligned = true>
void stream1_transformer_fmha_launch_typed(
    half* qkv,
    half* context,
    Stream1TransformerDims dims,
    std::uint32_t b_micro,
    cudaStream_t stream) {
    using Attention = AttentionKernel<
        Element,
        ArchTag,
        IsAligned,
        QueriesPerBlock,
        KeysPerBlock,
        MaxK,
        false,
        false>;
    typename Attention::Params params{};
    params.query_ptr = reinterpret_cast<Element*>(qkv);
    params.key_ptr = reinterpret_cast<Element*>(qkv + dims.d_model);
    params.value_ptr = reinterpret_cast<Element*>(qkv + 2ULL * dims.d_model);
    params.output_ptr = reinterpret_cast<Element*>(context);
    params.output_accum_ptr = nullptr;
    params.logsumexp_ptr = nullptr;
    params.scale = 0.1767766952966369f;
    params.num_heads = static_cast<int32_t>(dims.nhead);
    params.num_batches = static_cast<int32_t>(b_micro);
    params.head_dim = static_cast<int32_t>(dims.head_dim);
    params.head_dim_value = static_cast<int32_t>(dims.head_dim);
    params.num_queries = static_cast<int32_t>(dims.padded_seq_len);
    params.num_keys = static_cast<int32_t>(dims.seq_len);
    params.custom_mask_type = Attention::NoCustomMask;
    params.q_strideH = static_cast<int32_t>(dims.head_dim);
    params.k_strideH = static_cast<int32_t>(dims.head_dim);
    params.v_strideH = static_cast<int32_t>(dims.head_dim);
    params.q_strideM = static_cast<int32_t>((3U * dims.d_model));
    params.k_strideM = static_cast<int32_t>((3U * dims.d_model));
    params.v_strideM = static_cast<int32_t>((3U * dims.d_model));
    params.q_strideB = static_cast<int64_t>(dims.padded_seq_len * (3U * dims.d_model));
    params.k_strideB = static_cast<int64_t>(dims.padded_seq_len * (3U * dims.d_model));
    params.v_strideB = static_cast<int64_t>(dims.padded_seq_len * (3U * dims.d_model));
    params.o_strideM = static_cast<int32_t>(dims.d_model);
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

template <typename Element, typename ArchTag, int QueriesPerBlock, int KeysPerBlock, bool IsAligned = true>
void stream1_transformer_fmha_launch_policy(
    half* qkv,
    half* context,
    Stream1TransformerDims dims,
    std::uint32_t b_micro,
    cudaStream_t stream,
    Stream1TransformerAttentionMaxKPolicy max_k_policy) {
    if (max_k_policy == Stream1TransformerAttentionMaxKPolicy::Exact32) {
        stream1_transformer_fmha_launch_typed<Element, ArchTag, QueriesPerBlock, KeysPerBlock, 32, IsAligned>(
            qkv, context, dims, b_micro, stream);
        return;
    }
    stream1_transformer_fmha_launch_typed<Element, ArchTag, QueriesPerBlock, KeysPerBlock, 64, IsAligned>(
        qkv, context, dims, b_micro, stream);
}

template <typename Element, typename ArchTag, int QueriesPerBlock, int KeysPerBlock>
void stream1_transformer_fmha_cls_launch_typed(
    half* cls_query,
    half* qkv,
    half* cls_context,
    Stream1TransformerDims dims,
    std::uint32_t b_micro,
    cudaStream_t stream) {
    using Attention = AttentionKernel<
        Element,
        ArchTag,
        true,
        QueriesPerBlock,
        KeysPerBlock,
        64,
        false,
        false>;
    typename Attention::Params params{};
    const bool split_query = cls_query != nullptr;
    params.query_ptr = reinterpret_cast<Element*>(split_query ? cls_query : qkv);
    params.key_ptr = reinterpret_cast<Element*>(qkv + dims.d_model);
    params.value_ptr = reinterpret_cast<Element*>(qkv + 2ULL * dims.d_model);
    params.output_ptr = reinterpret_cast<Element*>(cls_context);
    params.output_accum_ptr = nullptr;
    params.logsumexp_ptr = nullptr;
    params.scale = 0.1767766952966369f;
    params.num_heads = static_cast<int32_t>(dims.nhead);
    params.num_batches = static_cast<int32_t>(b_micro);
    params.head_dim = static_cast<int32_t>(dims.head_dim);
    params.head_dim_value = static_cast<int32_t>(dims.head_dim);
    params.num_queries = 1;
    params.num_keys = static_cast<int32_t>(dims.seq_len);
    params.custom_mask_type = Attention::NoCustomMask;
    params.q_strideH = static_cast<int32_t>(dims.head_dim);
    params.k_strideH = static_cast<int32_t>(dims.head_dim);
    params.v_strideH = static_cast<int32_t>(dims.head_dim);
    params.q_strideM = static_cast<int32_t>(split_query ? dims.d_model : (3U * dims.d_model));
    params.k_strideM = static_cast<int32_t>((3U * dims.d_model));
    params.v_strideM = static_cast<int32_t>((3U * dims.d_model));
    params.q_strideB = static_cast<int64_t>(
        split_query ? dims.d_model : dims.padded_seq_len * (3U * dims.d_model));
    params.k_strideB = static_cast<int64_t>(dims.padded_seq_len * (3U * dims.d_model));
    params.v_strideB = static_cast<int64_t>(dims.padded_seq_len * (3U * dims.d_model));
    params.o_strideM = static_cast<int32_t>(dims.d_model);
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
    if (dims.seq_len == 0U || dims.padded_seq_len < dims.seq_len ||
        dims.d_model == 0U || dims.d_model != dims.nhead * dims.head_dim ||
        dims.padded_seq_len > 64U) {
        throw std::invalid_argument("Stream1 piece_transformer CUTLASS FMHA received unsupported logical/padded dimensions");
    }
    if (qkv == nullptr || context == nullptr) {
        throw std::invalid_argument("Stream1 piece_transformer CUTLASS FMHA requires qkv and context");
    }
#if BEAM_HAS_CUTLASS && BEAM_HAS_CUTLASS_FMHA
    (void)packed_qkv;
    const auto max_k_policy = parse_stream1_transformer_attention_max_k_policy(
        std::getenv("BEAM_STREAM1_TRANSFORMER_ATTENTION_MAX_K_POLICY"));
    if (sm75_fp16) {
        if (dims.dtype != STREAM1_DTYPE_FP16) {
            throw std::invalid_argument("Stream1 piece_transformer SM75 CUTLASS FMHA requires fp16");
        }
        const auto policy = parse_stream1_transformer_attention_tile_policy(std::getenv("BEAM_STREAM1_TRANSFORMER_ATTENTION_TILE_POLICY"));
        if (policy == Stream1TransformerAttentionTilePolicy::Q64K64V4) {
            stream1_transformer_fmha_launch_policy<cutlass::half_t, cutlass::arch::Sm75, 64, 64, false>(qkv, context, dims, b_micro, stream, max_k_policy);
        } else if (policy == Stream1TransformerAttentionTilePolicy::Q32K64) {
            stream1_transformer_fmha_launch_policy<cutlass::half_t, cutlass::arch::Sm75, 32, 64>(qkv, context, dims, b_micro, stream, max_k_policy);
        } else {
            stream1_transformer_fmha_launch_policy<cutlass::half_t, cutlass::arch::Sm75, 64, 64>(qkv, context, dims, b_micro, stream, max_k_policy);
        }
        return;
    }
    if (dims.dtype == STREAM1_DTYPE_BF16) {
        const auto policy = parse_stream1_transformer_attention_tile_policy(std::getenv("BEAM_STREAM1_TRANSFORMER_ATTENTION_TILE_POLICY"));
        if (policy == Stream1TransformerAttentionTilePolicy::Q64K64V4) {
            stream1_transformer_fmha_launch_policy<cutlass::bfloat16_t, cutlass::arch::Sm80, 64, 64, false>(qkv, context, dims, b_micro, stream, max_k_policy);
        } else if (policy == Stream1TransformerAttentionTilePolicy::Q32K64) {
            stream1_transformer_fmha_launch_policy<cutlass::bfloat16_t, cutlass::arch::Sm80, 32, 64>(qkv, context, dims, b_micro, stream, max_k_policy);
        } else {
            stream1_transformer_fmha_launch_policy<cutlass::bfloat16_t, cutlass::arch::Sm80, 64, 64>(qkv, context, dims, b_micro, stream, max_k_policy);
        }
        return;
    }
    if (dims.dtype == STREAM1_DTYPE_FP16) {
        const auto policy = parse_stream1_transformer_attention_tile_policy(std::getenv("BEAM_STREAM1_TRANSFORMER_ATTENTION_TILE_POLICY"));
        if (policy == Stream1TransformerAttentionTilePolicy::Q64K64V4) {
            stream1_transformer_fmha_launch_policy<cutlass::half_t, cutlass::arch::Sm80, 64, 64, false>(qkv, context, dims, b_micro, stream, max_k_policy);
        } else if (policy == Stream1TransformerAttentionTilePolicy::Q32K64) {
            stream1_transformer_fmha_launch_policy<cutlass::half_t, cutlass::arch::Sm80, 32, 64>(qkv, context, dims, b_micro, stream, max_k_policy);
        } else {
            stream1_transformer_fmha_launch_policy<cutlass::half_t, cutlass::arch::Sm80, 64, 64>(qkv, context, dims, b_micro, stream, max_k_policy);
        }
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
    if (dims.seq_len == 0U || dims.padded_seq_len < dims.seq_len ||
        dims.d_model == 0U || dims.d_model != dims.nhead * dims.head_dim ||
        dims.padded_seq_len > 64U) {
        throw std::invalid_argument("Stream1 piece_transformer CUTLASS CLS FMHA received unsupported logical/padded dimensions");
    }
    if (qkv == nullptr || cls_context == nullptr) {
        throw std::invalid_argument("Stream1 piece_transformer CUTLASS CLS FMHA requires qkv and cls_context");
    }
#if BEAM_HAS_CUTLASS && BEAM_HAS_CUTLASS_FMHA
    const auto cls_policy = parse_stream1_transformer_cls_attention_policy(
        std::getenv("BEAM_STREAM1_TRANSFORMER_CLS_ATTENTION_POLICY"));
    if (cls_policy == Stream1TransformerClsAttentionPolicy::Q32K64) {
        if (sm75_fp16) {
            stream1_transformer_fmha_cls_launch_typed<cutlass::half_t, cutlass::arch::Sm75, 32, 64>(nullptr, qkv, cls_context, dims, b_micro, stream);
        } else if (dims.dtype == STREAM1_DTYPE_BF16) {
            stream1_transformer_fmha_cls_launch_typed<cutlass::bfloat16_t, cutlass::arch::Sm80, 32, 64>(nullptr, qkv, cls_context, dims, b_micro, stream);
        } else {
            stream1_transformer_fmha_cls_launch_typed<cutlass::half_t, cutlass::arch::Sm80, 32, 64>(nullptr, qkv, cls_context, dims, b_micro, stream);
        }
        return;
    }    if (sm75_fp16) {
        if (dims.dtype != STREAM1_DTYPE_FP16) {
            throw std::invalid_argument("Stream1 piece_transformer SM75 CUTLASS CLS FMHA requires fp16");
        }
        stream1_transformer_fmha_cls_launch_typed<cutlass::half_t, cutlass::arch::Sm75, 64, 64>(nullptr, qkv, cls_context, dims, b_micro, stream);
        return;
    }
    if (dims.dtype == STREAM1_DTYPE_BF16) {
        stream1_transformer_fmha_cls_launch_typed<cutlass::bfloat16_t, cutlass::arch::Sm80, 64, 64>(nullptr, qkv, cls_context, dims, b_micro, stream);
        return;
    }
    if (dims.dtype == STREAM1_DTYPE_FP16) {
        stream1_transformer_fmha_cls_launch_typed<cutlass::half_t, cutlass::arch::Sm80, 64, 64>(nullptr, qkv, cls_context, dims, b_micro, stream);
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

void stream1_transformer_fmha_cls_attention_split_q_cuda(
    half* cls_query,
    half* qkv,
    half* cls_context,
    Stream1TransformerDims dims,
    bool sm75_fp16,
    std::uint32_t b_micro,
    cudaStream_t stream) {
    if (cls_query == nullptr || qkv == nullptr || cls_context == nullptr) {
        throw std::invalid_argument("Stream1 split-Q CLS FMHA requires query, qkv, and context");
    }
#if BEAM_HAS_CUTLASS && BEAM_HAS_CUTLASS_FMHA
    const auto cls_policy = parse_stream1_transformer_cls_attention_policy(
        std::getenv("BEAM_STREAM1_TRANSFORMER_CLS_ATTENTION_POLICY"));
    if (sm75_fp16 && dims.dtype != STREAM1_DTYPE_FP16) {
        throw std::invalid_argument("Stream1 split-Q SM75 CLS FMHA requires fp16");
    }
    if (cls_policy == Stream1TransformerClsAttentionPolicy::Q32K64) {
        if (sm75_fp16) {
            stream1_transformer_fmha_cls_launch_typed<cutlass::half_t, cutlass::arch::Sm75, 32, 64>(cls_query, qkv, cls_context, dims, b_micro, stream);
        } else if (dims.dtype == STREAM1_DTYPE_BF16) {
            stream1_transformer_fmha_cls_launch_typed<cutlass::bfloat16_t, cutlass::arch::Sm80, 32, 64>(cls_query, qkv, cls_context, dims, b_micro, stream);
        } else if (dims.dtype == STREAM1_DTYPE_FP16) {
            stream1_transformer_fmha_cls_launch_typed<cutlass::half_t, cutlass::arch::Sm80, 32, 64>(cls_query, qkv, cls_context, dims, b_micro, stream);
        } else {
            throw std::invalid_argument("Stream1 split-Q CLS FMHA dtype must be fp16 or bf16");
        }
        return;
    }
    if (sm75_fp16) {
        stream1_transformer_fmha_cls_launch_typed<cutlass::half_t, cutlass::arch::Sm75, 64, 64>(cls_query, qkv, cls_context, dims, b_micro, stream);
    } else if (dims.dtype == STREAM1_DTYPE_BF16) {
        stream1_transformer_fmha_cls_launch_typed<cutlass::bfloat16_t, cutlass::arch::Sm80, 64, 64>(cls_query, qkv, cls_context, dims, b_micro, stream);
    } else if (dims.dtype == STREAM1_DTYPE_FP16) {
        stream1_transformer_fmha_cls_launch_typed<cutlass::half_t, cutlass::arch::Sm80, 64, 64>(cls_query, qkv, cls_context, dims, b_micro, stream);
    } else {
        throw std::invalid_argument("Stream1 split-Q CLS FMHA dtype must be fp16 or bf16");
    }
#else
    (void)cls_query; (void)qkv; (void)cls_context; (void)dims;
    (void)sm75_fp16; (void)b_micro; (void)stream;
    throw std::invalid_argument("Stream1 split-Q CLS FMHA requires CUTLASS example 41 headers");
#endif
}
} // namespace beam
