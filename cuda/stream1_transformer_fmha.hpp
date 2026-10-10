#pragma once

#include "stream1.hpp"

namespace beam {

void stream1_transformer_fmha_attention_cuda(
    half* qkv,
    half* packed_qkv,
    half* context,
    Stream1TransformerDims dims,
    bool sm75_fp16,
    std::uint32_t b_micro,
    cudaStream_t stream);
void stream1_transformer_fmha_cls_attention_cuda(
    half* qkv,
    half* cls_context,
    Stream1TransformerDims dims,
    bool sm75_fp16,
    std::uint32_t b_micro,
    cudaStream_t stream);
void stream1_transformer_fmha_cls_attention_split_q_cuda(
    half* cls_query,
    half* qkv,
    half* cls_context,
    Stream1TransformerDims dims,
    bool sm75_fp16,
    std::uint32_t b_micro,
    cudaStream_t stream);
} // namespace beam
