#pragma once

#include "stream1.hpp"

namespace beam {

void stream1_transformer_fmha_attention_cuda(
    half* qkv,
    half* packed_qkv,
    half* context,
    Stream1TransformerDims dims,
    std::uint32_t b_micro,
    cudaStream_t stream);

} // namespace beam