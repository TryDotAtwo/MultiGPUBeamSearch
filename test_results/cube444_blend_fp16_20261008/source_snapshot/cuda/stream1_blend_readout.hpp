#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace beam {
// FP32 row-major A[rows,256], W[256,24], Q_mlp[rows,24], bias[24].
// Same CUDA stream must wait for the MLP producer before calling this readout.
// Writes clamp/round keys directly; no intermediate Transformer logits.
void stream1_blend_readout_cuda(const float* cls, const float* weight,
    const float* bias, const float* mlp_q, std::uint32_t* keys,
    std::uint32_t rows, std::uint32_t* numeric_error, cudaStream_t stream, bool transformer_only = false);
// FP16 Tensor Core operands, FP32 accumulator/bias/blend/quantization.
void stream1_blend_readout_fp16_cuda(const __half* cls, const __half* weight,
    const float* bias, const float* mlp_q, std::uint32_t* keys,
    std::uint32_t rows, std::uint32_t* numeric_error, cudaStream_t stream, bool transformer_only = false);
}
