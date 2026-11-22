#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace beam {
// Arbitrary-length ensemble: call once per head in stable manifest order on
// the same stream. First call uses initialize=true; only last call emits keys.
// A[rows,hidden], W[hidden,outputs] are contiguous FP16. Aligned Q heads use
// Tensor Cores; scalar/unaligned heads use a CUTLASS SIMT readout. FP32 accumulator
// is shared by all heads, never quantized
// between heads. The caller owns lifetime and cross-stream producer events.
void stream1_ensemble_head_fp16_cuda(const __half* activations, const __half* weight,
    const float* bias, float* accumulator, std::uint32_t* keys,
    std::uint32_t rows, std::uint32_t hidden, std::uint32_t outputs,
    float coefficient, bool initialize, bool finalize,
    std::uint32_t* numeric_error, cudaStream_t stream);
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
