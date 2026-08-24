#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

std::size_t stream1_transformer_cublaslt_fp16_workspace_bytes();
void stream1_transformer_cublaslt_fp16_linear_cuda(
    const half* input,
    const half* weight,
    half* output,
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    void* workspace,
    std::size_t workspace_bytes,
    cudaStream_t stream);
