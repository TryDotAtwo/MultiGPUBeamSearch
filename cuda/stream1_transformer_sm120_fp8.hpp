#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

bool stream1_transformer_sm120_fp8_supported();

std::size_t stream1_transformer_sm120_fp8_input_scale_elements(
    std::uint32_t rows,
    std::uint32_t input_cols);

std::size_t stream1_transformer_sm120_fp8_weight_scale_elements(
    std::uint32_t input_cols,
    std::uint32_t output_cols);

std::size_t stream1_transformer_sm120_fp8_workspace_bytes(
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols);

// Three-term residual encoding computes Ah*Bh + Al*Bh + Ah*Bl as one
// block-scaled GEMM with a packed 3K inner dimension. Weights are encoded
// offline from FP32; only activations are encoded dynamically.
std::size_t stream1_transformer_sm120_fp8_residual3_input_bytes(
    std::uint32_t rows, std::uint32_t input_cols);
std::size_t stream1_transformer_sm120_fp8_residual3_input_scale_elements(
    std::uint32_t rows, std::uint32_t input_cols);
std::size_t stream1_transformer_sm120_fp8_residual3_weight_bytes(
    std::uint32_t input_cols, std::uint32_t output_cols);
std::size_t stream1_transformer_sm120_fp8_residual3_weight_scale_elements(
    std::uint32_t input_cols, std::uint32_t output_cols);
std::size_t stream1_transformer_sm120_fp8_residual3_workspace_bytes(
    std::uint32_t rows, std::uint32_t input_cols, std::uint32_t output_cols);

void stream1_transformer_sm120_fp8_quantize_weight_cuda(
    const half* weight,
    std::uint8_t* quantized_weight,
    float* weight_scales,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    float scale_multiplier,
    cudaStream_t stream);

void stream1_transformer_sm120_fp8_quantize_weight_mse_cuda(
    const half* weight,
    std::uint8_t* quantized_weight,
    float* weight_scales,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    cudaStream_t stream);

void stream1_transformer_sm120_fp8_quantize_weight_from_fp32_cuda(
    const float* weight,
    std::uint8_t* quantized_weight,
    float* weight_scales,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    float scale_multiplier,
    cudaStream_t stream);

void stream1_transformer_sm120_fp8_quantize_weight_mse_from_fp32_cuda(
    const float* weight,
    std::uint8_t* quantized_weight,
    float* weight_scales,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    cudaStream_t stream);

void stream1_transformer_sm120_fp8_residual3_quantize_weight_from_fp32_cuda(
    const float* weight,
    std::uint8_t* packed_quantized_weight,
    float* packed_weight_scales,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    cudaStream_t stream);

void stream1_transformer_sm120_fp8_linear_cuda(
    const half* input,
    std::uint8_t* quantized_input,
    float* input_scales,
    const std::uint8_t* quantized_weight,
    const float* weight_scales,
    half* output,
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    float activation_scale_multiplier,
    void* workspace,
    std::size_t workspace_bytes,
    cudaStream_t stream);

void stream1_transformer_sm120_fp8_residual3_linear_cuda(
    const half* input,
    std::uint8_t* packed_quantized_input,
    float* packed_input_scales,
    const std::uint8_t* packed_quantized_weight,
    const float* packed_weight_scales,
    half* output,
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    void* workspace,
    std::size_t workspace_bytes,
    cudaStream_t stream);
