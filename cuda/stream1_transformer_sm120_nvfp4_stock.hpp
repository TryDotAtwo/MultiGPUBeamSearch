#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

// Conservative production control path for one Cube4 FFN layer on SM120.
// Unlike the zero-global-hidden research kernel, this path deliberately uses
// stock CUTLASS collectives and materializes the FF1 ReLU result as NVFP4.
// The materialization reuses existing Transformer scratch allocations, so the
// opt-in path does not increase the production StaticMemoryPlan.
struct Stream1Sm120Nvfp4StockScratchPlan {
    std::size_t input_value_offset = 0U;
    std::size_t input_value_bytes = 0U;
    std::size_t input_scale_offset = 0U;
    std::size_t input_scale_bytes = 0U;
    std::size_t input_total_bytes = 0U;
    std::size_t hidden_value_offset = 0U;
    std::size_t hidden_value_bytes = 0U;
    std::size_t hidden_scale_offset = 0U;
    std::size_t hidden_scale_bytes = 0U;
    std::size_t hidden_total_bytes = 0U;
};

inline constexpr std::size_t stream1_sm120_nvfp4_align_up(
    std::size_t value, std::size_t alignment) {
    return (value + alignment - 1U) / alignment * alignment;
}

inline constexpr std::size_t stream1_sm120_nvfp4_activation_value_bytes(
    std::uint32_t rows, std::uint32_t columns) {
    return (static_cast<std::size_t>(rows) * columns + 1U) / 2U;
}

// Sm1xxBlockScaledConfig<16> pads M to 128 and reserves four physical bytes
// per logical K16 scale.  This is the exact SFA/SFD storage contract used by
// CUTLASS SM120 NVFP4 collectives, not the smaller logical scale matrix.
inline constexpr std::size_t stream1_sm120_nvfp4_activation_scale_bytes(
    std::uint32_t rows, std::uint32_t columns) {
    const std::size_t padded_rows =
        (static_cast<std::size_t>(rows) + 127U) / 128U * 128U;
    const std::size_t k16_groups =
        (static_cast<std::size_t>(columns) + 15U) / 16U;
    return padded_rows * k16_groups * 4U;
}

inline constexpr Stream1Sm120Nvfp4StockScratchPlan
stream1_sm120_nvfp4_stock_scratch_plan(std::uint32_t rows) {
    Stream1Sm120Nvfp4StockScratchPlan plan{};
    plan.input_value_bytes =
        stream1_sm120_nvfp4_activation_value_bytes(rows, 256U);
    plan.input_scale_offset =
        stream1_sm120_nvfp4_align_up(plan.input_value_bytes, 128U);
    plan.input_scale_bytes =
        stream1_sm120_nvfp4_activation_scale_bytes(rows, 256U);
    plan.input_total_bytes = plan.input_scale_offset + plan.input_scale_bytes;
    plan.hidden_value_bytes =
        stream1_sm120_nvfp4_activation_value_bytes(rows, 1024U);
    plan.hidden_scale_offset =
        stream1_sm120_nvfp4_align_up(plan.hidden_value_bytes, 128U);
    plan.hidden_scale_bytes =
        stream1_sm120_nvfp4_activation_scale_bytes(rows, 1024U);
    plan.hidden_total_bytes = plan.hidden_scale_offset + plan.hidden_scale_bytes;
    return plan;
}

#if defined(__CUDACC__)

bool stream1_transformer_sm120_nvfp4_stock_supported();

void stream1_transformer_sm120_nvfp4_stock_ffn_cuda(
    const half* normalized_input_fp16,
    const half* residual_input_fp16,
    const std::uint8_t* ff1_weight_e2m1,
    const std::uint8_t* ff1_weight_ue4m3,
    const half* ff1_bias_fp16,
    const std::uint8_t* ff2_weight_e2m1,
    const std::uint8_t* ff2_weight_ue4m3,
    half* residual_output_fp16,
    void* input_scratch,
    std::size_t input_scratch_bytes,
    void* hidden_scratch,
    std::size_t hidden_scratch_bytes,
    std::uint32_t rows,
    cudaStream_t stream);

// Fuses the production bias+residual LayerNorm boundary with the K16 NVFP4
// input producer.  The residual tensor keeps the same rounded FP16 values as
// the ordinary LayerNorm path, while the normalized activation is rounded to
// FP16 and packed directly into input_scratch for FF1.  No normalized FP16
// tensor is materialized in global memory.
void stream1_transformer_sm120_nvfp4_stock_bias_layernorm_ffn_cuda(
    half* residual_input_output_fp16,
    const half* residual_bias_fp16,
    const half* layernorm_gamma_fp16,
    const half* layernorm_beta_fp16,
    const std::uint8_t* ff1_weight_e2m1,
    const std::uint8_t* ff1_weight_ue4m3,
    const half* ff1_bias_fp16,
    const std::uint8_t* ff2_weight_e2m1,
    const std::uint8_t* ff2_weight_ue4m3,
    void* input_scratch,
    std::size_t input_scratch_bytes,
    void* hidden_scratch,
    std::size_t hidden_scratch_bytes,
    std::uint32_t rows,
    cudaStream_t stream);

// Fuses the pre-attention residual completion (optional deferred FF2 bias),
// LayerNorm, NVFP4 activation packing, and the 256->768 QKV projection.  The
// scratch storage is transient and may alias the later FF hidden workspace;
// QKV output must not alias it.  When residual_bias_fp16 is null the residual
// is left untouched (the first transformer layer fast path).
void stream1_transformer_sm120_nvfp4_stock_layernorm_qkv_cuda(
    half* residual_input_output_fp16,
    const half* residual_bias_fp16,
    const half* layernorm_gamma_fp16,
    const half* layernorm_beta_fp16,
    const std::uint8_t* qkv_weight_e2m1,
    const std::uint8_t* qkv_weight_ue4m3,
    const half* qkv_bias_fp16,
    half* qkv_output_fp16,
    void* input_scratch,
    std::size_t input_scratch_bytes,
    std::uint32_t rows,
    cudaStream_t stream);

#endif
