#pragma once

#include "stream1_transformer_sm120_nvfp4_ffn_policy.hpp"

#include <cstddef>
#include <cstdint>

// Fixed-shape native contract for one non-final Cube4 Transformer FFN layer.
// Deliberately absent: a global-memory FF1 hidden pointer.  A conforming SM120
// implementation must retain the 256->1024 ReLU intermediate in registers or
// CTA-local shared storage while consuming it into the 1024->256 GEMM.
struct Stream1Sm120Nvfp4FfnArguments {
    const void* normalized_input_fp16 = nullptr;  // [rows, 256]
    const void* residual_input_fp16 = nullptr;    // [rows, 256]
    const void* ff1_weight_e2m1 = nullptr;        // immutable CUTLASS SM120 layout
    const void* ff1_weight_ue4m3 = nullptr;       // N x ceil(K / 16)
    const void* ff1_bias_fp16 = nullptr;          // [1024]
    const void* ff2_weight_e2m1 = nullptr;        // immutable CUTLASS SM120 layout
    const void* ff2_weight_ue4m3 = nullptr;       // N x ceil(K / 16)
    const void* ff2_bias_fp16 = nullptr;          // [256]
    const void* next_layernorm_gamma_fp16 = nullptr;  // [256]
    const void* next_layernorm_beta_fp16 = nullptr;   // [256]
    void* residual_output_fp16 = nullptr;         // [rows, 256]
    void* next_normalized_e2m1 = nullptr;          // [rows, 256], packed
    void* next_normalized_ue4m3 = nullptr;         // [rows, 16]
    std::uint32_t rows = 0;
    float layernorm_epsilon = 1.0e-5F;
};

struct Stream1Sm120Nvfp4FfnMemoryPlan {
    std::size_t global_hidden_bytes = 0;
    std::size_t output_token_bytes = 0;
    std::size_t next_activation_bytes = 0;
    std::size_t next_scale_bytes = 0;
};

inline constexpr Stream1Sm120Nvfp4FfnMemoryPlan
stream1_sm120_nvfp4_ffn_memory_plan(std::uint32_t rows) {
    return {
        0U,
        static_cast<std::size_t>(rows) * 256U * sizeof(std::uint16_t),
        static_cast<std::size_t>(rows) * 256U / 2U,
        static_cast<std::size_t>(rows) * (256U / 16U),
    };
}

// Exact semantic order required by the fused epilogue:
//   fp32_sum = fp32(ff2_accum) + fp32(ff2_bias) + fp32(residual_input_fp16)
//   residual_output_fp16 = round_fp16(fp32_sum)
//   normalized = LayerNorm(fp32_sum, epsilon, gamma, beta)
//   next_normalized = dynamic NVFP4(normalized), K16 UE4M3 scales
// LayerNorm must not read the rounded residual_output_fp16.
enum class Stream1Sm120Nvfp4FfnEpilogueOrder : std::uint32_t {
    ResidualFp32ThenStoreFp16ThenLayernormUnroundedThenNvfp4 = 1U,
};
