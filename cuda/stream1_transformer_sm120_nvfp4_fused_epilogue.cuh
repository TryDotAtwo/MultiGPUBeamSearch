#pragma once

#include "stream1_transformer_sm120_nvfp4_cutlass_dsm_fused.cuh"

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace stream1_sm120_nvfp4_cutlass_dsm {

// FF2's two physical CTAs own disjoint 128-column halves of one M128xN256
// output tile.  Once the last FF2 MMA has drained, its shared arena is dead and
// can be reused as a 64-KiB FP32 tile.  Keeping the unrounded FF2+bias+residual
// values here makes the following LayerNorm consume the exact FP32 sum while
// avoiding both a global FP32 FF2 tensor and a global normalized FP16 tensor.
struct Stream1Sm120Nvfp4FusedEpilogueContract {
    static constexpr std::uint32_t kRows = 128U;
    static constexpr std::uint32_t kColumnsPerRank = 128U;
    static constexpr std::uint32_t kColumns = 256U;
    static constexpr std::uint32_t kClusterCtas = 2U;
    static constexpr std::uint32_t kScaleVector = 16U;
    static constexpr std::size_t kSharedBytes =
        static_cast<std::size_t>(kRows) * kColumnsPerRank * sizeof(float);
};

static_assert(
    Stream1Sm120Nvfp4FusedEpilogueContract::kColumns == Contract::kDModel);
static_assert(
    Stream1Sm120Nvfp4FusedEpilogueContract::kColumnsPerRank ==
    Contract::kSliceColumns);

CUTLASS_DEVICE std::uint8_t stream1_sm120_nvfp4_encode_e2m1(
    float value, float inverse_scale) {
    return static_cast<std::uint8_t>(
        cutlass::float_e2m1_t(value * inverse_scale).raw() & 0x0fU);
}

template <class Cluster, class ScaleLayout>
CUTLASS_DEVICE void stream1_sm120_nvfp4_finish_ff2_tile(
    Cluster const& cluster,
    std::uint32_t rank,
    std::uint32_t m_tile,
    float* local_unrounded,
    ElementBias const* ff2_bias,
    ElementBias const* residual_input,
    ElementBias const* layernorm_gamma,
    ElementBias const* layernorm_beta,
    ElementBias* residual_output,
    std::uint8_t* next_values,
    ElementScale* next_scales,
    ScaleLayout const& next_scale_layout,
    float epsilon) {
    using E = Stream1Sm120Nvfp4FusedEpilogueContract;
    const std::uint32_t row = threadIdx.x;
    if (row >= E::kRows) {
        return;
    }

    // Both ranks intentionally reduce columns in the same logical order.
    // Local-first would reverse the two 128-column halves on rank 1 and could
    // make row scales diverge at floating-point rounding boundaries.
    auto* rank0 = rank == 0U
        ? local_unrounded
        : cluster.map_shared_rank(local_unrounded, 0U);
    auto* rank1 = rank == 1U
        ? local_unrounded
        : cluster.map_shared_rank(local_unrounded, 1U);

    float sum = 0.0F;
    // Preserve the canonical scalar order but do not fully unroll 128-term
    // reductions into the already register-heavy FF1+FF2 persistent kernel.
    // Full unrolling pushed the fused kernel to 255 registers and a 1040-byte
    // stack frame on sm_120a.
#pragma unroll 1
    for (std::uint32_t column = 0; column < E::kColumnsPerRank; ++column) {
        sum += rank0[row * E::kColumnsPerRank + column];
    }
#pragma unroll 1
    for (std::uint32_t column = 0; column < E::kColumnsPerRank; ++column) {
        sum += rank1[row * E::kColumnsPerRank + column];
    }
    const float mean = sum * (1.0F / static_cast<float>(E::kColumns));

    float variance_sum = 0.0F;
#pragma unroll 1
    for (std::uint32_t column = 0; column < E::kColumnsPerRank; ++column) {
        const float centered =
            rank0[row * E::kColumnsPerRank + column] - mean;
        variance_sum += centered * centered;
    }
#pragma unroll 1
    for (std::uint32_t column = 0; column < E::kColumnsPerRank; ++column) {
        const float centered =
            rank1[row * E::kColumnsPerRank + column] - mean;
        variance_sum += centered * centered;
    }
    const float inverse_std = rsqrtf(
        variance_sum * (1.0F / static_cast<float>(E::kColumns)) + epsilon);

    const std::uint32_t global_row = m_tile * E::kRows + row;
    const std::uint32_t first_column = rank * E::kColumnsPerRank;
    const std::size_t global_base =
        static_cast<std::size_t>(global_row) * E::kColumns;
    const float* local_row = local_unrounded + row * E::kColumnsPerRank;

#pragma unroll 1
    for (std::uint32_t group = 0;
         group < E::kColumnsPerRank / E::kScaleVector; ++group) {
        float max_abs = 0.0F;
#pragma unroll 1
        for (std::uint32_t lane = 0; lane < E::kScaleVector; ++lane) {
            const std::uint32_t local_column = group * E::kScaleVector + lane;
            const std::uint32_t global_column = first_column + local_column;
            const float x = local_row[local_column];
            residual_output[global_base + global_column] = ElementBias(x);
            const float y = (x - mean) * inverse_std *
                static_cast<float>(layernorm_gamma[global_column]) +
                static_cast<float>(layernorm_beta[global_column]);
            max_abs = fmaxf(max_abs, fabsf(y));
        }

        const ElementScale encoded_scale(
            max_abs == 0.0F ? 1.0F : max_abs / 6.0F);
        const float decoded_scale = static_cast<float>(encoded_scale);
        const float inverse_scale =
            decoded_scale == 0.0F ? 0.0F : 1.0F / decoded_scale;
        const std::uint32_t group_column =
            first_column + group * E::kScaleVector;
        auto scales = cute::make_tensor(next_scales, next_scale_layout);
        scales(
            static_cast<int>(global_row),
            static_cast<int>(group_column),
            0) = encoded_scale;

#pragma unroll 1
        for (std::uint32_t pair = 0; pair < E::kScaleVector / 2U; ++pair) {
            const std::uint32_t local_column =
                group * E::kScaleVector + pair * 2U;
            const std::uint32_t global_column = first_column + local_column;
            const float x0 = local_row[local_column];
            const float x1 = local_row[local_column + 1U];
            const float y0 = (x0 - mean) * inverse_std *
                static_cast<float>(layernorm_gamma[global_column]) +
                static_cast<float>(layernorm_beta[global_column]);
            const float y1 = (x1 - mean) * inverse_std *
                static_cast<float>(layernorm_gamma[global_column + 1U]) +
                static_cast<float>(layernorm_beta[global_column + 1U]);
            const std::uint8_t lo = stream1_sm120_nvfp4_encode_e2m1(
                y0, inverse_scale);
            const std::uint8_t hi = stream1_sm120_nvfp4_encode_e2m1(
                y1, inverse_scale);
            const std::size_t value_index = global_base + group_column + pair * 2U;
            next_values[value_index / 2U] = static_cast<std::uint8_t>(
                lo | static_cast<std::uint8_t>(hi << 4U));
        }
    }
}

}  // namespace stream1_sm120_nvfp4_cutlass_dsm
