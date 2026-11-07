#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "cute/arch/mma_sm120.hpp"
#include "cutlass/numeric_types.h"

namespace stream1_sm120_nvfp4_raw {

using Mma = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t,
    cutlass::float_e2m1_t,
    float,
    cutlass::float_ue4m3_t,
    16>;

__device__ inline std::uint8_t packed_nibble(
    const std::uint8_t* packed, std::size_t logical_index) {
    const std::uint8_t byte = packed[logical_index >> 1U];
    return static_cast<std::uint8_t>((byte >> ((logical_index & 1U) * 4U)) & 0x0fU);
}

__device__ inline typename Mma::RegTypeSF unit_scale_pair(bool compensate_4x) {
    // PTX's mxf4nvf4 scale_vec::4X has a fixed 0.5 product normalization.
    // Compensate exactly once across the A/B scale pair, never on both sides.
    const std::uint8_t one =
        cutlass::float_ue4m3_t(compensate_4x ? 2.0F : 1.0F).raw();
    return static_cast<typename Mma::RegTypeSF>(
        static_cast<std::uint16_t>(one) |
        (static_cast<std::uint16_t>(one) << 8U));
}

__device__ inline typename Mma::RegTypeSF scale_quad(
    std::uint8_t first, std::uint8_t second,
    std::uint8_t third, std::uint8_t fourth) {
    return static_cast<typename Mma::RegTypeSF>(
        static_cast<std::uint32_t>(first) |
        (static_cast<std::uint32_t>(second) << 8U) |
        (static_cast<std::uint32_t>(third) << 16U) |
        (static_cast<std::uint32_t>(fourth) << 24U));
}

// NVFP4 UE4M3 scales are stored as [M,K/16].  The scale_vec::4X instruction
// consumes four consecutive scale blocks for each K64 MMA.  Unlike the
// accidentally exercised K32 contract, the native K16 path needs no extra
// scale compensation.
__device__ inline typename Mma::RegTypeSF load_sfa_pair(
    const std::uint8_t* scales,
    std::uint32_t leading_scale_k,
    std::uint32_t row_begin,
    std::uint32_t k_begin,
    std::uint32_t lane) {
    const std::uint32_t row = row_begin + (lane & 1U) * 8U + (lane >> 2U);
    const std::uint32_t scale_k = k_begin / 16U;
    const auto* words = reinterpret_cast<const std::uint32_t*>(scales);
    return static_cast<typename Mma::RegTypeSF>(
        words[(static_cast<std::size_t>(row) * leading_scale_k + scale_k) / 4U]);
}

// B scales are stored as [N,K/16] with no compensation.
__device__ inline typename Mma::RegTypeSF load_sfb_pair(
    const std::uint8_t* scales,
    std::uint32_t leading_scale_k,
    std::uint32_t col_begin,
    std::uint32_t k_begin,
    std::uint32_t lane) {
    const std::uint32_t col = col_begin + (lane >> 2U);
    const std::uint32_t scale_k = k_begin / 16U;
    const auto* words = reinterpret_cast<const std::uint32_t*>(scales);
    return static_cast<typename Mma::RegTypeSF>(
        words[(static_cast<std::size_t>(col) * leading_scale_k + scale_k) / 4U]);
}

// Load the per-lane A registers described by CUTLASS's SM120 block-scaled
// ALayout: (T32,V32) -> (M16,K64).  A is logically row-major MxK.
__device__ inline void load_a_unit_scale(
    const std::uint8_t* packed_a,
    std::uint32_t leading_k,
    std::uint32_t row_begin,
    std::uint32_t k_begin,
    std::uint32_t lane,
    std::uint32_t (&registers)[4]) {
    const std::uint32_t thread_0 = lane & 3U;
    const std::uint32_t thread_1 = lane >> 2U;
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        std::uint32_t packed = 0U;
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            const std::uint32_t value = static_cast<std::uint32_t>(reg * 8 + item);
            const std::uint32_t value_0 = value & 7U;
            const std::uint32_t value_1 = (value >> 3U) & 1U;
            const std::uint32_t value_2 = value >> 4U;
            const std::uint32_t linear =
                thread_0 * 128U + thread_1 + value_0 * 16U +
                value_1 * 8U + value_2 * 512U;
            const std::uint32_t local_row = linear % 16U;
            const std::uint32_t local_k = linear / 16U;
            const std::size_t logical =
                static_cast<std::size_t>(row_begin + local_row) * leading_k +
                k_begin + local_k;
            packed |= static_cast<std::uint32_t>(packed_nibble(packed_a, logical))
                << (item * 4U);
        }
        registers[reg] = packed;
    }
}

// Load an A operand exported in instruction fragment order:
// [M16 tile, K64 tile, lane32, register4].
__device__ inline void load_a_prepacked(
    const std::uint32_t* packed_a,
    std::uint32_t total_k,
    std::uint32_t row_begin,
    std::uint32_t k_begin,
    std::uint32_t lane,
    std::uint32_t (&registers)[4]) {
    const std::uint32_t tile_m = row_begin / 16U;
    const std::uint32_t tile_k = k_begin / 64U;
    const std::uint32_t tiles_k = total_k / 64U;
    const std::size_t base =
        ((static_cast<std::size_t>(tile_m) * tiles_k + tile_k) * 32U + lane) * 4U;
    registers[0] = packed_a[base];
    registers[1] = packed_a[base + 1U];
    registers[2] = packed_a[base + 2U];
    registers[3] = packed_a[base + 3U];
}

// Load the per-lane B registers described by CUTLASS's SM120 block-scaled
// BLayout: (T32,V16) -> (N8,K64).  B is stored K-major with N contiguous,
// matching the immutable Stream1 weight artifact.
__device__ inline void load_b_unit_scale(
    const std::uint8_t* packed_b,
    std::uint32_t leading_n,
    std::uint32_t col_begin,
    std::uint32_t k_begin,
    std::uint32_t lane,
    std::uint32_t (&registers)[2]) {
    const std::uint32_t thread_0 = lane & 3U;
    const std::uint32_t thread_1 = lane >> 2U;
#pragma unroll
    for (int reg = 0; reg < 2; ++reg) {
        std::uint32_t packed = 0U;
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            const std::uint32_t value = static_cast<std::uint32_t>(reg * 8 + item);
            const std::uint32_t value_0 = value & 7U;
            const std::uint32_t value_1 = value >> 3U;
            const std::uint32_t linear =
                thread_0 * 64U + thread_1 + value_0 * 8U + value_1 * 256U;
            const std::uint32_t local_col = linear % 8U;
            const std::uint32_t local_k = linear / 8U;
            const std::size_t logical =
                static_cast<std::size_t>(k_begin + local_k) * leading_n +
                col_begin + local_col;
            packed |= static_cast<std::uint32_t>(packed_nibble(packed_b, logical))
                << (item * 4U);
        }
        registers[reg] = packed;
    }
}

// Load an immutable B operand that was exported directly in instruction
// fragment order: [N8 tile, K64 tile, lane32, register2].  This preserves the
// exact payload size while replacing sixteen nibble gathers with two aligned
// 32-bit loads per lane and MMA.
__device__ inline void load_b_prepacked(
    const std::uint32_t* packed_b,
    std::uint32_t leading_n,
    std::uint32_t total_k,
    std::uint32_t col_begin,
    std::uint32_t k_begin,
    std::uint32_t lane,
    std::uint32_t (&registers)[2]) {
    const std::uint32_t tile_n = col_begin / 8U;
    const std::uint32_t tile_k = k_begin / 64U;
    const std::uint32_t tiles_k = total_k / 64U;
    const std::size_t base =
        ((static_cast<std::size_t>(tile_n) * tiles_k + tile_k) * 32U + lane) * 2U;
    registers[0] = packed_b[base];
    registers[1] = packed_b[base + 1U];
}

__device__ inline void store_c(
    float* output,
    std::uint32_t leading_n,
    std::uint32_t row_begin,
    std::uint32_t col_begin,
    std::uint32_t lane,
    const float (&accumulators)[4]) {
    const std::uint32_t thread_0 = lane & 3U;
    const std::uint32_t thread_1 = lane >> 2U;
#pragma unroll
    for (int value = 0; value < 4; ++value) {
        const std::uint32_t value_0 = static_cast<std::uint32_t>(value) & 1U;
        const std::uint32_t value_1 = static_cast<std::uint32_t>(value) >> 1U;
        const std::uint32_t linear =
            thread_0 * 32U + thread_1 + value_0 * 16U + value_1 * 8U;
        const std::uint32_t local_row = linear % 16U;
        const std::uint32_t local_col = linear / 16U;
        output[static_cast<std::size_t>(row_begin + local_row) * leading_n +
               col_begin + local_col] = accumulators[value];
    }
}

}  // namespace stream1_sm120_nvfp4_raw
