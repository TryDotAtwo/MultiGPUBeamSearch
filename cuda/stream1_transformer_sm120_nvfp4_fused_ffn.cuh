#pragma once

#include "stream1_transformer_sm120_nvfp4_raw_mma.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace stream1_sm120_nvfp4_fused {

constexpr std::uint32_t kRowsPerCta = 16U;
constexpr std::uint32_t kRowTilesPerCta = 1U;
constexpr std::uint32_t kTotalRowsPerCta = kRowsPerCta * kRowTilesPerCta;
constexpr std::uint32_t kMaxWarpsPerCta = 8U;
constexpr std::uint32_t kDModel = 256U;
constexpr std::uint32_t kFfDim = 1024U;

struct alignas(128) SharedStorage {
    union {
        float ff1_warp_scratch[
            kMaxWarpsPerCta * kRowTilesPerCta * kRowsPerCta * 16U];
        std::uint32_t hidden_fragments[
            kRowTilesPerCta * kRowsPerCta * kFfDim / 8U];
        struct {
            std::uint32_t values[(kDModel / 64U) * 32U * 4U];
            std::uint32_t scales[(kRowsPerCta * (kDModel / 16U)) / 4U];
        } staged_input;
    };
    std::uint8_t hidden_values[
        kRowTilesPerCta * kRowsPerCta * kFfDim / 2U];
    std::uint8_t hidden_scales[
        kRowTilesPerCta * kRowsPerCta * (kFfDim / 16U)];
};

__device__ inline std::uint8_t pack_e2m1_pair_scaled(
    float first, float second, float reciprocal_scale) {
    std::uint32_t packed;
    asm volatile(
        "{\n"
        "  .reg .f32 scaled0, scaled1;\n"
        "  .reg .b8 byte0, byte1, byte2, byte3;\n"
        "  mul.f32 scaled0, %1, %3;\n"
        "  mul.f32 scaled1, %2, %3;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 byte0, scaled1, scaled0;\n"
        "  mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}\n"
        : "=r"(packed)
        : "f"(first), "f"(second), "f"(reciprocal_scale));
    return static_cast<std::uint8_t>(packed & 0xffU);
}

__global__ void fused_ffn_kernel(
    const std::uint8_t* input_values,
    const std::uint8_t* input_scales_compensated,
    const std::uint8_t* ff1_values,
    const std::uint8_t* ff1_scales,
    const float* ff1_bias,
    const std::uint8_t* ff2_values,
    const std::uint8_t* ff2_scales,
    const float* ff2_bias,
    float* output,
    float* debug_hidden_fp32,
    std::uint32_t rows) {
    using namespace stream1_sm120_nvfp4_raw;
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& shared = *reinterpret_cast<SharedStorage*>(shared_bytes);
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t warp_count = blockDim.x >> 5U;
    const std::uint32_t global_row_begin = blockIdx.x * kTotalRowsPerCta;
    const bool skip_ff1_pack = ff2_values == nullptr;

    // The same A tile is reused across every FF1 N group.  Stage its native
    // instruction fragments and K16 scales once per CTA in the storage that
    // is repurposed for FF2 fragments after FF1 completes.
    constexpr std::uint32_t input_fragment_words =
        (kDModel / 64U) * 32U * 4U;
    constexpr std::uint32_t input_scale_words =
        (kRowsPerCta * (kDModel / 16U)) / 4U;
    const auto* global_input_words =
        reinterpret_cast<const std::uint32_t*>(input_values) +
        static_cast<std::size_t>(blockIdx.x) * input_fragment_words;
    for (std::uint32_t word = threadIdx.x;
         word < input_fragment_words; word += blockDim.x) {
        shared.staged_input.values[word] = global_input_words[word];
    }
    const auto* global_input_scale_words =
        reinterpret_cast<const std::uint32_t*>(input_scales_compensated) +
        static_cast<std::size_t>(global_row_begin) * (kDModel / 16U) / 4U;
    for (std::uint32_t word = threadIdx.x;
         word < input_scale_words; word += blockDim.x) {
        shared.staged_input.scales[word] = global_input_scale_words[word];
    }
    __syncthreads();

    // FF1: each warp owns one complete N16 scale group.  Only a 16x16
    // warp-local FP32 tile is staged; ReLU and NVFP4 packing happen before the
    // warp advances, so the full 16x1024 hidden never exists in shared memory.
    constexpr std::uint32_t ff1_groups_per_warp_batch = 2U;
    for (std::uint32_t group_base = warp * ff1_groups_per_warp_batch;
         group_base < kFfDim / 16U;
         group_base += warp_count * ff1_groups_per_warp_batch) {
        float accumulators_0[ff1_groups_per_warp_batch][kRowTilesPerCta][4] = {};
        float accumulators_1[ff1_groups_per_warp_batch][kRowTilesPerCta][4] = {};
        for (std::uint32_t k_begin = 0; k_begin < kDModel; k_begin += 64U) {
#pragma unroll
            for (std::uint32_t row_tile = 0; row_tile < kRowTilesPerCta; ++row_tile) {
                std::uint32_t a[4];
                load_a_prepacked(
                    shared.staged_input.values,
                    kDModel, row_tile * kRowsPerCta,
                    k_begin, lane, a);
                const auto sfa = load_sfa_pair(
                    reinterpret_cast<const std::uint8_t*>(shared.staged_input.scales),
                    kDModel / 16U, row_tile * kRowsPerCta, k_begin, lane);
#pragma unroll
              for (std::uint32_t group_slot = 0;
                   group_slot < ff1_groups_per_warp_batch; ++group_slot) {
                const std::uint32_t group_n = group_base + group_slot;
                std::uint32_t b0[2];
                std::uint32_t b1[2];
                load_b_prepacked(
                    reinterpret_cast<const std::uint32_t*>(ff1_values),
                    kFfDim, kDModel, group_n * 16U, k_begin, lane, b0);
                load_b_prepacked(
                    reinterpret_cast<const std::uint32_t*>(ff1_values),
                    kFfDim, kDModel, group_n * 16U + 8U, k_begin, lane, b1);
                const auto sfb0 = load_sfb_pair(
                    ff1_scales, kDModel / 16U, group_n * 16U, k_begin, lane);
                const auto sfb1 = load_sfb_pair(
                    ff1_scales, kDModel / 16U, group_n * 16U + 8U, k_begin, lane);
                Mma::fma(
                    accumulators_0[group_slot][row_tile][0],
                    accumulators_0[group_slot][row_tile][1],
                    accumulators_0[group_slot][row_tile][2],
                    accumulators_0[group_slot][row_tile][3],
                    a[0], a[1], a[2], a[3], b0[0], b0[1],
                    accumulators_0[group_slot][row_tile][0],
                    accumulators_0[group_slot][row_tile][1],
                    accumulators_0[group_slot][row_tile][2],
                    accumulators_0[group_slot][row_tile][3], sfa, sfb0);
                Mma::fma(
                    accumulators_1[group_slot][row_tile][0],
                    accumulators_1[group_slot][row_tile][1],
                    accumulators_1[group_slot][row_tile][2],
                    accumulators_1[group_slot][row_tile][3],
                    a[0], a[1], a[2], a[3], b1[0], b1[1],
                    accumulators_1[group_slot][row_tile][0],
                    accumulators_1[group_slot][row_tile][1],
                    accumulators_1[group_slot][row_tile][2],
                    accumulators_1[group_slot][row_tile][3], sfa, sfb1);
              }
            }
        }
#pragma unroll
      for (std::uint32_t group_slot = 0;
           group_slot < ff1_groups_per_warp_batch; ++group_slot) {
        const std::uint32_t group_n = group_base + group_slot;
#pragma unroll
        for (std::uint32_t row_tile = 0; row_tile < kRowTilesPerCta; ++row_tile) {
            if (debug_hidden_fp32 != nullptr) {
#pragma unroll
              for (int value = 0; value < 4; ++value) {
                const std::uint32_t linear = (lane & 3U) * 32U + (lane >> 2U) +
                    (static_cast<std::uint32_t>(value) & 1U) * 16U +
                    (static_cast<std::uint32_t>(value) >> 1U) * 8U;
                const std::uint32_t row = linear % 16U;
                const std::uint32_t local_col = linear / 16U;
                debug_hidden_fp32[
                    static_cast<std::size_t>(global_row_begin + row_tile * kRowsPerCta + row) *
                    kFfDim + group_n * 16U + local_col] =
                    accumulators_0[group_slot][row_tile][value];
                debug_hidden_fp32[
                    static_cast<std::size_t>(global_row_begin + row_tile * kRowsPerCta + row) *
                    kFfDim + group_n * 16U + 8U + local_col] =
                    accumulators_1[group_slot][row_tile][value];
              }
            }
            if (!skip_ff1_pack) {
                const std::uint32_t lane_in_group = lane & 3U;
                const std::uint32_t row_low = lane >> 2U;
                const std::uint32_t row_high = row_low + 8U;
                const std::uint32_t col_pair = group_n * 16U + lane_in_group * 2U;
                float low_values[4] = {
                    fmaxf(accumulators_0[group_slot][row_tile][0] + ff1_bias[col_pair], 0.0F),
                    fmaxf(accumulators_0[group_slot][row_tile][1] + ff1_bias[col_pair + 1U], 0.0F),
                    fmaxf(accumulators_1[group_slot][row_tile][0] + ff1_bias[col_pair + 8U], 0.0F),
                    fmaxf(accumulators_1[group_slot][row_tile][1] + ff1_bias[col_pair + 9U], 0.0F)};
                float high_values[4] = {
                    fmaxf(accumulators_0[group_slot][row_tile][2] + ff1_bias[col_pair], 0.0F),
                    fmaxf(accumulators_0[group_slot][row_tile][3] + ff1_bias[col_pair + 1U], 0.0F),
                    fmaxf(accumulators_1[group_slot][row_tile][2] + ff1_bias[col_pair + 8U], 0.0F),
                    fmaxf(accumulators_1[group_slot][row_tile][3] + ff1_bias[col_pair + 9U], 0.0F)};
                float max_low = fmaxf(fmaxf(low_values[0], low_values[1]),
                                      fmaxf(low_values[2], low_values[3]));
                float max_high = fmaxf(fmaxf(high_values[0], high_values[1]),
                                       fmaxf(high_values[2], high_values[3]));
                max_low = fmaxf(max_low, __shfl_xor_sync(0xffffffffU, max_low, 1, 4));
                max_low = fmaxf(max_low, __shfl_xor_sync(0xffffffffU, max_low, 2, 4));
                max_high = fmaxf(max_high, __shfl_xor_sync(0xffffffffU, max_high, 1, 4));
                max_high = fmaxf(max_high, __shfl_xor_sync(0xffffffffU, max_high, 2, 4));
                const cutlass::float_ue4m3_t scale_low(
                    max_low == 0.0F ? 1.0F : max_low / 6.0F);
                const cutlass::float_ue4m3_t scale_high(
                    max_high == 0.0F ? 1.0F : max_high / 6.0F);
                const float logical_low = static_cast<float>(scale_low);
                const float logical_high = static_cast<float>(scale_high);
                const float reciprocal_low = __frcp_rn(logical_low);
                const float reciprocal_high = __frcp_rn(logical_high);
                const std::size_t logical_row_low =
                    static_cast<std::size_t>(row_tile) * kRowsPerCta + row_low;
                const std::size_t logical_row_high =
                    static_cast<std::size_t>(row_tile) * kRowsPerCta + row_high;
                if (lane_in_group == 0U) {
                    shared.hidden_scales[logical_row_low * (kFfDim / 16U) + group_n] =
                        scale_low.raw();
                    shared.hidden_scales[logical_row_high * (kFfDim / 16U) + group_n] =
                        scale_high.raw();
                }
                const std::size_t byte_low =
                    (logical_row_low * kFfDim + group_n * 16U) / 2U + lane_in_group;
                const std::size_t byte_high =
                    (logical_row_high * kFfDim + group_n * 16U) / 2U + lane_in_group;
                shared.hidden_values[byte_low] = pack_e2m1_pair_scaled(
                    low_values[0], low_values[1], reciprocal_low);
                shared.hidden_values[byte_low + 4U] = pack_e2m1_pair_scaled(
                    low_values[2], low_values[3], reciprocal_low);
                shared.hidden_values[byte_high] = pack_e2m1_pair_scaled(
                    high_values[0], high_values[1], reciprocal_high);
                shared.hidden_values[byte_high + 4U] = pack_e2m1_pair_scaled(
                    high_values[2], high_values[3], reciprocal_high);
            }
        }
      }
    }
    if (skip_ff1_pack) {
        return;
    }
    __syncthreads();
    if (ff2_bias == nullptr) {
        return;
    }

    // Convert the compact row-major CTA payload to native A-fragment order.
    constexpr std::uint32_t fragment_words =
        kRowTilesPerCta * kRowsPerCta * kFfDim / 8U;
    for (std::uint32_t word_index = threadIdx.x;
         word_index < fragment_words; word_index += blockDim.x) {
        constexpr std::uint32_t words_per_row_tile = kRowsPerCta * kFfDim / 8U;
        const std::uint32_t row_tile = word_index / words_per_row_tile;
        const std::uint32_t local_word = word_index % words_per_row_tile;
        const std::uint32_t reg = local_word & 3U;
        const std::uint32_t lane_index = (local_word >> 2U) & 31U;
        const std::uint32_t tile_k = local_word / (32U * 4U);
        const std::uint32_t thread_0 = lane_index & 3U;
        const std::uint32_t thread_1 = lane_index >> 2U;
        std::uint32_t word = 0U;
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            const std::uint32_t value = reg * 8U + static_cast<std::uint32_t>(item);
            const std::uint32_t value_0 = value & 7U;
            const std::uint32_t value_1 = (value >> 3U) & 1U;
            const std::uint32_t value_2 = value >> 4U;
            const std::uint32_t linear = thread_0 * 128U + thread_1 +
                value_0 * 16U + value_1 * 8U + value_2 * 512U;
            const std::uint32_t row = linear % 16U;
            const std::uint32_t col = tile_k * 64U + linear / 16U;
            word |= static_cast<std::uint32_t>(packed_nibble(
                shared.hidden_values,
                (static_cast<std::size_t>(row_tile) * kRowsPerCta + row) *
                    kFfDim + col))
                << (item * 4U);
        }
        shared.hidden_fragments[word_index] = word;
    }
    __syncthreads();
    if (output == nullptr) {
        return;
    }

    // FF2 consumes the CTA-local packed hidden tile directly.
    for (std::uint32_t tile_n = warp; tile_n < kDModel / 8U; tile_n += warp_count) {
        float accumulators[kRowTilesPerCta][4] = {};
        for (std::uint32_t k_begin = 0; k_begin < kFfDim; k_begin += 64U) {
            std::uint32_t b[2];
            load_b_prepacked(
                reinterpret_cast<const std::uint32_t*>(ff2_values),
                kDModel, kFfDim, tile_n * 8U, k_begin, lane, b);
            const auto sfb = load_sfb_pair(
                ff2_scales, kFfDim / 16U, tile_n * 8U, k_begin, lane);
#pragma unroll
            for (std::uint32_t row_tile = 0; row_tile < kRowTilesPerCta; ++row_tile) {
                std::uint32_t a[4];
                load_a_prepacked(
                    shared.hidden_fragments, kFfDim,
                    row_tile * kRowsPerCta, k_begin, lane, a);
                const auto sfa = load_sfa_pair(
                    shared.hidden_scales, kFfDim / 16U,
                    row_tile * kRowsPerCta, k_begin, lane);
                Mma::fma(
                    accumulators[row_tile][0], accumulators[row_tile][1],
                    accumulators[row_tile][2], accumulators[row_tile][3],
                    a[0], a[1], a[2], a[3], b[0], b[1],
                    accumulators[row_tile][0], accumulators[row_tile][1],
                    accumulators[row_tile][2], accumulators[row_tile][3], sfa, sfb);
            }
        }
        const std::uint32_t thread_0 = lane & 3U;
        const std::uint32_t thread_1 = lane >> 2U;
#pragma unroll
        for (int value = 0; value < 4; ++value) {
            const std::uint32_t linear = thread_0 * 32U + thread_1 +
                (static_cast<std::uint32_t>(value) & 1U) * 16U +
                (static_cast<std::uint32_t>(value) >> 1U) * 8U;
            const std::uint32_t row = linear % 16U;
            const std::uint32_t col = tile_n * 8U + linear / 16U;
#pragma unroll
            for (std::uint32_t row_tile = 0; row_tile < kRowTilesPerCta; ++row_tile) {
                const std::uint32_t global_row =
                    global_row_begin + row_tile * kRowsPerCta + row;
                if (global_row < rows) {
                    output[static_cast<std::size_t>(global_row) * kDModel + col] =
                        accumulators[row_tile][value] + ff2_bias[col];
                }
            }
        }
    }
}

}  // namespace stream1_sm120_nvfp4_fused
