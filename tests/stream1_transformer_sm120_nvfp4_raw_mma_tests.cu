#include "stream1_transformer_sm120_nvfp4_raw_mma.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr std::uint32_t kM = 16U;
constexpr std::uint32_t kN = 1024U;
constexpr std::uint32_t kK = 256U;

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

std::uint8_t raw_for(std::uint32_t seed) {
    return static_cast<std::uint8_t>(seed & 15U);
}

float scale_for(std::uint32_t seed) {
    constexpr float choices[] = {0.03125F, 0.04296875F, 0.0625F, 0.125F};
    return choices[(seed * 13U + 5U) & 3U];
}

float decode(std::uint8_t raw) {
    return static_cast<float>(cutlass::float_e2m1_t::bitcast(raw));
}

void set_nibble(std::vector<std::uint8_t>& packed, std::size_t logical, std::uint8_t raw) {
    packed[logical / 2U] = static_cast<std::uint8_t>(
        packed[logical / 2U] | ((raw & 0x0fU) << ((logical & 1U) * 4U)));
}

__global__ void raw_gemm_kernel(
    const std::uint8_t* a,
    const std::uint8_t* scales_a,
    const std::uint8_t* b,
    const std::uint8_t* scales_b,
    float* c) {
    using namespace stream1_sm120_nvfp4_raw;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    constexpr std::uint32_t tiles_n = kN / 8U;
    for (std::uint32_t tile_n = warp; tile_n < tiles_n; tile_n += blockDim.x / 32U) {
      constexpr std::uint32_t tile_m = 0U;
      float accumulators[4] = {0.0F, 0.0F, 0.0F, 0.0F};
      for (std::uint32_t k_begin = 0U; k_begin < kK; k_begin += 64U) {
        std::uint32_t a_registers[4];
        std::uint32_t b_registers[2];
        load_a_unit_scale(a, kK, tile_m * 16U, k_begin, lane, a_registers);
        load_b_unit_scale(b, kN, tile_n * 8U, k_begin, lane, b_registers);
        const auto scale_a = load_sfa_pair(
            scales_a, kK / 16U, tile_m * 16U, k_begin, lane);
        const auto scale_b = load_sfb_pair(
            scales_b, kK / 16U, tile_n * 8U, k_begin, lane);
        Mma::fma(
            accumulators[0], accumulators[1], accumulators[2], accumulators[3],
            a_registers[0], a_registers[1], a_registers[2], a_registers[3],
            b_registers[0], b_registers[1],
            accumulators[0], accumulators[1], accumulators[2], accumulators[3],
            scale_a, scale_b);
      }
      store_c(c, kN, tile_m * 16U, tile_n * 8U, lane, accumulators);
    }
}

}  // namespace

int main() {
    std::vector<std::uint8_t> a(kM * kK / 2U, 0U);
    std::vector<std::uint8_t> b(kK * kN / 2U, 0U);
    std::vector<std::uint8_t> scales_a(kM * (kK / 16U));
    std::vector<std::uint8_t> scales_b(kN * (kK / 16U));
    std::vector<float> logical_scales_a(scales_a.size());
    std::vector<float> logical_scales_b(scales_b.size());
    for (std::uint32_t m = 0; m < kM; ++m) {
        for (std::uint32_t group = 0; group < kK / 16U; ++group) {
            const float logical = scale_for(m * 17U + group * 7U);
            const std::size_t index = static_cast<std::size_t>(m) * (kK / 16U) + group;
            logical_scales_a[index] = logical;
            scales_a[index] = cutlass::float_ue4m3_t(logical).raw();
        }
    }
    for (std::uint32_t n = 0; n < kN; ++n) {
        for (std::uint32_t group = 0; group < kK / 16U; ++group) {
            const float logical = scale_for(n * 11U + group * 19U + 3U);
            const std::size_t index = static_cast<std::size_t>(n) * (kK / 16U) + group;
            logical_scales_b[index] = logical;
            scales_b[index] = cutlass::float_ue4m3_t(logical).raw();
        }
    }
    for (std::uint32_t m = 0; m < kM; ++m) {
        for (std::uint32_t k = 0; k < kK; ++k) {
            set_nibble(a, static_cast<std::size_t>(m) * kK + k, raw_for(m * 3U + k * 5U));
        }
    }
    for (std::uint32_t k = 0; k < kK; ++k) {
        for (std::uint32_t n = 0; n < kN; ++n) {
            set_nibble(b, static_cast<std::size_t>(k) * kN + n, raw_for(k * 7U + n * 11U));
        }
    }
    std::vector<float> reference(kM * kN, 0.0F);
    for (std::uint32_t m = 0; m < kM; ++m) {
        for (std::uint32_t n = 0; n < kN; ++n) {
            float sum = 0.0F;
            for (std::uint32_t k = 0; k < kK; ++k) {
                const std::uint32_t group = k / 16U;
                sum += decode(raw_for(m * 3U + k * 5U)) *
                       logical_scales_a[static_cast<std::size_t>(m) * (kK / 16U) + group] *
                       decode(raw_for(k * 7U + n * 11U)) *
                       logical_scales_b[static_cast<std::size_t>(n) * (kK / 16U) + group];
            }
            reference[static_cast<std::size_t>(m) * kN + n] = sum;
        }
    }

    std::uint8_t* device_a = nullptr;
    std::uint8_t* device_b = nullptr;
    std::uint8_t* device_scales_a = nullptr;
    std::uint8_t* device_scales_b = nullptr;
    float* device_c = nullptr;
    cudaMalloc(&device_a, a.size());
    cudaMalloc(&device_b, b.size());
    cudaMalloc(&device_scales_a, scales_a.size());
    cudaMalloc(&device_scales_b, scales_b.size());
    cudaMalloc(&device_c, reference.size() * sizeof(float));
    cudaMemcpy(device_a, a.data(), a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b.data(), b.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(device_scales_a, scales_a.data(), scales_a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(device_scales_b, scales_b.data(), scales_b.size(), cudaMemcpyHostToDevice);
    raw_gemm_kernel<<<1, 256>>>(
        device_a, device_scales_a, device_b, device_scales_b, device_c);
    require(cudaDeviceSynchronize() == cudaSuccess, "raw SM120 MMA kernel failed");
    std::vector<float> actual(reference.size());
    cudaMemcpy(actual.data(), device_c, actual.size() * sizeof(float), cudaMemcpyDeviceToHost);
    float max_abs_error = 0.0F;
    std::size_t mismatch_count = 0U;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        max_abs_error = std::max(max_abs_error, std::abs(actual[index] - reference[index]));
        if (actual[index] != reference[index]) {
            if (mismatch_count < 16U) {
                std::cerr << "mismatch index=" << index
                          << " actual=" << actual[index]
                          << " reference=" << reference[index] << '\n';
            }
            ++mismatch_count;
        }
    }
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_scales_a);
    cudaFree(device_scales_b);
    cudaFree(device_c);
    std::cout << "raw_mma_diagnostic max_abs_error=" << max_abs_error
              << " mismatch_count=" << mismatch_count << '\n';
    require(max_abs_error == 0.0F, "raw SM120 MMA fragment mapping mismatch");
    std::cout << "stream1_transformer_sm120_nvfp4_raw_mma_tests=pass"
              << " max_abs_error=" << max_abs_error << '\n';
    return EXIT_SUCCESS;
}
