#include "stream1_transformer_sm120_nvfp4_raw_mma.hpp"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr std::uint32_t kM = 16U;
constexpr std::uint32_t kN = 128U;
constexpr std::uint32_t kK = 256U;

struct alignas(128) DsmOperand {
    std::uint8_t values[kM * kK / 2U];
    std::uint8_t scales[kM * (kK / 16U)];
};

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void set_nibble(std::vector<std::uint8_t>& packed, std::size_t logical,
                std::uint8_t raw) {
    packed[logical / 2U] = static_cast<std::uint8_t>(
        packed[logical / 2U] |
        ((raw & 0x0fU) << ((logical & 1U) * 4U)));
}

__global__ __cluster_dims__(2, 1, 1) void dsm_operand_kernel(
    const std::uint8_t* global_a,
    const std::uint8_t* global_scales_a,
    const std::uint8_t* b,
    const std::uint8_t* scales_b,
    float* output) {
    namespace cg = cooperative_groups;
    using namespace stream1_sm120_nvfp4_raw;
    __shared__ DsmOperand local;
    const auto cluster = cg::this_cluster();
    const std::uint32_t rank = cluster.block_rank();
    if (rank == 0U) {
        for (std::uint32_t i = threadIdx.x; i < sizeof(local.values);
             i += blockDim.x) {
            local.values[i] = global_a[i];
        }
        for (std::uint32_t i = threadIdx.x; i < sizeof(local.scales);
             i += blockDim.x) {
            local.scales[i] = global_scales_a[i];
        }
    }
    cluster.sync();
    const auto* remote_values = cluster.map_shared_rank(local.values, 0);
    const auto* remote_scales = cluster.map_shared_rank(local.scales, 0);

    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    for (std::uint32_t tile_n = warp; tile_n < kN / 8U;
         tile_n += blockDim.x / 32U) {
        float accumulators[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        for (std::uint32_t k_begin = 0U; k_begin < kK; k_begin += 64U) {
            std::uint32_t a_registers[4];
            std::uint32_t b_registers[2];
            load_a_unit_scale(remote_values, kK, 0U, k_begin, lane,
                              a_registers);
            load_b_unit_scale(b, kN, tile_n * 8U, k_begin, lane,
                              b_registers);
            const auto scale_a = load_sfa_pair(
                remote_scales, kK / 16U, 0U, k_begin, lane);
            const auto scale_b = load_sfb_pair(
                scales_b, kK / 16U, tile_n * 8U, k_begin, lane);
            Mma::fma(
                accumulators[0], accumulators[1], accumulators[2],
                accumulators[3], a_registers[0], a_registers[1],
                a_registers[2], a_registers[3], b_registers[0],
                b_registers[1], accumulators[0], accumulators[1],
                accumulators[2], accumulators[3], scale_a, scale_b);
        }
        store_c(output + static_cast<std::size_t>(rank) * kM * kN,
                kN, 0U, tile_n * 8U, lane, accumulators);
    }
    cluster.sync();
}

}  // namespace

int main() {
    std::vector<std::uint8_t> a(kM * kK / 2U, 0U);
    std::vector<std::uint8_t> b(kK * kN / 2U, 0U);
    std::vector<std::uint8_t> scales_a(kM * (kK / 16U));
    std::vector<std::uint8_t> scales_b(kN * (kK / 16U));
    for (std::uint32_t m = 0; m < kM; ++m) {
        for (std::uint32_t k = 0; k < kK; ++k) {
            set_nibble(a, static_cast<std::size_t>(m) * kK + k,
                       static_cast<std::uint8_t>((m * 3U + k * 5U) & 15U));
        }
    }
    for (std::uint32_t k = 0; k < kK; ++k) {
        for (std::uint32_t n = 0; n < kN; ++n) {
            set_nibble(b, static_cast<std::size_t>(k) * kN + n,
                       static_cast<std::uint8_t>((k * 7U + n * 11U) & 15U));
        }
    }
    std::fill(scales_a.begin(), scales_a.end(),
              cutlass::float_ue4m3_t(1.0F).raw());
    std::fill(scales_b.begin(), scales_b.end(),
              cutlass::float_ue4m3_t(1.0F).raw());

    std::uint8_t *da = nullptr, *db = nullptr, *dsa = nullptr, *dsb = nullptr;
    float* dc = nullptr;
    cudaMalloc(&da, a.size());
    cudaMalloc(&db, b.size());
    cudaMalloc(&dsa, scales_a.size());
    cudaMalloc(&dsb, scales_b.size());
    cudaMalloc(&dc, 2U * kM * kN * sizeof(float));
    cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dsa, scales_a.data(), scales_a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dsb, scales_b.data(), scales_b.size(), cudaMemcpyHostToDevice);
    dsm_operand_kernel<<<2, 256>>>(da, dsa, db, dsb, dc);
    require(cudaDeviceSynchronize() == cudaSuccess,
            "SM120 DSM operand kernel failed");
    std::vector<float> output(2U * kM * kN);
    cudaMemcpy(output.data(), dc, output.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    float max_rank_error = 0.0F;
    for (std::size_t i = 0; i < kM * kN; ++i) {
        max_rank_error = std::max(
            max_rank_error, std::abs(output[i] - output[kM * kN + i]));
    }
    std::cout << "stream1_transformer_sm120_nvfp4_dsm_operand=pass"
              << " max_rank_error=" << max_rank_error << '\n';
    require(max_rank_error == 0.0F, "remote DSM MMA differs by CTA rank");
    cudaFree(da);
    cudaFree(db);
    cudaFree(dsa);
    cudaFree(dsb);
    cudaFree(dc);
    return EXIT_SUCCESS;
}
