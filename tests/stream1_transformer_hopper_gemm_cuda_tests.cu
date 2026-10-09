#include "stream1_transformer_hopper.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}

int main() {
    constexpr std::uint32_t M = 4096, N = 768, K = 256;
    std::vector<half> a(static_cast<std::size_t>(M) * K, __float2half(0.0f));
    std::vector<half> b_row(static_cast<std::size_t>(K) * N);
    std::vector<half> b_col(static_cast<std::size_t>(K) * N);
    std::vector<half> bias(N, __float2half(0.0f));
    a[0] = __float2half(1.0f);
    for (std::uint32_t k = 0; k < K; ++k) {
        for (std::uint32_t n = 0; n < N; ++n) {
            const half v = __float2half(static_cast<float>((k * 13U + n * 7U) % 31U) / 32.0f);
            b_row[static_cast<std::size_t>(k) * N + n] = v;
            b_col[static_cast<std::size_t>(n) * K + k] = v;
        }
    }
    half *da = nullptr, *db = nullptr, *dbias = nullptr, *dd = nullptr;
    check(cudaMalloc(&da, a.size() * sizeof(half)));
    check(cudaMalloc(&db, b_row.size() * sizeof(half)));
    check(cudaMalloc(&dbias, bias.size() * sizeof(half)));
    check(cudaMalloc(&dd, static_cast<std::size_t>(M) * N * sizeof(half)));
    check(cudaMemcpy(da, a.data(), a.size() * sizeof(half), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dbias, bias.data(), bias.size() * sizeof(half), cudaMemcpyHostToDevice));

    auto run = [&](const std::vector<half>& weights) {
        check(cudaMemcpy(db, weights.data(), weights.size() * sizeof(half), cudaMemcpyHostToDevice));
        beam::stream1_transformer_hopper_fp16_bias_activation<cutlass::epilogue::thread::Identity>(
            da, db, dbias, dd, M, K, N, nullptr);
        std::vector<half> first(N);
        check(cudaMemcpy(first.data(), dd, N * sizeof(half), cudaMemcpyDeviceToHost));
        double mae = 0.0;
        for (std::uint32_t n = 0; n < N; ++n) {
            mae += std::abs(__half2float(first[n]) - __half2float(b_row[n]));
        }
        return mae / N;
    };
    const double row_mae = run(b_row);
    const double col_mae = run(b_col);
    std::cout << "hopper_gemm_layout row_mae=" << row_mae << " col_mae=" << col_mae << "\n";
    if (!((row_mae == 0.0) != (col_mae == 0.0))) {
        throw std::runtime_error("exactly one Hopper B packing must match KxN semantics");
    }
    cudaFree(da); cudaFree(db); cudaFree(dbias); cudaFree(dd);

    // Reproduce the production FF1 large-M boundary. Every row has the same
    // input, so row zero must not depend on how many later rows are launched.
    constexpr std::uint32_t FF_M_SMALL = 384U * 57U;
    constexpr std::uint32_t FF_M_LARGE = 4096U * 57U;
    constexpr std::uint32_t FF_N = 1024U;
    std::vector<half> ff_weight(static_cast<std::size_t>(K) * FF_N);
    std::vector<half> ff_bias(FF_N);
    for (std::uint32_t n = 0; n < FF_N; ++n) {
        ff_bias[n] = __float2half(static_cast<float>(static_cast<int>(n % 9U) - 4) / 32.0f);
        for (std::uint32_t k = 0; k < K; ++k) {
            ff_weight[static_cast<std::size_t>(n) * K + k] =
                __float2half(static_cast<float>(static_cast<int>((k * 5U + n * 3U) % 17U) - 8) / 64.0f);
        }
    }
    std::vector<half> ff_input(static_cast<std::size_t>(FF_M_LARGE) * K, __float2half(0.0f));
    for (std::uint32_t m = 0; m < FF_M_LARGE; ++m) {
        for (std::uint32_t k = 0; k < K; ++k) {
            ff_input[static_cast<std::size_t>(m) * K + k] =
                __float2half(static_cast<float>(static_cast<int>(k % 13U) - 6) / 16.0f);
        }
    }
    half *dff_input = nullptr, *dff_weight = nullptr, *dff_bias = nullptr;
    half *dff_small = nullptr, *dff_large = nullptr;
    check(cudaMalloc(&dff_input, ff_input.size() * sizeof(half)));
    check(cudaMalloc(&dff_weight, ff_weight.size() * sizeof(half)));
    check(cudaMalloc(&dff_bias, ff_bias.size() * sizeof(half)));
    check(cudaMalloc(&dff_small, static_cast<std::size_t>(FF_M_SMALL) * FF_N * sizeof(half)));
    check(cudaMalloc(&dff_large, static_cast<std::size_t>(FF_M_LARGE) * FF_N * sizeof(half)));
    check(cudaMemcpy(dff_input, ff_input.data(), ff_input.size() * sizeof(half), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dff_weight, ff_weight.data(), ff_weight.size() * sizeof(half), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dff_bias, ff_bias.data(), ff_bias.size() * sizeof(half), cudaMemcpyHostToDevice));
    beam::stream1_transformer_hopper_fp16_bias_activation<cutlass::epilogue::thread::ReLu>(
        dff_input, dff_weight, dff_bias, dff_small, FF_M_SMALL, K, FF_N, nullptr);
    beam::stream1_transformer_hopper_fp16_bias_activation<cutlass::epilogue::thread::ReLu>(
        dff_input, dff_weight, dff_bias, dff_large, FF_M_LARGE, K, FF_N, nullptr);
    std::vector<half> ff_small_first(FF_N), ff_large_first(FF_N);
    check(cudaMemcpy(ff_small_first.data(), dff_small, FF_N * sizeof(half), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(ff_large_first.data(), dff_large, FF_N * sizeof(half), cudaMemcpyDeviceToHost));
    std::uint32_t ff_mismatch = 0U;
    std::uint32_t ff_oracle_mismatch = 0U;
    for (std::uint32_t n = 0; n < FF_N; ++n) {
        ff_mismatch += __half2float(ff_small_first[n]) != __half2float(ff_large_first[n]);
        float expected = __half2float(ff_bias[n]);
        for (std::uint32_t k = 0; k < K; ++k) {
            expected += __half2float(ff_input[k]) *
                __half2float(ff_weight[static_cast<std::size_t>(n) * K + k]);
        }
        expected = std::max(expected, 0.0f);
        ff_oracle_mismatch += std::abs(__half2float(ff_small_first[n]) - expected) > 0.01f;
    }
    std::cout << "hopper_ff1_large_m first_row_mismatch=" << ff_mismatch
              << " oracle_mismatch=" << ff_oracle_mismatch << "\n";
    if (ff_mismatch != 0U) throw std::runtime_error("Hopper FF1 output depends on trailing M rows");
    if (ff_oracle_mismatch != 0U) throw std::runtime_error("Hopper FF1 fused bias orientation is incorrect");
    cudaFree(dff_input); cudaFree(dff_weight); cudaFree(dff_bias);
    cudaFree(dff_small); cudaFree(dff_large);
    return 0;
}
