#include "stream1_transformer_hopper.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
    return 0;
}
