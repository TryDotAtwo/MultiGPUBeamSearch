#include "stream1_transformer_sm120_nvfp4_cutlass_dsm_fused.cuh"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace {

using namespace cute;
using namespace stream1_sm120_nvfp4_cutlass_dsm;

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main(int argc, char** argv) {
    using Handoff = BulkDsmFf2Collective;
    constexpr int m = 128;
    constexpr int n = 256;
    constexpr int k = 1024;
    const auto problem_shape = make_shape(m, n, k, 1);
    const typename Ff2Mainloop::StrideA stride_a =
        make_stride(std::int64_t{k}, _1{}, std::int64_t{m * k});
    const typename Ff2Mainloop::StrideB stride_b =
        make_stride(std::int64_t{k}, _1{}, std::int64_t{n * k});
    using ScaleConfig = typename Ff2Mainloop::Sm1xxBlkScaledConfig;
    const typename Ff2Mainloop::LayoutSFA layout_sfa =
        ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
    const typename Ff2Mainloop::LayoutSFB layout_sfb =
        ScaleConfig::tile_atom_to_shape_SFB(problem_shape);

    cutlass::float_e2m1_t *unused_a = nullptr, *b = nullptr;
    ElementScale *unused_sfa = nullptr, *sfb = nullptr;
    const int clusters = argc > 1 ? std::atoi(argv[1]) : 1;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 0;
    require(clusters > 0 && clusters <= 512,
            "clusters must be in [1, 512]");
    require(iterations >= 0, "iterations must be non-negative");
    float* rank_checksums = nullptr;
    const std::size_t a_bytes = static_cast<std::size_t>(m) * k / 2U;
    const std::size_t b_bytes = static_cast<std::size_t>(n) * k / 2U;
    const std::size_t sfa_bytes = static_cast<std::size_t>(
        size(filter_zeros(layout_sfa)));
    const std::size_t sfb_bytes = static_cast<std::size_t>(
        size(filter_zeros(layout_sfb)));
    require(cudaMalloc(&unused_a, a_bytes) == cudaSuccess,
            "cudaMalloc unused A failed");
    require(cudaMalloc(&b, b_bytes) == cudaSuccess, "cudaMalloc B failed");
    require(cudaMalloc(&unused_sfa, sfa_bytes) == cudaSuccess,
            "cudaMalloc unused SFA failed");
    require(cudaMalloc(&sfb, sfb_bytes) == cudaSuccess,
            "cudaMalloc SFB failed");
    require(cudaMalloc(&rank_checksums,
                       2U * clusters * sizeof(float)) == cudaSuccess,
            "cudaMalloc checksums failed");
    cudaMemset(unused_a, 0, a_bytes);
    cudaMemset(unused_sfa, 0, sfa_bytes);
    cudaMemset(b, 0x11, b_bytes);
    cudaMemset(sfb, ElementScale(1.0F).raw(), sfb_bytes);
    cudaMemset(rank_checksums, 0, 2U * clusters * sizeof(float));

    typename Ff2Mainloop::Arguments arguments{
        unused_a, stride_a, b, stride_b,
        unused_sfa, layout_sfa, sfb, layout_sfb};
    require(Ff2Mainloop::can_implement(problem_shape, arguments),
            "Ff2Mainloop rejected bulk-DSM descriptor inputs");
    const auto params = Ff2Mainloop::to_underlying_arguments(
        problem_shape, arguments, nullptr);
    require(cudaFuncSetAttribute(
                bulk_dsm_ff2_collective_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                sizeof(Handoff::SharedStorage)) == cudaSuccess,
            "bulk-DSM collective shared-memory attribute failed");
    bulk_dsm_ff2_collective_kernel<<<
        2 * clusters, Handoff::kThreads, sizeof(Handoff::SharedStorage)>>>(
            params, rank_checksums);
    require(cudaGetLastError() == cudaSuccess,
            "bulk-DSM collective launch failed");
    const cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        std::cerr << "bulk-DSM collective runtime failed: "
                  << cudaGetErrorString(status) << '\n';
        return EXIT_FAILURE;
    }
    float host[1024] = {};
    cudaMemcpy(host, rank_checksums,
               2U * clusters * sizeof(float), cudaMemcpyDeviceToHost);
    for (int cluster = 0; cluster < clusters; ++cluster) {
        require(host[2 * cluster] > 0.0F,
                "bulk-DSM collective checksum is zero");
        require(host[2 * cluster] == host[2 * cluster + 1],
                "bulk-DSM collective differs between FF2 output halves");
        require(host[2 * cluster] == host[0],
                "bulk-DSM collective differs between clusters");
    }
    std::cout << "stream1_transformer_sm120_nvfp4_cutlass_bulk_dsm=pass"
              << " rank0_checksum=" << host[0]
              << " rank1_checksum=" << host[1]
              << " global_hidden_bytes=" << Plan::kGlobalHiddenBytes
              << " transaction_bytes=" << Handoff::kTotalTransactionBytes
              << '\n';
    if (iterations > 0) {
        constexpr int warmups = 20;
        for (int i = 0; i < warmups; ++i) {
            bulk_dsm_ff2_collective_kernel<<<
                2 * clusters, Handoff::kThreads,
                sizeof(Handoff::SharedStorage)>>>(params, rank_checksums);
        }
        require(cudaDeviceSynchronize() == cudaSuccess,
                "bulk-DSM collective warmup failed");
        cudaEvent_t start{}, stop{};
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < iterations; ++i) {
            bulk_dsm_ff2_collective_kernel<<<
                2 * clusters, Handoff::kThreads,
                sizeof(Handoff::SharedStorage)>>>(params, rank_checksums);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float total_ms = 0.0F;
        cudaEventElapsedTime(&total_ms, start, stop);
        const double launch_ms = total_ms / iterations;
        const double flops = 2.0 * clusters * m * n * k;
        const double tflops = flops / (launch_ms * 1.0e9);
        std::cout << "stream1_transformer_sm120_nvfp4_cutlass_bulk_dsm_bench"
                  << " clusters=" << clusters
                  << " iterations=" << iterations
                  << " launch_ms=" << std::fixed << std::setprecision(6)
                  << launch_ms
                  << " ff2_tflops=" << std::setprecision(3) << tflops
                  << '\n';

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    cudaFree(rank_checksums);
    cudaFree(unused_a);
    cudaFree(b);
    cudaFree(unused_sfa);
    cudaFree(sfb);
    return EXIT_SUCCESS;
}
