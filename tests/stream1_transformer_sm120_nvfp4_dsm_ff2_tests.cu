#include "stream1_transformer_sm120_nvfp4_dsm_ff2.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

using namespace cute;
using namespace stream1_sm120_nvfp4_cutlass_dsm;
using namespace stream1_sm120_nvfp4_dsm_ff2;

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

int env_positive_int(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return fallback;
    }
    const int value = std::atoi(raw);
    return value > 0 ? value : fallback;
}

}  // namespace

int main() {
    static_assert(BOnlyCollective::kLoadsGlobalA == false);
    static_assert(BOnlyCollective::kLoadsGlobalSFA == false);
    static_assert(BOnlyCollective::kLoadsGlobalB == true);
    static_assert(BOnlyCollective::kLoadsGlobalSFB == true);
    static_assert(BOnlyCollective::kTransactionBytes ==
                  Ff2Mainloop::TmaTransactionBytesNK);
    static_assert(BOnlyCollective::kTransactionBytes <
                  Ff2Mainloop::TmaTransactionBytes);

    constexpr int m = 128;
    constexpr int n = 128;
    constexpr int k = 128;
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

    cutlass::float_e2m1_t *a = nullptr, *b = nullptr;
    ElementScale *sfa = nullptr, *sfb = nullptr;
    unsigned long long* checksum = nullptr;
    const std::size_t ab_bytes = static_cast<std::size_t>(m) * k / 2U;
    const std::size_t sfa_bytes = static_cast<std::size_t>(
        size(filter_zeros(layout_sfa)));
    const std::size_t sfb_bytes = static_cast<std::size_t>(
        size(filter_zeros(layout_sfb)));
    require(cudaMalloc(&a, ab_bytes) == cudaSuccess, "cudaMalloc A failed");
    require(cudaMalloc(&b, ab_bytes) == cudaSuccess, "cudaMalloc B failed");
    require(cudaMalloc(&sfa, sfa_bytes) == cudaSuccess,
            "cudaMalloc SFA failed");
    require(cudaMalloc(&sfb, sfb_bytes) == cudaSuccess,
            "cudaMalloc SFB failed");
    require(cudaMalloc(&checksum, sizeof(*checksum)) == cudaSuccess,
            "cudaMalloc checksum failed");
    constexpr std::uint8_t value_pattern = 0x11U;
    const std::uint8_t scale_pattern = ElementScale(1.0F).raw();
    cudaMemset(a, 0, ab_bytes);
    cudaMemset(b, value_pattern, ab_bytes);
    cudaMemset(sfa, 0, sfa_bytes);
    cudaMemset(sfb, scale_pattern, sfb_bytes);
    cudaMemset(checksum, 0, sizeof(*checksum));

    typename Ff2Mainloop::Arguments arguments{
        a, stride_a, b, stride_b, sfa, layout_sfa, sfb, layout_sfb};
    require(Ff2Mainloop::can_implement(problem_shape, arguments),
            "Ff2Mainloop rejected the descriptor inputs");
    const auto params = Ff2Mainloop::to_underlying_arguments(
        problem_shape, arguments, nullptr);
    b_sfb_tma_contract_kernel<<<
        1, BOnlyCollective::kThreads, sizeof(BOnlyCollective::SharedStorage)>>>(
        params, checksum);
    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        std::cerr << "B/SFB-only TMA launch failed: "
                  << cudaGetErrorString(launch_status) << '\n';
        return EXIT_FAILURE;
    }
    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
        std::cerr << "B/SFB-only TMA runtime failed: "
                  << cudaGetErrorString(sync_status) << '\n';
        return EXIT_FAILURE;
    }
    unsigned long long actual = 0;
    cudaMemcpy(&actual, checksum, sizeof(actual), cudaMemcpyDeviceToHost);
    const unsigned long long expected =
        static_cast<unsigned long long>(BOnlyCollective::kBBytes) *
            value_pattern +
        static_cast<unsigned long long>(BOnlyCollective::kSfbBytes) *
            scale_pattern;
    require(actual == expected, "B/SFB-only TMA checksum mismatch");
    std::cout << "stream1_transformer_sm120_nvfp4_dsm_ff2_tma=pass"
              << " transaction_bytes=" << BOnlyCollective::kTransactionBytes
              << " checksum=" << actual << '\n';

    unsigned long long* copy_checksums = nullptr;
    require(cudaMalloc(&copy_checksums, 2U * sizeof(unsigned long long)) ==
                cudaSuccess,
            "cudaMalloc DSM copy checksums failed");
    cudaMemset(copy_checksums, 0, 2U * sizeof(unsigned long long));
    dsm_to_local_copy_contract_kernel<<<
        2, BOnlyCollective::kThreads, sizeof(DsmASlot)>>>(copy_checksums);
    require(cudaGetLastError() == cudaSuccess,
            "DSM-to-local copy launch failed");
    const cudaError_t copy_status = cudaDeviceSynchronize();
    if (copy_status != cudaSuccess) {
        std::cerr << "DSM-to-local copy runtime failed: "
                  << cudaGetErrorString(copy_status) << '\n';
        return EXIT_FAILURE;
    }
    unsigned long long host_copy_checksums[2] = {};
    cudaMemcpy(host_copy_checksums, copy_checksums,
               sizeof(host_copy_checksums), cudaMemcpyDeviceToHost);
    require(host_copy_checksums[0] == host_copy_checksums[1],
            "DSM-to-local copy differs by CTA rank");
    require(host_copy_checksums[0] != 0U,
            "DSM-to-local copy checksum is zero");
    std::cout << "stream1_transformer_sm120_nvfp4_dsm_ff2_copy=pass"
              << " checksum=" << host_copy_checksums[0] << '\n';

    float* local_mma_checksum = nullptr;
    require(cudaMalloc(&local_mma_checksum, sizeof(float)) == cudaSuccess,
            "cudaMalloc local MMA checksum failed");
    cudaMemset(local_mma_checksum, 0, sizeof(float));
    local_a_mma_contract_kernel<<<
        1, BOnlyCollective::kThreads,
        sizeof(RemoteAMmaSharedStorage)>>>(params, local_mma_checksum);
    require(cudaGetLastError() == cudaSuccess,
            "local A/SFA MMA launch failed");
    const cudaError_t local_mma_status = cudaDeviceSynchronize();
    if (local_mma_status != cudaSuccess) {
        std::cerr << "local A/SFA MMA runtime failed: "
                  << cudaGetErrorString(local_mma_status) << '\n';
        return EXIT_FAILURE;
    }
    float host_local_mma_checksum = 0.0F;
    cudaMemcpy(&host_local_mma_checksum, local_mma_checksum, sizeof(float),
               cudaMemcpyDeviceToHost);
    require(host_local_mma_checksum > 0.0F,
            "local A/SFA MMA checksum is zero");
    std::cout << "stream1_transformer_sm120_nvfp4_dsm_ff2_local_mma=pass"
              << " checksum=" << host_local_mma_checksum << '\n';

    // The remote gate is opt-in while its synchronization protocol is under
    // test, so a failing cluster experiment cannot take down the Molab kernel
    // before the independent local-MMA evidence is recorded.
    const bool run_remote = std::getenv("RUN_REMOTE_DSM_FF2") != nullptr;
    const bool run_cluster_local =
        std::getenv("RUN_CLUSTER_LOCAL_FF2") != nullptr;
    const bool run_cluster_manual =
        std::getenv("RUN_CLUSTER_MANUAL_FF2") != nullptr;
    const bool run_multislice =
        std::getenv("RUN_MULTISLICE_DSM_FF2") != nullptr;
    if (!run_remote && !run_cluster_local && !run_cluster_manual) {
        return EXIT_SUCCESS;
    }

    constexpr int max_benchmark_clusters = 256;
    const int benchmark_clusters = env_positive_int(
        "DSM_FF2_BENCH_CLUSTERS", 1);
    require(benchmark_clusters <= max_benchmark_clusters,
            "DSM_FF2_BENCH_CLUSTERS exceeds test capacity");
    float* rank_checksums = nullptr;
    require(cudaMalloc(&rank_checksums,
                       2U * max_benchmark_clusters * sizeof(float)) == cudaSuccess,
            "cudaMalloc rank checksums failed");
    cudaMemset(rank_checksums, 0,
               2U * max_benchmark_clusters * sizeof(float));
    remote_a_mma_contract_kernel<<<
        2, BOnlyCollective::kThreads,
        sizeof(RemoteAMmaSharedStorage)>>>(
            params, rank_checksums, run_remote, !run_cluster_manual);
    const cudaError_t mma_launch_status = cudaGetLastError();
    if (mma_launch_status != cudaSuccess) {
        std::cerr << "remote A/SFA MMA launch failed: "
                  << cudaGetErrorString(mma_launch_status) << '\n';
        return EXIT_FAILURE;
    }
    const cudaError_t mma_sync_status = cudaDeviceSynchronize();
    if (mma_sync_status != cudaSuccess) {
        std::cerr << "remote A/SFA MMA runtime failed: "
                  << cudaGetErrorString(mma_sync_status) << '\n';
        return EXIT_FAILURE;
    }
    float host_rank_checksums[2] = {};
    cudaMemcpy(host_rank_checksums, rank_checksums,
               sizeof(host_rank_checksums), cudaMemcpyDeviceToHost);
    require(host_rank_checksums[0] > 0.0F,
            "remote A/SFA MMA checksum is zero");
    require(host_rank_checksums[0] == host_rank_checksums[1],
            "remote A/SFA MMA differs by CTA rank");
    std::cout << (run_remote
            ? "stream1_transformer_sm120_nvfp4_dsm_ff2_mma=pass"
            : (run_cluster_manual
                ? "stream1_transformer_sm120_nvfp4_cluster_manual_ff2_mma=pass"
                : "stream1_transformer_sm120_nvfp4_cluster_local_ff2_mma=pass"))
              << " rank0_checksum=" << host_rank_checksums[0]
              << " rank1_checksum=" << host_rank_checksums[1] << '\n';

    if (run_multislice) {
        cudaMemset(rank_checksums, 0, 2U * sizeof(float));
        remote_a_mma_multislice_contract_kernel<<<
            2, 256,
            sizeof(RemoteAMmaSharedStorage)>>>(
                params, rank_checksums, run_remote);
        const cudaError_t multislice_launch_status = cudaGetLastError();
        if (multislice_launch_status != cudaSuccess) {
            std::cerr << "multislice DSM FF2 launch failed: "
                      << cudaGetErrorString(multislice_launch_status) << '\n';
            return EXIT_FAILURE;
        }
        require(cudaDeviceSynchronize() == cudaSuccess,
                "multislice DSM FF2 runtime failed");
        cudaMemcpy(host_rank_checksums, rank_checksums,
                   sizeof(host_rank_checksums), cudaMemcpyDeviceToHost);
        require(host_rank_checksums[0] > 0.0F,
                "multislice DSM FF2 checksum is zero");
        require(host_rank_checksums[0] == host_rank_checksums[1],
                "multislice DSM FF2 differs by CTA rank");
        std::cout << "stream1_transformer_sm120_nvfp4_dsm_ff2_multislice=pass"
                  << " mode=" << (run_remote ? "remote_handoff" : "local")
                  << " slices=" << Contract::kSlices
                  << " rank0_checksum=" << host_rank_checksums[0]
                  << " rank1_checksum=" << host_rank_checksums[1] << '\n';
    }

    const int benchmark_iterations = env_positive_int(
        "DSM_FF2_BENCH_ITERS", 0);
    if (benchmark_iterations > 0 && !run_cluster_manual) {
        constexpr int warmup_iterations = 20;
        auto launch = [&](bool remote) {
            if (run_multislice) {
                remote_a_mma_multislice_contract_kernel<<<
                    2 * benchmark_clusters, 256,
                    sizeof(RemoteAMmaSharedStorage)>>>(
                        params, rank_checksums, remote);
            } else {
                remote_a_mma_contract_kernel<<<
                    2, BOnlyCollective::kThreads,
                    sizeof(RemoteAMmaSharedStorage)>>>(
                        params, rank_checksums, remote, true);
            }
        };
        for (int i = 0; i < warmup_iterations; ++i) {
            launch(run_remote);
        }
        require(cudaDeviceSynchronize() == cudaSuccess,
                "DSM FF2 benchmark warmup failed");
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        require(cudaEventCreate(&start) == cudaSuccess,
                "cudaEventCreate start failed");
        require(cudaEventCreate(&stop) == cudaSuccess,
                "cudaEventCreate stop failed");
        require(cudaEventRecord(start) == cudaSuccess,
                "cudaEventRecord start failed");
        for (int i = 0; i < benchmark_iterations; ++i) {
            launch(run_remote);
        }
        require(cudaEventRecord(stop) == cudaSuccess,
                "cudaEventRecord stop failed");
        require(cudaEventSynchronize(stop) == cudaSuccess,
                "DSM FF2 benchmark synchronize failed");
        float elapsed_ms = 0.0F;
        require(cudaEventElapsedTime(&elapsed_ms, start, stop) == cudaSuccess,
                "cudaEventElapsedTime failed");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        const double launch_us =
            static_cast<double>(elapsed_ms) * 1000.0 /
            static_cast<double>(benchmark_iterations);
        constexpr double flops_per_cluster =
            2.0 * static_cast<double>(Contract::kSlices) * 2.0 *
            128.0 * 128.0 * 128.0;
        const double tflops = run_multislice
            ? flops_per_cluster * static_cast<double>(benchmark_clusters) /
                (launch_us * 1.0e6)
            : 0.0;
        std::cout << "stream1_transformer_sm120_nvfp4_dsm_ff2_benchmark=pass"
                  << " mode=" << (run_remote ? "remote_handoff" : "local")
                  << " slices=" << (run_multislice ? Contract::kSlices : 1U)
                  << " clusters=" << benchmark_clusters
                  << " iterations=" << benchmark_iterations
                  << " launch_us=" << launch_us
                  << " tflops=" << tflops << '\n';
    }
    return EXIT_SUCCESS;
}
