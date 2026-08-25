#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#include "gemm_testbed_3x.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace sm120_sparse_mxfp8_bench {

using namespace cute;

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
using LayoutSFDTag = cutlass::layout::RowMajor;
#if defined(BEAM_BENCH_NVFP4)
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
#else
using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
#endif
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementSF = cutlass::float_ue8m0_t;
using ElementAccumulator = float;
using ElementCompute = float;
using ProblemShape = Shape<int, int, int, int>;
using ClusterShape = Shape<_1, _1, _1>;
using MainloopTileShape = Shape<_128, _128, _256>;
using EpilogueTileShape = Shape<_128, _128, _256>;
using ArchTag = cutlass::arch::Sm120;
using EpilogueSchedule = cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120;
#if defined(BEAM_BENCH_NVFP4)
using KernelSchedule = cutlass::gemm::KernelSparseTmaWarpSpecializedNvf4Sm120;
constexpr int kAlignmentA = 256;
constexpr int kAlignmentB = 128;
#else
using KernelSchedule = cutlass::gemm::KernelSparseTmaWarpSpecializedMxf8f6f4Acc2x4Sm120;
constexpr int kAlignmentA = 32;
constexpr int kAlignmentB = 16;
#endif
constexpr int kAlignmentC = 1;
constexpr int kAlignmentD = 4;
constexpr int kScaleVector = 64;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    cutlass::arch::OpClassTensorOp,
    EpilogueTileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    LayoutCTag,
    kAlignmentC,
    ElementD,
    LayoutDTag,
    kAlignmentD,
    EpilogueSchedule>::CollectiveOp;

using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    cutlass::arch::OpClassBlockScaledSparseTensorOp,
    ElementA,
    LayoutATag,
    kAlignmentA,
    ElementB,
    LayoutBTag,
    kAlignmentB,
    ElementAccumulator,
    MainloopTileShape,
    ClusterShape,
    StageCount,
    KernelSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::StreamKScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using Testbed = test::gemm::device::Testbed3x<Gemm>;

double benchmark(int m, int n, int k, int iterations) {
    ProblemShape problem{m, n, k, 1};
    Testbed testbed;
    auto& impl = testbed.impl_;
    if (!impl.sufficient() || !impl.initialize(problem, 1.0f, 0.0f)) {
        throw std::runtime_error("structured-sparse fixture initialization failed");
    }

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    typename GemmKernel::TileScheduler::Arguments scheduler_args{};
    auto mainloop_args = impl.collective_mma_inputs.to_args();
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        mainloop_args,
        impl.collective_epilogue.to_args(problem),
        hw_info,
        scheduler_args};

    Gemm gemm;
    if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
        throw std::runtime_error("structured-sparse GEMM cannot implement shape");
    }
    cutlass::device_memory::allocation<std::uint8_t> workspace(
        Gemm::get_workspace_size(arguments));
    if (gemm.initialize(arguments, workspace.get()) != cutlass::Status::kSuccess) {
        throw std::runtime_error("structured-sparse GEMM initialization failed");
    }
    for (int i = 0; i < 10; ++i) {
        if (gemm.run() != cutlass::Status::kSuccess) {
            throw std::runtime_error("structured-sparse GEMM warmup failed");
        }
    }
    cudaDeviceSynchronize();

    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        if (gemm.run() != cutlass::Status::kSuccess) {
            throw std::runtime_error("structured-sparse GEMM launch failed");
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    const double ms = static_cast<double>(total_ms) / iterations;
    const double dense_equivalent_flops = 2.0 * static_cast<double>(m) * n * k;
    const double tflops = dense_equivalent_flops / (ms * 1.0e9);
    std::cout <<
#if defined(BEAM_BENCH_NVFP4)
        "sm120_sparse_nvfp4"
#else
        "sm120_sparse_mxfp8"
#endif
              << " m=" << m << " n=" << n << " k=" << k
              << " iterations=" << iterations
              << " kernel_ms=" << std::fixed << std::setprecision(6) << ms
              << " dense_equivalent_tflops=" << std::setprecision(3) << tflops << '\n';
    return tflops;
}

}  // namespace sm120_sparse_mxfp8_bench

int main(int argc, char** argv) {
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, 0);
    if (properties.major != 12 || properties.minor != 0) {
        std::cerr << "SM120 is required\n";
        return EXIT_FAILURE;
    }
    const int m = argc > 1 ? std::atoi(argv[1]) : 8192;
    const int n = argc > 2 ? std::atoi(argv[2]) : 8192;
    const int k = argc > 3 ? std::atoi(argv[3]) : 8192;
    const int iterations = argc > 4 ? std::atoi(argv[4]) : 100;
    try {
        sm120_sparse_mxfp8_bench::benchmark(m, n, k, iterations);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
