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
#if defined(BEAM_BENCH_NVFP4) || defined(BEAM_BENCH_DENSE_NVFP4)
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
#else
using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
#endif
using ElementC = cutlass::bfloat16_t;
#if defined(BEAM_BENCH_DENSE_NVFP4_OUTPUT)
using ElementD = cutlass::float_e2m1_t;
using ElementSF = cutlass::float_ue4m3_t;
#else
using ElementD = cutlass::bfloat16_t;
using ElementSF = cutlass::float_ue8m0_t;
#endif
using ElementBias = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
using ProblemShape = Shape<int, int, int, int>;
using ClusterShape = Shape<_1, _1, _1>;
#if defined(BEAM_BENCH_TILE_N256)
using MainloopTileShape = Shape<_128, _256, _128>;
using EpilogueTileShape = Shape<_128, _256, _128>;
#elif defined(BEAM_BENCH_TILE_K128)
using MainloopTileShape = Shape<_128, _128, _128>;
using EpilogueTileShape = Shape<_128, _128, _128>;
#else
using MainloopTileShape = Shape<_128, _128, _256>;
using EpilogueTileShape = Shape<_128, _128, _256>;
#endif
using ArchTag = cutlass::arch::Sm120;
#if defined(BEAM_BENCH_DENSE_NVFP4)
using MainloopOpClass = cutlass::arch::OpClassBlockScaledTensorOp;
using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
constexpr int kAlignmentA = 32;
constexpr int kAlignmentB = 32;
#elif defined(BEAM_BENCH_NVFP4)
using MainloopOpClass = cutlass::arch::OpClassBlockScaledSparseTensorOp;
using EpilogueSchedule = cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120;
using KernelSchedule = cutlass::gemm::KernelSparseTmaWarpSpecializedNvf4Sm120;
constexpr int kAlignmentA = 256;
constexpr int kAlignmentB = 128;
#else
using MainloopOpClass = cutlass::arch::OpClassBlockScaledSparseTensorOp;
using EpilogueSchedule = cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120;
using KernelSchedule = cutlass::gemm::KernelSparseTmaWarpSpecializedMxf8f6f4Acc2x4Sm120;
constexpr int kAlignmentA = 32;
constexpr int kAlignmentB = 16;
#endif
constexpr int kAlignmentC = 1;
constexpr int kAlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int kScaleVector = 64;
#if defined(BEAM_BENCH_FUSED_KERNEL_SCAFFOLD)
constexpr int kFusedHiddenValueBytes = 2 * (128 * 128 / 2);
constexpr int kFusedHiddenScaleBytes = 2 * (128 * (128 / 16));
constexpr int kFusedHiddenBytes = kFusedHiddenValueBytes + kFusedHiddenScaleBytes;
#else
constexpr int kFusedHiddenBytes = 0;
#endif

#if defined(BEAM_BENCH_DENSE_NVFP4_OUTPUT)
using FusionOperation = cutlass::epilogue::fusion::LinCombPerRowBiasEltActBlockScaleFactor<
    cutlass::epilogue::thread::ReLU,
    16,
    ElementD,
    ElementCompute,
    ElementSF,
    LayoutSFDTag,
    ElementBias,
    ElementC>;
#endif

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
    EpilogueSchedule
#if defined(BEAM_BENCH_DENSE_NVFP4_OUTPUT)
    , FusionOperation
#endif
    >::CollectiveOp;

using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage)) +
    kFusedHiddenBytes>;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    MainloopOpClass,
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

#if defined(BEAM_BENCH_STATIC_SCHEDULER)
using TileScheduler = void;
#else
using TileScheduler = cutlass::gemm::StreamKScheduler;
#endif
using BaseGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
#if defined(BEAM_BENCH_FUSED_KERNEL_SCAFFOLD)
struct FusedFfnKernelScaffold : BaseGemmKernel {
    using Base = BaseGemmKernel;
    using Params = typename Base::Params;

    struct SharedStorage {
        alignas(128) typename Base::SharedStorage base;
        alignas(128) std::uint8_t hidden_values[kFusedHiddenValueBytes];
        alignas(128) std::uint8_t hidden_scales[kFusedHiddenScaleBytes];
    };

    static constexpr int SharedStorageSize = sizeof(SharedStorage);
    static_assert(SharedStorageSize <= ArchTag::kSharedMemoryCapacityBytes,
                  "fused FFN scaffold exceeds SM120 shared-memory capacity");

    CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
        auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
        Base::operator()(params, reinterpret_cast<char*>(&storage.base));
    }
};
using GemmKernel = FusedFfnKernelScaffold;
#else
using GemmKernel = BaseGemmKernel;
#endif
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using Testbed = test::gemm::device::Testbed3x<Gemm>;

#if defined(BEAM_BENCH_DENSE_NVFP4_PIPELINE)
using Ff2CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp, EpilogueTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    cutlass::bfloat16_t, LayoutCTag, 1,
    cutlass::bfloat16_t, LayoutDTag,
    128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
    EpilogueSchedule>::CollectiveOp;
using Ff2StageCount = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename Ff2CollectiveEpilogue::SharedStorage))>;
using Ff2CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, MainloopOpClass,
    ElementA, LayoutATag, kAlignmentA,
    ElementB, LayoutBTag, kAlignmentB,
    ElementAccumulator, MainloopTileShape, ClusterShape,
    Ff2StageCount, KernelSchedule>::CollectiveOp;
using Ff2Kernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, Ff2CollectiveMainloop, Ff2CollectiveEpilogue, void>;
using Ff2Gemm = cutlass::gemm::device::GemmUniversalAdapter<Ff2Kernel>;
using Ff2Testbed = test::gemm::device::Testbed3x<Ff2Gemm>;

double benchmark_ffn_pipeline(int m, int iterations) {
    constexpr int d_model = 256;
    constexpr int ff_dim = 1024;
    ProblemShape ff1_problem{m, ff_dim, d_model, 1};
    ProblemShape ff2_problem{m, d_model, ff_dim, 1};
    Testbed ff1_testbed;
    Ff2Testbed ff2_testbed;
    auto& ff1 = ff1_testbed.impl_;
    auto& ff2 = ff2_testbed.impl_;
    if (!ff1.sufficient() || !ff2.sufficient() ||
        !ff1.initialize(ff1_problem, 1.0f, 0.0f) ||
        !ff2.initialize(ff2_problem, 1.0f, 0.0f)) {
        throw std::runtime_error("dense NVFP4 FFN pipeline fixture initialization failed");
    }

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    typename GemmKernel::TileScheduler::Arguments ff1_scheduler_args{};
    typename Gemm::Arguments ff1_arguments{
        cutlass::gemm::GemmUniversalMode::kGemm, ff1_problem,
        ff1.collective_mma_inputs.to_args(),
        ff1.collective_epilogue.to_args(ff1_problem),
        hw_info, ff1_scheduler_args};

    using Ff2ArrayElementA = typename Ff2Kernel::CollectiveMainloop::ArrayElementA;
    using Ff2ArrayElementB = typename Ff2Kernel::CollectiveMainloop::ArrayElementB;
    typename Ff2Kernel::MainloopArguments ff2_mainloop_args{
        reinterpret_cast<Ff2ArrayElementA*>(ff1.collective_epilogue.tensor_D.device_data()),
        ff2.collective_mma_inputs.stride_a,
        reinterpret_cast<Ff2ArrayElementB*>(ff2.collective_mma_inputs.tensor_B.device_data()),
        ff2.collective_mma_inputs.stride_b,
        ff1.collective_epilogue.tensor_SFD.device_data(),
        ff2.collective_mma_inputs.layout_sfa,
        ff2.collective_mma_inputs.tensor_SFB.device_data(),
        ff2.collective_mma_inputs.layout_sfb};
    typename Ff2Kernel::TileScheduler::Arguments ff2_scheduler_args{};
    typename Ff2Gemm::Arguments ff2_arguments{
        cutlass::gemm::GemmUniversalMode::kGemm, ff2_problem,
        ff2_mainloop_args, ff2.collective_epilogue.to_args(ff2_problem),
        hw_info, ff2_scheduler_args};

    Gemm ff1_gemm;
    Ff2Gemm ff2_gemm;
    if (ff1_gemm.can_implement(ff1_arguments) != cutlass::Status::kSuccess ||
        ff2_gemm.can_implement(ff2_arguments) != cutlass::Status::kSuccess) {
        throw std::runtime_error("dense NVFP4 FFN pipeline cannot implement Cube4 shape");
    }
    cutlass::device_memory::allocation<std::uint8_t> ff1_workspace(
        Gemm::get_workspace_size(ff1_arguments));
    cutlass::device_memory::allocation<std::uint8_t> ff2_workspace(
        Ff2Gemm::get_workspace_size(ff2_arguments));
    if (ff1_gemm.initialize(ff1_arguments, ff1_workspace.get()) != cutlass::Status::kSuccess ||
        ff2_gemm.initialize(ff2_arguments, ff2_workspace.get()) != cutlass::Status::kSuccess) {
        throw std::runtime_error("dense NVFP4 FFN pipeline initialization failed");
    }
    auto check_device = [](const char* stage) {
        const cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string(stage) + ": " + cudaGetErrorString(error));
        }
    };
    if (ff1_gemm.run() != cutlass::Status::kSuccess) {
        throw std::runtime_error("dense NVFP4 fused FF1 probe launch failed");
    }
    check_device("dense NVFP4 fused FF1 probe failed");
    if (ff2_gemm.run() != cutlass::Status::kSuccess) {
        throw std::runtime_error("dense NVFP4 chained FF2 probe launch failed");
    }
    check_device("dense NVFP4 chained FF2 probe failed");
    for (int i = 0; i < 10; ++i) {
        if (ff1_gemm.run() != cutlass::Status::kSuccess ||
            ff2_gemm.run() != cutlass::Status::kSuccess) {
            throw std::runtime_error("dense NVFP4 FFN pipeline warmup failed");
        }
    }
    cudaDeviceSynchronize();

    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        if (ff1_gemm.run() != cutlass::Status::kSuccess ||
            ff2_gemm.run() != cutlass::Status::kSuccess) {
            throw std::runtime_error("dense NVFP4 FFN pipeline launch failed");
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    const double ms = static_cast<double>(total_ms) / iterations;
    const double flops = 4.0 * static_cast<double>(m) * d_model * ff_dim;
    const double tflops = flops / (ms * 1.0e9);
    std::cout << "sm120_dense_nvfp4_ffn_fused_transport"
              << " m=" << m << " iterations=" << iterations
              << " pipeline_ms=" << std::fixed << std::setprecision(6) << ms
              << " dense_equivalent_tflops=" << std::setprecision(3) << tflops
              << " intermediate_format=nvfp4_relu intermediate_bf16_bytes=0\n";
    return tflops;
}
#endif

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
#if defined(BEAM_BENCH_DENSE_NVFP4)
#if defined(BEAM_BENCH_DENSE_NVFP4_OUTPUT)
        "sm120_dense_nvfp4_relu_nvfp4_out"
#else
        "sm120_dense_nvfp4"
#endif
#elif defined(BEAM_BENCH_NVFP4)
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
#if defined(BEAM_BENCH_DENSE_NVFP4_OUTPUT) && defined(BEAM_BENCH_TILE_K128)
        std::cout << "sm120_nvfp4_kernel_plan"
                  << " stages="
                  << sm120_sparse_mxfp8_bench::CollectiveMainloop::DispatchPolicy::Stages
                  << " shared_bytes="
                  << sm120_sparse_mxfp8_bench::GemmKernel::SharedStorageSize
                  << " reserved_hidden_bytes="
                  << sm120_sparse_mxfp8_bench::kFusedHiddenBytes << '\n';
#endif
#if defined(BEAM_BENCH_DENSE_NVFP4_PIPELINE)
        const int pipeline_iterations = argc > 2 ? std::atoi(argv[2]) : 100;
        sm120_sparse_mxfp8_bench::benchmark_ffn_pipeline(m, pipeline_iterations);
#else
        sm120_sparse_mxfp8_bench::benchmark(m, n, k, iterations);
#endif
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
