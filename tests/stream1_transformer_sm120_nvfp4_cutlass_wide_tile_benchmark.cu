#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using namespace cute;

using ElementAB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementAccumulator = float;
using ElementCompute = float;
using ElementScale = cutlass::float_ue4m3_t;
using ElementOutput = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
using BaseTileShape = Shape<_128, _128, _128>;
using DeepTileShape = Shape<_128, _128, _256>;
using WideTileShape = Shape<_128, _256, _128>;
using ClusterShape = Shape<_1, _1, _1>;
using ProblemShape = Shape<int, int, int, int>;
using ArchTag = cutlass::arch::Sm120;
using KernelSchedule =
    cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120;
using EpilogueSchedule =
    cutlass::epilogue::collective::EpilogueScheduleAuto;

template <class Tile>
struct KernelTypes {
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, cutlass::arch::OpClassTensorOp,
        Tile, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementOutput, LayoutD, 8,
        ElementOutput, LayoutD, 8,
        EpilogueSchedule>::CollectiveOp;
    using Stages = cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epilogue::SharedStorage))>;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, cutlass::arch::OpClassBlockScaledTensorOp,
        ElementAB, LayoutA, 32,
        ElementAB, LayoutB, 32,
        ElementAccumulator, Tile, ClusterShape,
        Stages, KernelSchedule>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<
        ProblemShape, Mainloop, Epilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using Base = KernelTypes<BaseTileShape>;
using Deep = KernelTypes<DeepTileShape>;
using Wide = KernelTypes<WideTileShape>;

void require_cuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void require_cutlass(cutlass::Status status, const char* message) {
    if (status != cutlass::Status::kSuccess) {
        std::cerr << message << ": " << cutlassGetStatusString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

template <class Types>
typename Types::Gemm::Arguments make_arguments(
    const std::uint8_t* a,
    const std::uint8_t* b,
    const ElementScale* sfa,
    const ElementScale* sfb,
    const ElementOutput* c,
    ElementOutput* d,
    int rows) {
    constexpr int n = 256;
    constexpr int k = 1024;
    using Kernel = typename Types::Kernel;
    using Mainloop = typename Types::Mainloop;
    const ProblemShape problem{rows, n, k, 1};
    const auto stride_a = cutlass::make_cute_packed_stride(
        typename Kernel::StrideA{}, cute::make_shape(rows, k, 1));
    const auto stride_b = cutlass::make_cute_packed_stride(
        typename Kernel::StrideB{}, cute::make_shape(n, k, 1));
    const auto stride_c = cutlass::make_cute_packed_stride(
        typename Kernel::StrideC{}, cute::make_shape(rows, n, 1));
    const auto stride_d = cutlass::make_cute_packed_stride(
        typename Kernel::StrideD{}, cute::make_shape(rows, n, 1));
    using ScaleConfig = typename Mainloop::Sm1xxBlkScaledConfig;
    const auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(problem);
    const auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(problem);
    typename Kernel::MainloopArguments mainloop{
        reinterpret_cast<const typename Kernel::CollectiveMainloop::ArrayElementA*>(a),
        stride_a,
        reinterpret_cast<const typename Kernel::CollectiveMainloop::ArrayElementB*>(b),
        stride_b,
        sfa,
        layout_sfa,
        sfb,
        layout_sfb};
    typename Kernel::EpilogueArguments epilogue{
        {}, c, stride_c, d, stride_d};
    epilogue.thread.alpha = 1.0F;
    epilogue.thread.beta = 0.0F;
    cutlass::KernelHardwareInfo hardware{};
    hardware.device_id = 0;
    hardware.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    typename Kernel::TileScheduler::Arguments scheduler{};
    return {
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        mainloop,
        epilogue,
        hardware,
        scheduler};
}

template <class Types>
double benchmark(
    typename Types::Gemm& gemm,
    typename Types::Gemm::Arguments const& arguments,
    cudaStream_t stream,
    int iterations) {
    for (int iteration = 0; iteration < 10; ++iteration) {
        require_cutlass(gemm(arguments, nullptr, stream), "warmup launch failed");
    }
    require_cuda(cudaStreamSynchronize(stream), "warmup sync failed");
    cudaEvent_t begin{}, end{};
    require_cuda(cudaEventCreate(&begin), "create begin event failed");
    require_cuda(cudaEventCreate(&end), "create end event failed");
    require_cuda(cudaEventRecord(begin, stream), "record begin failed");
    for (int iteration = 0; iteration < iterations; ++iteration) {
        require_cutlass(gemm(arguments, nullptr, stream), "timed launch failed");
    }
    require_cuda(cudaEventRecord(end, stream), "record end failed");
    require_cuda(cudaEventSynchronize(end), "timed sync failed");
    float elapsed_ms = 0.0F;
    require_cuda(cudaEventElapsedTime(&elapsed_ms, begin, end),
                 "elapsed time failed");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return static_cast<double>(elapsed_ms) / iterations;
}

}  // namespace

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 131072;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 100;
    if (rows <= 0 || rows % 128 != 0 || iterations <= 0) {
        std::cerr << "rows must be a positive multiple of 128 and iterations positive\n";
        return EXIT_FAILURE;
    }
    constexpr int n = 256;
    constexpr int k = 1024;
    const ProblemShape problem{rows, n, k, 1};
    using BaseScaleConfig = typename Base::Mainloop::Sm1xxBlkScaledConfig;
    using DeepScaleConfig = typename Deep::Mainloop::Sm1xxBlkScaledConfig;
    using WideScaleConfig = typename Wide::Mainloop::Sm1xxBlkScaledConfig;
    const auto base_sfa_layout =
        BaseScaleConfig::tile_atom_to_shape_SFA(problem);
    const auto base_sfb_layout =
        BaseScaleConfig::tile_atom_to_shape_SFB(problem);
    const auto deep_sfa_layout =
        DeepScaleConfig::tile_atom_to_shape_SFA(problem);
    const auto deep_sfb_layout =
        DeepScaleConfig::tile_atom_to_shape_SFB(problem);
    const auto wide_sfa_layout =
        WideScaleConfig::tile_atom_to_shape_SFA(problem);
    const auto wide_sfb_layout =
        WideScaleConfig::tile_atom_to_shape_SFB(problem);
    const std::size_t sfa_bytes = std::max({
        static_cast<std::size_t>(size(filter_zeros(base_sfa_layout))),
        static_cast<std::size_t>(size(filter_zeros(deep_sfa_layout))),
        static_cast<std::size_t>(size(filter_zeros(wide_sfa_layout)))});
    const std::size_t sfb_bytes = std::max({
        static_cast<std::size_t>(size(filter_zeros(base_sfb_layout))),
        static_cast<std::size_t>(size(filter_zeros(deep_sfb_layout))),
        static_cast<std::size_t>(size(filter_zeros(wide_sfb_layout)))});

    std::uint8_t *a = nullptr, *b = nullptr;
    ElementScale *sfa = nullptr, *sfb = nullptr;
    ElementOutput *c = nullptr, *base_d = nullptr, *deep_d = nullptr,
                  *wide_d = nullptr;
    require_cuda(cudaMalloc(&a, static_cast<std::size_t>(rows) * k / 2U),
                 "cudaMalloc A failed");
    require_cuda(cudaMalloc(&b, static_cast<std::size_t>(n) * k / 2U),
                 "cudaMalloc B failed");
    require_cuda(cudaMalloc(&sfa, sfa_bytes), "cudaMalloc SFA failed");
    require_cuda(cudaMalloc(&sfb, sfb_bytes), "cudaMalloc SFB failed");
    const std::size_t output_bytes =
        static_cast<std::size_t>(rows) * n * sizeof(ElementOutput);
    require_cuda(cudaMalloc(&c, output_bytes), "cudaMalloc C failed");
    require_cuda(cudaMalloc(&base_d, output_bytes), "cudaMalloc base D failed");
    require_cuda(cudaMalloc(&deep_d, output_bytes), "cudaMalloc deep D failed");
    require_cuda(cudaMalloc(&wide_d, output_bytes), "cudaMalloc wide D failed");
    require_cuda(cudaMemset(a, 0x11, static_cast<std::size_t>(rows) * k / 2U),
                 "initialize A failed");
    require_cuda(cudaMemset(b, 0x11, static_cast<std::size_t>(n) * k / 2U),
                 "initialize B failed");
    require_cuda(cudaMemset(sfa, ElementScale(1.0F).raw(), sfa_bytes),
                 "initialize SFA failed");
    require_cuda(cudaMemset(sfb, ElementScale(1.0F).raw(), sfb_bytes),
                 "initialize SFB failed");
    require_cuda(cudaMemset(c, 0, output_bytes), "initialize C failed");
    require_cuda(cudaMemset(base_d, 0xff, output_bytes),
                 "initialize base D failed");
    require_cuda(cudaMemset(deep_d, 0xff, output_bytes),
                 "initialize deep D failed");
    require_cuda(cudaMemset(wide_d, 0xff, output_bytes),
                 "initialize wide D failed");

    auto base_arguments = make_arguments<Base>(
        a, b, sfa, sfb, c, base_d, rows);
    auto deep_arguments = make_arguments<Deep>(
        a, b, sfa, sfb, c, deep_d, rows);
    auto wide_arguments = make_arguments<Wide>(
        a, b, sfa, sfb, c, wide_d, rows);
    typename Base::Gemm base_gemm;
    typename Deep::Gemm deep_gemm;
    typename Wide::Gemm wide_gemm;
    require_cutlass(base_gemm.can_implement(base_arguments),
                    "base tile cannot implement shape");
    require_cutlass(wide_gemm.can_implement(wide_arguments),
                    "wide tile cannot implement shape");
    require_cutlass(deep_gemm.can_implement(deep_arguments),
                    "deep tile cannot implement shape");
    if (Base::Gemm::get_workspace_size(base_arguments) != 0U ||
        Deep::Gemm::get_workspace_size(deep_arguments) != 0U ||
        Wide::Gemm::get_workspace_size(wide_arguments) != 0U) {
        std::cerr << "unexpected workspace requirement\n";
        return EXIT_FAILURE;
    }

    cudaStream_t stream{};
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                 "create stream failed");
    require_cutlass(base_gemm(base_arguments, nullptr, stream),
                    "base numeric launch failed");
    require_cutlass(deep_gemm(deep_arguments, nullptr, stream),
                    "deep numeric launch failed");
    require_cutlass(wide_gemm(wide_arguments, nullptr, stream),
                    "wide numeric launch failed");
    require_cuda(cudaStreamSynchronize(stream), "numeric sync failed");
    std::vector<half> host_base(static_cast<std::size_t>(rows) * n);
    std::vector<half> host_deep(host_base.size());
    std::vector<half> host_wide(host_base.size());
    require_cuda(cudaMemcpy(host_base.data(), base_d, output_bytes,
                            cudaMemcpyDeviceToHost),
                 "copy base output failed");
    require_cuda(cudaMemcpy(host_wide.data(), wide_d, output_bytes,
                            cudaMemcpyDeviceToHost),
                 "copy wide output failed");
    require_cuda(cudaMemcpy(host_deep.data(), deep_d, output_bytes,
                            cudaMemcpyDeviceToHost),
                 "copy deep output failed");
    float max_abs_error = 0.0F;
    std::size_t wide_mismatches = 0U;
    std::size_t deep_mismatches = 0U;
    for (std::size_t index = 0; index < host_base.size(); ++index) {
        const float wide_error = std::abs(
            __half2float(host_base[index]) - __half2float(host_wide[index]));
        const float deep_error = std::abs(
            __half2float(host_base[index]) - __half2float(host_deep[index]));
        max_abs_error = std::max({max_abs_error, wide_error, deep_error});
        wide_mismatches += wide_error != 0.0F;
        deep_mismatches += deep_error != 0.0F;
    }

    const double base_ms = benchmark<Base>(
        base_gemm, base_arguments, stream, iterations);
    const double deep_ms = benchmark<Deep>(
        deep_gemm, deep_arguments, stream, iterations);
    const double wide_ms = benchmark<Wide>(
        wide_gemm, wide_arguments, stream, iterations);
    const double flops = 2.0 * rows * n * k;
    const double base_tflops = flops / (base_ms * 1.0e9);
    const double deep_tflops = flops / (deep_ms * 1.0e9);
    const double wide_tflops = flops / (wide_ms * 1.0e9);
    std::cout << std::fixed << std::setprecision(6)
              << "stream1_transformer_sm120_nvfp4_wide_tile_ab"
              << " rows=" << rows
              << " iterations=" << iterations
              << " base_tile=128x128x128"
              << " base_mma_threads="
              << cute::thr_size(typename Base::Mainloop::TiledMma{})
              << " base_ms=" << base_ms
              << " base_tflops=" << std::setprecision(3) << base_tflops
              << " deep_tile=128x128x256"
              << " deep_mma_threads="
              << cute::thr_size(typename Deep::Mainloop::TiledMma{})
              << " deep_ms=" << std::setprecision(6) << deep_ms
              << " deep_tflops=" << std::setprecision(3) << deep_tflops
              << " deep_speedup=" << base_ms / deep_ms
              << " wide_tile=128x256x128"
              << " wide_mma_threads="
              << cute::thr_size(typename Wide::Mainloop::TiledMma{})
              << " wide_ms=" << std::setprecision(6) << wide_ms
              << " wide_tflops=" << std::setprecision(3) << wide_tflops
              << " wide_speedup=" << base_ms / wide_ms
              << " max_abs_error=" << max_abs_error
              << " deep_mismatches=" << deep_mismatches
              << " wide_mismatches=" << wide_mismatches
              << '\n';

    cudaStreamDestroy(stream);
    cudaFree(a);
    cudaFree(b);
    cudaFree(sfa);
    cudaFree(sfb);
    cudaFree(c);
    cudaFree(base_d);
    cudaFree(deep_d);
    cudaFree(wide_d);
    return deep_mismatches == 0U && wide_mismatches == 0U
        ? EXIT_SUCCESS : EXIT_FAILURE;
}
