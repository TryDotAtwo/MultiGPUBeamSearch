#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

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
using WideTileShape = Shape<_128, _256, _128>;
using ClusterShape = Shape<_1, _1, _1>;
using ProblemShape = Shape<int, int, int, int>;
using ArchTag = cutlass::arch::Sm120;
using KernelSchedule =
    cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120;
using EpilogueSchedule =
    cutlass::epilogue::collective::EpilogueScheduleAuto;

using WideEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp,
    WideTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementOutput, LayoutD, 8,
    ElementOutput, LayoutD, 8,
    EpilogueSchedule>::CollectiveOp;

using WideStages = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename WideEpilogue::SharedStorage))>;

using WideMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementAB, LayoutA, 32,
    ElementAB, LayoutB, 32,
    ElementAccumulator, WideTileShape, ClusterShape,
    WideStages, KernelSchedule>::CollectiveOp;

using WideKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, WideMainloop, WideEpilogue, void>;

void require_cuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main() {
    constexpr int m = 128;
    constexpr int n = 256;
    constexpr int k = 1024;
    constexpr int mma_threads =
        cute::thr_size(typename WideMainloop::TiledMma{});
    constexpr int stages = WideMainloop::DispatchPolicy::Stages;
    constexpr int max_threads = WideKernel::MaxThreadsPerBlock;
    constexpr std::size_t shared_bytes =
        sizeof(typename WideKernel::SharedStorage);

    const auto problem_shape = make_shape(m, n, k, 1);
    const typename WideMainloop::StrideA stride_a =
        make_stride(std::int64_t{k}, _1{}, std::int64_t{m * k});
    const typename WideMainloop::StrideB stride_b =
        make_stride(std::int64_t{k}, _1{}, std::int64_t{n * k});
    using ScaleConfig = typename WideMainloop::Sm1xxBlkScaledConfig;
    const typename WideMainloop::LayoutSFA layout_sfa =
        ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
    const typename WideMainloop::LayoutSFB layout_sfb =
        ScaleConfig::tile_atom_to_shape_SFB(problem_shape);

    typename WideMainloop::ElementA* a = nullptr;
    typename WideMainloop::ElementB* b = nullptr;
    ElementScale* sfa = nullptr;
    ElementScale* sfb = nullptr;
    require_cuda(cudaMalloc(&a, static_cast<std::size_t>(m) * k / 2U),
                 "cudaMalloc A failed");
    require_cuda(cudaMalloc(&b, static_cast<std::size_t>(n) * k / 2U),
                 "cudaMalloc B failed");
    require_cuda(cudaMalloc(
                     &sfa, static_cast<std::size_t>(
                         size(filter_zeros(layout_sfa)))),
                 "cudaMalloc SFA failed");
    require_cuda(cudaMalloc(
                     &sfb, static_cast<std::size_t>(
                         size(filter_zeros(layout_sfb)))),
                 "cudaMalloc SFB failed");

    typename WideMainloop::Arguments arguments{
        a, stride_a, b, stride_b, sfa, layout_sfa, sfb, layout_sfb};
    const bool can_implement =
        WideMainloop::can_implement(problem_shape, arguments);

    std::cout << "stream1_transformer_sm120_nvfp4_wide_tile_probe"
              << " tile_m=128 tile_n=256 tile_k=128"
              << " mma_threads=" << mma_threads
              << " mma_warp_groups=" << mma_threads / 128
              << " max_threads_per_block=" << max_threads
              << " stages=" << stages
              << " shared_bytes=" << shared_bytes
              << " can_implement=" << (can_implement ? 1 : 0)
              << '\n';

    cudaFree(a);
    cudaFree(b);
    cudaFree(sfa);
    cudaFree(sfb);

    if (!can_implement) {
        std::cerr << "wide tile is not supported for the Cube4 FF2 shape\n";
        return EXIT_FAILURE;
    }
    if (mma_threads != 256) {
        std::cout << "two_mma_warp_group_hypothesis=rejected\n";
    }
    return EXIT_SUCCESS;
}
