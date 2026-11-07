#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace {

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC = cutlass::half_t;
using ElementD = cutlass::float_e2m1_t;
using ElementSFD = cutlass::float_ue4m3_t;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using ElementCompute = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

constexpr int kAlignmentA = 32;
constexpr int kAlignmentB = 32;
constexpr int kAlignmentC = 8;
constexpr int kAlignmentD = 32;
constexpr int kScaleVector = 16;

using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
    kScaleVector, ElementD, ElementCompute, ElementSFD, LayoutDTag, ElementC>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, kAlignmentC,
    ElementD, LayoutDTag, kAlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOperation>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, kAlignmentA,
    ElementB, LayoutBTag, kAlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;
using LayoutSFA = typename CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename CollectiveMainloop::LayoutSFB;
using OutputScaleConfig = cutlass::detail::Sm1xxBlockScaledOutputConfig<kScaleVector>;

void require_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
        std::exit(2);
    }
}

void require_cutlass(cutlass::Status status, const char* what) {
    if (status != cutlass::Status::kSuccess) {
        std::cerr << what << ": " << cutlassGetStatusString(status) << '\n';
        std::exit(3);
    }
}

template <class T>
T* allocate_bytes(std::size_t bytes) {
    T* pointer = nullptr;
    require_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc");
    return pointer;
}

}  // namespace

int main(int argc, char** argv) {
    const int m = argc > 1 ? std::atoi(argv[1]) : 32768;
    const int n = argc > 2 ? std::atoi(argv[2]) : 1024;
    const int k = argc > 3 ? std::atoi(argv[3]) : 256;
    const int iterations = argc > 4 ? std::atoi(argv[4]) : 100;
    const auto problem_shape = make_shape(m, n, k, 1);
    const auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    const auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    const auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    const auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    using ScaleConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
    const LayoutSFA layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
    const LayoutSFB layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);
    const auto layout_sfd = OutputScaleConfig::tile_atom_to_shape_SFD(problem_shape);

    const std::size_t a_elements = static_cast<std::size_t>(m) * k;
    const std::size_t b_elements = static_cast<std::size_t>(n) * k;
    const std::size_t c_elements = static_cast<std::size_t>(m) * n;
    const std::size_t sfa_elements = static_cast<std::size_t>(size(filter_zeros(layout_sfa)));
    const std::size_t sfb_elements = static_cast<std::size_t>(size(filter_zeros(layout_sfb)));
    const std::size_t sfd_elements = static_cast<std::size_t>(size(filter_zeros(layout_sfd)));

    auto* a = allocate_bytes<cutlass::float_e2m1_t>((a_elements + 1U) / 2U);
    auto* b = allocate_bytes<cutlass::float_e2m1_t>((b_elements + 1U) / 2U);
    auto* c = allocate_bytes<ElementC>(c_elements * sizeof(ElementC));
    auto* d = allocate_bytes<ElementD>((c_elements + 1U) / 2U);
    auto* sfa = allocate_bytes<cutlass::float_ue4m3_t>(sfa_elements);
    auto* sfb = allocate_bytes<cutlass::float_ue4m3_t>(sfb_elements);
    auto* sfd = allocate_bytes<ElementSFD>(sfd_elements);
    auto* norm = allocate_bytes<float>(sizeof(float));

    require_cuda(cudaMemset(a, 0x11, (a_elements + 1U) / 2U), "memset A");
    require_cuda(cudaMemset(b, 0x11, (b_elements + 1U) / 2U), "memset B");
    require_cuda(cudaMemset(c, 0, c_elements * sizeof(ElementC)), "memset C");
    require_cuda(cudaMemset(d, 0, (c_elements + 1U) / 2U), "memset D");
    const std::uint8_t one_scale = cutlass::float_ue4m3_t(1.0F).raw();
    require_cuda(cudaMemset(sfa, one_scale, sfa_elements), "memset SFA");
    require_cuda(cudaMemset(sfb, one_scale, sfb_elements), "memset SFB");
    const float norm_value = 6.0F;
    require_cuda(cudaMemcpy(norm, &norm_value, sizeof(float), cudaMemcpyHostToDevice),
                 "copy norm");

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_shape,
        {a, stride_a, b, stride_b, sfa, layout_sfa, sfb, layout_sfb},
        {{1.0F, 0.0F}, c, stride_c, d, stride_d}};
    arguments.epilogue.thread.block_scale_factor_ptr = sfd;
    arguments.epilogue.thread.norm_constant_ptr = norm;

    Gemm gemm;
    const std::size_t workspace_bytes = Gemm::get_workspace_size(arguments);
    auto* workspace = allocate_bytes<std::uint8_t>(workspace_bytes == 0U ? 1U : workspace_bytes);
    require_cutlass(gemm.can_implement(arguments), "can_implement");
    require_cutlass(gemm.initialize(arguments, workspace), "initialize");
    for (int warmup = 0; warmup < 5; ++warmup) {
        require_cutlass(gemm.run(), "warmup");
    }
    require_cuda(cudaDeviceSynchronize(), "warmup synchronize");

    cudaEvent_t begin{}, end{};
    require_cuda(cudaEventCreate(&begin), "event begin");
    require_cuda(cudaEventCreate(&end), "event end");
    require_cuda(cudaEventRecord(begin), "record begin");
    for (int iteration = 0; iteration < iterations; ++iteration) {
        require_cutlass(gemm.run(), "benchmark run");
    }
    require_cuda(cudaEventRecord(end), "record end");
    require_cuda(cudaEventSynchronize(end), "event synchronize");
    float total_ms = 0.0F;
    require_cuda(cudaEventElapsedTime(&total_ms, begin, end), "elapsed time");
    const double milliseconds = total_ms / iterations;
    const double tflops = 2.0 * static_cast<double>(m) * n * k /
        (milliseconds * 1.0e9);
    std::cout << "cutlass_sm120_nvfp4_tma"
              << " m=" << m << " n=" << n << " k=" << k
              << " iterations=" << iterations
              << " ms=" << milliseconds
              << " tflops=" << tflops
              << " workspace_bytes=" << workspace_bytes << '\n';
    return 0;
}
