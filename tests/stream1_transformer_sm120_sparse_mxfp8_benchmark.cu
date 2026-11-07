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
#if defined(BEAM_BENCH_DENSE_NVFP4_PIPELINE) && \
    !defined(STREAM1_FUSED_TEST_HIDDEN)
#define STREAM1_FUSED_TEST_HIDDEN 1024
#endif
#include "sm120_fused_numeric_reference.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace sm120_sparse_mxfp8_bench {

using namespace cute;

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
using LayoutSFDTag = cutlass::layout::RowMajor;
#if defined(BEAM_BENCH_MXFP4)
using ElementA = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
#elif defined(BEAM_BENCH_NVFP4) || defined(BEAM_BENCH_DENSE_NVFP4)
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
#else
using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
#endif
using ElementC = cutlass::bfloat16_t;
#if defined(BEAM_BENCH_FF1_VOID_C)
// The concurrent FFN fixture fixes beta=0; omit unused source storage.
using Ff1SourceElement = void;
#else
using Ff1SourceElement = ElementC;
#endif
#if defined(BEAM_BENCH_DENSE_NVFP4_OUTPUT)
using ElementD = cutlass::float_e2m1_t;
#if defined(BEAM_BENCH_MXFP4)
using ElementSF = cutlass::float_ue8m0_t;
#else
using ElementSF = cutlass::float_ue4m3_t;
#endif
#else
using ElementD = cutlass::bfloat16_t;
using ElementSF = cutlass::float_ue8m0_t;
#endif
using ElementBias = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
using ProblemShape = Shape<int, int, int, int>;
using ClusterShape = Shape<_1, _1, _1>;
#if defined(BEAM_BENCH_TILE_M256)
using MainloopTileShape = Shape<_256, _128, _128>;
using EpilogueTileShape = Shape<_256, _128, _128>;
#elif defined(BEAM_BENCH_TILE_N256)
using MainloopTileShape = Shape<_128, _256, _128>;
using EpilogueTileShape = Shape<_128, _256, _128>;
#elif defined(BEAM_BENCH_TILE_N64)
using MainloopTileShape = Shape<_128, _64, _128>;
using EpilogueTileShape = Shape<_128, _64, _128>;
#elif defined(BEAM_BENCH_TILE_K128)
using MainloopTileShape = Shape<_128, _128, _128>;
using EpilogueTileShape = Shape<_128, _128, _128>;
#else
using MainloopTileShape = Shape<_128, _128, _256>;
using EpilogueTileShape = Shape<_128, _128, _256>;
#endif
// Isolate FF1 accumulator pressure without changing the FF2 control tile.
#if defined(BEAM_BENCH_FF1_TILE_N64)
using Ff1TileShape = Shape<_128, _64, _256>;
#else
using Ff1TileShape = MainloopTileShape;
#endif
#if defined(BEAM_BENCH_FF1_EPILOGUE_128X128)
// Opt-in full-tile epilogue experiment; retain the validated 64x128 route.
using Ff1EpilogueSubtile = Shape<_128, _128>;
#elif defined(BEAM_BENCH_FF1_EPILOGUE_64X128)
// Diagnostic candidate: halve the epilogue subtile count versus 64x64.
// Keep the MMA tile, scale-vector length and FF2 implementation unchanged.
using Ff1EpilogueSubtile = Shape<_64, _128>;
#elif defined(BEAM_BENCH_FF1_EPILOGUE_64X64)
using Ff1EpilogueSubtile = Shape<_64, _64>;
#else
using Ff1EpilogueSubtile = cutlass::epilogue::collective::EpilogueTileAuto;
#endif
using ArchTag = cutlass::arch::Sm120;
#if defined(BEAM_BENCH_DENSE_NVFP4)
using MainloopOpClass = cutlass::arch::OpClassBlockScaledTensorOp;
using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
#if defined(BEAM_BENCH_PINGPONG)
#if defined(BEAM_BENCH_MXFP4)
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
#else
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120;
#endif
#else
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
#endif
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
#if defined(BEAM_BENCH_MXFP4)
constexpr int kScaleVector = 32;
#else
constexpr int kScaleVector = 16;
#endif
#if defined(BEAM_BENCH_FUSED_KERNEL_SCAFFOLD)
constexpr int kFusedHiddenValueBytes = 2 * (128 * 128 / 2);
constexpr int kFusedHiddenScaleBytes = 2 * (128 * (128 / 16));
constexpr int kFusedHiddenBytes = kFusedHiddenValueBytes + kFusedHiddenScaleBytes;
#else
constexpr int kFusedHiddenBytes = 0;
#endif

#if defined(BEAM_BENCH_DENSE_NVFP4_OUTPUT)
using FusionOperation = cutlass::epilogue::fusion::LinCombPerColBiasEltActBlockScaleFactor<
    cutlass::epilogue::thread::ReLU,
    kScaleVector,
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
    Ff1TileShape,
    ClusterShape,
    Ff1EpilogueSubtile,
    ElementAccumulator,
    ElementCompute,
    Ff1SourceElement,
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
    Ff1TileShape,
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
using Testbed = test::gemm::device::Testbed3x<
    Gemm, cutlass::epilogue::thread::ReLU>;

#if defined(BEAM_BENCH_DENSE_NVFP4_PIPELINE)
#if defined(BEAM_BENCH_FF2_VOID_C)
// Both FFN fixtures fix FF2 beta=0; this is not a residual-fusion path.
using Ff2SourceElement = void;
#else
using Ff2SourceElement = cutlass::bfloat16_t;
#endif
using Ff2CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp, EpilogueTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    Ff2SourceElement, LayoutCTag, 1,
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

template <class SourceLayout, class DestinationLayout>
__global__ void relayout_output_scales_to_input_scales(
    ElementSF* destination,
    ElementSF const* source,
    SourceLayout source_layout,
    DestinationLayout destination_layout,
    int rows,
    int columns) {
    const int scale_groups_per_row = columns / kScaleVector;
    const int linear = static_cast<int>(
        blockIdx.x * blockDim.x + threadIdx.x);
    const int total = rows * scale_groups_per_row;
    if (linear >= total) {
        return;
    }
    const int row = linear / scale_groups_per_row;
    const int column =
        (linear - row * scale_groups_per_row) * kScaleVector;
    destination[destination_layout(row, column, 0)] =
        source[source_layout(row, column, 0)];
}

double benchmark_ffn_pipeline(int m, int iterations) {
    constexpr int d_model = 256;
    constexpr int ff_dim = 1024;
    static_assert(
        sm120_fused_reference::kModel == d_model &&
        sm120_fused_reference::kHidden == ff_dim,
        "the independent FFN oracle must use the benchmark dimensions");
    ProblemShape ff1_problem{m, ff_dim, d_model, 1};
    ProblemShape ff2_problem{m, d_model, ff_dim, 1};
    using Ff1OutputScaleConfig =
        cutlass::detail::Sm1xxBlockScaledOutputConfig<kScaleVector>;
    using Ff2InputScaleConfig =
        typename Ff2CollectiveMainloop::Sm1xxBlkScaledConfig;
    const auto ff1_sfd_layout =
        Ff1OutputScaleConfig::tile_atom_to_shape_SFD(ff1_problem);
    const auto ff2_sfa_layout =
        Ff2InputScaleConfig::tile_atom_to_shape_SFA(ff2_problem);
    static_assert(
        std::is_same_v<
            std::remove_cv_t<decltype(ff1_sfd_layout)>,
            std::remove_cv_t<decltype(ff2_sfa_layout)>>,
        "direct FF1 SFD to FF2 SFA alias requires identical CUTLASS layouts");
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
        ff2.collective_mma_inputs.tensor_SFA.device_data(),
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
    auto launch_ff1 = [&](cudaStream_t stream = nullptr) {
#if defined(BEAM_BENCH_PDL)
        return ff1_gemm.run(stream, nullptr, true);
#else
        return ff1_gemm.run(stream);
#endif
    };
    auto launch_scale_relayout = [&](cudaStream_t stream = nullptr) {
        const int groups = m * ff_dim / kScaleVector;
        constexpr int threads = 256;
        relayout_output_scales_to_input_scales<<<
            (groups + threads - 1) / threads, threads, 0, stream>>>(
                ff2.collective_mma_inputs.tensor_SFA.device_data(),
                ff1.collective_epilogue.tensor_SFD.device_data(),
                ff1_sfd_layout, ff2_sfa_layout, m, ff_dim);
        return cudaPeekAtLastError();
    };
    auto launch_ff2 = [&](cudaStream_t stream = nullptr) {
#if defined(BEAM_BENCH_PDL)
        return ff2_gemm.run(stream, nullptr, true);
#else
        return ff2_gemm.run(stream);
#endif
    };
    auto check_device = [](const char* stage) {
        const cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string(stage) + ": " + cudaGetErrorString(error));
        }
    };
    if (launch_ff1() != cutlass::Status::kSuccess) {
        throw std::runtime_error("dense NVFP4 fused FF1 probe launch failed");
    }
    if (launch_scale_relayout() != cudaSuccess) {
        throw std::runtime_error("dense NVFP4 SFD-to-SFA relayout launch failed");
    }
    check_device("dense NVFP4 fused FF1 probe failed");
    if (launch_ff2() != cutlass::Status::kSuccess) {
        throw std::runtime_error("dense NVFP4 chained FF2 probe launch failed");
    }
    check_device("dense NVFP4 chained FF2 probe failed");
    for (int i = 0; i < 10; ++i) {
        if (launch_ff1() != cutlass::Status::kSuccess ||
            launch_scale_relayout() != cudaSuccess ||
            launch_ff2() != cutlass::Status::kSuccess) {
            throw std::runtime_error("dense NVFP4 FFN pipeline warmup failed");
        }
    }
    cudaDeviceSynchronize();

    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        if (launch_ff1() != cutlass::Status::kSuccess ||
            launch_scale_relayout() != cudaSuccess ||
            launch_ff2() != cutlass::Status::kSuccess) {
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
              << " intermediate_format="
#if defined(BEAM_BENCH_MXFP4)
              << "mxfp4_relu"
#else
              << "nvfp4_relu"
#endif
              << " intermediate_bf16_bytes=0 launch_mode="
#if defined(BEAM_BENCH_PDL)
              << "pdl"
#else
              << "sequential"
#endif
              << " scale_relayout=sfd_to_sfa\n";
    return tflops;
}

#if defined(BEAM_BENCH_PIPELINE_CONCURRENT)
double benchmark_ffn_pipeline_concurrent(
    int m, int iterations, int slots, bool persist_hidden_l2,
    bool verify_numeric_chain, std::uint32_t numeric_seed,
    bool use_scale_relayout, int diagnostic_mode = 0) {
    // 0: unchanged throughput control; 1/2: isolated FF1/FF2;
    // 3: event-delimited chain. Mode 3 perturbs PDL overlap and is diagnostic only.
    // 4: same complete chain, captured once per slot and replayed per iteration.
    if (diagnostic_mode < 0 || diagnostic_mode > 4 || iterations <= 0) {
        throw std::runtime_error("diagnostic mode must be [0,4] and iterations positive");
    }
    if (slots < 1 || slots > 8) {
        throw std::runtime_error("concurrent FFN slots must be in [1, 8]");
    }
    constexpr int d_model = 256;
    constexpr int ff_dim = 1024;
    static_assert(
        sm120_fused_reference::kModel == d_model &&
        sm120_fused_reference::kHidden == ff_dim,
        "the independent FFN oracle must use the benchmark dimensions");
    ProblemShape ff1_problem{m, ff_dim, d_model, 1};
    ProblemShape ff2_problem{m, d_model, ff_dim, 1};
    using Ff1OutputScaleConfig =
        cutlass::detail::Sm1xxBlockScaledOutputConfig<kScaleVector>;
    using Ff2InputScaleConfig =
        typename Ff2CollectiveMainloop::Sm1xxBlkScaledConfig;
    const auto ff1_sfd_layout =
        Ff1OutputScaleConfig::tile_atom_to_shape_SFD(ff1_problem);
    const auto ff2_sfa_layout =
        Ff2InputScaleConfig::tile_atom_to_shape_SFA(ff2_problem);
    static_assert(
        std::is_same_v<
            std::remove_cv_t<decltype(ff1_sfd_layout)>,
            std::remove_cv_t<decltype(ff2_sfa_layout)>>,
        "direct FF1 SFD to FF2 SFA alias requires identical CUTLASS layouts");
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

    std::vector<std::unique_ptr<Testbed>> ff1_testbeds;
    std::vector<std::unique_ptr<Ff2Testbed>> ff2_testbeds;
    std::vector<std::unique_ptr<Gemm>> ff1_gemms;
    std::vector<std::unique_ptr<Ff2Gemm>> ff2_gemms;
    std::vector<std::unique_ptr<cutlass::device_memory::allocation<std::uint8_t>>> ff1_workspaces;
    std::vector<std::unique_ptr<cutlass::device_memory::allocation<std::uint8_t>>> ff2_workspaces;
    std::vector<cudaStream_t> streams(static_cast<std::size_t>(slots));

    const std::size_t hidden_value_bytes =
        static_cast<std::size_t>(m) * ff_dim / 2U;
    int max_persisting_l2_bytes = 0;
    cudaDeviceGetAttribute(
        &max_persisting_l2_bytes, cudaDevAttrMaxPersistingL2CacheSize, 0);
    if (persist_hidden_l2) {
        const std::size_t requested = hidden_value_bytes *
            static_cast<std::size_t>(slots);
        cudaDeviceSetLimit(
            cudaLimitPersistingL2CacheSize,
            std::min(requested,
                     static_cast<std::size_t>(max_persisting_l2_bytes)));
    }

    for (int slot = 0; slot < slots; ++slot) {
        ff1_testbeds.push_back(std::make_unique<Testbed>());
        ff2_testbeds.push_back(std::make_unique<Ff2Testbed>());
        auto& ff1 = ff1_testbeds.back()->impl_;
        auto& ff2 = ff2_testbeds.back()->impl_;
        if (!ff1.sufficient() || !ff2.sufficient() ||
            !ff1.initialize(ff1_problem, 1.0f, 0.0f) ||
            !ff2.initialize(ff2_problem, 1.0f, 0.0f)) {
            throw std::runtime_error("concurrent FFN fixture initialization failed");
        }
        typename GemmKernel::TileScheduler::Arguments ff1_scheduler_args{};
        typename Gemm::Arguments ff1_arguments{
            cutlass::gemm::GemmUniversalMode::kGemm, ff1_problem,
            ff1.collective_mma_inputs.to_args(),
            ff1.collective_epilogue.to_args(ff1_problem),
            hw_info, ff1_scheduler_args};
        using Ff2ArrayElementA = typename Ff2Kernel::CollectiveMainloop::ArrayElementA;
        using Ff2ArrayElementB = typename Ff2Kernel::CollectiveMainloop::ArrayElementB;
        ElementSF* ff2_input_scale_ptr = use_scale_relayout
            ? ff2.collective_mma_inputs.tensor_SFA.device_data()
            : ff1.collective_epilogue.tensor_SFD.device_data();
        typename Ff2Kernel::MainloopArguments ff2_mainloop_args{
            reinterpret_cast<Ff2ArrayElementA*>(ff1.collective_epilogue.tensor_D.device_data()),
            ff2.collective_mma_inputs.stride_a,
            reinterpret_cast<Ff2ArrayElementB*>(ff2.collective_mma_inputs.tensor_B.device_data()),
            ff2.collective_mma_inputs.stride_b,
            ff2_input_scale_ptr,
            ff2.collective_mma_inputs.layout_sfa,
            ff2.collective_mma_inputs.tensor_SFB.device_data(),
            ff2.collective_mma_inputs.layout_sfb};
        typename Ff2Kernel::TileScheduler::Arguments ff2_scheduler_args{};
        typename Ff2Gemm::Arguments ff2_arguments{
            cutlass::gemm::GemmUniversalMode::kGemm, ff2_problem,
            ff2_mainloop_args, ff2.collective_epilogue.to_args(ff2_problem),
            hw_info, ff2_scheduler_args};
        auto ff1_workspace = std::make_unique<cutlass::device_memory::allocation<std::uint8_t>>(
            Gemm::get_workspace_size(ff1_arguments));
        auto ff2_workspace = std::make_unique<cutlass::device_memory::allocation<std::uint8_t>>(
            Ff2Gemm::get_workspace_size(ff2_arguments));
        auto ff1_gemm = std::make_unique<Gemm>();
        auto ff2_gemm = std::make_unique<Ff2Gemm>();
        if (ff1_gemm->can_implement(ff1_arguments) != cutlass::Status::kSuccess ||
            ff2_gemm->can_implement(ff2_arguments) != cutlass::Status::kSuccess ||
            ff1_gemm->initialize(ff1_arguments, ff1_workspace->get()) != cutlass::Status::kSuccess ||
            ff2_gemm->initialize(ff2_arguments, ff2_workspace->get()) != cutlass::Status::kSuccess) {
            throw std::runtime_error("concurrent FFN GEMM initialization failed");
        }
        cudaStreamCreateWithFlags(&streams[static_cast<std::size_t>(slot)], cudaStreamNonBlocking);
        if (persist_hidden_l2) {
            cudaStreamAttrValue attribute{};
            attribute.accessPolicyWindow.base_ptr =
                ff1.collective_epilogue.tensor_D.device_data();
            attribute.accessPolicyWindow.num_bytes = std::min(
                hidden_value_bytes,
                static_cast<std::size_t>(max_persisting_l2_bytes));
            attribute.accessPolicyWindow.hitRatio = 1.0F;
            attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(
                streams[static_cast<std::size_t>(slot)],
                cudaStreamAttributeAccessPolicyWindow,
                &attribute);
        }
        ff1_workspaces.push_back(std::move(ff1_workspace));
        ff2_workspaces.push_back(std::move(ff2_workspace));
        ff1_gemms.push_back(std::move(ff1_gemm));
        ff2_gemms.push_back(std::move(ff2_gemm));
    }

    auto launch_ff1_slot = [&](int slot) {
        const auto index = static_cast<std::size_t>(slot);
#if defined(BEAM_BENCH_PDL)
        return ff1_gemms[index]->run(streams[index], nullptr, true);
#else
        return ff1_gemms[index]->run(streams[index]);
#endif
    };
    auto launch_scale_relayout_slot = [&](int slot) {
        if (!use_scale_relayout) {
            return cudaSuccess;
        }
        const auto index = static_cast<std::size_t>(slot);
        auto& ff1 = ff1_testbeds[index]->impl_;
        auto& ff2 = ff2_testbeds[index]->impl_;
        const int groups = m * ff_dim / kScaleVector;
        constexpr int threads = 256;
        relayout_output_scales_to_input_scales<<<
            (groups + threads - 1) / threads, threads, 0, streams[index]>>>(
                ff2.collective_mma_inputs.tensor_SFA.device_data(),
                ff1.collective_epilogue.tensor_SFD.device_data(),
                ff1_sfd_layout, ff2_sfa_layout, m, ff_dim);
        return cudaPeekAtLastError();
    };
    auto launch_ff2_slot = [&](int slot) {
        const auto index = static_cast<std::size_t>(slot);
#if defined(BEAM_BENCH_PDL)
        return ff2_gemms[index]->run(streams[index], nullptr, true);
#else
        return ff2_gemms[index]->run(streams[index]);
#endif
    };

    for (int warmup = 0; warmup < 10; ++warmup) {
        for (int slot = 0; slot < slots; ++slot) {
            if (launch_ff1_slot(slot) != cutlass::Status::kSuccess ||
                launch_scale_relayout_slot(slot) != cudaSuccess ||
                launch_ff2_slot(slot) != cutlass::Status::kSuccess) {
                throw std::runtime_error("concurrent FFN warmup failed");
            }
        }
    }
    cudaDeviceSynchronize();

    std::vector<std::vector<float>> timed_output_reference(slots);
    int oracle_begin = 0, oracle_end = m;
    if (verify_numeric_chain) {
        auto read_bound = [](char const* name, int fallback) {
            char const* text = std::getenv(name);
            if (!text) return fallback;
            char* end = nullptr;
            const long value = std::strtol(text, &end, 10);
            if (end == text || *end || value < 0 || value > 51072)
                throw std::runtime_error("invalid oracle row bound");
            return static_cast<int>(value);
        };
        oracle_begin = read_bound("BEAM_ORACLE_ROW_BEGIN", 0);
        oracle_end = read_bound("BEAM_ORACLE_ROW_END", m);
        if (oracle_begin >= oracle_end || oracle_end > m)
            throw std::runtime_error("invalid oracle row range");
    }
    if (verify_numeric_chain) {
        if (m > 51072) {
            throw std::runtime_error(
                "full stock FFN host-reference gate is limited to m <= 51072");
        }
        for (int oracle_slot = 0; oracle_slot < slots; ++oracle_slot) {
        auto& ff1 = ff1_testbeds[oracle_slot]->impl_;
        auto& ff2 = ff2_testbeds[oracle_slot]->impl_;
        using Ff1Main = std::remove_reference_t<
            decltype(ff1.collective_mma_inputs)>;
        using Ff2Main = std::remove_reference_t<
            decltype(ff2.collective_mma_inputs)>;
        using Ff1ElementA = typename Ff1Main::ElementA;
        using Ff1ElementB = typename Ff1Main::ElementB;
        using Ff1ElementSF = typename Ff1Main::ElementSF;
        using Ff2ElementB = typename Ff2Main::ElementB;
        using Ff2ElementSF = typename Ff2Main::ElementSF;
        using Ff1Epilogue = std::remove_reference_t<
            decltype(ff1.collective_epilogue)>;
        using Ff1ElementBias = typename Ff1Epilogue::ElementBias;

        const auto slot_seed = numeric_seed + static_cast<std::uint32_t>(oracle_slot);
        auto fixture = sm120_fused_reference::make_fixture(slot_seed, m);
        auto ff1_sfa = sm120_fused_reference::encode_scales(
            fixture.a, ff1.collective_mma_inputs.layout_sfa,
            ff1.collective_mma_inputs.tensor_SFA.capacity());
        auto ff1_sfb = sm120_fused_reference::encode_scales(
            fixture.b1, ff1.collective_mma_inputs.layout_sfb,
            ff1.collective_mma_inputs.tensor_SFB.capacity());
        auto ff2_sfb = sm120_fused_reference::encode_scales(
            fixture.b2, ff2.collective_mma_inputs.layout_sfb,
            ff2.collective_mma_inputs.tensor_SFB.capacity());

        ff1.collective_mma_inputs.tensor_A.copy_in_host_to_host(
            reinterpret_cast<Ff1ElementA const*>(fixture.a.packed.data()));
        ff1.collective_mma_inputs.tensor_B.copy_in_host_to_host(
            reinterpret_cast<Ff1ElementB const*>(fixture.b1.packed.data()));
        ff1.collective_mma_inputs.tensor_SFA.copy_in_host_to_host(
            reinterpret_cast<Ff1ElementSF const*>(ff1_sfa.data()));
        ff1.collective_mma_inputs.tensor_SFB.copy_in_host_to_host(
            reinterpret_cast<Ff1ElementSF const*>(ff1_sfb.data()));
        ff2.collective_mma_inputs.tensor_B.copy_in_host_to_host(
            reinterpret_cast<Ff2ElementB const*>(fixture.b2.packed.data()));
        ff2.collective_mma_inputs.tensor_SFB.copy_in_host_to_host(
            reinterpret_cast<Ff2ElementSF const*>(ff2_sfb.data()));
        if (ff1.collective_epilogue.bias.capacity() !=
            static_cast<std::size_t>(ff_dim)) {
            throw std::runtime_error(
                "FF1 bias must be per output feature for Transformer semantics");
        }
        for (int column = 0; column < ff_dim; ++column) {
            const Ff1ElementBias rounded(fixture.bias[column]);
            fixture.bias[column] = float(rounded);
            ff1.collective_epilogue.bias.host_data(column) = rounded;
        }
        // CUTLASS computes SFD = quantized(amax * norm_constant / 6) and
        // scales D by norm_constant / SFD.  A value of one therefore makes
        // D * SFD reconstruct the quantized activation without an extra
        // matrix-wide multiplier.
        ff1.collective_epilogue.norm_constant.host_data(0) = 1.0F;

        ff1.collective_mma_inputs.tensor_A.sync_device();
        ff1.collective_mma_inputs.tensor_B.sync_device();
        ff1.collective_mma_inputs.tensor_SFA.sync_device();
        ff1.collective_mma_inputs.tensor_SFB.sync_device();
        ff1.collective_epilogue.bias.sync_device();
        ff1.collective_epilogue.norm_constant.sync_device();
        ff2.collective_mma_inputs.tensor_B.sync_device();
        ff2.collective_mma_inputs.tensor_SFB.sync_device();

        for (int first = oracle_begin; first < oracle_end; first += 128) {
            std::vector<int> rows;
            for (int row = first; row < std::min(first + 128, oracle_end); ++row) rows.push_back(row);
            sm120_fused_reference::compute_rows(fixture, rows);
            std::cout << "oracle_progress slot=" << oracle_slot
                      << " rows=" << std::min(first + 128, oracle_end) << '/' << m << std::endl;
        }
        if (launch_ff1_slot(oracle_slot) != cutlass::Status::kSuccess ||
            launch_scale_relayout_slot(oracle_slot) != cudaSuccess ||
            launch_ff2_slot(oracle_slot) != cutlass::Status::kSuccess) {
            throw std::runtime_error("stock FFN numerical launch failed");
        }
        const cudaError_t numeric_error = cudaDeviceSynchronize();
        if (numeric_error != cudaSuccess) {
            throw std::runtime_error(
                std::string("stock FFN numerical launch failed: ") +
                cudaGetErrorString(numeric_error));
        }

        auto& ff1_values = ff1.collective_epilogue.tensor_D;
        auto& ff1_output_scales = ff1.collective_epilogue.tensor_SFD;
        auto& ff2_values = ff2.collective_epilogue.tensor_D;
        ff1_values.sync_host();
        ff1_output_scales.sync_host();
        if (use_scale_relayout) {
            ff2.collective_mma_inputs.tensor_SFA.sync_host();
        }
        ff2_values.sync_host();
        std::vector<float> actual_hidden(static_cast<std::size_t>(oracle_end - oracle_begin) * ff_dim);
        std::vector<float> actual_hidden_sfd(actual_hidden.size());
        std::size_t scale_relayout_mismatches = 0;
        for (int row = oracle_begin; row < oracle_end; ++row) {
            for (int column = 0; column < ff_dim; ++column) {
                const auto source_scale_offset = static_cast<std::size_t>(
                    ff1_sfd_layout(row, column, 0));
                const auto destination_scale_offset = static_cast<std::size_t>(
                    ff2.collective_mma_inputs.layout_sfa(row, column, 0));
                const float output_scale =
                    float(ff1_output_scales.host_data(source_scale_offset));
                const float input_scale = use_scale_relayout
                    ? float(ff2.collective_mma_inputs.tensor_SFA.host_data(
                          destination_scale_offset))
                    : output_scale;
                scale_relayout_mismatches += output_scale != input_scale;
                const float value =
                    float(ff1_values.at(cutlass::make_Coord(row, column)));
                actual_hidden[static_cast<std::size_t>(row - oracle_begin) * ff_dim + column] =
                    value * input_scale;
                actual_hidden_sfd[
                    static_cast<std::size_t>(row - oracle_begin) * ff_dim + column] =
                    value * output_scale;
            }
        }
        std::vector<float> actual_ff2(static_cast<std::size_t>(oracle_end - oracle_begin) * d_model);
        std::vector<float> expected_ff2(actual_ff2.size());
        for (int row = oracle_begin; row < oracle_end; ++row) {
            for (int column = 0; column < d_model; ++column) {
                const auto index =
                    static_cast<std::size_t>(row - oracle_begin) * d_model + column;
                actual_ff2[index] =
                    float(ff2_values.at(cutlass::make_Coord(row, column)));
                expected_ff2[index] =
                    float(cutlass::bfloat16_t(fixture.ff2[static_cast<std::size_t>(row) * d_model + column]));
            }
        }
        const std::vector<float> expected_hidden(
            fixture.hidden.begin() + static_cast<std::size_t>(oracle_begin) * ff_dim,
            fixture.hidden.begin() + static_cast<std::size_t>(oracle_end) * ff_dim);
        const bool hidden_sfd_pass = sm120_fused_reference::compare(
            "stock_quantized_hidden_sfd", actual_hidden_sfd, expected_hidden);
        const bool hidden_pass = sm120_fused_reference::compare(
            "stock_quantized_hidden", actual_hidden, expected_hidden);
        const bool ff2_pass = sm120_fused_reference::compare(
            "stock_ff2_bf16", actual_ff2, expected_ff2);
        const bool chain_pass = hidden_sfd_pass && hidden_pass && ff2_pass &&
            scale_relayout_mismatches == 0;
        std::cout << "numeric_chain_pass=" << (chain_pass ? 1 : 0)
                  << " oracle=independent_scalar"
                  << " rows=" << m
                  << " seed=" << slot_seed
                  << " slot=" << oracle_slot
                  << " oracle_range_begin=" << oracle_begin
                  << " oracle_range_end=" << oracle_end
                  << " hidden_sfd_pass=" << (hidden_sfd_pass ? 1 : 0)
                  << " hidden_pass=" << (hidden_pass ? 1 : 0)
                  << " scale_relayout_mismatches="
                  << scale_relayout_mismatches
                  << " ff2_pass=" << (ff2_pass ? 1 : 0) << '\n';
        if (!chain_pass) {
            throw std::runtime_error("stock CUTLASS FFN numerical gate failed");
        }
        timed_output_reference[oracle_slot] = std::move(expected_ff2);
        }
    }

    auto check_profile_cuda = [](cudaError_t error) {
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }
    };
    struct StageEvents { cudaEvent_t begin{}, middle{}, end{}; };
    std::vector<StageEvents> stage_events;
    if (diagnostic_mode == 3) {
        if (iterations > 1000) {
            throw std::runtime_error("event diagnostic is bounded to 1000 iterations");
        }
        stage_events.resize(static_cast<std::size_t>(iterations) * slots);
        for (auto& events : stage_events) {
            check_profile_cuda(cudaEventCreate(&events.begin));
            check_profile_cuda(cudaEventCreate(&events.middle));
            check_profile_cuda(cudaEventCreate(&events.end));
        }
    }
    cudaEvent_t start{}, stop{};
    std::vector<cudaGraph_t> graphs(static_cast<std::size_t>(slots), nullptr);
    std::vector<cudaGraphExec_t> graph_execs(static_cast<std::size_t>(slots), nullptr);
    if (diagnostic_mode == 4) {
        check_profile_cuda(cudaDeviceSynchronize());
        for (int slot = 0; slot < slots; ++slot) {
            check_profile_cuda(cudaStreamBeginCapture(streams[slot], cudaStreamCaptureModeThreadLocal));
            if (launch_ff1_slot(slot) != cutlass::Status::kSuccess ||
                launch_scale_relayout_slot(slot) != cudaSuccess ||
                launch_ff2_slot(slot) != cutlass::Status::kSuccess) {
                throw std::runtime_error("graph FFN capture launch failed");
            }
            check_profile_cuda(cudaStreamEndCapture(streams[slot], &graphs[slot]));
            check_profile_cuda(cudaGraphInstantiate(&graph_execs[slot], graphs[slot], 0));
            check_profile_cuda(cudaGraphLaunch(graph_execs[slot], streams[slot]));
        }
        check_profile_cuda(cudaDeviceSynchronize());
    }
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (auto stream : streams) {
        cudaStreamWaitEvent(stream, start, 0);
    }
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int slot = 0; slot < slots; ++slot) {
            const auto index = static_cast<std::size_t>(iteration) * slots + slot;
            if (diagnostic_mode == 4) {
                check_profile_cuda(cudaGraphLaunch(graph_execs[slot], streams[slot]));
                continue;
            }
            if (diagnostic_mode == 3) {
                check_profile_cuda(cudaEventRecord(stage_events[index].begin, streams[slot]));
            }
            if (diagnostic_mode != 2) {
                if (launch_ff1_slot(slot) != cutlass::Status::kSuccess ||
                    launch_scale_relayout_slot(slot) != cudaSuccess) {
                    throw std::runtime_error("concurrent FF1 timed launch failed");
                }
            }
            if (diagnostic_mode == 3) {
                check_profile_cuda(cudaEventRecord(stage_events[index].middle, streams[slot]));
            }
            if (diagnostic_mode != 1 && launch_ff2_slot(slot) != cutlass::Status::kSuccess) {
                throw std::runtime_error("concurrent FF2 timed launch failed");
            }
            if (diagnostic_mode == 3) {
                check_profile_cuda(cudaEventRecord(stage_events[index].end, streams[slot]));
            }
        }
    }
    std::vector<cudaEvent_t> done(static_cast<std::size_t>(slots));
    for (int slot = 0; slot < slots; ++slot) {
        cudaEventCreateWithFlags(&done[static_cast<std::size_t>(slot)], cudaEventDisableTiming);
        cudaEventRecord(done[static_cast<std::size_t>(slot)], streams[static_cast<std::size_t>(slot)]);
        cudaStreamWaitEvent(0, done[static_cast<std::size_t>(slot)], 0);
    }
    check_profile_cuda(cudaEventRecord(stop, 0));
    check_profile_cuda(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    check_profile_cuda(cudaEventElapsedTime(&total_ms, start, stop));
    check_profile_cuda(cudaGetLastError());
    if (verify_numeric_chain) {
        for (int oracle_slot = 0; oracle_slot < slots; ++oracle_slot) {
        auto& values = ff2_testbeds[oracle_slot]->impl_.collective_epilogue.tensor_D;
        values.sync_host();
        std::vector<float> actual(timed_output_reference[oracle_slot].size());
        for (int row = oracle_begin; row < oracle_end; ++row) {
            for (int column = 0; column < d_model; ++column) {
                actual[static_cast<std::size_t>(row - oracle_begin) * d_model + column] =
                    float(values.at(cutlass::make_Coord(row, column)));
            }
        }
        if (!sm120_fused_reference::compare("post_timing_ff2", actual, timed_output_reference[oracle_slot])) {
            throw std::runtime_error("post-timing FFN output gate failed");
        }
        std::cout << "post_timing_output_pass=1 slot=" << oracle_slot << '\n';
        }
    }
    for (std::size_t index = 0; index < stage_events.size(); ++index) {
        auto& events = stage_events[index];
        float ff1_ms = 0, ff2_ms = 0;
        check_profile_cuda(cudaEventElapsedTime(&ff1_ms, events.begin, events.middle));
        check_profile_cuda(cudaEventElapsedTime(&ff2_ms, events.middle, events.end));
        std::cout << "ffn_stage_event iteration=" << index / slots
                  << " slot=" << index % slots
                  << " ff1_relu_pack_relayout_us=" << std::setprecision(9) << ff1_ms * 1000
                  << " ff2_store_us=" << ff2_ms * 1000
                  << " instrumented=1\n";
        check_profile_cuda(cudaEventDestroy(events.begin));
        check_profile_cuda(cudaEventDestroy(events.middle));
        check_profile_cuda(cudaEventDestroy(events.end));
    }
    for (int slot = 0; slot < slots; ++slot) {
        if (graph_execs[slot]) check_profile_cuda(cudaGraphExecDestroy(graph_execs[slot]));
        if (graphs[slot]) check_profile_cuda(cudaGraphDestroy(graphs[slot]));
        cudaEventDestroy(done[static_cast<std::size_t>(slot)]);
        cudaStreamDestroy(streams[static_cast<std::size_t>(slot)]);
    }
    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    const double ms = static_cast<double>(total_ms) / iterations;
    const double flops = (diagnostic_mode == 1 || diagnostic_mode == 2 ? 2.0 : 4.0)
        * static_cast<double>(m) * d_model * ff_dim * slots;
    const double tflops = flops / (ms * 1.0e9);
    std::cout << "sm120_dense_nvfp4_ffn_concurrent"
              << " m=" << m << " slots=" << slots << " iterations=" << iterations
              << " aggregate_ms=" << std::fixed << std::setprecision(6) << ms
              << " aggregate_tflops=" << std::setprecision(3) << tflops
              << " diagnostic_mode=" << diagnostic_mode
              << " persist_hidden_l2=" << (persist_hidden_l2 ? 1 : 0)
              << " hidden_value_bytes_per_slot=" << hidden_value_bytes
              << " max_persisting_l2_bytes=" << max_persisting_l2_bytes
              << " scale_handoff="
              << (use_scale_relayout ? "sfd_to_sfa_relayout" : "direct_alias")
              << " intermediate_format="
#if defined(BEAM_BENCH_MXFP4)
              << "mxfp4_relu\n";
#else
              << "nvfp4_relu\n";
#endif
    return tflops;
}
#endif
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
              << " dense_equivalent_tflops=" << std::setprecision(3) << tflops
#if defined(BEAM_BENCH_CODED_FF1)
              // The physical shape is (r*N)x(r*K) by (r*K)xM.  It represents
              // r independent logical MxK by KxN FF1 products, so the physical
              // dense-shape numerator is r times the useful numerator.
              << " coded_width=" << (k / 256)
              << " useful_ff1_tflops=" << std::setprecision(3)
              << (tflops / static_cast<double>(k / 256))
#endif
              << '\n';
    return tflops;
}

#if defined(BEAM_BENCH_CONCURRENT_SLOTS)
double benchmark_concurrent(int m, int n, int k, int iterations, int slots) {
    if (slots < 1 || slots > 8) {
        throw std::runtime_error("concurrent slots must be in [1, 8]");
    }
    ProblemShape problem{m, n, k, 1};
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

    std::vector<std::unique_ptr<Testbed>> testbeds;
    std::vector<std::unique_ptr<Gemm>> gemms;
    std::vector<std::unique_ptr<cutlass::device_memory::allocation<std::uint8_t>>> workspaces;
    std::vector<cudaStream_t> streams(static_cast<std::size_t>(slots));
    testbeds.reserve(static_cast<std::size_t>(slots));
    gemms.reserve(static_cast<std::size_t>(slots));
    workspaces.reserve(static_cast<std::size_t>(slots));

    for (int slot = 0; slot < slots; ++slot) {
        testbeds.push_back(std::make_unique<Testbed>());
        auto& impl = testbeds.back()->impl_;
        if (!impl.sufficient() || !impl.initialize(problem, 1.0f, 0.0f)) {
            throw std::runtime_error("concurrent NVFP4 fixture initialization failed");
        }
        typename GemmKernel::TileScheduler::Arguments scheduler_args{};
        typename Gemm::Arguments arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            problem,
            impl.collective_mma_inputs.to_args(),
            impl.collective_epilogue.to_args(problem),
            hw_info,
            scheduler_args};
        auto workspace = std::make_unique<cutlass::device_memory::allocation<std::uint8_t>>(
            Gemm::get_workspace_size(arguments));
        auto gemm = std::make_unique<Gemm>();
        if (gemm->can_implement(arguments) != cutlass::Status::kSuccess ||
            gemm->initialize(arguments, workspace->get()) != cutlass::Status::kSuccess) {
            throw std::runtime_error("concurrent NVFP4 GEMM initialization failed");
        }
        cudaStreamCreateWithFlags(&streams[static_cast<std::size_t>(slot)], cudaStreamNonBlocking);
        workspaces.push_back(std::move(workspace));
        gemms.push_back(std::move(gemm));
    }

    for (int warmup = 0; warmup < 10; ++warmup) {
        for (int slot = 0; slot < slots; ++slot) {
            if (gemms[static_cast<std::size_t>(slot)]->run(
                    streams[static_cast<std::size_t>(slot)]) != cutlass::Status::kSuccess) {
                throw std::runtime_error("concurrent NVFP4 warmup launch failed");
            }
        }
    }
    cudaDeviceSynchronize();

    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (cudaStream_t stream : streams) {
        cudaStreamWaitEvent(stream, start, 0);
    }
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int slot = 0; slot < slots; ++slot) {
            if (gemms[static_cast<std::size_t>(slot)]->run(
                    streams[static_cast<std::size_t>(slot)]) != cutlass::Status::kSuccess) {
                throw std::runtime_error("concurrent NVFP4 timed launch failed");
            }
        }
    }
    std::vector<cudaEvent_t> done(static_cast<std::size_t>(slots));
    for (int slot = 0; slot < slots; ++slot) {
        cudaEventCreateWithFlags(&done[static_cast<std::size_t>(slot)], cudaEventDisableTiming);
        cudaEventRecord(done[static_cast<std::size_t>(slot)], streams[static_cast<std::size_t>(slot)]);
        cudaStreamWaitEvent(0, done[static_cast<std::size_t>(slot)], 0);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float total_ms = 0.0F;
    cudaEventElapsedTime(&total_ms, start, stop);

#if defined(BEAM_BENCH_K_SWEEP_VERIFY)
    // Bounded host oracle, using the same timed kernel specialization.
    // Large timing fixtures deliberately skip the expensive host reference.
    if (m <= 256) {
        for (auto& bed : testbeds) {
            if (!bed->impl_.verify(problem, 1.0f, 0.0f)) {
                throw std::runtime_error("K sweep output oracle failed");
            }
        }
        std::cout << "k_sweep_output_verified=1 m=" << m << " k=" << k << '\n';
    }
    std::cout << "k_sweep_plan stages=" << CollectiveMainloop::DispatchPolicy::Stages
              << " shared_bytes=" << GemmKernel::SharedStorageSize << '\n';
#endif

    for (int slot = 0; slot < slots; ++slot) {
        cudaEventDestroy(done[static_cast<std::size_t>(slot)]);
        cudaStreamDestroy(streams[static_cast<std::size_t>(slot)]);
    }
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    const double ms = static_cast<double>(total_ms) / iterations;
    const double flops = 2.0 * static_cast<double>(m) * n * k * slots;
    const double tflops = flops / (ms * 1.0e9);
    std::cout <<
#if defined(BEAM_BENCH_CODED_FF1)
              "sm120_sparse_nvfp4_pair_ff1_concurrent"
#else
              "sm120_dense_nvfp4_concurrent"
#endif
              << " m=" << m << " n=" << n << " k=" << k
              << " slots=" << slots << " iterations=" << iterations
              << " aggregate_ms=" << std::fixed << std::setprecision(6) << ms
              << " aggregate_tflops=" << std::setprecision(3) << tflops
#if defined(BEAM_BENCH_CODED_FF1)
              << " useful_ff1_tflops=" << std::setprecision(3)
              << (tflops / static_cast<double>(k / 256))
#endif
              << '\n';
    return tflops;
}
#endif

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
#if defined(BEAM_BENCH_CODED_FF1)
        const int coded_width = k / 256;
        if ((coded_width != 1 && coded_width != 2 &&
             coded_width != 4 && coded_width != 8) ||
            k != coded_width * 256 || m != coded_width * 1024) {
            throw std::runtime_error(
                "coded FF1 physical shape must be m=width*1024,k=width*256 "
                "for width in {1,2,4,8}; width=1 is the exact W^T*X^T "
                "N-packed FF1; n is the original FF1 row count");
        }
#endif
#if defined(BEAM_BENCH_DENSE_NVFP4_OUTPUT) && defined(BEAM_BENCH_TILE_K128)
        std::cout << "sm120_nvfp4_kernel_plan"
                  << " stages="
                  << sm120_sparse_mxfp8_bench::CollectiveMainloop::DispatchPolicy::Stages
                  << " shared_bytes="
                  << sm120_sparse_mxfp8_bench::GemmKernel::SharedStorageSize
                  << " reserved_hidden_bytes="
                  << sm120_sparse_mxfp8_bench::kFusedHiddenBytes << '\n';
#endif
#if defined(BEAM_BENCH_PIPELINE_CONCURRENT)
        const int pipeline_iterations = argc > 2 ? std::atoi(argv[2]) : 100;
        const int slots = argc > 3 ? std::atoi(argv[3]) : 2;
        const bool persist_hidden_l2 = argc > 4 ? std::atoi(argv[4]) != 0 : false;
        const bool verify_numeric_chain = argc > 5 ? std::atoi(argv[5]) != 0 : false;
        const auto numeric_seed = argc > 6
            ? static_cast<std::uint32_t>(std::strtoul(argv[6], nullptr, 10))
            : 17U;
        const bool use_scale_relayout = argc > 7
            ? std::atoi(argv[7]) != 0
            : false;
        const int diagnostic_mode = argc > 8 ? std::atoi(argv[8]) : 0;
        sm120_sparse_mxfp8_bench::benchmark_ffn_pipeline_concurrent(
            m, pipeline_iterations, slots, persist_hidden_l2,
            verify_numeric_chain, numeric_seed, use_scale_relayout, diagnostic_mode);
#elif defined(BEAM_BENCH_DENSE_NVFP4_PIPELINE)
        const int pipeline_iterations = argc > 2 ? std::atoi(argv[2]) : 100;
        sm120_sparse_mxfp8_bench::benchmark_ffn_pipeline(m, pipeline_iterations);
#elif defined(BEAM_BENCH_CONCURRENT_SLOTS)
        const int slots = argc > 5 ? std::atoi(argv[5]) : 1;
        sm120_sparse_mxfp8_bench::benchmark_concurrent(m, n, k, iterations, slots);
#else
        sm120_sparse_mxfp8_bench::benchmark(m, n, k, iterations);
#endif
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
