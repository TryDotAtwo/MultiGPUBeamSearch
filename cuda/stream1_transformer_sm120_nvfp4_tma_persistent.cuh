#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>

#include <cstddef>
#include <cstdint>

namespace stream1_sm120_nvfp4_tma_persistent {

using namespace cute;

struct Contract {
    static constexpr std::uint32_t kRowsPerCta = 128U;
    static constexpr std::uint32_t kDModel = 256U;
    static constexpr std::uint32_t kFfDim = 1024U;
    static constexpr std::uint32_t kHiddenSliceColumns = 128U;
    static constexpr std::uint32_t kHiddenSlices =
        kFfDim / kHiddenSliceColumns;
    static constexpr std::uint32_t kRingBuffers = 2U;
    static constexpr std::uint32_t kScaleVector = 16U;
    static constexpr std::uint32_t kValueBytesPerBuffer =
        kRowsPerCta * kHiddenSliceColumns / 2U;
    static constexpr std::uint32_t kScaleBytesPerBuffer =
        kRowsPerCta * (kHiddenSliceColumns / kScaleVector);
    static constexpr std::uint32_t kHandoffBytes =
        kRingBuffers * (kValueBytesPerBuffer + kScaleBytesPerBuffer);
};

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementAccumulator = float;
using ElementCompute = float;
using ElementScale = cutlass::float_ue4m3_t;
using ElementBias = cutlass::half_t;
using ElementResidual = cutlass::half_t;
using ElementHidden = cutlass::float_e2m1_t;
using ElementOutput = cutlass::half_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;
using ProblemShape = Shape<int, int, int, int>;
using ArchTag = cutlass::arch::Sm120;
using MainloopOpClass = cutlass::arch::OpClassBlockScaledTensorOp;
using KernelSchedule =
    cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120;
using EpilogueSchedule =
    cutlass::epilogue::collective::EpilogueScheduleAuto;

static constexpr int kAlignmentA = 32;
static constexpr int kAlignmentB = 32;
static constexpr int kAlignmentResidual = 8;
static constexpr int kAlignmentHidden = 32;
static constexpr int kAlignmentOutput = 8;

using Ff1Fusion =
    cutlass::epilogue::fusion::LinCombPerRowBiasEltActBlockScaleFactor<
        cutlass::epilogue::thread::ReLU,
        Contract::kScaleVector,
        ElementHidden,
        ElementCompute,
        ElementScale,
        LayoutD,
        ElementBias,
        ElementResidual>;

using Ff1Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementResidual,
    LayoutC,
    kAlignmentResidual,
    ElementHidden,
    LayoutD,
    kAlignmentHidden,
    EpilogueSchedule,
    Ff1Fusion>::CollectiveOp;

using Ff1StageCount = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename Ff1Epilogue::SharedStorage)) +
    static_cast<int>(Contract::kHandoffBytes)>;

using Ff1Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    MainloopOpClass,
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    Ff1StageCount,
    KernelSchedule>::CollectiveOp;

using Ff1Kernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, Ff1Mainloop, Ff1Epilogue, void>;

using Ff2Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementOutput,
    LayoutC,
    kAlignmentOutput,
    ElementOutput,
    LayoutD,
    kAlignmentOutput,
    EpilogueSchedule>::CollectiveOp;

using Ff2StageCount = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename Ff2Epilogue::SharedStorage)) +
    static_cast<int>(Contract::kHandoffBytes)>;

using Ff2Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    MainloopOpClass,
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    Ff2StageCount,
    KernelSchedule>::CollectiveOp;

using Ff2Kernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, Ff2Mainloop, Ff2Epilogue, void>;

union alignas(128) CollectiveArena {
    typename Ff1Kernel::SharedStorage ff1;
    typename Ff2Kernel::SharedStorage ff2;
};

struct alignas(128) SharedStorage {
    CollectiveArena collective;
    alignas(128) std::uint8_t
        hidden_values[Contract::kRingBuffers][Contract::kValueBytesPerBuffer];
    alignas(128) std::uint8_t
        hidden_scales[Contract::kRingBuffers][Contract::kScaleBytesPerBuffer];
    alignas(16) std::uint64_t ready_barriers[Contract::kRingBuffers];
    alignas(16) std::uint64_t free_barriers[Contract::kRingBuffers];
};

struct Plan : Contract {
    static constexpr std::uint32_t kGlobalHiddenBytes = 0U;
    static constexpr std::uint32_t kSm120SharedCapacityBytes =
        ArchTag::kSharedMemoryCapacityBytes;
    static constexpr std::uint32_t kFf1MainloopStages =
        Ff1Mainloop::DispatchPolicy::Stages;
    static constexpr std::uint32_t kFf2MainloopStages =
        Ff2Mainloop::DispatchPolicy::Stages;
    static constexpr std::uint32_t kSharedStorageBytes = sizeof(SharedStorage);
};

static_assert(Plan::kHandoffBytes == 18432U);
static_assert(Plan::kSharedStorageBytes <= Plan::kSm120SharedCapacityBytes);

}  // namespace stream1_sm120_nvfp4_tma_persistent
