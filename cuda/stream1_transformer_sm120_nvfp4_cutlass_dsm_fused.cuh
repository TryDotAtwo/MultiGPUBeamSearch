#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/pipeline.hpp>
#include <cuda/__ptx/instructions/cp_async_bulk.h>
#include <cooperative_groups.h>

#include <cstddef>
#include <cstdint>

namespace stream1_sm120_nvfp4_cutlass_dsm {

using namespace cute;

struct Contract {
    static constexpr std::uint32_t kRows = 128U;
    static constexpr std::uint32_t kDModel = 256U;
    static constexpr std::uint32_t kFfDim = 1024U;
    static constexpr std::uint32_t kSliceColumns = 128U;
    static constexpr std::uint32_t kSlices = kFfDim / kSliceColumns;
    static constexpr std::uint32_t kClusterCtas = 2U;
    static constexpr std::uint32_t kRingSlots = 2U;
    static constexpr std::uint32_t kScaleVector = 16U;
    static constexpr std::uint32_t kValuesPerSlot =
        kRows * kSliceColumns / 2U;
    static constexpr std::uint32_t kScalesPerSlot =
        kRows * (kSliceColumns / kScaleVector);
    static constexpr std::uint32_t kRingBytes =
        kRingSlots * (kValuesPerSlot + kScalesPerSlot);
};

using ElementAB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementAccumulator = float;
using ElementCompute = float;
using ElementHidden = cutlass::float_e2m1_t;
using ElementScale = cutlass::float_ue4m3_t;
using ElementBias = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
using TileShape = Shape<_128, _128, _128>;
using CollectiveClusterShape = Shape<_1, _1, _1>;
using LaunchClusterShape = Shape<_2, _1, _1>;
using ProblemShape = Shape<int, int, int, int>;
using ArchTag = cutlass::arch::Sm120;
using KernelSchedule =
    cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120;
using EpilogueSchedule =
    cutlass::epilogue::collective::EpilogueScheduleAuto;

using Ff1Fusion =
    cutlass::epilogue::fusion::LinCombPerRowBiasEltActBlockScaleFactor<
        cutlass::epilogue::thread::ReLU,
        Contract::kScaleVector,
        ElementHidden,
        ElementCompute,
        ElementScale,
        LayoutD,
        ElementBias,
        ElementOutput>;

using Ff1Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp,
    TileShape, CollectiveClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementOutput, LayoutD, 8,
    ElementHidden, LayoutD, 32,
    EpilogueSchedule, Ff1Fusion>::CollectiveOp;

using Ff1Stages = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename Ff1Epilogue::SharedStorage)) +
    static_cast<int>(Contract::kRingBytes)>;

using Ff1Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementAB, LayoutA, 32,
    ElementAB, LayoutB, 32,
    ElementAccumulator, TileShape, CollectiveClusterShape,
    Ff1Stages, KernelSchedule>::CollectiveOp;

using Ff2Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp,
    TileShape, CollectiveClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementOutput, LayoutD, 8,
    ElementOutput, LayoutD, 8,
    EpilogueSchedule>::CollectiveOp;

using Ff2Stages = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename Ff2Epilogue::SharedStorage)) +
    static_cast<int>(Contract::kRingBytes)>;

using Ff2Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementAB, LayoutA, 32,
    ElementAB, LayoutB, 32,
    ElementAccumulator, TileShape, CollectiveClusterShape,
    Ff2Stages, KernelSchedule>::CollectiveOp;

using Ff1Kernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, Ff1Mainloop, Ff1Epilogue, void>;
using Ff2Kernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, Ff2Mainloop, Ff2Epilogue, void>;

// Production FF2 seam: A/SFA are delivered by the FF1 owner CTA through a
// shared-to-cluster-shared bulk copy, while B/SFB retain the stock CUTLASS TMA
// producer and FF2 retains Ff2Mainloop::mma().  Both transfers contribute to
// the same CUTLASS pipeline transaction barrier, so the consumer observes one
// ready event and no intermediate global hidden tensor.
struct BulkDsmFf2Collective {
    using MainloopPipeline = typename Ff2Mainloop::MainloopPipeline;
    using PipelineState = typename Ff2Mainloop::PipelineState;
    struct SharedStorage;

    static constexpr bool kLoadsGlobalA = false;
    static constexpr bool kLoadsGlobalSFA = false;
    static constexpr bool kLoadsGlobalB = true;
    static constexpr bool kLoadsGlobalSFB = true;
    static constexpr bool kUsesCutlassConsumerMma = true;
    static constexpr bool kUsesBulkDsm = true;
    static constexpr bool kUsesPipelineTransactionBarrier = true;
    static constexpr bool kUsesFullClusterSyncInHotLoop = false;
    static constexpr std::uint32_t kDsmTransactionBytes =
        Ff2Mainloop::TmaTransactionBytesMK;
    static constexpr std::uint32_t kTmaTransactionBytes =
        Ff2Mainloop::TmaTransactionBytesNK;
    static constexpr std::uint32_t kTotalTransactionBytes =
        Ff2Mainloop::TmaTransactionBytes;
    static constexpr std::uint32_t kThreads = 256U;
};

union alignas(128) CutlassArena {
    typename Ff1Kernel::SharedStorage ff1;
    typename Ff2Kernel::SharedStorage ff2;
};

// The remote operand is not merely a byte queue.  Its typed overlay is the
// exact A/SFA shared-memory representation consumed by the fast CUTLASS SM120
// mainloop.  FF1's fused epilogue will eventually target this layout directly,
// so FF2 can retain its register-copy/MMA path without a DSM repack.
struct alignas(1024) DsmASlot {
    alignas(1024) std::uint8_t values[Contract::kValuesPerSlot];
    alignas(16) std::uint8_t scales[Contract::kScalesPerSlot];
};

using Ff2StageLayoutA = decltype(
    typename Ff2Mainloop::SmemLayoutA{}(_, _, Int<0>{}));
using Ff2StageLayoutSFA = decltype(
    typename Ff2Mainloop::SmemLayoutSFA{}(_, _, Int<0>{}));

CUTLASS_DEVICE auto make_dsm_a_stage_view(std::uint8_t* values) {
    using Storage = typename Ff2Mainloop::SmemAllocTypeA;
    return make_tensor(
        make_smem_ptr(reinterpret_cast<Storage*>(values)),
        Ff2StageLayoutA{});
}

CUTLASS_DEVICE auto make_dsm_sfa_stage_view(std::uint8_t* scales) {
    return make_tensor(
        make_smem_ptr(reinterpret_cast<ElementScale*>(scales)),
        Ff2StageLayoutSFA{});
}

struct alignas(1024) BulkDsmFf2Collective::SharedStorage {
    alignas(1024) typename Ff2Mainloop::TensorStorage tensors;
    alignas(16) typename MainloopPipeline::SharedStorage pipeline;
    DsmASlot hidden[Contract::kRingSlots];
    alignas(8) std::uint64_t handoff_armed;
};

static_assert(BulkDsmFf2Collective::kDsmTransactionBytes ==
              Contract::kValuesPerSlot + Contract::kScalesPerSlot);
static_assert(BulkDsmFf2Collective::kTmaTransactionBytes +
                  BulkDsmFf2Collective::kDsmTransactionBytes ==
              BulkDsmFf2Collective::kTotalTransactionBytes);
static_assert(sizeof(BulkDsmFf2Collective::SharedStorage) <=
              ArchTag::kSharedMemoryCapacityBytes);

template <class LoadInputs>
CUTLASS_DEVICE void issue_ff2_b_sfb_tma(
    typename Ff2Mainloop::Params const& params,
    typename BulkDsmFf2Collective::MainloopPipeline& pipeline,
    typename BulkDsmFf2Collective::PipelineState write_state,
    LoadInputs const& load_inputs,
    std::uint32_t n_tile,
    std::uint32_t k_tile,
    BulkDsmFf2Collective::SharedStorage& storage) {
    auto [gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl] = load_inputs;
    (void)gA_mkl;
    (void)gSFA_mkl;
    auto block_tma_b = params.tma_load_b.get_slice(0);
    auto block_tma_sfb = params.tma_load_sfb.get_slice(0);
    Tensor gB = gB_nkl(_, _, n_tile, _, 0);
    auto broadcast_n = make_layout(
        make_shape(
            Int<size<1>(typename Ff2Mainloop::TileShapeSFB{}) /
                size<1>(TileShape{})>{},
            Int<cute::numeric_limits<int>::max()>{}),
        make_stride(
            _0{}, size<1>(typename Ff2Mainloop::TileShapeSFB{}) /
                size<1>(TileShape{})));
    Tensor gSFB = gSFB_nkl(_, _, broadcast_n(n_tile), _, 0);
    Tensor sB = make_tensor(
        make_smem_ptr(storage.tensors.smem_B.begin()),
        typename Ff2Mainloop::SmemLayoutB{});
    Tensor sSFB = make_tensor(
        make_smem_ptr(storage.tensors.smem_SFB.begin()),
        typename Ff2Mainloop::SmemLayoutSFB{});
    Tensor tBgB = block_tma_b.partition_S(gB);
    Tensor tBsB = block_tma_b.partition_D(sB);
    Tensor tBgSFB = block_tma_sfb.partition_S(gSFB);
    Tensor tBsSFB = block_tma_sfb.partition_D(sSFB);
    using Barrier = typename BulkDsmFf2Collective::MainloopPipeline::
        ProducerBarrierType;
    Barrier* barrier = pipeline.producer_get_barrier(write_state);
    const int stage = write_state.index();
    copy(params.tma_load_b.with(*barrier),
         tBgB(_, _, _, k_tile), tBsB(_, _, _, stage));
    copy(params.tma_load_sfb.with(*barrier),
         tBgSFB(_, _, _, k_tile), tBsSFB(_, _, _, stage));
}

CUTLASS_DEVICE void arrive_handoff_armed(
    std::uint64_t* local_barrier,
    std::uint32_t owner_rank) {
    cutlass::arch::ClusterBarrier::arrive(
        local_barrier, owner_rank, 1U);
}

CUTLASS_DEVICE void issue_ff2_a_sfa_bulk_dsm(
    typename BulkDsmFf2Collective::MainloopPipeline& pipeline,
    typename BulkDsmFf2Collective::PipelineState write_state,
    BulkDsmFf2Collective::SharedStorage& storage,
    std::uint32_t slot) {
    namespace cg = cooperative_groups;
    const auto cluster = cg::this_cluster();
    const int stage = write_state.index();
    auto* local_a = reinterpret_cast<std::uint8_t*>(
        &storage.tensors.smem_A) +
        static_cast<std::size_t>(stage) * Contract::kValuesPerSlot;
    auto* local_sfa = reinterpret_cast<std::uint8_t*>(
        &storage.tensors.smem_SFA) +
        static_cast<std::size_t>(stage) * Contract::kScalesPerSlot;
    auto* local_barrier = pipeline.producer_get_barrier(write_state);
    constexpr std::uint32_t values_bytes = Contract::kValuesPerSlot;
    constexpr std::uint32_t scales_bytes = Contract::kScalesPerSlot;

    for (std::uint32_t target = 0; target < Contract::kClusterCtas;
         ++target) {
        auto* target_a = cluster.map_shared_rank(local_a, target);
        auto* target_sfa = cluster.map_shared_rank(local_sfa, target);
        auto* target_barrier = cluster.map_shared_rank(
            local_barrier, target);
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_cluster, cuda::ptx::space_shared,
            target_a, storage.hidden[slot].values,
            values_bytes, target_barrier);
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_cluster, cuda::ptx::space_shared,
            target_sfa, storage.hidden[slot].scales,
            scales_bytes, target_barrier);
    }
}

// Executable integration gate for the production seam.  It deliberately
// delegates the complete consumer path to Ff2Mainloop::mma(): only its A/SFA
// producer is replaced.  The sole cluster-wide synchronization initializes
// cluster-visible barriers; the eight-slice hot loop uses per-phase mbarriers.
static __global__ __launch_bounds__(256, 1) __cluster_dims__(2, 1, 1)
void bulk_dsm_ff2_collective_kernel(
    CUTE_GRID_CONSTANT typename Ff2Mainloop::Params const params,
    float* rank_checksums) {
    namespace cg = cooperative_groups;
    extern __shared__ __align__(1024) unsigned char shared_bytes[];
    auto& storage = *reinterpret_cast<
        BulkDsmFf2Collective::SharedStorage*>(shared_bytes);
    const auto cluster = cg::this_cluster();
    const std::uint32_t rank = cluster.block_rank();
    const std::uint32_t warp_group = threadIdx.x / 128U;
    const std::uint32_t wg_thread = threadIdx.x % 128U;

    if (threadIdx.x == 0U) {
        cutlass::arch::ClusterBarrier::init(&storage.handoff_armed, 2U);
    }
    for (std::uint32_t slot = 0; slot < Contract::kRingSlots; ++slot) {
        for (std::uint32_t i = threadIdx.x;
             i < Contract::kValuesPerSlot; i += blockDim.x) {
            storage.hidden[slot].values[i] = 0x11U;
        }
        for (std::uint32_t i = threadIdx.x;
             i < Contract::kScalesPerSlot; i += blockDim.x) {
            storage.hidden[slot].scales[i] = ElementScale(1.0F).raw();
        }
    }

    using Pipeline = BulkDsmFf2Collective::MainloopPipeline;
    using PipelineState = BulkDsmFf2Collective::PipelineState;
    typename Pipeline::Params pipeline_params;
    pipeline_params.transaction_bytes =
        BulkDsmFf2Collective::kTotalTransactionBytes;
    pipeline_params.num_consumers = 128U;
    pipeline_params.is_leader = wg_thread == 0U;
    pipeline_params.role = warp_group == 0U
        ? Pipeline::ThreadCategory::Producer
        : Pipeline::ThreadCategory::Consumer;
    Pipeline pipeline(
        storage.pipeline, pipeline_params, LaunchClusterShape{});
    __syncthreads();
    cutlass::arch::fence_barrier_init();
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();

    constexpr auto problem_shape = make_shape(128, 256, 1024, 1);
    Ff2Mainloop mainloop;
    const auto load_inputs = mainloop.load_init(problem_shape, params);
    if (warp_group == 0U && wg_thread == 0U) {
        PipelineState write_state =
            cutlass::make_producer_start_state<Pipeline>();
        for (std::uint32_t slice = 0; slice < Contract::kSlices; ++slice) {
            pipeline.producer_acquire(write_state);
            issue_ff2_b_sfb_tma(
                params, pipeline, write_state, load_inputs,
                rank, slice, storage);
            const std::uint32_t owner = slice & 1U;
            arrive_handoff_armed(&storage.handoff_armed, owner);
            if (rank == owner) {
                cutlass::arch::ClusterBarrier::wait(
                    &storage.handoff_armed, (slice / 2U) & 1U);
                issue_ff2_a_sfa_bulk_dsm(
                    pipeline, write_state, storage,
                    slice % Contract::kRingSlots);
            }
            ++write_state;
        }
        pipeline.producer_tail(write_state);
    } else if (warp_group == 1U) {
        typename Ff2Mainloop::TiledMma tiled_mma;
        auto accum = partition_fragment_C(
            tiled_mma, take<0, 2>(TileShape{}));
        PipelineState read_state;
        mainloop.mma(
            pipeline, read_state, accum, Contract::kSlices,
            static_cast<int>(wg_thread), storage.tensors,
            params, make_coord(0, rank, 0, 0));
        float local = 0.0F;
        for (int i = 0; i < size(accum); ++i) {
            local += accum(i) < 0.0F ? -accum(i) : accum(i);
        }
        const std::uint32_t cluster_index = blockIdx.x / 2U;
        atomicAdd(rank_checksums + cluster_index * 2U + rank, local);
    }
}

struct alignas(128) SharedStorage {
    CutlassArena cutlass;
    DsmASlot hidden[Contract::kRingSlots];
    alignas(16) std::uint64_t ready[Contract::kRingSlots];
    alignas(16) std::uint64_t free[Contract::kRingSlots];
};

struct Plan : Contract {
    static constexpr std::uint32_t kGlobalHiddenBytes = 0U;
    static constexpr std::uint32_t kFf1ProducerCtasPerSlice = 1U;
    static constexpr std::uint32_t kFf2ConsumerCtas = 2U;
    static constexpr std::uint32_t kOutputColumnsPerConsumer = 128U;
    static constexpr std::uint32_t kSharedBytes = sizeof(SharedStorage);
    static constexpr std::uint32_t kFf1Stages =
        Ff1Mainloop::DispatchPolicy::Stages;
    static constexpr std::uint32_t kFf2Stages =
        Ff2Mainloop::DispatchPolicy::Stages;
    // FlashAttention-4-style current/previous schedule.  Four warps in the
    // owner CTA produce slice N while four consumer warps in both CTAs consume
    // slice N-1.  The final stage only drains the last produced slice.
    static constexpr std::uint32_t kContractProducerWarps = 4U;
    static constexpr std::uint32_t kContractConsumerWarps = 4U;
    static constexpr std::uint32_t kContractThreads =
        (kContractProducerWarps + kContractConsumerWarps) * 32U;
    // Production preserves the native SM120 CUTLASS ping-pong composition:
    // one DMA warpgroup and two independent math warpgroups.  The math groups
    // become FF1-current and FF2-previous instead of alternating two tiles of
    // one GEMM.  This is the direct analogue of FA4's two-tile ping-pong.
    static constexpr std::uint32_t kDmaWarps = 4U;
    static constexpr std::uint32_t kFf1MathWarps = 4U;
    static constexpr std::uint32_t kFf2MathWarps = 4U;
    static constexpr std::uint32_t kKernelThreads =
        (kDmaWarps + kFf1MathWarps + kFf2MathWarps) * 32U;
    static constexpr std::uint32_t kPipelineStages = kSlices + 1U;
    static constexpr std::uint32_t kMaximumLiveSlices = 2U;

    static constexpr std::int32_t produced_slice(std::uint32_t stage) {
        return stage < kSlices ? static_cast<std::int32_t>(stage) : -1;
    }

    static constexpr std::int32_t consumed_slice(std::uint32_t stage) {
        return stage > 0U ? static_cast<std::int32_t>(stage - 1U) : -1;
    }
};

static_assert(size(LaunchClusterShape{}) == Contract::kClusterCtas);
static_assert(Plan::kFf1ProducerCtasPerSlice == 1U);
static_assert(Plan::kFf2ConsumerCtas == 2U);
static_assert(Plan::kOutputColumnsPerConsumer * Plan::kFf2ConsumerCtas ==
              Contract::kDModel);
static_assert(Plan::kRingBytes == 18432U);
static_assert(sizeof(DsmASlot) ==
              Contract::kValuesPerSlot + Contract::kScalesPerSlot);
static_assert(offsetof(DsmASlot, scales) ==
              Contract::kValuesPerSlot);
static_assert(
    cutlass::bits_to_bytes(
        cosize(Ff2StageLayoutA{}) *
        cute::sizeof_bits_v<typename Ff2Mainloop::SmemAllocTypeA>) ==
    Contract::kValuesPerSlot);
static_assert(
    cutlass::bits_to_bytes(
        cosize(Ff2StageLayoutSFA{}) * cute::sizeof_bits_v<ElementScale>) ==
    Contract::kScalesPerSlot);
static_assert(Plan::kRingSlots >= Plan::kMaximumLiveSlices);
static_assert(Plan::kContractThreads == 256U);
static_assert(Plan::kKernelThreads == 384U);
static_assert(Plan::produced_slice(0U) == 0);
static_assert(Plan::consumed_slice(0U) == -1);
static_assert(Plan::produced_slice(Plan::kSlices) == -1);
static_assert(Plan::consumed_slice(Plan::kSlices) ==
              static_cast<std::int32_t>(Plan::kSlices - 1U));
static_assert(Plan::kSharedBytes <= ArchTag::kSharedMemoryCapacityBytes);

// Hardware ownership gate for the eventual CUTLASS collective kernel. Slice
// ownership alternates between the two CTAs; both consumers map the owner's
// shared ring through DSM before the slot can be reused.
static __global__ __cluster_dims__(2, 1, 1) void dsm_ring_contract_kernel(
    std::uint64_t* checksums, std::uint32_t* violations) {
    namespace cg = cooperative_groups;
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& storage = *reinterpret_cast<SharedStorage*>(shared_bytes);
    const auto cluster = cg::this_cluster();
    const std::uint32_t rank = cluster.block_rank();

    for (std::uint32_t slice = 0; slice < Contract::kSlices; ++slice) {
        const std::uint32_t owner = slice & 1U;
        const std::uint32_t slot = slice % Contract::kRingSlots;
        if (rank == owner) {
            for (std::uint32_t byte = threadIdx.x;
                 byte < Contract::kValuesPerSlot; byte += blockDim.x) {
                storage.hidden[slot].values[byte] = static_cast<std::uint8_t>(
                    (slice * 29U + byte * 7U) & 0xffU);
            }
            for (std::uint32_t byte = threadIdx.x;
                 byte < Contract::kScalesPerSlot; byte += blockDim.x) {
                storage.hidden[slot].scales[byte] = static_cast<std::uint8_t>(
                    (slice * 13U + byte * 3U) & 0xffU);
            }
        }
        cluster.sync();
        auto* remote_values = cluster.map_shared_rank(
            storage.hidden[slot].values, owner);
        auto* remote_scales = cluster.map_shared_rank(
            storage.hidden[slot].scales, owner);
        std::uint64_t local = 0U;
        for (std::uint32_t byte = threadIdx.x;
             byte < Contract::kValuesPerSlot; byte += blockDim.x) {
            const std::uint8_t expected = static_cast<std::uint8_t>(
                (slice * 29U + byte * 7U) & 0xffU);
            const std::uint8_t actual = remote_values[byte];
            local += actual;
            if (actual != expected) {
                atomicAdd(violations, 1U);
            }
        }
        for (std::uint32_t byte = threadIdx.x;
             byte < Contract::kScalesPerSlot; byte += blockDim.x) {
            const std::uint8_t expected = static_cast<std::uint8_t>(
                (slice * 13U + byte * 3U) & 0xffU);
            const std::uint8_t actual = remote_scales[byte];
            local += actual;
            if (actual != expected) {
                atomicAdd(violations, 1U);
            }
        }
        atomicAdd(
            reinterpret_cast<unsigned long long*>(
                checksums + static_cast<std::size_t>(rank) *
                    Contract::kSlices + slice),
            static_cast<unsigned long long>(local));
        cluster.sync();
    }
}

// Executable current/previous ownership gate.  This is deliberately separate
// from dsm_ring_contract_kernel: the older kernel proves remote addressing,
// while this one proves the production schedule does useful producer and
// consumer work in the same stage.  A cluster fence ends each stage; no fence
// exists between producing slice N and consuming slice N-1.
static __global__ __cluster_dims__(2, 1, 1) void dsm_pingpong_overlap_kernel(
    std::uint64_t* checksums, std::uint32_t* violations) {
    namespace cg = cooperative_groups;
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& storage = *reinterpret_cast<SharedStorage*>(shared_bytes);
    const auto cluster = cg::this_cluster();
    const std::uint32_t rank = cluster.block_rank();
    const std::uint32_t local_thread = threadIdx.x;
    constexpr std::uint32_t producer_threads =
        Plan::kContractProducerWarps * 32U;
    constexpr std::uint32_t consumer_threads =
        Plan::kContractConsumerWarps * 32U;
    const bool producer = local_thread < producer_threads;
    const std::uint32_t role_thread = producer
        ? local_thread : local_thread - producer_threads;

    for (std::uint32_t stage = 0; stage < Plan::kPipelineStages; ++stage) {
        const std::int32_t produce = Plan::produced_slice(stage);
        const std::int32_t consume = Plan::consumed_slice(stage);

        if (producer && produce >= 0) {
            const std::uint32_t slice = static_cast<std::uint32_t>(produce);
            const std::uint32_t owner = slice & 1U;
            const std::uint32_t slot = slice % Plan::kRingSlots;
            if (rank == owner) {
                for (std::uint32_t byte = role_thread;
                     byte < Contract::kValuesPerSlot;
                     byte += producer_threads) {
                    storage.hidden[slot].values[byte] =
                        static_cast<std::uint8_t>(
                            (slice * 29U + byte * 7U) & 0xffU);
                }
                for (std::uint32_t byte = role_thread;
                     byte < Contract::kScalesPerSlot;
                     byte += producer_threads) {
                    storage.hidden[slot].scales[byte] =
                        static_cast<std::uint8_t>(
                            (slice * 13U + byte * 3U) & 0xffU);
                }
            }
        }

        if (!producer && consume >= 0) {
            const std::uint32_t slice = static_cast<std::uint32_t>(consume);
            const std::uint32_t owner = slice & 1U;
            const std::uint32_t slot = slice % Plan::kRingSlots;
            auto* remote_values = cluster.map_shared_rank(
                storage.hidden[slot].values, owner);
            auto* remote_scales = cluster.map_shared_rank(
                storage.hidden[slot].scales, owner);
            std::uint64_t local = 0U;
            for (std::uint32_t byte = role_thread;
                 byte < Contract::kValuesPerSlot;
                 byte += consumer_threads) {
                const std::uint8_t expected = static_cast<std::uint8_t>(
                    (slice * 29U + byte * 7U) & 0xffU);
                const std::uint8_t actual = remote_values[byte];
                local += actual;
                if (actual != expected) {
                    atomicAdd(violations, 1U);
                }
            }
            for (std::uint32_t byte = role_thread;
                 byte < Contract::kScalesPerSlot;
                 byte += consumer_threads) {
                const std::uint8_t expected = static_cast<std::uint8_t>(
                    (slice * 13U + byte * 3U) & 0xffU);
                const std::uint8_t actual = remote_scales[byte];
                local += actual;
                if (actual != expected) {
                    atomicAdd(violations, 1U);
                }
            }
            atomicAdd(
                reinterpret_cast<unsigned long long*>(
                    checksums + static_cast<std::size_t>(rank) *
                        Plan::kSlices + slice),
                static_cast<unsigned long long>(local));
        }

        // Publishes the current slice, completes the previous consumption and
        // makes the slot from two stages ago reusable in one operation.
        cluster.sync();
    }
}

}  // namespace stream1_sm120_nvfp4_cutlass_dsm
