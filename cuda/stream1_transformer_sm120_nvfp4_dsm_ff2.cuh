#pragma once

#include "stream1_transformer_sm120_nvfp4_cutlass_dsm_fused.cuh"

#include <cuda/__ptx/instructions/cp_async_bulk.h>
#include <cuda/__ptx/ptx_dot_variants.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/pipeline.hpp>

#include <cstdint>

namespace stream1_sm120_nvfp4_dsm_ff2 {

using namespace cute;
using namespace stream1_sm120_nvfp4_cutlass_dsm;

struct BOnlyCollective {
    using Base = Ff2Mainloop;
    using MainloopPipeline = cutlass::PipelineTmaAsync<1>;
    using PipelineState = cutlass::PipelineState<1>;
    using StageLayoutB = decltype(
        typename Base::SmemLayoutB{}(_, _, Int<0>{}));
    using StageLayoutSFB = decltype(
        typename Base::SmemLayoutSFB{}(_, _, Int<0>{}));
    using StorageB = typename Base::SmemAllocTypeB;

    static constexpr bool kLoadsGlobalA = false;
    static constexpr bool kLoadsGlobalSFA = false;
    static constexpr bool kLoadsGlobalB = true;
    static constexpr bool kLoadsGlobalSFB = true;
    static constexpr std::uint32_t kTransactionBytes =
        Base::TmaTransactionBytesNK;
    static constexpr std::uint32_t kBBytes = cutlass::bits_to_bytes(
        cosize(StageLayoutB{}) * cute::sizeof_bits_v<StorageB>);
    static constexpr std::uint32_t kSfbBytes = cutlass::bits_to_bytes(
        cosize(StageLayoutSFB{}) * cute::sizeof_bits_v<ElementScale>);
    static constexpr std::uint32_t kThreads = 384U;

    struct alignas(1024) SharedStorage {
        alignas(1024) cute::ArrayEngine<
            StorageB, cute::cosize_v<StageLayoutB>> smem_B;
        alignas(16) cute::ArrayEngine<
            ElementScale, cute::cosize_v<StageLayoutSFB>> smem_SFB;
        alignas(16) typename MainloopPipeline::SharedStorage pipeline;
    };

    template <class LoadInputs>
    CUTLASS_DEVICE static void issue_load(
        typename Base::Params const& params,
        MainloopPipeline& pipeline,
        PipelineState write_state,
        LoadInputs const& load_inputs,
        SharedStorage& storage) {
        auto [gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl] = load_inputs;
        (void)gA_mkl;
        (void)gSFA_mkl;
        auto block_tma_b = params.tma_load_b.get_slice(0);
        auto block_tma_sfb = params.tma_load_sfb.get_slice(0);
        Tensor gB = gB_nkl(_, _, 0, _, 0);
        auto broadcast_n = make_layout(
            make_shape(
                Int<size<1>(typename Base::TileShapeSFB{}) /
                    size<1>(TileShape{})>{},
                Int<cute::numeric_limits<int>::max()>{}),
            make_stride(
                _0{},
                size<1>(typename Base::TileShapeSFB{}) /
                    size<1>(TileShape{})));
        Tensor gSFB = gSFB_nkl(_, _, broadcast_n(0), _, 0);
        Tensor sB = make_tensor(
            make_smem_ptr(storage.smem_B.begin()), StageLayoutB{});
        Tensor sSFB = make_tensor(
            make_smem_ptr(storage.smem_SFB.begin()), StageLayoutSFB{});
        Tensor tBgB = block_tma_b.partition_S(gB);
        Tensor tBsB = block_tma_b.partition_D(sB);
        Tensor tBgSFB = block_tma_sfb.partition_S(gSFB);
        Tensor tBsSFB = block_tma_sfb.partition_D(sSFB);
        using Barrier = typename MainloopPipeline::ProducerBarrierType;
        Barrier* barrier = pipeline.producer_get_barrier(write_state);
        copy(params.tma_load_b.with(*barrier),
             tBgB(_, _, _, 0), tBsB);
        copy(params.tma_load_sfb.with(*barrier),
             tBgSFB(_, _, _, 0), tBsSFB);
    }

    template <class Accumulator>
    CUTLASS_DEVICE static void mma_local_a_stage(
        std::uint8_t* local_a,
        std::uint8_t* local_sfa,
        SharedStorage& storage,
        int thread_idx,
        Accumulator& accum,
        bool reset_accumulator = true) {
        static_assert(is_rmem<Accumulator>::value);
        if (reset_accumulator) {
            clear(accum);
        }
        Tensor sA = make_dsm_a_stage_view(local_a);
        Tensor sSFA = make_dsm_sfa_stage_view(local_sfa);
        Tensor sB = make_tensor(
            make_smem_ptr(storage.smem_B.begin()), StageLayoutB{});
        Tensor sSFB = make_tensor(
            make_smem_ptr(storage.smem_SFB.begin()), StageLayoutSFB{});

        typename Base::TiledMma tiled_mma;
        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor tCrA = thread_mma.partition_fragment_A(sA);
        Tensor tCrB = thread_mma.partition_fragment_B(sB);
        Base mainloop;
        Tensor tCrSFA = mainloop.partition_fragment_SFA(sSFA, thread_mma);
        Tensor tCrSFB = mainloop.partition_fragment_SFB(sSFB, thread_mma);

        auto copy_a = make_tiled_copy_A(
            typename Base::SmemCopyAtomA{}, tiled_mma);
        auto copy_b = make_tiled_copy_B(
            typename Base::SmemCopyAtomB{}, tiled_mma);
        auto thread_copy_a = copy_a.get_thread_slice(thread_idx);
        auto thread_copy_b = copy_b.get_thread_slice(thread_idx);
        Tensor tCsA = thread_copy_a.partition_S(
            as_position_independent_swizzle_tensor(sA));
        Tensor tCsB = thread_copy_b.partition_S(
            as_position_independent_swizzle_tensor(sB));
        Tensor tCrAView = thread_copy_a.retile_D(tCrA);
        Tensor tCrBView = thread_copy_b.retile_D(tCrB);

        const auto mma_shape = tile_shape(tiled_mma);
        auto copy_sfa = make_tiled_copy_impl(
            typename Base::SmemCopyAtomSFA{},
            mainloop.get_layoutSFA_TV(tiled_mma),
            make_shape(size<0>(mma_shape), size<2>(mma_shape)));
        auto copy_sfb = make_tiled_copy_impl(
            typename Base::SmemCopyAtomSFB{},
            mainloop.get_layoutSFB_TV(tiled_mma),
            make_shape(size<1>(mma_shape), size<2>(mma_shape)));
        auto thread_copy_sfa = copy_sfa.get_thread_slice(thread_idx);
        auto thread_copy_sfb = copy_sfb.get_thread_slice(thread_idx);
        Tensor tCsSFA = thread_copy_sfa.partition_S(
            as_position_independent_swizzle_tensor(sSFA));
        Tensor tCsSFB = thread_copy_sfb.partition_S(
            as_position_independent_swizzle_tensor(sSFB));
        Tensor tCrSFAView = thread_copy_sfa.retile_D(tCrSFA);
        Tensor tCrSFBView = thread_copy_sfb.retile_D(tCrSFB);

        constexpr int kBlocks = size<2>(decltype(tCrA){});
        for_each(make_int_sequence<kBlocks>{}, [&](auto k_block) {
            copy(copy_a, tCsA(_, _, k_block),
                 tCrAView(_, _, k_block));
            copy(copy_b, tCsB(_, _, k_block),
                 tCrBView(_, _, k_block));
            using MmaOp = typename Base::TiledMma::MMA_Op;
            fp4_shift_A(MmaOp{}, tCrAView(_, _, k_block));
            fp4_shift_B(MmaOp{}, tCrBView(_, _, k_block));
            copy(tCsSFA(_, _, k_block),
                 tCrSFAView(_, _, k_block));
            copy(tCsSFB(_, _, k_block),
                 tCrSFBView(_, _, k_block));
            cute::gemm(
                tiled_mma,
                make_zip_tensor(
                    tCrA(_, _, k_block), tCrSFA(_, _, k_block)),
                make_zip_tensor(
                    tCrB(_, _, k_block), tCrSFB(_, _, k_block)),
                accum);
        });
    }
};

static_assert(BOnlyCollective::kTransactionBytes ==
              BOnlyCollective::kBBytes + BOnlyCollective::kSfbBytes);
static_assert(BOnlyCollective::kBBytes == 8192U);
static_assert(BOnlyCollective::kSfbBytes == 1024U);

__global__ __cluster_dims__(1, 1, 1) void b_sfb_tma_contract_kernel(
    CUTE_GRID_CONSTANT typename Ff2Mainloop::Params const params,
    unsigned long long* checksum) {
    extern __shared__ __align__(1024) unsigned char shared_bytes[];
    auto& storage = *reinterpret_cast<BOnlyCollective::SharedStorage*>(
        shared_bytes);
    using Pipeline = BOnlyCollective::MainloopPipeline;
    using PipelineState = BOnlyCollective::PipelineState;
    const std::uint32_t warp_group = threadIdx.x / 128U;
    const std::uint32_t wg_thread = threadIdx.x % 128U;

    typename Pipeline::Params pipeline_params;
    pipeline_params.transaction_bytes = BOnlyCollective::kTransactionBytes;
    pipeline_params.num_consumers = 128U;
    pipeline_params.is_leader = wg_thread == 0U;
    pipeline_params.role = warp_group == 0U
        ? Pipeline::ThreadCategory::Producer
        : (warp_group == 1U
            ? Pipeline::ThreadCategory::Consumer
            : Pipeline::ThreadCategory::NonParticipant);
    Pipeline pipeline(
        storage.pipeline, pipeline_params, Shape<_1, _1, _1>{});
    __syncthreads();
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();

    constexpr auto problem_shape = make_shape(128, 128, 128, 1);
    if (warp_group == 0U && wg_thread == 0U) {
        Ff2Mainloop mainloop;
        const auto load_inputs = mainloop.load_init(problem_shape, params);
        PipelineState write_state =
            cutlass::make_producer_start_state<Pipeline>();
        pipeline.producer_acquire(write_state);
        BOnlyCollective::issue_load(
            params, pipeline, write_state, load_inputs, storage);
        ++write_state;
        pipeline.producer_tail(write_state);
    } else if (warp_group == 1U) {
        PipelineState read_state;
        pipeline.consumer_wait(read_state);
        const auto* b = reinterpret_cast<const std::uint8_t*>(
            &storage.smem_B);
        const auto* sfb = reinterpret_cast<const std::uint8_t*>(
            &storage.smem_SFB);
        unsigned long long local = 0U;
        for (std::uint32_t i = wg_thread;
             i < BOnlyCollective::kBBytes; i += 128U) {
            local += b[i];
        }
        for (std::uint32_t i = wg_thread;
             i < BOnlyCollective::kSfbBytes; i += 128U) {
            local += sfb[i];
        }
        atomicAdd(checksum, local);
        pipeline.consumer_release(read_state);
    }
}

struct alignas(1024) RemoteAMmaSharedStorage {
    DsmASlot dsm_a;
    alignas(8) std::uint64_t dsm_copy_barrier;
    alignas(8) std::uint64_t slice_ready_barrier;
    alignas(8) std::uint64_t slice_done_barrier;
    BOnlyCollective::SharedStorage b_only;
};

CUTLASS_DEVICE void elected_cluster_barrier(
    std::uint64_t* local_barrier,
    std::uint32_t peer_rank,
    std::uint32_t phase) {
    if (threadIdx.x == 0U) {
        cutlass::arch::ClusterBarrier::arrive(local_barrier);
        cutlass::arch::ClusterBarrier::arrive(
            local_barrier, peer_rank, 1U);
        cutlass::arch::ClusterBarrier::wait(local_barrier, phase);
    }
    __syncthreads();
}

__global__ __cluster_dims__(2, 1, 1) void dsm_to_local_copy_contract_kernel(
    unsigned long long* rank_checksums) {
    namespace cg = cooperative_groups;
    extern __shared__ __align__(1024) unsigned char shared_bytes[];
    auto& local = *reinterpret_cast<DsmASlot*>(shared_bytes);
    const auto cluster = cg::this_cluster();
    const std::uint32_t rank = cluster.block_rank();
    constexpr std::uint8_t value_pattern = 0x11U;
    constexpr std::uint8_t scale_pattern = 0x7fU;
    if (rank == 0U) {
        for (std::uint32_t i = threadIdx.x;
             i < Contract::kValuesPerSlot; i += blockDim.x) {
            local.values[i] = value_pattern;
        }
        for (std::uint32_t i = threadIdx.x;
             i < Contract::kScalesPerSlot; i += blockDim.x) {
            local.scales[i] = scale_pattern;
        }
    }
    cluster.sync();
    if (rank != 0U) {
        const auto* remote_values = cluster.map_shared_rank(local.values, 0);
        const auto* remote_scales = cluster.map_shared_rank(local.scales, 0);
        for (std::uint32_t i = threadIdx.x;
             i < Contract::kValuesPerSlot; i += blockDim.x) {
            local.values[i] = remote_values[i];
        }
        for (std::uint32_t i = threadIdx.x;
             i < Contract::kScalesPerSlot; i += blockDim.x) {
            local.scales[i] = remote_scales[i];
        }
    }
    cluster.sync();
    unsigned long long sum = 0U;
    for (std::uint32_t i = threadIdx.x;
         i < Contract::kValuesPerSlot; i += blockDim.x) {
        sum += local.values[i];
    }
    for (std::uint32_t i = threadIdx.x;
         i < Contract::kScalesPerSlot; i += blockDim.x) {
        sum += local.scales[i];
    }
    atomicAdd(rank_checksums + rank, sum);
}

// Control kernel: proves the customized FF2 MMA path independently from DSM.
// Keeping this executable gate separate prevents a DSM synchronization fault
// from being misdiagnosed as an NVFP4/CUTLASS MMA fault.
__global__ __cluster_dims__(1, 1, 1) void local_a_mma_contract_kernel(
    CUTE_GRID_CONSTANT typename Ff2Mainloop::Params const params,
    float* checksum) {
    extern __shared__ __align__(1024) unsigned char shared_bytes[];
    auto& storage = *reinterpret_cast<RemoteAMmaSharedStorage*>(shared_bytes);
    const std::uint32_t warp_group = threadIdx.x / 128U;
    const std::uint32_t wg_thread = threadIdx.x % 128U;
    constexpr std::uint8_t value_pattern = 0x11U;
    const std::uint8_t scale_pattern = ElementScale(1.0F).raw();

    for (std::uint32_t i = threadIdx.x;
         i < Contract::kValuesPerSlot; i += blockDim.x) {
        storage.dsm_a.values[i] = value_pattern;
    }
    for (std::uint32_t i = threadIdx.x;
         i < Contract::kScalesPerSlot; i += blockDim.x) {
        storage.dsm_a.scales[i] = scale_pattern;
    }

    using Pipeline = BOnlyCollective::MainloopPipeline;
    using PipelineState = BOnlyCollective::PipelineState;
    typename Pipeline::Params pipeline_params;
    pipeline_params.transaction_bytes = BOnlyCollective::kTransactionBytes;
    pipeline_params.num_consumers = 128U;
    pipeline_params.is_leader = wg_thread == 0U;
    pipeline_params.role = warp_group == 0U
        ? Pipeline::ThreadCategory::Producer
        : (warp_group == 1U
            ? Pipeline::ThreadCategory::Consumer
            : Pipeline::ThreadCategory::NonParticipant);
    Pipeline pipeline(
        storage.b_only.pipeline, pipeline_params, Shape<_1, _1, _1>{});
    __syncthreads();

    constexpr auto problem_shape = make_shape(128, 128, 128, 1);
    if (warp_group == 0U && wg_thread == 0U) {
        Ff2Mainloop mainloop;
        const auto load_inputs = mainloop.load_init(problem_shape, params);
        PipelineState write_state =
            cutlass::make_producer_start_state<Pipeline>();
        pipeline.producer_acquire(write_state);
        BOnlyCollective::issue_load(
            params, pipeline, write_state, load_inputs, storage.b_only);
        ++write_state;
        pipeline.producer_tail(write_state);
    } else if (warp_group == 1U) {
        PipelineState read_state;
        pipeline.consumer_wait(read_state);
        typename Ff2Mainloop::TiledMma tiled_mma;
        auto accum = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
        BOnlyCollective::mma_local_a_stage(
            storage.dsm_a.values, storage.dsm_a.scales, storage.b_only,
            static_cast<int>(wg_thread), accum);
        float local = 0.0F;
        for (int i = 0; i < size(accum); ++i) {
            local += accum(i) < 0.0F ? -accum(i) : accum(i);
        }
        atomicAdd(checksum, local);
        pipeline.consumer_release(read_state);
    }
}

__global__ __cluster_dims__(2, 1, 1) void remote_a_mma_contract_kernel(
    CUTE_GRID_CONSTANT typename Ff2Mainloop::Params const params,
    float* rank_checksums,
    bool use_remote_handoff,
    bool use_tma) {
    namespace cg = cooperative_groups;
    extern __shared__ __align__(1024) unsigned char shared_bytes[];
    auto& storage = *reinterpret_cast<RemoteAMmaSharedStorage*>(shared_bytes);
    const auto cluster = cg::this_cluster();
    const std::uint32_t rank = cluster.block_rank();
    const std::uint32_t warp_group = threadIdx.x / 128U;
    const std::uint32_t wg_thread = threadIdx.x % 128U;
    constexpr std::uint8_t value_pattern = 0x11U;
    const std::uint8_t scale_pattern = ElementScale(1.0F).raw();

    if (rank == 0U || !use_remote_handoff) {
        for (std::uint32_t i = threadIdx.x;
             i < Contract::kValuesPerSlot; i += blockDim.x) {
            storage.dsm_a.values[i] = value_pattern;
        }
        for (std::uint32_t i = threadIdx.x;
             i < Contract::kScalesPerSlot; i += blockDim.x) {
            storage.dsm_a.scales[i] = scale_pattern;
        }
    }
    cluster.sync();

    // CUTLASS's SM120 warp-MMA copy atoms consume CTA-local shared memory.
    // Move only the remote DSM payload into the identical local byte layout;
    // B/SFB remain on their independent TMA path below.
    if (use_remote_handoff && rank != 0U) {
        const auto* remote_values = cluster.map_shared_rank(
            storage.dsm_a.values, 0);
        const auto* remote_scales = cluster.map_shared_rank(
            storage.dsm_a.scales, 0);
        for (std::uint32_t i = threadIdx.x;
             i < Contract::kValuesPerSlot; i += blockDim.x) {
            storage.dsm_a.values[i] = remote_values[i];
        }
        for (std::uint32_t i = threadIdx.x;
             i < Contract::kScalesPerSlot; i += blockDim.x) {
            storage.dsm_a.scales[i] = remote_scales[i];
        }
    }
    cluster.sync();

    using Pipeline = BOnlyCollective::MainloopPipeline;
    using PipelineState = BOnlyCollective::PipelineState;
    typename Pipeline::Params pipeline_params;
    pipeline_params.transaction_bytes = BOnlyCollective::kTransactionBytes;
    pipeline_params.num_consumers = 128U;
    pipeline_params.is_leader = wg_thread == 0U;
    pipeline_params.role = warp_group == 0U
        ? Pipeline::ThreadCategory::Producer
        : (warp_group == 1U
            ? Pipeline::ThreadCategory::Consumer
            : Pipeline::ThreadCategory::NonParticipant);
    Pipeline pipeline(
        storage.b_only.pipeline, pipeline_params, Shape<_2, _1, _1>{});
    __syncthreads();
    // PipelineTmaAsync initializes cluster-visible mbarriers.  Every CTA must
    // finish that initialization before any producer can acquire a stage.
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();

    if (!use_tma) {
        auto* b_bytes = reinterpret_cast<std::uint8_t*>(
            &storage.b_only.smem_B);
        auto* sfb_bytes = reinterpret_cast<std::uint8_t*>(
            &storage.b_only.smem_SFB);
        for (std::uint32_t i = threadIdx.x;
             i < BOnlyCollective::kBBytes; i += blockDim.x) {
            b_bytes[i] = value_pattern;
        }
        for (std::uint32_t i = threadIdx.x;
             i < BOnlyCollective::kSfbBytes; i += blockDim.x) {
            sfb_bytes[i] = scale_pattern;
        }
        __syncthreads();
        if (warp_group == 1U) {
            typename Ff2Mainloop::TiledMma tiled_mma;
            auto accum = partition_fragment_C(
                tiled_mma, take<0, 2>(TileShape{}));
            BOnlyCollective::mma_local_a_stage(
                storage.dsm_a.values, storage.dsm_a.scales, storage.b_only,
                static_cast<int>(wg_thread), accum);
            float local = 0.0F;
            for (int i = 0; i < size(accum); ++i) {
                local += accum(i) < 0.0F ? -accum(i) : accum(i);
            }
            atomicAdd(rank_checksums + rank, local);
        }
        return;
    }

    constexpr auto problem_shape = make_shape(128, 128, 128, 1);
    if (warp_group == 0U && wg_thread == 0U) {
        Ff2Mainloop mainloop;
        const auto load_inputs = mainloop.load_init(problem_shape, params);
        PipelineState write_state =
            cutlass::make_producer_start_state<Pipeline>();
        pipeline.producer_acquire(write_state);
        BOnlyCollective::issue_load(
            params, pipeline, write_state, load_inputs, storage.b_only);
        ++write_state;
        pipeline.producer_tail(write_state);
    } else if (warp_group == 1U) {
        PipelineState read_state;
        pipeline.consumer_wait(read_state);
        typename Ff2Mainloop::TiledMma tiled_mma;
        auto accum = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
        BOnlyCollective::mma_local_a_stage(
            storage.dsm_a.values, storage.dsm_a.scales, storage.b_only,
            static_cast<int>(wg_thread), accum);
        float local = 0.0F;
        for (int i = 0; i < size(accum); ++i) {
            local += accum(i) < 0.0F ? -accum(i) : accum(i);
        }
        atomicAdd(rank_checksums + rank, local);
        pipeline.consumer_release(read_state);
    }
}

// Full FF2-K contract: one persistent cluster/pipeline consumes all eight
// K128 hidden slices.  Slice ownership alternates between CTAs, so each rank
// exercises both local production and remote DSM handoff without materializing
// the 128x1024 hidden tile in global memory.
__global__ __launch_bounds__(256, 1) __cluster_dims__(2, 1, 1)
void remote_a_mma_multislice_contract_kernel(
    CUTE_GRID_CONSTANT typename Ff2Mainloop::Params const params,
    float* rank_checksums,
    bool use_remote_handoff) {
    namespace cg = cooperative_groups;
    extern __shared__ __align__(1024) unsigned char shared_bytes[];
    auto& storage = *reinterpret_cast<RemoteAMmaSharedStorage*>(shared_bytes);
    const auto cluster = cg::this_cluster();
    const std::uint32_t rank = cluster.block_rank();
    const std::uint32_t warp_group = threadIdx.x / 128U;
    const std::uint32_t wg_thread = threadIdx.x % 128U;
    constexpr std::uint8_t value_pattern = 0x11U;
    const std::uint8_t scale_pattern = ElementScale(1.0F).raw();

    if (threadIdx.x == 0U) {
        cutlass::arch::ClusterTransactionBarrier::init(
            &storage.dsm_copy_barrier, 1U);
        cutlass::arch::ClusterBarrier::init(
            &storage.slice_ready_barrier, 2U);
        cutlass::arch::ClusterBarrier::init(
            &storage.slice_done_barrier, 2U);
    }
    __syncthreads();
    cutlass::arch::fence_barrier_init();
    cluster.sync();

    using Pipeline = BOnlyCollective::MainloopPipeline;
    using PipelineState = BOnlyCollective::PipelineState;
    typename Pipeline::Params pipeline_params;
    pipeline_params.transaction_bytes = BOnlyCollective::kTransactionBytes;
    pipeline_params.num_consumers = 128U;
    pipeline_params.is_leader = wg_thread == 0U;
    pipeline_params.role = warp_group == 0U
        ? Pipeline::ThreadCategory::Producer
        : (warp_group == 1U
            ? Pipeline::ThreadCategory::Consumer
            : Pipeline::ThreadCategory::NonParticipant);
    Pipeline pipeline(
        storage.b_only.pipeline, pipeline_params, Shape<_2, _1, _1>{});
    __syncthreads();
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();

    typename Ff2Mainloop::TiledMma tiled_mma;
    auto accum = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
    clear(accum);
    PipelineState write_state = cutlass::make_producer_start_state<Pipeline>();
    PipelineState read_state;
    constexpr auto problem_shape = make_shape(128, 128, 128, 1);
    Ff2Mainloop mainloop;
    const auto load_inputs = mainloop.load_init(problem_shape, params);

    for (std::uint32_t slice = 0; slice < Contract::kSlices; ++slice) {
        const std::uint32_t owner = use_remote_handoff ? (slice & 1U) : rank;
        const std::uint32_t peer = rank ^ 1U;
        const std::uint32_t slice_phase = slice & 1U;
        if (rank == owner) {
            for (std::uint32_t i = threadIdx.x;
                 i < Contract::kValuesPerSlot; i += blockDim.x) {
                storage.dsm_a.values[i] = value_pattern;
            }
            for (std::uint32_t i = threadIdx.x;
                 i < Contract::kScalesPerSlot; i += blockDim.x) {
                storage.dsm_a.scales[i] = scale_pattern;
            }
        }
        if (use_remote_handoff && rank != owner && threadIdx.x == 0U) {
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                &storage.dsm_copy_barrier,
                Contract::kValuesPerSlot + Contract::kScalesPerSlot);
        }
        __syncthreads();
        elected_cluster_barrier(
            &storage.slice_ready_barrier, peer, slice_phase);
        if (use_remote_handoff && rank == owner && threadIdx.x == 0U) {
            const std::uint32_t consumer = owner ^ 1U;
            const std::uint32_t values_bytes = Contract::kValuesPerSlot;
            const std::uint32_t scales_bytes = Contract::kScalesPerSlot;
            auto* remote_values = cluster.map_shared_rank(
                storage.dsm_a.values, consumer);
            auto* remote_scales = cluster.map_shared_rank(
                storage.dsm_a.scales, consumer);
            auto* remote_barrier = cluster.map_shared_rank(
                &storage.dsm_copy_barrier, consumer);
            cuda::ptx::cp_async_bulk(
                cuda::ptx::space_cluster, cuda::ptx::space_shared,
                remote_values, storage.dsm_a.values,
                values_bytes, remote_barrier);
            cuda::ptx::cp_async_bulk(
                cuda::ptx::space_cluster, cuda::ptx::space_shared,
                remote_scales, storage.dsm_a.scales,
                scales_bytes, remote_barrier);
        } else if (use_remote_handoff && rank != owner && threadIdx.x == 0U) {
            const std::uint32_t receive_phase = (slice / 2U) & 1U;
            cutlass::arch::ClusterTransactionBarrier::wait(
                &storage.dsm_copy_barrier, receive_phase);
        }
        __syncthreads();

        if (warp_group == 0U && wg_thread == 0U) {
            pipeline.producer_acquire(write_state);
            BOnlyCollective::issue_load(
                params, pipeline, write_state, load_inputs, storage.b_only);
            ++write_state;
        } else if (warp_group == 1U) {
            pipeline.consumer_wait(read_state);
            BOnlyCollective::mma_local_a_stage(
                storage.dsm_a.values, storage.dsm_a.scales, storage.b_only,
                static_cast<int>(wg_thread), accum, false);
            pipeline.consumer_release(read_state);
            ++read_state;
        }
        __syncthreads();
        elected_cluster_barrier(
            &storage.slice_done_barrier, peer, slice_phase);
    }

    if (warp_group == 0U && wg_thread == 0U) {
        pipeline.producer_tail(write_state);
    } else if (warp_group == 1U) {
        float local = 0.0F;
        for (int i = 0; i < size(accum); ++i) {
            local += accum(i) < 0.0F ? -accum(i) : accum(i);
        }
        const std::uint32_t cluster_index = blockIdx.x / 2U;
        atomicAdd(rank_checksums + cluster_index * 2U + rank, local);
    }
}

}  // namespace stream1_sm120_nvfp4_dsm_ff2
