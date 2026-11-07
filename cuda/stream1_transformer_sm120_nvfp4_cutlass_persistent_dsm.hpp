#pragma once

#include "stream1_transformer_sm120_nvfp4_cutlass_dsm_fused.cuh"

#include <cuda/barrier>

namespace stream1_sm120_nvfp4_cutlass_dsm {

using PersistentClusterShape = Shape<_1, _2, _1>;

// A stock SM120 block-scaled FF1 collective consumes a complete 384-thread
// CTA.  It therefore cannot be placed in the 128-thread producer warpgroup of
// PersistentWorkTilePlan without replacing the measured CUTLASS mainloop.
// The production fusion candidate keeps both measured collectives intact and
// separates their ownership at CTA granularity:
//
//   rank 0/1: FF1 for the two adjacent H128 slices (ping/pong producers)
//   rank 2/3: FF2 for the two adjacent D128 output tiles (consumers)
//
// Each consumer reads both producer slots.  This is deliberately a 2x2
// physical cluster with logical-singleton CUTLASS collectives; SM120 does not
// provide the programmatic TMA multicast used by SM100 schedules, so A/SFA is
// published through explicit DSM while B/SFB remains local TMA.
using RoleSeparatedClusterShape = Shape<_2, _2, _1>;

struct RoleSeparatedCtaPlan {
    static constexpr std::uint32_t kProducerCtas = 2U;
    static constexpr std::uint32_t kConsumerCtas = 2U;
    static constexpr std::uint32_t kClusterCtas =
        kProducerCtas + kConsumerCtas;
    static constexpr std::uint32_t kThreadsPerCta = 384U;

    CUTLASS_HOST_DEVICE static constexpr bool is_producer(
        std::uint32_t rank) {
        return rank < kProducerCtas;
    }

    CUTLASS_HOST_DEVICE static constexpr bool is_consumer(
        std::uint32_t rank) {
        return rank >= kProducerCtas && rank < kClusterCtas;
    }

    CUTLASS_HOST_DEVICE static constexpr std::uint32_t producer_slot(
        std::uint32_t rank) {
        return rank;
    }

    CUTLASS_HOST_DEVICE static constexpr std::uint32_t consumer_n_tile(
        std::uint32_t rank) {
        return rank - kProducerCtas;
    }

    CUTLASS_HOST_DEVICE static constexpr std::uint32_t slice_for_pair(
        std::uint32_t pair, std::uint32_t producer_rank) {
        return pair * kProducerCtas + producer_slot(producer_rank);
    }
};

static_assert(cute::size(RoleSeparatedClusterShape{}) ==
              RoleSeparatedCtaPlan::kClusterCtas);
static_assert(RoleSeparatedCtaPlan::slice_for_pair(3U, 1U) == 7U);

// Static, zero-atomic work ownership for the production topology.  One
// physical two-CTA cluster owns two interleaved persistent work streams.  WG0
// is the only producer; WG1 and WG2 consume different M tiles and advance by
// the total number of active MMA work streams.  The formula is deliberately
// independent of barrier state so a tile can never be claimed twice.
struct PersistentWorkTilePlan {
    static constexpr std::uint32_t kProducerWarpGroup = 0U;
    static constexpr std::uint32_t kFirstMmaWarpGroup = 1U;
    static constexpr std::uint32_t kMmaWarpGroups = 2U;
    static constexpr std::uint32_t kThreadsPerWarpGroup = 128U;
    static constexpr std::uint32_t kThreads =
        (1U + kMmaWarpGroups) * kThreadsPerWarpGroup;

    CUTLASS_HOST_DEVICE static constexpr std::uint32_t cluster_index(
        std::uint32_t physical_block) {
        return physical_block / 2U;
    }

    CUTLASS_HOST_DEVICE static constexpr std::uint32_t first_tile(
        std::uint32_t cluster, std::uint32_t mma_slot) {
        return cluster * kMmaWarpGroups + mma_slot;
    }

    CUTLASS_HOST_DEVICE static constexpr std::uint32_t next_tile(
        std::uint32_t tile, std::uint32_t cluster_count) {
        return tile + cluster_count * kMmaWarpGroups;
    }
};

static_assert(PersistentWorkTilePlan::kThreads == 384U);

using PersistentBlockBarrier = cuda::barrier<cuda::thread_scope_block>;

// Two physical slots are sufficient because the tile scheduler permanently
// binds one slot to each MMA warpgroup.  Ready/free generations let WG0 fill
// the next FF1 result while the other consumer is executing FF2, without an
// atomic work queue and without allocating one full hidden buffer per logical
// transaction.
struct PersistentFf1Ff2Ring {
    static constexpr std::uint32_t kSlots =
        PersistentWorkTilePlan::kMmaWarpGroups;
    DsmASlot hidden[kSlots];
    PersistentBlockBarrier ready[kSlots];
    PersistentBlockBarrier free[kSlots];
    std::uint32_t generation[kSlots];

    CUTLASS_HOST_DEVICE static constexpr std::uint32_t slot_for_tile(
        std::uint32_t tile) {
        return tile % kSlots;
    }

    CUTLASS_HOST_DEVICE static constexpr std::uint32_t generation_for_tile(
        std::uint32_t tile, std::uint32_t cluster_count) {
        return tile / (cluster_count * kSlots);
    }
};

static_assert(PersistentFf1Ff2Ring::kSlots == 2U);

// The FF1 epilogue writes the exact physical layout consumed by FF2.  Keeping
// the views here makes the layout contract shared by both collectives and
// prevents a logical-row-major hidden tensor or a post-epilogue repack from
// entering the hot path.
struct Ff1HiddenRingWriter {
    static constexpr bool kWritesCutlassNativeLayout = true;
    static constexpr bool kWritesValuesAndScalesDirectly = true;
    static constexpr bool kMaterializesGlobalHidden = false;
    static constexpr bool kConsumesFourLanePairs = true;
    static constexpr std::uint32_t kSlotBytes =
        Contract::kValuesPerSlot + Contract::kScalesPerSlot;

    CUTLASS_DEVICE static auto values(DsmASlot& slot) {
        return make_dsm_a_stage_view(slot.values);
    }

    CUTLASS_DEVICE static auto scales(DsmASlot& slot) {
        return make_dsm_sfa_stage_view(slot.scales);
    }

    CUTLASS_DEVICE static void store_group(
        DsmASlot& slot,
        std::uint32_t row,
        std::uint32_t group,
        const std::uint8_t (&raw)[Contract::kScaleVector],
        std::uint8_t scale_raw) {
        const std::uint32_t first_column = group * Contract::kScaleVector;
        CUTLASS_PRAGMA_UNROLL
        for (std::uint32_t i = 0; i < Contract::kScaleVector; ++i) {
            const std::uint32_t nibble = static_cast<std::uint32_t>(
                Ff2PhysicalStageLayoutA{}(row, first_column + i));
            const std::uint32_t byte = nibble >> 1U;
            const std::uint8_t value = raw[i] & 0x0fU;
            if ((nibble & 1U) == 0U) {
                slot.values[byte] = static_cast<std::uint8_t>(
                    (slot.values[byte] & 0xf0U) | value);
            } else {
                slot.values[byte] = static_cast<std::uint8_t>(
                    (slot.values[byte] & 0x0fU) | (value << 4U));
            }
        }
        const std::uint32_t scale = static_cast<std::uint32_t>(
            Ff2StageLayoutSFA{}(row, first_column));
        slot.scales[scale] = scale_raw;
    }

    CUTLASS_DEVICE static void store_group_from_lane_pairs(
        DsmASlot& slot, std::uint32_t row, std::uint32_t group,
        std::uint8_t pair_lo, std::uint8_t pair_hi,
        std::uint8_t scale_raw) {
        const std::uint32_t lane = threadIdx.x & 3U;
        const std::uint32_t first = group * Contract::kScaleVector;
        store_pair(slot, row, first + lane * 2U, pair_lo);
        store_pair(slot, row, first + 8U + lane * 2U, pair_hi);
        if (lane == 0U) {
            const std::uint32_t scale = static_cast<std::uint32_t>(
                Ff2StageLayoutSFA{}(row, first));
            slot.scales[scale] = scale_raw;
        }
    }

    CUTLASS_DEVICE static void store_pair(
        DsmASlot& slot, std::uint32_t row, std::uint32_t column,
        std::uint8_t packed) {
        const std::uint32_t low = static_cast<std::uint32_t>(
            Ff2PhysicalStageLayoutA{}(row, column));
        const std::uint32_t high = static_cast<std::uint32_t>(
            Ff2PhysicalStageLayoutA{}(row, column + 1U));
        if ((low & 1U) == 0U && high == low + 1U) {
            slot.values[low >> 1U] = packed;
        } else {
            const std::uint32_t low_byte = low >> 1U;
            const std::uint32_t high_byte = high >> 1U;
            slot.values[low_byte] = static_cast<std::uint8_t>(
                (slot.values[low_byte] & 0xf0U) | (packed & 0x0fU));
            slot.values[high_byte] = static_cast<std::uint8_t>(
                (slot.values[high_byte] & 0x0fU) | (packed & 0xf0U));
        }
    }
};

// SM120's block-scaled builder intentionally rejects programmatic TMA
// multicast.  Build the unchanged fast collective with its supported 1-CTA
// shape and promote only the wrapper/kernel launch to a 2-CTA cluster.  The
// promoted operand is DSM, not TMA multicast.
using PersistentFf2Epilogue = Ff2Epilogue;
using PersistentBaseFf2Mainloop = Ff2Mainloop;
using PersistentBaseFf2TensorStorage =
    typename PersistentBaseFf2Mainloop::TensorStorage;

// Drop-in replacement for the stock mainloop producer.  The consumer MMA,
// two-MMA-warpgroup ping-pong kernel and persistent tile scheduler are inherited
// unchanged.  FF1 owns rank 0 and writes the typed hidden ring before this load
// is entered.  Only rank 0 publishes A/SFA to both CTAs; each CTA continues to
// issue its own B/SFB TMA into the same CUTLASS transaction barrier.
struct PersistentBulkDsmFf2Collective : PersistentBaseFf2Mainloop {
    using Base = PersistentBaseFf2Mainloop;
    using MainloopPipeline = typename Base::MainloopPipeline;
    using PipelineState = typename Base::PipelineState;
    using Params = typename Base::Params;
    using ClusterShape = PersistentClusterShape;

    // Qualify the parent explicitly.  Once this nested class name is injected,
    // `Base::TensorStorage` is diagnosed by optimized nvcc passes as the
    // incomplete class currently being declared rather than the parent
    // collective storage.
    struct TensorStorage : PersistentBaseFf2TensorStorage {
        PersistentFf1Ff2Ring ff1_ff2;
    };

    static constexpr bool kUsesStockPersistentScheduler = true;
    static constexpr std::uint32_t kMmaWarpGroups =
        PersistentWorkTilePlan::kMmaWarpGroups;
    static constexpr std::uint32_t kThreads =
        PersistentWorkTilePlan::kThreads;
    static constexpr bool kLoadsGlobalAOnlyOnOwner = true;
    static constexpr bool kPublishesAWithBulkDsm = true;
    static constexpr bool kUsesSameTransactionBarrier = true;
    static constexpr bool kLoadsGlobalBPerCta = true;
    static constexpr bool kHasNoGlobalHiddenTensor = true;
    static constexpr std::uint32_t kClusterCtas = 2U;
    static constexpr std::uint32_t kTransactionBytes = Base::TmaTransactionBytes;

    template <class TensorA, class TensorB, class TensorSFA, class TensorSFB,
              class KTileIterator, class BlockCoord>
    CUTLASS_DEVICE void load(
        Params const& params,
        MainloopPipeline pipeline,
        PipelineState write_state,
        cute::tuple<TensorA, TensorB, TensorSFA, TensorSFB> const& inputs,
        BlockCoord const& block_coord,
        KTileIterator k_tile_iter,
        int k_tile_count,
        int,
        std::uint32_t,
        TensorStorage& storage) {
        if (cute::elect_one_sync()) {
            auto [gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl] = inputs;
            (void)gA_mkl;
            (void)gSFA_mkl;
            auto [m_coord, n_coord, ignored_k, l_coord] = block_coord;
            (void)m_coord;
            (void)ignored_k;

            Tensor sB = make_tensor(make_smem_ptr(storage.smem_B.begin()),
                                    typename Base::SmemLayoutB{});
            Tensor sSFB = make_tensor(make_smem_ptr(storage.smem_SFB.begin()),
                                      typename Base::SmemLayoutSFB{});
            Tensor gB = gB_nkl(_, _, n_coord, _, l_coord);
            auto broadcast_n = make_layout(
                make_shape(Int<size<1>(typename Base::TileShapeSFB{}) /
                                   size<1>(TileShape{})>{},
                           Int<cute::numeric_limits<int>::max()>{}),
                make_stride(_0{}, size<1>(typename Base::TileShapeSFB{}) /
                                      size<1>(TileShape{})));
            Tensor gSFB = gSFB_nkl(_, _, broadcast_n(n_coord), _, l_coord);
            auto tma_b = params.tma_load_b.get_slice(0);
            auto tma_sfb = params.tma_load_sfb.get_slice(0);
            Tensor tBgB = tma_b.partition_S(gB);
            Tensor tBsB = tma_b.partition_D(sB);
            Tensor tBgSFB = tma_sfb.partition_S(gSFB);
            Tensor tBsSFB = tma_sfb.partition_D(sSFB);

            namespace cg = cooperative_groups;
            auto cluster = cg::this_cluster();
            const std::uint32_t rank = cluster.block_rank();
            constexpr std::uint32_t values_per_slot =
                Contract::kValuesPerSlot;
            constexpr std::uint32_t scales_per_slot =
                Contract::kScalesPerSlot;
            std::uint32_t slice = 0U;
            for (; k_tile_count > 0; --k_tile_count, ++slice) {
                pipeline.producer_acquire(write_state);
                using Barrier = typename MainloopPipeline::ProducerBarrierType;
                Barrier* barrier = pipeline.producer_get_barrier(write_state);
                const int stage = write_state.index();
                copy(params.tma_load_b.with(*barrier),
                     tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, stage));
                copy(params.tma_load_sfb.with(*barrier),
                     tBgSFB(_, _, _, *k_tile_iter), tBsSFB(_, _, _, stage));

                if (rank == 0U) {
                    auto* local_a = reinterpret_cast<std::uint8_t*>(&storage.smem_A) +
                        static_cast<std::size_t>(stage) * values_per_slot;
                    auto* local_sfa = reinterpret_cast<std::uint8_t*>(&storage.smem_SFA) +
                        static_cast<std::size_t>(stage) * scales_per_slot;
                    for (std::uint32_t target = 0; target < kClusterCtas; ++target) {
                        auto* target_a = cluster.map_shared_rank(local_a, target);
                        auto* target_sfa = cluster.map_shared_rank(local_sfa, target);
                        auto* target_barrier = cluster.map_shared_rank(barrier, target);
                        cuda::ptx::cp_async_bulk(
                            cuda::ptx::space_cluster, cuda::ptx::space_shared,
                            target_a, storage.ff1_ff2.hidden[
                                slice % Contract::kRingSlots].values,
                            values_per_slot, target_barrier);
                        cuda::ptx::cp_async_bulk(
                            cuda::ptx::space_cluster, cuda::ptx::space_shared,
                            target_sfa, storage.ff1_ff2.hidden[
                                slice % Contract::kRingSlots].scales,
                            scales_per_slot, target_barrier);
                    }
                }
                ++k_tile_iter;
                ++write_state;
            }
        }
        __syncwarp();
    }
};

static_assert(PersistentBulkDsmFf2Collective::kTransactionBytes == 18432U);

using PersistentFf2Kernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, PersistentBulkDsmFf2Collective,
    PersistentFf2Epilogue, void>;

}  // namespace stream1_sm120_nvfp4_cutlass_dsm
