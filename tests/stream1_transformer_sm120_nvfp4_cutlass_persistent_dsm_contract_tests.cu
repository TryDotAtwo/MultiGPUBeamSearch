#include "stream1_transformer_sm120_nvfp4_cutlass_persistent_dsm.hpp"

#include <cutlass/device_kernel.h>
#include <type_traits>
#include <iostream>
#include <limits>

using namespace stream1_sm120_nvfp4_cutlass_dsm;

static_assert(PersistentBulkDsmFf2Collective::kUsesStockPersistentScheduler);
static_assert(PersistentBulkDsmFf2Collective::kMmaWarpGroups == 2);
static_assert(PersistentBulkDsmFf2Collective::kThreads == 384);
static_assert(PersistentBulkDsmFf2Collective::kLoadsGlobalAOnlyOnOwner);
static_assert(PersistentBulkDsmFf2Collective::kPublishesAWithBulkDsm);
static_assert(PersistentBulkDsmFf2Collective::kUsesSameTransactionBarrier);
static_assert(PersistentBulkDsmFf2Collective::kLoadsGlobalBPerCta);
static_assert(PersistentBulkDsmFf2Collective::kHasNoGlobalHiddenTensor);
static_assert(PersistentBulkDsmFf2Collective::kClusterCtas == 2);
static_assert(PersistentBulkDsmFf2Collective::kTransactionBytes == 18432);

static_assert(std::is_same_v<
    typename PersistentFf2Kernel::CollectiveMainloop,
    PersistentBulkDsmFf2Collective>);
static_assert(PersistentFf2Kernel::MaxThreadsPerBlock == 384);
static_assert(Ff1HiddenRingWriter::kWritesCutlassNativeLayout);
static_assert(Ff1HiddenRingWriter::kWritesValuesAndScalesDirectly);
static_assert(Ff1HiddenRingWriter::kMaterializesGlobalHidden == false);
static_assert(Ff1HiddenRingWriter::kSlotBytes == 9216);
static_assert(PersistentWorkTilePlan::kProducerWarpGroup == 0);
static_assert(PersistentWorkTilePlan::kFirstMmaWarpGroup == 1);
static_assert(PersistentWorkTilePlan::first_tile(7, 0) == 14);
static_assert(PersistentWorkTilePlan::first_tile(7, 1) == 15);
static_assert(PersistentWorkTilePlan::next_tile(15, 94) == 203);
static_assert(PersistentFf1Ff2Ring::slot_for_tile(14) == 0);
static_assert(PersistentFf1Ff2Ring::slot_for_tile(15) == 1);
static_assert(PersistentFf1Ff2Ring::generation_for_tile(64, 16) == 2);
static_assert(RoleSeparatedCtaPlan::kClusterCtas == 4);
static_assert(RoleSeparatedCtaPlan::kThreadsPerCta == 384);
static_assert(RoleSeparatedCtaPlan::is_producer(0));
static_assert(RoleSeparatedCtaPlan::is_producer(1));
static_assert(!RoleSeparatedCtaPlan::is_producer(2));
static_assert(RoleSeparatedCtaPlan::is_consumer(2));
static_assert(RoleSeparatedCtaPlan::is_consumer(3));
static_assert(RoleSeparatedCtaPlan::producer_slot(1) == 1);
static_assert(RoleSeparatedCtaPlan::consumer_n_tile(2) == 0);
static_assert(RoleSeparatedCtaPlan::consumer_n_tile(3) == 1);
static_assert(RoleSeparatedCtaPlan::slice_for_pair(0, 0) == 0);
static_assert(RoleSeparatedCtaPlan::slice_for_pair(0, 1) == 1);
static_assert(RoleSeparatedCtaPlan::slice_for_pair(3, 0) == 6);
static_assert(RoleSeparatedCtaPlan::slice_for_pair(3, 1) == 7);

struct alignas(128) HandoffContractStorage {
    PersistentFf1Ff2Ring ring;
};

struct alignas(128) RoleSeparatedContractStorage {
    DsmASlot produced;
    DsmASlot received[RoleSeparatedCtaPlan::kProducerCtas];
    alignas(8) std::uint64_t receive[RoleSeparatedCtaPlan::kProducerCtas];
    alignas(8) std::uint64_t armed;
    alignas(8) std::uint64_t free;
};

// Hardware gate for the topology required by intact CUTLASS collectives.  It
// proves that both producer CTAs can push their native A/SFA slots to both
// consumer CTAs through one-way bulk DSM in a physical four-CTA cluster.  No
// cluster-wide rendezvous occurs in the hot loop: consumer-armed, transaction
// completion and producer-free generations are independent mbarriers.  This
// is intentionally a bounded transport oracle; the next gate replaces the
// deterministic writer with FF1 and the checksum reader with FF2.
__global__ __cluster_dims__(4, 1, 1) void role_separated_four_cta_handoff(
    std::uint32_t cluster_count, std::uint32_t pairs,
    std::uint32_t* partial_checksums, std::uint32_t* violations) {
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    namespace cg = cooperative_groups;
    const auto cluster = cg::this_cluster();
    const std::uint32_t rank = cluster.block_rank();
    const std::uint32_t cluster_id = blockIdx.x /
        RoleSeparatedCtaPlan::kClusterCtas;
    auto& storage = *reinterpret_cast<RoleSeparatedContractStorage*>(
        shared_bytes);
    constexpr std::uint32_t words_per_slot =
        sizeof(DsmASlot) / sizeof(std::uint32_t);

    if (threadIdx.x == 0U) {
        if (RoleSeparatedCtaPlan::is_producer(rank)) {
            cutlass::arch::ClusterBarrier::init(
                &storage.armed, RoleSeparatedCtaPlan::kConsumerCtas);
            cutlass::arch::ClusterBarrier::init(
                &storage.free, RoleSeparatedCtaPlan::kConsumerCtas);
        } else {
            for (std::uint32_t producer = 0;
                 producer < RoleSeparatedCtaPlan::kProducerCtas; ++producer) {
                cutlass::arch::ClusterTransactionBarrier::init(
                    &storage.receive[producer], 1U);
            }
        }
    }
    __syncthreads();
    cutlass::arch::fence_barrier_init();
    cluster.sync();

    for (std::uint32_t pair = 0; pair < pairs; ++pair) {
        if (RoleSeparatedCtaPlan::is_consumer(rank)) {
            if (threadIdx.x == 0U) {
                for (std::uint32_t producer = 0;
                     producer < RoleSeparatedCtaPlan::kProducerCtas;
                     ++producer) {
                    cutlass::arch::ClusterTransactionBarrier::
                        arrive_and_expect_tx(
                            &storage.receive[producer], sizeof(DsmASlot));
                    cutlass::arch::ClusterBarrier::arrive(
                        &storage.armed, producer, 1U);
                }
                for (std::uint32_t producer = 0;
                     producer < RoleSeparatedCtaPlan::kProducerCtas;
                     ++producer) {
                    cutlass::arch::ClusterTransactionBarrier::wait(
                        &storage.receive[producer], pair & 1U);
                }
            }
            __syncthreads();

            const std::uint32_t consumer =
                RoleSeparatedCtaPlan::consumer_n_tile(rank);
            std::uint32_t checksum = 0U;
            for (std::uint32_t producer = 0;
                 producer < RoleSeparatedCtaPlan::kProducerCtas; ++producer) {
                const auto* words = reinterpret_cast<const std::uint32_t*>(
                    &storage.received[producer]);
                for (std::uint32_t word = threadIdx.x; word < words_per_slot;
                     word += blockDim.x) {
                    const std::uint32_t expected = cluster_id * 65537U +
                        pair * 8191U + producer * 4099U + word;
                    const std::uint32_t actual = words[word];
                    checksum ^= actual + word * 17U + producer * 257U;
                    if (actual != expected) atomicAdd(violations, 1U);
                }
            }
            partial_checksums[
                ((static_cast<std::size_t>(cluster_id) * pairs + pair) *
                     RoleSeparatedCtaPlan::kConsumerCtas + consumer) *
                    RoleSeparatedCtaPlan::kThreadsPerCta + threadIdx.x] =
                checksum;
            __syncthreads();
            if (threadIdx.x == 0U) {
                for (std::uint32_t producer = 0;
                     producer < RoleSeparatedCtaPlan::kProducerCtas;
                     ++producer) {
                    cutlass::arch::ClusterBarrier::arrive(
                        &storage.free, producer, 1U);
                }
            }
        } else {
            if (threadIdx.x == 0U) {
                cutlass::arch::ClusterBarrier::wait(
                    &storage.armed, pair & 1U);
                if (pair > 0U) {
                    cutlass::arch::ClusterBarrier::wait(
                        &storage.free, (pair - 1U) & 1U);
                }
            }
            __syncthreads();
            auto* words = reinterpret_cast<std::uint32_t*>(&storage.produced);
            for (std::uint32_t word = threadIdx.x; word < words_per_slot;
                 word += blockDim.x) {
                words[word] = cluster_id * 65537U + pair * 8191U +
                    rank * 4099U + word;
            }
            __syncthreads();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (threadIdx.x == 0U) {
                constexpr std::uint32_t values_bytes =
                    Contract::kValuesPerSlot;
                constexpr std::uint32_t scales_bytes =
                    Contract::kScalesPerSlot;
                for (std::uint32_t consumer_rank =
                         RoleSeparatedCtaPlan::kProducerCtas;
                     consumer_rank < RoleSeparatedCtaPlan::kClusterCtas;
                     ++consumer_rank) {
                    auto* target_values = cluster.map_shared_rank(
                        storage.received[rank].values, consumer_rank);
                    auto* target_scales = cluster.map_shared_rank(
                        storage.received[rank].scales, consumer_rank);
                    auto* target_barrier = cluster.map_shared_rank(
                        &storage.receive[rank], consumer_rank);
                    cuda::ptx::cp_async_bulk(
                        cuda::ptx::space_cluster,
                        cuda::ptx::space_shared,
                        target_values, storage.produced.values,
                        values_bytes, target_barrier);
                    cuda::ptx::cp_async_bulk(
                        cuda::ptx::space_cluster,
                        cuda::ptx::space_shared,
                        target_scales, storage.produced.scales,
                        scales_bytes, target_barrier);
                }
            }
        }
    }
    if (RoleSeparatedCtaPlan::is_producer(rank) && threadIdx.x == 0U &&
        pairs > 0U) {
        cutlass::arch::ClusterBarrier::wait(
            &storage.free, (pairs - 1U) & 1U);
    }
    __syncthreads();
    cluster.sync();
    (void)cluster_count;
}

__global__ __cluster_dims__(2, 1, 1) void persistent_two_mma_handoff(
    std::uint32_t tile_count, std::uint32_t cluster_count,
    std::uint32_t* observed_generation, std::uint32_t* partial_checksums) {
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    namespace cg = cooperative_groups;
    const auto cluster = cg::this_cluster();
    if (cluster.block_rank() != 0U) return;
    auto& storage = *reinterpret_cast<HandoffContractStorage*>(shared_bytes);
    const std::uint32_t warp_group = threadIdx.x /
        PersistentWorkTilePlan::kThreadsPerWarpGroup;
    const std::uint32_t lane = threadIdx.x %
        PersistentWorkTilePlan::kThreadsPerWarpGroup;
    const std::uint32_t cluster_id =
        PersistentWorkTilePlan::cluster_index(blockIdx.x);

    if (threadIdx.x == 0U) {
        for (std::uint32_t slot = 0; slot < PersistentFf1Ff2Ring::kSlots;
             ++slot) {
            init(&storage.ring.ready[slot],
                 PersistentWorkTilePlan::kThreadsPerWarpGroup);
            init(&storage.ring.free[slot],
                 PersistentWorkTilePlan::kThreadsPerWarpGroup);
            storage.ring.generation[slot] =
                std::numeric_limits<std::uint32_t>::max();
        }
    }
    __syncthreads();

    if (warp_group == PersistentWorkTilePlan::kProducerWarpGroup) {
        for (std::uint32_t generation = 0;; ++generation) {
            bool any = false;
            for (std::uint32_t slot = 0; slot < PersistentFf1Ff2Ring::kSlots;
                 ++slot) {
                const std::uint32_t tile = cluster_id *
                    PersistentFf1Ff2Ring::kSlots + slot + generation *
                    cluster_count * PersistentFf1Ff2Ring::kSlots;
                if (tile >= tile_count) continue;
                any = true;
                if (generation != 0U) {
                    storage.ring.free[slot].wait_parity(
                        (generation - 1U) & 1U);
                }
                auto* words = reinterpret_cast<std::uint32_t*>(
                    &storage.ring.hidden[slot]);
                constexpr std::uint32_t words_per_slot =
                    sizeof(DsmASlot) / sizeof(std::uint32_t);
                for (std::uint32_t word = lane; word < words_per_slot;
                     word += PersistentWorkTilePlan::kThreadsPerWarpGroup) {
                    words[word] = tile * 4099U + word;
                }
                if (lane == 0U) storage.ring.generation[slot] = generation;
                (void)storage.ring.ready[slot].arrive();
            }
            if (!any) break;
        }
    } else if (warp_group >= PersistentWorkTilePlan::kFirstMmaWarpGroup) {
        const std::uint32_t slot =
            warp_group - PersistentWorkTilePlan::kFirstMmaWarpGroup;
        if (slot >= PersistentFf1Ff2Ring::kSlots) return;
        for (std::uint32_t tile = PersistentWorkTilePlan::first_tile(
                 cluster_id, slot);
             tile < tile_count;
             tile = PersistentWorkTilePlan::next_tile(tile, cluster_count)) {
            const std::uint32_t generation =
                PersistentFf1Ff2Ring::generation_for_tile(
                    tile, cluster_count);
            storage.ring.ready[slot].wait_parity(generation & 1U);
            if (lane == 0U) {
                observed_generation[tile] = storage.ring.generation[slot];
            }
            const auto* words = reinterpret_cast<const std::uint32_t*>(
                &storage.ring.hidden[slot]);
            constexpr std::uint32_t words_per_slot =
                sizeof(DsmASlot) / sizeof(std::uint32_t);
            std::uint32_t checksum = 0U;
            for (std::uint32_t word = lane; word < words_per_slot;
                 word += PersistentWorkTilePlan::kThreadsPerWarpGroup) {
                checksum ^= words[word] + word * 17U;
            }
            partial_checksums[static_cast<std::size_t>(tile) *
                PersistentWorkTilePlan::kThreadsPerWarpGroup + lane] = checksum;
            (void)storage.ring.free[slot].arrive();
        }
    }
}

__global__ __cluster_dims__(2, 1, 1) void persistent_tile_ownership(
    std::uint32_t tile_count, std::uint32_t cluster_count,
    std::uint32_t* owner, std::uint32_t* visits) {
    namespace cg = cooperative_groups;
    const auto cluster = cg::this_cluster();
    const std::uint32_t warp_group = threadIdx.x /
        PersistentWorkTilePlan::kThreadsPerWarpGroup;
    const std::uint32_t lane = threadIdx.x %
        PersistentWorkTilePlan::kThreadsPerWarpGroup;
    if (cluster.block_rank() != 0U || lane != 0U ||
        warp_group < PersistentWorkTilePlan::kFirstMmaWarpGroup) {
        return;
    }
    const std::uint32_t mma_slot =
        warp_group - PersistentWorkTilePlan::kFirstMmaWarpGroup;
    if (mma_slot >= PersistentWorkTilePlan::kMmaWarpGroups) {
        return;
    }
    const std::uint32_t cluster_id =
        PersistentWorkTilePlan::cluster_index(blockIdx.x);
    for (std::uint32_t tile = PersistentWorkTilePlan::first_tile(
             cluster_id, mma_slot);
         tile < tile_count;
         tile = PersistentWorkTilePlan::next_tile(tile, cluster_count)) {
        // Ownership is injective by construction; no atomic claim is needed.
        owner[tile] = cluster_id * PersistentWorkTilePlan::kMmaWarpGroups +
            mma_slot;
        visits[tile] = 1U;
    }
}

__global__ void writer_roundtrip(std::uint32_t* checksum) {
    __shared__ DsmASlot slot;
    const std::uint32_t group = threadIdx.x;
    if (group < 8U) {
        std::uint8_t raw[16];
        for (std::uint32_t i = 0; i < 16U; ++i) {
            raw[i] = static_cast<std::uint8_t>((group + i) & 0x0fU);
        }
        Ff1HiddenRingWriter::store_group(
            slot, 0U, group, raw,
            cutlass::float_ue4m3_t(1.0F).raw());
    }
    __syncthreads();
    for (std::uint32_t i = threadIdx.x; i < sizeof(DsmASlot); i += blockDim.x) {
        atomicAdd(checksum, reinterpret_cast<std::uint8_t*>(&slot)[i]);
    }
}

__global__ void writer_benchmark(std::uint32_t* sink) {
    __shared__ DsmASlot slot;
    for (std::uint32_t work = threadIdx.x; work < 128U * 8U;
         work += blockDim.x) {
        const std::uint32_t row = work / 8U;
        const std::uint32_t group = work % 8U;
        std::uint8_t raw[16];
        for (std::uint32_t i = 0; i < 16U; ++i) {
            raw[i] = static_cast<std::uint8_t>((row + group + i) & 0x0fU);
        }
        Ff1HiddenRingWriter::store_group(
            slot, row, group, raw,
            cutlass::float_ue4m3_t(1.0F).raw());
    }
    __syncthreads();
    if (threadIdx.x == 0U) {
        atomicAdd(sink, slot.values[0] + slot.scales[0]);
    }
}

int main() {
    auto* kernel = cutlass::device_kernel<PersistentFf2Kernel>;
    cudaFuncAttributes resources{};
    const auto resource_status = cudaFuncGetAttributes(&resources, kernel);
    std::cout << "persistent_kernel_resources cuda="
              << cudaGetErrorString(resource_status)
              << " registers=" << resources.numRegs
              << " static_shared_bytes=" << resources.sharedSizeBytes
              << " max_dynamic_shared_bytes=" << resources.maxDynamicSharedSizeBytes
              << " max_threads=" << resources.maxThreadsPerBlock
              << " shared_storage_bytes="
              << sizeof(typename PersistentFf2Kernel::SharedStorage)
              << '\n';
    constexpr int dynamic_shared_bytes = static_cast<int>(
        sizeof(typename PersistentFf2Kernel::SharedStorage));
    const auto shared_attribute_status = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_shared_bytes);
    int active_blocks_per_sm = 0;
    const auto occupancy_status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm, kernel,
        PersistentFf2Kernel::MaxThreadsPerBlock, dynamic_shared_bytes);
    std::cout << "persistent_kernel_occupancy shared_attribute_cuda="
              << cudaGetErrorString(shared_attribute_status)
              << " occupancy_cuda=" << cudaGetErrorString(occupancy_status)
              << " active_blocks_per_sm=" << active_blocks_per_sm << '\n';
    std::cout << "a_rank=" << cute::rank(Ff2StageLayoutA{})
              << " a_size=" << cute::size(Ff2StageLayoutA{})
              << " a_cosize=" << cute::cosize(Ff2StageLayoutA{})
              << " sfa_rank=" << cute::rank(Ff2StageLayoutSFA{})
              << " sfa_size=" << cute::size(Ff2StageLayoutSFA{})
              << " sfa_cosize=" << cute::cosize(Ff2StageLayoutSFA{})
              << " values_bytes=" << Contract::kValuesPerSlot
              << " scales_bytes=" << Contract::kScalesPerSlot << '\n';
    std::uint32_t* checksum = nullptr;
    cudaMallocManaged(&checksum, sizeof(*checksum));
    *checksum = 0U;
    writer_roundtrip<<<1, 128>>>(checksum);
    const auto status = cudaDeviceSynchronize();
    const bool ok = status == cudaSuccess && *checksum != 0U;
    std::cout << "ff1_hidden_writer_roundtrip=" << (ok ? "pass" : "fail")
              << " checksum=" << *checksum
              << " cuda=" << cudaGetErrorString(status) << '\n';
    constexpr std::uint32_t role_clusters = 8U;
    constexpr std::uint32_t role_pairs = 17U;
    constexpr std::size_t role_partials =
        static_cast<std::size_t>(role_clusters) *
        role_pairs *
        RoleSeparatedCtaPlan::kConsumerCtas *
        RoleSeparatedCtaPlan::kThreadsPerCta;
    std::uint32_t* role_checksums = nullptr;
    cudaMallocManaged(&role_checksums,
                      role_partials * sizeof(*role_checksums));
    cudaMemset(role_checksums, 0,
               role_partials * sizeof(*role_checksums));
    std::uint32_t* role_violations = nullptr;
    cudaMallocManaged(&role_violations, sizeof(*role_violations));
    *role_violations = 0U;
    role_separated_four_cta_handoff<<<
        role_clusters * RoleSeparatedCtaPlan::kClusterCtas,
        RoleSeparatedCtaPlan::kThreadsPerCta,
        sizeof(RoleSeparatedContractStorage)>>>(
            role_clusters, role_pairs, role_checksums, role_violations);
    const auto role_status = cudaDeviceSynchronize();
    bool role_ok = role_status == cudaSuccess;
    constexpr std::uint32_t role_words =
        sizeof(DsmASlot) / sizeof(std::uint32_t);
    role_ok = role_ok && *role_violations == 0U;
    for (std::uint32_t cluster_id = 0; cluster_id < role_clusters;
         ++cluster_id) for (std::uint32_t pair = 0; pair < role_pairs;
         ++pair) {
        std::uint32_t expected = 0U;
        for (std::uint32_t producer = 0;
             producer < RoleSeparatedCtaPlan::kProducerCtas; ++producer)
            for (std::uint32_t word = 0; word < role_words; ++word)
                expected ^= cluster_id * 65537U + pair * 8191U +
                    producer * 4099U + word + word * 17U + producer * 257U;
        for (std::uint32_t consumer = 0;
             consumer < RoleSeparatedCtaPlan::kConsumerCtas; ++consumer) {
            std::uint32_t observed = 0U;
            const std::size_t base =
                ((static_cast<std::size_t>(cluster_id) * role_pairs + pair) *
                     RoleSeparatedCtaPlan::kConsumerCtas + consumer) *
                RoleSeparatedCtaPlan::kThreadsPerCta;
            for (std::uint32_t lane = 0;
                 lane < RoleSeparatedCtaPlan::kThreadsPerCta; ++lane) {
                observed ^= role_checksums[base + lane];
            }
            role_ok = role_ok && observed == expected;
        }
    }
    std::cout << "role_separated_four_cta_handoff="
              << (role_ok ? "pass" : "fail")
              << " clusters=" << role_clusters
              << " pairs=" << role_pairs
              << " producers=" << RoleSeparatedCtaPlan::kProducerCtas
              << " consumers=" << RoleSeparatedCtaPlan::kConsumerCtas
              << " shared_bytes=" << sizeof(RoleSeparatedContractStorage)
              << " violations=" << *role_violations
              << " cuda=" << cudaGetErrorString(role_status) << '\n';
    cudaEvent_t begin{}, end{};
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    for (int i = 0; i < 20; ++i) writer_benchmark<<<1, 128>>>(checksum);
    const auto warmup_status = cudaDeviceSynchronize();
    cudaEventRecord(begin);
    constexpr int iterations = 2000;
    for (int i = 0; i < iterations; ++i) writer_benchmark<<<1, 128>>>(checksum);
    cudaEventRecord(end);
    const auto benchmark_status = cudaEventSynchronize(end);
    float elapsed_ms = 0.0F;
    cudaEventElapsedTime(&elapsed_ms, begin, end);
    const double us = elapsed_ms * 1000.0 / iterations;
    const double gbps = static_cast<double>(Ff1HiddenRingWriter::kSlotBytes) /
        (us * 1000.0);
    std::cout << "ff1_hidden_writer_benchmark slot_bytes="
              << Ff1HiddenRingWriter::kSlotBytes
              << " us=" << us << " effective_GBps=" << gbps
              << " warmup_cuda=" << cudaGetErrorString(warmup_status)
              << " benchmark_cuda=" << cudaGetErrorString(benchmark_status)
              << '\n';
    constexpr std::uint32_t scheduler_tiles = 1003U;
    constexpr std::uint32_t scheduler_clusters = 16U;
    std::uint32_t* scheduler_owner = nullptr;
    std::uint32_t* scheduler_visits = nullptr;
    cudaMallocManaged(&scheduler_owner, scheduler_tiles * sizeof(*scheduler_owner));
    cudaMallocManaged(&scheduler_visits, scheduler_tiles * sizeof(*scheduler_visits));
    for (std::uint32_t tile = 0; tile < scheduler_tiles; ++tile) {
        scheduler_owner[tile] = std::numeric_limits<std::uint32_t>::max();
        scheduler_visits[tile] = 0U;
    }
    persistent_tile_ownership<<<scheduler_clusters * 2U,
        PersistentWorkTilePlan::kThreads>>>(
        scheduler_tiles, scheduler_clusters, scheduler_owner,
        scheduler_visits);
    const auto scheduler_status = cudaDeviceSynchronize();
    bool scheduler_ok = scheduler_status == cudaSuccess;
    for (std::uint32_t tile = 0; tile < scheduler_tiles; ++tile) {
        const std::uint32_t expected_owner = tile %
            (scheduler_clusters * PersistentWorkTilePlan::kMmaWarpGroups);
        scheduler_ok = scheduler_ok && scheduler_visits[tile] == 1U &&
            scheduler_owner[tile] == expected_owner;
    }
    std::cout << "persistent_tile_scheduler="
              << (scheduler_ok ? "pass" : "fail")
              << " tiles=" << scheduler_tiles
              << " clusters=" << scheduler_clusters
              << " producer_wg="
              << PersistentWorkTilePlan::kProducerWarpGroup
              << " mma_wgs=" << PersistentWorkTilePlan::kMmaWarpGroups
              << " atomic_claims=0 cuda="
              << cudaGetErrorString(scheduler_status) << '\n';
    std::uint32_t* handoff_generation = nullptr;
    std::uint32_t* handoff_checksums = nullptr;
    cudaMallocManaged(&handoff_generation,
                      scheduler_tiles * sizeof(*handoff_generation));
    cudaMallocManaged(&handoff_checksums,
        static_cast<std::size_t>(scheduler_tiles) *
            PersistentWorkTilePlan::kThreadsPerWarpGroup *
            sizeof(*handoff_checksums));
    cudaMemset(handoff_generation, 0xff,
               scheduler_tiles * sizeof(*handoff_generation));
    cudaMemset(handoff_checksums, 0,
        static_cast<std::size_t>(scheduler_tiles) *
            PersistentWorkTilePlan::kThreadsPerWarpGroup *
            sizeof(*handoff_checksums));
    persistent_two_mma_handoff<<<scheduler_clusters * 2U,
        PersistentWorkTilePlan::kThreads,
        sizeof(HandoffContractStorage)>>>(
        scheduler_tiles, scheduler_clusters, handoff_generation,
        handoff_checksums);
    const auto handoff_status = cudaDeviceSynchronize();
    bool handoff_ok = handoff_status == cudaSuccess;
    for (std::uint32_t tile = 0; tile < scheduler_tiles; ++tile) {
        const std::uint32_t expected_generation =
            PersistentFf1Ff2Ring::generation_for_tile(
                tile, scheduler_clusters);
        handoff_ok = handoff_ok &&
            handoff_generation[tile] == expected_generation;
        std::uint32_t checksum = 0U;
        for (std::uint32_t lane = 0;
             lane < PersistentWorkTilePlan::kThreadsPerWarpGroup; ++lane) {
            checksum ^= handoff_checksums[static_cast<std::size_t>(tile) *
                PersistentWorkTilePlan::kThreadsPerWarpGroup + lane];
        }
        std::uint32_t expected_checksum = 0U;
        constexpr std::uint32_t words_per_slot =
            sizeof(DsmASlot) / sizeof(std::uint32_t);
        for (std::uint32_t word = 0; word < words_per_slot; ++word) {
            expected_checksum ^= tile * 4099U + word + word * 17U;
        }
        handoff_ok = handoff_ok && checksum == expected_checksum;
    }
    std::cout << "persistent_two_mma_handoff="
              << (handoff_ok ? "pass" : "fail")
              << " tiles=" << scheduler_tiles
              << " slots=" << PersistentFf1Ff2Ring::kSlots
              << " shared_bytes=" << sizeof(HandoffContractStorage)
              << " atomic_hot_path=0 cuda="
              << cudaGetErrorString(handoff_status) << '\n';
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaFree(scheduler_owner);
    cudaFree(scheduler_visits);
    cudaFree(handoff_generation);
    cudaFree(handoff_checksums);
    cudaFree(checksum);
    cudaFree(role_checksums);
    cudaFree(role_violations);
    return kernel == nullptr || resource_status != cudaSuccess ||
        shared_attribute_status != cudaSuccess || occupancy_status != cudaSuccess ||
        active_blocks_per_sm < 1 ||
        resources.maxThreadsPerBlock != PersistentFf2Kernel::MaxThreadsPerBlock ||
        !ok || !role_ok || !scheduler_ok || !handoff_ok ||
        warmup_status != cudaSuccess ||
        benchmark_status != cudaSuccess || elapsed_ms <= 0.0F;
}
