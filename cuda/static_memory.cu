#include "static_memory.hpp"

#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/iterator/constant_input_iterator.cuh>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace beam {

namespace {

std::size_t align_up_size(std::size_t value, std::size_t alignment) {
    const std::size_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

struct Stream3HashLessForCubTemp {
    __host__ __device__ bool operator()(Hash128 a, Hash128 b) const {
        if (a.hi != b.hi) {
            return a.hi < b.hi;
        }
        return a.lo < b.lo;
    }
};

struct Stream3MinValueForCubTemp {
    __host__ __device__ std::uint64_t operator()(std::uint64_t a, std::uint64_t b) const {
        return a < b ? a : b;
    }
};

struct Stream4HashLessForCubTemp {
    __host__ __device__ bool operator()(Hash128 a, Hash128 b) const {
        if (a.hi != b.hi) {
            return a.hi < b.hi;
        }
        return a.lo < b.lo;
    }
};

struct Stream4BestCandidateForCubTemp {
    __host__ __device__ CandidateMeta operator()(CandidateMeta a, CandidateMeta b) const {
        if (a.score_key != b.score_key) {
            return a.score_key < b.score_key ? a : b;
        }
        if (a.parent_idx != b.parent_idx) {
            return a.parent_idx < b.parent_idx ? a : b;
        }
        return a.route_packed <= b.route_packed ? a : b;
    }
};

struct ScoreCountSumForCubTemp {
    __host__ __device__ std::uint64_t operator()(std::uint64_t a, std::uint64_t b) const {
        return a + b;
    }
};

std::uint32_t shard_key_bits_for_count(std::uint32_t shard_count) {
    std::uint32_t values = shard_count + 1U;
    std::uint32_t bits = 0;
    --values;
    while (values != 0U) {
        ++bits;
        values >>= 1U;
    }
    return bits == 0U ? 1U : bits;
}

std::size_t stream3_cub_temp_bytes(std::uint32_t stream3_batch_candidates) {
    if (stream3_batch_candidates > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("stream3 batch candidates exceed CUB int count range");
    }
    Hash128* keys = nullptr;
    Hash128* unique_keys = nullptr;
    std::uint64_t* values = nullptr;
    std::uint64_t* unique_values = nullptr;
    std::uint32_t* unique_count = nullptr;
    std::uint32_t* block_counts = nullptr;
    std::uint32_t* block_offsets = nullptr;
    std::size_t sort_bytes = 0;
    std::size_t reduce_bytes = 0;
    std::size_t scan_bytes = 0;
    const int item_count = static_cast<int>(stream3_batch_candidates);
    const int block_count = static_cast<int>((stream3_batch_candidates + 255U) / 256U);
    cudaError_t status = cub::DeviceMergeSort::SortPairs(
        nullptr,
        sort_bytes,
        keys,
        values,
        item_count,
        Stream3HashLessForCubTemp{},
        0);
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    status = cub::DeviceReduce::ReduceByKey(
        nullptr,
        reduce_bytes,
        keys,
        unique_keys,
        values,
        unique_values,
        unique_count,
        Stream3MinValueForCubTemp{},
        item_count,
        0);
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    status = cub::DeviceScan::ExclusiveSum(
        nullptr,
        scan_bytes,
        block_counts,
        block_offsets,
        block_count,
        0);
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    return align_up_size(std::max(std::max(sort_bytes, reduce_bytes), scan_bytes), 256);
}

std::size_t stream3_partition_cub_temp_bytes(std::uint32_t max_candidates, std::uint32_t shard_count) {
    if (max_candidates > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("stream3 partition candidates exceed CUB int count range");
    }
    std::uint32_t* keys_in = nullptr;
    std::uint32_t* keys_out = nullptr;
    CandidateMeta* values_in = nullptr;
    CandidateMeta* values_out = nullptr;
    std::uint32_t* unique_keys = nullptr;
    std::uint32_t* unique_counts = nullptr;
    std::uint32_t* unique_count = nullptr;
    cub::ConstantInputIterator<std::uint32_t> ones(1U);
    std::size_t sort_bytes = 0;
    std::size_t reduce_bytes = 0;
    const int item_count = static_cast<int>(max_candidates);
    cudaError_t status = cub::DeviceRadixSort::SortPairs(
        nullptr,
        sort_bytes,
        keys_in,
        keys_out,
        values_in,
        values_out,
        item_count,
        0,
        static_cast<int>(shard_key_bits_for_count(shard_count)),
        0);
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    status = cub::DeviceReduce::ReduceByKey(
        nullptr,
        reduce_bytes,
        keys_out,
        unique_keys,
        ones,
        unique_counts,
        unique_count,
        cub::Sum{},
        item_count,
        0);
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    return align_up_size(std::max(sort_bytes, reduce_bytes), 256);
}

std::size_t stream4_cub_temp_bytes(std::uint32_t stream4_capacity) {
    if (stream4_capacity > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("stream4 capacity exceeds CUB int count range");
    }
    Hash128* keys = nullptr;
    Hash128* unique_keys = nullptr;
    CandidateMeta* values = nullptr;
    CandidateMeta* unique_values = nullptr;
    std::uint32_t* unique_count = nullptr;
    std::size_t sort_bytes = 0;
    std::size_t reduce_bytes = 0;
    const int item_count = static_cast<int>(stream4_capacity);
    cudaError_t status = cub::DeviceMergeSort::SortPairs(
        nullptr,
        sort_bytes,
        keys,
        values,
        item_count,
        Stream4HashLessForCubTemp{},
        0);
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    status = cub::DeviceReduce::ReduceByKey(
        nullptr,
        reduce_bytes,
        keys,
        unique_keys,
        values,
        unique_values,
        unique_count,
        Stream4BestCandidateForCubTemp{},
        item_count,
        0);
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }

    std::uint32_t* score_keys_in = nullptr;
    std::uint32_t* score_keys_out = nullptr;
    std::uint64_t* score_counts_in = nullptr;
    std::uint64_t* score_counts_out = nullptr;
    std::uint32_t* score_unique_count = nullptr;
    std::size_t score_sort_bytes = 0;
    std::size_t score_reduce_bytes = 0;
    status = cub::DeviceRadixSort::SortPairs(
        nullptr,
        score_sort_bytes,
        score_keys_in,
        score_keys_out,
        score_counts_in,
        score_counts_out,
        item_count,
        0,
        32,
        0);
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    status = cub::DeviceReduce::ReduceByKey(
        nullptr,
        score_reduce_bytes,
        score_keys_out,
        score_keys_in,
        score_counts_out,
        score_counts_in,
        score_unique_count,
        ScoreCountSumForCubTemp{},
        item_count,
        0);
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    return align_up_size(std::max(std::max(sort_bytes, reduce_bytes), std::max(score_sort_bytes, score_reduce_bytes)), 256);
}

struct Cursor {
    std::byte* base = nullptr;
    std::size_t offset = 0;

    template <typename T>
    T* take(std::uint64_t count, std::size_t alignment = alignof(T)) {
        offset = align_up_size(offset, alignment);
        T* ptr = reinterpret_cast<T*>(base + offset);
        offset += static_cast<std::size_t>(count) * sizeof(T);
        return ptr;
    }
};

std::size_t bytes_streams(const RuntimeConfig& config, const DerivedConfig& derived) {
    const std::uint64_t ring_slots = static_cast<std::uint64_t>(config.ring_count) * derived.ring_slot_count;
    const std::uint64_t ring_candidates = ring_slots * config.b_micro * MOVE_COUNT;
    const std::uint64_t stream3 = config.stream3_batch_candidates;
    const std::uint64_t stream3_partition = std::max<std::uint64_t>(stream3, config.global_spill_capacity);
    const std::uint64_t survivors = static_cast<std::uint64_t>(config.shard_count) * 2ULL * config.stream4_batch_candidates;
    const std::uint64_t stream4_capacity = 2ULL * config.stream4_batch_candidates;
    const std::uint64_t stream4_slots = config.stream4_active_sort_slots;
    const std::uint64_t spill = config.global_spill_capacity;
    std::size_t total = 0;
    total += ring_candidates * sizeof(std::uint32_t);
    total += ring_candidates * sizeof(Hash128);
    total += ring_slots * sizeof(std::uint64_t);
    total += ring_slots * sizeof(std::uint32_t);
    total += 3ULL * stream3 * sizeof(Hash128);
    total += 3ULL * stream3 * sizeof(std::uint64_t);
    const std::uint64_t stream3_blocks = (stream3 + 255ULL) / 256ULL;
    total += stream3 * sizeof(std::uint32_t);
    total += 2ULL * stream3_blocks * sizeof(std::uint32_t);
    total += stream3 * sizeof(std::uint32_t);
    total += 4ULL * config.shard_count * sizeof(std::uint32_t);
    total += 2ULL * config.shard_count * sizeof(std::uint32_t);
    total += sizeof(std::uint32_t);
    total += 2ULL * stream3_partition * sizeof(std::uint32_t);
    total += 2ULL * stream3_partition * sizeof(CandidateMeta);
    total += 2ULL * (static_cast<std::uint64_t>(config.shard_count) + 1ULL) * sizeof(std::uint32_t);
    total += sizeof(std::uint32_t);
    total += std::max(
        stream3_cub_temp_bytes(config.stream3_batch_candidates),
        stream3_partition_cub_temp_bytes(static_cast<std::uint32_t>(stream3_partition), config.shard_count));
    total += sizeof(std::uint32_t);
    total += 3ULL * stream3 * sizeof(CandidateMeta);
    total += sizeof(std::uint32_t);
    total += 2ULL * config.world_size * sizeof(std::uint32_t);
    total += 2ULL * (static_cast<std::uint64_t>(config.world_size) + 1ULL) * sizeof(std::uint32_t);
    total += survivors * sizeof(CandidateMeta);
    total += 2ULL * stream4_slots * stream4_capacity * sizeof(Hash128);
    total += 2ULL * stream4_slots * stream4_capacity * sizeof(CandidateMeta);
    total += 2ULL * stream4_slots * stream4_capacity * sizeof(std::uint32_t);
    total += 2ULL * stream4_slots * stream4_capacity * sizeof(std::uint64_t);
    const std::uint64_t stream4_blocks = (stream4_capacity + 255ULL) / 256ULL;
    total += stream4_slots * stream4_capacity * sizeof(std::uint32_t);
    total += 2ULL * stream4_slots * stream4_blocks * sizeof(std::uint32_t);
    total += stream4_slots * sizeof(std::uint32_t);
    total += stream4_slots * stream4_cub_temp_bytes(static_cast<std::uint32_t>(stream4_capacity));
    total += config.shard_count * sizeof(std::uint32_t);
    total += 3ULL * config.shard_count * sizeof(std::uint32_t);
    total += 2ULL * spill * sizeof(CandidateMeta);
    total += 2ULL * sizeof(std::uint32_t);
    total += sizeof(std::uint32_t);
    total += 2ULL * static_cast<std::uint64_t>(config.shard_count) * SCORE_BIN_COUNT * sizeof(std::uint32_t);
    total += 2ULL * config.shard_count * sizeof(std::uint32_t);
    total += 2ULL * SCORE_BIN_COUNT * sizeof(std::uint64_t);
    total += 2ULL * sizeof(std::uint32_t);
    return align_up_size(total, 256);
}

std::size_t bytes_final(const RuntimeConfig& config, const DerivedConfig& derived, std::size_t layout_streams_bytes) {
    (void)derived;
    const std::uint64_t frontier =
        (config.global_beam_width_max_safe + static_cast<std::uint64_t>(config.world_size) - 1ULL) /
        static_cast<std::uint64_t>(config.world_size);
    const std::uint64_t requests = frontier;
    const std::uint64_t survivors = static_cast<std::uint64_t>(config.shard_count) * 2ULL * config.stream4_batch_candidates;
    const std::uint64_t survivor_blocks = (survivors + 255ULL) / 256ULL;
    std::size_t total = 0;
    total += frontier * sizeof(State128);
    total = std::max(total, layout_streams_bytes);
    total += survivors * sizeof(std::uint32_t);
    total += 2ULL * survivor_blocks * sizeof(std::uint32_t);
    total += frontier * sizeof(CandidateMeta);
    total += sizeof(std::uint32_t);
    total += requests * sizeof(FinalRequest);
    total += sizeof(std::uint32_t);
    if (config.world_size > 1U) {
        total += requests * sizeof(FinalResponse);
    }
    total += 2ULL * config.world_size * sizeof(std::uint32_t);
    total += 2ULL * (static_cast<std::uint64_t>(config.world_size) + 1ULL) * sizeof(std::uint32_t);
    return align_up_size(total, 256);
}

} // namespace

StaticMemoryPlan make_static_memory_plan(const RuntimeConfig& config) {
    const DerivedConfig derived = derive_config(config);
    if (config.b_micro == 0 || config.ring_count == 0 || derived.ring_slot_count == 0 ||
        config.world_size == 0 || config.shard_count == 0 || config.stream4_active_sort_slots == 0 ||
        derived.global_beam_width_effective == 0) {
        throw std::invalid_argument("static memory config contains zero-sized architecture dimension");
    }
    StaticMemoryPlan plan;
    plan.config = config;
    plan.derived = derived;
    plan.frontier_states =
        (config.global_beam_width_max_safe + static_cast<std::uint64_t>(config.world_size) - 1ULL) /
        static_cast<std::uint64_t>(config.world_size);
    if (plan.frontier_states > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::invalid_argument("local frontier capacity exceeds uint32 final target index range");
    }
    plan.score_ring_count =
        static_cast<std::uint64_t>(config.ring_count) * derived.ring_slot_count * config.b_micro * MOVE_COUNT;
    plan.hash_ring_count = plan.score_ring_count;
    plan.parent_base_count = static_cast<std::uint64_t>(config.ring_count) * derived.ring_slot_count;
    plan.ring_count_count = plan.parent_base_count;
    plan.stream3_count = config.stream3_batch_candidates;
    plan.survivor_count = static_cast<std::uint64_t>(config.shard_count) * 2ULL * config.stream4_batch_candidates;
    plan.final_state_count = plan.frontier_states;
    const std::uint32_t stream3_partition_count = static_cast<std::uint32_t>(
        std::max<std::uint64_t>(plan.stream3_count, config.global_spill_capacity));
    plan.stream3_cub_temp_bytes = std::max(
        stream3_cub_temp_bytes(config.stream3_batch_candidates),
        stream3_partition_cub_temp_bytes(stream3_partition_count, config.shard_count));
    plan.stream4_cub_temp_bytes = stream4_cub_temp_bytes(2U * config.stream4_batch_candidates);
    plan.current_frontier_bytes = static_cast<std::size_t>(plan.frontier_states) * sizeof(State128);
    plan.solved_bytes =
        5ULL * sizeof(std::uint32_t) +
        static_cast<std::uint64_t>(config.solved_result_capacity) * sizeof(CandidateMeta) +
        static_cast<std::uint64_t>(config.solved_result_capacity) * sizeof(std::uint32_t);
    plan.layout_streams_bytes = bytes_streams(config, derived);
    plan.layout_final_bytes = bytes_final(config, derived, plan.layout_streams_bytes);
    plan.scratch_pool_bytes = std::max(plan.layout_streams_bytes, plan.layout_final_bytes);
    plan.total_device_bytes =
        align_up_size(plan.current_frontier_bytes, 256) +
        align_up_size(plan.solved_bytes, 256) +
        align_up_size(plan.scratch_pool_bytes, 256);
    return plan;
}

void allocate_static_device_memory(const StaticMemoryPlan& plan, StaticDeviceMemory& memory) {
    if (memory.allocation != nullptr) {
        throw std::invalid_argument("static device memory must be freed before reallocation");
    }
    cudaError_t err = cudaMalloc(&memory.allocation, plan.total_device_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    memory.allocation_bytes = plan.total_device_bytes;
    Cursor root{reinterpret_cast<std::byte*>(memory.allocation), 0};
    memory.current_frontier_states = root.take<State128>(plan.frontier_states, 256);
    root.offset = align_up_size(root.offset, 256);
    memory.solved_flag = root.take<std::uint32_t>(1);
    memory.stop_flag = root.take<std::uint32_t>(1);
    memory.solved_count = root.take<std::uint32_t>(1);
    memory.solved_overflow = root.take<std::uint32_t>(1);
    memory.solved_meta_list = root.take<CandidateMeta>(plan.config.solved_result_capacity);
    memory.solved_depth_list = root.take<std::uint32_t>(plan.config.solved_result_capacity);
    memory.current_depth = root.take<std::uint32_t>(1);
    root.offset = align_up_size(root.offset, 256);
    memory.scratch_pool = root.base + root.offset;
    memory.scratch_pool_bytes = plan.scratch_pool_bytes;

    Cursor streams{reinterpret_cast<std::byte*>(memory.scratch_pool), 0};
    memory.streams.score_ring = streams.take<std::uint32_t>(plan.score_ring_count);
    memory.streams.hash_ring = streams.take<Hash128>(plan.hash_ring_count);
    memory.streams.parent_base = streams.take<std::uint64_t>(plan.parent_base_count);
    memory.streams.count = streams.take<std::uint32_t>(plan.ring_count_count);
    memory.streams.stream3_key_a = streams.take<Hash128>(plan.stream3_count);
    memory.streams.stream3_key_b = streams.take<Hash128>(plan.stream3_count);
    memory.streams.stream3_val_a = streams.take<std::uint64_t>(plan.stream3_count);
    memory.streams.stream3_val_b = streams.take<std::uint64_t>(plan.stream3_count);
    const std::uint64_t stream3_blocks = (plan.stream3_count + 255ULL) / 256ULL;
    memory.streams.stream3_keep_flags = streams.take<std::uint32_t>(plan.stream3_count);
    memory.streams.stream3_block_counts = streams.take<std::uint32_t>(stream3_blocks);
    memory.streams.stream3_block_offsets = streams.take<std::uint32_t>(stream3_blocks);
    memory.streams.stream3_owner = streams.take<std::uint32_t>(plan.stream3_count);
    memory.streams.stream3_shard_counts = streams.take<std::uint32_t>(plan.config.shard_count);
    memory.streams.stream3_shard_offsets = streams.take<std::uint32_t>(plan.config.shard_count);
    memory.streams.stream3_spill_counts = streams.take<std::uint32_t>(plan.config.shard_count);
    memory.streams.stream3_spill_offsets = streams.take<std::uint32_t>(plan.config.shard_count);
    memory.streams.stream3_ready_flag = streams.take<std::uint32_t>(plan.config.shard_count);
    memory.streams.stream3_ready_shard_list = streams.take<std::uint32_t>(plan.config.shard_count);
    memory.streams.stream3_ready_count = streams.take<std::uint32_t>(1);
    const std::uint64_t stream3_partition_count =
        std::max<std::uint64_t>(plan.stream3_count, plan.config.global_spill_capacity);
    memory.streams.stream3_partition_key_a = streams.take<std::uint32_t>(stream3_partition_count);
    memory.streams.stream3_partition_key_b = streams.take<std::uint32_t>(stream3_partition_count);
    memory.streams.stream3_partition_val_a = streams.take<CandidateMeta>(stream3_partition_count);
    memory.streams.stream3_partition_val_b = streams.take<CandidateMeta>(stream3_partition_count);
    memory.streams.stream3_partition_unique_shard =
        streams.take<std::uint32_t>(static_cast<std::uint64_t>(plan.config.shard_count) + 1ULL);
    memory.streams.stream3_partition_unique_counts =
        streams.take<std::uint32_t>(static_cast<std::uint64_t>(plan.config.shard_count) + 1ULL);
    memory.streams.stream3_partition_unique_count = streams.take<std::uint32_t>(1);
    memory.streams.stream3_cub_temp = streams.take<std::byte>(plan.stream3_cub_temp_bytes, 256);
    memory.streams.stream3_cub_temp_bytes = plan.stream3_cub_temp_bytes;
    memory.streams.unique_key = streams.take<Hash128>(plan.stream3_count);
    memory.streams.unique_val = streams.take<std::uint64_t>(plan.stream3_count);
    memory.streams.unique_count = streams.take<std::uint32_t>(1);
    memory.streams.local_pending_buffer = streams.take<CandidateMeta>(plan.stream3_count);
    memory.streams.local_pending_count = streams.take<std::uint32_t>(1);
    memory.streams.remote_send_buffer = streams.take<CandidateMeta>(plan.stream3_count);
    memory.streams.remote_recv_buffer = streams.take<CandidateMeta>(plan.stream3_count);
    memory.streams.send_count = streams.take<std::uint32_t>(plan.config.world_size);
    memory.streams.send_offset = streams.take<std::uint32_t>(static_cast<std::uint64_t>(plan.config.world_size) + 1ULL);
    memory.streams.recv_count = streams.take<std::uint32_t>(plan.config.world_size);
    memory.streams.recv_offset = streams.take<std::uint32_t>(static_cast<std::uint64_t>(plan.config.world_size) + 1ULL);
    memory.streams.survivor_shard = streams.take<CandidateMeta>(plan.survivor_count);
    const std::uint64_t stream4_capacity = 2ULL * plan.config.stream4_batch_candidates;
    const std::uint64_t stream4_slots = plan.config.stream4_active_sort_slots;
    const std::uint64_t stream4_slot_items = stream4_slots * stream4_capacity;
    const std::uint64_t stream4_blocks_per_slot = (stream4_capacity + 255ULL) / 256ULL;
    memory.streams.stream4_key_a = streams.take<Hash128>(stream4_slot_items);
    memory.streams.stream4_key_b = streams.take<Hash128>(stream4_slot_items);
    memory.streams.stream4_val_a = streams.take<CandidateMeta>(stream4_slot_items);
    memory.streams.stream4_val_b = streams.take<CandidateMeta>(stream4_slot_items);
    memory.streams.stream4_score_key_a = streams.take<std::uint32_t>(stream4_slot_items);
    memory.streams.stream4_score_key_b = streams.take<std::uint32_t>(stream4_slot_items);
    memory.streams.stream4_score_count_a = streams.take<std::uint64_t>(stream4_slot_items);
    memory.streams.stream4_score_count_b = streams.take<std::uint64_t>(stream4_slot_items);
    memory.streams.stream4_keep_flags = streams.take<std::uint32_t>(stream4_slot_items);
    memory.streams.stream4_block_counts = streams.take<std::uint32_t>(stream4_slots * stream4_blocks_per_slot);
    memory.streams.stream4_block_offsets = streams.take<std::uint32_t>(stream4_slots * stream4_blocks_per_slot);
    memory.streams.stream4_count = streams.take<std::uint32_t>(stream4_slots);
    memory.streams.stream4_cub_temp =
        streams.take<std::byte>(stream4_slots * plan.stream4_cub_temp_bytes, 256);
    memory.streams.stream4_cub_temp_bytes = plan.stream4_cub_temp_bytes;
    memory.streams.clean_count = streams.take<std::uint32_t>(plan.config.shard_count);
    memory.streams.dirty_count = streams.take<std::uint32_t>(plan.config.shard_count);
    memory.streams.processing_flag = streams.take<std::uint32_t>(plan.config.shard_count);
    memory.streams.global_spill_buffer_a = streams.take<CandidateMeta>(plan.config.global_spill_capacity);
    memory.streams.global_spill_buffer_b = streams.take<CandidateMeta>(plan.config.global_spill_capacity);
    memory.streams.global_spill_count = streams.take<std::uint32_t>(2);
    memory.streams.global_spill_active_index = streams.take<std::uint32_t>(1);
    const std::uint64_t shard_hist_items =
        static_cast<std::uint64_t>(plan.config.shard_count) * SCORE_BIN_COUNT;
    memory.streams.shard_score_hist_a = streams.take<std::uint32_t>(shard_hist_items);
    memory.streams.shard_score_hist_b = streams.take<std::uint32_t>(shard_hist_items);
    memory.streams.shard_score_hist_active_index = streams.take<std::uint32_t>(plan.config.shard_count);
    memory.streams.threshold_hist_active_snapshot = streams.take<std::uint32_t>(plan.config.shard_count);
    memory.streams.local_score_hist = streams.take<std::uint64_t>(SCORE_BIN_COUNT);
    memory.streams.global_score_hist = streams.take<std::uint64_t>(SCORE_BIN_COUNT);
    memory.streams.current_threshold = streams.take<std::uint32_t>(1);
    memory.streams.threshold_initialized = streams.take<std::uint32_t>(1);

    Cursor final{reinterpret_cast<std::byte*>(memory.scratch_pool), 0};
    memory.final.next_frontier_states_tmp = final.take<State128>(plan.frontier_states);
    final.offset = std::max(final.offset, plan.layout_streams_bytes);
    const std::uint64_t final_survivor_blocks = (plan.survivor_count + 255ULL) / 256ULL;
    memory.final.final_keep_flags = final.take<std::uint32_t>(plan.survivor_count);
    memory.final.final_block_counts = final.take<std::uint32_t>(final_survivor_blocks);
    memory.final.final_block_offsets = final.take<std::uint32_t>(final_survivor_blocks);
    memory.final.final_candidate_buffer = final.take<CandidateMeta>(plan.frontier_states);
    memory.final.final_candidate_count = final.take<std::uint32_t>(1);
    memory.final.final_request_buffer = final.take<FinalRequest>(plan.frontier_states);
    memory.final.final_request_count = final.take<std::uint32_t>(1);
    if (plan.config.world_size > 1U) {
        memory.final.final_response_buffer = final.take<FinalResponse>(plan.frontier_states);
    }
    memory.final.final_send_count = final.take<std::uint32_t>(plan.config.world_size);
    memory.final.final_send_offset = final.take<std::uint32_t>(static_cast<std::uint64_t>(plan.config.world_size) + 1ULL);
    memory.final.final_recv_count = final.take<std::uint32_t>(plan.config.world_size);
    memory.final.final_recv_offset = final.take<std::uint32_t>(static_cast<std::uint64_t>(plan.config.world_size) + 1ULL);
}

void free_static_device_memory(StaticDeviceMemory& memory) {
    if (memory.allocation != nullptr) {
        cudaFree(memory.allocation);
    }
    memory = StaticDeviceMemory{};
}

} // namespace beam
