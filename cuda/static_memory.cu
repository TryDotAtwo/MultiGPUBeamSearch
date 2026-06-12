#include "static_memory.hpp"
#include "cub_compat.hpp"
#include "stream3.hpp"

#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>

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

constexpr std::uint32_t FINAL_MATERIALIZE_SLOT_COUNT = 3;

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
    std::uint32_t* score_key_in = nullptr;
    std::uint32_t* score_key_out = nullptr;
    std::uint64_t* score_count_in = nullptr;
    std::uint64_t* score_count_out = nullptr;
    std::uint32_t* score_unique_count = nullptr;
    std::size_t sort_bytes = 0;
    std::size_t reduce_bytes = 0;
    std::size_t scan_bytes = 0;
    std::size_t score_sort_bytes = 0;
    std::size_t score_reduce_bytes = 0;
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
    status = cub::DeviceRadixSort::SortPairs(
        nullptr,
        score_sort_bytes,
        score_key_in,
        score_key_out,
        score_count_in,
        score_count_out,
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
        score_key_out,
        score_key_in,
        score_count_out,
        score_count_in,
        score_unique_count,
        ScoreCountSumForCubTemp{},
        item_count,
        0);
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    return align_up_size(
        std::max(
            std::max(std::max(sort_bytes, reduce_bytes), scan_bytes),
            std::max(score_sort_bytes, score_reduce_bytes)),
        256);
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
    CubConstantInputIterator<std::uint32_t> ones(1U);
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
        CubSum{},
        item_count,
        0);
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    return align_up_size(std::max(sort_bytes, reduce_bytes), 256);
}

std::uint32_t storage_shard_count_for(const RuntimeConfig& config) {
    const std::uint64_t storage =
        static_cast<std::uint64_t>(config.shard_count) *
        static_cast<std::uint64_t>(config.shard_buffer_count);
    if (storage == 0ULL || storage > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::invalid_argument("storage shard count exceeds uint32 range");
    }
    return static_cast<std::uint32_t>(storage);
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

std::size_t final_materialize_cub_temp_bytes(std::uint32_t max_requests) {
    if (max_requests > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("final materialize request count exceeds CUB int count range");
    }
    std::uint32_t* key_in = nullptr;
    std::uint32_t* key_out = nullptr;
    FinalRequest* value_in = nullptr;
    FinalRequest* value_out = nullptr;
    std::size_t sort_bytes = 0;
    const int item_count = static_cast<int>(max_requests);
    cudaError_t status = cub::DeviceRadixSort::SortPairs(
        nullptr,
        sort_bytes,
        key_in,
        key_out,
        value_in,
        value_out,
        item_count,
        0,
        32,
        0);
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    return align_up_size(sort_bytes, 256);
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

struct LayoutSizeCursor {
    std::size_t offset = 0;

    template <typename T>
    void take(std::uint64_t count, std::size_t alignment = alignof(T)) {
        offset = align_up_size(offset, alignment);
        offset += static_cast<std::size_t>(count) * sizeof(T);
    }

    void take_bytes(std::size_t bytes, std::size_t alignment) {
        offset = align_up_size(offset, alignment);
        offset += bytes;
    }
};

std::size_t bytes_final_select(const RuntimeConfig& config, const DerivedConfig& derived);
std::size_t bytes_final_materialize(const RuntimeConfig& config, const DerivedConfig& derived);

std::size_t bytes_streams(const RuntimeConfig& config, const DerivedConfig& derived) {
    const std::uint32_t storage_shard_count = storage_shard_count_for(config);
    const std::uint64_t ring_slots = static_cast<std::uint64_t>(config.ring_count) * derived.ring_slot_count;
    const std::uint64_t ring_candidates = ring_slots * config.b_micro * MOVE_COUNT;
    const std::uint64_t stream3 = config.stream3_batch_candidates;
    const std::uint64_t stream5_slots = config.world_size > 1U ? config.ring_count : 1U;
    const std::uint64_t stream5_send = stream3 * stream5_slots;
    const std::uint64_t stream5_recv_slot = std::max<std::uint64_t>(
        stream3,
        (stream3 * static_cast<std::uint64_t>(config.stream5_recv_capacity_scale_ppm) + 999'999ULL) / 1'000'000ULL);
    const std::uint64_t stream5_recv = stream5_recv_slot * stream5_slots;
    const std::uint64_t stream3_partition = std::max<std::uint64_t>(stream3, config.global_spill_capacity);
    const std::uint64_t survivors =
        static_cast<std::uint64_t>(storage_shard_count) * config.shard_capacity_candidates;
    const std::uint64_t stream4_capacity = config.shard_capacity_candidates;
    const std::uint64_t stream4_slots = config.stream4_active_sort_slots;
    const std::uint64_t spill = config.global_spill_capacity;
    LayoutSizeCursor cursor;
    cursor.take<std::uint32_t>(ring_candidates);
    cursor.take<Hash128>(ring_candidates);
    cursor.take<std::uint64_t>(ring_slots);
    cursor.take<std::uint32_t>(ring_slots);
    cursor.take<Hash128>(stream3);
    cursor.take<Hash128>(stream3);
    cursor.take<std::uint64_t>(stream3);
    cursor.take<std::uint64_t>(stream3);
    const std::uint64_t stream3_blocks = (stream3 + 255ULL) / 256ULL;
    cursor.take<std::uint32_t>(stream3);
    cursor.take<std::uint32_t>(stream3_blocks);
    cursor.take<std::uint32_t>(stream3_blocks);
    cursor.take<std::uint32_t>(stream3);
    cursor.take<std::uint32_t>(config.shard_count);
    cursor.take<std::uint32_t>(config.shard_count);
    cursor.take<std::uint32_t>(config.shard_count);
    cursor.take<std::uint32_t>(config.shard_count);
    cursor.take<std::uint32_t>(storage_shard_count);
    cursor.take<std::uint32_t>(storage_shard_count);
    cursor.take<std::uint32_t>(1);
    cursor.take<std::uint32_t>(config.shard_count);
    cursor.take<std::uint32_t>(stream3_partition);
    cursor.take<std::uint32_t>(stream3_partition);
    cursor.take<CandidateMeta>(stream3_partition);
    cursor.take<CandidateMeta>(stream3_partition);
    cursor.take<std::uint32_t>(static_cast<std::uint64_t>(config.shard_count) + 1ULL);
    cursor.take<std::uint32_t>(static_cast<std::uint64_t>(config.shard_count) + 1ULL);
    cursor.take<std::uint32_t>(1);
    cursor.take<std::uint32_t>(stream3);
    cursor.take<std::uint32_t>(stream3);
    cursor.take<std::uint64_t>(stream3);
    cursor.take<std::uint64_t>(stream3);
    cursor.take<std::uint32_t>(1);
    cursor.take<std::uint64_t>(SCORE_BIN_COUNT);
    cursor.take_bytes(std::max(
        stream3_cub_temp_bytes(config.stream3_batch_candidates),
        stream3_partition_cub_temp_bytes(static_cast<std::uint32_t>(stream3_partition), config.shard_count)),
        256);
    cursor.take<Hash128>(stream3);
    cursor.take<std::uint64_t>(stream3);
    cursor.take<std::uint32_t>(1);
    cursor.take<CandidateMeta>(stream3);
    cursor.take<std::uint32_t>(1);
    cursor.take<CandidateMeta>(stream5_send);
    cursor.take<CandidateMeta>(stream5_recv);
    cursor.take<std::uint32_t>(config.world_size);
    cursor.take<std::uint32_t>(static_cast<std::uint64_t>(config.world_size) + 1ULL);
    cursor.take<std::uint32_t>(config.world_size);
    cursor.take<std::uint32_t>(static_cast<std::uint64_t>(config.world_size) + 1ULL);
    cursor.take<std::uint64_t>(1);
    cursor.take<std::uint64_t>(1);
    const std::uint64_t stream4_slot_items = stream4_slots * stream4_capacity;
    cursor.take<Hash128>(stream4_slot_items);
    cursor.take<Hash128>(stream4_slot_items);
    cursor.take<CandidateMeta>(stream4_slot_items);
    cursor.take<CandidateMeta>(stream4_slot_items);
    cursor.take<std::uint32_t>(stream4_slot_items);
    cursor.take<std::uint32_t>(stream4_slot_items);
    cursor.take<std::uint64_t>(stream4_slot_items);
    cursor.take<std::uint64_t>(stream4_slot_items);
    const std::uint64_t stream4_blocks = (stream4_capacity + 255ULL) / 256ULL;
    cursor.take<std::uint32_t>(stream4_slot_items);
    cursor.take<std::uint32_t>(stream4_slots * stream4_blocks);
    cursor.take<std::uint32_t>(stream4_slots * stream4_blocks);
    cursor.take<std::uint32_t>(stream4_slots);
    cursor.take_bytes(stream4_slots * stream4_cub_temp_bytes(static_cast<std::uint32_t>(stream4_capacity)), 256);
    cursor.take<std::uint32_t>(storage_shard_count);
    cursor.take<std::uint32_t>(storage_shard_count);
    cursor.take<CandidateMeta>(spill);
    cursor.take<CandidateMeta>(spill);
    cursor.take<std::uint32_t>(2);
    cursor.take<std::uint32_t>(1);
    cursor.take<std::uint32_t>(1);
    cursor.take<std::uint64_t>(STREAM_FATAL_TRACE_WORDS);
    const std::size_t final_layout_budget =
        std::max(bytes_final_select(config, derived), bytes_final_materialize(config, derived));
    cursor.offset = std::max(cursor.offset, final_layout_budget);
    cursor.take<CandidateMeta>(survivors);
    cursor.take<std::uint32_t>(storage_shard_count);
    const std::uint64_t shard_hist_items =
        static_cast<std::uint64_t>(storage_shard_count) * SCORE_BIN_COUNT;
    cursor.take<std::uint32_t>(shard_hist_items);
    cursor.take<std::uint32_t>(shard_hist_items);
    cursor.take<std::uint32_t>(storage_shard_count);
    cursor.take<std::uint32_t>(storage_shard_count);
    cursor.take<std::uint64_t>(SCORE_BIN_COUNT);
    cursor.take<std::uint64_t>(SCORE_BIN_COUNT);
    cursor.take<std::uint32_t>(1);
    cursor.take<std::uint32_t>(1);
    return align_up_size(cursor.offset, 256);
}

struct FinalLayoutShape {
    std::uint64_t frontier = 0;
    std::uint64_t requests = 0;
    std::uint64_t final_chunk = 0;
    std::uint64_t final_exchange = 0;
    std::uint64_t final_slots = 0;
    std::uint64_t survivors = 0;
    std::uint64_t selected_capacity = 0;
    std::uint64_t survivor_blocks = 0;
};

FinalLayoutShape make_final_layout_shape(const RuntimeConfig& config, const DerivedConfig& derived) {
    const std::uint32_t storage_shard_count = storage_shard_count_for(config);
    FinalLayoutShape shape;
    shape.frontier =
        (derived.global_beam_width_effective + static_cast<std::uint64_t>(config.world_size) - 1ULL) /
        static_cast<std::uint64_t>(config.world_size);
    shape.requests = shape.frontier;
    const std::uint64_t requested_final_chunk =
        config.final_materialize_chunk_candidates == 0U
            ? static_cast<std::uint64_t>(config.stream3_batch_candidates)
            : static_cast<std::uint64_t>(config.final_materialize_chunk_candidates);
    shape.final_chunk = std::max<std::uint64_t>(1ULL, std::min<std::uint64_t>(shape.frontier, requested_final_chunk));
    shape.final_exchange = std::max<std::uint64_t>(
        shape.final_chunk,
        (shape.final_chunk * static_cast<std::uint64_t>(config.final_materialize_exchange_scale_ppm) +
         999'999ULL) /
            1'000'000ULL);
    shape.final_slots = FINAL_MATERIALIZE_SLOT_COUNT;
    shape.survivors =
        static_cast<std::uint64_t>(storage_shard_count) * config.shard_capacity_candidates;
    shape.selected_capacity = std::min<std::uint64_t>(derived.global_beam_width_effective, shape.survivors);
    shape.survivor_blocks = (shape.survivors + 255ULL) / 256ULL;
    return shape;
}

template <typename CursorT>
void take_final_common(CursorT& cursor, const RuntimeConfig& config, const FinalLayoutShape& shape) {
    if (config.world_size > 1U) {
        cursor.template take<CandidateMeta>(shape.selected_capacity, alignof(CandidateMeta));
    } else {
        cursor.template take<CandidateMeta>(shape.frontier);
        cursor.template take<FinalRequest>(shape.requests);
    }
    cursor.template take<std::uint32_t>(1);
    cursor.template take<std::uint32_t>(1);
    cursor.template take<FinalRequestValidationError>(1, alignof(FinalRequestValidationError));
    cursor.template take<std::uint32_t>(config.world_size, 256);
    cursor.template take<std::uint32_t>(static_cast<std::uint64_t>(config.world_size) + 1ULL, 256);
    cursor.template take<std::uint32_t>(config.world_size, 256);
    cursor.template take<std::uint32_t>(static_cast<std::uint64_t>(config.world_size) + 1ULL, 256);
}

std::size_t bytes_final_select(const RuntimeConfig& config, const DerivedConfig& derived) {
    const FinalLayoutShape shape = make_final_layout_shape(config, derived);
    LayoutSizeCursor cursor;
    take_final_common(cursor, config, shape);
    cursor.offset = align_up_size(cursor.offset, 256);
    cursor.take<std::uint32_t>(shape.survivors);
    cursor.take<std::uint32_t>(shape.survivor_blocks);
    cursor.take<std::uint32_t>(shape.survivor_blocks);
    return align_up_size(cursor.offset, 256);
}

std::size_t bytes_final_materialize(const RuntimeConfig& config, const DerivedConfig& derived) {
    const FinalLayoutShape shape = make_final_layout_shape(config, derived);
    LayoutSizeCursor cursor;
    take_final_common(cursor, config, shape);
    cursor.offset = align_up_size(cursor.offset, 256);
    if (config.world_size > 1U) {
        cursor.take<CandidateMeta>(shape.frontier);
    }
    cursor.take<State128>(shape.frontier, alignof(State128));
    if (config.world_size > 1U) {
        cursor.take<std::uint32_t>(shape.final_slots * shape.final_exchange, 256);
        cursor.take<std::uint32_t>(shape.final_slots * shape.final_exchange, 256);
        cursor.take<FinalRequest>(shape.final_slots * shape.final_exchange, alignof(FinalRequest));
        cursor.take<FinalRequest>(shape.final_slots * shape.final_exchange, alignof(FinalRequest));
        cursor.take<FinalRequest>(shape.final_slots * shape.final_exchange, alignof(FinalRequest));
        cursor.take<FinalResponse>(shape.final_slots * shape.final_exchange, alignof(FinalResponse));
        cursor.take<FinalResponse>(shape.final_slots * shape.final_exchange, alignof(FinalResponse));
        cursor.take<FinalHistoryRecord>(shape.final_slots * shape.final_chunk, alignof(FinalHistoryRecord));
        cursor.take<FinalHistoryRecord>(shape.final_slots * shape.final_exchange, alignof(FinalHistoryRecord));
        cursor.take_bytes(final_materialize_cub_temp_bytes(static_cast<std::uint32_t>(shape.final_exchange)), 256);
    }
    return align_up_size(cursor.offset, 256);
}

std::size_t bytes_solved(const RuntimeConfig& config) {
    LayoutSizeCursor cursor;
    cursor.take<std::uint32_t>(1);
    cursor.take<std::uint32_t>(1);
    cursor.take<std::uint32_t>(1);
    cursor.take<std::uint32_t>(1);
    cursor.take<std::uint32_t>(1);
    cursor.take<CandidateMeta>(config.solved_result_capacity);
    cursor.take<std::uint32_t>(config.solved_result_capacity);
    cursor.take<std::uint32_t>(config.solved_result_capacity);
    cursor.take<std::uint32_t>(1);
    return cursor.offset;
}

} // namespace

StaticMemoryPlan make_static_memory_plan(const RuntimeConfig& config) {
    const DerivedConfig derived = derive_config(config);
    if (config.b_micro == 0 || config.ring_count == 0 || derived.ring_slot_count == 0 ||
        config.world_size == 0 || config.shard_count == 0 || config.stream4_active_sort_slots == 0 ||
        config.shard_capacity_candidates == 0 || derived.global_beam_width_effective == 0) {
        throw std::invalid_argument("static memory config contains zero-sized architecture dimension");
    }
    StaticMemoryPlan plan;
    plan.config = config;
    plan.derived = derived;
    plan.frontier_states =
        (derived.global_beam_width_effective + static_cast<std::uint64_t>(config.world_size) - 1ULL) /
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
    plan.stream5_slot_count = config.world_size > 1U ? config.ring_count : 1U;
    plan.stream5_send_slot_capacity = plan.stream3_count;
    plan.stream5_recv_slot_capacity = std::max<std::uint64_t>(
        plan.stream3_count,
        (plan.stream3_count * static_cast<std::uint64_t>(config.stream5_recv_capacity_scale_ppm) + 999'999ULL) /
            1'000'000ULL);
    plan.final_materialize_slot_count = FINAL_MATERIALIZE_SLOT_COUNT;
    const std::uint64_t requested_final_chunk =
        config.final_materialize_chunk_candidates == 0U
            ? plan.stream3_count
            : static_cast<std::uint64_t>(config.final_materialize_chunk_candidates);
    plan.final_materialize_chunk_capacity =
        std::max<std::uint64_t>(1ULL, std::min<std::uint64_t>(plan.frontier_states, requested_final_chunk));
    plan.final_materialize_exchange_capacity = std::max<std::uint64_t>(
        plan.final_materialize_chunk_capacity,
        (plan.final_materialize_chunk_capacity *
             static_cast<std::uint64_t>(config.final_materialize_exchange_scale_ppm) +
         999'999ULL) /
            1'000'000ULL);
    plan.storage_shard_count = storage_shard_count_for(config);
    plan.survivor_count =
        static_cast<std::uint64_t>(plan.storage_shard_count) * config.shard_capacity_candidates;
    plan.final_selected_candidate_capacity =
        std::min<std::uint64_t>(plan.derived.global_beam_width_effective, plan.survivor_count);
    plan.final_state_count = plan.frontier_states;
    const std::uint32_t stream3_partition_count = static_cast<std::uint32_t>(
        std::max<std::uint64_t>(plan.stream3_count, config.global_spill_capacity));
    plan.stream3_cub_temp_bytes = std::max(
        stream3_cub_temp_bytes(config.stream3_batch_candidates),
        stream3_partition_cub_temp_bytes(stream3_partition_count, config.shard_count));
    plan.stream4_cub_temp_bytes = stream4_cub_temp_bytes(config.shard_capacity_candidates);
    plan.final_materialize_cub_temp_bytes = final_materialize_cub_temp_bytes(
        static_cast<std::uint32_t>(plan.final_materialize_exchange_capacity));
    plan.current_frontier_bytes = static_cast<std::size_t>(plan.frontier_states) * sizeof(State128);
    plan.solved_bytes = bytes_solved(config);
    plan.layout_phase1_streams_bytes = bytes_streams(config, derived);
    plan.layout_phase2_select_bytes = bytes_final_select(config, derived);
    plan.layout_phase3_materialize_bytes = bytes_final_materialize(config, derived);
    plan.layout_streams_bytes = plan.layout_phase1_streams_bytes;
    plan.layout_final_budget_bytes = std::max(plan.layout_phase2_select_bytes, plan.layout_phase3_materialize_bytes);
    plan.layout_final_bytes = plan.layout_final_budget_bytes;
    plan.scratch_pool_bytes = std::max({
        plan.layout_phase1_streams_bytes,
        plan.layout_phase2_select_bytes,
        plan.layout_phase3_materialize_bytes,
    });
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
    memory.global_stop_flag = root.take<std::uint32_t>(1);
    memory.solved_meta_list = root.take<CandidateMeta>(plan.config.solved_result_capacity);
    memory.solved_depth_list = root.take<std::uint32_t>(plan.config.solved_result_capacity);
    memory.solved_suffix_list = root.take<std::uint32_t>(plan.config.solved_result_capacity);
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
    memory.streams.stream3_ready_flag = streams.take<std::uint32_t>(plan.storage_shard_count);
    memory.streams.stream3_ready_shard_list = streams.take<std::uint32_t>(plan.storage_shard_count);
    memory.streams.stream3_ready_count = streams.take<std::uint32_t>(1);
    memory.streams.stream3_write_buffer_index = streams.take<std::uint32_t>(plan.config.shard_count);
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
    memory.streams.stream3_score_key_a = streams.take<std::uint32_t>(plan.stream3_count);
    memory.streams.stream3_score_key_b = streams.take<std::uint32_t>(plan.stream3_count);
    memory.streams.stream3_score_count_a = streams.take<std::uint64_t>(plan.stream3_count);
    memory.streams.stream3_score_count_b = streams.take<std::uint64_t>(plan.stream3_count);
    memory.streams.stream3_score_unique_count = streams.take<std::uint32_t>(1);
    memory.streams.stream3_score_hist = streams.take<std::uint64_t>(SCORE_BIN_COUNT);
    memory.streams.stream3_cub_temp = streams.take<std::byte>(plan.stream3_cub_temp_bytes, 256);
    memory.streams.stream3_cub_temp_bytes = plan.stream3_cub_temp_bytes;
    memory.streams.unique_key = streams.take<Hash128>(plan.stream3_count);
    memory.streams.unique_val = streams.take<std::uint64_t>(plan.stream3_count);
    memory.streams.unique_count = streams.take<std::uint32_t>(1);
#if BEAM_DEBUG_DEPTH_FLOW_TRACE
    memory.streams.stream3_threshold_pass_count_by_ring =
        streams.take<std::uint32_t>(plan.config.ring_count);
    memory.streams.stream3_unique_count_by_ring =
        streams.take<std::uint32_t>(plan.config.ring_count);
#endif
    memory.streams.local_pending_buffer = streams.take<CandidateMeta>(plan.stream3_count);
    memory.streams.local_pending_count = streams.take<std::uint32_t>(1);
    memory.streams.remote_send_buffer =
        streams.take<CandidateMeta>(plan.stream5_slot_count * plan.stream5_send_slot_capacity);
    memory.streams.remote_recv_buffer =
        streams.take<CandidateMeta>(plan.stream5_slot_count * plan.stream5_recv_slot_capacity);
    memory.streams.send_count = streams.take<std::uint32_t>(plan.config.world_size);
    memory.streams.send_offset = streams.take<std::uint32_t>(static_cast<std::uint64_t>(plan.config.world_size) + 1ULL);
    memory.streams.recv_count = streams.take<std::uint32_t>(plan.config.world_size);
    memory.streams.recv_offset = streams.take<std::uint32_t>(static_cast<std::uint64_t>(plan.config.world_size) + 1ULL);
    memory.streams.stream5_local_round_count = streams.take<std::uint64_t>(1);
    memory.streams.stream5_global_round_count = streams.take<std::uint64_t>(1);
    const std::uint64_t stream4_capacity = plan.config.shard_capacity_candidates;
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
    memory.streams.dirty_count = streams.take<std::uint32_t>(plan.storage_shard_count);
    memory.streams.processing_flag = streams.take<std::uint32_t>(plan.storage_shard_count);
    memory.streams.global_spill_buffer_a = streams.take<CandidateMeta>(plan.config.global_spill_capacity);
    memory.streams.global_spill_buffer_b = streams.take<CandidateMeta>(plan.config.global_spill_capacity);
    memory.streams.global_spill_count = streams.take<std::uint32_t>(2);
    memory.streams.global_spill_active_index = streams.take<std::uint32_t>(1);
    memory.streams.fatal_error_flag = streams.take<std::uint32_t>(1);
    memory.streams.fatal_error_trace = streams.take<std::uint64_t>(STREAM_FATAL_TRACE_WORDS);
    streams.offset = std::max(streams.offset, plan.layout_final_budget_bytes);
    memory.streams.survivor_shard = streams.take<CandidateMeta>(plan.survivor_count);
    memory.streams.clean_count = streams.take<std::uint32_t>(plan.storage_shard_count);
    const std::uint64_t shard_hist_items =
        static_cast<std::uint64_t>(plan.storage_shard_count) * SCORE_BIN_COUNT;
    memory.streams.shard_score_hist_a = streams.take<std::uint32_t>(shard_hist_items);
    memory.streams.shard_score_hist_b = streams.take<std::uint32_t>(shard_hist_items);
    memory.streams.shard_score_hist_active_index = streams.take<std::uint32_t>(plan.storage_shard_count);
    memory.streams.threshold_hist_active_snapshot = streams.take<std::uint32_t>(plan.storage_shard_count);
    memory.streams.local_score_hist = streams.take<std::uint64_t>(SCORE_BIN_COUNT);
    memory.streams.global_score_hist = streams.take<std::uint64_t>(SCORE_BIN_COUNT);
    memory.streams.current_threshold = streams.take<std::uint32_t>(2);
    memory.streams.threshold_initialized = streams.take<std::uint32_t>(2);
    memory.streams.current_threshold_active_index = streams.take<std::uint32_t>(1);
    memory.streams.threshold_request_local = streams.take<std::uint32_t>(1);
    memory.streams.threshold_request_global = streams.take<std::uint32_t>(1);

    Cursor final_common{reinterpret_cast<std::byte*>(memory.scratch_pool), 0};
    if (plan.config.world_size > 1U) {
        memory.final.final_selected_buffer = final_common.take<CandidateMeta>(
            plan.final_selected_candidate_capacity,
            alignof(CandidateMeta));
    } else {
        memory.final.final_candidate_buffer = final_common.take<CandidateMeta>(plan.frontier_states);
        memory.final.final_request_buffer = final_common.take<FinalRequest>(plan.frontier_states);
    }
    memory.final.final_candidate_count = final_common.take<std::uint32_t>(1);
    memory.final.final_request_count = final_common.take<std::uint32_t>(1);
    memory.final.final_validation_error =
        final_common.take<FinalRequestValidationError>(1, alignof(FinalRequestValidationError));
    memory.final.final_send_count = final_common.take<std::uint32_t>(plan.config.world_size, 256);
    memory.final.final_send_offset =
        final_common.take<std::uint32_t>(static_cast<std::uint64_t>(plan.config.world_size) + 1ULL, 256);
    memory.final.final_recv_count = final_common.take<std::uint32_t>(plan.config.world_size, 256);
    memory.final.final_recv_offset =
        final_common.take<std::uint32_t>(static_cast<std::uint64_t>(plan.config.world_size) + 1ULL, 256);

    const std::size_t final_common_bytes = align_up_size(final_common.offset, 256);
    Cursor final_select{reinterpret_cast<std::byte*>(memory.scratch_pool), final_common_bytes};
    const std::uint64_t final_survivor_blocks = (plan.survivor_count + 255ULL) / 256ULL;
    memory.final.final_keep_flags = final_select.take<std::uint32_t>(plan.survivor_count);
    memory.final.final_block_counts = final_select.take<std::uint32_t>(final_survivor_blocks);
    memory.final.final_block_offsets = final_select.take<std::uint32_t>(final_survivor_blocks);

    Cursor final_materialize{reinterpret_cast<std::byte*>(memory.scratch_pool), final_common_bytes};
    if (plan.config.world_size > 1U) {
        memory.final.final_candidate_buffer = final_materialize.take<CandidateMeta>(plan.frontier_states);
    }
    memory.final.next_frontier_states_tmp =
        final_materialize.take<State128>(plan.frontier_states, alignof(State128));
    if (plan.config.world_size > 1U) {
        const std::uint64_t final_slot_items =
            static_cast<std::uint64_t>(plan.final_materialize_slot_count) *
            plan.final_materialize_exchange_capacity;
        const std::uint64_t final_history_send_items =
            static_cast<std::uint64_t>(plan.final_materialize_slot_count) *
            plan.final_materialize_chunk_capacity;
        memory.final.final_mat_key_a = final_materialize.take<std::uint32_t>(final_slot_items, 256);
        memory.final.final_mat_key_b = final_materialize.take<std::uint32_t>(final_slot_items, 256);
        memory.final.final_mat_request_a =
            final_materialize.take<FinalRequest>(final_slot_items, alignof(FinalRequest));
        memory.final.final_mat_request_b =
            final_materialize.take<FinalRequest>(final_slot_items, alignof(FinalRequest));
        memory.final.final_mat_request_recv =
            final_materialize.take<FinalRequest>(final_slot_items, alignof(FinalRequest));
        memory.final.final_mat_response_send =
            final_materialize.take<FinalResponse>(final_slot_items, alignof(FinalResponse));
        memory.final.final_mat_response_recv =
            final_materialize.take<FinalResponse>(final_slot_items, alignof(FinalResponse));
        memory.final.final_mat_history_send =
            final_materialize.take<FinalHistoryRecord>(final_history_send_items, alignof(FinalHistoryRecord));
        memory.final.final_mat_history_recv =
            final_materialize.take<FinalHistoryRecord>(final_slot_items, alignof(FinalHistoryRecord));
        memory.final.final_mat_cub_temp =
            final_materialize.take<std::byte>(plan.final_materialize_cub_temp_bytes, 256);
        memory.final.final_mat_cub_temp_bytes = plan.final_materialize_cub_temp_bytes;
    }
}

void free_static_device_memory(StaticDeviceMemory& memory) {
    if (memory.allocation != nullptr) {
        cudaFree(memory.allocation);
    }
    memory = StaticDeviceMemory{};
}

} // namespace beam
