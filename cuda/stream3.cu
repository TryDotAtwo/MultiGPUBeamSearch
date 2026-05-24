#include "stream3.hpp"

#include "config.hpp"
#include "hash.hpp"
#include "nvtx_ranges.hpp"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/iterator/constant_input_iterator.cuh>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace beam {

namespace {

__device__ std::uint8_t owner_from_hash128_stream3_device(Hash128 hash, std::uint32_t world_size) {
    return static_cast<std::uint8_t>(hash128_owner_distribution_key(hash) % world_size);
}

__device__ std::uint32_t pack_route_stream3_device(
    std::uint16_t source_rank,
    std::uint8_t owner,
    std::uint8_t move) {
    return (static_cast<std::uint32_t>(source_rank) << 16) |
           (static_cast<std::uint32_t>(owner) << 8) |
           static_cast<std::uint32_t>(move);
}

__device__ std::uint32_t shard_from_hash128_stream3_device(Hash128 hash, std::uint32_t shard_count) {
    return static_cast<std::uint32_t>(hash128_shard_distribution_key(hash) % shard_count);
}

struct Stream3HashLess {
    __host__ __device__ bool operator()(Hash128 a, Hash128 b) const {
        if (a.hi != b.hi) {
            return a.hi < b.hi;
        }
        return a.lo < b.lo;
    }
};

struct Stream3MinValue {
    __host__ __device__ std::uint64_t operator()(std::uint64_t a, std::uint64_t b) const {
        return a < b ? a : b;
    }
};

std::uint32_t stream3_shard_key_bits(std::uint32_t shard_count) {
    std::uint32_t values = shard_count + 1U;
    std::uint32_t bits = 0;
    --values;
    while (values != 0U) {
        ++bits;
        values >>= 1U;
    }
    return bits == 0U ? 1U : bits;
}

void check_cub(cudaError_t status, const char* op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
    }
}

__global__ void stream3_mark_counts_kernel(
    const std::uint32_t* score_ring,
    const std::uint32_t* count,
    std::uint32_t* keep_flags,
    std::uint32_t* block_counts,
    const std::uint32_t* current_threshold_device,
    std::uint32_t current_threshold_value,
    std::uint32_t b_micro,
    std::uint32_t stream3_batch_candidates) {
    __shared__ std::uint32_t flags[256];
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t i = blockIdx.x * blockDim.x + tid;
    const std::uint32_t candidates_per_slot = b_micro * static_cast<std::uint32_t>(MOVE_COUNT);
    std::uint32_t keep = 0;
    const std::uint32_t current_threshold =
        current_threshold_device == nullptr ? current_threshold_value : *current_threshold_device;
    if (i < stream3_batch_candidates) {
        const std::uint32_t ring_slot = i / candidates_per_slot;
        const std::uint32_t local_i = i % candidates_per_slot;
        const std::uint32_t parent_local = local_i / static_cast<std::uint32_t>(MOVE_COUNT);
        const std::uint32_t parent_count = count == nullptr ? b_micro : count[ring_slot];
        const std::uint32_t score_key = score_ring[i];
        keep = parent_local < parent_count && score_key <= current_threshold ? 1U : 0U;
        keep_flags[i] = keep;
    }
    flags[tid] = keep;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            flags[tid] += flags[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_counts[blockIdx.x] = flags[0];
    }
}

__global__ void stream3_finalize_block_scan_kernel(
    const std::uint32_t* block_counts,
    std::uint32_t* block_offsets,
    std::uint32_t* compact_count,
    std::uint32_t block_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const std::uint32_t last = block_count - 1U;
    *compact_count = block_offsets[last] + block_counts[last];
}

void stream3_scan_block_counts(
    const std::uint32_t* block_counts,
    std::uint32_t* block_offsets,
    std::uint32_t* compact_count,
    std::uint32_t block_count,
    void* cub_temp_storage,
    std::size_t cub_temp_storage_bytes,
    cudaStream_t stream) {
    std::size_t scan_temp_bytes = cub_temp_storage_bytes;
    check_cub(
        cub::DeviceScan::ExclusiveSum(
            cub_temp_storage,
            scan_temp_bytes,
            block_counts,
            block_offsets,
            static_cast<int>(block_count),
            stream),
        "cub::DeviceScan::ExclusiveSum stream3 block counts");
    stream3_finalize_block_scan_kernel<<<1, 1, 0, stream>>>(
        block_counts,
        block_offsets,
        compact_count,
        block_count);
}

__global__ void stream3_compact_kernel(
    const std::uint32_t* score_ring,
    const Hash128* hash_ring,
    const std::uint32_t* keep_flags,
    const std::uint32_t* block_offsets,
    Hash128* compact_key,
    std::uint64_t* compact_val,
    std::uint32_t stream3_batch_candidates) {
    __shared__ std::uint32_t scan[256];
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t i = blockIdx.x * blockDim.x + tid;
    const std::uint32_t keep = i < stream3_batch_candidates ? keep_flags[i] : 0U;
    scan[tid] = keep;
    __syncthreads();
    for (std::uint32_t offset = 1; offset < blockDim.x; offset <<= 1) {
        const std::uint32_t add = tid >= offset ? scan[tid - offset] : 0U;
        __syncthreads();
        scan[tid] += add;
        __syncthreads();
    }
    if (keep != 0U) {
        const std::uint32_t out = block_offsets[blockIdx.x] + scan[tid] - 1U;
        const std::uint32_t score_key = score_ring[i];
        compact_key[out] = hash_ring[i];
        compact_val[out] = (static_cast<std::uint64_t>(score_key) << 32) | static_cast<std::uint64_t>(i);
    }
}

__global__ void stream3_fill_sort_tail_kernel(
    Hash128* key,
    std::uint64_t* val,
    const std::uint32_t* compact_count,
    std::uint32_t stream3_batch_candidates) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count = *compact_count;
    if (i >= count && i < stream3_batch_candidates) {
        key[i] = Hash128{UINT64_MAX, UINT64_MAX};
        val[i] = UINT64_MAX;
    }
}

__global__ void stream3_mark_valid_unique_counts_kernel(
    const std::uint64_t* reduced_val,
    std::uint32_t* head_flags,
    std::uint32_t* block_counts,
    const std::uint32_t* reduced_count,
    std::uint32_t stream3_batch_candidates) {
    __shared__ std::uint32_t flags[256];
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t i = blockIdx.x * blockDim.x + tid;
    const std::uint32_t count = *reduced_count;
    std::uint32_t flag = 0;
    if (i < count) {
        flag = reduced_val[i] != UINT64_MAX ? 1U : 0U;
        head_flags[i] = flag;
    } else if (i < stream3_batch_candidates) {
        head_flags[i] = 0;
    }
    flags[tid] = flag;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            flags[tid] += flags[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_counts[blockIdx.x] = flags[0];
    }
}

__global__ void stream3_compact_valid_unique_kernel(
    const Hash128* reduced_key,
    const std::uint64_t* reduced_val,
    const std::uint32_t* head_flags,
    const std::uint32_t* block_offsets,
    Hash128* unique_key,
    std::uint64_t* unique_val,
    std::uint32_t stream3_batch_candidates) {
    __shared__ std::uint32_t scan[256];
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t i = blockIdx.x * blockDim.x + tid;
    const std::uint32_t keep = i < stream3_batch_candidates ? head_flags[i] : 0U;
    scan[tid] = keep;
    __syncthreads();
    for (std::uint32_t offset = 1; offset < blockDim.x; offset <<= 1) {
        const std::uint32_t add = tid >= offset ? scan[tid - offset] : 0U;
        __syncthreads();
        scan[tid] += add;
        __syncthreads();
    }
    if (keep != 0U) {
        const std::uint32_t out = block_offsets[blockIdx.x] + scan[tid] - 1U;
        unique_key[out] = reduced_key[i];
        unique_val[out] = reduced_val[i];
    }
}

__global__ void stream3_restore_owner_split_kernel(
    const Hash128* unique_key,
    const std::uint64_t* unique_val,
    const std::uint32_t* unique_count,
    const std::uint64_t* parent_base,
    CandidateMeta* meta_scratch,
    std::uint32_t* owner_scratch,
    std::uint16_t local_rank,
    std::uint32_t world_size,
    std::uint32_t b_micro) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count = *unique_count;
    if (i >= count) {
        return;
    }
    const std::uint32_t candidates_per_slot = b_micro * static_cast<std::uint32_t>(MOVE_COUNT);
    const Hash128 hash = unique_key[i];
    const std::uint64_t value = unique_val[i];
    const std::uint32_t score_key = static_cast<std::uint32_t>(value >> 32);
    const std::uint32_t payload_id = static_cast<std::uint32_t>(value & 0xffffffffULL);
    const std::uint32_t ring_slot = payload_id / candidates_per_slot;
    const std::uint32_t local_i = payload_id % candidates_per_slot;
    const std::uint32_t parent_local = local_i / static_cast<std::uint32_t>(MOVE_COUNT);
    const std::uint8_t move = static_cast<std::uint8_t>(local_i % static_cast<std::uint32_t>(MOVE_COUNT));
    const std::uint64_t parent_idx = (parent_base == nullptr ? 0ULL : parent_base[ring_slot]) + parent_local;
    const std::uint32_t owner = owner_from_hash128_stream3_device(hash, world_size);
    owner_scratch[i] = owner;
    meta_scratch[i] = CandidateMeta{hash, parent_idx, score_key, pack_route_stream3_device(local_rank, owner, move)};
}

__global__ void stream3_restore_single_owner_kernel(
    const Hash128* unique_key,
    const std::uint64_t* unique_val,
    const std::uint32_t* unique_count,
    const std::uint64_t* parent_base,
    CandidateMeta* local_pending_buffer,
    std::uint32_t* local_pending_count,
    std::uint32_t* send_count,
    std::uint32_t* send_offset,
    std::uint16_t local_rank,
    std::uint32_t b_micro) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count = *unique_count;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *local_pending_count = count;
        send_count[0] = 0;
        send_offset[0] = 0;
        send_offset[1] = 0;
    }
    if (i >= count) {
        return;
    }
    const std::uint32_t candidates_per_slot = b_micro * static_cast<std::uint32_t>(MOVE_COUNT);
    const Hash128 hash = unique_key[i];
    const std::uint64_t value = unique_val[i];
    const std::uint32_t score_key = static_cast<std::uint32_t>(value >> 32);
    const std::uint32_t payload_id = static_cast<std::uint32_t>(value & 0xffffffffULL);
    const std::uint32_t ring_slot = payload_id / candidates_per_slot;
    const std::uint32_t local_i = payload_id % candidates_per_slot;
    const std::uint32_t parent_local = local_i / static_cast<std::uint32_t>(MOVE_COUNT);
    const std::uint8_t move = static_cast<std::uint8_t>(local_i % static_cast<std::uint32_t>(MOVE_COUNT));
    const std::uint64_t parent_idx = (parent_base == nullptr ? 0ULL : parent_base[ring_slot]) + parent_local;
    local_pending_buffer[i] = CandidateMeta{
        hash,
        parent_idx,
        score_key,
        pack_route_stream3_device(local_rank, static_cast<std::uint8_t>(local_rank), move)};
}

__global__ void stream3_restore_single_owner_partition_kernel(
    const Hash128* unique_key,
    const std::uint64_t* unique_val,
    const std::uint32_t* unique_count,
    const std::uint64_t* parent_base,
    std::uint32_t* local_pending_count,
    std::uint32_t* send_count,
    std::uint32_t* send_offset,
    std::uint32_t* partition_key,
    CandidateMeta* partition_val,
    std::uint16_t local_rank,
    std::uint32_t b_micro,
    std::uint32_t max_candidates,
    std::uint32_t shard_count) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count = *unique_count;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *local_pending_count = count;
        send_count[0] = 0;
        send_offset[0] = 0;
        send_offset[1] = 0;
    }
    if (i >= max_candidates) {
        return;
    }
    if (i >= count) {
        partition_key[i] = shard_count;
        partition_val[i] = CandidateMeta{};
        return;
    }
    const std::uint32_t candidates_per_slot = b_micro * static_cast<std::uint32_t>(MOVE_COUNT);
    const Hash128 hash = unique_key[i];
    const std::uint64_t value = unique_val[i];
    const std::uint32_t score_key = static_cast<std::uint32_t>(value >> 32);
    const std::uint32_t payload_id = static_cast<std::uint32_t>(value & 0xffffffffULL);
    const std::uint32_t ring_slot = payload_id / candidates_per_slot;
    const std::uint32_t local_i = payload_id % candidates_per_slot;
    const std::uint32_t parent_local = local_i / static_cast<std::uint32_t>(MOVE_COUNT);
    const std::uint8_t move = static_cast<std::uint8_t>(local_i % static_cast<std::uint32_t>(MOVE_COUNT));
    const std::uint64_t parent_idx = (parent_base == nullptr ? 0ULL : parent_base[ring_slot]) + parent_local;
    const CandidateMeta candidate{
        hash,
        parent_idx,
        score_key,
        pack_route_stream3_device(local_rank, static_cast<std::uint8_t>(local_rank), move)};
    partition_key[i] = shard_from_hash128_stream3_device(hash, shard_count);
    partition_val[i] = candidate;
}

__global__ void stream3_count_owner_kernel(
    const std::uint32_t* owner_scratch,
    const std::uint32_t* unique_count,
    std::uint32_t* local_pending_count,
    std::uint32_t* send_count,
    std::uint16_t local_rank,
    std::uint32_t world_size) {
    __shared__ std::uint32_t sums[256];
    const std::uint32_t peer = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t count = *unique_count;
    std::uint32_t thread_sum = 0;
    for (std::uint32_t i = tid; i < count; i += blockDim.x) {
        thread_sum += owner_scratch[i] == peer ? 1U : 0U;
    }
    sums[tid] = thread_sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        if (peer == static_cast<std::uint32_t>(local_rank)) {
            *local_pending_count = sums[0];
            send_count[peer] = 0;
        } else if (peer < world_size) {
            send_count[peer] = sums[0];
        }
    }
}

__global__ void stream3_scan_send_counts_kernel(
    const std::uint32_t* send_count,
    std::uint32_t* send_offset,
    std::uint32_t world_size) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    std::uint32_t running = 0;
    send_offset[0] = 0;
    for (std::uint32_t peer = 0; peer < world_size; ++peer) {
        running += send_count[peer];
        send_offset[peer + 1U] = running;
    }
}

__global__ void stream3_scatter_owner_kernel(
    CandidateMeta* meta_scratch,
    const std::uint32_t* owner_scratch,
    const std::uint32_t* unique_count,
    CandidateMeta* remote_send_buffer,
    const std::uint32_t* send_offset,
    std::uint16_t local_rank,
    std::uint32_t world_size) {
    __shared__ std::uint32_t scan[256];
    __shared__ std::uint32_t tile_total;
    const std::uint32_t peer = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t count = *unique_count;
    std::uint32_t running = 0;
    for (std::uint32_t base = 0; base < count; base += blockDim.x) {
        const std::uint32_t i = base + tid;
        const bool match = i < count && owner_scratch[i] == peer;
        const CandidateMeta candidate = i < count ? meta_scratch[i] : CandidateMeta{};
        scan[tid] = match ? 1U : 0U;
        __syncthreads();
        for (std::uint32_t offset = 1; offset < blockDim.x; offset <<= 1) {
            const std::uint32_t add = tid >= offset ? scan[tid - offset] : 0U;
            __syncthreads();
            scan[tid] += add;
            __syncthreads();
        }
        if (tid == blockDim.x - 1U) {
            tile_total = scan[tid];
        }
        __syncthreads();
        if (match) {
            const std::uint32_t out = running + scan[tid] - 1U;
            if (peer == static_cast<std::uint32_t>(local_rank)) {
                meta_scratch[out] = candidate;
            } else {
                remote_send_buffer[send_offset[peer] + out] = candidate;
            }
        }
        running += tile_total;
        __syncthreads();
    }
}

__global__ void stream3_partition_fill_input_kernel(
    const CandidateMeta* input,
    const std::uint32_t* input_count,
    std::uint32_t* partition_key,
    CandidateMeta* partition_val,
    std::uint32_t max_candidates,
    std::uint32_t shard_count) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_candidates) {
        return;
    }
    const std::uint32_t count = *input_count;
    if (i < count) {
        const CandidateMeta candidate = input[i];
        partition_key[i] = shard_from_hash128_stream3_device(candidate.hash, shard_count);
        partition_val[i] = candidate;
    } else {
        partition_key[i] = shard_count;
        partition_val[i] = CandidateMeta{};
    }
}

__global__ void stream3_partition_fill_active_spill_kernel(
    const CandidateMeta* global_spill_buffer_a,
    const CandidateMeta* global_spill_buffer_b,
    const std::uint32_t* global_spill_count,
    const std::uint32_t* global_spill_active_index,
    std::uint32_t* partition_key,
    CandidateMeta* partition_val,
    std::uint32_t max_candidates,
    std::uint32_t shard_count) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_candidates) {
        return;
    }
    const std::uint32_t active = *global_spill_active_index & 1U;
    const CandidateMeta* input = active == 0U ? global_spill_buffer_a : global_spill_buffer_b;
    const std::uint32_t count = global_spill_count[active];
    if (i < count) {
        const CandidateMeta candidate = input[i];
        partition_key[i] = shard_from_hash128_stream3_device(candidate.hash, shard_count);
        partition_val[i] = candidate;
    } else {
        partition_key[i] = shard_count;
        partition_val[i] = CandidateMeta{};
    }
}

__global__ void stream3_prepare_partition_counts_kernel(
    const std::uint32_t* unique_shard,
    const std::uint32_t* unique_counts,
    const std::uint32_t* unique_count,
    const std::uint32_t* clean_count,
    const std::uint32_t* dirty_count,
    const std::uint32_t* processing_flag,
    std::uint32_t* shard_counts,
    std::uint32_t* shard_offsets,
    std::uint32_t* spill_counts,
    std::uint32_t* spill_offsets,
    std::uint32_t* global_spill_count,
    const std::uint32_t* global_spill_active_index,
    std::uint32_t* write_buffer_index,
    std::uint32_t shard_count,
    std::uint32_t shard_buffer_count,
    std::uint32_t shard_capacity_candidates,
    std::uint32_t stream4_batch_candidates,
    std::uint32_t global_spill_capacity,
    std::uint32_t append_to_active_spill,
    std::uint32_t* fatal_error_flag,
    std::uint64_t* fatal_error_trace) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    for (std::uint32_t shard = 0; shard < shard_count; ++shard) {
        shard_counts[shard] = 0;
        shard_offsets[shard] = 0;
        spill_counts[shard] = 0;
        spill_offsets[shard] = 0;
    }
    const std::uint32_t active = *global_spill_active_index & 1U;
    const std::uint32_t spill_index = append_to_active_spill != 0U ? active : (active ^ 1U);
    std::uint32_t source_running = 0;
    std::uint32_t spill_running = append_to_active_spill != 0U ? global_spill_count[active] : 0U;
    const std::uint32_t groups = *unique_count;
    const std::uint32_t shard_capacity = shard_capacity_candidates;
    for (std::uint32_t group = 0; group < groups; ++group) {
        const std::uint32_t shard = unique_shard[group];
        const std::uint32_t raw_count = unique_counts[group];
        if (shard >= shard_count) {
            continue;
        }
        std::uint32_t physical_shard = shard;
        std::uint32_t existing = clean_count[physical_shard] + dirty_count[physical_shard];
        std::uint32_t available =
            processing_flag[physical_shard] == 0U && existing < shard_capacity ? shard_capacity - existing : 0U;
        if (shard_buffer_count > 1U && write_buffer_index != nullptr) {
            const std::uint32_t current_buffer = write_buffer_index[shard] % shard_buffer_count;
            std::uint32_t best_buffer = current_buffer;
            std::uint32_t best_physical_shard = shard * shard_buffer_count + current_buffer;
            std::uint32_t best_existing = clean_count[best_physical_shard] + dirty_count[best_physical_shard];
            std::uint32_t best_available =
                processing_flag[best_physical_shard] == 0U && best_existing < shard_capacity ?
                shard_capacity - best_existing :
                0U;
            for (std::uint32_t step = 0; step < shard_buffer_count; ++step) {
                const std::uint32_t candidate_buffer = (current_buffer + step) % shard_buffer_count;
                const std::uint32_t candidate_physical_shard =
                    shard * shard_buffer_count + candidate_buffer;
                const std::uint32_t candidate_existing =
                    clean_count[candidate_physical_shard] + dirty_count[candidate_physical_shard];
                const std::uint32_t candidate_available =
                    processing_flag[candidate_physical_shard] == 0U && candidate_existing < shard_capacity ?
                    shard_capacity - candidate_existing :
                    0U;
                if (candidate_available >= raw_count) {
                    best_buffer = candidate_buffer;
                    best_physical_shard = candidate_physical_shard;
                    best_existing = candidate_existing;
                    best_available = candidate_available;
                    break;
                }
                if (candidate_available > best_available) {
                    best_buffer = candidate_buffer;
                    best_physical_shard = candidate_physical_shard;
                    best_existing = candidate_existing;
                    best_available = candidate_available;
                }
            }
            write_buffer_index[shard] = best_buffer;
            physical_shard = best_physical_shard;
            existing = best_existing;
            available = best_available;
        }
        shard_offsets[shard] = source_running;
        const std::uint32_t write_count = raw_count < available ? raw_count : available;
        shard_counts[shard] = write_count;
        const std::uint32_t spill_count = raw_count - write_count;
        spill_counts[shard] = spill_count;
        spill_offsets[shard] = spill_running;
        const std::uint64_t spill_end =
            static_cast<std::uint64_t>(spill_running) + static_cast<std::uint64_t>(spill_count);
        const bool double_buffer_overflow = shard_buffer_count > 1U && spill_count != 0U;
        const bool spill_overflow = shard_buffer_count <= 1U && spill_count != 0U && spill_end > global_spill_capacity;
        if (fatal_error_flag != nullptr && fatal_error_trace != nullptr &&
            (double_buffer_overflow || spill_overflow) && *fatal_error_flag == 0U) {
            const std::uint32_t code =
                double_buffer_overflow ? STREAM_FATAL_STREAM3_DOUBLE_BUFFER_OVERFLOW :
                STREAM_FATAL_STREAM3_SPILL_OVERFLOW;
            *fatal_error_flag = code;
            fatal_error_trace[FatalTraceCode] = code;
            fatal_error_trace[FatalTraceShard] = physical_shard;
            fatal_error_trace[FatalTraceGroup] = group;
            fatal_error_trace[FatalTraceExisting] = existing;
            fatal_error_trace[FatalTraceAvailable] = available;
            fatal_error_trace[FatalTraceRawCount] = raw_count;
            fatal_error_trace[FatalTraceWriteCount] = write_count;
            fatal_error_trace[FatalTraceSpillCount] = spill_count;
            fatal_error_trace[FatalTraceSpillOffset] = spill_running;
            fatal_error_trace[FatalTraceSpillEnd] = spill_end;
            fatal_error_trace[FatalTraceSpillCapacity] = global_spill_capacity;
            fatal_error_trace[FatalTraceCleanCount] = clean_count[physical_shard];
            fatal_error_trace[FatalTraceDirtyCount] = dirty_count[physical_shard];
            fatal_error_trace[FatalTraceProcessingFlag] = processing_flag[physical_shard];
            fatal_error_trace[FatalTraceShardCapacity] = shard_capacity;
            fatal_error_trace[FatalTraceStream4Batch] = stream4_batch_candidates;
            fatal_error_trace[FatalTraceAppendToActiveSpill] = append_to_active_spill;
        }
        spill_running = spill_end > static_cast<std::uint64_t>(UINT32_MAX) ? UINT32_MAX : static_cast<std::uint32_t>(spill_end);
        source_running += raw_count;
    }
    if (shard_buffer_count <= 1U) {
        global_spill_count[spill_index] = spill_running < global_spill_capacity ? spill_running : global_spill_capacity;
    }
}

__global__ void stream3_partition_scatter_kernel(
    const CandidateMeta* partition_val_sorted,
    const std::uint32_t* unique_shard,
    const std::uint32_t* unique_counts,
    const std::uint32_t* unique_count,
    CandidateMeta* survivor_shard,
    const std::uint32_t* clean_count,
    std::uint32_t* dirty_count,
    CandidateMeta* global_spill_buffer_a,
    CandidateMeta* global_spill_buffer_b,
    const std::uint32_t* global_spill_active_index,
    const std::uint32_t* write_buffer_index,
    const std::uint32_t* shard_counts,
    const std::uint32_t* shard_offsets,
    const std::uint32_t* spill_offsets,
    std::uint32_t shard_count,
    std::uint32_t shard_buffer_count,
    std::uint32_t shard_capacity_candidates,
    std::uint32_t stream4_batch_candidates,
    std::uint32_t global_spill_capacity,
    std::uint32_t append_to_active_spill,
    const std::uint32_t* fatal_error_flag) {
    const std::uint32_t group = blockIdx.x;
    const std::uint32_t groups = *unique_count;
    if (fatal_error_flag != nullptr && *fatal_error_flag != 0U) {
        return;
    }
    if (group >= groups) {
        return;
    }
    const std::uint32_t shard = unique_shard[group];
    if (shard >= shard_count) {
        return;
    }
    const std::uint32_t buffer = shard_buffer_count <= 1U ? 0U :
        (write_buffer_index[shard] % shard_buffer_count);
    const std::uint32_t physical_shard = shard * shard_buffer_count + buffer;
    const std::uint32_t raw_count = unique_counts[group];
    const std::uint32_t source_base = shard_offsets[shard];
    const std::uint32_t write_limit = shard_counts[shard];
    const std::uint32_t shard_capacity = shard_capacity_candidates;
    const std::uint32_t dirty_base = clean_count[physical_shard] + dirty_count[physical_shard];
    const std::uint32_t spill_base = spill_offsets[shard];
    const std::uint32_t active = *global_spill_active_index & 1U;
    CandidateMeta* global_spill_buffer =
        append_to_active_spill != 0U ?
        (active == 0U ? global_spill_buffer_a : global_spill_buffer_b) :
        (active == 0U ? global_spill_buffer_b : global_spill_buffer_a);
    for (std::uint32_t local = threadIdx.x; local < raw_count; local += blockDim.x) {
        const CandidateMeta candidate = partition_val_sorted[source_base + local];
        if (local < write_limit) {
            survivor_shard[static_cast<std::uint64_t>(physical_shard) * shard_capacity + dirty_base + local] = candidate;
        } else {
            const std::uint32_t spill_idx = spill_base + local - write_limit;
            if (shard_buffer_count <= 1U && spill_idx < global_spill_capacity) {
                global_spill_buffer[spill_idx] = candidate;
            }
        }
    }
    if (threadIdx.x == 0) {
        dirty_count[physical_shard] += write_limit;
    }
}

__global__ void stream3_finalize_drain_counts_kernel(
    std::uint32_t* global_spill_count,
    std::uint32_t* global_spill_active_index) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const std::uint32_t read = *global_spill_active_index & 1U;
    const std::uint32_t write = read ^ 1U;
    global_spill_count[read] = 0;
    *global_spill_active_index = write;
}

__global__ void stream3_build_ready_shard_queue_kernel(
    const std::uint32_t* clean_count,
    const std::uint32_t* dirty_count,
    std::uint32_t* processing_flag,
    std::uint32_t* write_buffer_index,
    std::uint32_t* ready_flag,
    std::uint32_t* ready_shard_list,
    std::uint32_t* ready_count,
    std::uint32_t shard_count,
    std::uint32_t shard_buffer_count,
    std::uint32_t shard_capacity_candidates,
    std::uint32_t stream3_batch_candidates,
    std::uint32_t stream4_trigger_candidates,
    std::uint32_t force_dirty_flush,
    std::uint32_t force_clean_flush) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    std::uint32_t out = 0;
    const std::uint32_t shard_capacity = shard_capacity_candidates;
    const std::uint32_t storage_shard_count = shard_count * shard_buffer_count;
    const std::uint32_t average_shard_write =
        shard_count == 0U ? stream3_batch_candidates :
        (stream3_batch_candidates + shard_count - 1U) / shard_count;
    const std::uint32_t write_margin = (average_shard_write + 3U) / 4U;
    const std::uint32_t unclamped_write_reserve =
        average_shard_write > UINT32_MAX - write_margin ? UINT32_MAX : average_shard_write + write_margin;
    const std::uint32_t write_reserve =
        unclamped_write_reserve < shard_capacity ? unclamped_write_reserve : shard_capacity;
    const std::uint32_t clean_ready_threshold =
        write_reserve < shard_capacity ? shard_capacity - write_reserve : 0U;
    for (std::uint32_t shard = 0; shard < storage_shard_count; ++shard) {
        ready_flag[shard] = 0;
        const std::uint32_t clean = clean_count[shard];
        const std::uint32_t dirty = dirty_count[shard];
        const std::uint32_t total = clean + dirty;
        const bool near_capacity = total >= clean_ready_threshold;
        const bool dirty_ready =
            dirty != 0U &&
            (force_dirty_flush != 0U || dirty >= stream4_trigger_candidates || near_capacity);
        const bool clean_ready =
            dirty == 0U && clean != 0U && (force_clean_flush != 0U || clean >= clean_ready_threshold);
        const bool ready =
            processing_flag[shard] == 0U &&
            (dirty_ready || clean_ready);
        if (ready) {
            processing_flag[shard] = 1;
            ready_flag[shard] = 1;
            ready_shard_list[out] = shard;
            ++out;
            if (shard_buffer_count > 1U) {
                const std::uint32_t logical_shard = shard / shard_buffer_count;
                const std::uint32_t current_buffer = write_buffer_index[logical_shard] % shard_buffer_count;
                if (shard == logical_shard * shard_buffer_count + current_buffer) {
                    for (std::uint32_t step = 1U; step < shard_buffer_count; ++step) {
                        const std::uint32_t candidate_buffer = (current_buffer + step) % shard_buffer_count;
                        const std::uint32_t candidate_shard =
                            logical_shard * shard_buffer_count + candidate_buffer;
                        const std::uint32_t candidate_total =
                            clean_count[candidate_shard] + dirty_count[candidate_shard];
                        if (processing_flag[candidate_shard] == 0U && candidate_total < shard_capacity) {
                            write_buffer_index[logical_shard] = candidate_buffer;
                            break;
                        }
                    }
                }
            }
        }
    }
    *ready_count = out;
}

} // namespace

void stream3_pack_threshold_cuda(
    const std::uint32_t* score_ring,
    const Hash128* hash_ring,
    const std::uint64_t*,
    const std::uint32_t* count,
    Hash128* compact_key,
    std::uint64_t* compact_val,
    Hash128* reduce_key_scratch,
    std::uint64_t* reduce_val_scratch,
    Hash128* unique_key,
    std::uint64_t* unique_val,
    std::uint32_t* keep_flags,
    std::uint32_t* block_counts,
    std::uint32_t* block_offsets,
    std::uint32_t* unique_count,
    void* cub_temp_storage,
    std::size_t cub_temp_storage_bytes,
    std::uint32_t current_threshold,
    std::uint32_t b_micro,
    std::uint32_t stream3_batch_candidates,
    cudaStream_t stream) {
    NvtxRange range("Stream3_threshold_compact_cub_sort_reduce_launch");
    if (cub_temp_storage == nullptr || cub_temp_storage_bytes == 0) {
        throw std::invalid_argument("stream3 CUB fixed temp storage is required");
    }
    const std::uint32_t block_size = 256;
    const std::uint32_t block_count = (stream3_batch_candidates + block_size - 1U) / block_size;
    const dim3 block(block_size);
    const dim3 grid(block_count);
    stream3_mark_counts_kernel<<<grid, block, 0, stream>>>(
        score_ring,
        count,
        keep_flags,
        block_counts,
        nullptr,
        current_threshold,
        b_micro,
        stream3_batch_candidates);
    stream3_scan_block_counts(
        block_counts,
        block_offsets,
        unique_count,
        block_count,
        cub_temp_storage,
        cub_temp_storage_bytes,
        stream);
    stream3_compact_kernel<<<grid, block, 0, stream>>>(
        score_ring,
        hash_ring,
        keep_flags,
        block_offsets,
        compact_key,
        compact_val,
        stream3_batch_candidates);
    stream3_fill_sort_tail_kernel<<<grid, block, 0, stream>>>(
        compact_key,
        compact_val,
        unique_count,
        stream3_batch_candidates);
    std::size_t sort_temp_bytes = cub_temp_storage_bytes;
    check_cub(
        cub::DeviceMergeSort::SortPairs(
            cub_temp_storage,
            sort_temp_bytes,
            compact_key,
            compact_val,
            stream3_batch_candidates,
            Stream3HashLess{},
            stream),
        "cub::DeviceMergeSort::SortPairs stream3");
    std::size_t reduce_temp_bytes = cub_temp_storage_bytes;
    check_cub(
        cub::DeviceReduce::ReduceByKey(
            cub_temp_storage,
            reduce_temp_bytes,
            compact_key,
            reduce_key_scratch,
            compact_val,
            reduce_val_scratch,
            unique_count,
            Stream3MinValue{},
            stream3_batch_candidates,
            stream),
        "cub::DeviceReduce::ReduceByKey stream3");
    stream3_mark_valid_unique_counts_kernel<<<grid, block, 0, stream>>>(
        reduce_val_scratch,
        keep_flags,
        block_counts,
        unique_count,
        stream3_batch_candidates);
    stream3_scan_block_counts(
        block_counts,
        block_offsets,
        unique_count,
        block_count,
        cub_temp_storage,
        cub_temp_storage_bytes,
        stream);
    stream3_compact_valid_unique_kernel<<<grid, block, 0, stream>>>(
        reduce_key_scratch,
        reduce_val_scratch,
        keep_flags,
        block_offsets,
        unique_key,
        unique_val,
        stream3_batch_candidates);
}

void stream3_pack_threshold_device_threshold_cuda(
    const std::uint32_t* score_ring,
    const Hash128* hash_ring,
    const std::uint64_t* parent_base,
    const std::uint32_t* count,
    Hash128* compact_key,
    std::uint64_t* compact_val,
    Hash128* reduce_key_scratch,
    std::uint64_t* reduce_val_scratch,
    Hash128* unique_key,
    std::uint64_t* unique_val,
    std::uint32_t* keep_flags,
    std::uint32_t* block_counts,
    std::uint32_t* block_offsets,
    std::uint32_t* unique_count,
    void* cub_temp_storage,
    std::size_t cub_temp_storage_bytes,
    const std::uint32_t* current_threshold,
    std::uint32_t b_micro,
    std::uint32_t stream3_batch_candidates,
    cudaStream_t stream) {
    NvtxRange range("Stream3_threshold_compact_cub_sort_reduce_device_threshold_launch");
    if (cub_temp_storage == nullptr || cub_temp_storage_bytes == 0) {
        throw std::invalid_argument("stream3 CUB fixed temp storage is required");
    }
    const std::uint32_t block_size = 256;
    const std::uint32_t block_count = (stream3_batch_candidates + block_size - 1U) / block_size;
    const dim3 block(block_size);
    const dim3 grid(block_count);
    stream3_mark_counts_kernel<<<grid, block, 0, stream>>>(
        score_ring,
        count,
        keep_flags,
        block_counts,
        current_threshold,
        UINT32_THRESHOLD_MAX,
        b_micro,
        stream3_batch_candidates);
    stream3_scan_block_counts(
        block_counts,
        block_offsets,
        unique_count,
        block_count,
        cub_temp_storage,
        cub_temp_storage_bytes,
        stream);
    stream3_compact_kernel<<<grid, block, 0, stream>>>(
        score_ring,
        hash_ring,
        keep_flags,
        block_offsets,
        compact_key,
        compact_val,
        stream3_batch_candidates);
    stream3_fill_sort_tail_kernel<<<grid, block, 0, stream>>>(
        compact_key,
        compact_val,
        unique_count,
        stream3_batch_candidates);
    std::size_t sort_temp_bytes = cub_temp_storage_bytes;
    check_cub(
        cub::DeviceMergeSort::SortPairs(
            cub_temp_storage,
            sort_temp_bytes,
            compact_key,
            compact_val,
            stream3_batch_candidates,
            Stream3HashLess{},
            stream),
        "cub::DeviceMergeSort::SortPairs stream3");
    std::size_t reduce_temp_bytes = cub_temp_storage_bytes;
    check_cub(
        cub::DeviceReduce::ReduceByKey(
            cub_temp_storage,
            reduce_temp_bytes,
            compact_key,
            reduce_key_scratch,
            compact_val,
            reduce_val_scratch,
            unique_count,
            Stream3MinValue{},
            stream3_batch_candidates,
            stream),
        "cub::DeviceReduce::ReduceByKey stream3");
    stream3_mark_valid_unique_counts_kernel<<<grid, block, 0, stream>>>(
        reduce_val_scratch,
        keep_flags,
        block_counts,
        unique_count,
        stream3_batch_candidates);
    stream3_scan_block_counts(
        block_counts,
        block_offsets,
        unique_count,
        block_count,
        cub_temp_storage,
        cub_temp_storage_bytes,
        stream);
    stream3_compact_valid_unique_kernel<<<grid, block, 0, stream>>>(
        reduce_key_scratch,
        reduce_val_scratch,
        keep_flags,
        block_offsets,
        unique_key,
        unique_val,
        stream3_batch_candidates);
    (void)parent_base;
}

void stream3_restore_owner_split_cuda(
    const Hash128* unique_key,
    const std::uint64_t* unique_val,
    const std::uint32_t* unique_count,
    const std::uint64_t* parent_base,
    CandidateMeta* local_pending_buffer,
    std::uint32_t* local_pending_count,
    CandidateMeta* remote_send_buffer,
    std::uint32_t* send_count,
    std::uint32_t* send_offset,
    std::uint32_t* owner_scratch,
    std::uint16_t local_rank,
    std::uint32_t world_size,
    std::uint32_t b_micro,
    std::uint32_t max_candidates,
    cudaStream_t stream) {
    NvtxRange range("Stream3_restore_owner_split_launch");
    const std::uint32_t block_size = 256;
    const dim3 block(block_size);
    const dim3 grid((max_candidates + block_size - 1U) / block_size);
    if (world_size == 1U) {
        stream3_restore_single_owner_kernel<<<grid, block, 0, stream>>>(
            unique_key,
            unique_val,
            unique_count,
            parent_base,
            local_pending_buffer,
            local_pending_count,
            send_count,
            send_offset,
            local_rank,
            b_micro);
        return;
    }
    stream3_restore_owner_split_kernel<<<grid, block, 0, stream>>>(
        unique_key,
        unique_val,
        unique_count,
        parent_base,
        local_pending_buffer,
        owner_scratch,
        local_rank,
        world_size,
        b_micro);
    stream3_count_owner_kernel<<<world_size, block, 0, stream>>>(
        owner_scratch,
        unique_count,
        local_pending_count,
        send_count,
        local_rank,
        world_size);
    stream3_scan_send_counts_kernel<<<1, 1, 0, stream>>>(send_count, send_offset, world_size);
    stream3_scatter_owner_kernel<<<world_size, block, 0, stream>>>(
        local_pending_buffer,
        owner_scratch,
        unique_count,
        remote_send_buffer,
        send_offset,
        local_rank,
        world_size);
}

namespace {

void stream3_partition_sort_reduce_scatter(
    std::uint32_t* partition_key_a,
    std::uint32_t* partition_key_b,
    CandidateMeta* partition_val_a,
    CandidateMeta* partition_val_b,
    std::uint32_t* partition_unique_shard,
    std::uint32_t* partition_unique_counts,
    std::uint32_t* partition_unique_count,
    void* cub_temp_storage,
    std::size_t cub_temp_storage_bytes,
    std::uint32_t max_candidates,
    CandidateMeta* survivor_shard,
    std::uint32_t* clean_count,
    std::uint32_t* dirty_count,
    const std::uint32_t* processing_flag,
    CandidateMeta* global_spill_buffer_a,
    CandidateMeta* global_spill_buffer_b,
    std::uint32_t* global_spill_count,
    const std::uint32_t* global_spill_active_index,
    std::uint32_t* write_buffer_index,
    std::uint32_t* shard_counts,
    std::uint32_t* shard_offsets,
    std::uint32_t* spill_counts,
    std::uint32_t* spill_offsets,
    std::uint32_t shard_count,
    std::uint32_t shard_buffer_count,
    std::uint32_t shard_capacity_candidates,
    std::uint32_t stream4_batch_candidates,
    std::uint32_t global_spill_capacity,
    bool append_to_active_spill,
    std::uint32_t* fatal_error_flag,
    std::uint64_t* fatal_error_trace,
    cudaStream_t stream) {
    if (cub_temp_storage == nullptr || cub_temp_storage_bytes == 0) {
        throw std::invalid_argument("stream3 partition CUB fixed temp storage is required");
    }
    std::size_t sort_temp_bytes = cub_temp_storage_bytes;
    check_cub(
        cub::DeviceRadixSort::SortPairs(
            cub_temp_storage,
            sort_temp_bytes,
            partition_key_a,
            partition_key_b,
            partition_val_a,
            partition_val_b,
            static_cast<int>(max_candidates),
            0,
            static_cast<int>(stream3_shard_key_bits(shard_count)),
            stream),
        "cub::DeviceRadixSort::SortPairs stream3 partition");
    cub::ConstantInputIterator<std::uint32_t> ones(1U);
    std::size_t reduce_temp_bytes = cub_temp_storage_bytes;
    check_cub(
        cub::DeviceReduce::ReduceByKey(
            cub_temp_storage,
            reduce_temp_bytes,
            partition_key_b,
            partition_unique_shard,
            ones,
            partition_unique_counts,
            partition_unique_count,
            cub::Sum{},
            static_cast<int>(max_candidates),
            stream),
        "cub::DeviceReduce::ReduceByKey stream3 partition");
    stream3_prepare_partition_counts_kernel<<<1, 1, 0, stream>>>(
        partition_unique_shard,
        partition_unique_counts,
        partition_unique_count,
        clean_count,
        dirty_count,
        processing_flag,
        shard_counts,
        shard_offsets,
        spill_counts,
        spill_offsets,
        global_spill_count,
        global_spill_active_index,
        write_buffer_index,
        shard_count,
        shard_buffer_count,
        shard_capacity_candidates,
        stream4_batch_candidates,
        global_spill_capacity,
        append_to_active_spill ? 1U : 0U,
        fatal_error_flag,
        fatal_error_trace);
    stream3_partition_scatter_kernel<<<shard_count + 1U, 256, 0, stream>>>(
        partition_val_b,
        partition_unique_shard,
        partition_unique_counts,
        partition_unique_count,
        survivor_shard,
        clean_count,
        dirty_count,
        global_spill_buffer_a,
        global_spill_buffer_b,
        global_spill_active_index,
        write_buffer_index,
        shard_counts,
        shard_offsets,
        spill_offsets,
        shard_count,
        shard_buffer_count,
        shard_capacity_candidates,
        stream4_batch_candidates,
        global_spill_capacity,
        append_to_active_spill ? 1U : 0U,
        fatal_error_flag);
}

} // namespace

void stream3_collect_local_pending_cuda(
    const CandidateMeta* local_pending_buffer,
    const std::uint32_t* local_pending_count,
    CandidateMeta* survivor_shard,
    std::uint32_t* clean_count,
    std::uint32_t* dirty_count,
    const std::uint32_t* processing_flag,
    CandidateMeta* global_spill_buffer_a,
    CandidateMeta* global_spill_buffer_b,
    std::uint32_t* global_spill_count,
    std::uint32_t* global_spill_active_index,
    std::uint32_t* write_buffer_index,
    std::uint32_t* shard_counts,
    std::uint32_t* shard_offsets,
    std::uint32_t* spill_counts,
    std::uint32_t* spill_offsets,
    std::uint32_t* partition_key_a,
    std::uint32_t* partition_key_b,
    CandidateMeta* partition_val_a,
    CandidateMeta* partition_val_b,
    std::uint32_t* partition_unique_shard,
    std::uint32_t* partition_unique_counts,
    std::uint32_t* partition_unique_count,
    void* cub_temp_storage,
    std::size_t cub_temp_storage_bytes,
    std::uint32_t max_candidates,
    std::uint32_t shard_count,
    std::uint32_t shard_buffer_count,
    std::uint32_t shard_capacity_candidates,
    std::uint32_t stream4_batch_candidates,
    std::uint32_t global_spill_capacity,
    cudaStream_t stream,
    std::uint32_t* fatal_error_flag,
    std::uint64_t* fatal_error_trace) {
    NvtxRange range("Stream3_collect_local_pending_partition_launch");
    const dim3 block(256);
    const dim3 grid((max_candidates + block.x - 1U) / block.x);
    stream3_partition_fill_input_kernel<<<grid, block, 0, stream>>>(
        local_pending_buffer,
        local_pending_count,
        partition_key_a,
        partition_val_a,
        max_candidates,
        shard_count);
    stream3_partition_sort_reduce_scatter(
        partition_key_a,
        partition_key_b,
        partition_val_a,
        partition_val_b,
        partition_unique_shard,
        partition_unique_counts,
        partition_unique_count,
        cub_temp_storage,
        cub_temp_storage_bytes,
        max_candidates,
        survivor_shard,
        clean_count,
        dirty_count,
        processing_flag,
        global_spill_buffer_a,
        global_spill_buffer_b,
        global_spill_count,
        global_spill_active_index,
        write_buffer_index,
        shard_counts,
        shard_offsets,
        spill_counts,
        spill_offsets,
        shard_count,
        shard_buffer_count,
        shard_capacity_candidates,
        stream4_batch_candidates,
        global_spill_capacity,
        true,
        fatal_error_flag,
        fatal_error_trace,
        stream);
}

void stream3_restore_collect_single_owner_cuda(
    const Hash128* unique_key,
    const std::uint64_t* unique_val,
    const std::uint32_t* unique_count,
    const std::uint64_t* parent_base,
    std::uint32_t* local_pending_count,
    std::uint32_t* send_count,
    std::uint32_t* send_offset,
    CandidateMeta* survivor_shard,
    std::uint32_t* clean_count,
    std::uint32_t* dirty_count,
    const std::uint32_t* processing_flag,
    CandidateMeta* global_spill_buffer_a,
    CandidateMeta* global_spill_buffer_b,
    std::uint32_t* global_spill_count,
    std::uint32_t* global_spill_active_index,
    std::uint32_t* write_buffer_index,
    std::uint32_t* shard_counts,
    std::uint32_t* shard_offsets,
    std::uint32_t* spill_counts,
    std::uint32_t* spill_offsets,
    std::uint32_t* partition_key_a,
    std::uint32_t* partition_key_b,
    CandidateMeta* partition_val_a,
    CandidateMeta* partition_val_b,
    std::uint32_t* partition_unique_shard,
    std::uint32_t* partition_unique_counts,
    std::uint32_t* partition_unique_count,
    void* cub_temp_storage,
    std::size_t cub_temp_storage_bytes,
    std::uint16_t local_rank,
    std::uint32_t b_micro,
    std::uint32_t max_candidates,
    std::uint32_t shard_count,
    std::uint32_t shard_buffer_count,
    std::uint32_t shard_capacity_candidates,
    std::uint32_t stream4_batch_candidates,
    std::uint32_t global_spill_capacity,
    cudaStream_t stream,
    std::uint32_t* fatal_error_flag,
    std::uint64_t* fatal_error_trace) {
    NvtxRange range("Stream3_restore_collect_single_owner_partition_launch");
    const dim3 block(256);
    const dim3 grid((max_candidates + block.x - 1U) / block.x);
    stream3_restore_single_owner_partition_kernel<<<grid, block, 0, stream>>>(
        unique_key,
        unique_val,
        unique_count,
        parent_base,
        local_pending_count,
        send_count,
        send_offset,
        partition_key_a,
        partition_val_a,
        local_rank,
        b_micro,
        max_candidates,
        shard_count);
    stream3_partition_sort_reduce_scatter(
        partition_key_a,
        partition_key_b,
        partition_val_a,
        partition_val_b,
        partition_unique_shard,
        partition_unique_counts,
        partition_unique_count,
        cub_temp_storage,
        cub_temp_storage_bytes,
        max_candidates,
        survivor_shard,
        clean_count,
        dirty_count,
        processing_flag,
        global_spill_buffer_a,
        global_spill_buffer_b,
        global_spill_count,
        global_spill_active_index,
        write_buffer_index,
        shard_counts,
        shard_offsets,
        spill_counts,
        spill_offsets,
        shard_count,
        shard_buffer_count,
        shard_capacity_candidates,
        stream4_batch_candidates,
        global_spill_capacity,
        true,
        fatal_error_flag,
        fatal_error_trace,
        stream);
}

void stream3_drain_global_spill_cuda(
    CandidateMeta* global_spill_buffer_a,
    CandidateMeta* global_spill_buffer_b,
    std::uint32_t* global_spill_count,
    std::uint32_t* global_spill_active_index,
    CandidateMeta* survivor_shard,
    std::uint32_t* clean_count,
    std::uint32_t* dirty_count,
    const std::uint32_t* processing_flag,
    std::uint32_t* shard_counts,
    std::uint32_t* shard_offsets,
    std::uint32_t* spill_counts,
    std::uint32_t* spill_offsets,
    std::uint32_t* partition_key_a,
    std::uint32_t* partition_key_b,
    CandidateMeta* partition_val_a,
    CandidateMeta* partition_val_b,
    std::uint32_t* partition_unique_shard,
    std::uint32_t* partition_unique_counts,
    std::uint32_t* partition_unique_count,
    void* cub_temp_storage,
    std::size_t cub_temp_storage_bytes,
    std::uint32_t shard_count,
    std::uint32_t global_spill_capacity,
    std::uint32_t shard_capacity_candidates,
    std::uint32_t stream4_batch_candidates,
    cudaStream_t stream,
    std::uint32_t* fatal_error_flag,
    std::uint64_t* fatal_error_trace) {
    NvtxRange range("Stream3_drain_global_spill_partition_launch");
    const dim3 block(256);
    const dim3 grid((global_spill_capacity + block.x - 1U) / block.x);
    stream3_partition_fill_active_spill_kernel<<<grid, block, 0, stream>>>(
        global_spill_buffer_a,
        global_spill_buffer_b,
        global_spill_count,
        global_spill_active_index,
        partition_key_a,
        partition_val_a,
        global_spill_capacity,
        shard_count);
    stream3_partition_sort_reduce_scatter(
        partition_key_a,
        partition_key_b,
        partition_val_a,
        partition_val_b,
        partition_unique_shard,
        partition_unique_counts,
        partition_unique_count,
        cub_temp_storage,
        cub_temp_storage_bytes,
        global_spill_capacity,
        survivor_shard,
        clean_count,
        dirty_count,
        processing_flag,
        global_spill_buffer_a,
        global_spill_buffer_b,
        global_spill_count,
        global_spill_active_index,
        nullptr,
        shard_counts,
        shard_offsets,
        spill_counts,
        spill_offsets,
        shard_count,
        1U,
        shard_capacity_candidates,
        stream4_batch_candidates,
        global_spill_capacity,
        false,
        fatal_error_flag,
        fatal_error_trace,
        stream);
    stream3_finalize_drain_counts_kernel<<<1, 1, 0, stream>>>(
        global_spill_count,
        global_spill_active_index);
}

void stream3_collect_remote_recv_cuda(
    const CandidateMeta* remote_recv_buffer,
    const std::uint32_t* recv_count,
    const std::uint32_t* recv_offset,
    CandidateMeta* survivor_shard,
    std::uint32_t* clean_count,
    std::uint32_t* dirty_count,
    const std::uint32_t* processing_flag,
    CandidateMeta* global_spill_buffer_a,
    CandidateMeta* global_spill_buffer_b,
    std::uint32_t* global_spill_count,
    std::uint32_t* global_spill_active_index,
    std::uint32_t* write_buffer_index,
    std::uint32_t* shard_counts,
    std::uint32_t* shard_offsets,
    std::uint32_t* spill_counts,
    std::uint32_t* spill_offsets,
    std::uint32_t* partition_key_a,
    std::uint32_t* partition_key_b,
    CandidateMeta* partition_val_a,
    CandidateMeta* partition_val_b,
    std::uint32_t* partition_unique_shard,
    std::uint32_t* partition_unique_counts,
    std::uint32_t* partition_unique_count,
    void* cub_temp_storage,
    std::size_t cub_temp_storage_bytes,
    std::uint32_t max_candidates,
    std::uint32_t world_size,
    std::uint32_t shard_count,
    std::uint32_t shard_buffer_count,
    std::uint32_t shard_capacity_candidates,
    std::uint32_t stream4_batch_candidates,
    std::uint32_t global_spill_capacity,
    cudaStream_t stream,
    std::uint32_t* fatal_error_flag,
    std::uint64_t* fatal_error_trace) {
    NvtxRange range("Stream3_collect_remote_recv_partition_launch");
    const dim3 block(256);
    const dim3 grid((max_candidates + block.x - 1U) / block.x);
    stream3_partition_fill_input_kernel<<<grid, block, 0, stream>>>(
        remote_recv_buffer,
        recv_offset + world_size,
        partition_key_a,
        partition_val_a,
        max_candidates,
        shard_count);
    stream3_partition_sort_reduce_scatter(
        partition_key_a,
        partition_key_b,
        partition_val_a,
        partition_val_b,
        partition_unique_shard,
        partition_unique_counts,
        partition_unique_count,
        cub_temp_storage,
        cub_temp_storage_bytes,
        max_candidates,
        survivor_shard,
        clean_count,
        dirty_count,
        processing_flag,
        global_spill_buffer_a,
        global_spill_buffer_b,
        global_spill_count,
        global_spill_active_index,
        write_buffer_index,
        shard_counts,
        shard_offsets,
        spill_counts,
        spill_offsets,
        shard_count,
        shard_buffer_count,
        shard_capacity_candidates,
        stream4_batch_candidates,
        global_spill_capacity,
        true,
        fatal_error_flag,
        fatal_error_trace,
        stream);
    (void)recv_count;
}

void stream3_build_ready_shard_queue_cuda(
    const std::uint32_t* clean_count,
    const std::uint32_t* dirty_count,
    std::uint32_t* processing_flag,
    std::uint32_t* write_buffer_index,
    std::uint32_t* ready_flag,
    std::uint32_t* ready_shard_list,
    std::uint32_t* ready_count,
    std::uint32_t shard_count,
    std::uint32_t shard_buffer_count,
    std::uint32_t shard_capacity_candidates,
    std::uint32_t stream3_batch_candidates,
    std::uint32_t stream4_trigger_candidates,
    bool force_dirty_flush,
    bool force_clean_flush,
    cudaStream_t stream) {
    NvtxRange range("Stream3_build_ready_shard_queue_launch");
    stream3_build_ready_shard_queue_kernel<<<1, 1, 0, stream>>>(
        clean_count,
        dirty_count,
        processing_flag,
        write_buffer_index,
        ready_flag,
        ready_shard_list,
        ready_count,
        shard_count,
        shard_buffer_count,
        shard_capacity_candidates,
        stream3_batch_candidates,
        stream4_trigger_candidates,
        force_dirty_flush ? 1U : 0U,
        force_clean_flush ? 1U : 0U);
}

} // namespace beam
