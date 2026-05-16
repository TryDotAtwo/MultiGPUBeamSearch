#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cuda/std/tuple>
#include <stdint.h>

#include "beam_types.hpp"

namespace beam_v6 {

struct Hash128Key {
    uint64_t hi;
    uint64_t lo;
};

struct Hash128KeyDecomposer {
    __host__ __device__ auto operator()(Hash128Key& key) const {
        return ::cuda::std::tie(key.hi, key.lo);
    }
};

__host__ __device__ __forceinline__ bool operator<(const Hash128Key& a, const Hash128Key& b) {
    return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
}

__host__ __device__ __forceinline__ bool operator==(const Hash128Key& a, const Hash128Key& b) {
    return a.hi == b.hi && a.lo == b.lo;
}

__device__ __forceinline__ uint32_t owner_from_hash128(uint64_t hi, uint64_t lo, uint32_t world_size) {
    if (world_size <= 1) {
        return 0;
    }
    const uint64_t mixed = hi ^ (lo * 0x9e3779b97f4a7c15ULL);
    return uint32_t(mixed % uint64_t(world_size));
}

extern "C" __global__ void kernel_v6_stream3_pack_threshold_compact(
    const uint32_t* __restrict__ score_ring,
    const Hash128* __restrict__ hash_ring,
    const uint64_t* __restrict__ parent_base,
    const uint32_t* __restrict__ count,
    Hash128Key* __restrict__ stream3_key_a,
    uint64_t* __restrict__ stream3_val_a,
    uint32_t* __restrict__ compact_count,
    uint32_t current_threshold,
    uint32_t ring,
    uint32_t ring_slot_count,
    uint32_t b_micro,
    uint32_t stream3_batch_candidates
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= stream3_batch_candidates) {
        return;
    }

    const uint32_t candidates_per_slot = b_micro * MOVE_COUNT;
    const uint32_t ring_slot = i / candidates_per_slot;
    const uint32_t local_i = i - ring_slot * candidates_per_slot;
    const uint32_t parent_local = local_i / MOVE_COUNT;
    const uint32_t slot_idx = ring * ring_slot_count + ring_slot;
    if (parent_local >= count[slot_idx]) {
        return;
    }

    const uint64_t ring_offset = uint64_t(slot_idx) * uint64_t(candidates_per_slot) + uint64_t(local_i);
    const uint32_t score_key = score_ring[ring_offset];
    if (score_key > current_threshold) {
        return;
    }

    const uint32_t compact_i = atomicAdd(compact_count, 1u);
    const Hash128 hash = hash_ring[ring_offset];
    stream3_key_a[compact_i] = Hash128Key{hash.hi, hash.lo};
    stream3_val_a[compact_i] = pack_stream3_val(score_key, i);
}

extern "C" __global__ void kernel_v6_stream3_dedup_sorted(
    const Hash128Key* __restrict__ sorted_key,
    const uint64_t* __restrict__ sorted_val,
    Hash128* __restrict__ unique_key,
    uint64_t* __restrict__ unique_val,
    uint32_t* __restrict__ unique_count,
    uint32_t compact_count
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= compact_count) {
        return;
    }

    const Hash128Key key = sorted_key[i];
    const uint64_t val = sorted_val[i];
    const bool starts_segment = (i == 0) || !(key == sorted_key[i - 1]);
    if (!starts_segment) {
        return;
    }

    uint64_t best = val;
    uint32_t j = i + 1;
    while (j < compact_count && sorted_key[j] == key) {
        const uint64_t candidate = sorted_val[j];
        if (candidate < best) {
            best = candidate;
        }
        ++j;
    }

    const uint32_t out_i = atomicAdd(unique_count, 1u);
    unique_key[out_i] = Hash128{key.lo, key.hi};
    unique_val[out_i] = best;
}

extern "C" __global__ void kernel_v6_stream3_restore_split(
    const Hash128* __restrict__ unique_key,
    const uint64_t* __restrict__ unique_val,
    const uint64_t* __restrict__ parent_base,
    CandidateMeta* __restrict__ local_pending_buffer,
    CandidateMeta* __restrict__ remote_send_buffer,
    uint32_t* __restrict__ local_count,
    uint32_t* __restrict__ send_count,
    uint32_t* __restrict__ send_offset,
    uint32_t unique_count,
    uint32_t local_rank,
    uint32_t world_size,
    uint32_t ring,
    uint32_t ring_slot_count,
    uint32_t b_micro
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= unique_count) {
        return;
    }

    const uint64_t val = unique_val[i];
    const uint32_t score_key = stream3_val_score_key(val);
    const uint32_t payload = stream3_val_payload_id(val);
    const uint32_t candidates_per_slot = b_micro * MOVE_COUNT;
    const uint32_t ring_slot = payload / candidates_per_slot;
    const uint32_t local_i = payload - ring_slot * candidates_per_slot;
    const uint32_t parent_local = local_i / MOVE_COUNT;
    const uint8_t move = uint8_t(local_i - parent_local * MOVE_COUNT);
    const uint32_t slot_idx = ring * ring_slot_count + ring_slot;
    const uint64_t parent_idx = parent_base[slot_idx] + uint64_t(parent_local);
    const Hash128 hash = unique_key[i];
    const uint32_t owner = owner_from_hash128(hash.hi, hash.lo, world_size);

    CandidateMeta meta;
    meta.hash = hash;
    meta.parent_idx = parent_idx;
    meta.score_key = score_key;
    meta.route_packed = pack_route(uint16_t(local_rank), uint8_t(owner), move);

    if (owner == local_rank) {
        const uint32_t out_i = atomicAdd(local_count, 1u);
        local_pending_buffer[out_i] = meta;
    } else {
        const uint32_t peer_pos = atomicAdd(send_count + owner, 1u);
        remote_send_buffer[send_offset[owner] + peer_pos] = meta;
    }
}

extern "C" __global__ void kernel_v6_stream3_scatter_remote_grouped(
    const CandidateMeta* __restrict__ remote_unsorted_buffer,
    CandidateMeta* __restrict__ remote_send_buffer,
    const uint32_t* __restrict__ send_offset,
    uint32_t* __restrict__ scatter_cursor,
    uint32_t remote_count
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= remote_count) {
        return;
    }
    const CandidateMeta meta = remote_unsorted_buffer[i];
    const uint32_t owner = unpack_owner(meta.route_packed);
    const uint32_t pos = atomicAdd(scatter_cursor + owner, 1u);
    remote_send_buffer[send_offset[owner] + pos] = meta;
}

} // namespace beam_v6

extern "C" void launch_v6_stream3_pack_threshold_compact(
    const uint32_t* score_ring,
    const beam_v6::Hash128* hash_ring,
    const uint64_t* parent_base,
    const uint32_t* count,
    beam_v6::Hash128Key* stream3_key_a,
    uint64_t* stream3_val_a,
    uint32_t* compact_count,
    uint32_t current_threshold,
    uint32_t ring,
    uint32_t ring_slot_count,
    uint32_t b_micro,
    uint32_t stream3_batch_candidates,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = int((stream3_batch_candidates + threads - 1) / threads);
    beam_v6::kernel_v6_stream3_pack_threshold_compact<<<blocks, threads, 0, stream>>>(
        score_ring,
        hash_ring,
        parent_base,
        count,
        stream3_key_a,
        stream3_val_a,
        compact_count,
        current_threshold,
        ring,
        ring_slot_count,
        b_micro,
        stream3_batch_candidates);
}

extern "C" void launch_v6_stream3_sort_pairs(
    void* temp_storage,
    size_t& temp_storage_bytes,
    const beam_v6::Hash128Key* key_in,
    beam_v6::Hash128Key* key_out,
    const uint64_t* val_in,
    uint64_t* val_out,
    int item_count,
    cudaStream_t stream
) {
    cub::DeviceRadixSort::SortPairs(
        temp_storage,
        temp_storage_bytes,
        key_in,
        key_out,
        val_in,
        val_out,
        item_count,
        beam_v6::Hash128KeyDecomposer{},
        0,
        int(sizeof(beam_v6::Hash128Key) * 8),
        stream);
}

extern "C" void launch_v6_stream3_dedup_sorted(
    const beam_v6::Hash128Key* sorted_key,
    const uint64_t* sorted_val,
    beam_v6::Hash128* unique_key,
    uint64_t* unique_val,
    uint32_t* unique_count,
    uint32_t compact_count,
    cudaStream_t stream
) {
    const int threads = 128;
    const int blocks = int((compact_count + threads - 1) / threads);
    beam_v6::kernel_v6_stream3_dedup_sorted<<<blocks, threads, 0, stream>>>(
        sorted_key,
        sorted_val,
        unique_key,
        unique_val,
        unique_count,
        compact_count);
}

extern "C" void launch_v6_stream3_restore_split(
    const beam_v6::Hash128* unique_key,
    const uint64_t* unique_val,
    const uint64_t* parent_base,
    beam_v6::CandidateMeta* local_pending_buffer,
    beam_v6::CandidateMeta* remote_send_buffer,
    uint32_t* local_count,
    uint32_t* send_count,
    uint32_t* send_offset,
    uint32_t unique_count,
    uint32_t local_rank,
    uint32_t world_size,
    uint32_t ring,
    uint32_t ring_slot_count,
    uint32_t b_micro,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = int((unique_count + threads - 1) / threads);
    beam_v6::kernel_v6_stream3_restore_split<<<blocks, threads, 0, stream>>>(
        unique_key,
        unique_val,
        parent_base,
        local_pending_buffer,
        remote_send_buffer,
        local_count,
        send_count,
        send_offset,
        unique_count,
        local_rank,
        world_size,
        ring,
        ring_slot_count,
        b_micro);
}
