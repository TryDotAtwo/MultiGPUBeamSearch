#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cuda/std/tuple>
#include <stdint.h>

#include "beam_types.hpp"

namespace beam_v6 {

struct Stream4HashKey {
    uint64_t hi;
    uint64_t lo;
};

struct Stream4HashKeyDecomposer {
    __host__ __device__ auto operator()(Stream4HashKey& key) const {
        return ::cuda::std::tie(key.hi, key.lo);
    }
};

__host__ __device__ __forceinline__ bool operator==(const Stream4HashKey& a, const Stream4HashKey& b) {
    return a.hi == b.hi && a.lo == b.lo;
}

__device__ __forceinline__ bool candidate_better(const CandidateMeta& a, const CandidateMeta& b) {
    if (a.score_key != b.score_key) {
        return a.score_key < b.score_key;
    }
    if (a.parent_idx != b.parent_idx) {
        return a.parent_idx < b.parent_idx;
    }
    return a.route_packed < b.route_packed;
}

extern "C" __global__ void kernel_v6_stream4_threshold_compact(
    const CandidateMeta* __restrict__ survivor_shard,
    Stream4HashKey* __restrict__ stream4_key_a,
    CandidateMeta* __restrict__ stream4_val_a,
    uint32_t* __restrict__ compact_count,
    uint32_t input_count,
    uint32_t stream4_job_threshold
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input_count) {
        return;
    }
    const CandidateMeta meta = survivor_shard[i];
    if (meta.score_key > stream4_job_threshold) {
        return;
    }
    const uint32_t out_i = atomicAdd(compact_count, 1u);
    stream4_key_a[out_i] = Stream4HashKey{meta.hash.hi, meta.hash.lo};
    stream4_val_a[out_i] = meta;
}

extern "C" __global__ void kernel_v6_stream4_dedup_sorted(
    const Stream4HashKey* __restrict__ sorted_key,
    const CandidateMeta* __restrict__ sorted_val,
    CandidateMeta* __restrict__ clean_tmp,
    uint32_t* __restrict__ new_clean_count,
    uint32_t compact_count
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= compact_count) {
        return;
    }
    const Stream4HashKey key = sorted_key[i];
    const bool starts_segment = (i == 0) || !(key == sorted_key[i - 1]);
    if (!starts_segment) {
        return;
    }

    CandidateMeta best = sorted_val[i];
    uint32_t j = i + 1;
    while (j < compact_count && sorted_key[j] == key) {
        const CandidateMeta candidate = sorted_val[j];
        if (candidate_better(candidate, best)) {
            best = candidate;
        }
        ++j;
    }

    const uint32_t out_i = atomicAdd(new_clean_count, 1u);
    clean_tmp[out_i] = best;
}

extern "C" __global__ void kernel_v6_stream4_write_clean(
    CandidateMeta* __restrict__ survivor_shard,
    const CandidateMeta* __restrict__ clean_tmp,
    uint32_t* __restrict__ clean_count,
    uint32_t* __restrict__ dirty_count,
    uint8_t* __restrict__ processing_flag,
    uint32_t new_clean_count
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < new_clean_count) {
        survivor_shard[i] = clean_tmp[i];
    }
    if (i == 0) {
        clean_count[0] = new_clean_count;
        dirty_count[0] = 0;
        processing_flag[0] = 0;
    }
}

} // namespace beam_v6

extern "C" void launch_v6_stream4_threshold_compact(
    const beam_v6::CandidateMeta* survivor_shard,
    beam_v6::Stream4HashKey* stream4_key_a,
    beam_v6::CandidateMeta* stream4_val_a,
    uint32_t* compact_count,
    uint32_t input_count,
    uint32_t stream4_job_threshold,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = int((input_count + threads - 1) / threads);
    beam_v6::kernel_v6_stream4_threshold_compact<<<blocks, threads, 0, stream>>>(
        survivor_shard,
        stream4_key_a,
        stream4_val_a,
        compact_count,
        input_count,
        stream4_job_threshold);
}

extern "C" void launch_v6_stream4_sort_pairs(
    void* temp_storage,
    size_t& temp_storage_bytes,
    const beam_v6::Stream4HashKey* key_in,
    beam_v6::Stream4HashKey* key_out,
    const beam_v6::CandidateMeta* val_in,
    beam_v6::CandidateMeta* val_out,
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
        beam_v6::Stream4HashKeyDecomposer{},
        0,
        int(sizeof(beam_v6::Stream4HashKey) * 8),
        stream);
}

extern "C" void launch_v6_stream4_dedup_sorted(
    const beam_v6::Stream4HashKey* sorted_key,
    const beam_v6::CandidateMeta* sorted_val,
    beam_v6::CandidateMeta* clean_tmp,
    uint32_t* new_clean_count,
    uint32_t compact_count,
    cudaStream_t stream
) {
    const int threads = 128;
    const int blocks = int((compact_count + threads - 1) / threads);
    beam_v6::kernel_v6_stream4_dedup_sorted<<<blocks, threads, 0, stream>>>(
        sorted_key,
        sorted_val,
        clean_tmp,
        new_clean_count,
        compact_count);
}

extern "C" void launch_v6_stream4_write_clean(
    beam_v6::CandidateMeta* survivor_shard,
    const beam_v6::CandidateMeta* clean_tmp,
    uint32_t* clean_count,
    uint32_t* dirty_count,
    uint8_t* processing_flag,
    uint32_t new_clean_count,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = int((new_clean_count + threads - 1) / threads);
    beam_v6::kernel_v6_stream4_write_clean<<<blocks, threads, 0, stream>>>(
        survivor_shard,
        clean_tmp,
        clean_count,
        dirty_count,
        processing_flag,
        new_clean_count);
}
