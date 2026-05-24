#include "stream4.hpp"

#include "nvtx_ranges.hpp"

#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace beam {

namespace {

struct Stream4HashLess {
    __host__ __device__ bool operator()(Hash128 a, Hash128 b) const {
        if (a.hi != b.hi) {
            return a.hi < b.hi;
        }
        return a.lo < b.lo;
    }
};

struct Stream4BestCandidate {
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

struct Stream4ScoreCountSum {
    __host__ __device__ std::uint64_t operator()(std::uint64_t a, std::uint64_t b) const {
        return a + b;
    }
};

__device__ CandidateMeta stream4_invalid_candidate_device() {
    return CandidateMeta{Hash128{UINT64_MAX, UINT64_MAX}, UINT64_MAX, UINT32_MAX, UINT32_MAX};
}

__global__ void stream4_clear_inactive_histogram_kernel(
    std::uint32_t* shard_score_hist_a,
    std::uint32_t* shard_score_hist_b,
    const std::uint32_t* shard_score_hist_active_index) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= SCORE_BIN_COUNT) {
        return;
    }
    std::uint32_t* inactive_hist =
        ((*shard_score_hist_active_index & 1U) == 0U) ? shard_score_hist_b : shard_score_hist_a;
    inactive_hist[i] = 0;
}

__global__ void stream4_fill_score_histogram_pairs_kernel(
    const CandidateMeta* clean_survivor,
    const std::uint32_t* clean_count,
    std::uint32_t* score_key,
    std::uint64_t* score_count,
    std::uint32_t capacity) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity) {
        return;
    }
    const std::uint32_t count = *clean_count;
    if (i < count && clean_survivor[i].score_key < SCORE_BIN_COUNT) {
        score_key[i] = clean_survivor[i].score_key;
        score_count[i] = 1ULL;
    } else {
        score_key[i] = SCORE_BIN_COUNT;
        score_count[i] = 0ULL;
    }
}

__global__ void stream4_scatter_score_histogram_kernel(
    const std::uint32_t* score_key,
    const std::uint64_t* score_count,
    const std::uint32_t* unique_count,
    std::uint32_t* shard_score_hist_a,
    std::uint32_t* shard_score_hist_b,
    const std::uint32_t* shard_score_hist_active_index,
    std::uint32_t capacity) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity || i >= *unique_count) {
        return;
    }
    const std::uint32_t key = score_key[i];
    if (key < SCORE_BIN_COUNT) {
        std::uint32_t* inactive_hist =
            ((*shard_score_hist_active_index & 1U) == 0U) ? shard_score_hist_b : shard_score_hist_a;
        inactive_hist[key] = static_cast<std::uint32_t>(score_count[i]);
    }
}

__global__ void stream4_finalize_score_histogram_kernel(
    std::uint32_t* shard_score_hist_active_index,
    std::uint32_t* processing_flag) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *shard_score_hist_active_index = (*shard_score_hist_active_index ^ 1U) & 1U;
        *processing_flag = 0;
    }
}

void check_cub(cudaError_t status, const char* op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
    }
}

__global__ void stream4_mark_threshold_counts_kernel(
    const CandidateMeta* input,
    const std::uint32_t* clean_count,
    const std::uint32_t* dirty_count,
    std::uint32_t* keep_flags,
    std::uint32_t* block_counts,
    const std::uint32_t* threshold_device,
    std::uint32_t threshold_value,
    std::uint32_t capacity) {
    __shared__ std::uint32_t flags[256];
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t i = blockIdx.x * blockDim.x + tid;
    const std::uint32_t input_count = *clean_count + *dirty_count;
    const std::uint32_t threshold = threshold_device == nullptr ? threshold_value : *threshold_device;
    std::uint32_t keep = 0;
    if (i < capacity) {
        keep = i < input_count && input[i].score_key <= threshold ? 1U : 0U;
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

__global__ void stream4_scan_block_counts_kernel(
    const std::uint32_t* block_counts,
    std::uint32_t* block_offsets,
    std::uint32_t* output_count,
    std::uint32_t block_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    std::uint32_t running = 0;
    for (std::uint32_t block = 0; block < block_count; ++block) {
        block_offsets[block] = running;
        running += block_counts[block];
    }
    *output_count = running;
}

__global__ void stream4_compact_threshold_kernel(
    const CandidateMeta* input,
    const std::uint32_t* keep_flags,
    const std::uint32_t* block_offsets,
    Hash128* compact_key,
    CandidateMeta* compact_value,
    std::uint32_t capacity) {
    __shared__ std::uint32_t scan[256];
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t i = blockIdx.x * blockDim.x + tid;
    const std::uint32_t keep = i < capacity ? keep_flags[i] : 0U;
    scan[tid] = keep;
    __syncthreads();
    for (std::uint32_t offset = 1; offset < blockDim.x; offset <<= 1) {
        const std::uint32_t add = tid >= offset ? scan[tid - offset] : 0U;
        __syncthreads();
        scan[tid] += add;
        __syncthreads();
    }
    if (keep != 0U) {
        const CandidateMeta candidate = input[i];
        const std::uint32_t out = block_offsets[blockIdx.x] + scan[tid] - 1U;
        compact_key[out] = candidate.hash;
        compact_value[out] = candidate;
    }
}

__global__ void stream4_fill_sort_tail_kernel(
    Hash128* key,
    CandidateMeta* value,
    const std::uint32_t* compact_count,
    std::uint32_t capacity) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count = *compact_count;
    if (i >= count && i < capacity) {
        key[i] = Hash128{UINT64_MAX, UINT64_MAX};
        value[i] = stream4_invalid_candidate_device();
    }
}

__global__ void stream4_mark_valid_unique_counts_kernel(
    const CandidateMeta* reduced_value,
    std::uint32_t* keep_flags,
    std::uint32_t* block_counts,
    const std::uint32_t* reduced_count,
    std::uint32_t capacity) {
    __shared__ std::uint32_t flags[256];
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t i = blockIdx.x * blockDim.x + tid;
    const std::uint32_t count = *reduced_count;
    std::uint32_t keep = 0;
    if (i < count) {
        keep = reduced_value[i].score_key != UINT32_MAX ? 1U : 0U;
        keep_flags[i] = keep;
    } else if (i < capacity) {
        keep_flags[i] = 0;
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

__global__ void stream4_compact_valid_unique_kernel(
    const CandidateMeta* reduced_value,
    const std::uint32_t* keep_flags,
    const std::uint32_t* block_offsets,
    CandidateMeta* output,
    std::uint32_t* clean_count,
    std::uint32_t* dirty_count,
    const std::uint32_t* output_count,
    std::uint32_t capacity) {
    __shared__ std::uint32_t scan[256];
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t i = blockIdx.x * blockDim.x + tid;
    const std::uint32_t keep = i < capacity ? keep_flags[i] : 0U;
    scan[tid] = keep;
    __syncthreads();
    for (std::uint32_t offset = 1; offset < blockDim.x; offset <<= 1) {
        const std::uint32_t add = tid >= offset ? scan[tid - offset] : 0U;
        __syncthreads();
        scan[tid] += add;
        __syncthreads();
    }
    if (keep != 0U) {
        output[block_offsets[blockIdx.x] + scan[tid] - 1U] = reduced_value[i];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *clean_count = *output_count;
        *dirty_count = 0;
    }
}

void stream4_write_shard_histogram(
    CandidateMeta* survivor_shard,
    const std::uint32_t* clean_count,
    std::uint32_t capacity,
    std::uint32_t* score_key_a,
    std::uint32_t* score_key_b,
    std::uint64_t* score_count_a,
    std::uint64_t* score_count_b,
    std::uint32_t* score_unique_count,
    std::uint32_t* shard_score_hist_a,
    std::uint32_t* shard_score_hist_b,
    std::uint32_t* shard_score_hist_active_index,
    std::uint32_t* processing_flag,
    void* cub_temp_storage,
    std::size_t cub_temp_storage_bytes,
    cudaStream_t stream) {
    const std::uint32_t block_size = 256;
    const std::uint32_t block_count = (capacity + block_size - 1U) / block_size;
    const dim3 block(block_size);
    const dim3 grid(block_count);
    const dim3 hist_grid((SCORE_BIN_COUNT + block_size - 1U) / block_size);

    stream4_clear_inactive_histogram_kernel<<<hist_grid, block, 0, stream>>>(
        shard_score_hist_a,
        shard_score_hist_b,
        shard_score_hist_active_index);
    stream4_fill_score_histogram_pairs_kernel<<<grid, block, 0, stream>>>(
        survivor_shard,
        clean_count,
        score_key_a,
        score_count_a,
        capacity);

    std::size_t sort_temp_bytes = cub_temp_storage_bytes;
    check_cub(
        cub::DeviceRadixSort::SortPairs(
            cub_temp_storage,
            sort_temp_bytes,
            score_key_a,
            score_key_b,
            score_count_a,
            score_count_b,
            static_cast<int>(capacity),
            0,
            32,
            stream),
        "cub::DeviceRadixSort::SortPairs stream4 score histogram");

    std::size_t reduce_temp_bytes = cub_temp_storage_bytes;
    check_cub(
        cub::DeviceReduce::ReduceByKey(
            cub_temp_storage,
            reduce_temp_bytes,
            score_key_b,
            score_key_a,
            score_count_b,
            score_count_a,
            score_unique_count,
            Stream4ScoreCountSum{},
            static_cast<int>(capacity),
            stream),
        "cub::DeviceReduce::ReduceByKey stream4 score histogram");

    stream4_scatter_score_histogram_kernel<<<grid, block, 0, stream>>>(
        score_key_a,
        score_count_a,
        score_unique_count,
        shard_score_hist_a,
        shard_score_hist_b,
        shard_score_hist_active_index,
        capacity);
    stream4_finalize_score_histogram_kernel<<<1, 1, 0, stream>>>(
        shard_score_hist_active_index,
        processing_flag);
}

} // namespace

void stream4_shard_job_cuda(
    CandidateMeta* survivor_shard,
    std::uint32_t* clean_count,
    std::uint32_t* dirty_count,
    std::uint32_t* processing_flag,
    std::uint32_t threshold,
    std::uint32_t capacity,
    Hash128* sort_key,
    Hash128* reduce_key,
    CandidateMeta* sort_value,
    CandidateMeta* reduce_value,
    std::uint32_t* score_key_a,
    std::uint32_t* score_key_b,
    std::uint64_t* score_count_a,
    std::uint64_t* score_count_b,
    std::uint32_t* keep_flags,
    std::uint32_t* block_counts,
    std::uint32_t* block_offsets,
    std::uint32_t* scratch_count,
    std::uint32_t* shard_score_hist_a,
    std::uint32_t* shard_score_hist_b,
    std::uint32_t* shard_score_hist_active_index,
    void* cub_temp_storage,
    std::size_t cub_temp_storage_bytes,
    cudaStream_t stream) {
    NvtxRange range("Stream4_threshold_compact_cub_sort_reduce_launch");
    if (capacity == 0) {
        throw std::invalid_argument("stream4 shard job capacity must be greater than zero");
    }
    if (cub_temp_storage == nullptr || cub_temp_storage_bytes == 0) {
        throw std::invalid_argument("stream4 CUB fixed temp storage is required");
    }
    const std::uint32_t block_size = 256;
    const std::uint32_t block_count = (capacity + block_size - 1U) / block_size;
    const dim3 block(block_size);
    const dim3 grid(block_count);

    stream4_mark_threshold_counts_kernel<<<grid, block, 0, stream>>>(
        survivor_shard,
        clean_count,
        dirty_count,
        keep_flags,
        block_counts,
        nullptr,
        threshold,
        capacity);
    stream4_scan_block_counts_kernel<<<1, 1, 0, stream>>>(
        block_counts,
        block_offsets,
        scratch_count,
        block_count);
    stream4_compact_threshold_kernel<<<grid, block, 0, stream>>>(
        survivor_shard,
        keep_flags,
        block_offsets,
        sort_key,
        sort_value,
        capacity);
    stream4_fill_sort_tail_kernel<<<grid, block, 0, stream>>>(sort_key, sort_value, scratch_count, capacity);

    std::size_t sort_temp_bytes = cub_temp_storage_bytes;
    check_cub(
        cub::DeviceMergeSort::SortPairs(
            cub_temp_storage,
            sort_temp_bytes,
            sort_key,
            sort_value,
            capacity,
            Stream4HashLess{},
            stream),
        "cub::DeviceMergeSort::SortPairs stream4");

    std::size_t reduce_temp_bytes = cub_temp_storage_bytes;
    check_cub(
        cub::DeviceReduce::ReduceByKey(
            cub_temp_storage,
            reduce_temp_bytes,
            sort_key,
            reduce_key,
            sort_value,
            reduce_value,
            scratch_count,
            Stream4BestCandidate{},
            capacity,
            stream),
        "cub::DeviceReduce::ReduceByKey stream4");

    stream4_mark_valid_unique_counts_kernel<<<grid, block, 0, stream>>>(
        reduce_value,
        keep_flags,
        block_counts,
        scratch_count,
        capacity);
    stream4_scan_block_counts_kernel<<<1, 1, 0, stream>>>(
        block_counts,
        block_offsets,
        scratch_count,
        block_count);
    stream4_compact_valid_unique_kernel<<<grid, block, 0, stream>>>(
        reduce_value,
        keep_flags,
        block_offsets,
        survivor_shard,
        clean_count,
        dirty_count,
        scratch_count,
        capacity);
    stream4_write_shard_histogram(
        survivor_shard,
        clean_count,
        capacity,
        score_key_a,
        score_key_b,
        score_count_a,
        score_count_b,
        scratch_count,
        shard_score_hist_a,
        shard_score_hist_b,
        shard_score_hist_active_index,
        processing_flag,
        cub_temp_storage,
        cub_temp_storage_bytes,
        stream);
}

void stream4_shard_job_device_threshold_cuda(
    CandidateMeta* survivor_shard,
    std::uint32_t* clean_count,
    std::uint32_t* dirty_count,
    std::uint32_t* processing_flag,
    const std::uint32_t* threshold,
    std::uint32_t capacity,
    Hash128* sort_key,
    Hash128* reduce_key,
    CandidateMeta* sort_value,
    CandidateMeta* reduce_value,
    std::uint32_t* score_key_a,
    std::uint32_t* score_key_b,
    std::uint64_t* score_count_a,
    std::uint64_t* score_count_b,
    std::uint32_t* keep_flags,
    std::uint32_t* block_counts,
    std::uint32_t* block_offsets,
    std::uint32_t* scratch_count,
    std::uint32_t* shard_score_hist_a,
    std::uint32_t* shard_score_hist_b,
    std::uint32_t* shard_score_hist_active_index,
    void* cub_temp_storage,
    std::size_t cub_temp_storage_bytes,
    cudaStream_t stream) {
    NvtxRange range("Stream4_threshold_compact_cub_sort_reduce_device_threshold_launch");
    if (capacity == 0) {
        throw std::invalid_argument("stream4 shard job capacity must be greater than zero");
    }
    if (cub_temp_storage == nullptr || cub_temp_storage_bytes == 0) {
        throw std::invalid_argument("stream4 CUB fixed temp storage is required");
    }
    const std::uint32_t block_size = 256;
    const std::uint32_t block_count = (capacity + block_size - 1U) / block_size;
    const dim3 block(block_size);
    const dim3 grid(block_count);

    stream4_mark_threshold_counts_kernel<<<grid, block, 0, stream>>>(
        survivor_shard,
        clean_count,
        dirty_count,
        keep_flags,
        block_counts,
        threshold,
        UINT32_MAX,
        capacity);
    stream4_scan_block_counts_kernel<<<1, 1, 0, stream>>>(
        block_counts,
        block_offsets,
        scratch_count,
        block_count);
    stream4_compact_threshold_kernel<<<grid, block, 0, stream>>>(
        survivor_shard,
        keep_flags,
        block_offsets,
        sort_key,
        sort_value,
        capacity);
    stream4_fill_sort_tail_kernel<<<grid, block, 0, stream>>>(sort_key, sort_value, scratch_count, capacity);

    std::size_t sort_temp_bytes = cub_temp_storage_bytes;
    check_cub(
        cub::DeviceMergeSort::SortPairs(
            cub_temp_storage,
            sort_temp_bytes,
            sort_key,
            sort_value,
            capacity,
            Stream4HashLess{},
            stream),
        "cub::DeviceMergeSort::SortPairs stream4");

    std::size_t reduce_temp_bytes = cub_temp_storage_bytes;
    check_cub(
        cub::DeviceReduce::ReduceByKey(
            cub_temp_storage,
            reduce_temp_bytes,
            sort_key,
            reduce_key,
            sort_value,
            reduce_value,
            scratch_count,
            Stream4BestCandidate{},
            capacity,
            stream),
        "cub::DeviceReduce::ReduceByKey stream4");

    stream4_mark_valid_unique_counts_kernel<<<grid, block, 0, stream>>>(
        reduce_value,
        keep_flags,
        block_counts,
        scratch_count,
        capacity);
    stream4_scan_block_counts_kernel<<<1, 1, 0, stream>>>(
        block_counts,
        block_offsets,
        scratch_count,
        block_count);
    stream4_compact_valid_unique_kernel<<<grid, block, 0, stream>>>(
        reduce_value,
        keep_flags,
        block_offsets,
        survivor_shard,
        clean_count,
        dirty_count,
        scratch_count,
        capacity);
    stream4_write_shard_histogram(
        survivor_shard,
        clean_count,
        capacity,
        score_key_a,
        score_key_b,
        score_count_a,
        score_count_b,
        scratch_count,
        shard_score_hist_a,
        shard_score_hist_b,
        shard_score_hist_active_index,
        processing_flag,
        cub_temp_storage,
        cub_temp_storage_bytes,
        stream);
}

} // namespace beam
