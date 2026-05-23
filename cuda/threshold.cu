#include "threshold.hpp"

#include "config.hpp"
#include "nvtx_ranges.hpp"

#include <cuda_runtime.h>

#include <climits>
#include <stdexcept>
#include <string>

namespace beam {

namespace {

void check_nccl_threshold(ncclResult_t status, const char* op) {
    if (status != ncclSuccess) {
        throw std::runtime_error(std::string(op) + ": " + ncclGetErrorString(status));
    }
}

__device__ std::uint8_t unpack_move_threshold_device(std::uint32_t route_packed) {
    return static_cast<std::uint8_t>(route_packed & 0xff);
}

__device__ std::uint64_t ceil_div_u64_device(std::uint64_t a, std::uint64_t b) {
    return b == 0ULL ? 0ULL : (a + b - 1ULL) / b;
}

__global__ void threshold_snapshot_active_histogram_kernel(
    const std::uint32_t* shard_score_hist_active_index,
    std::uint32_t* threshold_hist_active_snapshot,
    std::uint32_t shard_count) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < shard_count) {
        threshold_hist_active_snapshot[i] = shard_score_hist_active_index[i] & 1U;
    }
}

__global__ void threshold_sum_shard_histograms_kernel(
    const std::uint32_t* shard_score_hist_a,
    const std::uint32_t* shard_score_hist_b,
    const std::uint32_t* threshold_hist_active_snapshot,
    std::uint64_t* local_score_hist,
    std::uint32_t shard_count) {
    const std::uint32_t score = blockIdx.x * blockDim.x + threadIdx.x;
    if (score >= SCORE_BIN_COUNT) {
        return;
    }
    std::uint64_t sum = 0;
    for (std::uint32_t shard = 0; shard < shard_count; ++shard) {
        const std::uint64_t offset = static_cast<std::uint64_t>(shard) * SCORE_BIN_COUNT + score;
        const std::uint32_t* hist =
            threshold_hist_active_snapshot[shard] == 0U ? shard_score_hist_a : shard_score_hist_b;
        sum += static_cast<std::uint64_t>(hist[offset]);
    }
    local_score_hist[score] = sum;
}

__global__ void threshold_select_kernel(
    const std::uint64_t* global_score_hist,
    std::uint32_t* current_threshold,
    std::uint64_t global_beam_width_effective) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    std::uint64_t cumulative = 0;
    for (std::uint32_t score = 0; score < SCORE_BIN_COUNT; ++score) {
        cumulative += global_score_hist[score];
        if (cumulative >= global_beam_width_effective) {
            *current_threshold = score;
            return;
        }
    }
    *current_threshold = UINT32_THRESHOLD_MAX;
}

__global__ void threshold_update_periodic_kernel(
    const std::uint64_t* global_score_hist,
    std::uint32_t* current_threshold,
    std::uint32_t* threshold_initialized,
    std::uint64_t global_beam_width_effective) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    std::uint64_t total = 0;
    std::uint64_t cumulative = 0;
    std::uint32_t selected = UINT32_THRESHOLD_MAX;
    for (std::uint32_t score = 0; score < SCORE_BIN_COUNT; ++score) {
        const std::uint64_t bin = global_score_hist[score];
        total += bin;
        if (selected == UINT32_THRESHOLD_MAX) {
            cumulative += bin;
            if (cumulative >= global_beam_width_effective) {
                selected = score;
            }
        }
    }
    if (*threshold_initialized == 0U && total < global_beam_width_effective) {
        *current_threshold = UINT32_THRESHOLD_MAX;
        return;
    }
    if (total >= global_beam_width_effective) {
        if (*threshold_initialized == 0U || selected < *current_threshold) {
            *current_threshold = selected;
        }
        *threshold_initialized = 1U;
    }
}

__global__ void final_mark_counts_kernel(
    const CandidateMeta* survivor_shard,
    const std::uint32_t* clean_count,
    std::uint32_t* keep_flags,
    std::uint32_t* block_counts,
    std::uint32_t final_threshold,
    std::uint32_t shard_count,
    std::uint32_t stream4_batch_candidates) {
    __shared__ std::uint32_t flags[256];
    const std::uint32_t tid = threadIdx.x;
    const std::uint64_t shard_capacity = 2ULL * stream4_batch_candidates;
    const std::uint64_t total = static_cast<std::uint64_t>(shard_count) * shard_capacity;
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + tid;
    std::uint32_t keep = 0;
    if (i < total) {
        const std::uint32_t shard = static_cast<std::uint32_t>(i / shard_capacity);
        const std::uint32_t local = static_cast<std::uint32_t>(i - static_cast<std::uint64_t>(shard) * shard_capacity);
        keep = local < clean_count[shard] && survivor_shard[i].score_key <= final_threshold ? 1U : 0U;
        keep_flags[i] = keep;
    }
    flags[tid] = keep;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride > 0; stride >>= 1U) {
        if (tid < stride) {
            flags[tid] += flags[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_counts[blockIdx.x] = flags[0];
    }
}

__global__ void final_scan_block_counts_kernel(
    const std::uint32_t* block_counts,
    std::uint32_t* block_offsets,
    std::uint32_t* final_candidate_count,
    std::uint32_t block_count,
    std::uint64_t global_prefix_for_rank,
    std::uint64_t global_keep_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    std::uint32_t running = 0;
    for (std::uint32_t block = 0; block < block_count; ++block) {
        block_offsets[block] = running;
        running += block_counts[block];
    }
    if (global_keep_count == 0ULL || global_prefix_for_rank >= global_keep_count) {
        *final_candidate_count = 0;
        return;
    }
    const std::uint64_t remaining_global_keep = global_keep_count - global_prefix_for_rank;
    const std::uint64_t capped =
        static_cast<std::uint64_t>(running) < remaining_global_keep
            ? static_cast<std::uint64_t>(running)
            : remaining_global_keep;
    *final_candidate_count = static_cast<std::uint32_t>(capped);
}

__global__ void final_init_send_ranges_kernel(
    const std::uint32_t* final_candidate_count,
    std::uint32_t* final_request_count,
    std::uint32_t* final_send_count,
    std::uint32_t* final_send_offset,
    std::uint32_t local_rank,
    std::uint32_t world_size,
    std::uint64_t global_prefix_for_rank,
    std::uint64_t global_keep_count,
    std::uint32_t final_capacity) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
        final_send_count[rank] = 0;
        final_send_offset[rank] = 0;
    }
    final_send_offset[world_size] = 0;
    *final_request_count = 0;
    const std::uint32_t local_keep_total = *final_candidate_count;
    if (global_keep_count == 0ULL || world_size == 0U || local_keep_total > final_capacity) {
        return;
    }
    const std::uint64_t balance_keep_count =
        world_size == 1U ? static_cast<std::uint64_t>(local_keep_total) : global_keep_count;
    const std::uint64_t local_begin = global_prefix_for_rank;
    const std::uint64_t local_end = global_prefix_for_rank + local_keep_total;
    std::uint32_t running = 0;
    final_send_offset[0] = 0;
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
        const std::uint64_t rank_begin =
            ceil_div_u64_device(static_cast<std::uint64_t>(rank) * balance_keep_count, world_size);
        const std::uint64_t rank_end =
            ceil_div_u64_device(static_cast<std::uint64_t>(rank + 1U) * balance_keep_count, world_size);
        const std::uint64_t begin = local_begin > rank_begin ? local_begin : rank_begin;
        const std::uint64_t end = local_end < rank_end ? local_end : rank_end;
        const std::uint32_t count = end > begin ? static_cast<std::uint32_t>(end - begin) : 0U;
        final_send_count[rank] = count;
        running += count;
        final_send_offset[rank + 1U] = running;
    }
    *final_request_count = local_rank < world_size ? final_send_count[local_rank] : 0U;
}

__global__ void final_scatter_load_balance_kernel(
    const CandidateMeta* survivor_shard,
    const std::uint32_t* keep_flags,
    const std::uint32_t* block_offsets,
    CandidateMeta* final_candidate_buffer,
    std::uint32_t* final_candidate_count,
    FinalRequest* final_request_buffer,
    std::uint32_t local_rank,
    std::uint32_t world_size,
    std::uint64_t global_prefix_for_rank,
    std::uint64_t global_keep_count,
    std::uint32_t final_capacity,
    std::uint32_t shard_count,
    std::uint32_t stream4_batch_candidates) {
    __shared__ std::uint32_t scan[256];
    const std::uint32_t tid = threadIdx.x;
    const std::uint64_t shard_capacity = 2ULL * stream4_batch_candidates;
    const std::uint64_t total = static_cast<std::uint64_t>(shard_count) * shard_capacity;
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + tid;
    const std::uint32_t keep = i < total ? keep_flags[i] : 0U;
    scan[tid] = keep;
    __syncthreads();
    for (std::uint32_t offset = 1U; offset < blockDim.x; offset <<= 1U) {
        const std::uint32_t add = tid >= offset ? scan[tid - offset] : 0U;
        __syncthreads();
        scan[tid] += add;
        __syncthreads();
    }
    const std::uint64_t balance_keep_count =
        world_size == 1U ? static_cast<std::uint64_t>(*final_candidate_count) : global_keep_count;
    if (keep == 0U || *final_candidate_count > final_capacity || balance_keep_count == 0ULL || world_size == 0U) {
        return;
    }
    const std::uint32_t local_out = block_offsets[blockIdx.x] + scan[tid] - 1U;
    if (local_out >= *final_candidate_count) {
        return;
    }
    const std::uint64_t global_idx = global_prefix_for_rank + local_out;
    std::uint32_t target_rank = static_cast<std::uint32_t>((global_idx * world_size) / balance_keep_count);
    if (target_rank >= world_size) {
        target_rank = world_size - 1U;
    }
    const std::uint64_t target_rank_begin =
        ceil_div_u64_device(static_cast<std::uint64_t>(target_rank) * balance_keep_count, world_size);
    const std::uint32_t target_local_idx = static_cast<std::uint32_t>(global_idx - target_rank_begin);
    const CandidateMeta candidate = survivor_shard[i];
    final_candidate_buffer[local_out] = candidate;
    if (target_rank == local_rank) {
        FinalRequest request{};
        request.parent_idx = candidate.parent_idx;
        request.target_local_idx = target_local_idx;
        request.return_rank = static_cast<std::uint16_t>(target_rank);
        request.move = unpack_move_threshold_device(candidate.route_packed);
        final_request_buffer[target_local_idx] = request;
    }
}

} // namespace

void threshold_build_local_histogram_cuda(
    const std::uint32_t* shard_score_hist_a,
    const std::uint32_t* shard_score_hist_b,
    const std::uint32_t* shard_score_hist_active_index,
    std::uint32_t* threshold_hist_active_snapshot,
    std::uint64_t* local_score_hist,
    std::uint32_t shard_count,
    cudaStream_t stream) {
    NvtxRange range("Threshold_sum_shard_histograms_launch");
    const std::uint32_t block_size = 256;
    const dim3 block(block_size);
    const dim3 shard_grid((shard_count + block_size - 1U) / block_size);
    const dim3 hist_grid((SCORE_BIN_COUNT + block_size - 1U) / block_size);
    threshold_snapshot_active_histogram_kernel<<<shard_grid, block, 0, stream>>>(
        shard_score_hist_active_index,
        threshold_hist_active_snapshot,
        shard_count);
    threshold_sum_shard_histograms_kernel<<<hist_grid, block, 0, stream>>>(
        shard_score_hist_a,
        shard_score_hist_b,
        threshold_hist_active_snapshot,
        local_score_hist,
        shard_count);
}

void threshold_select_cuda(
    const std::uint64_t* global_score_hist,
    std::uint32_t* current_threshold,
    std::uint64_t global_beam_width_effective,
    cudaStream_t stream) {
    NvtxRange range("Threshold_select_launch");
    threshold_select_kernel<<<1, 1, 0, stream>>>(global_score_hist, current_threshold, global_beam_width_effective);
}

void threshold_update_periodic_cuda(
    const std::uint64_t* global_score_hist,
    std::uint32_t* current_threshold,
    std::uint32_t* threshold_initialized,
    std::uint64_t global_beam_width_effective,
    cudaStream_t stream) {
    NvtxRange range("Threshold_update_periodic_launch");
    threshold_update_periodic_kernel<<<1, 1, 0, stream>>>(
        global_score_hist,
        current_threshold,
        threshold_initialized,
        global_beam_width_effective);
}

void threshold_allreduce_histogram_nccl_cuda(
    const std::uint64_t* local_score_hist,
    std::uint64_t* global_score_hist,
    ncclComm_t comm,
    cudaStream_t stream) {
    NvtxRange range("Threshold_NCCL_AllReduce_histogram_launch");
    check_nccl_threshold(
        ncclAllReduce(
            local_score_hist,
            global_score_hist,
            SCORE_BIN_COUNT,
            ncclUint64,
            ncclSum,
            comm,
            stream),
        "ncclAllReduce histogram");
}

void final_allgather_counts_nccl_cuda(
    const std::uint32_t* local_keep_count,
    std::uint32_t* all_keep_counts,
    ncclComm_t comm,
    cudaStream_t stream) {
    NvtxRange range("Final_NCCL_AllGather_counts_launch");
    check_nccl_threshold(
        ncclAllGather(
            local_keep_count,
            all_keep_counts,
            1,
            ncclUint32,
            comm,
            stream),
        "ncclAllGather counts");
}

void final_filter_load_balance_cuda(
    const CandidateMeta* survivor_shard,
    const std::uint32_t* clean_count,
    std::uint32_t* keep_flags,
    std::uint32_t* block_counts,
    std::uint32_t* block_offsets,
    CandidateMeta* final_candidate_buffer,
    std::uint32_t* final_candidate_count,
    FinalRequest* final_request_buffer,
    std::uint32_t* final_request_count,
    std::uint32_t* final_send_count,
    std::uint32_t* final_send_offset,
    std::uint32_t final_threshold,
    std::uint32_t local_rank,
    std::uint32_t world_size,
    std::uint64_t global_prefix_for_rank,
    std::uint64_t global_keep_count,
    std::uint32_t final_capacity,
    std::uint32_t shard_count,
    std::uint32_t stream4_batch_candidates,
    cudaStream_t stream) {
    NvtxRange range("Final_filter_load_balance_launch");
    const std::uint64_t item_count = static_cast<std::uint64_t>(shard_count) * 2ULL * stream4_batch_candidates;
    const std::uint32_t block_size = 256;
    const std::uint32_t block_count = static_cast<std::uint32_t>((item_count + block_size - 1ULL) / block_size);
    const dim3 block(block_size);
    const dim3 grid(block_count);
    final_mark_counts_kernel<<<grid, block, 0, stream>>>(
        survivor_shard,
        clean_count,
        keep_flags,
        block_counts,
        final_threshold,
        shard_count,
        stream4_batch_candidates);
    final_scan_block_counts_kernel<<<1, 1, 0, stream>>>(
        block_counts,
        block_offsets,
        final_candidate_count,
        block_count,
        global_prefix_for_rank,
        global_keep_count);
    final_init_send_ranges_kernel<<<1, 1, 0, stream>>>(
        final_candidate_count,
        final_request_count,
        final_send_count,
        final_send_offset,
        local_rank,
        world_size,
        global_prefix_for_rank,
        global_keep_count,
        final_capacity);
    final_scatter_load_balance_kernel<<<grid, block, 0, stream>>>(
        survivor_shard,
        keep_flags,
        block_offsets,
        final_candidate_buffer,
        final_candidate_count,
        final_request_buffer,
        local_rank,
        world_size,
        global_prefix_for_rank,
        global_keep_count,
        final_capacity,
        shard_count,
        stream4_batch_candidates);
}

} // namespace beam
