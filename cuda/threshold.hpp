#pragma once

#include "types.hpp"

#include <nccl.h>

#include <cstddef>

namespace beam {

void threshold_build_local_histogram_cuda(
    const std::uint32_t* shard_score_hist_a,
    const std::uint32_t* shard_score_hist_b,
    const std::uint32_t* shard_score_hist_active_index,
    std::uint32_t* threshold_hist_active_snapshot,
    std::uint64_t* local_score_hist,
    std::uint32_t shard_count,
    cudaStream_t stream);

void threshold_select_cuda(
    const std::uint64_t* global_score_hist,
    std::uint32_t* current_threshold,
    std::uint64_t global_beam_width_effective,
    cudaStream_t stream);

void threshold_update_periodic_cuda(
    const std::uint64_t* global_score_hist,
    std::uint32_t* current_threshold,
    std::uint32_t* threshold_initialized,
    std::uint64_t global_beam_width_effective,
    cudaStream_t stream);

void threshold_allreduce_histogram_nccl_cuda(
    const std::uint64_t* local_score_hist,
    std::uint64_t* global_score_hist,
    ncclComm_t comm,
    cudaStream_t stream);

void final_allgather_counts_nccl_cuda(
    const std::uint32_t* local_keep_count,
    std::uint32_t* all_keep_counts,
    ncclComm_t comm,
    cudaStream_t stream);

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
    cudaStream_t stream);

} // namespace beam
