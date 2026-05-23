#pragma once

#include "types.hpp"

#include <cstddef>

namespace beam {

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
    cudaStream_t stream);

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
    cudaStream_t stream);

} // namespace beam
