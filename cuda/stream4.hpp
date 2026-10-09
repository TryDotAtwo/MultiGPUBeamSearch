#pragma once

#include "types.hpp"

#include <cstddef>

namespace beam {

// Final phase only: callers have drained all writers. Two adjacent physical
// buffers form one logical shard. All scratch arrays hold 2*capacity entries;
// block arrays hold ceil(2*capacity/256), and CUB temp is sized for 2*capacity.
// Histograms/publication indices have two physical-buffer slices. No local top-k.
// Only clean_count is input metadata. dirty_count/processing_flag are two-cell
// output housekeeping arrays and may belong to the final-union scratch phase.
// sort_item_bound=0 retains full-capacity sorting; a positive proven upper
// bound shrinks CUB item count only. GPU traps if compaction exceeds the bound.
void stream4_finalize_logical_shard_union_cuda(
    CandidateMeta* survivor_pair, std::uint32_t* clean_count,
    std::uint32_t* dirty_count, std::uint32_t* processing_flag,
    std::uint32_t threshold, std::uint32_t capacity,
    Hash128* sort_key, Hash128* reduce_key,
    CandidateMeta* sort_value, CandidateMeta* reduce_value,
    std::uint32_t* score_key_a, std::uint32_t* score_key_b,
    std::uint64_t* score_count_a, std::uint64_t* score_count_b,
    std::uint32_t* keep_flags, std::uint32_t* block_counts,
    std::uint32_t* block_offsets, std::uint32_t* scratch_count,
    std::uint32_t* hist_a, std::uint32_t* hist_b, std::uint32_t* hist_active,
    void* cub_temp, std::size_t cub_temp_bytes, cudaStream_t stream,
    std::uint32_t sort_item_bound = 0);

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
    const std::uint32_t* threshold_active_index,
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
