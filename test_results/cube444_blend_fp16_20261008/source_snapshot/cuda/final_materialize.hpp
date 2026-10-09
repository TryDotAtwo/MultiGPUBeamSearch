#pragma once

#include "types.hpp"

#include <cstddef>

namespace beam {

void final_materialize_cuda(
    const State128* current_frontier_states,
    const FinalRequest* requests,
    const std::uint8_t* generators,
    FinalResponse* responses,
    State128* next_frontier_states_tmp,
    std::uint32_t request_count,
    cudaStream_t stream);

void final_materialize_responses_cuda(
    const State128* current_frontier_states,
    const FinalRequest* requests,
    const std::uint8_t* generators,
    FinalResponse* responses,
    std::uint32_t request_count,
    cudaStream_t stream);

void final_scatter_responses_cuda(
    const FinalResponse* responses,
    State128* next_frontier_states_tmp,
    std::uint32_t response_count,
    cudaStream_t stream);

void final_build_materialize_chunk_cuda(
    const CandidateMeta* candidates,
    std::uint32_t chunk_begin,
    std::uint32_t chunk_count,
    std::uint32_t local_less_count,
    std::uint64_t less_prefix_for_rank,
    std::uint64_t global_less,
    std::uint64_t equal_prefix_for_rank,
    std::uint64_t global_keep_count,
    std::uint32_t local_rank,
    std::uint32_t world_size,
    std::uint32_t* source_rank_keys,
    FinalRequest* requests,
    FinalHistoryRecord* history_records,
    cudaStream_t stream);

void final_sort_requests_by_key_cuda(
    const std::uint32_t* keys_in,
    std::uint32_t* keys_out,
    const FinalRequest* requests_in,
    FinalRequest* requests_out,
    std::uint32_t request_count,
    void* cub_temp,
    std::size_t cub_temp_bytes,
    cudaStream_t stream);

void final_count_sorted_rank_keys_cuda(
    const std::uint32_t* sorted_keys,
    std::uint32_t item_count,
    std::uint32_t* counts,
    std::uint32_t* offsets,
    std::uint32_t world_size,
    cudaStream_t stream);

void final_build_return_rank_keys_cuda(
    const FinalRequest* requests,
    std::uint32_t* return_rank_keys,
    std::uint32_t request_count,
    cudaStream_t stream);

void final_scatter_history_records_cuda(
    const FinalHistoryRecord* records,
    CandidateMeta* history_candidates,
    std::uint32_t record_count,
    cudaStream_t stream);

void validate_final_requests_cuda(
    const FinalRequest* requests,
    std::uint32_t request_count,
    std::uint64_t current_frontier_size,
    std::uint32_t target_count,
    FinalRequestValidationError* device_error,
    cudaStream_t stream);

} // namespace beam
