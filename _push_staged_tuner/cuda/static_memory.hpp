#pragma once

#include "config.hpp"
#include "types.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#ifndef BEAM_DEBUG_DEPTH_FLOW_TRACE
#define BEAM_DEBUG_DEPTH_FLOW_TRACE 0
#endif

namespace beam {

struct StaticMemoryPlan {
    RuntimeConfig config;
    DerivedConfig derived;
    std::uint64_t frontier_states = 0;
    std::uint64_t score_ring_count = 0;
    std::uint64_t hash_ring_count = 0;
    std::uint64_t parent_base_count = 0;
    std::uint64_t ring_count_count = 0;
    std::uint64_t stream3_count = 0;
    std::uint64_t stream5_slot_count = 0;
    std::uint64_t stream5_send_slot_capacity = 0;
    std::uint64_t stream5_recv_slot_capacity = 0;
    std::uint32_t final_materialize_slot_count = 0;
    std::uint64_t final_materialize_chunk_capacity = 0;
    std::uint64_t final_materialize_exchange_capacity = 0;
    std::uint64_t final_selected_candidate_capacity = 0;
    std::uint32_t storage_shard_count = 0;
    std::uint64_t survivor_count = 0;
    std::uint64_t final_state_count = 0;
    std::size_t stream3_cub_temp_bytes = 0;
    std::size_t stream4_cub_temp_bytes = 0;
    std::size_t final_materialize_cub_temp_bytes = 0;
    std::size_t current_frontier_bytes = 0;
    std::size_t solved_bytes = 0;
    std::size_t layout_phase1_streams_bytes = 0;
    std::size_t layout_phase2_select_bytes = 0;
    std::size_t layout_phase3_materialize_bytes = 0;
    std::size_t layout_streams_bytes = 0;
    std::size_t layout_final_budget_bytes = 0;
    std::size_t layout_final_bytes = 0;
    std::size_t scratch_pool_bytes = 0;
    std::size_t total_device_bytes = 0;
};

struct LayoutStreamsView {
    std::uint32_t* score_ring = nullptr;
    Hash128* hash_ring = nullptr;
    std::uint64_t* parent_base = nullptr;
    std::uint32_t* count = nullptr;
    Hash128* stream3_key_a = nullptr;
    Hash128* stream3_key_b = nullptr;
    std::uint64_t* stream3_val_a = nullptr;
    std::uint64_t* stream3_val_b = nullptr;
    std::uint32_t* stream3_keep_flags = nullptr;
    std::uint32_t* stream3_block_counts = nullptr;
    std::uint32_t* stream3_block_offsets = nullptr;
    std::uint32_t* stream3_owner = nullptr;
    std::uint32_t* stream3_shard_counts = nullptr;
    std::uint32_t* stream3_shard_offsets = nullptr;
    std::uint32_t* stream3_spill_counts = nullptr;
    std::uint32_t* stream3_spill_offsets = nullptr;
    std::uint32_t* stream3_ready_flag = nullptr;
    std::uint32_t* stream3_ready_shard_list = nullptr;
    std::uint32_t* stream3_ready_count = nullptr;
    std::uint32_t* stream3_write_buffer_index = nullptr;
    std::uint32_t* stream3_partition_key_a = nullptr;
    std::uint32_t* stream3_partition_key_b = nullptr;
    CandidateMeta* stream3_partition_val_a = nullptr;
    CandidateMeta* stream3_partition_val_b = nullptr;
    std::uint32_t* stream3_partition_unique_shard = nullptr;
    std::uint32_t* stream3_partition_unique_counts = nullptr;
    std::uint32_t* stream3_partition_unique_count = nullptr;
    std::uint32_t* stream3_score_key_a = nullptr;
    std::uint32_t* stream3_score_key_b = nullptr;
    std::uint64_t* stream3_score_count_a = nullptr;
    std::uint64_t* stream3_score_count_b = nullptr;
    std::uint32_t* stream3_score_unique_count = nullptr;
    std::uint64_t* stream3_score_hist = nullptr;
    void* stream3_cub_temp = nullptr;
    std::size_t stream3_cub_temp_bytes = 0;
    Hash128* unique_key = nullptr;
    std::uint64_t* unique_val = nullptr;
    std::uint32_t* unique_count = nullptr;
#if BEAM_DEBUG_DEPTH_FLOW_TRACE
    std::uint32_t* stream3_threshold_pass_count_by_ring = nullptr;
    std::uint32_t* stream3_unique_count_by_ring = nullptr;
#endif
    CandidateMeta* local_pending_buffer = nullptr;
    std::uint32_t* local_pending_count = nullptr;
    CandidateMeta* remote_send_buffer = nullptr;
    CandidateMeta* remote_recv_buffer = nullptr;
    std::uint32_t* send_count = nullptr;
    std::uint32_t* send_offset = nullptr;
    std::uint32_t* recv_count = nullptr;
    std::uint32_t* recv_offset = nullptr;
    std::uint64_t* stream5_local_round_count = nullptr;
    std::uint64_t* stream5_global_round_count = nullptr;
    CandidateMeta* survivor_shard = nullptr;
    Hash128* stream4_key_a = nullptr;
    Hash128* stream4_key_b = nullptr;
    CandidateMeta* stream4_val_a = nullptr;
    CandidateMeta* stream4_val_b = nullptr;
    std::uint32_t* stream4_score_key_a = nullptr;
    std::uint32_t* stream4_score_key_b = nullptr;
    std::uint64_t* stream4_score_count_a = nullptr;
    std::uint64_t* stream4_score_count_b = nullptr;
    std::uint32_t* stream4_keep_flags = nullptr;
    std::uint32_t* stream4_block_counts = nullptr;
    std::uint32_t* stream4_block_offsets = nullptr;
    std::uint32_t* stream4_count = nullptr;
    void* stream4_cub_temp = nullptr;
    std::size_t stream4_cub_temp_bytes = 0;
    std::uint32_t* clean_count = nullptr;
    std::uint32_t* dirty_count = nullptr;
    std::uint32_t* processing_flag = nullptr;
    CandidateMeta* global_spill_buffer_a = nullptr;
    CandidateMeta* global_spill_buffer_b = nullptr;
    std::uint32_t* global_spill_count = nullptr;
    std::uint32_t* global_spill_active_index = nullptr;
    std::uint32_t* fatal_error_flag = nullptr;
    std::uint64_t* fatal_error_trace = nullptr;
    std::uint32_t* shard_score_hist_a = nullptr;
    std::uint32_t* shard_score_hist_b = nullptr;
    std::uint32_t* shard_score_hist_active_index = nullptr;
    std::uint32_t* threshold_hist_active_snapshot = nullptr;
    std::uint64_t* local_score_hist = nullptr;
    std::uint64_t* global_score_hist = nullptr;
    std::uint32_t* current_threshold = nullptr;
    std::uint32_t* threshold_initialized = nullptr;
    std::uint32_t* current_threshold_active_index = nullptr;
    std::uint32_t* threshold_request_local = nullptr;
    std::uint32_t* threshold_request_global = nullptr;
};

struct LayoutFinalView {
    State128* next_frontier_states_tmp = nullptr;
    std::uint32_t* final_keep_flags = nullptr;
    std::uint32_t* final_block_counts = nullptr;
    std::uint32_t* final_block_offsets = nullptr;
    CandidateMeta* final_selected_buffer = nullptr;
    CandidateMeta* final_candidate_buffer = nullptr;
    std::uint32_t* final_candidate_count = nullptr;
    FinalRequest* final_request_buffer = nullptr;
    std::uint32_t* final_request_count = nullptr;
    FinalResponse* final_response_buffer = nullptr;
    FinalRequestValidationError* final_validation_error = nullptr;
    std::uint32_t* final_mat_key_a = nullptr;
    std::uint32_t* final_mat_key_b = nullptr;
    FinalRequest* final_mat_request_a = nullptr;
    FinalRequest* final_mat_request_b = nullptr;
    FinalRequest* final_mat_request_recv = nullptr;
    FinalResponse* final_mat_response_send = nullptr;
    FinalResponse* final_mat_response_recv = nullptr;
    FinalHistoryRecord* final_mat_history_send = nullptr;
    FinalHistoryRecord* final_mat_history_recv = nullptr;
    void* final_mat_cub_temp = nullptr;
    std::size_t final_mat_cub_temp_bytes = 0;
    std::uint32_t* final_send_count = nullptr;
    std::uint32_t* final_send_offset = nullptr;
    std::uint32_t* final_recv_count = nullptr;
    std::uint32_t* final_recv_offset = nullptr;
};

struct StaticDeviceMemory {
    void* allocation = nullptr;
    std::size_t allocation_bytes = 0;
    State128* current_frontier_states = nullptr;
    std::uint32_t* solved_flag = nullptr;
    std::uint32_t* stop_flag = nullptr;
    std::uint32_t* solved_count = nullptr;
    std::uint32_t* solved_overflow = nullptr;
    std::uint32_t* global_stop_flag = nullptr;
    CandidateMeta* solved_meta_list = nullptr;
    std::uint32_t* solved_depth_list = nullptr;
    std::uint32_t* solved_suffix_list = nullptr;
    std::uint32_t* current_depth = nullptr;
    void* scratch_pool = nullptr;
    std::size_t scratch_pool_bytes = 0;
    LayoutStreamsView streams;
    LayoutFinalView final;
};

StaticMemoryPlan make_static_memory_plan(const RuntimeConfig& config);
void allocate_static_device_memory(const StaticMemoryPlan& plan, StaticDeviceMemory& memory);
void free_static_device_memory(StaticDeviceMemory& memory);

} // namespace beam
