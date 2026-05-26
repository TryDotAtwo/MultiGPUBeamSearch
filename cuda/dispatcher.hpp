#pragma once

#include "static_memory.hpp"
#include "stream1.hpp"
#include "stream2.hpp"

#include <cuda_runtime.h>
#include <nccl.h>

#include <array>
#include <cstdint>
#include <vector>

namespace beam {

struct DispatcherStreams {
    cudaStream_t stream1 = nullptr;
    cudaStream_t stream2 = nullptr;
    cudaStream_t stream3 = nullptr;
    cudaStream_t stream4 = nullptr;
    cudaStream_t stream5 = nullptr;
    std::vector<cudaStream_t> stream4_slot_streams;
    std::vector<cudaEvent_t> stream4_slot_done;
};

struct DispatcherEvents {
    cudaEvent_t stream1_done = nullptr;
    cudaEvent_t stream2_done = nullptr;
    cudaEvent_t stream3_done = nullptr;
};

struct CudaGraphJobTemplates {
    std::vector<cudaGraph_t> ring_slot_graphs;
    std::vector<cudaGraphExec_t> ring_slot_execs;
    std::vector<cudaGraph_t> stream3_ring_graphs;
    std::vector<cudaGraphExec_t> stream3_ring_execs;
    std::vector<cudaGraph_t> stream4_shard_graphs;
    std::vector<cudaGraphExec_t> stream4_shard_execs;
};

struct DispatcherDeviceTables {
    const std::uint8_t* generators = nullptr;
    const State128* central_state = nullptr;
    const Hash128* zobrist = nullptr;
};

struct DispatcherNetwork {
    Stream1NetworkView view;
    Stream1CutlassScratch scratch;
};

struct DispatcherCollective {
    ncclComm_t comm = nullptr;
};

struct GeneratedTrackRequest {
    bool enabled = false;
    std::uint64_t parent_idx = UINT64_MAX;
    std::uint8_t move = 0;
};

struct GeneratedTrackResult {
    bool enabled = false;
    bool found = false;
    std::uint64_t request_parent_idx = UINT64_MAX;
    std::uint8_t request_move = 0;
    std::uint32_t ring = UINT32_MAX;
    std::uint32_t ring_slot = UINT32_MAX;
    std::uint32_t job = UINT32_MAX;
    std::uint64_t parent_base = 0;
    std::uint32_t count = 0;
    std::uint32_t parent_local = UINT32_MAX;
    std::uint64_t payload_id = UINT64_MAX;
    std::uint64_t score_ring_offset = UINT64_MAX;
    std::uint32_t score_key = UINT32_MAX;
    Hash128 hash{UINT64_MAX, UINT64_MAX};
    std::uint8_t owner = UINT8_MAX;
    std::uint32_t shard = UINT32_MAX;
    std::uint32_t current_threshold = UINT32_MAX;
    bool parent_state_copied = false;
    State128 parent_state{};
    bool all_move_scores_copied = false;
    std::array<std::uint32_t, MOVE_COUNT> move_score_keys{};
};

struct Stream4TrackResult {
    bool enabled = false;
    Hash128 hash{UINT64_MAX, UINT64_MAX};
    std::uint32_t score_key = UINT32_MAX;
    std::uint32_t shard = UINT32_MAX;

    bool after_stream3_scanned = false;
    bool after_stream3_found = false;
    std::uint32_t after_stream3_location = UINT32_MAX;
    std::uint32_t after_stream3_local = UINT32_MAX;
    std::uint32_t after_stream3_clean_count = 0;
    std::uint32_t after_stream3_dirty_count = 0;
    std::uint32_t after_stream3_active_spill_count = 0;
    std::uint32_t after_stream3_inactive_spill_count = 0;
    std::uint32_t after_stream3_threshold = UINT32_MAX;

    std::uint32_t input_scan_count = 0;
    bool input_found = false;
    std::uint32_t input_slot = UINT32_MAX;
    std::uint32_t input_job = UINT32_MAX;
    std::uint32_t input_location = UINT32_MAX;
    std::uint32_t input_local = UINT32_MAX;
    std::uint32_t input_clean_count = 0;
    std::uint32_t input_dirty_count = 0;
    std::uint32_t input_threshold = UINT32_MAX;

    std::uint32_t output_scan_count = 0;
    bool output_found = false;
    std::uint32_t output_slot = UINT32_MAX;
    std::uint32_t output_job = UINT32_MAX;
    std::uint32_t output_local = UINT32_MAX;
    std::uint32_t output_clean_count = 0;
    std::uint32_t output_dirty_count = 0;
    std::uint32_t output_threshold = UINT32_MAX;
};

struct Stream4TrackEvent {
    std::uint32_t phase = 0;
    bool found = false;
    std::uint32_t shard = UINT32_MAX;
    std::uint32_t slot = UINT32_MAX;
    std::uint32_t job = UINT32_MAX;
    std::uint32_t location = UINT32_MAX;
    std::uint32_t local = UINT32_MAX;
    std::uint32_t clean_count = 0;
    std::uint32_t dirty_count = 0;
    std::uint32_t active_spill_count = 0;
    std::uint32_t inactive_spill_count = 0;
    std::uint32_t threshold = UINT32_MAX;
    std::uint32_t score_key = UINT32_MAX;
};

struct Stream3TrackResult {
    bool enabled = false;
    bool scanned = false;
    bool unique_found = false;
    std::uint32_t unique_local = UINT32_MAX;
    std::uint32_t unique_count = 0;
    std::uint32_t unique_score_key = UINT32_MAX;
    std::uint32_t unique_payload_id = UINT32_MAX;
    std::uint64_t unique_parent_idx = UINT64_MAX;
    std::uint8_t unique_move = UINT8_MAX;

    bool partition_found = false;
    std::uint32_t partition_local = UINT32_MAX;
    std::uint32_t local_pending_count = 0;
    std::uint32_t partition_unique_count = 0;
    std::uint32_t group_offset = UINT32_MAX;
    std::uint32_t group_raw_count = 0;
    std::uint32_t local_in_group = UINT32_MAX;
    std::uint32_t shard_write_count = 0;
    std::uint32_t shard_spill_count = 0;
    std::uint32_t shard_spill_offset = UINT32_MAX;
    std::uint32_t spill_idx = UINT32_MAX;
    std::uint32_t spill_capacity = 0;
    bool spill_capacity_drop = false;

    std::uint32_t clean_count = 0;
    std::uint32_t dirty_count = 0;
    std::uint32_t processing_flag = 0;
    std::uint32_t active_spill_count = 0;
    std::uint32_t inactive_spill_count = 0;
};

struct DepthDispatchState {
    std::uint64_t frontier_cursor = 0;
    std::uint64_t frontier_size = 0;
    std::uint32_t ring_slot_jobs_launched = 0;
    std::uint32_t stream3_jobs_launched = 0;
    std::uint32_t stream4_jobs_launched = 0;
    std::uint32_t stream4_active_sort_slots_used = 0;
    std::uint32_t threshold_updates = 0;
    double stream12_ms_total = 0.0;
    double stream12_ms_max = 0.0;
    double stream3_ring_ms_total = 0.0;
    double stream3_ring_ms_max = 0.0;
    double stream3_spill_drain_ms_total = 0.0;
    double stream4_ms_total = 0.0;
    double stream4_ms_max = 0.0;
    double stream5_ms_total = 0.0;
    std::uint32_t stream4_pending_shards_max = 0;
    std::uint32_t stream4_busy_slots_max = 0;
    std::uint32_t global_spill_peak = 0;
    bool depth_drained = false;
    bool stop_requested = false;
    GeneratedTrackResult tracked_generated;
    Stream3TrackResult tracked_stream3;
    Stream4TrackResult tracked_stream4;
    std::vector<Stream4TrackEvent> tracked_stream4_events;
};

struct FinalizeDepthState {
    std::uint64_t next_frontier_size = 0;
    std::uint32_t final_threshold = 0;
    std::uint32_t final_candidate_count = 0;
    std::uint32_t final_request_count = 0;
    bool tracked_prefinal_enabled = false;
    std::uint32_t tracked_prefinal_matches = 0;
    std::uint32_t tracked_prefinal_best_score_key = UINT32_MAX;
    std::uint64_t tracked_prefinal_first_index = 0;
    std::uint64_t tracked_prefinal_best_index = 0;
    std::uint32_t tracked_prefinal_best_shard = UINT32_MAX;
    std::uint32_t tracked_prefinal_best_local = UINT32_MAX;
    std::uint64_t tracked_prefinal_best_parent_idx = UINT64_MAX;
    std::uint32_t tracked_prefinal_best_route_packed = UINT32_MAX;
    double stream5_threshold_ms = 0.0;
    double stream3_final_filter_ms = 0.0;
    double stream3_final_materialize_ms = 0.0;
    double stream3_reset_ms = 0.0;
};

void create_dispatcher_streams(DispatcherStreams& streams);
void destroy_dispatcher_streams(DispatcherStreams& streams);
void create_dispatcher_events(DispatcherEvents& events);
void destroy_dispatcher_events(DispatcherEvents& events);

void instantiate_cuda_graph_job_templates(
    const StaticMemoryPlan& plan,
    StaticDeviceMemory& memory,
    const DispatcherDeviceTables& tables,
    const DispatcherNetwork& network,
    Stream2SolvedBuffers solved,
    DispatcherStreams& streams,
    DispatcherEvents& events,
    CudaGraphJobTemplates& graphs);

void destroy_cuda_graph_job_templates(CudaGraphJobTemplates& graphs);

DepthDispatchState run_depth_cuda_graphs(
    const StaticMemoryPlan& plan,
    StaticDeviceMemory& memory,
    CudaGraphJobTemplates& graphs,
    DispatcherStreams& streams,
    std::uint64_t frontier_size,
    GeneratedTrackRequest track_request = {},
    const DispatcherCollective* collective = nullptr);

FinalizeDepthState finalize_depth_single_gpu(
    const StaticMemoryPlan& plan,
    StaticDeviceMemory& memory,
    const DispatcherDeviceTables& tables,
    DispatcherStreams& streams,
    std::uint64_t current_frontier_size,
    CandidateMeta* history_host_buffer = nullptr,
    std::uint32_t history_host_capacity = 0,
    cudaStream_t history_stream = nullptr,
    cudaEvent_t history_copy_done = nullptr,
    const Hash128* tracked_prefinal_hash = nullptr,
    const DispatcherCollective* collective = nullptr);

} // namespace beam
