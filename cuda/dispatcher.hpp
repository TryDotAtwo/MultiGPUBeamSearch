#pragma once

#include "static_memory.hpp"
#include "stream1.hpp"
#include "stream2.hpp"

#include <cuda_runtime.h>

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

struct DepthDispatchState {
    std::uint64_t frontier_cursor = 0;
    std::uint64_t frontier_size = 0;
    std::uint32_t ring_slot_jobs_launched = 0;
    std::uint32_t stream3_jobs_launched = 0;
    std::uint32_t stream4_jobs_launched = 0;
    std::uint32_t stream4_active_sort_slots_used = 0;
    std::uint32_t threshold_updates = 0;
    bool depth_drained = false;
    bool stop_requested = false;
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
    std::uint64_t frontier_size);

FinalizeDepthState finalize_depth_single_gpu(
    const StaticMemoryPlan& plan,
    StaticDeviceMemory& memory,
    const DispatcherDeviceTables& tables,
    DispatcherStreams& streams,
    CandidateMeta* history_host_buffer = nullptr,
    std::uint32_t history_host_capacity = 0,
    cudaStream_t history_stream = nullptr,
    cudaEvent_t history_copy_done = nullptr,
    const Hash128* tracked_prefinal_hash = nullptr);

} // namespace beam
