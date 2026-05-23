#include "dispatcher.hpp"

#include "config.hpp"
#include "final_materialize.hpp"
#include "nvtx_ranges.hpp"
#include "stream3.hpp"
#include "stream4.hpp"
#include "threshold.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <stdexcept>
#include <string>
#include <thread>

namespace beam {

namespace {

void check_cuda(cudaError_t status, const char* op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
    }
}

void instantiate_captured_graph(cudaStream_t stream, cudaGraph_t& graph, cudaGraphExec_t& exec) {
    check_cuda(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    check_cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "cudaGraphInstantiate");
}

void ensure_stream4_slot_resources(DispatcherStreams& streams, std::uint32_t slot_count) {
    if (streams.stream4_slot_streams.size() == slot_count && streams.stream4_slot_done.size() == slot_count) {
        return;
    }
    for (cudaStream_t stream : streams.stream4_slot_streams) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
    for (cudaEvent_t event : streams.stream4_slot_done) {
        if (event) {
            cudaEventDestroy(event);
        }
    }
    streams.stream4_slot_streams.assign(slot_count, nullptr);
    streams.stream4_slot_done.assign(slot_count, nullptr);
    for (std::uint32_t slot = 0; slot < slot_count; ++slot) {
        check_cuda(cudaStreamCreateWithFlags(&streams.stream4_slot_streams[slot], cudaStreamNonBlocking), "cudaStreamCreate stream4 slot");
        check_cuda(cudaEventCreateWithFlags(&streams.stream4_slot_done[slot], cudaEventDisableTiming), "cudaEventCreate stream4 slot done");
    }
}

void update_threshold_single_gpu(
    const StaticMemoryPlan& plan,
    StaticDeviceMemory& memory,
    cudaStream_t stream,
    bool periodic) {
    const std::uint64_t threshold_width = plan.derived.global_beam_width_effective;
    threshold_build_local_histogram_cuda(
        memory.streams.shard_score_hist_a,
        memory.streams.shard_score_hist_b,
        memory.streams.shard_score_hist_active_index,
        memory.streams.threshold_hist_active_snapshot,
        memory.streams.local_score_hist,
        plan.config.shard_count,
        stream);
    check_cuda(cudaMemcpyAsync(
        memory.streams.global_score_hist,
        memory.streams.local_score_hist,
        SCORE_BIN_COUNT * sizeof(std::uint64_t),
        cudaMemcpyDeviceToDevice,
        stream), "cudaMemcpyAsync single gpu histogram");
    if (periodic) {
        threshold_update_periodic_cuda(
            memory.streams.global_score_hist,
            memory.streams.current_threshold,
            memory.streams.threshold_initialized,
            threshold_width,
            stream);
    } else {
        threshold_select_cuda(
            memory.streams.global_score_hist,
            memory.streams.current_threshold,
            threshold_width,
            stream);
    }
}

} // namespace

void create_dispatcher_streams(DispatcherStreams& streams) {
    NvtxRange range("Dispatcher_create_streams");
    check_cuda(cudaStreamCreateWithFlags(&streams.stream1, cudaStreamNonBlocking), "cudaStreamCreate stream1");
    check_cuda(cudaStreamCreateWithFlags(&streams.stream2, cudaStreamNonBlocking), "cudaStreamCreate stream2");
    check_cuda(cudaStreamCreateWithFlags(&streams.stream3, cudaStreamNonBlocking), "cudaStreamCreate stream3");
    check_cuda(cudaStreamCreateWithFlags(&streams.stream4, cudaStreamNonBlocking), "cudaStreamCreate stream4");
    check_cuda(cudaStreamCreateWithFlags(&streams.stream5, cudaStreamNonBlocking), "cudaStreamCreate stream5");
}

void destroy_dispatcher_streams(DispatcherStreams& streams) {
    if (streams.stream1) {
        cudaStreamDestroy(streams.stream1);
    }
    if (streams.stream2) {
        cudaStreamDestroy(streams.stream2);
    }
    if (streams.stream3) {
        cudaStreamDestroy(streams.stream3);
    }
    if (streams.stream4) {
        cudaStreamDestroy(streams.stream4);
    }
    if (streams.stream5) {
        cudaStreamDestroy(streams.stream5);
    }
    for (cudaStream_t stream : streams.stream4_slot_streams) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
    for (cudaEvent_t event : streams.stream4_slot_done) {
        if (event) {
            cudaEventDestroy(event);
        }
    }
    streams = DispatcherStreams{};
}

void create_dispatcher_events(DispatcherEvents& events) {
    check_cuda(cudaEventCreateWithFlags(&events.stream1_done, cudaEventDisableTiming), "cudaEventCreate stream1_done");
    check_cuda(cudaEventCreateWithFlags(&events.stream2_done, cudaEventDisableTiming), "cudaEventCreate stream2_done");
    check_cuda(cudaEventCreateWithFlags(&events.stream3_done, cudaEventDisableTiming), "cudaEventCreate stream3_done");
}

void destroy_dispatcher_events(DispatcherEvents& events) {
    if (events.stream1_done) {
        cudaEventDestroy(events.stream1_done);
    }
    if (events.stream2_done) {
        cudaEventDestroy(events.stream2_done);
    }
    if (events.stream3_done) {
        cudaEventDestroy(events.stream3_done);
    }
    events = DispatcherEvents{};
}

void instantiate_cuda_graph_job_templates(
    const StaticMemoryPlan& plan,
    StaticDeviceMemory& memory,
    const DispatcherDeviceTables& tables,
    const DispatcherNetwork& network,
    Stream2SolvedBuffers solved,
    DispatcherStreams& streams,
    DispatcherEvents& events,
    CudaGraphJobTemplates& graphs) {
    NvtxRange range("Dispatcher_instantiate_cuda_graph_job_templates");
    if (!memory.current_frontier_states || !memory.scratch_pool || !tables.generators || !tables.central_state || !tables.zobrist) {
        throw std::invalid_argument("dispatcher graph templates require preallocated architecture memory and read-only tables");
    }
    ensure_stream4_slot_resources(streams, plan.config.stream4_active_sort_slots);

    const std::uint32_t ring_slot_job_count = plan.config.ring_count * plan.derived.ring_slot_count;
    graphs.ring_slot_graphs.resize(ring_slot_job_count, nullptr);
    graphs.ring_slot_execs.resize(ring_slot_job_count, nullptr);
    const std::uint64_t candidates_per_slot = static_cast<std::uint64_t>(plan.config.b_micro) * MOVE_COUNT;
    for (std::uint32_t job = 0; job < ring_slot_job_count; ++job) {
        const std::uint64_t candidate_offset = static_cast<std::uint64_t>(job) * candidates_per_slot;
        check_cuda(cudaStreamBeginCapture(streams.stream1, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture ring_slot_graph");
        check_cuda(cudaEventRecord(events.stream1_done, streams.stream1), "cudaEventRecord ring_slot_fork");
        check_cuda(cudaStreamWaitEvent(streams.stream2, events.stream1_done, 0), "cudaStreamWaitEvent stream2_fork");
        stream1_inference_cutlass_cuda(
            memory.current_frontier_states,
            memory.streams.parent_base + job,
            memory.streams.count + job,
            network.view,
            network.scratch,
            memory.streams.score_ring + candidate_offset,
            plan.config.b_micro,
            streams.stream1);
        stream2_hash_goal_cuda(
            memory.current_frontier_states,
            memory.streams.parent_base + job,
            memory.streams.count + job,
            tables.generators,
            tables.central_state,
            tables.zobrist,
            memory.streams.hash_ring + candidate_offset,
            0,
            0,
            plan.config.b_micro,
            0,
            plan.config.local_rank,
            solved,
            streams.stream2);
        check_cuda(cudaEventRecord(events.stream2_done, streams.stream2), "cudaEventRecord ring_slot_join");
        check_cuda(cudaStreamWaitEvent(streams.stream1, events.stream2_done, 0), "cudaStreamWaitEvent stream1_join");
        instantiate_captured_graph(streams.stream1, graphs.ring_slot_graphs[job], graphs.ring_slot_execs[job]);
    }

    graphs.stream3_ring_graphs.resize(plan.config.ring_count, nullptr);
    graphs.stream3_ring_execs.resize(plan.config.ring_count, nullptr);
    for (std::uint32_t ring = 0; ring < plan.config.ring_count; ++ring) {
        const std::uint64_t ring_candidate_offset = static_cast<std::uint64_t>(ring) * plan.config.stream3_batch_candidates;
        check_cuda(cudaStreamBeginCapture(streams.stream3, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture stream3_ring_graph");
        stream3_pack_threshold_device_threshold_cuda(
            memory.streams.score_ring + ring_candidate_offset,
            memory.streams.hash_ring + ring_candidate_offset,
            memory.streams.parent_base + static_cast<std::uint64_t>(ring) * plan.derived.ring_slot_count,
            memory.streams.count + static_cast<std::uint64_t>(ring) * plan.derived.ring_slot_count,
            memory.streams.stream3_key_a,
            memory.streams.stream3_val_a,
            memory.streams.stream3_key_b,
            memory.streams.stream3_val_b,
            memory.streams.unique_key,
            memory.streams.unique_val,
            memory.streams.stream3_keep_flags,
            memory.streams.stream3_block_counts,
            memory.streams.stream3_block_offsets,
            memory.streams.unique_count,
            memory.streams.stream3_cub_temp,
            memory.streams.stream3_cub_temp_bytes,
            memory.streams.current_threshold,
            plan.config.b_micro,
            plan.config.stream3_batch_candidates,
            streams.stream3);
        if (plan.config.world_size != 1U) {
            stream3_restore_owner_split_cuda(
                memory.streams.unique_key,
                memory.streams.unique_val,
                memory.streams.unique_count,
                memory.streams.parent_base + static_cast<std::uint64_t>(ring) * plan.derived.ring_slot_count,
                memory.streams.local_pending_buffer,
                memory.streams.local_pending_count,
                memory.streams.remote_send_buffer,
                memory.streams.send_count,
                memory.streams.send_offset,
                memory.streams.stream3_owner,
                static_cast<std::uint16_t>(plan.config.local_rank),
                plan.config.world_size,
                plan.config.b_micro,
                plan.config.stream3_batch_candidates,
                streams.stream3);
        }
        stream3_drain_global_spill_cuda(
            memory.streams.global_spill_buffer_a,
            memory.streams.global_spill_buffer_b,
            memory.streams.global_spill_count,
            memory.streams.global_spill_active_index,
            memory.streams.survivor_shard,
            memory.streams.clean_count,
            memory.streams.dirty_count,
            memory.streams.processing_flag,
            memory.streams.stream3_shard_counts,
            memory.streams.stream3_shard_offsets,
            memory.streams.stream3_spill_counts,
            memory.streams.stream3_spill_offsets,
            memory.streams.stream3_partition_key_a,
            memory.streams.stream3_partition_key_b,
            memory.streams.stream3_partition_val_a,
            memory.streams.stream3_partition_val_b,
            memory.streams.stream3_partition_unique_shard,
            memory.streams.stream3_partition_unique_counts,
            memory.streams.stream3_partition_unique_count,
            memory.streams.stream3_cub_temp,
            memory.streams.stream3_cub_temp_bytes,
            plan.config.shard_count,
            plan.config.global_spill_capacity,
            plan.config.stream4_batch_candidates,
            streams.stream3);
        if (plan.config.world_size == 1U) {
            stream3_restore_collect_single_owner_cuda(
                memory.streams.unique_key,
                memory.streams.unique_val,
                memory.streams.unique_count,
                memory.streams.parent_base + static_cast<std::uint64_t>(ring) * plan.derived.ring_slot_count,
                memory.streams.local_pending_count,
                memory.streams.send_count,
                memory.streams.send_offset,
                memory.streams.survivor_shard,
                memory.streams.clean_count,
                memory.streams.dirty_count,
                memory.streams.processing_flag,
                memory.streams.global_spill_buffer_a,
                memory.streams.global_spill_buffer_b,
                memory.streams.global_spill_count,
                memory.streams.global_spill_active_index,
                memory.streams.stream3_shard_counts,
                memory.streams.stream3_shard_offsets,
                memory.streams.stream3_spill_counts,
                memory.streams.stream3_spill_offsets,
                memory.streams.stream3_partition_key_a,
                memory.streams.stream3_partition_key_b,
                memory.streams.stream3_partition_val_a,
                memory.streams.stream3_partition_val_b,
                memory.streams.stream3_partition_unique_shard,
                memory.streams.stream3_partition_unique_counts,
                memory.streams.stream3_partition_unique_count,
                memory.streams.stream3_cub_temp,
                memory.streams.stream3_cub_temp_bytes,
                static_cast<std::uint16_t>(plan.config.local_rank),
                plan.config.b_micro,
                plan.config.stream3_batch_candidates,
                plan.config.shard_count,
                plan.config.stream4_batch_candidates,
                plan.config.global_spill_capacity,
                streams.stream3);
        } else {
            stream3_collect_local_pending_cuda(
                memory.streams.local_pending_buffer,
                memory.streams.local_pending_count,
                memory.streams.survivor_shard,
                memory.streams.clean_count,
                memory.streams.dirty_count,
                memory.streams.processing_flag,
                memory.streams.global_spill_buffer_a,
                memory.streams.global_spill_buffer_b,
                memory.streams.global_spill_count,
                memory.streams.global_spill_active_index,
                memory.streams.stream3_shard_counts,
                memory.streams.stream3_shard_offsets,
                memory.streams.stream3_spill_counts,
                memory.streams.stream3_spill_offsets,
                memory.streams.stream3_partition_key_a,
                memory.streams.stream3_partition_key_b,
                memory.streams.stream3_partition_val_a,
                memory.streams.stream3_partition_val_b,
                memory.streams.stream3_partition_unique_shard,
                memory.streams.stream3_partition_unique_counts,
                memory.streams.stream3_partition_unique_count,
                memory.streams.stream3_cub_temp,
                memory.streams.stream3_cub_temp_bytes,
                plan.config.stream3_batch_candidates,
                plan.config.shard_count,
                plan.config.stream4_batch_candidates,
                plan.config.global_spill_capacity,
                streams.stream3);
        }
        stream3_build_ready_shard_queue_cuda(
            memory.streams.clean_count,
            memory.streams.dirty_count,
            memory.streams.processing_flag,
            memory.streams.stream3_ready_flag,
            memory.streams.stream3_ready_shard_list,
            memory.streams.stream3_ready_count,
            plan.config.shard_count,
            plan.config.stream4_batch_candidates,
            false,
            streams.stream3);
        instantiate_captured_graph(streams.stream3, graphs.stream3_ring_graphs[ring], graphs.stream3_ring_execs[ring]);
    }

    const std::uint32_t stream4_slot_count = plan.config.stream4_active_sort_slots;
    graphs.stream4_shard_graphs.resize(
        static_cast<std::uint64_t>(plan.config.shard_count) * stream4_slot_count,
        nullptr);
    graphs.stream4_shard_execs.resize(
        static_cast<std::uint64_t>(plan.config.shard_count) * stream4_slot_count,
        nullptr);
    const std::uint32_t stream4_capacity = 2U * plan.config.stream4_batch_candidates;
    const std::uint32_t stream4_block_count = (stream4_capacity + 255U) / 256U;
    for (std::uint32_t shard = 0; shard < plan.config.shard_count; ++shard) {
        const std::uint64_t shard_candidate_offset = static_cast<std::uint64_t>(shard) * stream4_capacity;
        for (std::uint32_t slot = 0; slot < stream4_slot_count; ++slot) {
            const std::uint64_t graph_idx = static_cast<std::uint64_t>(shard) * stream4_slot_count + slot;
            const std::uint64_t slot_candidate_offset = static_cast<std::uint64_t>(slot) * stream4_capacity;
            const std::uint64_t slot_block_offset = static_cast<std::uint64_t>(slot) * stream4_block_count;
            auto* slot_cub_temp =
                static_cast<std::byte*>(memory.streams.stream4_cub_temp) +
                static_cast<std::uint64_t>(slot) * memory.streams.stream4_cub_temp_bytes;
            check_cuda(cudaStreamBeginCapture(streams.stream4, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture stream4_shard_graph");
            stream4_shard_job_device_threshold_cuda(
                memory.streams.survivor_shard + shard_candidate_offset,
                memory.streams.clean_count + shard,
                memory.streams.dirty_count + shard,
                memory.streams.processing_flag + shard,
                memory.streams.current_threshold,
                stream4_capacity,
                memory.streams.stream4_key_a + slot_candidate_offset,
                memory.streams.stream4_key_b + slot_candidate_offset,
                memory.streams.stream4_val_a + slot_candidate_offset,
                memory.streams.stream4_val_b + slot_candidate_offset,
                memory.streams.stream4_score_key_a + slot_candidate_offset,
                memory.streams.stream4_score_key_b + slot_candidate_offset,
                memory.streams.stream4_score_count_a + slot_candidate_offset,
                memory.streams.stream4_score_count_b + slot_candidate_offset,
                memory.streams.stream4_keep_flags + slot_candidate_offset,
                memory.streams.stream4_block_counts + slot_block_offset,
                memory.streams.stream4_block_offsets + slot_block_offset,
                memory.streams.stream4_count + slot,
                memory.streams.shard_score_hist_a + static_cast<std::uint64_t>(shard) * SCORE_BIN_COUNT,
                memory.streams.shard_score_hist_b + static_cast<std::uint64_t>(shard) * SCORE_BIN_COUNT,
                memory.streams.shard_score_hist_active_index + shard,
                slot_cub_temp,
                memory.streams.stream4_cub_temp_bytes,
                streams.stream4);
            instantiate_captured_graph(
                streams.stream4,
                graphs.stream4_shard_graphs[graph_idx],
                graphs.stream4_shard_execs[graph_idx]);
        }
    }
}

void destroy_cuda_graph_job_templates(CudaGraphJobTemplates& graphs) {
    for (cudaGraphExec_t exec : graphs.ring_slot_execs) {
        if (exec) {
            cudaGraphExecDestroy(exec);
        }
    }
    for (cudaGraph_t graph : graphs.ring_slot_graphs) {
        if (graph) {
            cudaGraphDestroy(graph);
        }
    }
    for (cudaGraphExec_t exec : graphs.stream3_ring_execs) {
        if (exec) {
            cudaGraphExecDestroy(exec);
        }
    }
    for (cudaGraph_t graph : graphs.stream3_ring_graphs) {
        if (graph) {
            cudaGraphDestroy(graph);
        }
    }
    for (cudaGraphExec_t exec : graphs.stream4_shard_execs) {
        if (exec) {
            cudaGraphExecDestroy(exec);
        }
    }
    for (cudaGraph_t graph : graphs.stream4_shard_graphs) {
        if (graph) {
            cudaGraphDestroy(graph);
        }
    }
    graphs = CudaGraphJobTemplates{};
}

DepthDispatchState run_depth_cuda_graphs(
    const StaticMemoryPlan& plan,
    StaticDeviceMemory& memory,
    CudaGraphJobTemplates& graphs,
    DispatcherStreams& streams,
    std::uint64_t frontier_size) {
    NvtxRange range("Dispatcher_depth_cuda_graphs");
    const std::uint32_t ring_slot_job_count = plan.config.ring_count * plan.derived.ring_slot_count;
    if (graphs.ring_slot_execs.size() != ring_slot_job_count ||
        graphs.stream3_ring_execs.size() != plan.config.ring_count ||
        graphs.stream4_shard_execs.size() !=
            static_cast<std::uint64_t>(plan.config.shard_count) * plan.config.stream4_active_sort_slots) {
        throw std::invalid_argument("depth dispatcher graph template counts do not match static memory plan");
    }

    DepthDispatchState state;
    state.frontier_size = frontier_size;
    std::vector<std::uint32_t> host_dirty(plan.config.shard_count);
    std::vector<std::uint32_t> host_ready_shards(plan.config.shard_count);
    std::vector<std::uint32_t> pending_stream4_shards;
    pending_stream4_shards.reserve(plan.config.shard_count * plan.config.ring_count);
    std::uint32_t pending_stream4_head = 0;
    std::vector<bool> stream4_slot_busy(plan.config.stream4_active_sort_slots, false);
    std::vector<std::uint32_t> stream4_slot_shard(plan.config.stream4_active_sort_slots, plan.config.shard_count);
    std::deque<std::uint32_t> stream4_free_slots;
    std::deque<std::uint32_t> stream4_busy_slots;
    for (std::uint32_t slot = 0; slot < plan.config.stream4_active_sort_slots; ++slot) {
        stream4_free_slots.push_back(slot);
    }
    std::uint32_t stream4_jobs_since_threshold_update = 0;

    std::vector<cudaEvent_t> ring_done(plan.config.ring_count, nullptr);
    std::vector<cudaEvent_t> stream3_done(plan.config.ring_count, nullptr);
    for (std::uint32_t ring = 0; ring < plan.config.ring_count; ++ring) {
        check_cuda(cudaEventCreateWithFlags(&ring_done[ring], cudaEventDisableTiming), "cudaEventCreate ring done");
        check_cuda(cudaEventCreateWithFlags(&stream3_done[ring], cudaEventDisableTiming), "cudaEventCreate stream3 done");
    }
    struct EventCleanup {
        std::vector<cudaEvent_t>& a;
        std::vector<cudaEvent_t>& b;
        ~EventCleanup() {
            for (cudaEvent_t event : a) {
                if (event) {
                    cudaEventDestroy(event);
                }
            }
            for (cudaEvent_t event : b) {
                if (event) {
                    cudaEventDestroy(event);
                }
            }
        }
    } event_cleanup{ring_done, stream3_done};

    const auto refresh_stop_requested = [&]() -> bool {
        if (state.stop_requested) {
            return true;
        }
        std::uint32_t stop_value = 0;
        check_cuda(cudaMemcpy(
            &stop_value,
            memory.stop_flag,
            sizeof(stop_value),
            cudaMemcpyDeviceToHost), "cudaMemcpy stop flag to host scheduler");
        state.stop_requested = stop_value != 0U;
        return state.stop_requested;
    };

    enum class RingState : std::uint8_t {
        Free,
        Stream1Running,
        ReadyForStream3,
        Stream3Running
    };
    std::vector<RingState> ring_state(plan.config.ring_count, RingState::Free);
    std::deque<std::uint32_t> stream1_running_rings;
    std::deque<std::uint32_t> stream3_ready_rings;
    bool stream3_active = false;
    std::uint32_t stream3_active_ring = plan.config.ring_count;

    const auto mark_stream4_slot_complete = [&](std::uint32_t slot) {
        stream4_slot_busy[slot] = false;
        stream4_slot_shard[slot] = plan.config.shard_count;
        stream4_free_slots.push_back(slot);
        ++stream4_jobs_since_threshold_update;
    };

    const auto release_completed_stream4_slots_nonblocking = [&]() -> bool {
        bool progressed = false;
        while (!stream4_busy_slots.empty()) {
            const std::uint32_t slot = stream4_busy_slots.front();
            const cudaError_t status = cudaEventQuery(streams.stream4_slot_done[slot]);
            if (status == cudaSuccess) {
                stream4_busy_slots.pop_front();
                mark_stream4_slot_complete(slot);
                progressed = true;
            } else if (status != cudaErrorNotReady) {
                check_cuda(status, "cudaEventQuery stream4 slot done");
            } else {
                break;
            }
        }
        return progressed;
    };

    const auto wait_all_stream4_slots = [&]() {
        while (!stream4_busy_slots.empty()) {
            const std::uint32_t slot = stream4_busy_slots.front();
            check_cuda(cudaEventSynchronize(streams.stream4_slot_done[slot]), "cudaEventSynchronize stream4 drain slot");
            stream4_busy_slots.pop_front();
            mark_stream4_slot_complete(slot);
            release_completed_stream4_slots_nonblocking();
        }
    };

    const auto wait_oldest_stream4_slot = [&]() -> bool {
        if (stream4_busy_slots.empty()) {
            return false;
        }
        const std::uint32_t slot = stream4_busy_slots.front();
        check_cuda(cudaEventSynchronize(streams.stream4_slot_done[slot]), "cudaEventSynchronize stream4 sort slot");
        stream4_busy_slots.pop_front();
        mark_stream4_slot_complete(slot);
        release_completed_stream4_slots_nonblocking();
        return true;
    };

    const auto acquire_stream4_slot_nonblocking = [&]() -> std::uint32_t {
        release_completed_stream4_slots_nonblocking();
        if (!stream4_free_slots.empty()) {
            const std::uint32_t slot = stream4_free_slots.front();
            stream4_free_slots.pop_front();
            return slot;
        }
        return plan.config.stream4_active_sort_slots;
    };

    const auto acquire_stream4_slot_blocking = [&]() -> std::uint32_t {
        const std::uint32_t free_slot = acquire_stream4_slot_nonblocking();
        if (free_slot < plan.config.stream4_active_sort_slots) {
            return free_slot;
        }
        if (!wait_oldest_stream4_slot()) {
            throw std::runtime_error("stream4 blocking slot acquire has no busy slot to wait on");
        }
        const std::uint32_t acquired_slot = stream4_free_slots.front();
        stream4_free_slots.pop_front();
        return acquired_slot;
    };

    const auto launch_free_rings = [&]() -> bool {
        if (state.stop_requested) {
            return false;
        }
        bool launched_any = false;
        for (std::uint32_t ring = 0; ring < plan.config.ring_count && state.frontier_cursor < frontier_size; ++ring) {
            if (ring_state[ring] != RingState::Free) {
                continue;
            }
            ring_state[ring] = RingState::Stream1Running;
            for (std::uint32_t slot = 0; slot < plan.derived.ring_slot_count; ++slot) {
                const std::uint32_t job = ring * plan.derived.ring_slot_count + slot;
                std::uint64_t parent_base_value = 0;
                std::uint32_t count_value = 0;
                if (state.frontier_cursor < frontier_size) {
                    parent_base_value = state.frontier_cursor;
                    const std::uint64_t remaining_for_job = frontier_size - state.frontier_cursor;
                    count_value = static_cast<std::uint32_t>(
                        std::min<std::uint64_t>(plan.config.b_micro, remaining_for_job));
                    state.frontier_cursor += count_value;
                }
                check_cuda(cudaMemcpyAsync(
                    memory.streams.parent_base + job,
                    &parent_base_value,
                    sizeof(parent_base_value),
                    cudaMemcpyHostToDevice,
                    streams.stream1), "cudaMemcpyAsync parent_base");
                check_cuda(cudaMemcpyAsync(
                    memory.streams.count + job,
                    &count_value,
                    sizeof(count_value),
                    cudaMemcpyHostToDevice,
                    streams.stream1), "cudaMemcpyAsync count");
                if (count_value != 0U) {
                    check_cuda(cudaGraphLaunch(graphs.ring_slot_execs[job], streams.stream1), "cudaGraphLaunch ring_slot");
                    ++state.ring_slot_jobs_launched;
                }
            }
            check_cuda(cudaEventRecord(ring_done[ring], streams.stream1), "cudaEventRecord ring score hash done");
            stream1_running_rings.push_back(ring);
            launched_any = true;
        }
        return launched_any;
    };

    const auto compact_pending_stream4_queue = [&]() {
        if (pending_stream4_head == 0U) {
            return;
        }
        if (pending_stream4_head == pending_stream4_shards.size()) {
            pending_stream4_shards.clear();
            pending_stream4_head = 0;
            return;
        }
        if (pending_stream4_head * 2U >= pending_stream4_shards.size()) {
            pending_stream4_shards.erase(
                pending_stream4_shards.begin(),
                pending_stream4_shards.begin() + static_cast<std::ptrdiff_t>(pending_stream4_head));
            pending_stream4_head = 0;
        }
    };

    const auto append_stream3_ready_queue = [&]() -> std::uint32_t {
        std::uint32_t ready_count = 0;
        check_cuda(cudaMemcpy(
            &ready_count,
            memory.streams.stream3_ready_count,
            sizeof(ready_count),
            cudaMemcpyDeviceToHost), "cudaMemcpy stream3 ready count to host scheduler");
        if (ready_count == 0U) {
            return 0;
        }
        if (ready_count > plan.config.shard_count) {
            throw std::runtime_error("stream3 ready shard count exceeds shard count");
        }
        check_cuda(cudaMemcpy(
            host_ready_shards.data(),
            memory.streams.stream3_ready_shard_list,
            static_cast<std::uint64_t>(ready_count) * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost), "cudaMemcpy stream3 ready shard list to host scheduler");
        for (std::uint32_t i = 0; i < ready_count; ++i) {
            const std::uint32_t shard = host_ready_shards[i];
            if (shard >= plan.config.shard_count) {
                throw std::runtime_error("stream3 ready shard index exceeds shard count");
            }
            pending_stream4_shards.push_back(shard);
        }
        return ready_count;
    };

    const auto periodic_threshold_due = [&]() -> bool {
        return plan.config.global_threshold_update_period_shards != 0 &&
            stream4_jobs_since_threshold_update >= plan.config.global_threshold_update_period_shards;
    };

    const auto force_periodic_threshold_update = [&]() {
        if (stream3_active) {
            throw std::runtime_error("periodic threshold update requested while stream3 is active");
        }
        wait_all_stream4_slots();
        update_threshold_single_gpu(plan, memory, streams.stream5, true);
        check_cuda(cudaStreamSynchronize(streams.stream5), "cudaStreamSynchronize stream5 periodic threshold update");
        ++state.threshold_updates;
        stream4_jobs_since_threshold_update = 0;
    };

    const auto launch_pending_stream4_shards = [&]() -> std::uint32_t {
        release_completed_stream4_slots_nonblocking();
        std::uint32_t launched = 0;
        while (pending_stream4_head < pending_stream4_shards.size()) {
            const std::uint32_t slot = acquire_stream4_slot_nonblocking();
            if (slot >= plan.config.stream4_active_sort_slots) {
                break;
            }
            const std::uint32_t shard = pending_stream4_shards[pending_stream4_head];
            ++pending_stream4_head;
            const std::uint64_t graph_idx =
                static_cast<std::uint64_t>(shard) * plan.config.stream4_active_sort_slots + slot;
            check_cuda(
                cudaGraphLaunch(graphs.stream4_shard_execs[graph_idx], streams.stream4_slot_streams[slot]),
                "cudaGraphLaunch stream4");
            check_cuda(
                cudaEventRecord(streams.stream4_slot_done[slot], streams.stream4_slot_streams[slot]),
                "cudaEventRecord stream4 slot done");
            stream4_slot_busy[slot] = true;
            stream4_slot_shard[slot] = shard;
            stream4_busy_slots.push_back(slot);
            ++launched;
            ++state.stream4_jobs_launched;
            state.stream4_active_sort_slots_used = std::max(state.stream4_active_sort_slots_used, slot + 1U);
        }
        compact_pending_stream4_queue();
        return launched;
    };

    const auto drain_pending_stream4_shards = [&]() -> std::uint32_t {
        std::uint32_t launched = 0;
        while (pending_stream4_head < pending_stream4_shards.size()) {
            launched += launch_pending_stream4_shards();
            if (pending_stream4_head < pending_stream4_shards.size()) {
                const std::uint32_t slot = acquire_stream4_slot_blocking();
                const std::uint32_t shard = pending_stream4_shards[pending_stream4_head];
                ++pending_stream4_head;
                const std::uint64_t graph_idx =
                    static_cast<std::uint64_t>(shard) * plan.config.stream4_active_sort_slots + slot;
                check_cuda(
                    cudaGraphLaunch(graphs.stream4_shard_execs[graph_idx], streams.stream4_slot_streams[slot]),
                    "cudaGraphLaunch stream4");
                check_cuda(
                    cudaEventRecord(streams.stream4_slot_done[slot], streams.stream4_slot_streams[slot]),
                    "cudaEventRecord stream4 slot done");
                stream4_slot_busy[slot] = true;
                stream4_slot_shard[slot] = shard;
                stream4_busy_slots.push_back(slot);
                ++launched;
                ++state.stream4_jobs_launched;
                state.stream4_active_sort_slots_used = std::max(state.stream4_active_sort_slots_used, slot + 1U);
            }
            compact_pending_stream4_queue();
        }
        wait_all_stream4_slots();
        return launched;
    };

    const auto any_active_ring = [&]() -> bool {
        for (RingState value : ring_state) {
            if (value != RingState::Free) {
                return true;
            }
        }
        return false;
    };

    const auto discard_ready_rings_after_stop = [&]() -> bool {
        bool progressed = false;
        while (!stream3_ready_rings.empty()) {
            const std::uint32_t ring = stream3_ready_rings.front();
            stream3_ready_rings.pop_front();
            if (ring_state[ring] == RingState::ReadyForStream3) {
                ring_state[ring] = RingState::Free;
                progressed = true;
            }
        }
        return progressed;
    };

    const auto try_launch_stream3 = [&]() -> bool {
        if (stream3_active || stream3_ready_rings.empty()) {
            return false;
        }
        const std::uint32_t ring = stream3_ready_rings.front();
        stream3_ready_rings.pop_front();
        if (ring_state[ring] != RingState::ReadyForStream3) {
            throw std::runtime_error("stream3 ready queue contains a non-ready ring");
        }
        check_cuda(cudaGraphLaunch(graphs.stream3_ring_execs[ring], streams.stream3), "cudaGraphLaunch stream3");
        check_cuda(cudaEventRecord(stream3_done[ring], streams.stream3), "cudaEventRecord stream3 ring done");
        ring_state[ring] = RingState::Stream3Running;
        stream3_active = true;
        stream3_active_ring = ring;
        ++state.stream3_jobs_launched;
        return true;
    };

    const auto maybe_update_threshold = [&]() -> bool {
        if (!periodic_threshold_due() || stream3_active) {
            return false;
        }
        drain_pending_stream4_shards();
        force_periodic_threshold_update();
        return true;
    };

    const auto release_completed_rings_nonblocking = [&]() -> bool {
        bool progressed = false;
        while (!stream1_running_rings.empty()) {
            const std::uint32_t ring = stream1_running_rings.front();
            const cudaError_t status = cudaEventQuery(ring_done[ring]);
            if (status == cudaSuccess) {
                stream1_running_rings.pop_front();
                ring_state[ring] = RingState::ReadyForStream3;
                stream3_ready_rings.push_back(ring);
                progressed = true;
            } else if (status != cudaErrorNotReady) {
                check_cuda(status, "cudaEventQuery ring done");
            } else {
                break;
            }
        }
        return progressed;
    };

    const auto wait_oldest_ring = [&]() -> bool {
        if (stream1_running_rings.empty()) {
            return false;
        }
        const std::uint32_t ring = stream1_running_rings.front();
        check_cuda(cudaEventSynchronize(ring_done[ring]), "cudaEventSynchronize ring done");
        stream1_running_rings.pop_front();
        ring_state[ring] = RingState::ReadyForStream3;
        stream3_ready_rings.push_back(ring);
        release_completed_rings_nonblocking();
        refresh_stop_requested();
        return true;
    };

    const auto complete_stream3_ring = [&]() {
        if (!stream3_active || stream3_active_ring >= plan.config.ring_count) {
            throw std::runtime_error("stream3 completion without active ring");
        }
        const std::uint32_t ring = stream3_active_ring;
        stream3_active = false;
        stream3_active_ring = plan.config.ring_count;
        ring_state[ring] = RingState::Free;
        if (!state.stop_requested) {
            append_stream3_ready_queue();
        }
    };

    const auto release_stream3_nonblocking = [&]() -> bool {
        if (!stream3_active) {
            return false;
        }
        const cudaError_t status = cudaEventQuery(stream3_done[stream3_active_ring]);
        if (status == cudaSuccess) {
            complete_stream3_ring();
            return true;
        }
        if (status != cudaErrorNotReady) {
            check_cuda(status, "cudaEventQuery stream3 done");
        }
        return false;
    };

    const auto wait_stream3 = [&]() -> bool {
        if (!stream3_active) {
            return false;
        }
        check_cuda(cudaEventSynchronize(stream3_done[stream3_active_ring]), "cudaEventSynchronize stream3 done");
        complete_stream3_ring();
        return true;
    };

    launch_free_rings();
    while ((!state.stop_requested && state.frontier_cursor < frontier_size) ||
        any_active_ring() ||
        stream3_active ||
        !stream4_busy_slots.empty()) {
        bool progressed = false;
        const bool stream4_slots_completed = release_completed_stream4_slots_nonblocking();
        progressed = stream4_slots_completed || progressed;
        const bool rings_completed = release_completed_rings_nonblocking();
        progressed = rings_completed || progressed;
        progressed = release_stream3_nonblocking() || progressed;
        if (rings_completed && refresh_stop_requested()) {
            progressed = discard_ready_rings_after_stop() || progressed;
        }

        if (state.stop_requested) {
            if (!progressed) {
                if (stream3_active) {
                    wait_stream3();
                } else if (!stream1_running_rings.empty()) {
                    wait_oldest_ring();
                } else if (!stream4_busy_slots.empty()) {
                    wait_oldest_stream4_slot();
                } else {
                    discard_ready_rings_after_stop();
                }
            }
            continue;
        }

        if (!stream3_active) {
            progressed = maybe_update_threshold() || progressed;
        }
        const std::uint32_t launched_shards = launch_pending_stream4_shards();
        progressed = (launched_shards != 0U) || progressed;
        progressed = launch_free_rings() || progressed;
        progressed = try_launch_stream3() || progressed;
        if (!stream3_active) {
            progressed = maybe_update_threshold() || progressed;
        }

        if (!progressed) {
            if (stream3_active) {
                wait_stream3();
            } else if (!stream1_running_rings.empty()) {
                wait_oldest_ring();
            } else if (!stream4_busy_slots.empty()) {
                wait_oldest_stream4_slot();
            } else {
                std::this_thread::yield();
            }
        }
    }

    if (state.stop_requested) {
        wait_all_stream4_slots();
        state.depth_drained = true;
        return state;
    }

    drain_pending_stream4_shards();
    force_periodic_threshold_update();

    for (std::uint32_t flush_round = 0; flush_round < plan.config.shard_count + 2U; ++flush_round) {
        stream3_drain_global_spill_cuda(
            memory.streams.global_spill_buffer_a,
            memory.streams.global_spill_buffer_b,
            memory.streams.global_spill_count,
            memory.streams.global_spill_active_index,
            memory.streams.survivor_shard,
            memory.streams.clean_count,
            memory.streams.dirty_count,
            memory.streams.processing_flag,
            memory.streams.stream3_shard_counts,
            memory.streams.stream3_shard_offsets,
            memory.streams.stream3_spill_counts,
            memory.streams.stream3_spill_offsets,
            memory.streams.stream3_partition_key_a,
            memory.streams.stream3_partition_key_b,
            memory.streams.stream3_partition_val_a,
            memory.streams.stream3_partition_val_b,
            memory.streams.stream3_partition_unique_shard,
            memory.streams.stream3_partition_unique_counts,
            memory.streams.stream3_partition_unique_count,
            memory.streams.stream3_cub_temp,
            memory.streams.stream3_cub_temp_bytes,
            plan.config.shard_count,
            plan.config.global_spill_capacity,
            plan.config.stream4_batch_candidates,
            streams.stream3);
        stream3_build_ready_shard_queue_cuda(
            memory.streams.clean_count,
            memory.streams.dirty_count,
            memory.streams.processing_flag,
            memory.streams.stream3_ready_flag,
            memory.streams.stream3_ready_shard_list,
            memory.streams.stream3_ready_count,
            plan.config.shard_count,
            plan.config.stream4_batch_candidates,
            true,
            streams.stream3);
        check_cuda(cudaStreamSynchronize(streams.stream3), "cudaStreamSynchronize final spill drain");
        append_stream3_ready_queue();
        drain_pending_stream4_shards();
        std::uint32_t spill_counts[2]{};
        std::uint32_t spill_active = 0;
        check_cuda(cudaMemcpy(
            spill_counts,
            memory.streams.global_spill_count,
            sizeof(spill_counts),
            cudaMemcpyDeviceToHost), "cudaMemcpy global spill counts final flush");
        check_cuda(cudaMemcpy(
            &spill_active,
            memory.streams.global_spill_active_index,
            sizeof(spill_active),
            cudaMemcpyDeviceToHost), "cudaMemcpy global spill active final flush");
        check_cuda(cudaMemcpy(
            host_dirty.data(),
            memory.streams.dirty_count,
            static_cast<std::uint64_t>(plan.config.shard_count) * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost), "cudaMemcpy dirty_count final flush");
        bool any_dirty = false;
        for (std::uint32_t dirty : host_dirty) {
            any_dirty = any_dirty || dirty != 0U;
        }
        const bool spill_remaining = spill_counts[spill_active & 1U] != 0U || any_dirty;
        if (stream4_jobs_since_threshold_update != 0U &&
            (periodic_threshold_due() || spill_remaining)) {
            force_periodic_threshold_update();
        }
        if (!spill_remaining) {
            break;
        }
        if (flush_round + 1U == plan.config.shard_count + 2U) {
            std::uint32_t debug_threshold = UINT32_THRESHOLD_MAX;
            check_cuda(cudaMemcpy(
                &debug_threshold,
                memory.streams.current_threshold,
                sizeof(debug_threshold),
                cudaMemcpyDeviceToHost), "cudaMemcpy current threshold final flush failure");
            throw std::runtime_error(
                "final stream3 spill flush did not converge: active_spill_count=" +
                std::to_string(spill_counts[spill_active & 1U]) +
                " inactive_spill_count=" + std::to_string(spill_counts[(spill_active + 1U) & 1U]) +
                " any_dirty=" + std::to_string(any_dirty ? 1U : 0U) +
                " current_threshold=" + std::to_string(debug_threshold) +
                " stream4_jobs_since_threshold=" +
                std::to_string(stream4_jobs_since_threshold_update) +
                " threshold_updates=" + std::to_string(state.threshold_updates) +
                " stream4_jobs_launched=" + std::to_string(state.stream4_jobs_launched));
        }
    }
    state.depth_drained = true;
    return state;
}

FinalizeDepthState finalize_depth_single_gpu(
    const StaticMemoryPlan& plan,
    StaticDeviceMemory& memory,
    const DispatcherDeviceTables& tables,
    DispatcherStreams& streams,
    CandidateMeta* history_host_buffer,
    std::uint32_t history_host_capacity,
    cudaStream_t history_stream,
    cudaEvent_t history_copy_done) {
    NvtxRange range("Dispatcher_finalize_depth_single_gpu");
    if (plan.config.world_size != 1 || plan.config.local_rank != 0) {
        throw std::invalid_argument("single gpu finalization requires WORLD_SIZE=1 and LOCAL_RANK=0");
    }
    if (tables.generators == nullptr) {
        throw std::invalid_argument("finalization requires generators");
    }

    update_threshold_single_gpu(plan, memory, streams.stream5, false);
    check_cuda(cudaStreamSynchronize(streams.stream5), "cudaStreamSynchronize stream5 final threshold");
    std::uint32_t final_threshold = UINT32_THRESHOLD_MAX;
    check_cuda(cudaMemcpy(
        &final_threshold,
        memory.streams.current_threshold,
        sizeof(final_threshold),
        cudaMemcpyDeviceToHost), "cudaMemcpy final threshold");

    final_filter_load_balance_cuda(
        memory.streams.survivor_shard,
        memory.streams.clean_count,
        memory.final.final_keep_flags,
        memory.final.final_block_counts,
        memory.final.final_block_offsets,
        memory.final.final_candidate_buffer,
        memory.final.final_candidate_count,
        memory.final.final_request_buffer,
        memory.final.final_request_count,
        memory.final.final_send_count,
        memory.final.final_send_offset,
        final_threshold,
        plan.config.local_rank,
        plan.config.world_size,
        0,
        plan.derived.global_beam_width_effective,
        static_cast<std::uint32_t>(plan.frontier_states),
        plan.config.shard_count,
        plan.config.stream4_batch_candidates,
        streams.stream3);
    check_cuda(cudaStreamSynchronize(streams.stream3), "cudaStreamSynchronize final filter load balance");

    std::uint32_t final_candidate_count = 0;
    std::uint32_t final_request_count = 0;
    check_cuda(cudaMemcpy(
        &final_candidate_count,
        memory.final.final_candidate_count,
        sizeof(final_candidate_count),
        cudaMemcpyDeviceToHost), "cudaMemcpy final candidate count");
    check_cuda(cudaMemcpy(
        &final_request_count,
        memory.final.final_request_count,
        sizeof(final_request_count),
        cudaMemcpyDeviceToHost), "cudaMemcpy final request count");
    if (final_candidate_count > plan.frontier_states || final_request_count > plan.frontier_states) {
        throw std::runtime_error(
            "final output count exceeds allocated local frontier capacity: candidates=" +
            std::to_string(final_candidate_count) +
            " requests=" + std::to_string(final_request_count) +
            " capacity=" + std::to_string(plan.frontier_states));
    }
    if (history_host_buffer != nullptr) {
        if (history_stream == nullptr || history_copy_done == nullptr) {
            throw std::invalid_argument("history copy requires history stream and completion event");
        }
        if (final_candidate_count > history_host_capacity) {
            throw std::runtime_error("history host buffer capacity is smaller than final candidate count");
        }
        if (final_candidate_count != 0U) {
            check_cuda(cudaMemcpyAsync(
                history_host_buffer,
                memory.final.final_candidate_buffer,
                static_cast<std::uint64_t>(final_candidate_count) * sizeof(CandidateMeta),
                cudaMemcpyDeviceToHost,
                history_stream), "cudaMemcpyAsync final candidates to host history");
        }
        check_cuda(cudaEventRecord(history_copy_done, history_stream), "cudaEventRecord history copy done");
    }

    if (final_request_count != 0) {
        final_materialize_cuda(
            memory.current_frontier_states,
            memory.final.final_request_buffer,
            tables.generators,
            memory.final.final_response_buffer,
            memory.final.next_frontier_states_tmp,
            final_request_count,
            streams.stream3);
        check_cuda(cudaMemcpyAsync(
            memory.current_frontier_states,
            memory.final.next_frontier_states_tmp,
            static_cast<std::uint64_t>(final_request_count) * sizeof(State128),
            cudaMemcpyDeviceToDevice,
            streams.stream3), "cudaMemcpyAsync next frontier to current");
    }
    check_cuda(cudaStreamSynchronize(streams.stream3), "cudaStreamSynchronize final materialize");
    check_cuda(cudaMemsetAsync(
        memory.streams.clean_count,
        0,
        static_cast<std::uint64_t>(plan.config.shard_count) * sizeof(std::uint32_t),
        streams.stream3), "cudaMemsetAsync reset clean count");
    check_cuda(cudaMemsetAsync(
        memory.streams.dirty_count,
        0,
        static_cast<std::uint64_t>(plan.config.shard_count) * sizeof(std::uint32_t),
        streams.stream3), "cudaMemsetAsync reset dirty count");
    check_cuda(cudaMemsetAsync(
        memory.streams.processing_flag,
        0,
        static_cast<std::uint64_t>(plan.config.shard_count) * sizeof(std::uint32_t),
        streams.stream3), "cudaMemsetAsync reset processing flag");
    check_cuda(cudaMemsetAsync(
        memory.streams.stream3_ready_flag,
        0,
        static_cast<std::uint64_t>(plan.config.shard_count) * sizeof(std::uint32_t),
        streams.stream3), "cudaMemsetAsync reset stream3 ready flag");
    check_cuda(cudaMemsetAsync(
        memory.streams.stream3_ready_shard_list,
        0,
        static_cast<std::uint64_t>(plan.config.shard_count) * sizeof(std::uint32_t),
        streams.stream3), "cudaMemsetAsync reset stream3 ready shard list");
    check_cuda(cudaMemsetAsync(
        memory.streams.stream3_ready_count,
        0,
        sizeof(std::uint32_t),
        streams.stream3), "cudaMemsetAsync reset stream3 ready count");
    check_cuda(cudaMemsetAsync(
        memory.streams.global_spill_count,
        0,
        2ULL * sizeof(std::uint32_t),
        streams.stream3), "cudaMemsetAsync reset global spill count");
    check_cuda(cudaMemsetAsync(
        memory.streams.global_spill_active_index,
        0,
        sizeof(std::uint32_t),
        streams.stream3), "cudaMemsetAsync reset global spill active index");
    check_cuda(cudaMemsetAsync(
        memory.streams.shard_score_hist_a,
        0,
        static_cast<std::uint64_t>(plan.config.shard_count) * SCORE_BIN_COUNT * sizeof(std::uint32_t),
        streams.stream3), "cudaMemsetAsync reset shard score hist a");
    check_cuda(cudaMemsetAsync(
        memory.streams.shard_score_hist_b,
        0,
        static_cast<std::uint64_t>(plan.config.shard_count) * SCORE_BIN_COUNT * sizeof(std::uint32_t),
        streams.stream3), "cudaMemsetAsync reset shard score hist b");
    check_cuda(cudaMemsetAsync(
        memory.streams.shard_score_hist_active_index,
        0,
        static_cast<std::uint64_t>(plan.config.shard_count) * sizeof(std::uint32_t),
        streams.stream3), "cudaMemsetAsync reset shard score hist active index");
    check_cuda(cudaMemsetAsync(memory.streams.current_threshold, 0xff, sizeof(std::uint32_t), streams.stream3),
        "cudaMemsetAsync reset current threshold");
    check_cuda(cudaMemsetAsync(memory.streams.threshold_initialized, 0, sizeof(std::uint32_t), streams.stream3),
        "cudaMemsetAsync reset threshold initialized");
    check_cuda(cudaStreamSynchronize(streams.stream3), "cudaStreamSynchronize threshold reset");

    return FinalizeDepthState{
        final_request_count,
        final_threshold,
        final_candidate_count,
        final_request_count};
}

} // namespace beam
