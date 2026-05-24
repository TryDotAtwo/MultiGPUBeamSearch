#include "dispatcher.hpp"

#include "config.hpp"
#include "final_materialize.hpp"
#include "hash.hpp"
#include "nvtx_ranges.hpp"
#include "stream3.hpp"
#include "stream4.hpp"
#include "threshold.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace beam {

namespace {

void check_cuda(cudaError_t status, const char* op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
    }
}

std::uint32_t host_shard_from_hash128(Hash128 hash, std::uint32_t shard_count) {
    return static_cast<std::uint32_t>(hash128_shard_distribution_key(hash) % shard_count);
}

__device__ CandidateMeta invalid_track_candidate_device() {
    return CandidateMeta{Hash128{UINT64_MAX, UINT64_MAX}, UINT64_MAX, UINT32_MAX, UINT32_MAX};
}

__device__ bool track_candidate_less_device(CandidateMeta a, CandidateMeta b) {
    if (a.score_key != b.score_key) {
        return a.score_key < b.score_key;
    }
    if (a.parent_idx != b.parent_idx) {
        return a.parent_idx < b.parent_idx;
    }
    return a.route_packed < b.route_packed;
}

bool track_candidate_less_host(CandidateMeta a, CandidateMeta b) {
    if (a.score_key != b.score_key) {
        return a.score_key < b.score_key;
    }
    if (a.parent_idx != b.parent_idx) {
        return a.parent_idx < b.parent_idx;
    }
    return a.route_packed < b.route_packed;
}

enum TrackLocation : std::uint32_t {
    TrackLocationNone = 0,
    TrackLocationClean = 1,
    TrackLocationDirty = 2,
    TrackLocationActiveSpill = 3,
    TrackLocationInactiveSpill = 4
};

enum TrackStream4Phase : std::uint32_t {
    TrackStream4PhaseAfterStream3 = 1,
    TrackStream4PhaseInput = 2,
    TrackStream4PhaseOutput = 3
};

__global__ void track_clean_survivor_hash_kernel(
    const CandidateMeta* survivor_shard,
    const std::uint32_t* clean_count,
    std::uint32_t* block_matches,
    std::uint32_t* block_best_score,
    std::uint64_t* block_first_index,
    std::uint64_t* block_best_index,
    CandidateMeta* block_best_candidate,
    Hash128 target_hash,
    std::uint32_t shard_count,
    std::uint32_t stream4_batch_candidates) {
    __shared__ std::uint32_t match_count[256];
    __shared__ std::uint32_t best_score[256];
    __shared__ std::uint64_t first_index[256];
    __shared__ std::uint64_t best_index[256];
    __shared__ CandidateMeta best_candidate[256];
    const std::uint32_t tid = threadIdx.x;
    const std::uint64_t shard_capacity = 2ULL * stream4_batch_candidates;
    const std::uint64_t total = static_cast<std::uint64_t>(shard_count) * shard_capacity;
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + tid;
    bool match = false;
    std::uint32_t score = UINT32_MAX;
    CandidateMeta candidate = invalid_track_candidate_device();
    if (i < total) {
        const std::uint32_t shard = static_cast<std::uint32_t>(i / shard_capacity);
        const std::uint32_t local = static_cast<std::uint32_t>(i - static_cast<std::uint64_t>(shard) * shard_capacity);
        candidate = survivor_shard[i];
        match = local < clean_count[shard] && candidate.hash == target_hash;
        score = match ? candidate.score_key : UINT32_MAX;
    }
    match_count[tid] = match ? 1U : 0U;
    best_score[tid] = score;
    first_index[tid] = match ? i : UINT64_MAX;
    best_index[tid] = match ? i : UINT64_MAX;
    best_candidate[tid] = match ? candidate : invalid_track_candidate_device();
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride > 0; stride >>= 1U) {
        if (tid < stride) {
            match_count[tid] += match_count[tid + stride];
            if (best_score[tid + stride] < best_score[tid]) {
                best_score[tid] = best_score[tid + stride];
            }
            if (first_index[tid + stride] < first_index[tid]) {
                first_index[tid] = first_index[tid + stride];
            }
            if (track_candidate_less_device(best_candidate[tid + stride], best_candidate[tid])) {
                best_candidate[tid] = best_candidate[tid + stride];
                best_index[tid] = best_index[tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_matches[blockIdx.x] = match_count[0];
        block_best_score[blockIdx.x] = best_score[0];
        block_first_index[blockIdx.x] = first_index[0];
        block_best_index[blockIdx.x] = best_index[0];
        block_best_candidate[blockIdx.x] = best_candidate[0];
    }
}

void scan_tracked_prefinal_hash(
    const StaticMemoryPlan& plan,
    StaticDeviceMemory& memory,
    DispatcherStreams& streams,
    Hash128 target_hash,
    FinalizeDepthState& state) {
    NvtxRange range("Dispatcher_track_clean_survivor_hash");
    const std::uint32_t block_size = 256;
    const std::uint64_t item_count = plan.survivor_count;
    const std::uint32_t block_count = static_cast<std::uint32_t>((item_count + block_size - 1ULL) / block_size);
    auto* block_first_index = reinterpret_cast<std::uint64_t*>(memory.final.final_request_buffer);
    auto* block_best_index = block_first_index + block_count;
    track_clean_survivor_hash_kernel<<<block_count, block_size, 0, streams.stream3>>>(
        memory.streams.survivor_shard,
        memory.streams.clean_count,
        memory.final.final_block_counts,
        memory.final.final_block_offsets,
        block_first_index,
        block_best_index,
        memory.final.final_candidate_buffer,
        target_hash,
        plan.config.shard_count,
        plan.config.stream4_batch_candidates);
    check_cuda(cudaStreamSynchronize(streams.stream3), "cudaStreamSynchronize tracked prefinal hash scan");

    std::vector<std::uint32_t> matches(block_count);
    std::vector<std::uint32_t> best_score(block_count);
    std::vector<std::uint64_t> first_index(block_count);
    std::vector<std::uint64_t> best_index(block_count);
    std::vector<CandidateMeta> best_candidate(block_count);
    check_cuda(cudaMemcpy(matches.data(), memory.final.final_block_counts, static_cast<std::uint64_t>(block_count) * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "cudaMemcpy tracked prefinal matches");
    check_cuda(cudaMemcpy(best_score.data(), memory.final.final_block_offsets, static_cast<std::uint64_t>(block_count) * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "cudaMemcpy tracked prefinal best score");
    check_cuda(cudaMemcpy(first_index.data(), block_first_index, static_cast<std::uint64_t>(block_count) * sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "cudaMemcpy tracked prefinal first index");
    check_cuda(cudaMemcpy(best_index.data(), block_best_index, static_cast<std::uint64_t>(block_count) * sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "cudaMemcpy tracked prefinal best index");
    check_cuda(cudaMemcpy(best_candidate.data(), memory.final.final_candidate_buffer, static_cast<std::uint64_t>(block_count) * sizeof(CandidateMeta), cudaMemcpyDeviceToHost), "cudaMemcpy tracked prefinal best candidate");

    state.tracked_prefinal_enabled = true;
    state.tracked_prefinal_matches = 0;
    state.tracked_prefinal_best_score_key = UINT32_MAX;
    state.tracked_prefinal_first_index = 0;
    CandidateMeta best{Hash128{UINT64_MAX, UINT64_MAX}, UINT64_MAX, UINT32_MAX, UINT32_MAX};
    for (std::uint32_t block = 0; block < block_count; ++block) {
        if (matches[block] == 0U) {
            continue;
        }
        if (state.tracked_prefinal_matches == 0U) {
            state.tracked_prefinal_first_index = first_index[block];
        } else {
            state.tracked_prefinal_first_index = std::min(state.tracked_prefinal_first_index, first_index[block]);
        }
        state.tracked_prefinal_matches += matches[block];
        state.tracked_prefinal_best_score_key = std::min(state.tracked_prefinal_best_score_key, best_score[block]);
        if (track_candidate_less_host(best_candidate[block], best)) {
            best = best_candidate[block];
            state.tracked_prefinal_best_index = best_index[block];
        }
    }
    if (state.tracked_prefinal_matches != 0U) {
        const std::uint64_t shard_capacity = 2ULL * plan.config.stream4_batch_candidates;
        state.tracked_prefinal_best_score_key = best.score_key;
        state.tracked_prefinal_best_shard = static_cast<std::uint32_t>(state.tracked_prefinal_best_index / shard_capacity);
        state.tracked_prefinal_best_local = static_cast<std::uint32_t>(state.tracked_prefinal_best_index % shard_capacity);
        state.tracked_prefinal_best_parent_idx = best.parent_idx;
        state.tracked_prefinal_best_route_packed = best.route_packed;
    }
}

void dump_final_spill_debug(
    const StaticMemoryPlan& plan,
    StaticDeviceMemory& memory,
    const std::uint32_t spill_counts[2],
    std::uint32_t spill_active,
    std::uint32_t current_threshold) {
    const std::uint32_t shard_count = plan.config.shard_count;
    const std::uint32_t shard_capacity = 2U * plan.config.stream4_batch_candidates;
    const std::uint32_t active_index = spill_active & 1U;
    const std::uint32_t active_spill_count = spill_counts[active_index];
    std::vector<std::uint32_t> clean(shard_count);
    std::vector<std::uint32_t> dirty(shard_count);
    std::vector<std::uint32_t> processing(shard_count);
    std::vector<std::uint32_t> ready(shard_count);
    std::vector<std::uint32_t> last_write(shard_count);
    std::vector<std::uint32_t> last_write_offset(shard_count);
    std::vector<std::uint32_t> last_spill(shard_count);
    std::vector<std::uint32_t> last_spill_offset(shard_count);
    check_cuda(cudaMemcpy(clean.data(), memory.streams.clean_count, static_cast<std::uint64_t>(shard_count) * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "cudaMemcpy debug clean_count");
    check_cuda(cudaMemcpy(dirty.data(), memory.streams.dirty_count, static_cast<std::uint64_t>(shard_count) * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "cudaMemcpy debug dirty_count");
    check_cuda(cudaMemcpy(processing.data(), memory.streams.processing_flag, static_cast<std::uint64_t>(shard_count) * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "cudaMemcpy debug processing_flag");
    check_cuda(cudaMemcpy(ready.data(), memory.streams.stream3_ready_flag, static_cast<std::uint64_t>(shard_count) * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "cudaMemcpy debug stream3_ready_flag");
    check_cuda(cudaMemcpy(last_write.data(), memory.streams.stream3_shard_counts, static_cast<std::uint64_t>(shard_count) * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "cudaMemcpy debug stream3_shard_counts");
    check_cuda(cudaMemcpy(last_write_offset.data(), memory.streams.stream3_shard_offsets, static_cast<std::uint64_t>(shard_count) * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "cudaMemcpy debug stream3_shard_offsets");
    check_cuda(cudaMemcpy(last_spill.data(), memory.streams.stream3_spill_counts, static_cast<std::uint64_t>(shard_count) * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "cudaMemcpy debug stream3_spill_counts");
    check_cuda(cudaMemcpy(last_spill_offset.data(), memory.streams.stream3_spill_offsets, static_cast<std::uint64_t>(shard_count) * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "cudaMemcpy debug stream3_spill_offsets");

    std::vector<std::uint64_t> spill_by_shard(shard_count, 0);
    std::vector<std::uint64_t> spill_score_sum(shard_count, 0);
    std::vector<std::uint64_t> spill_score_pass(shard_count, 0);
    std::vector<std::uint64_t> spill_score_reject(shard_count, 0);
    std::vector<std::uint32_t> spill_min_score(shard_count, std::numeric_limits<std::uint32_t>::max());
    std::vector<std::uint32_t> spill_max_score(shard_count, 0);
    if (active_spill_count != 0U) {
        std::vector<CandidateMeta> spill(active_spill_count);
        CandidateMeta* active_buffer =
            active_index == 0U ? memory.streams.global_spill_buffer_a : memory.streams.global_spill_buffer_b;
        check_cuda(cudaMemcpy(
            spill.data(),
            active_buffer,
            static_cast<std::uint64_t>(active_spill_count) * sizeof(CandidateMeta),
            cudaMemcpyDeviceToHost), "cudaMemcpy debug active spill buffer");
        for (const CandidateMeta& candidate : spill) {
            const std::uint32_t shard = host_shard_from_hash128(candidate.hash, shard_count);
            ++spill_by_shard[shard];
            spill_score_sum[shard] += candidate.score_key;
            spill_min_score[shard] = std::min(spill_min_score[shard], candidate.score_key);
            spill_max_score[shard] = std::max(spill_max_score[shard], candidate.score_key);
            if (candidate.score_key <= current_threshold) {
                ++spill_score_pass[shard];
            } else {
                ++spill_score_reject[shard];
            }
        }
    }

    std::uint64_t total_clean = 0;
    std::uint64_t total_dirty = 0;
    std::uint64_t total_spill_by_shard = 0;
    std::uint32_t full_shards = 0;
    std::uint32_t spill_shards = 0;
    std::uint32_t processing_shards = 0;
    for (std::uint32_t shard = 0; shard < shard_count; ++shard) {
        total_clean += clean[shard];
        total_dirty += dirty[shard];
        total_spill_by_shard += spill_by_shard[shard];
        full_shards += clean[shard] + dirty[shard] >= shard_capacity ? 1U : 0U;
        spill_shards += spill_by_shard[shard] != 0U ? 1U : 0U;
        processing_shards += processing[shard] != 0U ? 1U : 0U;
    }
    std::cerr
        << "final_spill_debug"
        << " active_index=" << active_index
        << " active_spill_count=" << active_spill_count
        << " inactive_spill_count=" << spill_counts[active_index ^ 1U]
        << " threshold=" << current_threshold
        << " shard_count=" << shard_count
        << " shard_capacity=" << shard_capacity
        << " total_clean=" << total_clean
        << " total_dirty=" << total_dirty
        << " total_spill_by_shard=" << total_spill_by_shard
        << " full_shards=" << full_shards
        << " spill_shards=" << spill_shards
        << " processing_shards=" << processing_shards
        << '\n';
    for (std::uint32_t shard = 0; shard < shard_count; ++shard) {
        const bool interesting =
            clean[shard] != 0U ||
            dirty[shard] != 0U ||
            processing[shard] != 0U ||
            ready[shard] != 0U ||
            last_write[shard] != 0U ||
            last_spill[shard] != 0U ||
            spill_by_shard[shard] != 0U;
        if (!interesting) {
            continue;
        }
        const std::uint32_t occupied = clean[shard] + dirty[shard];
        const std::uint32_t free_slots = occupied < shard_capacity ? shard_capacity - occupied : 0U;
        const std::uint64_t avg_score =
            spill_by_shard[shard] == 0U ? 0U : spill_score_sum[shard] / spill_by_shard[shard];
        const std::uint32_t min_score =
            spill_by_shard[shard] == 0U ? 0U : spill_min_score[shard];
        std::cerr
            << "shard_debug"
            << " shard=" << shard
            << " clean=" << clean[shard]
            << " dirty=" << dirty[shard]
            << " processing=" << processing[shard]
            << " ready=" << ready[shard]
            << " free=" << free_slots
            << " last_write=" << last_write[shard]
            << " last_write_offset=" << last_write_offset[shard]
            << " last_spill=" << last_spill[shard]
            << " last_spill_offset=" << last_spill_offset[shard]
            << " active_spill=" << spill_by_shard[shard]
            << " spill_score_pass=" << spill_score_pass[shard]
            << " spill_score_reject=" << spill_score_reject[shard]
            << " spill_min_score=" << min_score
            << " spill_max_score=" << spill_max_score[shard]
            << " spill_avg_score=" << avg_score
            << '\n';
    }
    std::cerr.flush();
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
    std::uint64_t frontier_size,
    GeneratedTrackRequest track_request) {
    NvtxRange range("Dispatcher_depth_cuda_graphs");
    const std::uint32_t ring_slot_job_count = plan.config.ring_count * plan.derived.ring_slot_count;
    const std::uint64_t candidates_per_slot = static_cast<std::uint64_t>(plan.config.b_micro) * MOVE_COUNT;
    if (graphs.ring_slot_execs.size() != ring_slot_job_count ||
        graphs.stream3_ring_execs.size() != plan.config.ring_count ||
        graphs.stream4_shard_execs.size() !=
            static_cast<std::uint64_t>(plan.config.shard_count) * plan.config.stream4_active_sort_slots) {
        throw std::invalid_argument("depth dispatcher graph template counts do not match static memory plan");
    }
    if (track_request.enabled && track_request.move >= MOVE_COUNT) {
        throw std::invalid_argument("generated track request move exceeds MOVE_COUNT");
    }

    DepthDispatchState state;
    state.frontier_size = frontier_size;
    state.tracked_generated.enabled = track_request.enabled;
    state.tracked_generated.request_parent_idx = track_request.parent_idx;
    state.tracked_generated.request_move = track_request.move;
    state.tracked_stream4.enabled = track_request.enabled;
    std::vector<std::uint32_t> host_dirty(plan.config.shard_count);
    std::vector<std::uint32_t> host_clean(plan.config.shard_count);
    std::vector<std::uint32_t> host_ready_shards(plan.config.shard_count);
    std::vector<std::uint64_t> host_parent_base(ring_slot_job_count, 0);
    std::vector<std::uint32_t> host_count(ring_slot_job_count, 0);
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
    std::function<void(std::uint32_t)> scan_tracked_stream4_output;

    const auto mark_stream4_slot_complete = [&](std::uint32_t slot) {
        if (scan_tracked_stream4_output) {
            scan_tracked_stream4_output(slot);
        }
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

    const auto read_current_threshold_host = [&]() -> std::uint32_t {
        std::uint32_t threshold = UINT32_THRESHOLD_MAX;
        check_cuda(cudaMemcpy(
            &threshold,
            memory.streams.current_threshold,
            sizeof(threshold),
            cudaMemcpyDeviceToHost), "cudaMemcpy tracked stream4 threshold");
        return threshold;
    };

    const auto scan_candidate_array_for_hash = [&](
        const CandidateMeta* device_items,
        std::uint32_t count,
        Hash128 target_hash,
        std::uint32_t& local,
        CandidateMeta& candidate) -> bool {
        if (count == 0U) {
            return false;
        }
        std::vector<CandidateMeta> items(count);
        check_cuda(cudaMemcpy(
            items.data(),
            device_items,
            static_cast<std::uint64_t>(count) * sizeof(CandidateMeta),
            cudaMemcpyDeviceToHost), "cudaMemcpy tracked candidate array");
        for (std::uint32_t i = 0; i < count; ++i) {
            if (items[i].hash == target_hash) {
                local = i;
                candidate = items[i];
                return true;
            }
        }
        return false;
    };

    const auto scan_tracked_shard = [&](
        std::uint32_t shard,
        bool include_dirty,
        std::uint32_t& clean,
        std::uint32_t& dirty,
        std::uint32_t& location,
        std::uint32_t& local,
        CandidateMeta& candidate) -> bool {
        clean = 0;
        dirty = 0;
        location = TrackLocationNone;
        local = UINT32_MAX;
        if (!state.tracked_generated.found || shard >= plan.config.shard_count) {
            return false;
        }
        check_cuda(cudaMemcpy(
            &clean,
            memory.streams.clean_count + shard,
            sizeof(clean),
            cudaMemcpyDeviceToHost), "cudaMemcpy tracked shard clean count");
        check_cuda(cudaMemcpy(
            &dirty,
            memory.streams.dirty_count + shard,
            sizeof(dirty),
            cudaMemcpyDeviceToHost), "cudaMemcpy tracked shard dirty count");
        const std::uint32_t shard_capacity = 2U * plan.config.stream4_batch_candidates;
        const std::uint32_t scan_count = std::min<std::uint32_t>(
            include_dirty ? clean + dirty : clean,
            shard_capacity);
        const CandidateMeta* shard_base =
            memory.streams.survivor_shard + static_cast<std::uint64_t>(shard) * shard_capacity;
        if (!scan_candidate_array_for_hash(shard_base, scan_count, state.tracked_generated.hash, local, candidate)) {
            return false;
        }
        location = local < clean ? TrackLocationClean : TrackLocationDirty;
        return true;
    };

    const auto scan_tracked_spill = [&](
        std::uint32_t& location,
        std::uint32_t& local,
        std::uint32_t& active_count,
        std::uint32_t& inactive_count,
        CandidateMeta& candidate) -> bool {
        location = TrackLocationNone;
        local = UINT32_MAX;
        active_count = 0;
        inactive_count = 0;
        if (!state.tracked_generated.found) {
            return false;
        }
        std::uint32_t counts[2]{};
        std::uint32_t active = 0;
        check_cuda(cudaMemcpy(
            counts,
            memory.streams.global_spill_count,
            sizeof(counts),
            cudaMemcpyDeviceToHost), "cudaMemcpy tracked spill counts");
        check_cuda(cudaMemcpy(
            &active,
            memory.streams.global_spill_active_index,
            sizeof(active),
            cudaMemcpyDeviceToHost), "cudaMemcpy tracked spill active index");
        active &= 1U;
        const std::uint32_t inactive = active ^ 1U;
        active_count = counts[active];
        inactive_count = counts[inactive];
        CandidateMeta* buffers[2] = {memory.streams.global_spill_buffer_a, memory.streams.global_spill_buffer_b};
        if (scan_candidate_array_for_hash(
                buffers[active],
                std::min(active_count, plan.config.global_spill_capacity),
                state.tracked_generated.hash,
                local,
                candidate)) {
            location = TrackLocationActiveSpill;
            return true;
        }
        if (scan_candidate_array_for_hash(
                buffers[inactive],
                std::min(inactive_count, plan.config.global_spill_capacity),
                state.tracked_generated.hash,
                local,
                candidate)) {
            location = TrackLocationInactiveSpill;
            return true;
        }
        return false;
    };

    const auto scan_tracked_after_stream3 = [&](std::uint32_t ring) {
        if (!state.tracked_generated.found ||
            state.tracked_generated.ring != ring ||
            state.tracked_stream4.after_stream3_scanned) {
            return;
        }
        state.tracked_stream4.after_stream3_scanned = true;
        state.tracked_stream4.after_stream3_threshold = read_current_threshold_host();
        CandidateMeta candidate{};
        std::uint32_t location = TrackLocationNone;
        std::uint32_t local = UINT32_MAX;
        std::uint32_t clean = 0;
        std::uint32_t dirty = 0;
        Stream4TrackEvent event{};
        event.phase = TrackStream4PhaseAfterStream3;
        event.shard = state.tracked_stream4.shard;
        event.threshold = state.tracked_stream4.after_stream3_threshold;
        event.score_key = state.tracked_stream4.score_key;
        if (scan_tracked_shard(state.tracked_stream4.shard, true, clean, dirty, location, local, candidate)) {
            state.tracked_stream4.after_stream3_found = true;
            state.tracked_stream4.after_stream3_location = location;
            state.tracked_stream4.after_stream3_local = local;
            state.tracked_stream4.after_stream3_clean_count = clean;
            state.tracked_stream4.after_stream3_dirty_count = dirty;
            event.found = true;
            event.location = location;
            event.local = local;
            event.clean_count = clean;
            event.dirty_count = dirty;
            state.tracked_stream4_events.push_back(event);
            return;
        }
        state.tracked_stream4.after_stream3_clean_count = clean;
        state.tracked_stream4.after_stream3_dirty_count = dirty;
        event.clean_count = clean;
        event.dirty_count = dirty;
        std::uint32_t active_count = 0;
        std::uint32_t inactive_count = 0;
        if (scan_tracked_spill(location, local, active_count, inactive_count, candidate)) {
            state.tracked_stream4.after_stream3_found = true;
            state.tracked_stream4.after_stream3_location = location;
            state.tracked_stream4.after_stream3_local = local;
            event.found = true;
            event.location = location;
            event.local = local;
        }
        state.tracked_stream4.after_stream3_active_spill_count = active_count;
        state.tracked_stream4.after_stream3_inactive_spill_count = inactive_count;
        event.active_spill_count = active_count;
        event.inactive_spill_count = inactive_count;
        state.tracked_stream4_events.push_back(event);
    };

    const auto scan_tracked_stream4_input = [&](std::uint32_t shard, std::uint32_t slot, std::uint64_t graph_idx) {
        if (!state.tracked_generated.found ||
            shard != state.tracked_stream4.shard) {
            return;
        }
        ++state.tracked_stream4.input_scan_count;
        const std::uint32_t threshold = read_current_threshold_host();
        CandidateMeta candidate{};
        std::uint32_t location = TrackLocationNone;
        std::uint32_t local = UINT32_MAX;
        std::uint32_t clean = 0;
        std::uint32_t dirty = 0;
        const bool found = scan_tracked_shard(shard, true, clean, dirty, location, local, candidate);
        Stream4TrackEvent event{};
        event.phase = TrackStream4PhaseInput;
        event.found = found;
        event.shard = shard;
        event.slot = slot;
        event.job = static_cast<std::uint32_t>(graph_idx);
        event.location = found ? location : TrackLocationNone;
        event.local = found ? local : UINT32_MAX;
        event.clean_count = clean;
        event.dirty_count = dirty;
        event.threshold = threshold;
        event.score_key = state.tracked_stream4.score_key;
        state.tracked_stream4_events.push_back(event);
        if (found && !state.tracked_stream4.input_found) {
            state.tracked_stream4.input_slot = slot;
            state.tracked_stream4.input_job = static_cast<std::uint32_t>(graph_idx);
            state.tracked_stream4.input_threshold = threshold;
            state.tracked_stream4.input_clean_count = clean;
            state.tracked_stream4.input_dirty_count = dirty;
            state.tracked_stream4.input_found = true;
            state.tracked_stream4.input_location = location;
            state.tracked_stream4.input_local = local;
        } else if (!state.tracked_stream4.input_found) {
            state.tracked_stream4.input_slot = slot;
            state.tracked_stream4.input_job = static_cast<std::uint32_t>(graph_idx);
            state.tracked_stream4.input_threshold = threshold;
            state.tracked_stream4.input_clean_count = clean;
            state.tracked_stream4.input_dirty_count = dirty;
        }
    };

    scan_tracked_stream4_output = [&](std::uint32_t slot) {
        const std::uint32_t shard = stream4_slot_shard[slot];
        if (!state.tracked_generated.found ||
            shard != state.tracked_stream4.shard) {
            return;
        }
        ++state.tracked_stream4.output_scan_count;
        const std::uint32_t job =
            static_cast<std::uint32_t>(static_cast<std::uint64_t>(shard) * plan.config.stream4_active_sort_slots + slot);
        const std::uint32_t threshold = read_current_threshold_host();
        CandidateMeta candidate{};
        std::uint32_t location = TrackLocationNone;
        std::uint32_t local = UINT32_MAX;
        std::uint32_t clean = 0;
        std::uint32_t dirty = 0;
        const bool found = scan_tracked_shard(shard, false, clean, dirty, location, local, candidate);
        Stream4TrackEvent event{};
        event.phase = TrackStream4PhaseOutput;
        event.found = found;
        event.shard = shard;
        event.slot = slot;
        event.job = job;
        event.location = found ? location : TrackLocationNone;
        event.local = found ? local : UINT32_MAX;
        event.clean_count = clean;
        event.dirty_count = dirty;
        event.threshold = threshold;
        event.score_key = state.tracked_stream4.score_key;
        state.tracked_stream4_events.push_back(event);
        if (found && !state.tracked_stream4.output_found) {
            state.tracked_stream4.output_slot = slot;
            state.tracked_stream4.output_job = job;
            state.tracked_stream4.output_threshold = threshold;
            state.tracked_stream4.output_clean_count = clean;
            state.tracked_stream4.output_dirty_count = dirty;
            state.tracked_stream4.output_found = true;
            state.tracked_stream4.output_local = local;
        } else if (!state.tracked_stream4.output_found) {
            state.tracked_stream4.output_slot = slot;
            state.tracked_stream4.output_job = job;
            state.tracked_stream4.output_threshold = threshold;
            state.tracked_stream4.output_clean_count = clean;
            state.tracked_stream4.output_dirty_count = dirty;
        }
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
                host_parent_base[job] = parent_base_value;
                host_count[job] = count_value;
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
            scan_tracked_stream4_input(shard, slot, graph_idx);
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
                scan_tracked_stream4_input(shard, slot, graph_idx);
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

    const auto scan_tracked_generated_candidate = [&](std::uint32_t ring) -> bool {
        if (!track_request.enabled || state.tracked_generated.found) {
            return false;
        }
        for (std::uint32_t slot = 0; slot < plan.derived.ring_slot_count; ++slot) {
            const std::uint32_t job = ring * plan.derived.ring_slot_count + slot;
            const std::uint64_t base = host_parent_base[job];
            const std::uint32_t count = host_count[job];
            if (count == 0U || track_request.parent_idx < base) {
                continue;
            }
            const std::uint64_t parent_delta = track_request.parent_idx - base;
            if (parent_delta >= count) {
                continue;
            }
            const std::uint64_t payload_id =
                static_cast<std::uint64_t>(slot) * candidates_per_slot +
                parent_delta * MOVE_COUNT +
                track_request.move;
            const std::uint64_t global_offset =
                static_cast<std::uint64_t>(job) * candidates_per_slot +
                parent_delta * MOVE_COUNT +
                track_request.move;
            std::uint32_t score_key = UINT32_MAX;
            Hash128 hash{UINT64_MAX, UINT64_MAX};
            std::uint32_t threshold = UINT32_THRESHOLD_MAX;
            check_cuda(cudaMemcpy(
                &score_key,
                memory.streams.score_ring + global_offset,
                sizeof(score_key),
                cudaMemcpyDeviceToHost), "cudaMemcpy tracked generated score");
            check_cuda(cudaMemcpy(
                &hash,
                memory.streams.hash_ring + global_offset,
                sizeof(hash),
                cudaMemcpyDeviceToHost), "cudaMemcpy tracked generated hash");
            check_cuda(cudaMemcpy(
                &threshold,
                memory.streams.current_threshold,
                sizeof(threshold),
                cudaMemcpyDeviceToHost), "cudaMemcpy tracked generated threshold");
            State128 parent_state{};
            check_cuda(cudaMemcpy(
                &parent_state,
                memory.current_frontier_states + track_request.parent_idx,
                sizeof(parent_state),
                cudaMemcpyDeviceToHost), "cudaMemcpy tracked generated parent state");
            std::array<std::uint32_t, MOVE_COUNT> move_score_keys{};
            check_cuda(cudaMemcpy(
                move_score_keys.data(),
                memory.streams.score_ring + global_offset - track_request.move,
                MOVE_COUNT * sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost), "cudaMemcpy tracked generated all move scores");
            state.tracked_generated.found = true;
            state.tracked_generated.ring = ring;
            state.tracked_generated.ring_slot = slot;
            state.tracked_generated.job = job;
            state.tracked_generated.parent_base = base;
            state.tracked_generated.count = count;
            state.tracked_generated.parent_local = static_cast<std::uint32_t>(parent_delta);
            state.tracked_generated.payload_id = payload_id;
            state.tracked_generated.score_ring_offset = global_offset;
            state.tracked_generated.score_key = score_key;
            state.tracked_generated.hash = hash;
            state.tracked_generated.owner = owner_from_hash128(hash, plan.config.world_size);
            state.tracked_generated.shard = shard_from_hash128(hash, plan.config.shard_count);
            state.tracked_generated.current_threshold = threshold;
            state.tracked_generated.parent_state_copied = true;
            state.tracked_generated.parent_state = parent_state;
            state.tracked_generated.all_move_scores_copied = true;
            state.tracked_generated.move_score_keys = move_score_keys;
            state.tracked_stream4.hash = hash;
            state.tracked_stream4.score_key = score_key;
            state.tracked_stream4.shard = state.tracked_generated.shard;
            return true;
        }
        return false;
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
        scan_tracked_generated_candidate(ring);
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
        scan_tracked_after_stream3(ring);
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
            host_clean.data(),
            memory.streams.clean_count,
            static_cast<std::uint64_t>(plan.config.shard_count) * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost), "cudaMemcpy clean_count final flush");
        check_cuda(cudaMemcpy(
            host_dirty.data(),
            memory.streams.dirty_count,
            static_cast<std::uint64_t>(plan.config.shard_count) * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost), "cudaMemcpy dirty_count final flush");
        bool any_dirty = false;
        for (std::uint32_t dirty : host_dirty) {
            any_dirty = any_dirty || dirty != 0U;
        }
        std::uint64_t total_clean = 0;
        for (std::uint32_t clean : host_clean) {
            total_clean += clean;
        }
        const bool spill_remaining = spill_counts[spill_active & 1U] != 0U || any_dirty;
        if (stream4_jobs_since_threshold_update != 0U &&
            (periodic_threshold_due() || spill_remaining)) {
            force_periodic_threshold_update();
        }
        if (!spill_remaining ||
            (!any_dirty && total_clean >= plan.derived.global_beam_width_effective)) {
            break;
        }
        if (flush_round + 1U == plan.config.shard_count + 2U) {
            std::uint32_t debug_threshold = UINT32_THRESHOLD_MAX;
            check_cuda(cudaMemcpy(
                &debug_threshold,
                memory.streams.current_threshold,
                sizeof(debug_threshold),
                cudaMemcpyDeviceToHost), "cudaMemcpy current threshold final flush failure");
            dump_final_spill_debug(plan, memory, spill_counts, spill_active, debug_threshold);
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
    cudaEvent_t history_copy_done,
    const Hash128* tracked_prefinal_hash) {
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

    FinalizeDepthState result{};
    result.final_threshold = final_threshold;
    if (tracked_prefinal_hash != nullptr) {
        scan_tracked_prefinal_hash(plan, memory, streams, *tracked_prefinal_hash, result);
    }

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

    result.next_frontier_size = final_request_count;
    result.final_threshold = final_threshold;
    result.final_candidate_count = final_candidate_count;
    result.final_request_count = final_request_count;
    return result;
}

} // namespace beam
