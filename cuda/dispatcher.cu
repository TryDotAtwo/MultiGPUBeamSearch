#include "dispatcher.hpp"

#include "config.hpp"
#include "final_materialize.hpp"
#include "hash.hpp"
#include "nvtx_ranges.hpp"
#include "stream3.hpp"
#include "stream4.hpp"
#include "stream5.hpp"
#include "threshold.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef BEAM_DEBUG_STREAM_TIMING
#define BEAM_DEBUG_STREAM_TIMING 0
#endif

#ifndef BEAM_DEBUG_PATH_TRACE
#define BEAM_DEBUG_PATH_TRACE 0
#endif

#ifndef BEAM_DEBUG_FINAL_VALIDATE
#define BEAM_DEBUG_FINAL_VALIDATE 0
#endif

#ifndef BEAM_DEBUG_FINAL_EXCHANGE_TRACE
#define BEAM_DEBUG_FINAL_EXCHANGE_TRACE 0
#endif

namespace beam {

namespace {

void check_cuda(cudaError_t status, const char* op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
    }
}

void check_nccl_dispatcher(ncclResult_t status, const char* op) {
    if (status != ncclSuccess) {
        throw std::runtime_error(std::string(op) + ": " + ncclGetErrorString(status));
    }
}

#if BEAM_DEBUG_STREAM_TIMING
void accumulate_elapsed_ms(
    cudaEvent_t start,
    cudaEvent_t stop,
    double& total_ms,
    double* max_ms,
    const char* op) {
    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), op);
    total_ms += static_cast<double>(elapsed_ms);
    if (max_ms != nullptr) {
        *max_ms = std::max(*max_ms, static_cast<double>(elapsed_ms));
    }
}
#endif

#if BEAM_DEBUG_FINAL_EXCHANGE_TRACE
void log_final_exchange_counts(
    const char* label,
    std::uint32_t local_rank,
    const std::vector<std::uint32_t>& counts,
    const std::vector<std::uint32_t>& offsets) {
    std::cout << "final_exchange_trace"
              << " rank=" << local_rank
              << " label=" << label
              << " count_size=" << counts.size()
              << " offset_size=" << offsets.size();
    for (std::uint32_t i = 0; i < counts.size(); ++i) {
        std::cout << " count" << i << "=" << counts[i];
    }
    for (std::uint32_t i = 0; i < offsets.size(); ++i) {
        std::cout << " offset" << i << "=" << offsets[i];
    }
    std::cout << "\n";
}

void log_final_exchange_request(
    const char* label,
    std::uint32_t local_rank,
    std::uint32_t index,
    const FinalRequest& request) {
    std::cout << "final_exchange_trace_request"
              << " rank=" << local_rank
              << " label=" << label
              << " index=" << index
              << " parent_idx=" << request.parent_idx
              << " target_local_idx=" << request.target_local_idx
              << " return_rank=" << request.return_rank
              << " move=" << static_cast<std::uint32_t>(request.move)
              << " pad=" << static_cast<std::uint32_t>(request.pad)
              << "\n";
}

void log_final_exchange_requests(
    const char* label,
    std::uint32_t local_rank,
    const std::vector<FinalRequest>& requests) {
    std::cout << "final_exchange_trace_requests"
              << " rank=" << local_rank
              << " label=" << label
              << " total=" << requests.size()
              << "\n";
    constexpr std::uint32_t head_limit = 8;
    constexpr std::uint32_t tail_limit = 4;
    const std::uint32_t total = static_cast<std::uint32_t>(requests.size());
    const std::uint32_t head = std::min<std::uint32_t>(head_limit, total);
    for (std::uint32_t i = 0; i < head; ++i) {
        log_final_exchange_request(label, local_rank, i, requests[i]);
    }
    if (total > head + tail_limit) {
        std::cout << "final_exchange_trace_requests_gap"
                  << " rank=" << local_rank
                  << " label=" << label
                  << " skipped=" << (total - head - tail_limit)
                  << "\n";
    }
    const std::uint32_t tail_begin = total > head + tail_limit ? total - tail_limit : head;
    for (std::uint32_t i = tail_begin; i < total; ++i) {
        log_final_exchange_request(label, local_rank, i, requests[i]);
    }
}

void log_final_exchange_candidate(
    const char* label,
    std::uint32_t local_rank,
    std::uint32_t index,
    const CandidateMeta& candidate,
    std::uint32_t target_rank,
    std::uint64_t global_idx) {
    std::cout << "final_exchange_trace_candidate"
              << " rank=" << local_rank
              << " label=" << label
              << " index=" << index
              << " hash_lo=" << candidate.hash.lo
              << " hash_hi=" << candidate.hash.hi
              << " parent_idx=" << candidate.parent_idx
              << " score_key=" << candidate.score_key
              << " route_packed=" << candidate.route_packed
              << " source_rank=" << unpack_source_rank(candidate.route_packed)
              << " owner=" << static_cast<std::uint32_t>(unpack_owner(candidate.route_packed))
              << " move=" << static_cast<std::uint32_t>(unpack_move(candidate.route_packed))
              << " target_rank=" << target_rank
              << " global_idx=" << global_idx
              << "\n";
}
#endif

std::uint32_t host_shard_from_hash128(Hash128 hash, std::uint32_t shard_count) {
    return static_cast<std::uint32_t>(hash128_shard_distribution_key(hash) % shard_count);
}

std::string stream_fatal_error_message(
    const char* phase,
    std::uint32_t flag,
    const std::array<std::uint64_t, STREAM_FATAL_TRACE_WORDS>& trace) {
    return std::string("cuda stream fatal error: phase=") + phase +
        " flag=" + std::to_string(flag) +
        " code=" + std::to_string(trace[FatalTraceCode]) +
        " shard=" + std::to_string(trace[FatalTraceShard]) +
        " group=" + std::to_string(trace[FatalTraceGroup]) +
        " existing=" + std::to_string(trace[FatalTraceExisting]) +
        " available=" + std::to_string(trace[FatalTraceAvailable]) +
        " raw_count=" + std::to_string(trace[FatalTraceRawCount]) +
        " write_count=" + std::to_string(trace[FatalTraceWriteCount]) +
        " spill_count=" + std::to_string(trace[FatalTraceSpillCount]) +
        " spill_offset=" + std::to_string(trace[FatalTraceSpillOffset]) +
        " spill_end=" + std::to_string(trace[FatalTraceSpillEnd]) +
        " spill_capacity=" + std::to_string(trace[FatalTraceSpillCapacity]) +
        " clean_count=" + std::to_string(trace[FatalTraceCleanCount]) +
        " dirty_count=" + std::to_string(trace[FatalTraceDirtyCount]) +
        " processing_flag=" + std::to_string(trace[FatalTraceProcessingFlag]) +
        " shard_capacity_candidates=" + std::to_string(trace[FatalTraceShardCapacity]) +
        " stream4_batch_candidates=" + std::to_string(trace[FatalTraceStream4Batch]) +
        " append_to_active_spill=" + std::to_string(trace[FatalTraceAppendToActiveSpill]);
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
    std::uint32_t shard_capacity_candidates,
    std::uint32_t stream4_batch_candidates) {
    __shared__ std::uint32_t match_count[256];
    __shared__ std::uint32_t best_score[256];
    __shared__ std::uint64_t first_index[256];
    __shared__ std::uint64_t best_index[256];
    __shared__ CandidateMeta best_candidate[256];
    const std::uint32_t tid = threadIdx.x;
    const std::uint64_t shard_capacity = shard_capacity_candidates;
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
        plan.storage_shard_count,
        plan.config.shard_capacity_candidates,
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
        const std::uint64_t shard_capacity = plan.config.shard_capacity_candidates;
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
    const std::uint32_t shard_capacity = plan.config.shard_capacity_candidates;
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

void update_threshold_global(
    const StaticMemoryPlan& plan,
    StaticDeviceMemory& memory,
    cudaStream_t stream,
    bool periodic,
    const DispatcherCollective* collective) {
    const std::uint64_t threshold_width = plan.derived.global_beam_width_effective;
    threshold_build_local_histogram_cuda(
        memory.streams.shard_score_hist_a,
        memory.streams.shard_score_hist_b,
        memory.streams.shard_score_hist_active_index,
        memory.streams.threshold_hist_active_snapshot,
        memory.streams.local_score_hist,
        plan.storage_shard_count,
        stream);
    if (plan.config.world_size == 1U) {
        check_cuda(cudaMemcpyAsync(
            memory.streams.global_score_hist,
            memory.streams.local_score_hist,
            SCORE_BIN_COUNT * sizeof(std::uint64_t),
            cudaMemcpyDeviceToDevice,
            stream), "cudaMemcpyAsync single gpu histogram");
    } else {
        if (collective == nullptr || collective->comm == nullptr) {
            throw std::invalid_argument("multi rank threshold update requires NCCL collective");
        }
        threshold_allreduce_histogram_nccl_cuda(
            memory.streams.local_score_hist,
            memory.streams.global_score_hist,
            collective->comm,
            stream);
    }
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

struct alignas(32) FinalHistoryRecord {
    CandidateMeta meta;
    std::uint32_t target_local_idx;
    std::uint32_t reserved0;
    std::uint64_t reserved1[3];
};

static_assert(sizeof(FinalHistoryRecord) == 64);
static_assert(alignof(FinalHistoryRecord) == 32);
static_assert(sizeof(FinalHistoryRecord) % sizeof(std::uint64_t) == 0);

struct alignas(64) FinalExchangePlan {
    std::vector<std::uint32_t> count;
    std::vector<std::uint32_t> offset;
    std::uint32_t total = 0;
};

static_assert(alignof(FinalExchangePlan) == 64);

FinalExchangePlan make_final_exchange_plan(const std::vector<std::uint32_t>& counts) {
    FinalExchangePlan plan{};
    plan.count = counts;
    plan.offset.resize(static_cast<std::uint64_t>(counts.size()) + 1ULL);
    std::uint64_t running = 0;
    plan.offset[0] = 0;
    for (std::uint32_t i = 0; i < counts.size(); ++i) {
        running += counts[i];
        if (running > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::runtime_error("exchange item count exceeds uint32 range");
        }
        plan.offset[i + 1U] = static_cast<std::uint32_t>(running);
    }
    plan.total = static_cast<std::uint32_t>(running);
    return plan;
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
        const std::uint64_t stream5_send_slot =
            plan.config.world_size > 1U ? static_cast<std::uint64_t>(ring) : 0ULL;
        CandidateMeta* ring_remote_send_buffer =
            memory.streams.remote_send_buffer + stream5_send_slot * plan.stream5_send_slot_capacity;
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
                ring_remote_send_buffer,
                memory.streams.send_count,
                memory.streams.send_offset,
                memory.streams.stream3_owner,
                static_cast<std::uint16_t>(plan.config.local_rank),
                plan.config.world_size,
                plan.config.b_micro,
                plan.config.stream3_batch_candidates,
                streams.stream3);
        }
        if (plan.config.shard_buffer_count == 1U) {
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
                plan.config.shard_capacity_candidates,
                plan.config.stream4_batch_candidates,
                streams.stream3,
                memory.streams.fatal_error_flag,
                memory.streams.fatal_error_trace);
        }
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
                memory.streams.stream3_write_buffer_index,
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
                plan.config.shard_buffer_count,
                plan.config.shard_capacity_candidates,
                plan.config.stream4_batch_candidates,
                plan.config.global_spill_capacity,
                streams.stream3,
                memory.streams.fatal_error_flag,
                memory.streams.fatal_error_trace);
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
                memory.streams.stream3_write_buffer_index,
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
                plan.config.shard_buffer_count,
                plan.config.shard_capacity_candidates,
                plan.config.stream4_batch_candidates,
                plan.config.global_spill_capacity,
                streams.stream3,
                memory.streams.fatal_error_flag,
                memory.streams.fatal_error_trace);
        }
        stream3_build_ready_shard_queue_cuda(
            memory.streams.clean_count,
            memory.streams.dirty_count,
            memory.streams.processing_flag,
            memory.streams.stream3_write_buffer_index,
            memory.streams.stream3_ready_flag,
            memory.streams.stream3_ready_shard_list,
            memory.streams.stream3_ready_count,
            plan.config.shard_count,
            plan.config.shard_buffer_count,
            plan.config.shard_capacity_candidates,
            plan.config.stream3_batch_candidates,
            plan.config.stream4_trigger_candidates,
            false,
            false,
            streams.stream3);
        instantiate_captured_graph(streams.stream3, graphs.stream3_ring_graphs[ring], graphs.stream3_ring_execs[ring]);
    }

    const std::uint32_t stream4_slot_count = plan.config.stream4_active_sort_slots;
    graphs.stream4_shard_graphs.resize(
        static_cast<std::uint64_t>(plan.storage_shard_count) * stream4_slot_count,
        nullptr);
    graphs.stream4_shard_execs.resize(
        static_cast<std::uint64_t>(plan.storage_shard_count) * stream4_slot_count,
        nullptr);
    const std::uint32_t stream4_capacity = plan.config.shard_capacity_candidates;
    const std::uint32_t stream4_block_count = (stream4_capacity + 255U) / 256U;
    for (std::uint32_t shard = 0; shard < plan.storage_shard_count; ++shard) {
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
    GeneratedTrackRequest track_request,
    const DispatcherCollective* collective) {
    NvtxRange range("Dispatcher_depth_cuda_graphs");
#if !BEAM_DEBUG_PATH_TRACE
    track_request = {};
#endif
    const std::uint32_t ring_slot_job_count = plan.config.ring_count * plan.derived.ring_slot_count;
    const std::uint64_t candidates_per_slot = static_cast<std::uint64_t>(plan.config.b_micro) * MOVE_COUNT;
    if (graphs.ring_slot_execs.size() != ring_slot_job_count ||
        graphs.stream3_ring_execs.size() != plan.config.ring_count ||
        graphs.stream4_shard_execs.size() !=
            static_cast<std::uint64_t>(plan.storage_shard_count) * plan.config.stream4_active_sort_slots) {
        throw std::invalid_argument("depth dispatcher graph template counts do not match static memory plan");
    }
    if (track_request.enabled && track_request.move >= MOVE_COUNT) {
        throw std::invalid_argument("generated track request move exceeds MOVE_COUNT");
    }
    const bool multi_rank = plan.config.world_size > 1U;
    if (multi_rank && (collective == nullptr || collective->comm == nullptr)) {
        throw std::invalid_argument("multi rank depth dispatch requires NCCL collective");
    }
    const std::uint64_t parents_per_stream3_round =
        static_cast<std::uint64_t>(plan.derived.ring_slot_count) * plan.config.b_micro;
    const std::uint64_t local_exchange_rounds =
        frontier_size == 0U
            ? 0U
            : (frontier_size + parents_per_stream3_round - 1ULL) / parents_per_stream3_round;
    std::uint64_t global_exchange_rounds = local_exchange_rounds;
    if (multi_rank) {
        check_cuda(cudaMemcpyAsync(
            memory.streams.stream5_local_round_count,
            &local_exchange_rounds,
            sizeof(local_exchange_rounds),
            cudaMemcpyHostToDevice,
            streams.stream5), "cudaMemcpyAsync local stream5 round count");
        check_nccl_dispatcher(
            ncclAllReduce(
                memory.streams.stream5_local_round_count,
                memory.streams.stream5_global_round_count,
                1,
                ncclUint64,
                ncclMax,
                collective->comm,
                streams.stream5),
            "ncclAllReduce stream5 round count");
        check_cuda(cudaMemcpyAsync(
            &global_exchange_rounds,
            memory.streams.stream5_global_round_count,
            sizeof(global_exchange_rounds),
            cudaMemcpyDeviceToHost,
            streams.stream5), "cudaMemcpyAsync global stream5 round count");
        check_cuda(cudaStreamSynchronize(streams.stream5), "cudaStreamSynchronize stream5 round count");
    }

    DepthDispatchState state;
    state.frontier_size = frontier_size;
    state.tracked_generated.enabled = track_request.enabled;
    state.tracked_generated.request_parent_idx = track_request.parent_idx;
    state.tracked_generated.request_move = track_request.move;
    state.tracked_stream3.enabled = track_request.enabled;
    state.tracked_stream4.enabled = track_request.enabled;
    std::vector<std::uint32_t> host_dirty(plan.storage_shard_count);
    std::vector<std::uint32_t> host_clean(plan.storage_shard_count);
    std::vector<std::uint32_t> host_processing(plan.storage_shard_count);
    std::vector<std::uint32_t> host_ready_shards(plan.storage_shard_count);
    std::vector<std::uint32_t> host_send_count(plan.config.world_size);
    std::vector<std::uint32_t> host_send_offset(static_cast<std::uint64_t>(plan.config.world_size) + 1ULL);
    std::vector<std::uint32_t> host_recv_count(plan.config.world_size);
    std::vector<std::uint32_t> host_recv_offset(static_cast<std::uint64_t>(plan.config.world_size) + 1ULL);
    std::vector<std::uint64_t> host_parent_base(ring_slot_job_count, 0);
    std::vector<std::uint32_t> host_count(ring_slot_job_count, 0);
    std::vector<std::uint32_t> pending_stream4_shards;
    pending_stream4_shards.reserve(plan.storage_shard_count * plan.config.ring_count);
    std::uint32_t pending_stream4_head = 0;
    std::vector<bool> stream4_slot_busy(plan.config.stream4_active_sort_slots, false);
    std::vector<std::uint32_t> stream4_slot_shard(plan.config.stream4_active_sort_slots, plan.storage_shard_count);
    std::deque<std::uint32_t> stream4_free_slots;
    std::deque<std::uint32_t> stream4_busy_slots;
    for (std::uint32_t slot = 0; slot < plan.config.stream4_active_sort_slots; ++slot) {
        stream4_free_slots.push_back(slot);
    }
    std::uint32_t stream4_jobs_since_threshold_update = 0;
    const bool pipeline_stats_enabled = std::getenv("BEAM_DEBUG_PIPELINE_STATS") != nullptr;

    std::vector<cudaEvent_t> ring_done(plan.config.ring_count, nullptr);
    std::vector<cudaEvent_t> stream3_done(plan.config.ring_count, nullptr);
#if BEAM_DEBUG_STREAM_TIMING
    std::vector<cudaEvent_t> ring_timing_start(plan.config.ring_count, nullptr);
    std::vector<cudaEvent_t> ring_timing_stop(plan.config.ring_count, nullptr);
    std::vector<cudaEvent_t> stream3_timing_start(plan.config.ring_count, nullptr);
    std::vector<cudaEvent_t> stream3_timing_stop(plan.config.ring_count, nullptr);
    std::vector<cudaEvent_t> stream4_timing_start(plan.config.stream4_active_sort_slots, nullptr);
    std::vector<cudaEvent_t> stream4_timing_stop(plan.config.stream4_active_sort_slots, nullptr);
    std::vector<cudaEvent_t> stream5_timing_start(1, nullptr);
    std::vector<cudaEvent_t> stream5_timing_stop(1, nullptr);
    std::vector<cudaEvent_t> stream3_spill_drain_timing_start(1, nullptr);
    std::vector<cudaEvent_t> stream3_spill_drain_timing_stop(1, nullptr);
#endif
    for (std::uint32_t ring = 0; ring < plan.config.ring_count; ++ring) {
        check_cuda(cudaEventCreateWithFlags(&ring_done[ring], cudaEventDisableTiming), "cudaEventCreate ring done");
        check_cuda(cudaEventCreateWithFlags(&stream3_done[ring], cudaEventDisableTiming), "cudaEventCreate stream3 done");
#if BEAM_DEBUG_STREAM_TIMING
        check_cuda(cudaEventCreate(&ring_timing_start[ring]), "cudaEventCreate ring timing start");
        check_cuda(cudaEventCreate(&ring_timing_stop[ring]), "cudaEventCreate ring timing stop");
        check_cuda(cudaEventCreate(&stream3_timing_start[ring]), "cudaEventCreate stream3 timing start");
        check_cuda(cudaEventCreate(&stream3_timing_stop[ring]), "cudaEventCreate stream3 timing stop");
#endif
    }
#if BEAM_DEBUG_STREAM_TIMING
    for (std::uint32_t slot = 0; slot < plan.config.stream4_active_sort_slots; ++slot) {
        check_cuda(cudaEventCreate(&stream4_timing_start[slot]), "cudaEventCreate stream4 timing start");
        check_cuda(cudaEventCreate(&stream4_timing_stop[slot]), "cudaEventCreate stream4 timing stop");
    }
    check_cuda(cudaEventCreate(&stream5_timing_start[0]), "cudaEventCreate stream5 timing start");
    check_cuda(cudaEventCreate(&stream5_timing_stop[0]), "cudaEventCreate stream5 timing stop");
    check_cuda(cudaEventCreate(&stream3_spill_drain_timing_start[0]), "cudaEventCreate stream3 spill drain timing start");
    check_cuda(cudaEventCreate(&stream3_spill_drain_timing_stop[0]), "cudaEventCreate stream3 spill drain timing stop");
#endif
    struct EventCleanup {
        std::vector<std::vector<cudaEvent_t>*>& groups;
        ~EventCleanup() {
            for (std::vector<cudaEvent_t>* group : groups) {
                for (cudaEvent_t event : *group) {
                    if (event) {
                        cudaEventDestroy(event);
                    }
                }
            }
        }
    };
    std::vector<std::vector<cudaEvent_t>*> event_groups{
        &ring_done,
        &stream3_done
#if BEAM_DEBUG_STREAM_TIMING
        ,
        &ring_timing_start,
        &ring_timing_stop,
        &stream3_timing_start,
        &stream3_timing_stop,
        &stream4_timing_start,
        &stream4_timing_stop,
        &stream5_timing_start,
        &stream5_timing_stop,
        &stream3_spill_drain_timing_start,
        &stream3_spill_drain_timing_stop
#endif
    };
    EventCleanup event_cleanup{event_groups};

    const auto refresh_stop_requested = [&]() -> bool {
        if (multi_rank) {
            return false;
        }
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

    const auto throw_if_stream_fatal_error = [&](const char* phase) {
        std::uint32_t flag = 0;
        check_cuda(cudaMemcpy(
            &flag,
            memory.streams.fatal_error_flag,
            sizeof(flag),
            cudaMemcpyDeviceToHost), "cudaMemcpy stream fatal flag");
        if (flag == 0U) {
            return;
        }
        std::array<std::uint64_t, STREAM_FATAL_TRACE_WORDS> trace{};
        check_cuda(cudaMemcpy(
            trace.data(),
            memory.streams.fatal_error_trace,
            trace.size() * sizeof(std::uint64_t),
            cudaMemcpyDeviceToHost), "cudaMemcpy stream fatal trace");
        throw std::runtime_error(stream_fatal_error_message(phase, flag, trace));
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

    const auto pending_stream4_count = [&]() -> std::uint32_t {
        return static_cast<std::uint32_t>(pending_stream4_shards.size() - pending_stream4_head);
    };

    const auto debug_pipeline_stats = [&](
        const char* phase,
        std::uint32_t a = UINT32_MAX,
        std::uint32_t b = UINT32_MAX) {
        if (!pipeline_stats_enabled) {
            return;
        }
        std::vector<std::uint32_t> clean(plan.storage_shard_count);
        std::vector<std::uint32_t> dirty(plan.storage_shard_count);
        std::vector<std::uint32_t> processing(plan.storage_shard_count);
        std::vector<std::uint32_t> last_spill(plan.config.shard_count);
        std::uint32_t spill_counts[2]{};
        std::uint32_t spill_active = 0;
        check_cuda(cudaMemcpy(
            clean.data(),
            memory.streams.clean_count,
            static_cast<std::uint64_t>(plan.storage_shard_count) * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost), "cudaMemcpy pipeline clean");
        check_cuda(cudaMemcpy(
            dirty.data(),
            memory.streams.dirty_count,
            static_cast<std::uint64_t>(plan.storage_shard_count) * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost), "cudaMemcpy pipeline dirty");
        check_cuda(cudaMemcpy(
            processing.data(),
            memory.streams.processing_flag,
            static_cast<std::uint64_t>(plan.storage_shard_count) * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost), "cudaMemcpy pipeline processing");
        check_cuda(cudaMemcpy(
            last_spill.data(),
            memory.streams.stream3_spill_counts,
            static_cast<std::uint64_t>(plan.config.shard_count) * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost), "cudaMemcpy pipeline last spill");
        check_cuda(cudaMemcpy(
            spill_counts,
            memory.streams.global_spill_count,
            sizeof(spill_counts),
            cudaMemcpyDeviceToHost), "cudaMemcpy pipeline spill counts");
        check_cuda(cudaMemcpy(
            &spill_active,
            memory.streams.global_spill_active_index,
            sizeof(spill_active),
            cudaMemcpyDeviceToHost), "cudaMemcpy pipeline spill active");
        spill_active &= 1U;
        std::uint64_t clean_total = 0;
        std::uint64_t dirty_total = 0;
        std::uint64_t last_spill_total = 0;
        std::uint32_t busy_shards = 0;
        std::uint32_t dirty_shards = 0;
        for (std::uint32_t shard = 0; shard < plan.storage_shard_count; ++shard) {
            clean_total += clean[shard];
            dirty_total += dirty[shard];
            busy_shards += processing[shard] != 0U ? 1U : 0U;
            dirty_shards += dirty[shard] != 0U ? 1U : 0U;
        }
        for (std::uint32_t shard = 0; shard < plan.config.shard_count; ++shard) {
            last_spill_total += last_spill[shard];
        }
        std::cout << "pipeline_stats"
                  << " phase=" << phase
                  << " a=" << a
                  << " b=" << b
                  << " stream3_jobs=" << state.stream3_jobs_launched
                  << " stream4_jobs=" << state.stream4_jobs_launched
                  << " pending=" << pending_stream4_count()
                  << " busy_slots=" << stream4_busy_slots.size()
                  << " free_slots=" << stream4_free_slots.size()
                  << " busy_shards=" << busy_shards
                  << " dirty_shards=" << dirty_shards
                  << " clean_total=" << clean_total
                  << " dirty_total=" << dirty_total
                  << " last_spill_total=" << last_spill_total
                  << " spill_active_idx=" << spill_active
                  << " active_spill=" << spill_counts[spill_active]
                  << " inactive_spill=" << spill_counts[spill_active ^ 1U]
                  << "\n";
        if (plan.storage_shard_count <= 32U) {
            auto print_array = [&](const char* name, const std::vector<std::uint32_t>& values) {
                std::cout << " " << name << "=";
                for (std::uint32_t shard = 0; shard < values.size(); ++shard) {
                    if (shard != 0U) {
                        std::cout << ",";
                    }
                    std::cout << values[shard];
                }
            };
            std::cout << "pipeline_shards"
                      << " phase=" << phase
                      << " a=" << a
                      << " b=" << b;
            print_array("clean", clean);
            print_array("dirty", dirty);
            print_array("processing", processing);
            print_array("last_spill", last_spill);
            std::cout << "\n";
        }
    };

    const auto update_stream4_queue_peaks = [&]() {
        state.stream4_pending_shards_max =
            std::max(state.stream4_pending_shards_max, pending_stream4_count());
        state.stream4_busy_slots_max =
            std::max<std::uint32_t>(
                state.stream4_busy_slots_max,
                static_cast<std::uint32_t>(stream4_busy_slots.size()));
    };

    const auto stream3_has_writable_buffer = [&]() -> bool {
        if (plan.config.shard_buffer_count <= 1U) {
            return true;
        }
        const std::uint32_t average_shard_write =
            plan.config.shard_count == 0U ? plan.config.stream3_batch_candidates :
            (plan.config.stream3_batch_candidates + plan.config.shard_count - 1U) / plan.config.shard_count;
        const std::uint32_t write_margin = (average_shard_write + 3U) / 4U;
        const std::uint32_t unclamped_write_reserve =
            average_shard_write > UINT32_MAX - write_margin ? UINT32_MAX : average_shard_write + write_margin;
        const std::uint32_t write_reserve =
            unclamped_write_reserve < plan.config.shard_capacity_candidates ?
            unclamped_write_reserve :
            plan.config.shard_capacity_candidates;
        check_cuda(cudaMemcpy(
            host_clean.data(),
            memory.streams.clean_count,
            static_cast<std::uint64_t>(plan.storage_shard_count) * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost), "cudaMemcpy stream3 writable clean");
        check_cuda(cudaMemcpy(
            host_dirty.data(),
            memory.streams.dirty_count,
            static_cast<std::uint64_t>(plan.storage_shard_count) * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost), "cudaMemcpy stream3 writable dirty");
        check_cuda(cudaMemcpy(
            host_processing.data(),
            memory.streams.processing_flag,
            static_cast<std::uint64_t>(plan.storage_shard_count) * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost), "cudaMemcpy stream3 writable processing");
        for (std::uint32_t logical_shard = 0; logical_shard < plan.config.shard_count; ++logical_shard) {
            bool writable = false;
            for (std::uint32_t buffer = 0; buffer < plan.config.shard_buffer_count; ++buffer) {
                const std::uint32_t physical_shard =
                    logical_shard * plan.config.shard_buffer_count + buffer;
                const std::uint32_t occupied =
                    host_clean[physical_shard] + host_dirty[physical_shard];
                if (host_processing[physical_shard] == 0U &&
                    occupied < plan.config.shard_capacity_candidates &&
                    plan.config.shard_capacity_candidates - occupied >= write_reserve) {
                    writable = true;
                    break;
                }
            }
            if (!writable) {
                debug_pipeline_stats("stream3_backpressure", logical_shard, UINT32_MAX);
                return false;
            }
        }
        return true;
    };

#if BEAM_DEBUG_STREAM_TIMING
    const auto update_global_spill_peak = [&]() {
        std::uint32_t spill_counts[2]{};
        check_cuda(cudaMemcpy(
            spill_counts,
            memory.streams.global_spill_count,
            sizeof(spill_counts),
            cudaMemcpyDeviceToHost), "cudaMemcpy global spill counts timing");
        state.global_spill_peak = std::max(state.global_spill_peak, spill_counts[0]);
        state.global_spill_peak = std::max(state.global_spill_peak, spill_counts[1]);
    };
#else
    const auto update_global_spill_peak = [&]() {};
#endif

    const auto mark_stream4_slot_complete = [&](std::uint32_t slot) {
#if BEAM_DEBUG_STREAM_TIMING
        accumulate_elapsed_ms(
            stream4_timing_start[slot],
            stream4_timing_stop[slot],
            state.stream4_ms_total,
            &state.stream4_ms_max,
            "cudaEventElapsedTime stream4 job");
#endif
        if (scan_tracked_stream4_output) {
            scan_tracked_stream4_output(slot);
        }
        debug_pipeline_stats("stream4_complete", stream4_slot_shard[slot], slot);
        stream4_slot_busy[slot] = false;
        stream4_slot_shard[slot] = plan.storage_shard_count;
        stream4_free_slots.push_back(slot);
        ++stream4_jobs_since_threshold_update;
        update_stream4_queue_peaks();
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
        const std::uint32_t shard_capacity = plan.config.shard_capacity_candidates;
        for (std::uint32_t buffer = 0; buffer < plan.config.shard_buffer_count; ++buffer) {
            const std::uint32_t physical_shard = shard * plan.config.shard_buffer_count + buffer;
            std::uint32_t physical_clean = 0;
            std::uint32_t physical_dirty = 0;
            check_cuda(cudaMemcpy(
                &physical_clean,
                memory.streams.clean_count + physical_shard,
                sizeof(physical_clean),
                cudaMemcpyDeviceToHost), "cudaMemcpy tracked shard clean count");
            check_cuda(cudaMemcpy(
                &physical_dirty,
                memory.streams.dirty_count + physical_shard,
                sizeof(physical_dirty),
                cudaMemcpyDeviceToHost), "cudaMemcpy tracked shard dirty count");
            clean += physical_clean;
            dirty += physical_dirty;
            const std::uint32_t scan_count = std::min<std::uint32_t>(
                include_dirty ? physical_clean + physical_dirty : physical_clean,
                shard_capacity);
            const CandidateMeta* shard_base =
                memory.streams.survivor_shard + static_cast<std::uint64_t>(physical_shard) * shard_capacity;
            if (scan_candidate_array_for_hash(shard_base, scan_count, state.tracked_generated.hash, local, candidate)) {
                location = local < physical_clean ? TrackLocationClean : TrackLocationDirty;
                local += buffer * shard_capacity;
                return true;
            }
        }
        return false;
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

    const auto scan_tracked_stream3_path = [&](std::uint32_t ring) {
        if (!state.tracked_generated.found ||
            state.tracked_generated.ring != ring ||
            state.tracked_stream3.scanned) {
            return;
        }
        state.tracked_stream3.scanned = true;
        std::uint32_t unique_count = 0;
        check_cuda(cudaMemcpy(
            &unique_count,
            memory.streams.unique_count,
            sizeof(unique_count),
            cudaMemcpyDeviceToHost), "cudaMemcpy tracked stream3 unique count");
        unique_count = std::min(unique_count, plan.config.stream3_batch_candidates);
        state.tracked_stream3.unique_count = unique_count;
        if (unique_count != 0U) {
            std::vector<Hash128> unique_key(unique_count);
            std::vector<std::uint64_t> unique_val(unique_count);
            check_cuda(cudaMemcpy(
                unique_key.data(),
                memory.streams.unique_key,
                static_cast<std::uint64_t>(unique_count) * sizeof(Hash128),
                cudaMemcpyDeviceToHost), "cudaMemcpy tracked stream3 unique keys");
            check_cuda(cudaMemcpy(
                unique_val.data(),
                memory.streams.unique_val,
                static_cast<std::uint64_t>(unique_count) * sizeof(std::uint64_t),
                cudaMemcpyDeviceToHost), "cudaMemcpy tracked stream3 unique values");
            for (std::uint32_t i = 0; i < unique_count; ++i) {
                if (unique_key[i] == state.tracked_generated.hash) {
                    const std::uint64_t value = unique_val[i];
                    const std::uint32_t payload_id = static_cast<std::uint32_t>(value & 0xffffffffULL);
                    const std::uint32_t ring_slot = payload_id / static_cast<std::uint32_t>(candidates_per_slot);
                    const std::uint32_t local_i = payload_id % static_cast<std::uint32_t>(candidates_per_slot);
                    const std::uint32_t parent_local = local_i / static_cast<std::uint32_t>(MOVE_COUNT);
                    const std::uint32_t job = ring * plan.derived.ring_slot_count + ring_slot;
                    state.tracked_stream3.unique_found = true;
                    state.tracked_stream3.unique_local = i;
                    state.tracked_stream3.unique_score_key = static_cast<std::uint32_t>(value >> 32);
                    state.tracked_stream3.unique_payload_id = payload_id;
                    state.tracked_stream3.unique_parent_idx = host_parent_base[job] + parent_local;
                    state.tracked_stream3.unique_move =
                        static_cast<std::uint8_t>(local_i % static_cast<std::uint32_t>(MOVE_COUNT));
                    break;
                }
            }
        }

        std::uint32_t local_pending_count = 0;
        check_cuda(cudaMemcpy(
            &local_pending_count,
            memory.streams.local_pending_count,
            sizeof(local_pending_count),
            cudaMemcpyDeviceToHost), "cudaMemcpy tracked stream3 local pending count");
        local_pending_count = std::min(local_pending_count, plan.config.stream3_batch_candidates);
        state.tracked_stream3.local_pending_count = local_pending_count;
        if (local_pending_count != 0U) {
            std::vector<CandidateMeta> partition(local_pending_count);
            check_cuda(cudaMemcpy(
                partition.data(),
                memory.streams.stream3_partition_val_b,
                static_cast<std::uint64_t>(local_pending_count) * sizeof(CandidateMeta),
                cudaMemcpyDeviceToHost), "cudaMemcpy tracked stream3 partition values");
            for (std::uint32_t i = 0; i < local_pending_count; ++i) {
                if (partition[i].hash == state.tracked_generated.hash) {
                    state.tracked_stream3.partition_found = true;
                    state.tracked_stream3.partition_local = i;
                    break;
                }
            }
        }

        std::uint32_t partition_unique_count = 0;
        check_cuda(cudaMemcpy(
            &partition_unique_count,
            memory.streams.stream3_partition_unique_count,
            sizeof(partition_unique_count),
            cudaMemcpyDeviceToHost), "cudaMemcpy tracked stream3 partition unique count");
        partition_unique_count = std::min(partition_unique_count, plan.config.shard_count + 1U);
        state.tracked_stream3.partition_unique_count = partition_unique_count;
        if (partition_unique_count != 0U) {
            std::vector<std::uint32_t> unique_shard(partition_unique_count);
            std::vector<std::uint32_t> unique_counts(partition_unique_count);
            check_cuda(cudaMemcpy(
                unique_shard.data(),
                memory.streams.stream3_partition_unique_shard,
                static_cast<std::uint64_t>(partition_unique_count) * sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost), "cudaMemcpy tracked stream3 partition unique shards");
            check_cuda(cudaMemcpy(
                unique_counts.data(),
                memory.streams.stream3_partition_unique_counts,
                static_cast<std::uint64_t>(partition_unique_count) * sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost), "cudaMemcpy tracked stream3 partition unique counts");
            for (std::uint32_t group = 0; group < partition_unique_count; ++group) {
                if (unique_shard[group] == state.tracked_generated.shard) {
                    state.tracked_stream3.group_raw_count = unique_counts[group];
                    break;
                }
            }
        }

        const std::uint32_t shard = state.tracked_generated.shard;
        if (shard < plan.config.shard_count) {
            check_cuda(cudaMemcpy(
                &state.tracked_stream3.group_offset,
                memory.streams.stream3_shard_offsets + shard,
                sizeof(state.tracked_stream3.group_offset),
                cudaMemcpyDeviceToHost), "cudaMemcpy tracked stream3 shard offset");
            check_cuda(cudaMemcpy(
                &state.tracked_stream3.shard_write_count,
                memory.streams.stream3_shard_counts + shard,
                sizeof(state.tracked_stream3.shard_write_count),
                cudaMemcpyDeviceToHost), "cudaMemcpy tracked stream3 shard write count");
            check_cuda(cudaMemcpy(
                &state.tracked_stream3.shard_spill_count,
                memory.streams.stream3_spill_counts + shard,
                sizeof(state.tracked_stream3.shard_spill_count),
                cudaMemcpyDeviceToHost), "cudaMemcpy tracked stream3 shard spill count");
            check_cuda(cudaMemcpy(
                &state.tracked_stream3.shard_spill_offset,
                memory.streams.stream3_spill_offsets + shard,
                sizeof(state.tracked_stream3.shard_spill_offset),
                cudaMemcpyDeviceToHost), "cudaMemcpy tracked stream3 shard spill offset");
            check_cuda(cudaMemcpy(
                &state.tracked_stream3.clean_count,
                memory.streams.clean_count + shard,
                sizeof(state.tracked_stream3.clean_count),
                cudaMemcpyDeviceToHost), "cudaMemcpy tracked stream3 clean count");
            check_cuda(cudaMemcpy(
                &state.tracked_stream3.dirty_count,
                memory.streams.dirty_count + shard,
                sizeof(state.tracked_stream3.dirty_count),
                cudaMemcpyDeviceToHost), "cudaMemcpy tracked stream3 dirty count");
            check_cuda(cudaMemcpy(
                &state.tracked_stream3.processing_flag,
                memory.streams.processing_flag + shard,
                sizeof(state.tracked_stream3.processing_flag),
                cudaMemcpyDeviceToHost), "cudaMemcpy tracked stream3 processing flag");
        }
        std::uint32_t spill_counts[2]{};
        std::uint32_t spill_active = 0;
        check_cuda(cudaMemcpy(
            spill_counts,
            memory.streams.global_spill_count,
            sizeof(spill_counts),
            cudaMemcpyDeviceToHost), "cudaMemcpy tracked stream3 spill counts");
        check_cuda(cudaMemcpy(
            &spill_active,
            memory.streams.global_spill_active_index,
            sizeof(spill_active),
            cudaMemcpyDeviceToHost), "cudaMemcpy tracked stream3 spill active");
        spill_active &= 1U;
        state.tracked_stream3.active_spill_count = spill_counts[spill_active];
        state.tracked_stream3.inactive_spill_count = spill_counts[spill_active ^ 1U];
        state.tracked_stream3.spill_capacity = plan.config.global_spill_capacity;
        if (state.tracked_stream3.partition_found &&
            state.tracked_stream3.partition_local >= state.tracked_stream3.group_offset) {
            state.tracked_stream3.local_in_group =
                state.tracked_stream3.partition_local - state.tracked_stream3.group_offset;
            if (state.tracked_stream3.local_in_group >= state.tracked_stream3.shard_write_count) {
                state.tracked_stream3.spill_idx =
                    state.tracked_stream3.shard_spill_offset +
                    state.tracked_stream3.local_in_group -
                    state.tracked_stream3.shard_write_count;
                state.tracked_stream3.spill_capacity_drop =
                    state.tracked_stream3.spill_idx >= plan.config.global_spill_capacity;
            }
        }
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
#if BEAM_DEBUG_STREAM_TIMING
            check_cuda(cudaEventRecord(ring_timing_start[ring], streams.stream1), "cudaEventRecord stream12 timing start");
#endif
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
#if BEAM_DEBUG_STREAM_TIMING
            check_cuda(cudaEventRecord(ring_timing_stop[ring], streams.stream1), "cudaEventRecord stream12 timing stop");
#endif
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
        if (ready_count > plan.storage_shard_count) {
            throw std::runtime_error("stream3 ready shard count exceeds storage shard count");
        }
        check_cuda(cudaMemcpy(
            host_ready_shards.data(),
            memory.streams.stream3_ready_shard_list,
            static_cast<std::uint64_t>(ready_count) * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost), "cudaMemcpy stream3 ready shard list to host scheduler");
        for (std::uint32_t i = 0; i < ready_count; ++i) {
            const std::uint32_t shard = host_ready_shards[i];
            if (shard >= plan.storage_shard_count) {
                throw std::runtime_error("stream3 ready shard index exceeds storage shard count");
            }
            pending_stream4_shards.push_back(shard);
        }
        update_stream4_queue_peaks();
        debug_pipeline_stats("ready_queue_append", ready_count, UINT32_MAX);
        return ready_count;
    };

    const auto periodic_threshold_due = [&]() -> bool {
        return !multi_rank &&
            plan.config.global_threshold_update_period_shards != 0 &&
            stream4_jobs_since_threshold_update >= plan.config.global_threshold_update_period_shards;
    };

    std::uint64_t completed_exchange_rounds = 0;
    const auto run_stream5_exchange_and_collect = [&](std::uint32_t ring, bool zero_send) {
        if (!multi_rank) {
            return;
        }
        if (ring >= plan.stream5_slot_count) {
            throw std::runtime_error("stream5 exchange ring exceeds slot count");
        }
        CandidateMeta* send_buffer =
            memory.streams.remote_send_buffer + static_cast<std::uint64_t>(ring) * plan.stream5_send_slot_capacity;
        CandidateMeta* recv_buffer =
            memory.streams.remote_recv_buffer + static_cast<std::uint64_t>(ring) * plan.stream5_recv_slot_capacity;
        if (zero_send) {
            std::fill(host_send_count.begin(), host_send_count.end(), 0U);
            std::fill(host_send_offset.begin(), host_send_offset.end(), 0U);
            check_cuda(cudaMemcpyAsync(
                memory.streams.send_count,
                host_send_count.data(),
                static_cast<std::uint64_t>(plan.config.world_size) * sizeof(std::uint32_t),
                cudaMemcpyHostToDevice,
                streams.stream5), "cudaMemcpyAsync zero stream5 send counts");
            check_cuda(cudaMemcpyAsync(
                memory.streams.send_offset,
                host_send_offset.data(),
                (static_cast<std::uint64_t>(plan.config.world_size) + 1ULL) * sizeof(std::uint32_t),
                cudaMemcpyHostToDevice,
                streams.stream5), "cudaMemcpyAsync zero stream5 send offsets");
        } else {
            check_cuda(cudaMemcpy(
                host_send_count.data(),
                memory.streams.send_count,
                static_cast<std::uint64_t>(plan.config.world_size) * sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost), "cudaMemcpy stream5 send counts");
            check_cuda(cudaMemcpy(
                host_send_offset.data(),
                memory.streams.send_offset,
                (static_cast<std::uint64_t>(plan.config.world_size) + 1ULL) * sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost), "cudaMemcpy stream5 send offsets");
            if (host_send_offset[plan.config.world_size] > plan.stream5_send_slot_capacity) {
                throw std::runtime_error("stream5 send count exceeds slot capacity");
            }
        }

#if BEAM_DEBUG_STREAM_TIMING
        check_cuda(cudaEventRecord(stream5_timing_start[0], streams.stream5), "cudaEventRecord stream5 exchange timing start");
#endif
        stream5_exchange_counts_nccl_cuda(
            memory.streams.send_count,
            memory.streams.recv_count,
            plan.config.local_rank,
            plan.config.world_size,
            collective->comm,
            streams.stream5);
        check_cuda(cudaStreamSynchronize(streams.stream5), "cudaStreamSynchronize stream5 count exchange");
        check_cuda(cudaMemcpy(
            host_recv_count.data(),
            memory.streams.recv_count,
            static_cast<std::uint64_t>(plan.config.world_size) * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost), "cudaMemcpy stream5 recv counts");
        std::uint64_t recv_total_64 = 0;
        host_recv_offset[0] = 0;
        for (std::uint32_t peer = 0; peer < plan.config.world_size; ++peer) {
            recv_total_64 += host_recv_count[peer];
            if (recv_total_64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw std::runtime_error("stream5 recv count exceeds uint32 range");
            }
            host_recv_offset[peer + 1U] = static_cast<std::uint32_t>(recv_total_64);
        }
        if (recv_total_64 > plan.stream5_recv_slot_capacity) {
            throw std::runtime_error(
                "stream5 recv count exceeds slot capacity: recv=" +
                std::to_string(recv_total_64) +
                " capacity=" + std::to_string(plan.stream5_recv_slot_capacity) +
                " scale_ppm=" + std::to_string(plan.config.stream5_recv_capacity_scale_ppm));
        }
        stream5_write_recv_offsets_cuda(
            memory.streams.recv_offset,
            host_recv_offset.data(),
            plan.config.world_size,
            streams.stream5);
        stream5_exchange_payload_nccl_cuda(
            send_buffer,
            memory.streams.send_count,
            memory.streams.send_offset,
            recv_buffer,
            memory.streams.recv_offset,
            host_send_count.data(),
            host_send_offset.data(),
            host_recv_count.data(),
            host_recv_offset.data(),
            plan.config.local_rank,
            plan.config.world_size,
            collective->comm,
            streams.stream5);
#if BEAM_DEBUG_STREAM_TIMING
        check_cuda(cudaEventRecord(stream5_timing_stop[0], streams.stream5), "cudaEventRecord stream5 exchange timing stop");
#endif
        check_cuda(cudaStreamSynchronize(streams.stream5), "cudaStreamSynchronize stream5 payload exchange");
#if BEAM_DEBUG_STREAM_TIMING
        accumulate_elapsed_ms(
            stream5_timing_start[0],
            stream5_timing_stop[0],
            state.stream5_ms_total,
            nullptr,
            "cudaEventElapsedTime stream5 exchange");
#endif
        if (recv_total_64 != 0ULL) {
            stream3_collect_remote_recv_cuda(
                recv_buffer,
                memory.streams.recv_count,
                memory.streams.recv_offset,
                memory.streams.survivor_shard,
                memory.streams.clean_count,
                memory.streams.dirty_count,
                memory.streams.processing_flag,
                memory.streams.global_spill_buffer_a,
                memory.streams.global_spill_buffer_b,
                memory.streams.global_spill_count,
                memory.streams.global_spill_active_index,
                memory.streams.stream3_write_buffer_index,
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
                static_cast<std::uint32_t>(recv_total_64),
                plan.config.world_size,
                plan.config.shard_count,
                plan.config.shard_buffer_count,
                plan.config.shard_capacity_candidates,
                plan.config.stream4_batch_candidates,
                plan.config.global_spill_capacity,
                streams.stream3,
                memory.streams.fatal_error_flag,
                memory.streams.fatal_error_trace);
            check_cuda(cudaStreamSynchronize(streams.stream3), "cudaStreamSynchronize stream3 remote recv collect");
            throw_if_stream_fatal_error("stream3_remote_recv_collect");
            update_global_spill_peak();
            append_stream3_ready_queue();
        }
        ++completed_exchange_rounds;
    };

    const auto force_periodic_threshold_update = [&]() {
        if (stream3_active) {
            throw std::runtime_error("periodic threshold update requested while stream3 is active");
        }
        wait_all_stream4_slots();
#if BEAM_DEBUG_STREAM_TIMING
        check_cuda(cudaEventRecord(stream5_timing_start[0], streams.stream5), "cudaEventRecord stream5 timing start");
#endif
        update_threshold_global(plan, memory, streams.stream5, true, collective);
#if BEAM_DEBUG_STREAM_TIMING
        check_cuda(cudaEventRecord(stream5_timing_stop[0], streams.stream5), "cudaEventRecord stream5 timing stop");
#endif
        check_cuda(cudaStreamSynchronize(streams.stream5), "cudaStreamSynchronize stream5 periodic threshold update");
#if BEAM_DEBUG_STREAM_TIMING
        accumulate_elapsed_ms(
            stream5_timing_start[0],
            stream5_timing_stop[0],
            state.stream5_ms_total,
            nullptr,
            "cudaEventElapsedTime stream5 threshold");
#endif
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
            debug_pipeline_stats("stream4_launch", shard, slot);
#if BEAM_DEBUG_STREAM_TIMING
            check_cuda(
                cudaEventRecord(stream4_timing_start[slot], streams.stream4_slot_streams[slot]),
                "cudaEventRecord stream4 timing start");
#endif
            check_cuda(
                cudaGraphLaunch(graphs.stream4_shard_execs[graph_idx], streams.stream4_slot_streams[slot]),
                "cudaGraphLaunch stream4");
#if BEAM_DEBUG_STREAM_TIMING
            check_cuda(
                cudaEventRecord(stream4_timing_stop[slot], streams.stream4_slot_streams[slot]),
                "cudaEventRecord stream4 timing stop");
#endif
            check_cuda(
                cudaEventRecord(streams.stream4_slot_done[slot], streams.stream4_slot_streams[slot]),
                "cudaEventRecord stream4 slot done");
            stream4_slot_busy[slot] = true;
            stream4_slot_shard[slot] = shard;
            stream4_busy_slots.push_back(slot);
            update_stream4_queue_peaks();
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
                debug_pipeline_stats("stream4_launch_blocking", shard, slot);
#if BEAM_DEBUG_STREAM_TIMING
                check_cuda(
                    cudaEventRecord(stream4_timing_start[slot], streams.stream4_slot_streams[slot]),
                    "cudaEventRecord stream4 timing start");
#endif
                check_cuda(
                    cudaGraphLaunch(graphs.stream4_shard_execs[graph_idx], streams.stream4_slot_streams[slot]),
                    "cudaGraphLaunch stream4");
#if BEAM_DEBUG_STREAM_TIMING
                check_cuda(
                    cudaEventRecord(stream4_timing_stop[slot], streams.stream4_slot_streams[slot]),
                    "cudaEventRecord stream4 timing stop");
#endif
                check_cuda(
                    cudaEventRecord(streams.stream4_slot_done[slot], streams.stream4_slot_streams[slot]),
                    "cudaEventRecord stream4 slot done");
                stream4_slot_busy[slot] = true;
                stream4_slot_shard[slot] = shard;
                stream4_busy_slots.push_back(slot);
                update_stream4_queue_peaks();
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
        if (!stream3_has_writable_buffer()) {
            stream3_build_ready_shard_queue_cuda(
                memory.streams.clean_count,
                memory.streams.dirty_count,
                memory.streams.processing_flag,
                memory.streams.stream3_write_buffer_index,
                memory.streams.stream3_ready_flag,
                memory.streams.stream3_ready_shard_list,
                memory.streams.stream3_ready_count,
                plan.config.shard_count,
                plan.config.shard_buffer_count,
                plan.config.shard_capacity_candidates,
                plan.config.stream3_batch_candidates,
                plan.config.stream4_trigger_candidates,
                false,
                false,
                streams.stream3);
            check_cuda(cudaStreamSynchronize(streams.stream3), "cudaStreamSynchronize stream3 backpressure ready queue");
            throw_if_stream_fatal_error("stream3_backpressure_ready_queue");
            const std::uint32_t queued = append_stream3_ready_queue();
            const std::uint32_t launched = launch_pending_stream4_shards();
            if (queued == 0U && launched == 0U && pending_stream4_count() == 0U && stream4_busy_slots.empty()) {
                throw std::runtime_error("stream3 double-buffer backpressure has no ready stream4 work");
            }
            return queued != 0U || launched != 0U;
        }
        const std::uint32_t ring = stream3_ready_rings.front();
        stream3_ready_rings.pop_front();
        if (ring_state[ring] != RingState::ReadyForStream3) {
            throw std::runtime_error("stream3 ready queue contains a non-ready ring");
        }
        scan_tracked_generated_candidate(ring);
        debug_pipeline_stats("stream3_launch", ring, UINT32_MAX);
#if BEAM_DEBUG_STREAM_TIMING
        check_cuda(cudaEventRecord(stream3_timing_start[ring], streams.stream3), "cudaEventRecord stream3 timing start");
#endif
        check_cuda(cudaGraphLaunch(graphs.stream3_ring_execs[ring], streams.stream3), "cudaGraphLaunch stream3");
#if BEAM_DEBUG_STREAM_TIMING
        check_cuda(cudaEventRecord(stream3_timing_stop[ring], streams.stream3), "cudaEventRecord stream3 timing stop");
#endif
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
#if BEAM_DEBUG_STREAM_TIMING
                accumulate_elapsed_ms(
                    ring_timing_start[ring],
                    ring_timing_stop[ring],
                    state.stream12_ms_total,
                    &state.stream12_ms_max,
                    "cudaEventElapsedTime stream12 ring");
#endif
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
#if BEAM_DEBUG_STREAM_TIMING
        accumulate_elapsed_ms(
            ring_timing_start[ring],
            ring_timing_stop[ring],
            state.stream12_ms_total,
            &state.stream12_ms_max,
            "cudaEventElapsedTime stream12 ring");
#endif
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
        throw_if_stream_fatal_error("stream3_ring_complete");
#if BEAM_DEBUG_STREAM_TIMING
        accumulate_elapsed_ms(
            stream3_timing_start[ring],
            stream3_timing_stop[ring],
            state.stream3_ring_ms_total,
            &state.stream3_ring_ms_max,
            "cudaEventElapsedTime stream3 ring");
#endif
        update_global_spill_peak();
        stream3_active = false;
        stream3_active_ring = plan.config.ring_count;
        run_stream5_exchange_and_collect(ring, false);
        ring_state[ring] = RingState::Free;
        if (!state.stop_requested) {
            append_stream3_ready_queue();
        }
        scan_tracked_stream3_path(ring);
        scan_tracked_after_stream3(ring);
        debug_pipeline_stats("stream3_complete", ring, UINT32_MAX);
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

    check_cuda(cudaMemset(memory.streams.fatal_error_flag, 0, sizeof(std::uint32_t)), "cudaMemset stream fatal flag");
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

    while (multi_rank && completed_exchange_rounds < global_exchange_rounds) {
        const std::uint32_t ring =
            static_cast<std::uint32_t>(completed_exchange_rounds % plan.stream5_slot_count);
        run_stream5_exchange_and_collect(ring, true);
        launch_pending_stream4_shards();
    }

    drain_pending_stream4_shards();
    if (!multi_rank) {
        force_periodic_threshold_update();
    }

    for (std::uint32_t flush_round = 0; flush_round < plan.storage_shard_count + 2U; ++flush_round) {
#if BEAM_DEBUG_STREAM_TIMING
        check_cuda(
            cudaEventRecord(stream3_spill_drain_timing_start[0], streams.stream3),
            "cudaEventRecord stream3 spill drain timing start");
#endif
        if (plan.config.shard_buffer_count == 1U) {
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
                plan.config.shard_capacity_candidates,
                plan.config.stream4_batch_candidates,
                streams.stream3,
                memory.streams.fatal_error_flag,
                memory.streams.fatal_error_trace);
        }
        stream3_build_ready_shard_queue_cuda(
            memory.streams.clean_count,
            memory.streams.dirty_count,
            memory.streams.processing_flag,
            memory.streams.stream3_write_buffer_index,
            memory.streams.stream3_ready_flag,
            memory.streams.stream3_ready_shard_list,
            memory.streams.stream3_ready_count,
            plan.config.shard_count,
            plan.config.shard_buffer_count,
            plan.config.shard_capacity_candidates,
            plan.config.stream3_batch_candidates,
            plan.config.stream4_trigger_candidates,
            true,
            true,
            streams.stream3);
#if BEAM_DEBUG_STREAM_TIMING
        check_cuda(
            cudaEventRecord(stream3_spill_drain_timing_stop[0], streams.stream3),
            "cudaEventRecord stream3 spill drain timing stop");
#endif
        check_cuda(cudaStreamSynchronize(streams.stream3), "cudaStreamSynchronize final spill drain");
#if BEAM_DEBUG_STREAM_TIMING
        accumulate_elapsed_ms(
            stream3_spill_drain_timing_start[0],
            stream3_spill_drain_timing_stop[0],
            state.stream3_spill_drain_ms_total,
            nullptr,
            "cudaEventElapsedTime stream3 spill drain");
#endif
        throw_if_stream_fatal_error("final_spill_drain");
        update_global_spill_peak();
        debug_pipeline_stats("final_spill_drain", flush_round, UINT32_MAX);
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
            static_cast<std::uint64_t>(plan.storage_shard_count) * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost), "cudaMemcpy clean_count final flush");
        check_cuda(cudaMemcpy(
            host_dirty.data(),
            memory.streams.dirty_count,
            static_cast<std::uint64_t>(plan.storage_shard_count) * sizeof(std::uint32_t),
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
        if (!multi_rank &&
            stream4_jobs_since_threshold_update != 0U &&
            (periodic_threshold_due() || spill_remaining)) {
            force_periodic_threshold_update();
        }
        if (!spill_remaining ||
            (!any_dirty && total_clean >= plan.derived.global_beam_width_effective)) {
            break;
        }
        if (flush_round + 1U == plan.storage_shard_count + 2U) {
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
    std::uint64_t current_frontier_size,
    CandidateMeta* history_host_buffer,
    std::uint32_t history_host_capacity,
    cudaStream_t history_stream,
    cudaEvent_t history_copy_done,
    const Hash128* tracked_prefinal_hash,
    const DispatcherCollective* collective) {
    NvtxRange range("Dispatcher_finalize_depth_single_gpu");
    const bool multi_rank = plan.config.world_size > 1U;
    if (!multi_rank && plan.config.local_rank != 0) {
        throw std::invalid_argument("single gpu finalization requires WORLD_SIZE=1 and LOCAL_RANK=0");
    }
    if (multi_rank && (collective == nullptr || collective->comm == nullptr)) {
        throw std::invalid_argument("multi rank finalization requires NCCL collective");
    }
    if (tables.generators == nullptr) {
        throw std::invalid_argument("finalization requires generators");
    }

    FinalizeDepthState result{};
#if BEAM_DEBUG_STREAM_TIMING
    cudaEvent_t timing_start = nullptr;
    cudaEvent_t timing_stop = nullptr;
    check_cuda(cudaEventCreate(&timing_start), "cudaEventCreate finalize timing start");
    check_cuda(cudaEventCreate(&timing_stop), "cudaEventCreate finalize timing stop");
    struct FinalizeTimingCleanup {
        cudaEvent_t& start;
        cudaEvent_t& stop;
        ~FinalizeTimingCleanup() {
            if (start) {
                cudaEventDestroy(start);
            }
            if (stop) {
                cudaEventDestroy(stop);
            }
        }
    } finalize_timing_cleanup{timing_start, timing_stop};

    auto record_timing_start = [&](cudaStream_t stream, const char* op) {
        check_cuda(cudaEventRecord(timing_start, stream), op);
    };
    auto record_timing_stop_and_sync = [&](cudaStream_t stream, double& output_ms, const char* record_op, const char* sync_op, const char* elapsed_op) {
        check_cuda(cudaEventRecord(timing_stop, stream), record_op);
        check_cuda(cudaStreamSynchronize(stream), sync_op);
        accumulate_elapsed_ms(timing_start, timing_stop, output_ms, nullptr, elapsed_op);
    };
#endif
#if BEAM_DEBUG_STREAM_TIMING
    record_timing_start(streams.stream5, "cudaEventRecord final threshold timing start");
#endif
    update_threshold_global(plan, memory, streams.stream5, false, collective);
#if BEAM_DEBUG_STREAM_TIMING
    record_timing_stop_and_sync(
        streams.stream5,
        result.stream5_threshold_ms,
        "cudaEventRecord final threshold timing stop",
        "cudaStreamSynchronize stream5 final threshold",
        "cudaEventElapsedTime stream5 final threshold");
#else
    check_cuda(cudaStreamSynchronize(streams.stream5), "cudaStreamSynchronize stream5 final threshold");
#endif
    std::uint32_t final_threshold = UINT32_THRESHOLD_MAX;
    check_cuda(cudaMemcpy(
        &final_threshold,
        memory.streams.current_threshold,
        sizeof(final_threshold),
        cudaMemcpyDeviceToHost), "cudaMemcpy final threshold");

    result.final_threshold = final_threshold;
    if (tracked_prefinal_hash != nullptr) {
        scan_tracked_prefinal_hash(plan, memory, streams, *tracked_prefinal_hash, result);
    }

    if (multi_rank) {
        const std::uint32_t world_size = plan.config.world_size;
        const std::uint32_t local_rank = plan.config.local_rank;
        const auto ceil_div_local = [](std::uint64_t a, std::uint64_t b) {
            return b == 0ULL ? 0ULL : (a + b - 1ULL) / b;
        };
        const auto target_rank_for_global = [&](std::uint64_t global_idx, std::uint64_t global_count) {
            std::uint32_t target =
                static_cast<std::uint32_t>((global_idx * static_cast<std::uint64_t>(world_size)) / global_count);
            if (target >= world_size) {
                target = world_size - 1U;
            }
            return target;
        };
        const auto exchange_u64_items = [&](
            const char* exchange_name,
            const void* host_send,
            const FinalExchangePlan& send_plan,
            void* device_send,
            std::uint32_t device_send_capacity,
            void* device_recv,
            std::uint32_t device_recv_capacity,
            std::uint32_t words_per_item,
            void* host_recv,
            std::uint32_t& recv_total_out) {
            const FinalExchangePlan send_plan_snapshot = send_plan;
#if BEAM_DEBUG_FINAL_EXCHANGE_TRACE
            std::cout << "final_exchange_trace"
                      << " rank=" << local_rank
                      << " label=" << exchange_name
                      << " phase=entry"
                      << " send_total=" << send_plan_snapshot.total
                      << " device_send_capacity=" << device_send_capacity
                      << " device_recv_capacity=" << device_recv_capacity
                      << " words_per_item=" << words_per_item
                      << " host_send=" << (host_send != nullptr ? 1 : 0)
                      << " host_recv=" << (host_recv != nullptr ? 1 : 0)
                      << "\n";
            log_final_exchange_counts(
                (std::string(exchange_name) + "_send_snapshot").c_str(),
                local_rank,
                send_plan_snapshot.count,
                send_plan_snapshot.offset);
#endif
            if (send_plan_snapshot.total > device_send_capacity) {
                throw std::runtime_error("exchange send total exceeds device capacity");
            }
            if (send_plan_snapshot.total != 0U && host_send != nullptr) {
                check_cuda(cudaMemcpyAsync(
                    device_send,
                    host_send,
                    static_cast<std::uint64_t>(send_plan_snapshot.total) * words_per_item * sizeof(std::uint64_t),
                    cudaMemcpyHostToDevice,
                    streams.stream5), "cudaMemcpyAsync exchange host send to device");
            }
            check_cuda(cudaMemcpyAsync(
                memory.final.final_send_count,
                send_plan_snapshot.count.data(),
                static_cast<std::uint64_t>(world_size) * sizeof(std::uint32_t),
                cudaMemcpyHostToDevice,
                streams.stream5), "cudaMemcpyAsync exchange send counts");
            check_cuda(cudaMemcpyAsync(
                memory.final.final_send_offset,
                send_plan_snapshot.offset.data(),
                (static_cast<std::uint64_t>(world_size) + 1ULL) * sizeof(std::uint32_t),
                cudaMemcpyHostToDevice,
                streams.stream5), "cudaMemcpyAsync exchange send offsets");
            stream5_exchange_counts_nccl_cuda(
                memory.final.final_send_count,
                memory.final.final_recv_count,
                local_rank,
                world_size,
                collective->comm,
                streams.stream5);
            check_cuda(cudaStreamSynchronize(streams.stream5), "cudaStreamSynchronize exchange counts");
            std::vector<std::uint32_t> recv_counts(world_size);
            check_cuda(cudaMemcpy(
                recv_counts.data(),
                memory.final.final_recv_count,
                static_cast<std::uint64_t>(world_size) * sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost), "cudaMemcpy exchange recv counts");
            const FinalExchangePlan recv_plan = make_final_exchange_plan(recv_counts);
#if BEAM_DEBUG_FINAL_EXCHANGE_TRACE
            log_final_exchange_counts(
                (std::string(exchange_name) + "_recv").c_str(),
                local_rank,
                recv_plan.count,
                recv_plan.offset);
            std::cout << "final_exchange_trace"
                      << " rank=" << local_rank
                      << " label=" << exchange_name
                      << " phase=recv_ready"
                      << " recv_total=" << recv_plan.total
                      << "\n";
#endif
            if (recv_plan.total > device_recv_capacity) {
                throw std::runtime_error("exchange recv total exceeds device capacity");
            }
            stream5_write_recv_offsets_cuda(
                memory.final.final_recv_offset,
                recv_plan.offset.data(),
                world_size,
                streams.stream5);
            stream5_exchange_u64_payload_nccl_cuda(
                device_send,
                device_recv,
                send_plan_snapshot.count.data(),
                send_plan_snapshot.offset.data(),
                recv_plan.count.data(),
                recv_plan.offset.data(),
                words_per_item,
                local_rank,
                world_size,
                collective->comm,
                streams.stream5);
            if (recv_plan.total != 0U && host_recv != nullptr) {
                check_cuda(cudaMemcpyAsync(
                    host_recv,
                    device_recv,
                    static_cast<std::uint64_t>(recv_plan.total) * words_per_item * sizeof(std::uint64_t),
                    cudaMemcpyDeviceToHost,
                    streams.stream5), "cudaMemcpyAsync exchange device recv to host");
            }
            check_cuda(cudaStreamSynchronize(streams.stream5), "cudaStreamSynchronize exchange payload");
            recv_total_out = recv_plan.total;
        };

        final_count_score_phase_cuda(
            memory.streams.survivor_shard,
            memory.streams.clean_count,
            memory.final.final_keep_flags,
            memory.final.final_block_counts,
            memory.final.final_block_offsets,
            memory.final.final_candidate_count,
            final_threshold,
            0,
            plan.storage_shard_count,
            plan.config.shard_capacity_candidates,
            plan.config.stream4_batch_candidates,
            streams.stream3);
        check_cuda(cudaStreamSynchronize(streams.stream3), "cudaStreamSynchronize final less count");
        final_allgather_counts_nccl_cuda(
            memory.final.final_candidate_count,
            memory.final.final_recv_count,
            collective->comm,
            streams.stream5);
        check_cuda(cudaStreamSynchronize(streams.stream5), "cudaStreamSynchronize final less allgather");
        std::vector<std::uint32_t> less_counts(world_size);
        check_cuda(cudaMemcpy(
            less_counts.data(),
            memory.final.final_recv_count,
            static_cast<std::uint64_t>(world_size) * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost), "cudaMemcpy final less counts");

        final_count_score_phase_cuda(
            memory.streams.survivor_shard,
            memory.streams.clean_count,
            memory.final.final_keep_flags,
            memory.final.final_block_counts,
            memory.final.final_block_offsets,
            memory.final.final_request_count,
            final_threshold,
            1,
            plan.storage_shard_count,
            plan.config.shard_capacity_candidates,
            plan.config.stream4_batch_candidates,
            streams.stream3);
        check_cuda(cudaStreamSynchronize(streams.stream3), "cudaStreamSynchronize final equal count");
        final_allgather_counts_nccl_cuda(
            memory.final.final_request_count,
            memory.final.final_send_count,
            collective->comm,
            streams.stream5);
        check_cuda(cudaStreamSynchronize(streams.stream5), "cudaStreamSynchronize final equal allgather");
        std::vector<std::uint32_t> equal_counts(world_size);
        check_cuda(cudaMemcpy(
            equal_counts.data(),
            memory.final.final_send_count,
            static_cast<std::uint64_t>(world_size) * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost), "cudaMemcpy final equal counts");

        std::uint64_t less_prefix = 0;
        std::uint64_t equal_prefix = 0;
        std::uint64_t global_less = 0;
        std::uint64_t global_equal = 0;
        for (std::uint32_t rank = 0; rank < world_size; ++rank) {
            if (rank < local_rank) {
                less_prefix += less_counts[rank];
                equal_prefix += equal_counts[rank];
            }
            global_less += less_counts[rank];
            global_equal += equal_counts[rank];
        }
        const std::uint64_t total_available = global_less + global_equal;
        const std::uint64_t global_keep_count =
            std::min<std::uint64_t>(plan.derived.global_beam_width_effective, total_available);
#if BEAM_DEBUG_FINAL_EXCHANGE_TRACE
        std::cout << "final_exchange_trace"
                  << " rank=" << local_rank
                  << " label=threshold"
                  << " final_threshold=" << final_threshold
                  << " less_prefix=" << less_prefix
                  << " equal_prefix=" << equal_prefix
                  << " global_less=" << global_less
                  << " global_equal=" << global_equal
                  << " total_available=" << total_available
                  << " global_keep_count=" << global_keep_count
                  << " global_beam_width_effective=" << plan.derived.global_beam_width_effective
                  << "\n";
        const FinalExchangePlan trace_less_plan = make_final_exchange_plan(less_counts);
        const FinalExchangePlan trace_equal_plan = make_final_exchange_plan(equal_counts);
        log_final_exchange_counts("less_counts", local_rank, trace_less_plan.count, trace_less_plan.offset);
        log_final_exchange_counts("equal_counts", local_rank, trace_equal_plan.count, trace_equal_plan.offset);
#endif
        const std::uint32_t candidate_capacity = static_cast<std::uint32_t>(
            (plan.frontier_states * sizeof(State128)) / sizeof(CandidateMeta));
        CandidateMeta* selected_device = reinterpret_cast<CandidateMeta*>(memory.final.next_frontier_states_tmp);
        final_filter_load_balance_exact_cuda(
            memory.streams.survivor_shard,
            memory.streams.clean_count,
            memory.final.final_keep_flags,
            memory.final.final_block_counts,
            memory.final.final_block_offsets,
            selected_device,
            memory.final.final_candidate_count,
            memory.final.final_request_count,
            final_threshold,
            less_prefix,
            global_less + equal_prefix,
            global_keep_count,
            candidate_capacity,
            plan.storage_shard_count,
            plan.config.shard_capacity_candidates,
            plan.config.stream4_batch_candidates,
            streams.stream3);
        check_cuda(cudaStreamSynchronize(streams.stream3), "cudaStreamSynchronize final exact filter");
        std::uint32_t selected_count = 0;
        check_cuda(cudaMemcpy(
            &selected_count,
            memory.final.final_candidate_count,
            sizeof(selected_count),
            cudaMemcpyDeviceToHost), "cudaMemcpy final selected count");
        if (selected_count > candidate_capacity) {
            throw std::runtime_error("final selected count exceeds candidate capacity");
        }
        std::vector<CandidateMeta> selected(selected_count);
        if (selected_count != 0U) {
            check_cuda(cudaMemcpy(
                selected.data(),
                selected_device,
                static_cast<std::uint64_t>(selected_count) * sizeof(CandidateMeta),
                cudaMemcpyDeviceToHost), "cudaMemcpy final selected candidates");
        }

        std::vector<std::uint32_t> history_count(world_size, 0U);
        std::vector<std::uint32_t> request_count(world_size, 0U);
        std::uint64_t less_seen = 0;
        std::uint64_t equal_seen = 0;
        for (const CandidateMeta& candidate : selected) {
            const bool less = candidate.score_key < final_threshold;
            const std::uint64_t global_idx = less
                ? less_prefix + less_seen++
                : global_less + equal_prefix + equal_seen++;
            if (global_idx >= global_keep_count || global_keep_count == 0ULL) {
                continue;
            }
            const std::uint32_t target_rank = target_rank_for_global(global_idx, global_keep_count);
            const std::uint32_t source_rank = unpack_source_rank(candidate.route_packed);
            if (source_rank >= world_size) {
                throw std::runtime_error("final candidate source rank exceeds WORLD_SIZE");
            }
            ++history_count[target_rank];
            ++request_count[source_rank];
        }
        const FinalExchangePlan history_plan = make_final_exchange_plan(history_count);
        const FinalExchangePlan request_plan = make_final_exchange_plan(request_count);
#if BEAM_DEBUG_FINAL_EXCHANGE_TRACE
        std::cout << "final_exchange_trace"
                  << " rank=" << local_rank
                  << " label=selected_summary"
                  << " selected_count=" << selected_count
                  << " history_send_total=" << history_plan.total
                  << " request_send_total=" << request_plan.total
                  << "\n";
        log_final_exchange_counts("history_send_counts", local_rank, history_plan.count, history_plan.offset);
        log_final_exchange_counts("request_send_counts", local_rank, request_plan.count, request_plan.offset);
#endif
        std::vector<std::uint32_t> history_cursor = history_plan.offset;
        std::vector<std::uint32_t> request_cursor = request_plan.offset;
        std::vector<FinalHistoryRecord> history_send(history_plan.total);
        std::vector<FinalRequest> request_send(request_plan.total);
        less_seen = 0;
        equal_seen = 0;
        for (std::uint32_t selected_index = 0; selected_index < selected_count; ++selected_index) {
            const CandidateMeta& candidate = selected[selected_index];
            const bool less = candidate.score_key < final_threshold;
            const std::uint64_t global_idx = less
                ? less_prefix + less_seen++
                : global_less + equal_prefix + equal_seen++;
            if (global_idx >= global_keep_count || global_keep_count == 0ULL) {
                continue;
            }
            const std::uint32_t target_rank = target_rank_for_global(global_idx, global_keep_count);
            const std::uint64_t target_begin =
                ceil_div_local(static_cast<std::uint64_t>(target_rank) * global_keep_count, world_size);
            const std::uint32_t target_local_idx = static_cast<std::uint32_t>(global_idx - target_begin);
            const std::uint32_t source_rank = unpack_source_rank(candidate.route_packed);
#if BEAM_DEBUG_FINAL_EXCHANGE_TRACE
            if (selected_index < 8U || selected_count - selected_index <= 4U) {
                log_final_exchange_candidate(
                    "selected",
                    local_rank,
                    selected_index,
                    candidate,
                    target_rank,
                    global_idx);
            } else if (selected_index == 8U) {
                std::cout << "final_exchange_trace_candidate_gap"
                          << " rank=" << local_rank
                          << " label=selected"
                          << " skipped=" << (selected_count > 12U ? selected_count - 12U : 0U)
                          << "\n";
            }
#endif
            history_send[history_cursor[target_rank]++] =
                FinalHistoryRecord{candidate, target_local_idx, 0U, {0ULL, 0ULL, 0ULL}};
            FinalRequest request{};
            request.parent_idx = candidate.parent_idx;
            request.target_local_idx = target_local_idx;
            request.return_rank = static_cast<std::uint16_t>(target_rank);
            request.move = unpack_move(candidate.route_packed);
            request_send[request_cursor[source_rank]++] = request;
        }
#if BEAM_DEBUG_FINAL_EXCHANGE_TRACE
        log_final_exchange_requests("request_send", local_rank, request_send);
#endif

        const std::uint32_t history_record_capacity = static_cast<std::uint32_t>(
            (plan.frontier_states * sizeof(FinalResponse)) / sizeof(FinalHistoryRecord));
        std::uint32_t history_recv_total = 0;
        std::vector<FinalHistoryRecord> history_recv(history_record_capacity);
        exchange_u64_items(
            "history",
            history_send.data(),
            history_plan,
            memory.final.final_response_buffer,
            history_record_capacity,
            memory.final.next_frontier_states_tmp,
            history_record_capacity,
            static_cast<std::uint32_t>(sizeof(FinalHistoryRecord) / sizeof(std::uint64_t)),
            history_recv.data(),
            history_recv_total);
        if (history_recv_total > history_record_capacity) {
            throw std::runtime_error("history recv total exceeds host capacity");
        }
        if (history_host_buffer != nullptr) {
            if (history_stream == nullptr || history_copy_done == nullptr) {
                throw std::invalid_argument("history copy requires history stream and completion event");
            }
            if (history_recv_total > history_host_capacity) {
                throw std::runtime_error("history host buffer capacity is smaller than received history count");
            }
            for (std::uint32_t i = 0; i < history_recv_total; ++i) {
                if (history_recv[i].target_local_idx >= history_host_capacity) {
                    throw std::runtime_error("history target local index exceeds host capacity");
                }
                history_host_buffer[history_recv[i].target_local_idx] = history_recv[i].meta;
            }
            check_cuda(cudaEventRecord(history_copy_done, history_stream), "cudaEventRecord history copy done");
        }

        const std::uint32_t request_device_capacity =
            static_cast<std::uint32_t>(plan.frontier_states);
        std::uint32_t request_recv_total = 0;
        std::vector<FinalRequest> request_recv(static_cast<std::uint64_t>(plan.frontier_states) * 2ULL);
        exchange_u64_items(
            "request",
            request_send.data(),
            request_plan,
            memory.final.next_frontier_states_tmp,
            static_cast<std::uint32_t>((plan.frontier_states * sizeof(State128)) / sizeof(FinalRequest)),
            memory.final.final_candidate_buffer,
            static_cast<std::uint32_t>((plan.frontier_states * sizeof(CandidateMeta)) / sizeof(FinalRequest)),
            static_cast<std::uint32_t>(sizeof(FinalRequest) / sizeof(std::uint64_t)),
            request_recv.data(),
            request_recv_total);
        if (request_recv_total > request_recv.size()) {
            throw std::runtime_error("request recv total exceeds host capacity");
        }
#if BEAM_DEBUG_FINAL_EXCHANGE_TRACE
        std::vector<FinalRequest> request_recv_view(
            request_recv.begin(),
            request_recv.begin() + static_cast<std::ptrdiff_t>(request_recv_total));
        log_final_exchange_requests("request_recv", local_rank, request_recv_view);
#endif

        std::vector<std::uint32_t> response_count(world_size, 0U);
        for (std::uint32_t i = 0; i < request_recv_total; ++i) {
            if (request_recv[i].return_rank >= world_size) {
#if BEAM_DEBUG_FINAL_EXCHANGE_TRACE
                std::cout << "final_exchange_trace_invalid_request"
                          << " rank=" << local_rank
                          << " index=" << i
                          << " request_recv_total=" << request_recv_total
                          << " world_size=" << world_size
                          << " parent_idx=" << request_recv[i].parent_idx
                          << " target_local_idx=" << request_recv[i].target_local_idx
                          << " return_rank=" << request_recv[i].return_rank
                          << " move=" << static_cast<std::uint32_t>(request_recv[i].move)
                          << " pad=" << static_cast<std::uint32_t>(request_recv[i].pad)
                          << "\n";
#endif
                throw std::runtime_error("final request return rank exceeds WORLD_SIZE");
            }
            ++response_count[request_recv[i].return_rank];
        }
        const FinalExchangePlan response_plan = make_final_exchange_plan(response_count);
#if BEAM_DEBUG_FINAL_EXCHANGE_TRACE
        std::cout << "final_exchange_trace"
                  << " rank=" << local_rank
                  << " label=response_summary"
                  << " request_recv_total=" << request_recv_total
                  << " response_send_total=" << response_plan.total
                  << "\n";
        log_final_exchange_counts("response_send_counts", local_rank, response_plan.count, response_plan.offset);
#endif
        if (response_plan.total > request_device_capacity) {
            throw std::runtime_error("response send request count exceeds device capacity");
        }
        std::vector<std::uint32_t> response_cursor = response_plan.offset;
        std::vector<FinalRequest> response_requests(response_plan.total);
        for (std::uint32_t i = 0; i < request_recv_total; ++i) {
            response_requests[response_cursor[request_recv[i].return_rank]++] = request_recv[i];
        }
#if BEAM_DEBUG_FINAL_EXCHANGE_TRACE
        log_final_exchange_requests("response_requests", local_rank, response_requests);
#endif
        if (response_plan.total != 0U) {
            check_cuda(cudaMemcpyAsync(
                memory.final.final_request_buffer,
                response_requests.data(),
                static_cast<std::uint64_t>(response_plan.total) * sizeof(FinalRequest),
                cudaMemcpyHostToDevice,
                streams.stream3), "cudaMemcpyAsync response requests to device");
            final_materialize_responses_cuda(
                memory.current_frontier_states,
                memory.final.final_request_buffer,
                tables.generators,
                reinterpret_cast<FinalResponse*>(memory.final.next_frontier_states_tmp),
                response_plan.total,
                streams.stream3);
        }
        check_cuda(cudaStreamSynchronize(streams.stream3), "cudaStreamSynchronize final response materialize");

        std::uint32_t response_recv_total = 0;
        exchange_u64_items(
            "response",
            nullptr,
            response_plan,
            memory.final.next_frontier_states_tmp,
            request_device_capacity,
            memory.final.final_response_buffer,
            request_device_capacity,
            static_cast<std::uint32_t>(sizeof(FinalResponse) / sizeof(std::uint64_t)),
            nullptr,
            response_recv_total);
        final_scatter_responses_cuda(
            memory.final.final_response_buffer,
            memory.final.next_frontier_states_tmp,
            response_recv_total,
            streams.stream3);
        if (history_host_buffer != nullptr && history_recv_total != response_recv_total) {
            throw std::runtime_error("history recv count does not match response recv count");
        }
        if (response_recv_total != 0U) {
            check_cuda(cudaMemcpyAsync(
                memory.current_frontier_states,
                memory.final.next_frontier_states_tmp,
                static_cast<std::uint64_t>(response_recv_total) * sizeof(State128),
                cudaMemcpyDeviceToDevice,
                streams.stream3), "cudaMemcpyAsync multi next frontier to current");
        }
        check_cuda(cudaStreamSynchronize(streams.stream3), "cudaStreamSynchronize multi final materialize");

        result.final_candidate_count = response_recv_total;
        result.final_request_count = request_recv_total;
        result.next_frontier_size = response_recv_total;

        check_cuda(cudaMemsetAsync(
            memory.streams.clean_count,
            0,
            static_cast<std::uint64_t>(plan.storage_shard_count) * sizeof(std::uint32_t),
            streams.stream3), "cudaMemsetAsync reset clean count");
        check_cuda(cudaMemsetAsync(
            memory.streams.dirty_count,
            0,
            static_cast<std::uint64_t>(plan.storage_shard_count) * sizeof(std::uint32_t),
            streams.stream3), "cudaMemsetAsync reset dirty count");
        check_cuda(cudaMemsetAsync(
            memory.streams.processing_flag,
            0,
            static_cast<std::uint64_t>(plan.storage_shard_count) * sizeof(std::uint32_t),
            streams.stream3), "cudaMemsetAsync reset processing flag");
        check_cuda(cudaMemsetAsync(
            memory.streams.global_spill_count,
            0,
            2ULL * sizeof(std::uint32_t),
            streams.stream3), "cudaMemsetAsync reset global spill counts");
        check_cuda(cudaMemsetAsync(
            memory.streams.global_spill_active_index,
            0,
            sizeof(std::uint32_t),
            streams.stream3), "cudaMemsetAsync reset global spill active index");
        check_cuda(cudaMemsetAsync(
            memory.streams.threshold_initialized,
            0,
            sizeof(std::uint32_t),
            streams.stream3), "cudaMemsetAsync reset threshold initialized");
        check_cuda(cudaMemsetAsync(
            memory.streams.current_threshold,
            0xff,
            sizeof(std::uint32_t),
            streams.stream3), "cudaMemsetAsync reset threshold");
        check_cuda(cudaStreamSynchronize(streams.stream3), "cudaStreamSynchronize multi final reset");
        return result;
    }

#if BEAM_DEBUG_STREAM_TIMING
    record_timing_start(streams.stream3, "cudaEventRecord final filter timing start");
#endif
#if BEAM_DEBUG_FINAL_VALIDATE
    check_cuda(cudaMemsetAsync(
        memory.final.final_request_buffer,
        0xff,
        static_cast<std::uint64_t>(plan.frontier_states) * sizeof(FinalRequest),
        streams.stream3), "cudaMemsetAsync debug init final request buffer");
#endif
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
        plan.storage_shard_count,
        plan.config.shard_capacity_candidates,
        plan.config.stream4_batch_candidates,
        streams.stream3);
#if BEAM_DEBUG_STREAM_TIMING
    record_timing_stop_and_sync(
        streams.stream3,
        result.stream3_final_filter_ms,
        "cudaEventRecord final filter timing stop",
        "cudaStreamSynchronize final filter load balance",
        "cudaEventElapsedTime stream3 final filter");
#else
    check_cuda(cudaStreamSynchronize(streams.stream3), "cudaStreamSynchronize final filter load balance");
#endif

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

#if BEAM_DEBUG_STREAM_TIMING
    record_timing_start(streams.stream3, "cudaEventRecord final materialize timing start");
#endif
    if (final_request_count != 0) {
#if BEAM_DEBUG_FINAL_VALIDATE
        validate_final_requests_cuda(
            memory.final.final_request_buffer,
            final_request_count,
            current_frontier_size,
            final_request_count,
            streams.stream3);
#endif
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
#if BEAM_DEBUG_STREAM_TIMING
    record_timing_stop_and_sync(
        streams.stream3,
        result.stream3_final_materialize_ms,
        "cudaEventRecord final materialize timing stop",
        "cudaStreamSynchronize final materialize",
        "cudaEventElapsedTime stream3 final materialize");
#else
    check_cuda(cudaStreamSynchronize(streams.stream3), "cudaStreamSynchronize final materialize");
#endif
#if BEAM_DEBUG_STREAM_TIMING
    record_timing_start(streams.stream3, "cudaEventRecord final reset timing start");
#endif
    check_cuda(cudaMemsetAsync(
        memory.streams.clean_count,
        0,
        static_cast<std::uint64_t>(plan.storage_shard_count) * sizeof(std::uint32_t),
        streams.stream3), "cudaMemsetAsync reset clean count");
    check_cuda(cudaMemsetAsync(
        memory.streams.dirty_count,
        0,
        static_cast<std::uint64_t>(plan.storage_shard_count) * sizeof(std::uint32_t),
        streams.stream3), "cudaMemsetAsync reset dirty count");
    check_cuda(cudaMemsetAsync(
        memory.streams.processing_flag,
        0,
        static_cast<std::uint64_t>(plan.storage_shard_count) * sizeof(std::uint32_t),
        streams.stream3), "cudaMemsetAsync reset processing flag");
    check_cuda(cudaMemsetAsync(
        memory.streams.stream3_ready_flag,
        0,
        static_cast<std::uint64_t>(plan.storage_shard_count) * sizeof(std::uint32_t),
        streams.stream3), "cudaMemsetAsync reset stream3 ready flag");
    check_cuda(cudaMemsetAsync(
        memory.streams.stream3_ready_shard_list,
        0,
        static_cast<std::uint64_t>(plan.storage_shard_count) * sizeof(std::uint32_t),
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
        memory.streams.stream3_write_buffer_index,
        0,
        static_cast<std::uint64_t>(plan.config.shard_count) * sizeof(std::uint32_t),
        streams.stream3), "cudaMemsetAsync reset stream3 write buffer index");
    check_cuda(cudaMemsetAsync(
        memory.streams.shard_score_hist_a,
        0,
        static_cast<std::uint64_t>(plan.storage_shard_count) * SCORE_BIN_COUNT * sizeof(std::uint32_t),
        streams.stream3), "cudaMemsetAsync reset shard score hist a");
    check_cuda(cudaMemsetAsync(
        memory.streams.shard_score_hist_b,
        0,
        static_cast<std::uint64_t>(plan.storage_shard_count) * SCORE_BIN_COUNT * sizeof(std::uint32_t),
        streams.stream3), "cudaMemsetAsync reset shard score hist b");
    check_cuda(cudaMemsetAsync(
        memory.streams.shard_score_hist_active_index,
        0,
        static_cast<std::uint64_t>(plan.storage_shard_count) * sizeof(std::uint32_t),
        streams.stream3), "cudaMemsetAsync reset shard score hist active index");
    check_cuda(cudaMemsetAsync(memory.streams.current_threshold, 0xff, sizeof(std::uint32_t), streams.stream3),
        "cudaMemsetAsync reset current threshold");
    check_cuda(cudaMemsetAsync(memory.streams.threshold_initialized, 0, sizeof(std::uint32_t), streams.stream3),
        "cudaMemsetAsync reset threshold initialized");
#if BEAM_DEBUG_STREAM_TIMING
    record_timing_stop_and_sync(
        streams.stream3,
        result.stream3_reset_ms,
        "cudaEventRecord final reset timing stop",
        "cudaStreamSynchronize threshold reset",
        "cudaEventElapsedTime stream3 final reset");
#else
    check_cuda(cudaStreamSynchronize(streams.stream3), "cudaStreamSynchronize threshold reset");
#endif

    result.next_frontier_size = final_request_count;
    result.final_threshold = final_threshold;
    result.final_candidate_count = final_candidate_count;
    result.final_request_count = final_request_count;
    return result;
}

} // namespace beam
