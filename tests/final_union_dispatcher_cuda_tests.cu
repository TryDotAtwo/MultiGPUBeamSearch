#include "../cuda/dispatcher.hpp"
#include "../cuda/stream3.hpp"
#include "../cuda/stream4.hpp"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace beam;
static void cu(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void nc(ncclResult_t e) { if (e != ncclSuccess) throw std::runtime_error(ncclGetErrorString(e)); }
static int env(const char* key) {
    const char* s = std::getenv(key);
    if (!s) throw std::runtime_error(std::string("missing ") + key);
    return std::stoi(s);
}

// Force a legal busy-sibling schedule through the real Stream3 collector.
// Arrival1 contains x,y; arrival2 repeats x while physical A is unavailable.
static void collect_two_arrivals(const StaticMemoryPlan& plan, StaticDeviceMemory& memory,
                                 DispatcherStreams& streams, const CandidateMeta* input) {
    auto& s = memory.streams;
    const auto& c = plan.config;
    auto collect = [&](unsigned count) {
        const std::uint32_t counts[2]{count, 0}, offsets[3]{0, count, count};
        cu(cudaMemcpy(s.remote_recv_buffer, input, count * sizeof(CandidateMeta), cudaMemcpyHostToDevice));
        cu(cudaMemcpy(s.recv_count, counts, sizeof(counts), cudaMemcpyHostToDevice));
        cu(cudaMemcpy(s.recv_offset, offsets, sizeof(offsets), cudaMemcpyHostToDevice));
        stream3_collect_remote_recv_cuda(s.remote_recv_buffer, s.recv_count, s.recv_offset,
            s.survivor_shard, s.clean_count, s.dirty_count, s.processing_flag,
            s.global_spill_buffer_a, s.global_spill_buffer_b, s.global_spill_count, s.global_spill_active_index,
            s.stream3_write_buffer_index, s.stream3_shard_counts, s.stream3_shard_offsets,
            s.stream3_spill_counts, s.stream3_spill_offsets, s.stream3_partition_key_a, s.stream3_partition_key_b,
            s.stream3_partition_val_a, s.stream3_partition_val_b, s.stream3_partition_unique_shard,
            s.stream3_partition_unique_counts, s.stream3_partition_unique_count,
            s.stream3_cub_temp, s.stream3_cub_temp_bytes, count, c.world_size, c.shard_count,
            c.shard_buffer_count, c.shard_capacity_candidates, c.stream4_batch_candidates,
            c.global_spill_capacity, streams.stream3, s.fatal_error_flag, s.fatal_error_trace);
        cu(cudaStreamSynchronize(streams.stream3));
        std::uint32_t fatal = 0;
        cu(cudaMemcpy(&fatal, s.fatal_error_flag, sizeof(fatal), cudaMemcpyDeviceToHost));
        if (fatal) throw std::runtime_error("Stream3 controlled arrival failed: " + std::to_string(fatal));
    };
    auto finish = [&](unsigned physical) {
        stream4_shard_job_cuda(s.survivor_shard + physical * c.shard_capacity_candidates,
            s.clean_count + physical, s.dirty_count + physical, s.processing_flag + physical,
            UINT32_THRESHOLD_MAX, c.shard_capacity_candidates, s.stream4_key_a, s.stream4_key_b,
            s.stream4_val_a, s.stream4_val_b, s.stream4_score_key_a, s.stream4_score_key_b,
            s.stream4_score_count_a, s.stream4_score_count_b, s.stream4_keep_flags,
            s.stream4_block_counts, s.stream4_block_offsets, s.stream4_count,
            s.shard_score_hist_a + physical * SCORE_BIN_COUNT, s.shard_score_hist_b + physical * SCORE_BIN_COUNT,
            s.shard_score_hist_active_index + physical, s.stream4_cub_temp, s.stream4_cub_temp_bytes, streams.stream4);
        cu(cudaStreamSynchronize(streams.stream4));
    };
    collect(2); finish(0);
    const std::uint32_t busy[2]{1, 0}, idle[2]{0, 0};
    cu(cudaMemcpy(s.processing_flag, busy, sizeof(busy), cudaMemcpyHostToDevice));
    collect(1);
    cu(cudaMemcpy(s.processing_flag, idle, sizeof(idle), cudaMemcpyHostToDevice));
    finish(1);
    std::uint32_t clean[2]{};
    cu(cudaMemcpy(clean, s.clean_count, sizeof(clean), cudaMemcpyDeviceToHost));
    if (clean[0] != 2 || clean[1] != 1) throw std::runtime_error("Stream3 did not reproduce cross-physical duplicates");
}

// Real final selection, NCCL materialization and history, with synthetic
// prefinal candidates. This is not an inference or Stream3 scheduling test.
int main(int argc, char** argv) {
    try {
        if ((argc != 2 && argc != 3) || env("WORLD_SIZE") != 2) throw std::runtime_error("requires torchrun 2 ranks and rendezvous path");
        const bool collect_mode = argc == 3 && std::string(argv[2]) == "collect";
        const bool empty_rank_mode = argc == 3 && std::string(argv[2]) == "pipeline_empty_rank";
        const bool pipeline_mode = argc == 3 && (std::string(argv[2]) == "pipeline" || empty_rank_mode);
        if (argc == 3 && !collect_mode && !pipeline_mode) throw std::runtime_error("unknown fixture mode");
        const int rank = env("RANK"), device = env("LOCAL_RANK");
        const std::uint64_t local_parent_count = empty_rank_mode && rank == 1 ? 0U : 2U;
        if (rank != device || rank < 0 || rank > 1) throw std::runtime_error("requires distinct local devices 0 and 1");
        cu(cudaSetDevice(device));
        ncclUniqueId id{};
        const std::filesystem::path rendezvous(argv[1]);
        if (rank == 0) {
            nc(ncclGetUniqueId(&id));
            auto temp = rendezvous.string() + ".tmp";
            { std::ofstream f(temp, std::ios::binary); f.write(reinterpret_cast<const char*>(&id), sizeof(id));
              if (!f) throw std::runtime_error("NCCL rendezvous write failed"); }
            std::filesystem::rename(temp, rendezvous);
        } else {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            while (!std::filesystem::exists(rendezvous)) {
                if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("NCCL rendezvous timeout");
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            std::ifstream f(rendezvous, std::ios::binary);
            f.read(reinterpret_cast<char*>(&id), sizeof(id));
            if (!f) throw std::runtime_error("NCCL rendezvous read failed");
        }
        DispatcherCollective collective{};
        nc(ncclCommInitRank(&collective.comm, 2, id, rank));
        RuntimeConfig config;
        config.world_size = 2; config.local_rank = rank;
        config.b_micro = 1; config.ring_count = 1;
        config.stream3_batch_candidates = 2 * MOVE_COUNT;
        config.stream4_batch_candidates = 4; config.stream4_batch_alignment = 1;
        config.shard_count = 1; config.shard_buffer_count = 2;
        config.shard_capacity_candidates = 256; config.user_global_beam_width = 4;
        config.global_spill_capacity = 0; config.final_materialize_chunk_candidates = 2;
        if (pipeline_mode) {
            config.ring_count = 2;
            config.stream3_batch_candidates = MOVE_COUNT;
            config.stream4_trigger_candidates = 1;
        }
        const auto plan = make_static_memory_plan(config);
        if (plan.derived.global_beam_width_effective != 4) throw std::runtime_error("fixture beam rounded unexpectedly");
        StaticDeviceMemory memory;
        allocate_static_device_memory(plan, memory);
        cu(cudaMemset(memory.allocation, 0, memory.allocation_bytes));
        DispatcherStreams streams{};
        create_dispatcher_streams(streams);
        std::vector<State128> parents(2);
        for (int i = 0; i < 2; ++i) parents[i].v[0] = 2 * rank + i + 1;
        cu(cudaMemcpy(memory.current_frontier_states, parents.data(), 2 * sizeof(State128), cudaMemcpyHostToDevice));
        std::vector<std::uint8_t> generators(MOVE_COUNT * STATE_STORAGE_LEN);
        for (unsigned m = 0; m < MOVE_COUNT; ++m)
            for (unsigned p = 0; p < STATE_STORAGE_LEN; ++p) generators[m * STATE_STORAGE_LEN + p] = p;
        std::uint8_t* device_generators = nullptr;
        cu(cudaMalloc(&device_generators, generators.size()));
        cu(cudaMemcpy(device_generators, generators.data(), generators.size(), cudaMemcpyHostToDevice));
        DispatcherDeviceTables tables{}; tables.generators = device_generators;
        std::vector<CandidateMeta> candidates(512);
        candidates[0] = CandidateMeta{Hash128{static_cast<std::uint64_t>(2 * rank + 1), 0}, 0, 1, pack_route(rank, rank, 0)};
        candidates[1] = CandidateMeta{Hash128{static_cast<std::uint64_t>(2 * rank + 2), 0}, 1, 2, pack_route(rank, rank, 0)};
        candidates[256] = candidates[0];
        if (!collect_mode && !pipeline_mode) {
        cu(cudaMemcpy(memory.streams.survivor_shard, candidates.data(), candidates.size() * sizeof(CandidateMeta), cudaMemcpyHostToDevice));
        const std::uint32_t counts[2]{2, 1};
        cu(cudaMemcpy(memory.streams.clean_count, counts, sizeof(counts), cudaMemcpyHostToDevice));
        std::vector<std::uint32_t> hist(2ULL * SCORE_BIN_COUNT);
        hist[1] = hist[2] = hist[SCORE_BIN_COUNT + 1] = 1;
        cu(cudaMemcpy(memory.streams.shard_score_hist_a, hist.data(), hist.size() * sizeof(hist[0]), cudaMemcpyHostToDevice));
        } else if (collect_mode) {
            collect_two_arrivals(plan, memory, streams, candidates.data());
        }
        const std::uint32_t thresholds[2]{UINT32_THRESHOLD_MAX, UINT32_THRESHOLD_MAX};
        cu(cudaMemcpy(memory.streams.current_threshold, thresholds, sizeof(thresholds), cudaMemcpyHostToDevice));
        State128* central = nullptr;
        Hash128* zobrist_device = nullptr;
        DispatcherEvents events{};
        CudaGraphJobTemplates graphs{};
        std::vector<std::uint32_t> final_clean_counts;
        if (pipeline_mode) {
            cu(cudaMalloc(&central, sizeof(State128)));
            cu(cudaMemset(central, 0, sizeof(State128)));
            std::vector<Hash128> zobrist(STATE_STORAGE_LEN * STATE_VALUE_PAD);
            for (unsigned value = 1; value <= 4; ++value) zobrist[value].lo = value;
            cu(cudaMalloc(&zobrist_device, zobrist.size() * sizeof(Hash128)));
            cu(cudaMemcpy(zobrist_device, zobrist.data(), zobrist.size() * sizeof(Hash128), cudaMemcpyHostToDevice));
            tables.central_state = central; tables.zobrist = zobrist_device;
            DispatcherNetwork network{}; network.uniform_score = true;
            Stream2SolvedBuffers solved{memory.solved_flag, memory.stop_flag, memory.solved_count,
                memory.solved_overflow, memory.solved_meta_list, memory.solved_depth_list,
                config.solved_result_capacity};
            create_dispatcher_events(events);
            instantiate_cuda_graph_job_templates(plan, memory, tables, network, solved, streams, events, graphs);
            const auto depth = run_depth_cuda_graphs(plan, memory, graphs, streams, local_parent_count, {}, &collective);
            final_clean_counts = depth.final_clean_counts;
            std::cout << "rank=" << rank << " drain_observation=" << depth.depth_drained
                      << " stop=" << depth.stop_requested << " cursor=" << depth.frontier_cursor
                      << " local_parents=" << local_parent_count
                      << " stream3_jobs=" << depth.stream3_jobs_launched
                      << " stream4_jobs=" << depth.stream4_jobs_launched << std::endl;
            if (final_clean_counts.size() != plan.storage_shard_count)
                throw std::runtime_error("pipeline omitted final clean-count snapshot");
            if (!depth.depth_drained || depth.stop_requested || depth.frontier_cursor != local_parent_count ||
                depth.stream3_jobs_launched != local_parent_count ||
                (local_parent_count != 0 && depth.stream4_jobs_launched == 0))
                throw std::runtime_error("uniform-score pipeline did not fully drain expected work");
            std::cout << "rank=" << rank << " full_dispatch_drained=1 threshold_updates="
                      << depth.threshold_updates << std::endl;
        }
        CandidateMeta* history = nullptr;
        cu(cudaMallocHost(&history, 2 * sizeof(CandidateMeta)));
        cudaStream_t history_stream; cudaEvent_t history_done;
        cu(cudaStreamCreate(&history_stream)); cu(cudaEventCreateWithFlags(&history_done, cudaEventDisableTiming));
        cu(cudaDeviceSynchronize());
        // Host validation must reject malformed snapshots before enqueueing
        // work or entering any finalization collective, on both ranks.
        for (const auto& bad : {std::vector<std::uint32_t>{0},
                               std::vector<std::uint32_t>{257, 0}}) {
            bool rejected = false;
            try {
                finalize_depth_single_gpu(plan, memory, tables, streams, local_parent_count,
                    history, 2, history_stream, history_done, nullptr, &collective, &bad);
            } catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("invalid final snapshot was accepted");
        }
        const auto result = finalize_depth_single_gpu(plan, memory, tables, streams, local_parent_count,
            history, 2, history_stream, history_done, nullptr, &collective,
            pipeline_mode ? &final_clean_counts : nullptr);
        cu(cudaDeviceSynchronize());
        std::vector<State128> output(result.next_frontier_size);
        cu(cudaMemcpy(output.data(), memory.current_frontier_states, output.size() * sizeof(State128), cudaMemcpyDeviceToHost));
        int hits[4]{}, global_hits[4]{};
        bool valid = result.next_frontier_size == (empty_rank_mode ? 1U : 2U) &&
            result.final_threshold == (pipeline_mode && !empty_rank_mode ? 0U : empty_rank_mode ? UINT32_THRESHOLD_MAX : 2U);
        for (std::size_t i = 0; i < output.size(); ++i) {
            unsigned value = output[i].v[0];
            if (value < 1 || value > 4) valid = false; else ++hits[value - 1];
            for (unsigned p = 1; p < STATE_STORAGE_LEN; ++p) if (output[i].v[p] != 0) valid = false;
            if (i < 2 && value != 2 * unpack_source_rank(history[i].route_packed) + history[i].parent_idx + 1) valid = false;
            if (i < 2 && (history[i].hash.lo != value || history[i].hash.hi != 0 ||
                history[i].score_key != (pipeline_mode ? 0U : (value - 1) % 2 + 1) || unpack_move(history[i].route_packed) != 0)) valid = false;
        }
        int *send = nullptr, *recv = nullptr;
        cu(cudaMalloc(&send, sizeof(hits))); cu(cudaMalloc(&recv, sizeof(hits)));
        cu(cudaMemcpy(send, hits, sizeof(hits), cudaMemcpyHostToDevice));
        nc(ncclAllReduce(send, recv, 4, ncclInt, ncclSum, collective.comm, streams.stream5));
        cu(cudaStreamSynchronize(streams.stream5));
        cu(cudaMemcpy(global_hits, recv, sizeof(global_hits), cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < 4; ++i)
            if (global_hits[i] != (empty_rank_mode && i >= 2 ? 0 : 1)) valid = false;
        std::cout << "rank=" << rank << " final_threshold=" << result.final_threshold << " global_hits="
                  << global_hits[0] << ',' << global_hits[1] << ',' << global_hits[2] << ',' << global_hits[3]
                  << " dispatcher_union=" << (valid ? "PASS" : "FAIL")
                  << " input_mode=" << (empty_rank_mode ? "uniform_pipeline_empty_rank" : pipeline_mode ? "uniform_pipeline" : collect_mode ? "stream3_arrivals" : "prefinal") << std::endl;
        cu(cudaFree(send)); cu(cudaFree(recv)); cu(cudaFreeHost(history));
        cu(cudaEventDestroy(history_done)); cu(cudaStreamDestroy(history_stream));
        if (pipeline_mode) { destroy_cuda_graph_job_templates(graphs); destroy_dispatcher_events(events); }
        destroy_dispatcher_streams(streams); free_static_device_memory(memory);
        cu(cudaFree(central)); cu(cudaFree(zobrist_device));
        cu(cudaFree(device_generators)); nc(ncclCommDestroy(collective.comm));
        return valid ? 0 : 1;
    } catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << std::endl; return 2; }
}
