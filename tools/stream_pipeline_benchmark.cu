#include "stream_benchmark_common.hpp"
#include "../cuda/dispatcher.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace beam;
using namespace beam::bench;

namespace {

std::uint32_t env_u32(const char* name, std::uint32_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return static_cast<std::uint32_t>(parse_u64(value, name));
}

std::uint64_t env_u64(const char* name, std::uint64_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return parse_u64(value, name);
}

std::filesystem::path env_path(const char* name, std::filesystem::path fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::filesystem::path(value);
}

DepthDispatchStopStage parse_stop_stage(const std::string& mode) {
    if (mode == "stream12") {
        return DepthDispatchStopStage::AfterStream12;
    }
    if (mode == "stream123") {
        return DepthDispatchStopStage::AfterStream3;
    }
    throw std::invalid_argument("BEAM_PIPELINE_BENCH_MODE must be stream12 or stream123");
}

std::string stop_stage_name(DepthDispatchStopStage stage) {
    switch (stage) {
    case DepthDispatchStopStage::AfterStream12:
        return "stream12";
    case DepthDispatchStopStage::AfterStream3:
        return "stream123";
    case DepthDispatchStopStage::Full:
        return "full";
    }
    return "unknown";
}


void reset_pipeline_memory(
    const StaticMemoryPlan& plan,
    StaticDeviceMemory& memory,
    const std::vector<State128>& states,
    const State128& central_state,
    State128* d_central_state) {
    BEAM_CUDA_CHECK(cudaMemset(memory.allocation, 0, memory.allocation_bytes));
    if (states.size() > plan.frontier_states) {
        throw std::runtime_error("pipeline smoke state batch exceeds current frontier capacity");
    }
    BEAM_CUDA_CHECK(cudaMemcpy(
        memory.current_frontier_states,
        states.data(),
        states.size() * sizeof(State128),
        cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_central_state, &central_state, sizeof(State128), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemset(memory.streams.current_threshold, 0xff, 2ULL * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(memory.streams.threshold_initialized, 0, 2ULL * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(memory.streams.current_threshold_active_index, 0, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(memory.streams.threshold_request_local, 0, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(memory.streams.threshold_request_global, 0, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 1 && argc != 2) {
        std::cerr << "usage: stream_pipeline_benchmark [puzzle_id]\n";
        return 2;
    }
    std::cout << std::unitbuf;
    const std::uint64_t puzzle_id = argc == 2 ? parse_u64(argv[1], "puzzle_id") : 990ULL;
    const std::uint32_t cuda_device = env_u32("BEAM_CUDA_DEVICE", 0U);
    BEAM_CUDA_CHECK(cudaSetDevice(static_cast<int>(cuda_device)));

    const std::string mode_env = [] {
        const char* value = std::getenv("BEAM_PIPELINE_BENCH_MODE");
        return value != nullptr && value[0] != '\0' ? std::string(value) : std::string("stream12");
    }();
    const DepthDispatchStopStage stop_stage = parse_stop_stage(mode_env);
    const std::string mode = stop_stage_name(stop_stage);

    std::size_t free_before = 0;
    std::size_t total_before = 0;
    BEAM_CUDA_CHECK(cudaMemGetInfo(&free_before, &total_before));

    const std::filesystem::path generator_path = env_path("BEAM_GENERATOR_PATH", "FullBeamNice/generators/p900.json");
    const std::filesystem::path puzzle_info_path = env_path("BEAM_PUZZLE_INFO_PATH", "data/puzzle_info.json");
    const std::filesystem::path test_csv_path = env_path("BEAM_TEST_CSV_PATH", "data/test.csv");
    const std::filesystem::path weight_dir = env_path("BEAM_WEIGHT_DIR", "weights/megaminx_vlad_transformer_fp16");

    const std::vector<std::uint8_t> host_generators = load_p900_generators(generator_path);
    const State128 host_central = load_central_state(puzzle_info_path);
    const State128 host_initial = load_initial_state_from_test_csv(test_csv_path, puzzle_id);
    const ZobristTable host_zobrist = make_deterministic_zobrist(0xC0DEC0DEULL);
    const stream1_weights::HostWeightBytes host_weights = stream1_weights::load_stream1_weights(weight_dir);
    const Stream1ModelConfig& stream1_model = host_weights.model;
    if (stream1_model.backend != STREAM1_BACKEND_PIECE_TRANSFORMER) {
        throw std::runtime_error("stream_pipeline_benchmark requires piece_transformer Stream1 weights");
    }

    RuntimeConfig config;
    config.b_micro = env_u32("BEAM_B_MICRO", env_u32("BEAM_PIPELINE_B_MICRO", 512U));
    config.inference_parallelism = env_u32("BEAM_STREAM1_CONCURRENCY", 2U);
    const std::uint32_t ring_slots = env_u32("BEAM_STREAM3_RING_SLOTS", 8U);
    config.stream3_batch_candidates = config.b_micro * static_cast<std::uint32_t>(MOVE_COUNT) * ring_slots;
    config.stream4_batch_candidates = env_u32("STREAM4_BATCH_CANDIDATES", env_u32("BEAM_STREAM4_BATCH_CANDIDATES", 262144U));
    config.stream4_trigger_candidates = env_u32("STREAM4_TRIGGER_CANDIDATES", env_u32("BEAM_STREAM4_TRIGGER_CANDIDATES", 524288U));
    config.stream4_batch_alignment = env_u32("STREAM4_BATCH_ALIGNMENT", env_u32("BEAM_STREAM4_BATCH_ALIGNMENT", 1024U));
    config.stream4_active_sort_slots = env_u32("BEAM_STREAM4_ACTIVE_SORT_SLOTS", 4U);
    config.ring_count = env_u32("BEAM_PIPELINE_SMOKE_RINGS", env_u32("BEAM_RING_COUNT", 32U));
    config.world_size = 1U;
    config.local_rank = 0U;
    config.shard_count = env_u32("BEAM_SHARD_COUNT", 32U);
    config.shard_buffer_count = env_u32("BEAM_SHARD_BUFFER_COUNT", 2U);
    config.global_spill_capacity = env_u32("BEAM_GLOBAL_SPILL_CAPACITY", 1048576U);
    config.final_materialize_chunk_candidates = env_u32("BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES", 98304U);
    config.final_materialize_exchange_scale_ppm = env_u32("BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM", 2000000U);
    config.solved_result_capacity = 0U;

    const std::uint64_t default_frontier =
        static_cast<std::uint64_t>(config.ring_count) * ring_slots * config.b_micro;
    const std::uint64_t frontier_size = env_u64("BEAM_PIPELINE_SMOKE_FRONTIER", default_frontier);
    if (frontier_size == 0U) {
        throw std::runtime_error("BEAM_PIPELINE_SMOKE_FRONTIER must be positive");
    }
    if (frontier_size > UINT32_MAX) {
        throw std::runtime_error("BEAM_PIPELINE_SMOKE_FRONTIER exceeds benchmark host-state batch limit");
    }
    const std::uint64_t generated_candidates = frontier_size * static_cast<std::uint64_t>(MOVE_COUNT);
    const std::uint64_t avg_shard_candidates =
        (generated_candidates + std::max(1U, config.shard_count) - 1ULL) / std::max(1U, config.shard_count);
    const std::uint64_t auto_shard_capacity = std::max<std::uint64_t>({
        1048576ULL,
        config.stream4_batch_candidates,
        config.stream3_batch_candidates,
        avg_shard_candidates * 4ULL});
    config.shard_capacity_candidates = env_u32(
        "BEAM_SHARD_CAPACITY_CANDIDATES",
        static_cast<std::uint32_t>(std::min<std::uint64_t>(auto_shard_capacity, UINT32_MAX)));
    config.user_global_beam_width = std::max<std::uint64_t>(
        frontier_size,
        static_cast<std::uint64_t>(config.shard_count) * config.stream4_batch_alignment);

    const StaticMemoryPlan plan = make_static_memory_plan(config);
    if (frontier_size > plan.frontier_states) {
        throw std::runtime_error("pipeline smoke frontier exceeds static memory frontier capacity");
    }

    const bool synthetic_states = std::getenv("BEAM_STREAM1_SYNTHETIC_STATES") != nullptr;
    const std::vector<State128> host_states = synthetic_states
        ? make_synthetic_state_batch(static_cast<std::uint32_t>(frontier_size), stream1_model.state_len, stream1_model.num_classes)
        : make_state_batch(host_initial, static_cast<std::uint32_t>(frontier_size), stream1_model.num_classes);

    StaticDeviceMemory memory;
    allocate_static_device_memory(plan, memory);
    std::uint8_t* d_generators = device_alloc<std::uint8_t>(MOVE_COUNT * STATE_STORAGE_LEN);
    State128* d_central = device_alloc<State128>(1);
    Hash128* d_zobrist = device_alloc<Hash128>(STATE_STORAGE_LEN * STATE_VALUE_PAD);
    BEAM_CUDA_CHECK(cudaMemcpy(d_generators, host_generators.data(), host_generators.size(), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_zobrist, &host_zobrist[0][0], STATE_STORAGE_LEN * STATE_VALUE_PAD * sizeof(Hash128), cudaMemcpyHostToDevice));
    reset_pipeline_memory(plan, memory, host_states, host_central, d_central);

    stream1_weights::DeviceWeights device_weights = stream1_weights::upload_weights(host_weights);
    stream1_weights::ScratchAllocation stream1_scratch =
        stream1_weights::alloc_stream1_scratch(stream1_model, config.b_micro, config.inference_parallelism);
    stream1_weights::TransformerNetworkViewHolder transformer_view_holder =
        stream1_weights::transformer_network_view(device_weights.transformer, stream1_model);

    DispatcherNetwork network{};
    network.backend = DispatcherStream1Backend::PieceTransformer;
    network.transformer_view = transformer_view_holder.view;
    network.transformer_scratch_lanes.reserve(config.inference_parallelism);
    for (std::uint32_t lane = 0; lane < config.inference_parallelism; ++lane) {
        network.transformer_scratch_lanes.push_back(
            stream1_weights::transformer_scratch_view(stream1_scratch, stream1_model, config.b_micro, lane));
    }

    DispatcherStreams streams;
    DispatcherEvents events;
    CudaGraphJobTemplates graphs;
    create_dispatcher_streams(streams);
    create_dispatcher_events(events);
    DispatcherDeviceTables tables{d_generators, d_central, d_zobrist};
    Stream2SolvedBuffers solved{
        memory.solved_flag,
        memory.stop_flag,
        memory.solved_count,
        memory.solved_overflow,
        memory.solved_meta_list,
        memory.solved_depth_list,
        config.solved_result_capacity,
        memory.current_depth,
        {},
        {},
        memory.solved_suffix_list,
        0U};

    instantiate_cuda_graph_job_templates(plan, memory, tables, network, solved, streams, events, graphs);

    const auto host_start = std::chrono::steady_clock::now();
    const DepthDispatchState state = run_depth_cuda_graphs(
        plan,
        memory,
        graphs,
        streams,
        frontier_size,
        {},
        nullptr,
        nullptr,
        stop_stage);
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    const auto host_stop = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(host_stop - host_start).count();
    const std::uint64_t measured_candidates =
        static_cast<std::uint64_t>(state.ring_slot_jobs_launched) * config.b_micro * MOVE_COUNT;
    const double candidates_per_sec = elapsed_ms > 0.0
        ? static_cast<double>(measured_candidates) * 1000.0 / elapsed_ms
        : 0.0;

    const std::uint32_t graph_window_jobs = graphs.ring_slot_window_jobs;
    const std::uint32_t physical_jobs = graphs.ring_slot_physical_jobs;
    const std::uint32_t requested_window = env_u32("BEAM_RING_GRAPH_EXECS_PER_LANE", graph_window_jobs);
    std::cout << "stream_pipeline_benchmark"
              << " mode=" << mode
              << " window=" << requested_window
              << " b_micro=" << config.b_micro
              << " concurrency=" << config.inference_parallelism
              << " ring_slots=" << plan.derived.ring_slot_count
              << " stream3_batch=" << config.stream3_batch_candidates
              << " graph_window_jobs=" << graph_window_jobs
              << " physical_jobs=" << physical_jobs
              << " frontier_size=" << frontier_size
              << " ring_slot_jobs=" << state.ring_slot_jobs_launched
              << " stream3_jobs=" << state.stream3_jobs_launched
              << " stream4_jobs=" << state.stream4_jobs_launched
              << " candidates=" << measured_candidates
              << " depth_like_ms=" << elapsed_ms
              << " candidates_per_sec=" << candidates_per_sec
              << " shard_capacity=" << config.shard_capacity_candidates
              << " allocation_bytes=" << memory.allocation_bytes
              << " status=OK\n";

    if (const char* report_env = std::getenv("BEAM_PIPELINE_BENCH_REPORT"); report_env != nullptr && report_env[0] != '\0') {
        std::ofstream report(report_env, std::ios::app);
        report << "mode=" << mode
               << " window=" << requested_window
               << " b_micro=" << config.b_micro
               << " concurrency=" << config.inference_parallelism
               << " ring_slots=" << plan.derived.ring_slot_count
               << " stream3_batch=" << config.stream3_batch_candidates
               << " physical_jobs=" << physical_jobs
               << " candidates_per_sec=" << candidates_per_sec
               << " depth_like_ms=" << elapsed_ms
               << " status=OK\n";
    }

    destroy_cuda_graph_job_templates(graphs);
    destroy_dispatcher_events(events);
    destroy_dispatcher_streams(streams);
    stream1_weights::free_transformer_network_view(transformer_view_holder);
    stream1_weights::free_stream1_scratch(stream1_scratch);
    stream1_weights::free_weights(device_weights);
    free_static_device_memory(memory);
    cudaFree(d_generators);
    cudaFree(d_central);
    cudaFree(d_zobrist);
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    (void)total_before;
    (void)free_before;
    return 0;
}