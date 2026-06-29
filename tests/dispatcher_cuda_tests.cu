#include "cuda_check.hpp"
#include "../cuda/dispatcher.hpp"
#include "../cuda/runtime_config.hpp"
#include "../tools/stream1_weight_io.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace beam;

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void set_test_env(const char* name, const char* value) {
#if defined(_WIN32)
    if (_putenv_s(name, value) != 0) {
        throw std::runtime_error(std::string("failed to set environment variable ") + name);
    }
#else
    if (setenv(name, value, 1) != 0) {
        throw std::runtime_error(std::string("failed to set environment variable ") + name);
    }
#endif
}

struct ScopedTestEnv {
    const char* name;
    bool had_previous = false;
    std::string previous_value;

    ScopedTestEnv(const char* env_name, const char* value) : name(env_name) {
        if (const char* previous = std::getenv(name)) {
            had_previous = true;
            previous_value = previous;
        }
        set_test_env(name, value);
    }

    ~ScopedTestEnv() {
#if defined(_WIN32)
        _putenv_s(name, had_previous ? previous_value.c_str() : "");
#else
        if (had_previous) {
            setenv(name, previous_value.c_str(), 1);
        } else {
            unsetenv(name);
        }
#endif
    }

    ScopedTestEnv(const ScopedTestEnv&) = delete;
    ScopedTestEnv& operator=(const ScopedTestEnv&) = delete;
};

template <typename T>
T* cuda_alloc_count(std::uint64_t count, const char* name) {
    T* ptr = nullptr;
    cudaError_t status = cudaMalloc(&ptr, count * sizeof(T));
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed for ") + name + ": " + cudaGetErrorString(status));
    }
    BEAM_CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(T)));
    return ptr;
}

struct TransformerDispatcherFixture {
    std::vector<half*> half_allocations;
    std::uint16_t* piece_positions = nullptr;
    std::uint8_t* piece_mask = nullptr;
    std::uint8_t* piece_types = nullptr;
    std::vector<Stream1TransformerBlockView> blocks;
    Stream1TransformerNetworkView view{};
    Stream1TransformerScratchView scratch{};
    stream1_weights::ScratchAllocation scratch_allocation{};

    half* alloc_half(std::uint64_t count, const char* name) {
        half* ptr = cuda_alloc_count<half>(count, name);
        half_allocations.push_back(ptr);
        return ptr;
    }

    void destroy() {
        for (half* ptr : half_allocations) {
            cudaFree(ptr);
        }
        half_allocations.clear();
        cudaFree(piece_positions);
        cudaFree(piece_mask);
        cudaFree(piece_types);
        piece_positions = nullptr;
        piece_mask = nullptr;
        piece_types = nullptr;
        stream1_weights::free_stream1_scratch(scratch_allocation);
        view = {};
        scratch = {};
        blocks.clear();
    }
};

Stream1ModelConfig tiny_transformer_model() {
    Stream1ModelConfig model;
    model.backend = STREAM1_BACKEND_PIECE_TRANSFORMER;
    model.state_len = STATE_LEN;
    model.num_classes = stream1_weights::TRANSFORMER_NUM_CLASSES;
    model.output_dim = static_cast<std::uint32_t>(MOVE_COUNT);
    model.num_pieces = stream1_weights::TRANSFORMER_NUM_PIECES;
    model.max_piece_size = stream1_weights::TRANSFORMER_MAX_PIECE_SIZE;
    model.seq_len = stream1_weights::TRANSFORMER_SEQ_LEN;
    model.d_model = stream1_weights::TRANSFORMER_D_MODEL;
    model.nhead = stream1_weights::TRANSFORMER_NHEAD;
    model.head_dim = stream1_weights::TRANSFORMER_HEAD_DIM;
    model.transformer_layers = stream1_weights::TRANSFORMER_LAYERS;
    model.ff_dim = stream1_weights::TRANSFORMER_FF_DIM;
    model.dtype = STREAM1_DTYPE_FP16;
    return model;
}

TransformerDispatcherFixture make_transformer_dispatcher_fixture(
    const Stream1ModelConfig& model,
    std::uint32_t b_micro) {
    TransformerDispatcherFixture fixture;
    fixture.blocks.resize(model.transformer_layers);
    fixture.view.fast_slot_projected =
        fixture.alloc_half(static_cast<std::uint64_t>(model.max_piece_size) * model.num_classes * model.d_model, "fast_slot_projected");
    fixture.view.fast_piece_static =
        fixture.alloc_half(static_cast<std::uint64_t>(model.num_pieces) * model.d_model, "fast_piece_static");
    fixture.view.cls_token = fixture.alloc_half(model.d_model, "cls_token");
    fixture.view.input_ln_gamma = fixture.alloc_half(model.d_model, "input_ln_gamma");
    fixture.view.input_ln_beta = fixture.alloc_half(model.d_model, "input_ln_beta");
    fixture.view.output_ln_gamma = fixture.alloc_half(model.d_model, "output_ln_gamma");
    fixture.view.output_ln_beta = fixture.alloc_half(model.d_model, "output_ln_beta");
    for (std::uint32_t layer = 0; layer < model.transformer_layers; ++layer) {
        fixture.blocks[layer] = Stream1TransformerBlockView{
            fixture.alloc_half(model.d_model, "ln1_gamma"),
            fixture.alloc_half(model.d_model, "ln1_beta"),
            fixture.alloc_half(static_cast<std::uint64_t>(model.d_model) * 3ULL * model.d_model, "attn_qkv_weight"),
            fixture.alloc_half(3ULL * model.d_model, "attn_qkv_bias"),
            fixture.alloc_half(static_cast<std::uint64_t>(model.d_model) * model.d_model, "attn_out_weight"),
            fixture.alloc_half(model.d_model, "attn_out_bias"),
            fixture.alloc_half(model.d_model, "ln2_gamma"),
            fixture.alloc_half(model.d_model, "ln2_beta"),
            fixture.alloc_half(static_cast<std::uint64_t>(model.d_model) * model.ff_dim, "ff1_weight"),
            fixture.alloc_half(model.ff_dim, "ff1_bias"),
            fixture.alloc_half(static_cast<std::uint64_t>(model.ff_dim) * model.d_model, "ff2_weight"),
            fixture.alloc_half(model.d_model, "ff2_bias")};
    }
    fixture.view.blocks = fixture.blocks.data();
    fixture.view.output_weight =
        fixture.alloc_half(static_cast<std::uint64_t>(model.d_model) * model.output_dim, "output_weight");
    fixture.view.output_bias = fixture.alloc_half(model.output_dim, "output_bias");
    fixture.piece_positions = cuda_alloc_count<std::uint16_t>(
        static_cast<std::uint64_t>(model.num_pieces) * model.max_piece_size,
        "piece_positions");
    fixture.piece_mask = cuda_alloc_count<std::uint8_t>(
        static_cast<std::uint64_t>(model.num_pieces) * model.max_piece_size,
        "piece_mask");
    fixture.piece_types = cuda_alloc_count<std::uint8_t>(
        model.num_pieces,
        "piece_types");
    fixture.view.piece_positions = fixture.piece_positions;
    fixture.view.piece_mask = fixture.piece_mask;
    fixture.view.piece_types = fixture.piece_types;
    fixture.view.dims = stream1_weights::transformer_dims(model);
    fixture.scratch_allocation = stream1_weights::alloc_stream1_scratch(model, b_micro, 1);
    fixture.scratch = stream1_weights::transformer_scratch_view(fixture.scratch_allocation);
    return fixture;
}

void run_transformer_runtime_weight_estimate_test(std::ofstream& report) {
    const Stream1ModelConfig model = tiny_transformer_model();
    Stream1ModelConfig larger_state_model = model;
    larger_state_model.state_len += 7;
    RuntimeConfigBuild runtime_estimate;
    RuntimeConfigBuild larger_state_estimate;
    {
        const ScopedTestEnv runtime_mode("BEAM_RUNTIME_CONFIG_MODE", "manual");
        const ScopedTestEnv gpu_headroom("BEAM_GPU_HEADROOM_BYTES", "0");
        const ScopedTestEnv stream3_ring_slots("BEAM_STREAM3_RING_SLOTS", "1");
        const ScopedTestEnv shard_count("BEAM_SHARD_COUNT", "1");
        const ScopedTestEnv shard_buffer_count("BEAM_SHARD_BUFFER_COUNT", "2");
        const ScopedTestEnv shard_capacity("BEAM_SHARD_CAPACITY_CANDIDATES", "262144");
        const ScopedTestEnv stream4_batch("BEAM_STREAM4_BATCH_CANDIDATES", "1024");
        const ScopedTestEnv stream4_trigger("BEAM_STREAM4_TRIGGER_CANDIDATES", "1024");
        const ScopedTestEnv stream4_alignment("BEAM_STREAM4_BATCH_ALIGNMENT", "1");
        const ScopedTestEnv ring_count("BEAM_RING_COUNT", "1");
        runtime_estimate =
            build_runtime_config_from_budget(1024, 1, 0, model, 1024ULL * 1024ULL * 1024ULL * 1024ULL);
        larger_state_estimate =
            build_runtime_config_from_budget(1024, 1, 0, larger_state_model, 1024ULL * 1024ULL * 1024ULL * 1024ULL);
    }
    require(
        runtime_estimate.estimated_non_static_device_bytes ==
            larger_state_estimate.estimated_non_static_device_bytes,
        "piece_transformer runtime weight estimate must size fast_slot_projected by max_piece_size, not state_len");
    report << "- transformer_runtime_weight_estimate=pass\n";
}
void run_transformer_dispatcher_graph_test(std::ofstream& report) {
    const Stream1ModelConfig model = tiny_transformer_model();

    RuntimeConfig config;
    config.b_micro = 4;
    config.stream3_batch_candidates = 4 * static_cast<std::uint32_t>(MOVE_COUNT);
    config.stream4_batch_candidates = 64;
    config.stream4_trigger_candidates = 64;
    config.stream4_active_sort_slots = 1;
    config.ring_count = 1;
    config.shard_count = 1;
    config.shard_capacity_candidates = 128;
    config.global_spill_capacity = 128;
    config.stream4_batch_alignment = 1;
    config.user_global_beam_width = 64;
    config.solved_result_capacity = 8;
    const StaticMemoryPlan plan = make_static_memory_plan(config);
    StaticDeviceMemory memory;
    allocate_static_device_memory(plan, memory);
    BEAM_CUDA_CHECK(cudaMemset(memory.allocation, 0, memory.allocation_bytes));

    std::uint8_t* generators = nullptr;
    State128* central_state = nullptr;
    Hash128* zobrist = nullptr;
    BEAM_CUDA_CHECK(cudaMalloc(&generators, MOVE_COUNT * STATE_STORAGE_LEN));
    BEAM_CUDA_CHECK(cudaMalloc(&central_state, sizeof(State128)));
    BEAM_CUDA_CHECK(cudaMalloc(&zobrist, STATE_STORAGE_LEN * STATE_VALUE_PAD * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMemset(generators, 0, MOVE_COUNT * STATE_STORAGE_LEN));
    BEAM_CUDA_CHECK(cudaMemset(central_state, 255, sizeof(State128)));
    BEAM_CUDA_CHECK(cudaMemset(zobrist, 0, STATE_STORAGE_LEN * STATE_VALUE_PAD * sizeof(Hash128)));

    TransformerDispatcherFixture fixture = make_transformer_dispatcher_fixture(model, config.b_micro);
    DispatcherStreams streams;
    DispatcherEvents events;
    CudaGraphJobTemplates graphs;
    create_dispatcher_streams(streams);
    create_dispatcher_events(events);
    DispatcherNetwork network{};
    network.backend = DispatcherStream1Backend::PieceTransformer;
    network.transformer_view = fixture.view;
    network.transformer_scratch_lanes = {fixture.scratch};
    DispatcherDeviceTables tables{generators, central_state, zobrist};
    Stream2SolvedBuffers solved{
        memory.solved_flag,
        memory.stop_flag,
        memory.solved_count,
        memory.solved_overflow,
        memory.solved_meta_list,
        memory.solved_depth_list,
        config.solved_result_capacity};

    instantiate_cuda_graph_job_templates(plan, memory, tables, network, solved, streams, events, graphs);
    const DepthDispatchState state = run_depth_cuda_graphs(plan, memory, graphs, streams, config.b_micro);
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    require(state.depth_drained, "transformer depth dispatcher must drain the depth");
    require(state.ring_slot_jobs_launched == 1U, "transformer ring slot launch count mismatch");
    report << "- transformer_dispatcher_branch=pass\n";

    destroy_cuda_graph_job_templates(graphs);
    destroy_dispatcher_events(events);
    destroy_dispatcher_streams(streams);
    fixture.destroy();
    free_static_device_memory(memory);
    cudaFree(generators);
    cudaFree(central_state);
    cudaFree(zobrist);
}
} // namespace

int main() {
    std::filesystem::create_directories("test_results");
    std::ofstream report("test_results/dispatcher_cuda_tests_2026-05-22.md");
    report << "# Dispatcher CUDA Graph Template Tests 2026-05-22\n\n";
    BEAM_CUDA_CHECK(cudaSetDevice(0));

    RuntimeConfig config;
    config.b_micro = 16;
    config.stream3_batch_candidates = 16 * static_cast<std::uint32_t>(MOVE_COUNT) * 2;
    config.stream4_batch_candidates = 64;
    config.stream4_trigger_candidates = 64;
    config.stream4_active_sort_slots = 2;
    config.ring_count = 2;
    config.shard_count = 2;
    config.shard_capacity_candidates = 128;
    config.global_spill_capacity = 128;
    config.stream4_batch_alignment = 1;
    config.user_global_beam_width = 128;
    config.solved_result_capacity = 8;
    const StaticMemoryPlan plan = make_static_memory_plan(config);
    StaticDeviceMemory memory;
    allocate_static_device_memory(plan, memory);
    BEAM_CUDA_CHECK(cudaMemset(memory.allocation, 0, memory.allocation_bytes));

    std::uint8_t* generators = nullptr;
    State128* central_state = nullptr;
    Hash128* zobrist = nullptr;
    half* input_weight = nullptr;
    half* input_bias = nullptr;
    half* hidden_weight = nullptr;
    half* hidden_bias = nullptr;
    half* output_weight = nullptr;
    half* output_bias = nullptr;
    half* hidden1 = nullptr;
    half* hidden2 = nullptr;
    half* residual = nullptr;
    half* output = nullptr;
    constexpr std::uint32_t hidden1_cols = 16;
    constexpr std::uint32_t hidden2_cols = 8;
    BEAM_CUDA_CHECK(cudaMalloc(&generators, MOVE_COUNT * STATE_STORAGE_LEN));
    BEAM_CUDA_CHECK(cudaMalloc(&central_state, sizeof(State128)));
    BEAM_CUDA_CHECK(cudaMalloc(&zobrist, STATE_STORAGE_LEN * STATE_VALUE_PAD * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMalloc(&input_weight, STATE_LEN * STATE_VALUE_PAD * hidden1_cols * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&input_bias, hidden1_cols * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&hidden_weight, hidden1_cols * hidden2_cols * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&hidden_bias, hidden2_cols * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&output_weight, hidden2_cols * MOVE_COUNT * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&output_bias, MOVE_COUNT * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&hidden1, config.b_micro * hidden1_cols * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&hidden2, config.b_micro * hidden2_cols * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&residual, config.b_micro * hidden2_cols * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&output, config.b_micro * MOVE_COUNT * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMemset(generators, 0, MOVE_COUNT * STATE_STORAGE_LEN));
    BEAM_CUDA_CHECK(cudaMemset(central_state, 255, sizeof(State128)));
    BEAM_CUDA_CHECK(cudaMemset(zobrist, 0, STATE_STORAGE_LEN * STATE_VALUE_PAD * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMemset(input_weight, 0, STATE_LEN * STATE_VALUE_PAD * hidden1_cols * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMemset(input_bias, 0, hidden1_cols * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMemset(hidden_weight, 0, hidden1_cols * hidden2_cols * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMemset(hidden_bias, 0, hidden2_cols * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMemset(output_weight, 0, hidden2_cols * MOVE_COUNT * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMemset(output_bias, 0, MOVE_COUNT * sizeof(half)));

    DispatcherStreams streams;
    create_dispatcher_streams(streams);
    DispatcherEvents events;
    create_dispatcher_events(events);
    CudaGraphJobTemplates graphs;
    Stream1NetworkDims dims{STATE_LEN, STATE_VALUE_PAD, hidden1_cols, hidden2_cols, 0, static_cast<std::uint32_t>(MOVE_COUNT), STREAM1_DTYPE_FP16, STREAM1_NORM_NONE};
    DispatcherNetwork network{};
    network.backend = DispatcherStream1Backend::Mlp;
    network.mlp_view = Stream1NetworkView{input_weight, input_bias, nullptr, nullptr, hidden_weight, hidden_bias, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, output_weight, output_bias, dims};
    network.mlp_scratch_lanes = std::vector<Stream1CutlassScratch>{Stream1CutlassScratch{hidden1, hidden2, residual, output}};
    DispatcherDeviceTables tables{generators, central_state, zobrist};
    Stream2SolvedBuffers solved{
        memory.solved_flag,
        memory.stop_flag,
        memory.solved_count,
        memory.solved_overflow,
        memory.solved_meta_list,
        memory.solved_depth_list,
        config.solved_result_capacity};

    instantiate_cuda_graph_job_templates(plan, memory, tables, network, solved, streams, events, graphs);
    const std::uint32_t ring_slot_job_count = config.ring_count * plan.derived.ring_slot_count;
    require(graphs.ring_slot_graphs.size() == ring_slot_job_count, "ring slot graph count failed");
    require(graphs.ring_slot_execs.size() == ring_slot_job_count, "ring slot graph exec count failed");
    for (std::uint32_t job = 0; job < ring_slot_job_count; ++job) {
        require(graphs.ring_slot_graphs[job] != nullptr, "ring slot graph template missing");
        require(graphs.ring_slot_execs[job] != nullptr, "ring slot graph executable missing");
    }
    require(graphs.stream3_ring_graphs.size() == config.ring_count, "stream3 graph count failed");
    require(graphs.stream3_ring_execs.size() == config.ring_count, "stream3 graph exec count failed");
    for (std::uint32_t ring = 0; ring < config.ring_count; ++ring) {
        require(graphs.stream3_ring_graphs[ring] != nullptr, "stream3 graph template missing");
        require(graphs.stream3_ring_execs[ring] != nullptr, "stream3 graph executable missing");
    }
    const std::uint32_t stream4_graph_count = plan.storage_shard_count * config.stream4_active_sort_slots;
    require(graphs.stream4_shard_graphs.size() == stream4_graph_count, "stream4 graph count failed");
    require(graphs.stream4_shard_execs.size() == stream4_graph_count, "stream4 graph exec count failed");
    for (std::uint32_t graph = 0; graph < stream4_graph_count; ++graph) {
        require(graphs.stream4_shard_graphs[graph] != nullptr, "stream4 graph template missing");
        require(graphs.stream4_shard_execs[graph] != nullptr, "stream4 graph executable missing");
    }
    const std::uint64_t frontier_size =
        static_cast<std::uint64_t>(config.b_micro) * ring_slot_job_count + 3ULL;
    const DepthDispatchState state = run_depth_cuda_graphs(plan, memory, graphs, streams, frontier_size);
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    require(state.depth_drained, "depth dispatcher must drain the depth");
    require(state.frontier_cursor == frontier_size, "depth dispatcher frontier cursor mismatch");
    require(state.ring_slot_jobs_launched == ring_slot_job_count + 1U, "ring slot launch count mismatch");
    const std::uint32_t expected_parent_jobs = static_cast<std::uint32_t>(
        (frontier_size + config.b_micro - 1ULL) / config.b_micro);
    const std::uint32_t expected_stream3_jobs =
        (expected_parent_jobs + plan.derived.ring_slot_count - 1U) / plan.derived.ring_slot_count;
    require(state.stream3_jobs_launched == expected_stream3_jobs, "stream3 launch count mismatch");
    require(state.stream4_jobs_launched > 0, "stream4 conditional scheduler did not launch dirty flush");
    require(state.stream4_active_sort_slots_used > 0, "stream4 active sort slot usage missing");
    require(state.stream4_active_sort_slots_used <= config.stream4_active_sort_slots, "stream4 active sort slot usage mismatch");
    report << "- stream4_conditional_launches=" << state.stream4_jobs_launched << "\n";
    report << "- stream4_active_sort_slots_used=" << state.stream4_active_sort_slots_used << "\n";
    run_transformer_dispatcher_graph_test(report);
    run_transformer_runtime_weight_estimate_test(report);

    destroy_cuda_graph_job_templates(graphs);
    destroy_dispatcher_events(events);
    destroy_dispatcher_streams(streams);
    free_static_device_memory(memory);
    cudaFree(generators);
    cudaFree(central_state);
    cudaFree(zobrist);
    cudaFree(input_weight);
    cudaFree(input_bias);
    cudaFree(hidden_weight);
    cudaFree(hidden_bias);
    cudaFree(output_weight);
    cudaFree(output_bias);
    cudaFree(hidden1);
    cudaFree(hidden2);
    cudaFree(residual);
    cudaFree(output);

    report << "- cuda_graph_ring_slot_templates=pass\n";
    report << "- cuda_graph_stream3_ring_templates=pass\n";
    report << "- cuda_graph_stream4_shard_templates=pass\n";
    report << "- depth_cuda_graph_dispatch=pass\n";
    report << "- static_memory_used_for_pipeline_buffers=pass\n";
    report << "\nstatus=pass\n";
    std::cout << "dispatcher_cuda_tests=pass\n";
    return 0;
}
