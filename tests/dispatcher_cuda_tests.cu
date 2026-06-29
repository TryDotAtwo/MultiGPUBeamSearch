#include "cuda_check.hpp"
#include "../cuda/dispatcher.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace beam;

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
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
    DispatcherNetwork network{
        Stream1NetworkView{input_weight, input_bias, nullptr, nullptr, hidden_weight, hidden_bias, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, output_weight, output_bias, dims},
        std::vector<Stream1CutlassScratch>{Stream1CutlassScratch{hidden1, hidden2, residual, output}}};
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
    require(state.stream4_jobs_launched <= config.shard_count * 2U, "stream4 conditional scheduler launched too many jobs");
    require(state.stream4_active_sort_slots_used > 0, "stream4 active sort slot usage missing");
    require(state.stream4_active_sort_slots_used <= config.stream4_active_sort_slots, "stream4 active sort slot usage mismatch");
    report << "- stream4_conditional_launches=" << state.stream4_jobs_launched << "\n";
    report << "- stream4_active_sort_slots_used=" << state.stream4_active_sort_slots_used << "\n";

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
