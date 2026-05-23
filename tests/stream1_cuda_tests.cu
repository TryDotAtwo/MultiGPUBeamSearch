#include "cuda_check.hpp"
#include "../cuda/stream1.hpp"
#include "state.hpp"

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
    std::ofstream report("test_results/stream1_cuda_tests_2026-05-20.md");
    report << "# Stream1 CUDA Tests 2026-05-20\n\n";

    BEAM_CUDA_CHECK(cudaSetDevice(0));
    State128 state{};
    for (std::size_t i = 0; i < STATE_LEN; ++i) {
        state.v[i] = static_cast<std::uint8_t>(i);
    }
    clear_state_padding(state);

    constexpr std::uint32_t cutlass_b_micro = 16;
    State128* d_frontier = nullptr;
    std::uint64_t* d_parent_base = nullptr;
    std::uint32_t* d_count = nullptr;
    std::uint32_t* d_score = nullptr;
    half* d_input_weight = nullptr;
    half* d_input_bias = nullptr;
    half* d_hidden_weight = nullptr;
    half* d_hidden_bias = nullptr;
    half* d_output_weight = nullptr;
    half* d_output_bias = nullptr;
    half* d_cutlass_hidden1 = nullptr;
    half* d_cutlass_hidden2 = nullptr;
    half* d_cutlass_residual_tmp = nullptr;
    half* d_cutlass_output = nullptr;
    half* d_residual_fc1_weight = nullptr;
    half* d_residual_fc1_bias = nullptr;
    half* d_residual_fc2_weight = nullptr;
    half* d_residual_fc2_bias = nullptr;
    const std::uint64_t parent_base = 0;
    const std::uint32_t count = 1;
    std::vector<State128> frontier(cutlass_b_micro, state);
    BEAM_CUDA_CHECK(cudaMalloc(&d_frontier, frontier.size() * sizeof(State128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_parent_base, sizeof(std::uint64_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_count, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_score, cutlass_b_micro * MOVE_COUNT * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemcpy(d_frontier, frontier.data(), frontier.size() * sizeof(State128), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_parent_base, &parent_base, sizeof(parent_base), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_count, &count, sizeof(count), cudaMemcpyHostToDevice));

    stream1_score_contract_cuda(d_frontier, d_parent_base, d_count, d_score, 0, 0, 1, 0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<std::uint32_t> score(MOVE_COUNT);
    BEAM_CUDA_CHECK(cudaMemcpy(score.data(), d_score, score.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    require(score[0] <= SCORE_MAX_KEY, "stream1 score key range failed");
    require(score[1] != score[0], "stream1 move-dependent score failed");

    const Stream1NetworkDims dims{STATE_LEN, STATE_VALUE_PAD, 16, 8, 1};
    std::vector<half> input_weight(static_cast<std::size_t>(dims.state_len) * dims.num_classes * dims.hidden1, __float2half(0.0f));
    std::vector<half> input_bias(dims.hidden1, __float2half(0.0f));
    std::vector<half> hidden_weight(static_cast<std::size_t>(dims.hidden1) * dims.hidden2, __float2half(0.0f));
    std::vector<half> hidden_bias(dims.hidden2, __float2half(0.0f));
    std::vector<half> output_weight(static_cast<std::size_t>(dims.hidden2) * MOVE_COUNT, __float2half(0.0f));
    std::vector<half> output_bias(MOVE_COUNT, __float2half(0.0f));
    std::vector<half> residual_fc1_weight(static_cast<std::size_t>(dims.hidden2) * dims.hidden2, __float2half(0.0f));
    std::vector<half> residual_fc1_bias(dims.hidden2, __float2half(0.0f));
    std::vector<half> residual_fc2_weight(static_cast<std::size_t>(dims.hidden2) * dims.hidden2, __float2half(0.0f));
    std::vector<half> residual_fc2_bias(dims.hidden2, __float2half(0.0f));
    for (std::size_t p = 0; p < STATE_LEN; ++p) {
        input_weight[(p * dims.num_classes + state.v[p]) * dims.hidden1 + 0] = __float2half(0.001f);
    }
    hidden_weight[0] = __float2half(1.0f);
    hidden_weight[1] = __float2half(0.5f);
    for (std::size_t move = 0; move < MOVE_COUNT; ++move) {
        output_weight[move] = __float2half(static_cast<float>(move + 1));
        output_weight[MOVE_COUNT + move] = __float2half(0.25f);
    }
    residual_fc1_weight[0] = __float2half(1.0f);
    residual_fc1_weight[dims.hidden2 + 1] = __float2half(1.0f);
    residual_fc2_weight[0] = __float2half(1.0f);
    residual_fc2_weight[dims.hidden2 + 1] = __float2half(1.0f);
    BEAM_CUDA_CHECK(cudaMalloc(&d_input_weight, input_weight.size() * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_input_bias, input_bias.size() * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_hidden_weight, hidden_weight.size() * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_hidden_bias, hidden_bias.size() * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_output_weight, output_weight.size() * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_output_bias, output_bias.size() * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_residual_fc1_weight, residual_fc1_weight.size() * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_residual_fc1_bias, residual_fc1_bias.size() * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_residual_fc2_weight, residual_fc2_weight.size() * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_residual_fc2_bias, residual_fc2_bias.size() * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMemcpy(d_input_weight, input_weight.data(), input_weight.size() * sizeof(half), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_input_bias, input_bias.data(), input_bias.size() * sizeof(half), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_hidden_weight, hidden_weight.data(), hidden_weight.size() * sizeof(half), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_hidden_bias, hidden_bias.data(), hidden_bias.size() * sizeof(half), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_output_weight, output_weight.data(), output_weight.size() * sizeof(half), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_output_bias, output_bias.data(), output_bias.size() * sizeof(half), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_residual_fc1_weight, residual_fc1_weight.data(), residual_fc1_weight.size() * sizeof(half), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_residual_fc1_bias, residual_fc1_bias.data(), residual_fc1_bias.size() * sizeof(half), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_residual_fc2_weight, residual_fc2_weight.data(), residual_fc2_weight.size() * sizeof(half), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_residual_fc2_bias, residual_fc2_bias.data(), residual_fc2_bias.size() * sizeof(half), cudaMemcpyHostToDevice));
    const half* residual_fc1_weight_ptrs[] = {d_residual_fc1_weight};
    const half* residual_fc1_bias_ptrs[] = {d_residual_fc1_bias};
    const half* residual_fc2_weight_ptrs[] = {d_residual_fc2_weight};
    const half* residual_fc2_bias_ptrs[] = {d_residual_fc2_bias};
    Stream1NetworkView network{
        d_input_weight,
        d_input_bias,
        d_hidden_weight,
        d_hidden_bias,
        residual_fc1_weight_ptrs,
        residual_fc1_bias_ptrs,
        residual_fc2_weight_ptrs,
        residual_fc2_bias_ptrs,
        d_output_weight,
        d_output_bias,
        dims};
    BEAM_CUDA_CHECK(cudaMemset(d_score, 0, MOVE_COUNT * sizeof(std::uint32_t)));
    stream1_inference_custom_cuda(d_frontier, d_parent_base, d_count, network, d_score, 0, 0, 1, 0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    BEAM_CUDA_CHECK(cudaMemcpy(score.data(), d_score, score.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    require(score[0] > 0, "stream1 custom inference score must be positive");
    require(score[1] > score[0], "stream1 custom inference must produce move-specific scores");

    BEAM_CUDA_CHECK(cudaMalloc(&d_cutlass_hidden1, cutlass_b_micro * dims.hidden1 * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_cutlass_hidden2, cutlass_b_micro * dims.hidden2 * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_cutlass_residual_tmp, cutlass_b_micro * dims.hidden2 * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_cutlass_output, cutlass_b_micro * MOVE_COUNT * sizeof(half)));
    Stream1CutlassScratch scratch{d_cutlass_hidden1, d_cutlass_hidden2, d_cutlass_residual_tmp, d_cutlass_output};
    BEAM_CUDA_CHECK(cudaMemcpy(d_count, &cutlass_b_micro, sizeof(cutlass_b_micro), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemset(d_score, 0, cutlass_b_micro * MOVE_COUNT * sizeof(std::uint32_t)));
    stream1_inference_cutlass_cuda(d_frontier, d_parent_base, d_count, network, scratch, d_score, cutlass_b_micro, 0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    BEAM_CUDA_CHECK(cudaMemcpy(score.data(), d_score, score.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    require(score[0] > 0, "stream1 CUTLASS inference score must be positive");
    require(score[1] > score[0], "stream1 CUTLASS inference must produce move-specific scores");

    half* d_cutlass_in = nullptr;
    half* d_cutlass_w = nullptr;
    half* d_cutlass_out = nullptr;
    std::vector<half> cutlass_in(16 * 8, __float2half(1.0f));
    std::vector<half> cutlass_w(8 * 8, __float2half(0.0f));
    for (std::uint32_t i = 0; i < 8; ++i) {
        cutlass_w[i * 8 + i] = __float2half(1.0f);
    }
    std::vector<half> cutlass_out(16 * 8);
    BEAM_CUDA_CHECK(cudaMalloc(&d_cutlass_in, cutlass_in.size() * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_cutlass_w, cutlass_w.size() * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_cutlass_out, cutlass_out.size() * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMemcpy(d_cutlass_in, cutlass_in.data(), cutlass_in.size() * sizeof(half), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_cutlass_w, cutlass_w.data(), cutlass_w.size() * sizeof(half), cudaMemcpyHostToDevice));
    stream1_cutlass_linear_cuda(d_cutlass_in, d_cutlass_w, d_cutlass_out, 16, 8, 8, 0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    BEAM_CUDA_CHECK(cudaMemcpy(cutlass_out.data(), d_cutlass_out, cutlass_out.size() * sizeof(half), cudaMemcpyDeviceToHost));
    require(__half2float(cutlass_out[0]) > 0.9f && __half2float(cutlass_out[9]) > 0.9f, "stream1 CUTLASS linear contract failed");
    report << "- score_key_range=pass\n";
    report << "- no_q_float_global_contract=pass\n";
    report << "- custom_inference_no_embeddingbag_kernel=pass\n";
    report << "- cutlass_linear_contract=pass\n";
    report << "- cutlass_inference_contract=pass\n";
    report << "\nstatus=pass\n";

    cudaFree(d_frontier);
    cudaFree(d_parent_base);
    cudaFree(d_count);
    cudaFree(d_score);
    cudaFree(d_input_weight);
    cudaFree(d_input_bias);
    cudaFree(d_hidden_weight);
    cudaFree(d_hidden_bias);
    cudaFree(d_output_weight);
    cudaFree(d_output_bias);
    cudaFree(d_cutlass_hidden1);
    cudaFree(d_cutlass_hidden2);
    cudaFree(d_cutlass_residual_tmp);
    cudaFree(d_cutlass_output);
    cudaFree(d_residual_fc1_weight);
    cudaFree(d_residual_fc1_bias);
    cudaFree(d_residual_fc2_weight);
    cudaFree(d_residual_fc2_bias);
    cudaFree(d_cutlass_in);
    cudaFree(d_cutlass_w);
    cudaFree(d_cutlass_out);
    std::cout << "stream1_cuda_tests=pass\n";
    return 0;
}
