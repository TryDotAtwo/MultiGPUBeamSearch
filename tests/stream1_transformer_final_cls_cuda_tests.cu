#include "cuda_check.hpp"
#include "../cuda/stream1.hpp"
#include "../cuda/stream1_transformer_shape.hpp"
#include "../tools/stream1_weight_io.hpp"
#include "state.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
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

void set_final_cls_only(bool enabled) {
#if defined(_WIN32)
    if (_putenv_s("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY", enabled ? "1" : "0") != 0) {
        throw std::runtime_error("failed to set final CLS-only test environment");
    }
#else
    if (setenv("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY", enabled ? "1" : "0", 1) != 0) {
        throw std::runtime_error("failed to set final CLS-only test environment");
    }
#endif
}

void set_final_cls_attention(bool enabled) {
#if defined(_WIN32)
    if (_putenv_s("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION", enabled ? "1" : "0") != 0) {
        throw std::runtime_error("failed to set final CLS-attention test environment");
    }
#else
    if (setenv("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION", enabled ? "1" : "0", 1) != 0) {
        throw std::runtime_error("failed to set final CLS-attention test environment");
    }
#endif
}

void set_final_cls_split_qkv(bool enabled) {
#if defined(_WIN32)
    if (_putenv_s("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV", enabled ? "1" : "0") != 0) {
        throw std::runtime_error("failed to set final CLS split-QKV test environment");
    }
#else
    if (setenv("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV", enabled ? "1" : "0", 1) != 0) {
        throw std::runtime_error("failed to set final CLS split-QKV test environment");
    }
#endif
}

void set_legacy_padding_zero(bool enabled) {
#if defined(_WIN32)
    if (_putenv_s("BEAM_STREAM1_TRANSFORMER_LEGACY_PADDING_ZERO", enabled ? "1" : "0") != 0) {
        throw std::runtime_error("failed to set legacy padding-zero test environment");
    }
#else
    if (setenv("BEAM_STREAM1_TRANSFORMER_LEGACY_PADDING_ZERO", enabled ? "1" : "0", 1) != 0) {
        throw std::runtime_error("failed to set legacy padding-zero test environment");
    }
#endif
}

void set_fused_input_layernorm(bool enabled) {
#if defined(_WIN32)
    if (_putenv_s("BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM", enabled ? "1" : "0") != 0) {
        throw std::runtime_error("failed to set fused input LayerNorm test environment");
    }
#else
    if (setenv("BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM", enabled ? "1" : "0", 1) != 0) {
        throw std::runtime_error("failed to set fused input LayerNorm test environment");
    }
#endif
}

std::vector<std::uint32_t> run_scores(
    bool final_cls_only,
    const State128* frontier,
    const std::uint64_t* parent_base,
    const std::uint32_t* count,
    const Stream1TransformerNetworkView& network,
    const Stream1TransformerScratchView& scratch,
    std::uint32_t* score,
    std::uint32_t rows) {
    set_final_cls_only(final_cls_only);
    BEAM_CUDA_CHECK(cudaMemset(score, 0, static_cast<std::size_t>(rows) * MOVE_COUNT * sizeof(std::uint32_t)));
    stream1_transformer_inference_cuda(
        frontier,
        parent_base,
        count,
        network,
        scratch,
        score,
        rows,
        0U,
        0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::uint32_t> result(static_cast<std::size_t>(rows) * MOVE_COUNT);
    BEAM_CUDA_CHECK(cudaMemcpy(
        result.data(),
        score,
        result.size() * sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost));
    return result;
}

std::uint32_t env_u32(const char* name, std::uint32_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    const unsigned long parsed = std::stoul(value);
    if (parsed == 0UL || parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string(name) + " must be a positive uint32");
    }
    return static_cast<std::uint32_t>(parsed);
}

float time_graph_ms(
    bool final_cls_only,
    const State128* frontier,
    const std::uint64_t* parent_base,
    const std::uint32_t* count,
    const Stream1TransformerNetworkView& network,
    const Stream1TransformerScratchView& scratch,
    std::uint32_t* score,
    std::uint32_t rows,
    std::uint32_t iterations) {
    set_final_cls_only(final_cls_only);
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    BEAM_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    stream1_transformer_inference_cuda(
        frontier, parent_base, count, network, scratch, score, rows, 0U, stream);
    BEAM_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    BEAM_CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (std::uint32_t warmup = 0; warmup < 3U; ++warmup) {
        BEAM_CUDA_CHECK(cudaGraphLaunch(executable, stream));
    }
    BEAM_CUDA_CHECK(cudaStreamSynchronize(stream));
    BEAM_CUDA_CHECK(cudaEventCreate(&start));
    BEAM_CUDA_CHECK(cudaEventCreate(&stop));
    BEAM_CUDA_CHECK(cudaEventRecord(start, stream));
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        BEAM_CUDA_CHECK(cudaGraphLaunch(executable, stream));
    }
    BEAM_CUDA_CHECK(cudaEventRecord(stop, stream));
    BEAM_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    BEAM_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return elapsed_ms / static_cast<float>(iterations);
}

} // namespace

int main() {
    const char* weights_override = std::getenv("BEAM_STREAM1_TRANSFORMER_WEIGHTS_DIR");
    const std::filesystem::path weights_dir =
        weights_override != nullptr && weights_override[0] != '\0'
        ? std::filesystem::path(weights_override)
        : std::filesystem::path("test_results/stream1_transformer_reference/weights_fp16");
    if (!std::filesystem::exists(weights_dir / "manifest.json")) {
        std::cout << "stream1_transformer_final_cls_cuda_tests=skip missing_weights_fixture\n";
        return 0;
    }

    const std::uint32_t rows = env_u32("BEAM_STREAM1_FINAL_CLS_TEST_ROWS", 64U);
    std::vector<State128> states(rows);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t pos = 0; pos < STATE_LEN; ++pos) {
            states[row].v[pos] = static_cast<std::uint8_t>((row * 17U + pos * 5U) % 6U);
        }
        clear_state_padding(states[row]);
    }

    stream1_weights::HostWeightBytes host_weights = stream1_weights::load_stream1_weights(weights_dir);
    require(host_weights.model.backend == STREAM1_BACKEND_PIECE_TRANSFORMER, "fixture must be piece_transformer");
    stream1_weights::DeviceWeights device_weights = stream1_weights::upload_weights(host_weights);
    stream1_weights::ScratchAllocation scratch_allocation =
        stream1_weights::alloc_stream1_scratch(host_weights.model, rows, 1U);
    stream1_weights::TransformerNetworkViewHolder view_holder =
        stream1_weights::transformer_network_view(device_weights.transformer, host_weights.model);
    const Stream1TransformerDims dims = view_holder.view.dims;
    require(
        stream1_transformer_supports_generic_final_cls_only(
            dims.seq_len,
            dims.padded_seq_len,
            dims.d_model,
            dims.nhead,
            dims.head_dim,
            dims.transformer_layers,
            dims.ff_dim,
            dims.output_dim),
        "fixture must exercise the generic final CLS-only path");

    State128* d_frontier = nullptr;
    std::uint64_t* d_parent_base = nullptr;
    std::uint32_t* d_count = nullptr;
    std::uint32_t* d_score = nullptr;
    const std::uint64_t parent_base_value = 0ULL;
    const std::uint32_t count_value = rows;
    BEAM_CUDA_CHECK(cudaMalloc(&d_frontier, states.size() * sizeof(State128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_parent_base, sizeof(std::uint64_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_count, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_score, static_cast<std::size_t>(rows) * MOVE_COUNT * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemcpy(d_frontier, states.data(), states.size() * sizeof(State128), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_parent_base, &parent_base_value, sizeof(parent_base_value), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_count, &count_value, sizeof(count_value), cudaMemcpyHostToDevice));

    const Stream1TransformerScratchView scratch = stream1_weights::transformer_scratch_view(scratch_allocation);
    set_legacy_padding_zero(true);
    const std::vector<std::uint32_t> legacy_padding = run_scores(
        false, d_frontier, d_parent_base, d_count, view_holder.view, scratch, d_score, rows);
    set_legacy_padding_zero(false);
    const std::vector<std::uint32_t> baseline = run_scores(
        false, d_frontier, d_parent_base, d_count, view_holder.view, scratch, d_score, rows);
    Stream1TransformerNetworkView silu_network = view_holder.view;
    silu_network.dims.activation = STREAM1_ACTIVATION_SILU;
    const std::vector<std::uint32_t> silu_scores = run_scores(
        false, d_frontier, d_parent_base, d_count, silu_network, scratch, d_score, rows);
    set_final_cls_attention(false);
    const std::vector<std::uint32_t> optimized = run_scores(
        true, d_frontier, d_parent_base, d_count, view_holder.view, scratch, d_score, rows);
    set_final_cls_attention(true);
    const std::vector<std::uint32_t> cls_attention = run_scores(
        true, d_frontier, d_parent_base, d_count, view_holder.view, scratch, d_score, rows);
    set_final_cls_split_qkv(true);
    const std::vector<std::uint32_t> split_qkv = run_scores(
        true, d_frontier, d_parent_base, d_count, view_holder.view, scratch, d_score, rows);
    set_fused_input_layernorm(true);
    const std::vector<std::uint32_t> fused_input_layernorm = run_scores(
        true, d_frontier, d_parent_base, d_count, view_holder.view, scratch, d_score, rows);
    set_fused_input_layernorm(false);
    set_final_cls_split_qkv(false);
    set_final_cls_attention(false);
    set_final_cls_only(false);
    require(baseline == legacy_padding, "tail-only padding zero must preserve every score key");
    require(baseline != silu_scores, "ReLU fixture scores must differ from an explicit SiLU forward");
    require(optimized == baseline, "generic final CLS-only score keys must be byte exact");
    require(cls_attention == baseline, "generic final CLS-attention score keys must be byte exact");
    require(split_qkv == baseline, "generic final CLS split-QKV score keys must be byte exact");
    require(fused_input_layernorm == baseline, "generic fused input LayerNorm score keys must be byte exact");

    const char* benchmark_iterations_env = std::getenv("BEAM_STREAM1_FINAL_CLS_BENCH_ITERATIONS");
    if (benchmark_iterations_env != nullptr && benchmark_iterations_env[0] != '\0') {
        const std::uint32_t iterations = env_u32("BEAM_STREAM1_FINAL_CLS_BENCH_ITERATIONS", 10U);
        set_legacy_padding_zero(true);
        const float legacy_padding_ms = time_graph_ms(
            false, d_frontier, d_parent_base, d_count, view_holder.view, scratch, d_score, rows, iterations);
        set_legacy_padding_zero(false);
        const float baseline_ms = time_graph_ms(
            false, d_frontier, d_parent_base, d_count, view_holder.view, scratch, d_score, rows, iterations);
        const float optimized_ms = time_graph_ms(
            true, d_frontier, d_parent_base, d_count, view_holder.view, scratch, d_score, rows, iterations);
        set_final_cls_attention(true);
        const float cls_attention_ms = time_graph_ms(
            true, d_frontier, d_parent_base, d_count, view_holder.view, scratch, d_score, rows, iterations);
        set_final_cls_split_qkv(true);
        const float split_qkv_ms = time_graph_ms(
            true, d_frontier, d_parent_base, d_count, view_holder.view, scratch, d_score, rows, iterations);
        set_fused_input_layernorm(true);
        const float fused_input_layernorm_ms = time_graph_ms(
            true, d_frontier, d_parent_base, d_count, view_holder.view, scratch, d_score, rows, iterations);
        set_fused_input_layernorm(false);
        set_final_cls_split_qkv(false);
        set_final_cls_attention(false);
        std::cout << "stream1_transformer_final_cls_benchmark"
                  << " rows=" << rows
                  << " iterations=" << iterations
                  << " legacy_padding_ms=" << legacy_padding_ms
                  << " tail_padding_ms=" << baseline_ms
                  << " tail_padding_speedup=" << (legacy_padding_ms / baseline_ms)
                  << " baseline_ms=" << baseline_ms
                  << " optimized_ms=" << optimized_ms
                  << " speedup=" << (baseline_ms / optimized_ms)
                  << " cls_attention_ms=" << cls_attention_ms
                  << " cls_attention_speedup=" << (optimized_ms / cls_attention_ms)
                  << " split_qkv_ms=" << split_qkv_ms
                  << " split_qkv_speedup=" << (cls_attention_ms / split_qkv_ms)
                  << " fused_input_layernorm_ms=" << fused_input_layernorm_ms
                  << " fused_input_layernorm_speedup=" << (split_qkv_ms / fused_input_layernorm_ms)
                  << "\n";
    }
    set_final_cls_only(false);
    set_final_cls_attention(false);
    set_final_cls_split_qkv(false);
    set_fused_input_layernorm(false);
    set_legacy_padding_zero(false);

    cudaFree(d_frontier);
    cudaFree(d_parent_base);
    cudaFree(d_count);
    cudaFree(d_score);
    stream1_weights::free_stream1_scratch(scratch_allocation);
    stream1_weights::free_weights(device_weights);
    std::cout << "stream1_transformer_final_cls_cuda_tests=pass rows=" << rows << "\n";
    return 0;
}
