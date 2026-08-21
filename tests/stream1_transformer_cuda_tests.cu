#include "cuda_check.hpp"
#include "../cuda/stream1.hpp"
#include "../cuda/stream1_transformer_shape.hpp"
#include "../tools/stream1_weight_io.hpp"
#include "state.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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

std::string read_text(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open reference file: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

std::vector<double> parse_json_numbers_after(
    const std::string& text,
    const char* key,
    std::size_t expected_count) {
    const std::string quoted = std::string("\"") + key + "\"";
    std::size_t pos = text.find(quoted);
    if (pos == std::string::npos) {
        throw std::runtime_error("reference JSON missing key: " + std::string(key));
    }
    pos = text.find('[', pos + quoted.size());
    if (pos == std::string::npos) {
        throw std::runtime_error("reference JSON key is not an array: " + std::string(key));
    }
    std::vector<double> values;
    values.reserve(expected_count);
    while (pos < text.size() && values.size() < expected_count) {
        const char c = text[pos];
        if (c == '-' || c == '+' || c == '.' || std::isdigit(static_cast<unsigned char>(c))) {
            char* end = nullptr;
            const double value = std::strtod(text.c_str() + pos, &end);
            if (end == text.c_str() + pos) {
                throw std::runtime_error("failed to parse reference JSON number");
            }
            values.push_back(value);
            pos = static_cast<std::size_t>(end - text.c_str());
            continue;
        }
        ++pos;
    }
    if (values.size() != expected_count) {
        throw std::runtime_error(
            "reference JSON number count mismatch for " + std::string(key) +
            " expected=" + std::to_string(expected_count) +
            " actual=" + std::to_string(values.size()));
    }
    return values;
}

std::uint32_t reference_score_key(double q) {
    const double clamped = std::min(std::max(q, 0.0), static_cast<double>(SCORE_MAX_Q));
    return static_cast<std::uint32_t>(std::llround(clamped * static_cast<double>(SCORE_SCALE)));
}

std::filesystem::path reference_fixture_root() {
    const char* override_path = std::getenv("BEAM_STREAM1_TRANSFORMER_REFERENCE_DIR");
    if (override_path != nullptr && override_path[0] != '\0') {
        return std::filesystem::path(override_path);
    }
    return std::filesystem::path("test_results/stream1_transformer_reference");
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

} // namespace

int main() {
    std::filesystem::create_directories("test_results");
    std::ofstream report("test_results/stream1_transformer_cuda_tests_2026-06-29.md");
    report << "# Stream1 Transformer CUDA Tests 2026-06-29\n\n";

    const std::filesystem::path fixture_root = reference_fixture_root();
    const std::filesystem::path weights_dir = fixture_root / "weights_fp16";
    const std::filesystem::path manifest_path = weights_dir / "manifest.json";
    const std::filesystem::path reference_path = fixture_root / "reference.json";
    const std::uint32_t reference_count = 8;

    if (!std::filesystem::exists(manifest_path) || !std::filesystem::exists(reference_path)) {
        const std::string skip_line = "stream1_transformer_cuda_tests=skip missing_reference_fixture";
        report << skip_line << "\n";
        report << "- fixture_root=" << fixture_root.generic_string() << "\n";
        report << "- manifest_json=" << manifest_path.generic_string() << "\n";
        report << "- reference_json=" << reference_path.generic_string() << "\n";
        report << "\nstatus=skip missing_reference_fixture\n";
        std::cout << skip_line << "\n";
        return 0;
    }

    BEAM_CUDA_CHECK(cudaSetDevice(0));

    const std::string reference_json = read_text(reference_path);
    const std::vector<double> state_numbers =
        parse_json_numbers_after(reference_json, "states", reference_count * STATE_LEN);
    const std::vector<double> reference_scores =
        parse_json_numbers_after(reference_json, "scores_fp32", reference_count * MOVE_COUNT);

    std::vector<State128> states(reference_count);
    for (std::uint32_t row = 0; row < reference_count; ++row) {
        for (std::uint32_t p = 0; p < STATE_LEN; ++p) {
            const double value = state_numbers[static_cast<std::size_t>(row) * STATE_LEN + p];
            states[row].v[p] = static_cast<std::uint8_t>(value);
        }
        clear_state_padding(states[row]);
    }

    stream1_weights::HostWeightBytes host_weights = stream1_weights::load_stream1_weights(weights_dir);
    require(host_weights.model.backend == STREAM1_BACKEND_PIECE_TRANSFORMER, "fixture must be piece_transformer");
    stream1_weights::DeviceWeights device_weights = stream1_weights::upload_weights(host_weights);
    stream1_weights::ScratchAllocation scratch_allocation =
        stream1_weights::alloc_stream1_scratch(host_weights.model, reference_count, 1);

    State128* d_frontier = nullptr;
    std::uint64_t* d_parent_base = nullptr;
    std::uint32_t* d_count = nullptr;
    std::uint32_t* d_score = nullptr;
    const std::uint64_t parent_base = 0;
    const std::uint32_t count = reference_count;
    BEAM_CUDA_CHECK(cudaMalloc(&d_frontier, states.size() * sizeof(State128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_parent_base, sizeof(std::uint64_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_count, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_score, reference_count * MOVE_COUNT * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemcpy(d_frontier, states.data(), states.size() * sizeof(State128), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_parent_base, &parent_base, sizeof(parent_base), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_count, &count, sizeof(count), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemset(d_score, 0, reference_count * MOVE_COUNT * sizeof(std::uint32_t)));

    stream1_weights::TransformerNetworkViewHolder view_holder =
        stream1_weights::transformer_network_view(device_weights.transformer, host_weights.model);
    const Stream1TransformerScratchView scratch_view =
        stream1_weights::transformer_scratch_view(scratch_allocation);

    set_final_cls_only(false);
    stream1_transformer_inference_cuda(
        d_frontier,
        d_parent_base,
        d_count,
        view_holder.view,
        scratch_view,
        d_score,
        reference_count,
        0U,
        0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<std::uint32_t> cuda_scores(reference_count * MOVE_COUNT);
    BEAM_CUDA_CHECK(cudaMemcpy(cuda_scores.data(), d_score, cuda_scores.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));

    const Stream1TransformerDims runtime_dims = view_holder.view.dims;
    const bool generic_final_cls_shape = stream1_transformer_supports_generic_final_cls_only(
        runtime_dims.seq_len,
        runtime_dims.padded_seq_len,
        runtime_dims.d_model,
        runtime_dims.nhead,
        runtime_dims.head_dim,
        runtime_dims.transformer_layers,
        runtime_dims.ff_dim,
        runtime_dims.output_dim);
    if (generic_final_cls_shape) {
        BEAM_CUDA_CHECK(cudaMemset(d_score, 0, reference_count * MOVE_COUNT * sizeof(std::uint32_t)));
        set_final_cls_only(true);
        stream1_transformer_inference_cuda(
            d_frontier,
            d_parent_base,
            d_count,
            view_holder.view,
            scratch_view,
            d_score,
            reference_count,
            0U,
            0);
        BEAM_CUDA_CHECK(cudaGetLastError());
        BEAM_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint32_t> final_cls_scores(reference_count * MOVE_COUNT);
        BEAM_CUDA_CHECK(cudaMemcpy(
            final_cls_scores.data(),
            d_score,
            final_cls_scores.size() * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost));
        require(final_cls_scores == cuda_scores, "generic final CLS-only score keys must be byte exact");
        report << "- generic_final_cls_only_exact=pass\n";
    }
    set_final_cls_only(false);

    std::uint32_t max_abs_error = 0;
    for (std::size_t i = 0; i < cuda_scores.size(); ++i) {
        const std::uint32_t expected = reference_score_key(reference_scores[i]);
        const std::uint32_t actual = cuda_scores[i];
        const std::uint32_t error = expected > actual ? expected - actual : actual - expected;
        max_abs_error = std::max(max_abs_error, error);
    }
    report << "- reference_rows=" << reference_count << "\n";
    report << "- max_abs_score_key_error=" << max_abs_error << "\n";
    report.flush();
    require(max_abs_error <= 3072U, "transformer CUDA score keys drifted beyond tolerance");
    report << "- transformer_forward_reference=pass\n";
    report << "- standalone_transformer_forward=pass\n";
    report << "\nstatus=pass\n";

    cudaFree(d_frontier);
    cudaFree(d_parent_base);
    cudaFree(d_count);
    cudaFree(d_score);
    stream1_weights::free_stream1_scratch(scratch_allocation);
    stream1_weights::free_weights(device_weights);
    std::cout << "stream1_transformer_cuda_tests=pass\n";
    return 0;
}
