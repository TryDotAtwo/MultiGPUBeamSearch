#include "cuda_check.hpp"
#include "../cuda/stream1.hpp"
#include "../cuda/stream1_transformer_shape.hpp"
#include "../tools/stream1_weight_io.hpp"
#include "../cuda/stream1_layout_inventory.hpp"
#include "../cuda/stream1_execution_shape.hpp"
#include "../cuda/stream1_executor_contract.hpp"
#include "../cuda/stream1_kernel_description.hpp"
#include "../cuda/stream1_transformer_policy_snapshot.hpp"
#include "state.hpp"
#include "../third_party/nlohmann/json.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include <set>

using namespace beam;
#include "stream1_graph_lane_probe.hpp"

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
    using Json = nlohmann::ordered_json;
    std::vector<std::set<std::string>> keys;
    const auto root = Json::parse(text, [&](int depth, Json::parse_event_t event, Json& value) {
        require(depth <= 64, "reference nesting exceeds limit");
        if (event == Json::parse_event_t::object_start) keys.emplace_back();
        if (event == Json::parse_event_t::key)
            require(keys.back().insert(value.get<std::string>()).second, "duplicate reference key");
        if (event == Json::parse_event_t::object_end) keys.pop_back();
        return true;
    });
    require(root.is_object() && root.contains(key), "reference must contain matrix");
    const auto& matrix = root.at(key);
    const bool states = std::string(key) == "states";
    const std::size_t width = states ? STATE_LEN : MOVE_COUNT;
    require(matrix.is_array() && matrix.size() == expected_count / width,
            "reference row count mismatch");
    std::vector<double> values;
    values.reserve(expected_count);
    for (const auto& row : matrix) {
        require(row.is_array() && row.size() == width, "reference column count mismatch");
        for (const auto& entry : row) {
            require(entry.is_number(), "reference entry must be numeric");
            const double value = entry.get<double>();
            require(std::isfinite(value), "nonfinite reference entry");
            if (states) require(entry.is_number_unsigned() && value < 6,
                                "Cube4 reference state must be integer in 0..5");
            values.push_back(value);
        }
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

const char* observed_cls_flag(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return "0";
    if (std::string(value) != "0" && std::string(value) != "1") {
        throw std::invalid_argument(std::string(name) + " must be unset, 0, or 1");
    }
    return value;
}

} // namespace

int main(int argc, char** argv) {
    Stream1KernelObservation actual_kernel_observation(1024);
    Stream1KernelObservationScope kernel_observation_scope(actual_kernel_observation);
    bool require_reference = false;
    bool runtime_final_cls_scores = false;
    bool runtime_cuda_graph_scores = false;
    bool verify_packed_small_batch = false;
    bool verify_graph_input_reuse = false;
    bool verify_graph_job_bounds = false;
    bool verify_frozen_policies = false;
    bool verify_graph_lanes = false;
    bool verify_production_lanes = false;
    std::uint32_t probe_micro = 1024, probe_lanes = 4, probe_inner = 0;
    bool micro_supplied = false, lanes_supplied = false, inner_supplied = false;
    for (int arg = 1; arg < argc; ++arg) {
        const std::string option(argv[arg]);
        if (option == "--probe-microbatch" || option == "--probe-lanes" || option == "--probe-transformer-micro") {
            const bool is_micro = option == "--probe-microbatch";
            const bool is_inner = option == "--probe-transformer-micro";
            bool& supplied = is_inner ? inner_supplied : is_micro ? micro_supplied : lanes_supplied;
            std::uint32_t value = 0;
            if (supplied || arg + 1 == argc) {
                std::cerr << "invalid lane profile option\n"; return 2;
            }
            const std::string text(argv[++arg]);
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
            if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
                value == 0 || value > ((is_micro || is_inner) ? 65536U : 64U)) {
                std::cerr << "invalid lane profile value\n"; return 2;
            }
            supplied = true;
            (is_inner ? probe_inner : is_micro ? probe_micro : probe_lanes) = value;
            continue;
        }
        if (option == "--require-reference" && !require_reference) {
            require_reference = true;
        } else if (option == "--verify-packed-small-batch" && !verify_packed_small_batch) {
            verify_packed_small_batch = true;
        } else if (option == "--runtime-final-cls-scores" && !runtime_final_cls_scores) {
            runtime_final_cls_scores = true;
        } else if (option == "--runtime-cuda-graph-scores" && !runtime_cuda_graph_scores) {
            runtime_cuda_graph_scores = true;
        } else if (option == "--verify-graph-input-reuse" && !verify_graph_input_reuse) {
            verify_graph_input_reuse = true;
        } else if (option == "--verify-graph-job-bounds" && !verify_graph_job_bounds) {
            verify_graph_job_bounds = true;
        } else if (option == "--verify-frozen-policies" && !verify_frozen_policies) {
            verify_frozen_policies = true;
        } else if (option == "--verify-graph-lanes" && !verify_graph_lanes) {
            verify_graph_lanes = true;
        } else if (option == "--verify-production-lanes" && !verify_production_lanes) {
            verify_production_lanes = true;
        } else {
            std::cerr << "usage: stream1_transformer_cuda_tests [--require-reference] [--runtime-final-cls-scores] [--runtime-cuda-graph-scores]\n";
            return 2;
        }
    }
    if ((micro_supplied || lanes_supplied || inner_supplied) && !verify_production_lanes) {
        std::cerr << "lane profile requires --verify-production-lanes\n";
        return 2;
    }
    if (inner_supplied && probe_inner > probe_micro) {
        std::cerr << "inner probe micro exceeds outer micro\n";
        return 2;
    }
    if ((verify_packed_small_batch || verify_graph_input_reuse || verify_graph_job_bounds || verify_graph_lanes || verify_production_lanes) && !runtime_cuda_graph_scores) {
        std::cerr << "graph input reuse probe requires --runtime-cuda-graph-scores\n";
        return 2;
    }
    // Runtime score mode is never allowed to silently skip a missing fixture.
    if (const char* requested_executor = std::getenv("BEAM_STREAM1_EXECUTOR")) {
        if (requested_executor[0] != '\0') {
            try {
                const auto expected = runtime_cuda_graph_scores ? Stream1Executor::NativeGraph : Stream1Executor::NativeEager;
                if (resolve_stream1_executor(requested_executor) != expected) {
                    std::cerr << "requested executor differs from actual probe execution\n";
                    return 2;
                }
            } catch (const std::invalid_argument& error) {
                std::cerr << error.what() << '\n'; return 2;
            }
        }
    }
    require_reference = require_reference || runtime_final_cls_scores || runtime_cuda_graph_scores;
    runtime_final_cls_scores = runtime_final_cls_scores || runtime_cuda_graph_scores;
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
        if (require_reference) {
            report << "production_gate=fail reference_required\n";
            std::cerr << "production_gate=fail reference_required\n";
            return 1;
        }
        return 77;
    }

    std::vector<double> state_numbers, reference_scores;
    try {
        require(std::filesystem::file_size(reference_path) <= 8 * 1024 * 1024,
                "reference exceeds byte limit");
        const std::string reference_json = read_text(reference_path);
        state_numbers = parse_json_numbers_after(reference_json, "states", reference_count * STATE_LEN);
        reference_scores = parse_json_numbers_after(reference_json, "scores_fp32", reference_count * MOVE_COUNT);
    } catch (const std::exception& error) {
        std::cerr << "invalid_reference: " << error.what() << '\n';
        return 1;
    }
    BEAM_CUDA_CHECK(cudaSetDevice(0));

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
    if (verify_packed_small_batch) {
        require(host_weights.model.dtype == STREAM1_DTYPE_FP16,
                "packed small batch requires FP16 reference");
        require(device_weights.transformer.blocks.size() == 4,
                "packed small batch requires four transformer blocks");
        for (std::size_t layer = 0; layer < 4; ++layer) {
            const auto& block = device_weights.transformer.blocks[layer];
            require(block.qkv_hopper_fp16 == (layer < 3) &&
                    block.ff1_hopper_fp16 == (layer < 3),
                    "packed small batch requires loaded QKV/FF1 layout and unpacked final block");
        }
    }
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
    // Actual host-view observation only; not resolved kernel/tensor attestation.
    std::ofstream loaded_view("test_results/stream1_loaded_view.json");
    require(static_cast<bool>(loaded_view), "cannot create loaded view inventory");
    write_stream1_loaded_view_inventory(loaded_view, view_holder.view);
    loaded_view << '\n';
    loaded_view.close();
    require(static_cast<bool>(loaded_view), "cannot finish loaded view inventory");
    int shape_device = 0;
    cudaDeviceProp shape_properties{};
    BEAM_CUDA_CHECK(cudaGetDevice(&shape_device));
    BEAM_CUDA_CHECK(cudaGetDeviceProperties(&shape_properties, shape_device));
    std::ofstream shape("test_results/stream1_reference_execution_shape.json");
    require(static_cast<bool>(shape), "cannot create reference execution shape");
    write_stream1_execution_shape(shape, reference_count, reference_count, 1, shape_device,
        shape_properties.major * 10 + shape_properties.minor);
    shape << '\n';
    shape.close();
    require(static_cast<bool>(shape), "cannot finish reference execution shape");
    std::ofstream identity("test_results/stream1_reference_device_identity.json");
    require(static_cast<bool>(identity), "cannot create reference device identity");
    write_stream1_device_identity(identity, shape_device,
        shape_properties.major * 10 + shape_properties.minor, shape_properties.uuid.bytes);
    identity << '\n';
    identity.close();
    require(static_cast<bool>(identity), "cannot finish reference device identity");
    const std::uint64_t packed_probe_token_rows =
        static_cast<std::uint64_t>(reference_count) * view_holder.view.dims.padded_seq_len;
    if (verify_packed_small_batch) {
        require(packed_probe_token_rows > 0 && packed_probe_token_rows < 4096,
                "packed small batch must exercise below4096 actual padded token rows");
    }
    const Stream1TransformerScratchView scratch_view =
        stream1_weights::transformer_scratch_view(scratch_allocation);

    if (!runtime_final_cls_scores) set_final_cls_only(false);
    const char* score_mode = runtime_cuda_graph_scores ? "runtime_cuda_graph" :
        (runtime_final_cls_scores ? "runtime_final_cls" : "baseline");
    report << "- raw_score_mode=" << score_mode << "\n";
    report << "- production_quality_accepted=false\n";
    cudaStream_t inference_stream = nullptr;
    if (runtime_cuda_graph_scores) {
        BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&inference_stream, cudaStreamNonBlocking));
    }
    auto infer = [&]() { stream1_transformer_inference_cuda(
        d_frontier,
        d_parent_base,
        d_count,
        view_holder.view,
        scratch_view,
        d_score,
        reference_count,
        0U,
        inference_stream); };
    infer(); // Warm up kernel policies/resources before capture.
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    if (runtime_cuda_graph_scores) {
        std::vector<std::uint32_t> eager(reference_count * MOVE_COUNT), observed(eager.size());
        BEAM_CUDA_CHECK(cudaMemcpy(eager.data(), d_score, eager.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        BEAM_CUDA_CHECK(cudaStreamBeginCapture(inference_stream, cudaStreamCaptureModeThreadLocal));
        BEAM_CUDA_CHECK(cudaMemsetAsync(d_score, 0, reference_count * MOVE_COUNT * sizeof(std::uint32_t), inference_stream));
        infer();
        BEAM_CUDA_CHECK(cudaStreamEndCapture(inference_stream, &graph));
        BEAM_CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (unsigned replay = 0; replay < 3; ++replay) {
            std::vector<State128> replay_states(states.size());
            std::vector<unsigned> indices(states.size());
            for (unsigned row = 0; row < reference_count; ++row) {
                indices[row] = !verify_graph_input_reuse || replay == 2 ? row :
                    replay == 0 ? (row + 1) % reference_count : reference_count - row - 1;
                replay_states[row] = states[indices[row]];
            }
            if (verify_graph_input_reuse) {
                BEAM_CUDA_CHECK(cudaMemcpyAsync(d_frontier, replay_states.data(),
                    replay_states.size() * sizeof(State128), cudaMemcpyHostToDevice, inference_stream));
            }
            BEAM_CUDA_CHECK(cudaGraphLaunch(executable, inference_stream));
            BEAM_CUDA_CHECK(cudaStreamSynchronize(inference_stream));
            if (verify_graph_input_reuse) {
                BEAM_CUDA_CHECK(cudaMemcpy(observed.data(), d_score,
                    observed.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
                for (unsigned row = 0; row < reference_count; ++row)
                    for (unsigned move = 0; move < MOVE_COUNT; ++move)
                        require(observed[row * MOVE_COUNT + move] == eager[indices[row] * MOVE_COUNT + move],
                                "graph replay reused stale input or mismatched score row");
                report << "graph_input_reuse_replay=" << replay << " status=pass\n";
            }
        }
        BEAM_CUDA_CHECK(cudaGraphExecDestroy(executable));
        BEAM_CUDA_CHECK(cudaGraphDestroy(graph));
        BEAM_CUDA_CHECK(cudaStreamDestroy(inference_stream));
    }

    std::vector<std::uint32_t> cuda_scores(reference_count * MOVE_COUNT);
    BEAM_CUDA_CHECK(cudaMemcpy(cuda_scores.data(), d_score, cuda_scores.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));

    if (verify_frozen_policies) {
        // Mutating ambient policy after capture must not change an issued launch.
        const auto frozen = capture_stream1_transformer_launch_policies();
        const char* key = "BEAM_STREAM1_TRANSFORMER_FF1_POLICY";
#if defined(_WIN32)
        require(_putenv_s(key, "deliberately-invalid-after-capture") == 0, "mutate launch policy");
#else
        require(setenv(key, "deliberately-invalid-after-capture", 1) == 0, "mutate launch policy");
#endif
        {
            launch_stream1_transformer_chunks_cuda(d_frontier, d_parent_base, d_count,
                nullptr, false, view_holder.view, scratch_view, d_score,
                reference_count, reference_count, nullptr, Stream1NoChunkObserver{}, &frozen);
            BEAM_CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<std::uint32_t> actual(cuda_scores.size());
            BEAM_CUDA_CHECK(cudaMemcpy(actual.data(), d_score, actual.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
            require(actual == cuda_scores, "frozen launch changed all-row score keys");
            std::uint32_t* job = nullptr;
            BEAM_CUDA_CHECK(cudaMalloc(&job, sizeof(std::uint32_t)));
            BEAM_CUDA_CHECK(cudaMemset(job, 0, sizeof(std::uint32_t)));
            cudaStream_t frozen_stream;
            BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&frozen_stream, cudaStreamNonBlocking));
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            BEAM_CUDA_CHECK(cudaStreamBeginCapture(frozen_stream, cudaStreamCaptureModeThreadLocal));
            launch_stream1_transformer_chunks_cuda(d_frontier, d_parent_base, d_count,
                job, true, view_holder.view, scratch_view, d_score,
                reference_count, reference_count, frozen_stream, Stream1NoChunkObserver{}, &frozen);
            BEAM_CUDA_CHECK(cudaStreamEndCapture(frozen_stream, &graph));
            BEAM_CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            BEAM_CUDA_CHECK(cudaMemsetAsync(d_score, 0, actual.size() * sizeof(std::uint32_t), frozen_stream));
            BEAM_CUDA_CHECK(cudaGraphLaunch(executable, frozen_stream));
            BEAM_CUDA_CHECK(cudaStreamSynchronize(frozen_stream));
            BEAM_CUDA_CHECK(cudaMemcpy(actual.data(), d_score, actual.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
            require(actual == cuda_scores, "frozen windowed graph changed all-row score keys");
            BEAM_CUDA_CHECK(cudaGraphExecDestroy(executable));
            BEAM_CUDA_CHECK(cudaGraphDestroy(graph));
            BEAM_CUDA_CHECK(cudaStreamDestroy(frozen_stream));
            BEAM_CUDA_CHECK(cudaFree(job));
        }
        const auto* original = frozen.get(key);
#if defined(_WIN32)
        require(_putenv_s(key, original ? original : "") == 0, "restore launch policy");
#else
        require((original ? setenv(key, original, 1) : unsetenv(key)) == 0, "restore launch policy");
#endif
        report << "frozen_launch_policy=eager_and_windowed_graph_all_rows_pass\n";
    }

    // Preserve every raw FP16 score before clamping/quantization. Logits omit
    // the output bias; add it in float exactly as score_quantize_kernel does.
    require(host_weights.model.dtype == STREAM1_DTYPE_FP16, "full score fixture requires FP16");
    std::vector<half> raw_logits(cuda_scores.size()), raw_bias(MOVE_COUNT);
    BEAM_CUDA_CHECK(cudaMemcpy(raw_logits.data(), scratch_view.logits,
        raw_logits.size() * sizeof(half), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(raw_bias.data(), view_holder.view.output_bias,
        raw_bias.size() * sizeof(half), cudaMemcpyDeviceToHost));
    std::ofstream full_scores("test_results/stream1_full_scores.json");
    require(static_cast<bool>(full_scores), "cannot create full score artifact");
    full_scores << std::setprecision(9) << "{\"scores\":[";
    double packed_probe_max_abs_error = 0.;
    for (std::uint32_t row = 0; row < reference_count; ++row) {
        if (row) full_scores << ',';
        full_scores << '[';
        for (std::uint32_t move = 0; move < MOVE_COUNT; ++move) {
            if (move) full_scores << ',';
            const float score = __half2float(raw_logits[row * MOVE_COUNT + move]) + __half2float(raw_bias[move]);
            if (verify_packed_small_batch) {
                const double expected = reference_scores[row * MOVE_COUNT + move];
                packed_probe_max_abs_error = std::max(packed_probe_max_abs_error,
                    std::abs(static_cast<double>(score) - expected));
                require(std::isfinite(score) && std::isfinite(expected) &&
                        std::abs(static_cast<double>(score) - expected) <= 0.05 + 0.001 * std::abs(expected),
                        "packed small batch raw score differs from independent reference");
            }
            if (std::isfinite(score)) full_scores << score;
            else full_scores << "null"; // Invalid values remain visible and fail the full-tensor gate.
        }
        full_scores << ']';
    }
    full_scores << "]}\n";
    full_scores.close();
    require(static_cast<bool>(full_scores), "cannot finish full score artifact");
    std::ofstream execution("test_results/stream1_score_execution.json");
    require(static_cast<bool>(execution), "cannot create score execution receipt");
    execution << "{\"score_mode\":\""
              << score_mode
              << "\",\"rows\":" << reference_count
              << ",\"microbatch\":" << reference_count << ",\"lane\":0,\"cls_environment\":{"
              << "\"BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY\":\""
              << observed_cls_flag("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY") << "\","
              << "\"BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION\":\""
              << observed_cls_flag("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION") << "\","
              << "\"BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV\":\""
              << observed_cls_flag("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV") << "\"},"
              << "\"fused_input_layernorm\":"
              << (observed_cls_flag("BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM")[0] == '1'
                  ? "true" : "false")
              << ",\"compact57\":"
              << (view_holder.view.dims.seq_len == 57U && view_holder.view.dims.padded_seq_len == 57U
                  ? "true" : "false") << "}\n";
    execution.close();
    require(static_cast<bool>(execution), "cannot finish score execution receipt");

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
    if (generic_final_cls_shape && !runtime_final_cls_scores) {
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
    if (!runtime_final_cls_scores) set_final_cls_only(false);

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
    if (verify_graph_lanes)
        verify_stream1_graph_lanes(host_weights.model, view_holder.view, d_frontier, cuda_scores, report);
    if (verify_production_lanes)
        verify_stream1_graph_lanes(host_weights.model, view_holder.view, d_frontier, cuda_scores, report, true,
                                  probe_micro, probe_lanes, probe_inner);
    if (verify_graph_job_bounds) {
        // Exercise the dispatcher entry point, two score slots and device job
        // metadata. Keep independent ordinary-entry score rows as the oracle.
        std::uint64_t* bases = nullptr;
        std::uint32_t *counts = nullptr, *job = nullptr, *scores = nullptr;
        BEAM_CUDA_CHECK(cudaMalloc(&bases, 2 * sizeof(std::uint64_t)));
        BEAM_CUDA_CHECK(cudaMalloc(&counts, 2 * sizeof(std::uint32_t)));
        BEAM_CUDA_CHECK(cudaMalloc(&job, sizeof(std::uint32_t)));
        const std::size_t score_bytes = 2 * reference_count * MOVE_COUNT * sizeof(std::uint32_t);
        BEAM_CUDA_CHECK(cudaMalloc(&scores, score_bytes));
        cudaStream_t probe_stream;
        BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&probe_stream, cudaStreamNonBlocking));
        const std::uint64_t initial_bases[2]{0, 1};
        const std::uint32_t initial_counts[2]{8, 4}, initial_job = 1;
        BEAM_CUDA_CHECK(cudaMemcpy(bases, initial_bases, sizeof(initial_bases), cudaMemcpyHostToDevice));
        BEAM_CUDA_CHECK(cudaMemcpy(counts, initial_counts, sizeof(initial_counts), cudaMemcpyHostToDevice));
        BEAM_CUDA_CHECK(cudaMemcpy(job, &initial_job, sizeof(initial_job), cudaMemcpyHostToDevice));
        auto job_infer = [&]() { stream1_transformer_inference_graph_job_cuda(
            d_frontier, bases, counts, job, view_holder.view, scratch_view,
            scores, reference_count, reference_count, 1U, probe_stream); };
        BEAM_CUDA_CHECK(cudaMemset(scores, 0, score_bytes));
        job_infer();
        BEAM_CUDA_CHECK(cudaStreamSynchronize(probe_stream));
        cudaGraph_t probe_graph;
        cudaGraphExec_t probe_exec;
        BEAM_CUDA_CHECK(cudaStreamBeginCapture(probe_stream, cudaStreamCaptureModeThreadLocal));
        BEAM_CUDA_CHECK(cudaMemsetAsync(scores, 0, score_bytes, probe_stream));
        job_infer();
        BEAM_CUDA_CHECK(cudaStreamEndCapture(probe_stream, &probe_graph));
        BEAM_CUDA_CHECK(cudaGraphInstantiate(&probe_exec, probe_graph, nullptr, nullptr, 0));
        std::vector<std::uint32_t> actual(2 * reference_count * MOVE_COUNT);
        for (unsigned test = 0; test < 3; ++test) {
            const std::uint32_t selected = test == 0 ? 1 : 0;
            const std::uint32_t host_counts[2]{test == 1 ? 0U : 8U, 4U};
            BEAM_CUDA_CHECK(cudaMemcpyAsync(counts, host_counts, sizeof(host_counts), cudaMemcpyHostToDevice, probe_stream));
            BEAM_CUDA_CHECK(cudaMemcpyAsync(job, &selected, sizeof(selected), cudaMemcpyHostToDevice, probe_stream));
            BEAM_CUDA_CHECK(cudaGraphLaunch(probe_exec, probe_stream));
            BEAM_CUDA_CHECK(cudaStreamSynchronize(probe_stream));
            BEAM_CUDA_CHECK(cudaMemcpy(actual.data(), scores, score_bytes, cudaMemcpyDeviceToHost));
            const unsigned active = host_counts[selected] == 0 ? 0 : host_counts[selected] - 1;
            for (unsigned slot = 0; slot < 2; ++slot)
                for (unsigned row = 0; row < reference_count; ++row)
                    for (unsigned move = 0; move < MOVE_COUNT; ++move) {
                        const auto expected = slot == selected && row >= 1 && row <= active ?
                            cuda_scores[(initial_bases[selected] + row) * MOVE_COUNT + move] : 0U;
                        require(actual[(slot * reference_count + row) * MOVE_COUNT + move] == expected,
                                "graph job bounds/slot/count/parent-base mismatch");
                    }
            report << "graph_job_bounds_case=" << test << " status=pass\n";
        }
        BEAM_CUDA_CHECK(cudaGraphExecDestroy(probe_exec));
        BEAM_CUDA_CHECK(cudaGraphDestroy(probe_graph));
        BEAM_CUDA_CHECK(cudaStreamDestroy(probe_stream));
        cudaFree(bases); cudaFree(counts); cudaFree(job); cudaFree(scores);
    }
    if (verify_packed_small_batch) {
        std::ofstream packed_report("test_results/packed_small_probe.json");
        require(static_cast<bool>(packed_report), "cannot create packed probe receipt");
        packed_report << std::setprecision(9)
            << "{\"scope\":\"eight-row packed QKV/FF1 raw-score diagnostic, not production admission\","
            << "\"status\":\"pass\",\"graph_mode\":true,\"rows\":" << reference_count
            << ",\"padded_token_rows\":" << packed_probe_token_rows
            << ",\"compared_scores\":" << reference_count * MOVE_COUNT
            << ",\"max_abs_error\":" << packed_probe_max_abs_error
            << ",\"packed_layers\":[0,1,2],\"unpacked_final_layer\":3,"
            << "\"full_execution_identity_attested\":false}\n";
        packed_report.close();
        require(static_cast<bool>(packed_report), "cannot finish packed probe receipt");
    }
    report << "- transformer_forward_reference=pass\n";
    report << "- standalone_transformer_forward=pass\n";
    report << "\nstatus=pass\n";

    require(!actual_kernel_observation.records().empty(), "no actual linear+bias kernel observations");
    nlohmann::json kernel_records=nlohmann::json::array();
    for (const auto& d:actual_kernel_observation.records()) {
        kernel_records.push_back({{"family",d.family},{"implementation",d.implementation},
            {"dtype",d.dtype},{"epilogue",d.epilogue},{"architecture",d.architecture},
            {"rows",d.rows},{"input_cols",d.input_cols},{"output_cols",d.output_cols},
            {"tile",{d.tile_m,d.tile_n,d.tile_k}},{"warp",{d.warp_m,d.warp_n,d.warp_k}},
            {"instruction",{d.instruction_m,d.instruction_n,d.instruction_k}},
            {"stages",d.stages},{"swizzle",d.swizzle},{"packed_weight",d.packed_weight}});
    }
    std::ofstream kernel_receipt("test_results/stream1_kernel_observations.json");
    require(static_cast<bool>(kernel_receipt), "cannot create actual kernel observations");
    kernel_receipt << nlohmann::json({{"schema_version",1},
        {"scope","host issued linear_bias launches, including graph capture; not GPU completion"},
        {"coverage_complete",false},{"production_quality_accepted",false},
        {"records",kernel_records}}).dump(2) << '\n';
    kernel_receipt.close();
    require(static_cast<bool>(kernel_receipt), "cannot finish actual kernel observations");
    cudaFree(d_frontier);
    cudaFree(d_parent_base);
    cudaFree(d_count);
    cudaFree(d_score);
    stream1_weights::free_stream1_scratch(scratch_allocation);
    stream1_weights::free_weights(device_weights);
    std::cout << "stream1_transformer_cuda_tests=pass\n";
    return 0;
}
