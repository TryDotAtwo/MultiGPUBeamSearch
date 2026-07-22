#include <sstream>
#include <cstring>
#include <fstream>
#include "stream1_transformer_score_dump.hpp"
#include "stream_benchmark_common.hpp"

namespace beam::bench {

constexpr std::uint64_t FNV64_OFFSET = 1469598103934665603ULL;
constexpr std::uint64_t FNV64_PRIME = 1099511628211ULL;

struct ScoreSummary {
    std::uint64_t checksum = 0;
    std::uint64_t score_key_digest = FNV64_OFFSET;
    std::string first_score_keys;
};

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "1") == 0;
}
std::uint64_t estimate_transformer_dense_flops_per_parent(const Stream1ModelConfig& model) {
    const std::uint64_t s = model.seq_len;
    const std::uint64_t d = model.d_model;
    const std::uint64_t ff = model.ff_dim;
    const std::uint64_t out = model.output_dim;
    const std::uint64_t full_layer =
        6ULL * s * d * d +
        2ULL * s * d * d +
        2ULL * s * d * ff +
        2ULL * s * ff * d +
        4ULL * s * s * d;
    const bool block51_final_cls =
        env_flag_enabled("BEAM_STREAM1_TRANSFORMER_BLOCK51") &&
        env_flag_enabled("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY") &&
        model.seq_len == 51U && model.d_model == 256U &&
        model.nhead == 8U && model.head_dim == 32U &&
        model.transformer_layers == 4U && model.ff_dim == 1024U;
    if (block51_final_cls) {
        const std::uint64_t final_attention = env_flag_enabled("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION")
            ? 4ULL * s * d
            : 4ULL * s * s * d;
        const std::uint64_t final_layer =
            6ULL * s * d * d +
            final_attention +
            2ULL * d * d +
            2ULL * d * ff +
            2ULL * ff * d +
            2ULL * d * out;
        return 3ULL * full_layer + final_layer;
    }
    return static_cast<std::uint64_t>(model.transformer_layers) * full_layer + 2ULL * d * out;
}


ScoreSummary summarize_score_buffers(
    const std::vector<std::uint32_t*>& score_buffers,
    std::uint32_t b_micro,
    std::uint32_t concurrent) {
    std::ofstream score_dump;
    if (const char* dump_path = std::getenv("BEAM_STREAM1_TRANSFORMER_SCORE_DUMP");
        dump_path != nullptr && dump_path[0] != '\0') {
        score_dump.open(dump_path, std::ios::binary | std::ios::trunc);
        if (!score_dump) {
            throw std::runtime_error("failed to open Stream1 transformer score dump");
        }
        const Stream1TransformerScoreDumpHeader header = make_stream1_transformer_score_dump_header(
            concurrent, static_cast<std::uint64_t>(b_micro) * MOVE_COUNT);
        score_dump.write(reinterpret_cast<const char*>(&header), sizeof(header));
        if (!score_dump) {
            throw std::runtime_error("failed to write Stream1 transformer score dump header");
        }
    }
    ScoreSummary summary;
    std::vector<std::uint32_t> host(static_cast<std::uint64_t>(b_micro) * MOVE_COUNT);
    for (std::uint32_t lane = 0; lane < concurrent; ++lane) {
        BEAM_CUDA_CHECK(cudaMemcpy(
            host.data(),
            score_buffers[lane],
            host.size() * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost));
        if (score_dump.is_open()) {
            score_dump.write(
                reinterpret_cast<const char*>(host.data()),
                static_cast<std::streamsize>(host.size() * sizeof(std::uint32_t)));
            if (!score_dump) {
                throw std::runtime_error("failed to write Stream1 transformer score dump payload");
            }
        }
        for (std::uint32_t value : host) {
            summary.checksum += value;
            for (int shift = 0; shift < 32; shift += 8) {
                summary.score_key_digest ^= static_cast<std::uint64_t>((value >> shift) & 0xFFU);
                summary.score_key_digest *= FNV64_PRIME;
            }
        }
        if (lane == 0 && !host.empty()) {
            std::ostringstream out;
            for (std::uint32_t move = 0; move < MOVE_COUNT; ++move) {
                if (move != 0) {
                    out << ',';
                }
                out << host[move];
            }
            summary.first_score_keys = out.str();
        }
    }
    return summary;
}

struct Stream1TransformerBenchmarkResources {
    std::vector<cudaStream_t> streams;
    stream1_weights::ScratchAllocation scratch{};
    std::vector<std::uint64_t*> parent_base;
    std::vector<std::uint32_t*> count;
    std::vector<std::uint32_t*> score;

    Stream1TransformerBenchmarkResources() = default;
    Stream1TransformerBenchmarkResources(const Stream1TransformerBenchmarkResources&) = delete;
    Stream1TransformerBenchmarkResources& operator=(const Stream1TransformerBenchmarkResources&) = delete;

    ~Stream1TransformerBenchmarkResources() {
        cleanup();
    }

    void create_streams_checked(std::uint32_t stream_count) {
        streams.reserve(stream_count);
        for (std::uint32_t i = 0; i < stream_count; ++i) {
            cudaStream_t stream = nullptr;
            BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            streams.push_back(stream);
        }
    }

    void allocate(
        const Stream1ModelConfig& model,
        std::uint32_t b_micro,
        std::uint32_t concurrent) {
        scratch = stream1_weights::alloc_stream1_scratch(model, b_micro, concurrent);
        parent_base.assign(concurrent, nullptr);
        count.assign(concurrent, nullptr);
        score.assign(concurrent, nullptr);
        for (std::uint32_t i = 0; i < concurrent; ++i) {
            parent_base[i] = device_alloc<std::uint64_t>(1);
            count[i] = device_alloc<std::uint32_t>(1);
            score[i] = device_alloc<std::uint32_t>(static_cast<std::uint64_t>(b_micro) * MOVE_COUNT);
        }
    }

    void cleanup() noexcept {
        cudaDeviceSynchronize();
        for (std::uint64_t* ptr : parent_base) {
            cudaFree(ptr);
        }
        for (std::uint32_t* ptr : count) {
            cudaFree(ptr);
        }
        for (std::uint32_t* ptr : score) {
            cudaFree(ptr);
        }
        parent_base.clear();
        count.clear();
        score.clear();
        stream1_weights::free_stream1_scratch(scratch);
        destroy_streams(streams);
    }
};

float time_transformer_graph_ms(
    const std::vector<cudaStream_t>& streams,
    std::uint32_t iterations,
    const std::vector<cudaGraphExec_t>& execs) {
    return time_gpu_ms(streams, iterations, [&]() {
        for (std::uint32_t i = 0; i < execs.size(); ++i) {
            BEAM_CUDA_CHECK(cudaGraphLaunch(execs[i], streams[i]));
        }
    });
}
std::vector<Stream1Result> benchmark_stream1_transformer(
    const stream1_weights::DeviceWeights& weights,
    const Stream1ModelConfig& model,
    const State128* states,
    std::uint32_t max_states,
    std::ofstream& report) {
    if (model.backend != STREAM1_BACKEND_PIECE_TRANSFORMER) {
        throw std::runtime_error("benchmark_stream1_transformer requires piece_transformer backend");
    }
    std::vector<Stream1Result> results;
    stream1_weights::TransformerNetworkViewHolder view_holder =
        stream1_weights::transformer_network_view(weights.transformer, model);

    const bool graph_bench = std::getenv("BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH") != nullptr;
    const char* only_b_micro_env = std::getenv("BEAM_STREAM1_TRANSFORMER_B_MICRO");
    const char* only_concurrency_env = std::getenv("BEAM_STREAM1_TRANSFORMER_CONCURRENCY");
    const std::uint32_t only_b_micro = only_b_micro_env != nullptr && only_b_micro_env[0] != '\0'
        ? static_cast<std::uint32_t>(parse_u64(only_b_micro_env, "BEAM_STREAM1_TRANSFORMER_B_MICRO"))
        : 0U;
    const std::uint32_t only_concurrency = only_concurrency_env != nullptr && only_concurrency_env[0] != '\0'
        ? static_cast<std::uint32_t>(parse_u64(only_concurrency_env, "BEAM_STREAM1_TRANSFORMER_CONCURRENCY"))
        : 0U;

    report << "## Stream1 Piece Transformer\n\n";
    report << "- graph_bench=" << (graph_bench ? 1 : 0) << "\n";
    if (only_b_micro != 0U || only_concurrency != 0U) {
        report << "- filter_b_micro=" << only_b_micro << "\n";
        report << "- filter_concurrency=" << only_concurrency << "\n\n";
    }
    report << "| b_micro | concurrency | rows_per_launch_group | ms_per_launch_group | parents_per_sec | candidates_per_sec | estimated_flops_per_parent | achieved_tflops | checksum | score_key_digest | first_score_keys | scratch_bytes | token_bytes | qkv_bytes | attention_bytes | context_bytes | ff_hidden_bytes | logits_bytes |\n";
    report << "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|\n";
    for (std::uint32_t b_micro : TRANSFORMER_B_MICRO_SWEEP) {
        for (std::uint32_t concurrent : TRANSFORMER_STREAM1_CONCURRENCY_SWEEP) {
            if ((only_b_micro != 0U && b_micro != only_b_micro) ||
                (only_concurrency != 0U && concurrent != only_concurrency)) {
                continue;
            }
            const std::uint64_t rows_per_launch_group = static_cast<std::uint64_t>(b_micro) * concurrent;
            if (rows_per_launch_group > max_states) {
                report << "|" << b_micro << "|" << concurrent << "|" << rows_per_launch_group
                       << "|skip|skip|skip|skip|skip|skip|0: exceeds prepared state batch|skip|skip|skip|skip|skip|skip|\n";
                std::cout << "stream1_transformer_micro_skip"
                          << " b_micro=" << b_micro
                          << " concurrency=" << concurrent
                          << " reason=exceeds_prepared_state_batch\n";
                continue;
            }

            const stream1_weights::TransformerScratchBytePlan scratch_plan =
                stream1_weights::transformer_scratch_byte_plan(model, rows_per_launch_group);
            const std::uint64_t scratch_bytes = scratch_plan.total_bytes();
            const std::uint64_t io_bytes =
                rows_per_launch_group *
                (sizeof(std::uint64_t) + sizeof(std::uint32_t) +
                 static_cast<std::uint64_t>(MOVE_COUNT) * sizeof(std::uint32_t));
            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;
            BEAM_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            constexpr std::uint64_t safety_margin_bytes = 256ULL * 1024ULL * 1024ULL;
            if (scratch_bytes + io_bytes + safety_margin_bytes > free_bytes) {
                report << "|" << b_micro << "|" << concurrent << "|" << rows_per_launch_group
                       << "|skip|skip|skip|skip|skip|skip|" << scratch_bytes
                       << ": estimated allocation exceeds available GPU memory"
                       << " free_bytes=" << free_bytes
                       << " io_bytes=" << io_bytes << "|skip|skip|skip|skip|skip|skip|\n";
                std::cout << "stream1_transformer_micro_skip"
                          << " b_micro=" << b_micro
                          << " concurrency=" << concurrent
                          << " scratch_bytes=" << scratch_bytes
                          << " free_bytes=" << free_bytes
                          << " reason=estimated_allocation_exceeds_available_memory\n";
                continue;
            }

            Stream1TransformerBenchmarkResources resources;
            resources.create_streams_checked(concurrent);
            resources.allocate(model, b_micro, concurrent);
            std::vector<Stream1TransformerScratchView> scratch_views;
            scratch_views.reserve(concurrent);
            for (std::uint32_t i = 0; i < concurrent; ++i) {
                scratch_views.push_back(stream1_weights::transformer_scratch_view(resources.scratch, model, b_micro, i));
                const std::uint64_t base = static_cast<std::uint64_t>(i) * b_micro;
                BEAM_CUDA_CHECK(cudaMemcpy(resources.parent_base[i], &base, sizeof(base), cudaMemcpyHostToDevice));
                BEAM_CUDA_CHECK(cudaMemcpy(resources.count[i], &b_micro, sizeof(b_micro), cudaMemcpyHostToDevice));
            }


            const std::uint32_t iterations = b_micro >= 4096 ? 4U : 6U;
            float ms = 0.0f;
            if (graph_bench) {
                std::vector<cudaGraph_t> graphs(concurrent, nullptr);
                std::vector<cudaGraphExec_t> execs(concurrent, nullptr);
                for (std::uint32_t i = 0; i < concurrent; ++i) {
                    BEAM_CUDA_CHECK(cudaStreamBeginCapture(resources.streams[i], cudaStreamCaptureModeGlobal));
                    stream1_transformer_inference_cuda(
                        states,
                        resources.parent_base[i],
                        resources.count[i],
                        view_holder.view,
                        scratch_views[i],
                        resources.score[i],
                        b_micro,
                        0U,
                        resources.streams[i]);
                    BEAM_CUDA_CHECK(cudaStreamEndCapture(resources.streams[i], &graphs[i]));
                    BEAM_CUDA_CHECK(cudaGraphInstantiate(&execs[i], graphs[i], nullptr, nullptr, 0));
                }
                ms = time_transformer_graph_ms(resources.streams, iterations, execs);
                for (cudaGraphExec_t exec : execs) {
                    cudaGraphExecDestroy(exec);
                }
                for (cudaGraph_t graph : graphs) {
                    cudaGraphDestroy(graph);
                }
            } else {
                ms = time_gpu_ms(resources.streams, iterations, [&]() {
                    for (std::uint32_t i = 0; i < concurrent; ++i) {
                        stream1_transformer_inference_cuda(
                            states,
                            resources.parent_base[i],
                            resources.count[i],
                            view_holder.view,
                            scratch_views[i],
                            resources.score[i],
                            b_micro,
                            0U,
                            resources.streams[i]);
                    }
                });
            }
            const ScoreSummary score_summary = summarize_score_buffers(resources.score, b_micro, concurrent);
            const double parents = static_cast<double>(rows_per_launch_group);
            const double parent_per_sec = parents * 1000.0 / static_cast<double>(ms);
            const double candidate_per_sec = parents * static_cast<double>(MOVE_COUNT) * 1000.0 / static_cast<double>(ms);
            const std::uint64_t flops_per_parent = estimate_transformer_dense_flops_per_parent(model);
            const double achieved_tflops = parent_per_sec * static_cast<double>(flops_per_parent) / 1.0e12;
            results.push_back(Stream1Result{b_micro, concurrent, ms, parent_per_sec, candidate_per_sec});
            report << "|" << b_micro
                   << "|" << concurrent
                   << "|" << rows_per_launch_group
                   << "|" << std::fixed << std::setprecision(4) << ms
                   << "|" << std::setprecision(1) << parent_per_sec
                   << "|" << candidate_per_sec
                   << "|" << flops_per_parent
                   << "|" << std::setprecision(3) << achieved_tflops
                   << "|" << score_summary.checksum
                   << "|" << score_summary.score_key_digest
                   << "|`" << score_summary.first_score_keys << "`"
                   << "|" << scratch_bytes
                   << "|" << scratch_plan.token_bytes
                   << "|" << scratch_plan.qkv_bytes
                   << "|" << scratch_plan.attention_bytes
                   << "|" << scratch_plan.context_bytes
                   << "|" << scratch_plan.ff_hidden_bytes
                   << "|" << scratch_plan.logits_bytes << "|\n";
            std::cout << "stream1_transformer_micro"
                      << " b_micro=" << b_micro
                      << " concurrency=" << concurrent
                      << " rows_per_launch_group=" << rows_per_launch_group
                      << " ms_per_launch_group=" << std::fixed << std::setprecision(4) << ms
                      << " parents_per_sec=" << std::setprecision(1) << parent_per_sec
                      << " candidates_per_sec=" << candidate_per_sec
                      << " estimated_flops_per_parent=" << flops_per_parent
                      << " achieved_tflops=" << std::setprecision(3) << achieved_tflops
                      << " checksum=" << score_summary.checksum
                      << " score_key_digest=" << score_summary.score_key_digest
                      << " first_score_keys=" << score_summary.first_score_keys
                      << " scratch_bytes=" << scratch_bytes
                      << " token_bytes=" << scratch_plan.token_bytes
                      << " qkv_bytes=" << scratch_plan.qkv_bytes
                      << " attention_bytes=" << scratch_plan.attention_bytes
                      << " context_bytes=" << scratch_plan.context_bytes
                      << " ff_hidden_bytes=" << scratch_plan.ff_hidden_bytes
                      << " logits_bytes=" << scratch_plan.logits_bytes
                      << "\n";
        }
    }
    report << "\n";
    return results;
}


} // namespace beam::bench
