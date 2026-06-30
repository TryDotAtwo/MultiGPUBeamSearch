#include "stream_benchmark_common.hpp"

namespace beam::bench {

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

    const char* only_b_micro_env = std::getenv("BEAM_STREAM1_TRANSFORMER_B_MICRO");
    const char* only_concurrency_env = std::getenv("BEAM_STREAM1_TRANSFORMER_CONCURRENCY");
    const std::uint32_t only_b_micro = only_b_micro_env != nullptr && only_b_micro_env[0] != '\0'
        ? static_cast<std::uint32_t>(parse_u64(only_b_micro_env, "BEAM_STREAM1_TRANSFORMER_B_MICRO"))
        : 0U;
    const std::uint32_t only_concurrency = only_concurrency_env != nullptr && only_concurrency_env[0] != '\0'
        ? static_cast<std::uint32_t>(parse_u64(only_concurrency_env, "BEAM_STREAM1_TRANSFORMER_CONCURRENCY"))
        : 0U;

    report << "## Stream1 Piece Transformer\n\n";
    if (only_b_micro != 0U || only_concurrency != 0U) {
        report << "- filter_b_micro=" << only_b_micro << "\n";
        report << "- filter_concurrency=" << only_concurrency << "\n\n";
    }
    report << "| b_micro | concurrency | rows_per_launch_group | ms_per_launch_group | parents_per_sec | candidates_per_sec | scratch_bytes |\n";
    report << "|---:|---:|---:|---:|---:|---:|---:|\n";
    for (std::uint32_t b_micro : TRANSFORMER_B_MICRO_SWEEP) {
        for (std::uint32_t concurrent : TRANSFORMER_STREAM1_CONCURRENCY_SWEEP) {
            if ((only_b_micro != 0U && b_micro != only_b_micro) ||
                (only_concurrency != 0U && concurrent != only_concurrency)) {
                continue;
            }
            const std::uint64_t rows_per_launch_group = static_cast<std::uint64_t>(b_micro) * concurrent;
            if (rows_per_launch_group > max_states) {
                report << "|" << b_micro << "|" << concurrent << "|" << rows_per_launch_group
                       << "|skip|skip|skip|0: exceeds prepared state batch|\n";
                std::cout << "stream1_transformer_micro_skip"
                          << " b_micro=" << b_micro
                          << " concurrency=" << concurrent
                          << " reason=exceeds_prepared_state_batch\n";
                continue;
            }

            const std::uint64_t scratch_bytes =
                stream1_weights::stream1_scratch_bytes(model, b_micro, concurrent);
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
                       << "|skip|skip|skip|" << scratch_bytes
                       << ": estimated allocation exceeds available GPU memory"
                       << " free_bytes=" << free_bytes
                       << " io_bytes=" << io_bytes << "|\n";
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
            const float ms = time_gpu_ms(resources.streams, iterations, [&]() {
                for (std::uint32_t i = 0; i < concurrent; ++i) {
                    stream1_transformer_inference_cuda(
                        states,
                        resources.parent_base[i],
                        resources.count[i],
                        view_holder.view,
                        scratch_views[i],
                        resources.score[i],
                        b_micro,
                        resources.streams[i]);
                }
            });
            const double parents = static_cast<double>(rows_per_launch_group);
            const double parent_per_sec = parents * 1000.0 / static_cast<double>(ms);
            const double candidate_per_sec = parents * static_cast<double>(MOVE_COUNT) * 1000.0 / static_cast<double>(ms);
            results.push_back(Stream1Result{b_micro, concurrent, ms, parent_per_sec, candidate_per_sec});
            report << "|" << b_micro
                   << "|" << concurrent
                   << "|" << rows_per_launch_group
                   << "|" << std::fixed << std::setprecision(4) << ms
                   << "|" << std::setprecision(1) << parent_per_sec
                   << "|" << candidate_per_sec
                   << "|" << scratch_bytes << "|\n";
            std::cout << "stream1_transformer_micro"
                      << " b_micro=" << b_micro
                      << " concurrency=" << concurrent
                      << " rows_per_launch_group=" << rows_per_launch_group
                      << " ms_per_launch_group=" << std::fixed << std::setprecision(4) << ms
                      << " parents_per_sec=" << std::setprecision(1) << parent_per_sec
                      << " candidates_per_sec=" << candidate_per_sec
                      << " scratch_bytes=" << scratch_bytes
                      << "\n";
        }
    }
    report << "\n";
    return results;
}


} // namespace beam::bench
