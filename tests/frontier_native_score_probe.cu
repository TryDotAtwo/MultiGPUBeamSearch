// Explicit Cube4 diagnostic: native graph-job scores for a saved rank frontier.
#include "../tools/stream1_weight_io.hpp"
#include "../cuda/stream1.hpp"
#include "cuda_check.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
using namespace beam;

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    try {
        static_assert(STATE_LEN == 96 && sizeof(State128) == 112, "Cube4 probe specialization required");
        const auto bytes = std::filesystem::file_size(argv[2]);
        if (bytes == 0 || bytes % sizeof(State128) || bytes / sizeof(State128) > 1048576 ||
            std::filesystem::exists(argv[3])) throw std::runtime_error("invalid probe input/output");
        std::vector<State128> states(bytes / sizeof(State128));
        std::ifstream input(argv[2], std::ios::binary);
        input.read(reinterpret_cast<char*>(states.data()), bytes);
        if (!input) throw std::runtime_error("short frontier read");
        for (const auto& state : states) {
            for (unsigned i = 0; i < STATE_LEN; ++i)
                if (state.v[i] >= 6) throw std::runtime_error("invalid Cube4 state");
            for (unsigned i = STATE_LEN; i < sizeof(State128); ++i)
                if (state.v[i]) throw std::runtime_error("nonzero frontier padding");
        }
        constexpr unsigned micro = 32;
        auto host = stream1_weights::load_stream1_weights(argv[1]);
        if (host.model.backend != STREAM1_BACKEND_PIECE_TRANSFORMER)
            throw std::runtime_error("probe requires piece transformer weights");
        auto weights = stream1_weights::upload_weights(host);
        auto scratch = stream1_weights::alloc_stream1_scratch(host.model, micro, 1);
        auto network = stream1_weights::transformer_network_view(weights.transformer, host.model);
        auto view = stream1_weights::transformer_scratch_view(scratch, host.model, micro, 0);
        State128* frontier; std::uint64_t* base; std::uint32_t *count, *job, *scores;
        BEAM_CUDA_CHECK(cudaMalloc(&frontier, micro * sizeof(State128)));
        BEAM_CUDA_CHECK(cudaMalloc(&base, sizeof(*base)));
        BEAM_CUDA_CHECK(cudaMalloc(&count, sizeof(*count)));
        BEAM_CUDA_CHECK(cudaMalloc(&job, sizeof(*job)));
        BEAM_CUDA_CHECK(cudaMalloc(&scores, micro * MOVE_COUNT * sizeof(*scores)));
        BEAM_CUDA_CHECK(cudaMemset(base, 0, sizeof(*base)));
        BEAM_CUDA_CHECK(cudaMemset(job, 0, sizeof(*job)));
        unsigned active = std::min<std::size_t>(micro, states.size());
        BEAM_CUDA_CHECK(cudaMemcpy(frontier, states.data(), active * sizeof(State128), cudaMemcpyHostToDevice));
        BEAM_CUDA_CHECK(cudaMemcpy(count, &active, sizeof(active), cudaMemcpyHostToDevice));
        cudaStream_t stream; cudaGraph_t graph; cudaGraphExec_t executable;
        BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        auto infer = [&] { stream1_transformer_inference_graph_job_cuda(frontier, base, count, job,
            network.view, view, scores, micro, micro, 0, stream); };
        infer(); BEAM_CUDA_CHECK(cudaStreamSynchronize(stream));
        BEAM_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        BEAM_CUDA_CHECK(cudaMemsetAsync(scores, 0, micro * MOVE_COUNT * sizeof(*scores), stream));
        infer();
        BEAM_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        BEAM_CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        std::ofstream output(argv[3], std::ios::binary);
        std::vector<std::uint32_t> observed(micro * MOVE_COUNT);
        for (std::size_t offset = 0; offset < states.size(); offset += micro) {
            active = std::min<std::size_t>(micro, states.size() - offset);
            BEAM_CUDA_CHECK(cudaMemcpy(frontier, states.data() + offset, active * sizeof(State128), cudaMemcpyHostToDevice));
            BEAM_CUDA_CHECK(cudaMemcpy(count, &active, sizeof(active), cudaMemcpyHostToDevice));
            BEAM_CUDA_CHECK(cudaGraphLaunch(executable, stream));
            BEAM_CUDA_CHECK(cudaStreamSynchronize(stream));
            BEAM_CUDA_CHECK(cudaMemcpy(observed.data(), scores, observed.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
            for (unsigned i = active * MOVE_COUNT; i < observed.size(); ++i)
                if (observed[i]) throw std::runtime_error("nonzero inactive score tail");
            output.write(reinterpret_cast<const char*>(observed.data()), active * MOVE_COUNT * sizeof(std::uint32_t));
        }
        output.close(); if (!output) throw std::runtime_error("score write failed");
        cudaGraphExecDestroy(executable); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
        cudaFree(frontier); cudaFree(base); cudaFree(count); cudaFree(job); cudaFree(scores);
        stream1_weights::free_stream1_scratch(scratch); stream1_weights::free_weights(weights);
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
