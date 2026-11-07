#pragma once
#include "../cuda/stream1_transformer_chunk_launch.hpp"
inline bool stream1_graph_probe_score_matches(
    std::uint32_t expected, std::uint32_t actual, bool production_size) {
    if (!production_size) return expected == actual;
    // Different GEMM row counts can round differently. Raw FP32 reference
    // comparison remains mandatory in preflight; this is only a key guard.
    const std::uint32_t error = expected > actual ? expected - actual : actual - expected;
    return error <= 3072U;
}
// Test-only helper: production scratch allocation/slicing and graph-job entry.
inline void verify_stream1_graph_lanes(
    const beam::Stream1ModelConfig& model,
    const beam::Stream1TransformerNetworkView& network,
    const beam::State128* frontier,
    const std::vector<std::uint32_t>& oracle,
    std::ostream& report, bool production_size = false,
    std::uint32_t requested_micro = 1024, std::uint32_t requested_lanes = 4,
    std::uint32_t requested_inner = 0) {
    using namespace beam;
    const std::uint32_t micro = production_size ? requested_micro : 8;
    const std::uint32_t lanes = production_size ? requested_lanes : 2;
    const std::uint32_t inner = requested_inner ? requested_inner : micro;
    if (!micro || !lanes || !inner || inner > micro)
        throw std::invalid_argument("invalid diagnostic chunk dimensions");
    State128* expanded_frontier = nullptr;
    if (production_size) {
        std::vector<State128> fixture(8), expanded(micro);
        BEAM_CUDA_CHECK(cudaMemcpy(fixture.data(), frontier, fixture.size() * sizeof(State128), cudaMemcpyDeviceToHost));
        for (unsigned row = 0; row < micro; ++row) expanded[row] = fixture[row % fixture.size()];
        BEAM_CUDA_CHECK(cudaMalloc(&expanded_frontier, expanded.size() * sizeof(State128)));
        BEAM_CUDA_CHECK(cudaMemcpy(expanded_frontier, expanded.data(), expanded.size() * sizeof(State128), cudaMemcpyHostToDevice));
        frontier = expanded_frontier;
    }
    auto scratch = stream1_weights::alloc_stream1_scratch(model, micro, lanes);
    std::uint64_t* bases = nullptr;
    std::uint32_t *counts = nullptr, *jobs = nullptr, *scores = nullptr;
    BEAM_CUDA_CHECK(cudaMalloc(&bases, lanes * sizeof(std::uint64_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&counts, lanes * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&jobs, lanes * sizeof(std::uint32_t)));
    const std::size_t lane_bytes = micro * MOVE_COUNT * sizeof(std::uint32_t);
    BEAM_CUDA_CHECK(cudaMalloc(&scores, lanes * lane_bytes));
    // Diagnostic-only storage: scratch logits are overwritten by the next chunk.
    half* saved_logits = nullptr;
    BEAM_CUDA_CHECK(cudaMalloc(&saved_logits,
        static_cast<std::size_t>(lanes) * micro * MOVE_COUNT * sizeof(half)));
    std::vector<std::uint64_t> host_bases(lanes);
    std::vector<std::uint32_t> host_counts(lanes), host_jobs(lanes);
    for (unsigned lane = 0; lane < lanes; ++lane) {
        // Support micro smaller than lane count without unsigned underflow or
        // reading beyond the repeated frontier. Default 1024x4 is unchanged.
        host_bases[lane] = lane % micro;
        host_counts[lane] = micro - lane % micro;
        host_jobs[lane] = lane;
    }
    BEAM_CUDA_CHECK(cudaMemcpy(bases, host_bases.data(), lanes * sizeof(std::uint64_t), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(counts, host_counts.data(), lanes * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(jobs, host_jobs.data(), lanes * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    std::vector<cudaStream_t> streams(lanes);
    std::vector<cudaGraph_t> graphs(lanes);
    std::vector<cudaGraphExec_t> execs(lanes);
    for (unsigned lane = 0; lane < lanes; ++lane) {
        BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&streams[lane], cudaStreamNonBlocking));
        const auto view = stream1_weights::transformer_scratch_view(scratch, model, micro, lane);
        auto infer = [&]() { launch_stream1_transformer_chunks_cuda(
            frontier, bases, counts, jobs + lane, true, network, view, scores,
            micro, inner, streams[lane], [&](std::uint32_t offset, std::uint32_t count) {
                BEAM_CUDA_CHECK(cudaMemcpyAsync(
                    saved_logits + (static_cast<std::size_t>(lane) * micro + offset) * MOVE_COUNT,
                    view.logits, static_cast<std::size_t>(count) * MOVE_COUNT * sizeof(half),
                    cudaMemcpyDeviceToDevice, streams[lane]));
            }); };
        infer();
        BEAM_CUDA_CHECK(cudaStreamSynchronize(streams[lane]));
        BEAM_CUDA_CHECK(cudaStreamBeginCapture(streams[lane], cudaStreamCaptureModeThreadLocal));
        BEAM_CUDA_CHECK(cudaMemsetAsync(scores + lane * micro * MOVE_COUNT, 0, lane_bytes, streams[lane]));
        infer();
        BEAM_CUDA_CHECK(cudaStreamEndCapture(streams[lane], &graphs[lane]));
        BEAM_CUDA_CHECK(cudaGraphInstantiate(&execs[lane], graphs[lane], nullptr, nullptr, 0));
    }
    std::vector<std::uint32_t> actual(lanes * micro * MOVE_COUNT);
    for (unsigned replay = 0; replay < 3; ++replay) {
        for (unsigned lane = 0; lane < lanes; ++lane)
            BEAM_CUDA_CHECK(cudaGraphLaunch(execs[lane], streams[lane]));
        for (unsigned lane = 0; lane < lanes; ++lane)
            BEAM_CUDA_CHECK(cudaStreamSynchronize(streams[lane]));
        BEAM_CUDA_CHECK(cudaMemcpy(actual.data(), scores, actual.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
        for (unsigned lane = 0; lane < lanes; ++lane)
            for (unsigned row = 0; row < micro; ++row)
                for (unsigned move = 0; move < MOVE_COUNT; ++move) {
                    const auto expected = row < host_counts[lane] ?
                        oracle[((host_bases[lane] + row) % 8) * MOVE_COUNT + move] : 0U;
                    const auto observed = actual[(lane * micro + row) * MOVE_COUNT + move];
                    // Inactive slots always require exact zero, regardless of
                    // numerical tolerance for active rows of different shapes.
                    if (row < host_counts[lane] ?
                        !stream1_graph_probe_score_matches(expected, observed, production_size) : observed != 0U)
                        throw std::runtime_error("concurrent graph scratch lane/output mismatch");
                }
        report << "graph_lanes_replay=" << replay << " micro=" << micro << " lanes=" << lanes << " status=pass\n";
    }
    // Preserve final-replay raw scores for an independent FP32 checker. Skip
    // inactive tails explicitly; canonical row order is lane then active row.
    std::vector<half> bias(MOVE_COUNT), logits(micro * MOVE_COUNT);
    BEAM_CUDA_CHECK(cudaMemcpy(bias.data(), network.output_bias,
        bias.size() * sizeof(half), cudaMemcpyDeviceToHost));
    std::ofstream raw("test_results/stream1_lane_scores.json");
    if (!raw) throw std::runtime_error("cannot create raw lane score artifact");
    raw << std::setprecision(9) << "{\"microbatch\":" << micro;
    if (requested_inner) raw << ",\"transformer_microbatch\":" << inner;
    raw << ",\"lane_count\":" << lanes << ",\"parent_bases\":[";
    for (unsigned lane = 0; lane < lanes; ++lane) {
        if (lane) raw << ',';
        raw << host_bases[lane];
    }
    raw << "],\"active_counts\":[";
    for (unsigned lane = 0; lane < lanes; ++lane) {
        if (lane) raw << ',';
        raw << host_counts[lane];
    }
    raw << "],\"scores\":[";
    bool first = true;
    for (unsigned lane = 0; lane < lanes; ++lane) {
        BEAM_CUDA_CHECK(cudaMemcpy(logits.data(), saved_logits + static_cast<std::size_t>(lane) * micro * MOVE_COUNT,
            logits.size() * sizeof(half), cudaMemcpyDeviceToHost));
        for (unsigned row = 0; row < host_counts[lane]; ++row) {
            if (!first) raw << ',';
            first = false; raw << '[';
            for (unsigned move = 0; move < MOVE_COUNT; ++move) {
                if (move) raw << ',';
                const float value = __half2float(logits[row * MOVE_COUNT + move]) + __half2float(bias[move]);
                if (std::isfinite(value)) raw << value;
                else raw << "null";
            }
            raw << ']';
        }
    }
    raw << "]}\n";
    raw.close();
    if (!raw) throw std::runtime_error("cannot finish raw lane score artifact");
    for (unsigned lane = 0; lane < lanes; ++lane) {
        BEAM_CUDA_CHECK(cudaGraphExecDestroy(execs[lane]));
        BEAM_CUDA_CHECK(cudaGraphDestroy(graphs[lane]));
        BEAM_CUDA_CHECK(cudaStreamDestroy(streams[lane]));
    }
    cudaFree(bases); cudaFree(counts); cudaFree(jobs); cudaFree(scores);
    cudaFree(saved_logits);
    stream1_weights::free_stream1_scratch(scratch);
    cudaFree(expanded_frontier);
}
