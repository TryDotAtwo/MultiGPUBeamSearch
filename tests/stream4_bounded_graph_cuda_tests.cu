#define BEAM_TEST_UNION_CAPACITY 4097
#define main inherited_union_fixture_main
#include "stream4_logical_union_cuda_tests.cu"
#undef main
#include <cstdlib>

int main() {
    try {
        int devices = 0; check(cudaGetDeviceCount(&devices));
        if (!devices) throw std::runtime_error("requires CUDA device");
        for (int gpu = 0; gpu < devices; ++gpu) {
            check(cudaSetDevice(gpu)); Fixture f;
            auto* threshold = f.alloc<unsigned>(2);
            auto* threshold_index = f.alloc<unsigned>(1);
            cudaGraph_t graphs[2]{}; cudaGraphExec_t executions[2]{};
            for (unsigned mode = 0; mode < 2; ++mode) {
                setenv("BEAM_STREAM4_BOUNDED_SORT", mode ? "1" : "0", 1);
                check(cudaStreamBeginCapture(f.stream, cudaStreamCaptureModeGlobal));
                stream4_shard_job_device_threshold_cuda(f.states, f.clean, f.dirty, f.processing,
                    threshold, threshold_index, Fixture::cap, f.key, f.reduced_key,
                    f.value, f.reduced_value, f.score_a, f.score_b, f.count_a, f.count_b,
                    f.flags, f.blocks, f.offsets, f.count, f.hist_a, f.hist_b, f.active,
                    f.temp, Fixture::temp_bytes, f.stream);
                check(cudaStreamEndCapture(f.stream, &graphs[mode]));
                check(cudaGraphInstantiate(&executions[mode], graphs[mode], 0));
            }
            const unsigned quarter = (Fixture::cap + 3) / 4, half = (Fixture::cap + 1) / 2;
            for (unsigned n : {0U, 1U, quarter - 1, quarter, quarter + 1,
                               half - 1, half, half + 1, Fixture::cap, 0U, half, 1U}) {
                for (unsigned cutoff : {0U, 17U, SCORE_MAX_KEY}) {
                    std::vector<CandidateMeta> input(Fixture::cap,
                        CandidateMeta{{999999, 999999}, 0, 0, 0});
                    for (unsigned i = 0; i < n; ++i) {
                        const auto id = i / 3;
                        input[i] = CandidateMeta{{id * 11400714819323198485ULL, id % 7},
                            i % 31, (n - i) % 23, i % 24};
                    }
                    std::vector<CandidateMeta> outputs[2]; std::vector<unsigned> histograms[2];
                    for (unsigned mode = 0; mode < 2; ++mode) {
                        const unsigned clean = n / 2, dirty = n - clean;
                        const unsigned thresholds[2] = {cutoff, cutoff};
                        check(cudaMemcpy(f.states, input.data(), input.size() * sizeof(CandidateMeta), cudaMemcpyHostToDevice));
                        check(cudaMemcpy(f.clean, &clean, sizeof(clean), cudaMemcpyHostToDevice));
                        check(cudaMemcpy(f.dirty, &dirty, sizeof(dirty), cudaMemcpyHostToDevice));
                        check(cudaMemcpy(threshold, thresholds, sizeof(thresholds), cudaMemcpyHostToDevice));
                        check(cudaMemset(f.active, 0, 2 * sizeof(unsigned)));
                        check(cudaMemset(f.processing, 1, 2 * sizeof(unsigned)));
                        check(cudaGraphLaunch(executions[mode], f.stream));
                        check(cudaStreamSynchronize(f.stream)); check(cudaGetLastError());
                        unsigned count = 0, dirty_after = 1, processing_after = 1, active = 0;
                        check(cudaMemcpy(&count, f.clean, sizeof(count), cudaMemcpyDeviceToHost));
                        check(cudaMemcpy(&dirty_after, f.dirty, sizeof(dirty_after), cudaMemcpyDeviceToHost));
                        check(cudaMemcpy(&processing_after, f.processing, sizeof(processing_after), cudaMemcpyDeviceToHost));
                        check(cudaMemcpy(&active, f.active, sizeof(active), cudaMemcpyDeviceToHost));
                        if (count > n || dirty_after || processing_after || active != 1)
                            throw std::runtime_error("invalid bounded publication");
                        outputs[mode].resize(count);
                        if (count) check(cudaMemcpy(outputs[mode].data(), f.states, count * sizeof(CandidateMeta), cudaMemcpyDeviceToHost));
                        histograms[mode].resize(SCORE_BIN_COUNT);
                        check(cudaMemcpy(histograms[mode].data(), f.hist_b,
                            SCORE_BIN_COUNT * sizeof(unsigned), cudaMemcpyDeviceToHost));
                    }
                    if (outputs[0].size() != outputs[1].size() || histograms[0] != histograms[1])
                        throw std::runtime_error("bounded count/histogram mismatch");
                    for (unsigned i = 0; i < outputs[0].size(); ++i) {
                        const auto& a = outputs[0][i]; const auto& b = outputs[1][i];
                        if (!(a.hash == b.hash) || a.score_key != b.score_key ||
                            a.parent_idx != b.parent_idx || a.route_packed != b.route_packed)
                            throw std::runtime_error("bounded winner/order mismatch");
                    }
                }
            }
            for (unsigned mode = 0; mode < 2; ++mode) {
                check(cudaGraphExecDestroy(executions[mode])); check(cudaGraphDestroy(graphs[mode]));
            }
            std::cout << "gpu=" << gpu << " bounded_graph_equivalence=pass cases=36\n";
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
