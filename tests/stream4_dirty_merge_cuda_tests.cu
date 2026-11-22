#ifndef BEAM_TEST_UNION_CAPACITY
#define BEAM_TEST_UNION_CAPACITY 1025
#endif
#define main inherited_union_fixture_main
#include "stream4_logical_union_cuda_tests.cu"
#undef main

#include <cstdlib>

namespace {
using Key = std::pair<std::uint64_t, std::uint64_t>;
Key key(const CandidateMeta& c) { return {c.hash.hi, c.hash.lo}; }
auto rank(const CandidateMeta& c) { return std::make_tuple(c.score_key, c.parent_idx, c.route_packed); }

std::vector<CandidateMeta> oracle(const std::vector<CandidateMeta>& clean,
                                  const std::vector<CandidateMeta>& dirty,
                                  unsigned threshold) {
    std::map<Key, CandidateMeta> best;
    for (auto* range : {&clean, &dirty}) for (const auto& c : *range) {
        if (c.score_key > threshold || c.score_key == UINT32_MAX) continue;
        const auto k = key(c);
        if (!best.count(k) || rank(c) < rank(best.at(k))) best[k] = c;
    }
    std::vector<CandidateMeta> output;
    for (const auto& entry : best) output.push_back(entry.second);
    return output;
}

void verify_case(Fixture& f, cudaGraphExec_t* graphs, unsigned* threshold,
                 unsigned* threshold_index, const std::vector<CandidateMeta>& clean,
                 const std::vector<CandidateMeta>& dirty, unsigned cutoff, unsigned toggle,
                 unsigned modes) {
    if (clean.size() + dirty.size() > Fixture::cap) throw std::runtime_error("fixture overflow");
    for (std::size_t i = 1; i < clean.size(); ++i)
        if (!(key(clean[i - 1]) < key(clean[i]))) throw std::runtime_error("clean fixture not sorted/unique");
    const auto expected = oracle(clean, dirty, cutoff);
    std::vector<CandidateMeta> input(Fixture::cap,
        CandidateMeta{{UINT64_MAX, UINT64_MAX}, 0, 0, 0});
    std::copy(clean.begin(), clean.end(), input.begin());
    std::copy(dirty.begin(), dirty.end(), input.begin() + clean.size());
    const unsigned nclean = static_cast<unsigned>(clean.size());
    const unsigned ndirty = static_cast<unsigned>(dirty.size());
    const unsigned limits[2] = {toggle ? SCORE_MAX_KEY : cutoff, toggle ? cutoff : SCORE_MAX_KEY};
    const unsigned busy = 1, zero = 0;
    for (unsigned mode = 0; mode < 2; ++mode) {
        if (!(modes & (1U << mode))) continue;
        check(cudaMemcpy(f.states, input.data(), input.size() * sizeof(CandidateMeta), cudaMemcpyHostToDevice));
        check(cudaMemcpy(f.clean, &nclean, sizeof(nclean), cudaMemcpyHostToDevice));
        check(cudaMemcpy(f.dirty, &ndirty, sizeof(ndirty), cudaMemcpyHostToDevice));
        check(cudaMemcpy(f.processing, &busy, sizeof(busy), cudaMemcpyHostToDevice));
        check(cudaMemcpy(f.active, &zero, sizeof(zero), cudaMemcpyHostToDevice));
        check(cudaMemcpy(threshold, limits, sizeof(limits), cudaMemcpyHostToDevice));
        check(cudaMemcpy(threshold_index, &toggle, sizeof(toggle), cudaMemcpyHostToDevice));
        check(cudaGraphLaunch(graphs[mode], f.stream));
        check(cudaStreamSynchronize(f.stream)); check(cudaGetLastError());
        unsigned count = 0, remaining_dirty = 1, processing = 1, active = 0;
        check(cudaMemcpy(&count, f.clean, sizeof(count), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&remaining_dirty, f.dirty, sizeof(remaining_dirty), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&processing, f.processing, sizeof(processing), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&active, f.active, sizeof(active), cudaMemcpyDeviceToHost));
        if (count != expected.size() || remaining_dirty || processing || active != 1)
            throw std::runtime_error("wrong count or publication mode=" + std::to_string(mode));
        std::vector<CandidateMeta> actual(count);
        if (count) check(cudaMemcpy(actual.data(), f.states, count * sizeof(CandidateMeta), cudaMemcpyDeviceToHost));
        std::vector<unsigned> histogram(SCORE_BIN_COUNT, 0), observed(SCORE_BIN_COUNT);
        for (unsigned i = 0; i < count; ++i) {
            if (key(actual[i]) != key(expected[i]) || rank(actual[i]) != rank(expected[i]))
                throw std::runtime_error("wrong sorted winner at=" + std::to_string(i) +
                                         " mode=" + std::to_string(mode));
            if (actual[i].score_key < SCORE_BIN_COUNT) ++histogram[actual[i].score_key];
        }
        check(cudaMemcpy(observed.data(), f.hist_b, observed.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
        if (histogram != observed) throw std::runtime_error("wrong histogram");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        unsigned modes = 3;
        if (argc == 2 && std::string(argv[1]) == "--unbounded-only") modes = 1;
        else if (argc == 2 && std::string(argv[1]) == "--bounded-only") modes = 2;
        else if (argc != 1) throw std::runtime_error("unknown graph mode");
        int devices = 0; check(cudaGetDeviceCount(&devices));
        if (devices < 2) throw std::runtime_error("requires two physical GPUs");
        for (int gpu = 0; gpu < 2; ++gpu) {
            check(cudaSetDevice(gpu)); Fixture f;
            auto* threshold = f.alloc<unsigned>(2);
            auto* threshold_index = f.alloc<unsigned>(1);
            cudaGraph_t graph[2]{}; cudaGraphExec_t exec[2]{};
            for (unsigned mode = 0; mode < 2; ++mode) {
                if (!(modes & (1U << mode))) continue;
                setenv("BEAM_STREAM4_BOUNDED_SORT", mode ? "1" : "0", 1);
                check(cudaStreamBeginCapture(f.stream, cudaStreamCaptureModeGlobal));
                stream4_shard_job_device_threshold_cuda(f.states, f.clean, f.dirty, f.processing,
                    threshold, threshold_index, Fixture::cap, f.key, f.reduced_key,
                    f.value, f.reduced_value, f.score_a, f.score_b, f.count_a, f.count_b,
                    f.flags, f.blocks, f.offsets, f.count, f.hist_a, f.hist_b, f.active,
                    f.temp, Fixture::temp_bytes, f.stream);
                check(cudaStreamEndCapture(f.stream, &graph[mode]));
                check(cudaGraphInstantiate(&exec[mode], graph[mode], 0));
            }
            std::vector<CandidateMeta> clean{
                {{0, 0}, 9, 3, 4}, {{9, 0}, 8, 5, 9}, {{0, 1}, 7, 5, 8},
                {{UINT64_MAX, UINT64_MAX}, 6, 8, 7}};
            std::vector<CandidateMeta> dirty{
                {{0, 0}, 8, 3, 4}, {{9, 0}, 8, 5, 3}, {{0, 1}, 3, 5, 8},
                {{0, 2}, 2, 1, 99}, {{1, 1}, 5, 2, 9},
                {{UINT64_MAX, UINT64_MAX}, 2, 8, 6}, {{UINT64_MAX, UINT64_MAX}, 5, 9, 2}};
            unsigned cases = 0;
            if (Fixture::cap >= clean.size() + dirty.size()) {
                for (unsigned threshold_value : {0U, 3U, 8U, SCORE_MAX_KEY}) {
                    verify_case(f, exec, threshold, threshold_index, clean, dirty, threshold_value, cases & 1U, modes);
                    ++cases;
                }
                // Reuse the logical result as a clean prefix in the next graph
                // invocation, then append a new unsorted dirty suffix.
                clean = oracle(clean, dirty, SCORE_MAX_KEY);
                dirty = {{{9, 0}, 1, 5, 1}, {{7, 2}, 5, 4, 9}, {{0, 0}, 4, 2, 7}};
                verify_case(f, exec, threshold, threshold_index, clean, dirty, SCORE_MAX_KEY, cases & 1U, modes); ++cases;
            }
            std::mt19937_64 random(0xD17C1EAULL);
            for (unsigned nd : {0U, 1U, 255U, 256U, 257U, 511U, 512U, 513U, 514U, Fixture::cap}) {
                if (nd > Fixture::cap) continue;
                clean.clear(); dirty.clear();
                const unsigned nc = std::min(70U, Fixture::cap - nd);
                for (unsigned i = 0; i < nc; ++i)
                    clean.push_back({{2ULL * i, i % 7}, i + 13ULL, i % 23, 1000U + i});
                std::sort(clean.begin(), clean.end(), [](auto a, auto b) { return key(a) < key(b); });
                for (unsigned i = 0; i < nd; ++i) {
                    const auto x = random();
                    const auto h = (x % 31 == 0) ? Hash128{UINT64_MAX, UINT64_MAX}
                                                 : Hash128{(x % 110) * 2, (x >> 17) % 7};
                    dirty.push_back({h, x % 27, static_cast<unsigned>((x >> 8) % 23), i});
                }
                for (unsigned cutoff : {0U, 17U, SCORE_MAX_KEY}) {
                    verify_case(f, exec, threshold, threshold_index, clean, dirty, cutoff, cases & 1U, modes);
                    ++cases;
                }
            }
            // Exercise a one-item conditional sort bound and the sentinel
            // boundary with both an empty and a real all-ones hash. For small
            // capacities the graph chooses its one-item body even when a
            // separate clean prefix must still participate in the merge.
            if (Fixture::cap >= 1) {
                const CandidateMeta maximum{{UINT64_MAX, UINT64_MAX}, 7, 5, 11};
                verify_case(f, exec, threshold, threshold_index, {}, {maximum}, 5, cases & 1U, modes); ++cases;
                verify_case(f, exec, threshold, threshold_index, {maximum}, {}, 5, cases & 1U, modes); ++cases;
                verify_case(f, exec, threshold, threshold_index, {}, {maximum}, 4, cases & 1U, modes); ++cases;
            }
            if (Fixture::cap >= 2) {
                const CandidateMeta maximum{{UINT64_MAX, UINT64_MAX}, 7, 5, 11};
                const CandidateMeta better{{UINT64_MAX, UINT64_MAX}, 9, 2, 12};
                const CandidateMeta low{{0, 0}, 3, 1, 13};
                verify_case(f, exec, threshold, threshold_index, {maximum}, {better}, 5, cases & 1U, modes); ++cases;
                verify_case(f, exec, threshold, threshold_index, {maximum}, {low}, 5, cases & 1U, modes); ++cases;
                verify_case(f, exec, threshold, threshold_index, {low}, {maximum}, 5, cases & 1U, modes); ++cases;
            }
            for (unsigned mode = 0; mode < 2; ++mode) {
                if (!(modes & (1U << mode))) continue;
                check(cudaGraphExecDestroy(exec[mode])); check(cudaGraphDestroy(graph[mode]));
            }
            std::cout << "gpu=" << gpu << " dirty_merge_oracle=pass cases=" << cases
                      << " graph_mode_mask=" << modes << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
