#include <cuda_runtime.h>
#include "../cuda/stream4.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace beam;
namespace {
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Fixture {
#ifndef BEAM_TEST_UNION_CAPACITY
#define BEAM_TEST_UNION_CAPACITY 257
#endif
    static constexpr unsigned cap = BEAM_TEST_UNION_CAPACITY;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    unsigned graph_threshold = 0;
    unsigned graph_sort_bound = 0;
    std::vector<void*> allocations;
    template<class T> T* alloc(std::size_t n) {
        T* p = nullptr; check(cudaMalloc(&p, n * sizeof(T))); allocations.push_back(p);
        check(cudaMemset(p, 0, n * sizeof(T))); return p;
    }
    CandidateMeta* states = alloc<CandidateMeta>(2 * cap);
    Hash128 *key = alloc<Hash128>(2 * cap), *reduced_key = alloc<Hash128>(2 * cap);
    CandidateMeta *value = alloc<CandidateMeta>(2 * cap), *reduced_value = alloc<CandidateMeta>(2 * cap);
    unsigned *score_a = alloc<unsigned>(2 * cap), *score_b = alloc<unsigned>(2 * cap);
    std::uint64_t *count_a = alloc<std::uint64_t>(2 * cap), *count_b = alloc<std::uint64_t>(2 * cap);
    unsigned *flags = alloc<unsigned>(2 * cap), *blocks = alloc<unsigned>((2ULL * cap + 255) / 256), *offsets = alloc<unsigned>((2ULL * cap + 255) / 256);
    unsigned *count = alloc<unsigned>(1), *clean = alloc<unsigned>(2), *dirty = alloc<unsigned>(2);
    unsigned *processing = alloc<unsigned>(2), *hist_a = alloc<unsigned>(2ULL * SCORE_BIN_COUNT);
    unsigned *hist_b = alloc<unsigned>(2ULL * SCORE_BIN_COUNT), *active = alloc<unsigned>(2);
#ifndef BEAM_TEST_UNION_TEMP_BYTES
#define BEAM_TEST_UNION_TEMP_BYTES (8ULL << 20)
#endif
    static constexpr std::size_t temp_bytes = BEAM_TEST_UNION_TEMP_BYTES;
    void* temp = alloc<unsigned char>(temp_bytes);
    Fixture() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    ~Fixture() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        for (void* p : allocations) cudaFree(p);
    }
    void run(const std::vector<CandidateMeta>& a, const std::vector<CandidateMeta>& b, unsigned threshold) {
        if (a.size() > cap || b.size() > cap) throw std::runtime_error("bad test fixture size");
        // Holes contain valid-looking low-score states; counts must exclude them.
        std::vector<CandidateMeta> input(2 * cap, CandidateMeta{Hash128{99999, 99999}, 0, 0, 0});
        std::copy(a.begin(), a.end(), input.begin());
        std::copy(b.begin(), b.end(), input.begin() + cap);
        unsigned counts[2]{static_cast<unsigned>(a.size()), static_cast<unsigned>(b.size())};
        check(cudaMemcpy(states, input.data(), input.size() * sizeof(input[0]), cudaMemcpyHostToDevice));
        check(cudaMemcpy(clean, counts, sizeof(counts), cudaMemcpyHostToDevice));
        check(cudaMemset(dirty, 0, 2 * sizeof(unsigned)));
        unsigned sort_bound = 0;
#ifdef BEAM_TEST_UNION_SORT_BOUND
        sort_bound = std::max(1U, static_cast<unsigned>(a.size() + b.size()));
        std::cout << "sort_item_bound=" << sort_bound << " physical_items=" << 2 * cap << '\n';
#endif
// Fault injection: the first nonempty fixture exceeds this bound. The process
// must fail at the device guard, never return a truncated successful union.
#ifdef BEAM_TEST_UNION_BAD_SORT_BOUND
        sort_bound = 1;
        std::cout << "fault_injection=underbound sort_item_bound=1" << std::endl;
#endif
        const auto enqueue = [&]() {
#ifdef BEAM_TEST_LOGICAL_UNION
        stream4_finalize_logical_shard_union_cuda(states, clean, dirty, processing, threshold, cap,
            key, reduced_key, value, reduced_value, score_a, score_b, count_a, count_b,
            flags, blocks, offsets, count, hist_a, hist_b, active, temp, temp_bytes, stream, sort_bound);
#else
        for (unsigned physical = 0; physical < 2; ++physical) {
            stream4_shard_job_cuda(states + physical * cap, clean + physical, dirty + physical,
                processing + physical, threshold, cap, key, reduced_key, value, reduced_value,
                score_a, score_b, count_a, count_b, flags, blocks, offsets, count,
                hist_a + physical * SCORE_BIN_COUNT, hist_b + physical * SCORE_BIN_COUNT,
                active + physical, temp, temp_bytes, stream);
        }
#endif
        };
#ifdef BEAM_TEST_UNION_GRAPH
        if (!executable || graph_threshold != threshold || graph_sort_bound != sort_bound) {
            if (executable) { check(cudaGraphExecDestroy(executable)); executable = nullptr; }
            if (graph) { check(cudaGraphDestroy(graph)); graph = nullptr; }
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            enqueue();
            check(cudaStreamEndCapture(stream, &graph));
            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            graph_threshold = threshold;
            graph_sort_bound = sort_bound;
            std::cout << "union_graph=captured threshold=" << threshold << '\n';
        } else {
            std::cout << "union_graph=reused threshold=" << threshold << '\n';
        }
        check(cudaGraphLaunch(executable, stream));
#else
        enqueue();
#endif
        check(cudaGetLastError()); check(cudaStreamSynchronize(stream));
    }
    bool verify(const std::vector<CandidateMeta>& a, const std::vector<CandidateMeta>& b, unsigned threshold) {
        using Key = std::pair<std::uint64_t, std::uint64_t>;
        std::map<Key, CandidateMeta> oracle, observed;
        auto order = [](const CandidateMeta& c) { return std::make_tuple(c.score_key, c.parent_idx, c.route_packed); };
        for (const auto* values : {&a, &b}) for (auto c : *values) {
            if (c.score_key > threshold) continue;
            const Key k{c.hash.hi, c.hash.lo};
            if (!oracle.count(k) || order(c) < order(oracle.at(k))) oracle[k] = c;
        }
        std::array<unsigned, 2> counts{}, dirt{}, busy{}, published{};
        check(cudaMemcpy(counts.data(), clean, sizeof(counts), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(dirt.data(), dirty, sizeof(dirt), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(busy.data(), processing, sizeof(busy), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(published.data(), active, sizeof(published), cudaMemcpyDeviceToHost));
        std::cout << "expected_unique=" << oracle.size() << " observed_count=" << counts[0]+counts[1] << '\n';
        if (counts[0] + counts[1] != oracle.size()) return false;
        std::vector<CandidateMeta> output(2 * cap);
        check(cudaMemcpy(output.data(), states, output.size() * sizeof(output[0]), cudaMemcpyDeviceToHost));
        for (unsigned physical = 0; physical < 2; ++physical) {
            if (counts[physical] > cap || dirt[physical] || busy[physical]) return false;
            std::vector<unsigned> expected(SCORE_BIN_COUNT, 0), actual(SCORE_BIN_COUNT);
            for (unsigned i = 0; i < counts[physical]; ++i) {
                auto c = output[physical * cap + i];
                if (c.score_key >= SCORE_BIN_COUNT) return false;
                ++expected[c.score_key];
                if (!observed.emplace(Key{c.hash.hi, c.hash.lo}, c).second) return false;
            }
            auto* h = (published[physical] & 1) ? hist_b : hist_a;
            check(cudaMemcpy(actual.data(), h + physical * SCORE_BIN_COUNT,
                             actual.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
            if (expected != actual) return false;
        }
        if (oracle.size() != observed.size()) return false;
        for (auto item : oracle) {
            if (!observed.count(item.first) || order(item.second) != order(observed.at(item.first))) return false;
        }
        return true;
    }
};
}
int main() {
    try {
        int n = 0; check(cudaGetDeviceCount(&n));
#ifdef BEAM_TEST_SINGLE_GPU
        if (n < 1) throw std::runtime_error("requires a physical GPU");
        n = 1;
        std::cout << "scope=local_single_gpu_only\n";
#else
        if (n != 2) throw std::runtime_error("requires two physical GPUs");
#endif
        bool passed = true;
        for (int gpu = 0; gpu < n; ++gpu) {
            check(cudaSetDevice(gpu)); Fixture f;
            std::vector<CandidateMeta> a{{{1, 0}, 9, 5, 3}, {{2, 0}, 8, 2, 2}, {{3, 0}, 7, 7, 1}};
            std::vector<CandidateMeta> b{{{1, 0}, 10, 3, 0}, {{2, 0}, 4, 2, 1}};
            f.run(a, b, SCORE_MAX_KEY);
            std::cout << "gpu=" << gpu << ' ';
            passed = f.verify(a, b, SCORE_MAX_KEY) && passed;
#ifdef BEAM_TEST_LOGICAL_UNION
            // Same threshold, changed occupancy: graph replay must not retain
            // old counts or treat a legitimate all-ones Hash128 as a tail hole.
            const auto replay_edge = [&]() {
                f.run(a, b, SCORE_MAX_KEY);
                if (!f.verify(a, b, SCORE_MAX_KEY)) throw std::runtime_error("union occupancy edge mismatch");
            };
            a.clear(); b.clear(); replay_edge();
            for (unsigned i = 0; i < Fixture::cap; ++i)
                a.push_back(CandidateMeta{{i + 1U, 5}, SCORE_MAX_KEY, i, 0});
            replay_edge();
            b.swap(a); replay_edge();
            a = b; replay_edge();
            for (auto& candidate : b) candidate.hash.hi = 6;
            replay_edge(); // All 514 unique entries must survive: no shard cap.
            a.clear(); b.clear(); replay_edge();
            a.push_back(CandidateMeta{{UINT64_MAX, UINT64_MAX}, SCORE_MAX_KEY, 7, 3});
            b.push_back(CandidateMeta{{UINT64_MAX, UINT64_MAX}, SCORE_MAX_KEY, 8, 2});
            replay_edge();
            a.clear(); b.clear(); replay_edge();
            std::mt19937 rng(918U + gpu);
            for (unsigned trial = 0; trial < 20; ++trial) {
                a.clear(); b.clear();
                for (unsigned i = 0; i < Fixture::cap; ++i) {
                    auto make = [&]() { return CandidateMeta{{rng() % (trial & 1 ? 40 : 9999), rng() % 3},
                                                              rng() % 50, static_cast<unsigned>(rng() % 17),
                                                              static_cast<unsigned>(rng() % 24)}; };
                    if (trial != 0) a.push_back(make());
                    if (trial > 1) b.push_back(make());
                }
                const unsigned threshold = trial % 3 == 0 ? 8 : SCORE_MAX_KEY;
                f.run(a, b, threshold);
                if (!f.verify(a, b, threshold)) throw std::runtime_error("union oracle mismatch trial=" + std::to_string(trial));
            }
#endif
        }
        if (!passed) { std::cerr << "FAIL: cross-buffer unique union missing\n"; return 1; }
        std::cout << "PASS: logical union and exact physical histograms\n"; return 0;
    } catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 2; }
}
