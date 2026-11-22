#include <cuda_runtime.h>
#include "../cuda/threshold.hpp"
#include <array>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <map>
#include <random>
#include <vector>

using namespace beam;
namespace {
void cu(cudaError_t value) {
    if (value != cudaSuccess) throw std::runtime_error(cudaGetErrorString(value));
}
void nc(ncclResult_t value) {
    if (value != ncclSuccess) throw std::runtime_error(ncclGetErrorString(value));
}
template<class T> T* allocate(std::size_t count) {
    T* ptr = nullptr;
    cu(cudaMalloc(&ptr, count * sizeof(T)));
    cu(cudaMemset(ptr, 0, count * sizeof(T)));
    return ptr;
}
struct Rank {
    std::uint32_t *a, *b, *active, *snapshot, *threshold, *initialized, *index;
    std::uint64_t *local, *global;
    cudaStream_t stream;
};
#ifdef BEAM_TEST_CONSERVATIVE
void check_random_histograms(Rank& r, unsigned seed) {
    std::mt19937 rng(seed);
    for (unsigned trial = 0; trial < 32; ++trial) {
        std::map<unsigned, unsigned> a, b, united;
        for (unsigned i = 0; i < 128; ++i) {
            auto& target = (rng() & 1) ? a : b;
            const unsigned key = rng() % 48;
            const unsigned score = rng() % 3 == 0 ? SCORE_BIN_COUNT - 1 : rng() % 1025;
            const auto found = target.find(key);
            if (found == target.end() || score < found->second) target[key] = score;
        }
        if (trial == 0) { a.clear(); b.clear(); }
        if (trial == 1) b.clear();
        if (trial == 2) b = a;
        std::vector<std::uint32_t> ha(2ULL * SCORE_BIN_COUNT, 0), hb(ha.size(), 0);
        for (auto item : a) { ++ha[item.second]; united[item.first] = item.second; }
        for (auto item : b) {
            ++hb[SCORE_BIN_COUNT + item.second];
            if (!united.count(item.first) || item.second < united[item.first]) united[item.first] = item.second;
        }
        cu(cudaMemcpy(r.a, ha.data(), ha.size() * sizeof(ha[0]), cudaMemcpyHostToDevice));
        cu(cudaMemcpy(r.b, hb.data(), hb.size() * sizeof(hb[0]), cudaMemcpyHostToDevice));
        threshold_build_conservative_histogram_cuda(r.a, r.b, r.active, r.snapshot, r.local, 1, 2, r.stream);
        cu(cudaStreamSynchronize(r.stream));
        std::vector<std::uint64_t> actual(SCORE_BIN_COUNT);
        cu(cudaMemcpy(actual.data(), r.local, actual.size() * sizeof(actual[0]), cudaMemcpyDeviceToHost));
        std::uint64_t cumulative = 0;
        for (unsigned score = 0; score < SCORE_BIN_COUNT; ++score) {
            cumulative += actual[score];
            if (score > 1024 && score + 1 < SCORE_BIN_COUNT) {
                if (actual[score] != 0) throw std::runtime_error("unexpected mass in empty score range");
                continue;
            }
            // Independent set oracle; every possible score boundary is checked.
            const auto count = [score](const auto& states) {
                return std::count_if(states.begin(), states.end(), [score](auto p) { return p.second <= score; });
            };
            if (cumulative != static_cast<std::uint64_t>(std::max(count(a), count(b))) ||
                cumulative > static_cast<std::uint64_t>(count(united))) {
                throw std::runtime_error("CDF oracle mismatch at trial=" + std::to_string(trial) +
                                         " score=" + std::to_string(score));
            }
        }
    }
    std::cout << "PASS: 32 seeded CDF fixtures seed=" << seed << std::endl;
}
#endif
}

// Real production histogram + NCCL + threshold functions, no model fixture.
// Each rank owns a disjoint logical hash domain. Within it A={x:1,y:2},
// B={x:1}. K_global=4 requires score 2, not the multiset cutoff 1.
int main(int argc, char** argv) {
    try {
        const bool histogram_only = argc == 2 && std::string(argv[1]) == "--histogram-only";
        if (argc > 1 && !histogram_only) throw std::runtime_error("unknown test argument");
        std::cout << "test_mode=" << (histogram_only ? "histogram_only_no_nccl" : "two_gpu_nccl") << std::endl;
        int devices = 0;
        cu(cudaGetDeviceCount(&devices));
        if (devices != 2) throw std::runtime_error("requires exactly two real GPUs");
        std::array<Rank, 2> ranks{};
        std::array<ncclComm_t, 2> comms{};
        int ids[2]{0, 1};
        if (!histogram_only) nc(ncclCommInitAll(comms.data(), 2, ids));
        for (int rank = 0; rank < 2; ++rank) {
            cu(cudaSetDevice(rank));
            auto& r = ranks[rank];
            cu(cudaStreamCreate(&r.stream));
            r.a = allocate<std::uint32_t>(2ULL * SCORE_BIN_COUNT);
            r.b = allocate<std::uint32_t>(2ULL * SCORE_BIN_COUNT);
            r.active = allocate<std::uint32_t>(2);
            r.snapshot = allocate<std::uint32_t>(2);
            r.local = allocate<std::uint64_t>(SCORE_BIN_COUNT);
            r.global = allocate<std::uint64_t>(SCORE_BIN_COUNT);
            r.threshold = allocate<std::uint32_t>(2);
            r.initialized = allocate<std::uint32_t>(2);
            r.index = allocate<std::uint32_t>(1);
            // Histogram publication ping-pong is distinct from physical A/B.
            // Use publication slot 0 for physical A, slot 1 for physical B.
            std::vector<std::uint32_t> ha(2ULL * SCORE_BIN_COUNT, 0), hb(ha.size(), 0);
            ha[1] = 1; ha[2] = 1;
            hb[SCORE_BIN_COUNT + 1] = 1;
            const std::uint32_t active[2]{0, 1};
            cu(cudaMemcpy(r.a, ha.data(), ha.size() * sizeof(ha[0]), cudaMemcpyHostToDevice));
            cu(cudaMemcpy(r.b, hb.data(), hb.size() * sizeof(hb[0]), cudaMemcpyHostToDevice));
            cu(cudaMemcpy(r.active, active, sizeof(active), cudaMemcpyHostToDevice));
#ifdef BEAM_TEST_CONSERVATIVE
            threshold_build_conservative_histogram_cuda(r.a, r.b, r.active, r.snapshot,
                                                        r.local, 1, 2, r.stream);
#else
            threshold_build_local_histogram_cuda(r.a, r.b, r.active, r.snapshot,
                                                r.local, 2, r.stream);
#endif
        }
        if (!histogram_only) nc(ncclGroupStart());
        for (int rank = 0; rank < 2; ++rank) {
            cu(cudaSetDevice(rank));
            auto& r = ranks[rank];
            if (histogram_only) {
                cu(cudaMemcpyAsync(r.global, r.local, SCORE_BIN_COUNT * sizeof(std::uint64_t),
                                   cudaMemcpyDeviceToDevice, r.stream));
            } else {
                threshold_allreduce_histogram_nccl_cuda(r.local, r.global, comms[rank], r.stream);
            }
        }
        if (!histogram_only) nc(ncclGroupEnd());
        bool correct = true;
        for (int rank = 0; rank < 2; ++rank) {
            cu(cudaSetDevice(rank));
            auto& r = ranks[rank];
            threshold_update_periodic_cuda(r.global, r.threshold, r.initialized, r.index,
                                           histogram_only ? 2 : 4, r.stream);
            cu(cudaStreamSynchronize(r.stream));
            std::uint32_t active = 0, cutoff[2]{};
            cu(cudaMemcpy(&active, r.index, sizeof(active), cudaMemcpyDeviceToHost));
            cu(cudaMemcpy(cutoff, r.threshold, sizeof(cutoff), cudaMemcpyDeviceToHost));
            std::cout << "rank=" << rank << " expected_cutoff=2 observed_cutoff="
                      << cutoff[active & 1] << std::endl;
            correct = correct && cutoff[active & 1] == 2;
#ifdef BEAM_TEST_CONSERVATIVE
            check_random_histograms(r, 718U + rank);
#endif
            for (void* p : {static_cast<void*>(r.a), static_cast<void*>(r.b),
                           static_cast<void*>(r.active), static_cast<void*>(r.snapshot),
                           static_cast<void*>(r.threshold), static_cast<void*>(r.initialized),
                           static_cast<void*>(r.index), static_cast<void*>(r.local),
                           static_cast<void*>(r.global)}) cu(cudaFree(p));
            cu(cudaStreamDestroy(r.stream));
            if (!histogram_only) nc(ncclCommDestroy(comms[rank]));
        }
        if (!correct) { std::cerr << "FAIL: cross-buffer duplicates tighten unique threshold\n"; return 1; }
        std::cout << "PASS: two-GPU conservative threshold\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 2;
    }
}
