#include "../cuda/stream4_merge.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace beam;

namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

__global__ void partitions(const Hash128* a, unsigned na, const Hash128* b,
                           unsigned nb, unsigned* output) {
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d <= na + nb) {
        output[d] = detail::stream4_merge_partition(
            detail::Stream4HashKeys{a}, na, detail::Stream4HashKeys{b}, nb, d);
    }
}

void one_case(const std::vector<Hash128>& a, const std::vector<Hash128>& b) {
    struct Origin { Hash128 key; bool from_a; };
    std::vector<Origin> left, right, merged;
    for (auto key : a) left.push_back({key, true});
    for (auto key : b) right.push_back({key, false});
    std::merge(left.begin(), left.end(), right.begin(), right.end(),
               std::back_inserter(merged), [](const Origin& x, const Origin& y) {
                   return x.key.hi < y.key.hi ||
                          (x.key.hi == y.key.hi && x.key.lo < y.key.lo);
               });
    std::vector<unsigned> expected(merged.size() + 1, 0);
    for (std::size_t i = 0; i < merged.size(); ++i)
        expected[i + 1] = expected[i] + unsigned(merged[i].from_a);

    Hash128 *da = nullptr, *db = nullptr;
    unsigned* result = nullptr;
    check(cudaMalloc(&da, std::max<std::size_t>(1, a.size()) * sizeof(Hash128)));
    check(cudaMalloc(&db, std::max<std::size_t>(1, b.size()) * sizeof(Hash128)));
    check(cudaMalloc(&result, expected.size() * sizeof(unsigned)));
    if (!a.empty()) check(cudaMemcpy(da, a.data(), a.size() * sizeof(Hash128), cudaMemcpyHostToDevice));
    if (!b.empty()) check(cudaMemcpy(db, b.data(), b.size() * sizeof(Hash128), cudaMemcpyHostToDevice));
    partitions<<<(expected.size() + 255) / 256, 256>>>(
        da, static_cast<unsigned>(a.size()), db, static_cast<unsigned>(b.size()), result);
    check(cudaGetLastError());
    std::vector<unsigned> actual(expected.size());
    check(cudaMemcpy(actual.data(), result, actual.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
    for (unsigned d = 0; d < expected.size(); ++d) {
        const auto cpu = detail::stream4_merge_partition(
            detail::Stream4HashKeys{a.data()}, static_cast<unsigned>(a.size()),
            detail::Stream4HashKeys{b.data()}, static_cast<unsigned>(b.size()), d);
        if (actual[d] != expected[d] || cpu != expected[d])
            throw std::runtime_error("partition mismatch diagonal=" + std::to_string(d) +
                " a=" + std::to_string(a.size()) + " b=" + std::to_string(b.size()) +
                " expected=" + std::to_string(expected[d]) + " gpu=" + std::to_string(actual[d]) +
                " cpu=" + std::to_string(cpu));
    }
    check(cudaFree(result)); check(cudaFree(db)); check(cudaFree(da));
}
} // namespace

int main() {
    try {
        int devices = 0; check(cudaGetDeviceCount(&devices));
        if (devices < 2) throw std::runtime_error("requires two physical GPUs");
        for (int gpu = 0; gpu < 2; ++gpu) {
            check(cudaSetDevice(gpu));
            one_case({}, {});
            one_case({{0, 0}}, {});
            one_case({}, {{UINT64_MAX, UINT64_MAX}});
            one_case({{0, 0}, {1, 0}, {0, 1}, {UINT64_MAX, UINT64_MAX}},
                     {{0, 0}, {2, 0}, {0, 1}, {UINT64_MAX, UINT64_MAX}});
            std::mt19937_64 random(0xBEA40DULL);
            unsigned cases = 4;
            const unsigned sizes[] = {0, 1, 2, 7, 31, 255, 256, 257, 1025};
            for (unsigned na : sizes) for (unsigned nb : sizes) {
                std::vector<Hash128> a(na), b(nb);
                for (auto* v : {&a, &b}) for (auto& key : *v) {
                    // Repeated keys, independent halves, and the maximum hash.
                    const auto x = random();
                    key = (x % 19 == 0) ? Hash128{UINT64_MAX, UINT64_MAX}
                                         : Hash128{x % 17, (x >> 17) % 11};
                }
                auto less = [](Hash128 x, Hash128 y) {
                    return x.hi < y.hi || (x.hi == y.hi && x.lo < y.lo);
                };
                std::sort(a.begin(), a.end(), less);
                std::sort(b.begin(), b.end(), less);
                one_case(a, b); ++cases;
            }
            std::cout << "gpu=" << gpu << " merge_partition=pass cases=" << cases << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
