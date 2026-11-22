#include "../cuda/radix128.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using beam::CandidateMeta;
using beam::Hash128;

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <typename T> T value_for(std::size_t index, Hash128 key);
template <> std::uint64_t value_for(std::size_t index, Hash128) {
    // The value is the original candidate ID, including gaps from compaction.
    return static_cast<std::uint64_t>(index) * 17 + 5;
}
template <> CandidateMeta value_for(std::size_t index, Hash128 key) {
    return CandidateMeta{key, static_cast<std::uint64_t>(index * 19 + 1),
                         static_cast<std::uint32_t>(index % 3),
                         static_cast<std::uint32_t>(index % 5)};
}
bool equal(std::uint64_t a, std::uint64_t b) { return a == b; }
bool equal(CandidateMeta a, CandidateMeta b) {
    return a.hash.lo == b.hash.lo && a.hash.hi == b.hash.hi &&
           a.parent_idx == b.parent_idx && a.score_key == b.score_key &&
           a.route_packed == b.route_packed;
}

template <typename Value>
void run_case(std::size_t count, bool graph_replay) {
    std::mt19937_64 random(0x12345678U + count);
    std::vector<Hash128> keys(count);
    std::vector<Value> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        switch (i % 11) {
        case 0: keys[i] = {0, 0}; break;
        case 1: keys[i] = {UINT64_MAX, UINT64_MAX}; break;
        case 2: keys[i] = {UINT64_MAX, 0}; break;
        case 3: keys[i] = {0, UINT64_MAX}; break;
        case 4: keys[i] = {1, 7}; break;
        case 5: keys[i] = {2, 7}; break; // Different low half.
        case 6: keys[i] = {1, 8}; break; // Different high half.
        case 7: keys[i] = {1, 7}; break; // Equal full key, distinct payload.
        default: keys[i] = {random(), random()}; break;
        }
        values[i] = value_for<Value>(i, keys[i]);
    }

    Hash128 *input_key = nullptr, *output_key = nullptr;
    Value *input_value = nullptr, *output_value = nullptr;
    void* scratch = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    const auto allocation_count = std::max<std::size_t>(count, 1);
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaMalloc(&input_key, allocation_count * sizeof(Hash128)));
    check(cudaMalloc(&output_key, allocation_count * sizeof(Hash128)));
    check(cudaMalloc(&input_value, allocation_count * sizeof(Value)));
    check(cudaMalloc(&output_value, allocation_count * sizeof(Value)));
    std::size_t scratch_bytes = 0;
    check(beam::radix128_sort_pairs(nullptr, scratch_bytes, input_key, output_key,
                                    input_value, output_value, count, stream));
    check(cudaMalloc(&scratch, std::max<std::size_t>(scratch_bytes, 1)));

    const auto verify = [&]() {
        std::vector<std::size_t> order(count);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
            return keys[a].hi < keys[b].hi ||
                   (keys[a].hi == keys[b].hi && keys[a].lo < keys[b].lo);
        });
        std::vector<Hash128> actual_keys(count);
        std::vector<Value> actual_values(count);
        if (count) {
            check(cudaMemcpyAsync(actual_keys.data(), output_key, count * sizeof(Hash128),
                                  cudaMemcpyDeviceToHost, stream));
            check(cudaMemcpyAsync(actual_values.data(), output_value, count * sizeof(Value),
                                  cudaMemcpyDeviceToHost, stream));
        }
        check(cudaStreamSynchronize(stream));
        for (std::size_t i = 0; i < count; ++i) {
            const auto expected = order[i];
            if (actual_keys[i].hi != keys[expected].hi || actual_keys[i].lo != keys[expected].lo ||
                !equal(actual_values[i], values[expected])) {
                throw std::runtime_error("radix key, stability or original payload mismatch");
            }
        }
    };
    const auto upload = [&]() {
        if (count) {
            check(cudaMemcpyAsync(input_key, keys.data(), count * sizeof(Hash128),
                                  cudaMemcpyHostToDevice, stream));
            check(cudaMemcpyAsync(input_value, values.data(), count * sizeof(Value),
                                  cudaMemcpyHostToDevice, stream));
        }
    };
    upload();
    if (graph_replay) {
        check(cudaStreamSynchronize(stream));
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        auto bytes = scratch_bytes;
        check(beam::radix128_sort_pairs(scratch, bytes, input_key, output_key,
                                        input_value, output_value, count, stream));
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        check(cudaGraphLaunch(executable, stream));
        verify();
        std::reverse(keys.begin(), keys.end());
        std::reverse(values.begin(), values.end());
        upload();
        check(cudaGraphLaunch(executable, stream));
        verify();
    } else {
        auto bytes = scratch_bytes;
        check(beam::radix128_sort_pairs(scratch, bytes, input_key, output_key,
                                        input_value, output_value, count, stream));
        verify();
    }
    if (executable) check(cudaGraphExecDestroy(executable));
    if (graph) check(cudaGraphDestroy(graph));
    check(cudaFree(scratch));
    check(cudaFree(output_value));
    check(cudaFree(input_value));
    check(cudaFree(output_key));
    check(cudaFree(input_key));
    check(cudaStreamDestroy(stream));
}
} // namespace

int main() {
    try {
        int devices = 0;
        check(cudaGetDeviceCount(&devices));
        if (devices < 1) throw std::runtime_error("requires a physical CUDA device");
        for (int device = 0; device < devices; ++device) {
            check(cudaSetDevice(device));
            for (const auto count : {std::size_t{0}, std::size_t{1}, std::size_t{257}, std::size_t{1025}}) {
                run_case<std::uint64_t>(count, count == 1025);
                run_case<CandidateMeta>(count, count == 1025);
            }
            std::cout << "PASS full Hash128 radix keys and payloads device=" << device << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
