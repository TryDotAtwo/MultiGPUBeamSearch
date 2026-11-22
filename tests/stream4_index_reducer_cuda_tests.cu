#include "../cuda/stream4_index.cuh"

#include <cuda_runtime.h>

#include <array>
#include <iostream>
#include <stdexcept>

using namespace beam;

namespace {
void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

__global__ void probe(Stream4BestIndex best, std::uint32_t* output) {
    if (blockIdx.x || threadIdx.x) return;
    output[0] = best(0, 1);                    // parent tie break
    output[1] = best(1, 2);                    // route tie break
    output[2] = best(UINT32_MAX, 1);           // padded left input
    output[3] = best(1, UINT32_MAX);           // padded right input
    output[4] = best(0, 3);                    // index just beyond live range
    output[5] = best(3, 0);
    output[6] = best(3, UINT32_MAX);           // no valid input
    output[7] = best(1024, 0);                // arbitrary inactive lane value
}
}

int main() {
    try {
        int devices = 0;
        check(cudaGetDeviceCount(&devices));
        if (devices < 2) throw std::runtime_error("requires two physical GPUs");
        const std::array<CandidateMeta, 3> host{{
            CandidateMeta{{7, 9}, 10, 3, 4},
            CandidateMeta{{7, 9}, 8, 3, 6},
            CandidateMeta{{7, 9}, 8, 3, 1},
        }};
        const std::array<std::uint32_t, 8> expected{{1, 2, 1, 1, 0, 0, UINT32_MAX, 0}};
        for (int device = 0; device < 2; ++device) {
            check(cudaSetDevice(device));
            CandidateMeta* metadata = nullptr;
            std::uint32_t *count = nullptr, *output = nullptr;
            check(cudaMalloc(&metadata, sizeof(host)));
            check(cudaMalloc(&count, sizeof(std::uint32_t)));
            check(cudaMalloc(&output, sizeof(expected)));
            check(cudaMemcpy(metadata, host.data(), sizeof(host), cudaMemcpyHostToDevice));
            std::uint32_t live = 3;
            check(cudaMemcpy(count, &live, sizeof(live), cudaMemcpyHostToDevice));
            probe<<<1, 1>>>(Stream4BestIndex{metadata, count, 3}, output);
            check(cudaGetLastError());
            std::array<std::uint32_t, 8> actual{};
            check(cudaMemcpy(actual.data(), output, sizeof(actual), cudaMemcpyDeviceToHost));
            if (actual != expected) throw std::runtime_error("index reducer mismatch");
            live = 0;
            check(cudaMemcpy(count, &live, sizeof(live), cudaMemcpyHostToDevice));
            probe<<<1, 1>>>(Stream4BestIndex{metadata, count, 3}, output);
            check(cudaGetLastError());
            check(cudaMemcpy(actual.data(), output, sizeof(actual), cudaMemcpyDeviceToHost));
            for (auto result : actual)
                if (result != UINT32_MAX) throw std::runtime_error("empty reducer accepted an index");
            check(cudaFree(output)); check(cudaFree(count)); check(cudaFree(metadata));
            std::cout << "gpu=" << device << " index_reducer=pass\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
