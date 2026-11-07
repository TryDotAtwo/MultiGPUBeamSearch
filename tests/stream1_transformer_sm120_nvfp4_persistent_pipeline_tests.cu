#include "stream1_transformer_sm120_nvfp4_persistent_pipeline.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main() {
    using Pipeline = stream1_sm120_nvfp4_persistent::Pipeline;
    constexpr std::uint32_t slices = Pipeline::kHiddenSlices;
    constexpr std::uint32_t words = Pipeline::kWordsPerBuffer;
    std::vector<std::uint32_t> source(static_cast<std::size_t>(slices) * words);
    std::vector<std::uint64_t> expected(slices, 0U);
    for (std::uint32_t slice = 0; slice < slices; ++slice) {
        for (std::uint32_t word = 0; word < words; ++word) {
            const std::uint32_t value =
                1U + slice * 131U + (word * 17U + word / 7U) % 251U;
            source[static_cast<std::size_t>(slice) * words + word] = value;
            expected[slice] += value;
        }
    }

    std::uint32_t* device_source = nullptr;
    std::uint64_t* device_checksums = nullptr;
    std::uint32_t* device_violations = nullptr;
    cudaMalloc(&device_source, source.size() * sizeof(source.front()));
    cudaMalloc(&device_checksums, expected.size() * sizeof(expected.front()));
    cudaMalloc(&device_violations, sizeof(*device_violations));
    cudaMemcpy(device_source, source.data(), source.size() * sizeof(source.front()),
               cudaMemcpyHostToDevice);
    cudaMemset(device_checksums, 0, expected.size() * sizeof(expected.front()));
    cudaMemset(device_violations, 0, sizeof(*device_violations));

    stream1_sm120_nvfp4_persistent::persistent_handoff_contract_kernel<<<
        1, Pipeline::kThreads,
        sizeof(stream1_sm120_nvfp4_persistent::HandoffStorage)>>>(
        device_source, device_checksums, device_violations);
    require(cudaDeviceSynchronize() == cudaSuccess,
            "persistent handoff contract kernel failed");

    std::vector<std::uint64_t> actual(expected.size());
    std::uint32_t violations = 0;
    cudaMemcpy(actual.data(), device_checksums,
               actual.size() * sizeof(actual.front()), cudaMemcpyDeviceToHost);
    cudaMemcpy(&violations, device_violations, sizeof(violations),
               cudaMemcpyDeviceToHost);
    require(actual == expected, "persistent handoff generation checksum mismatch");
    require(violations == 0U, "persistent handoff ownership violation");

    std::cout << "stream1_transformer_sm120_nvfp4_persistent_pipeline_tests=pass"
              << " slices=" << slices << " ring_buffers=" << Pipeline::kRingBuffers
              << " words_per_buffer=" << words << '\n';
    return EXIT_SUCCESS;
}
