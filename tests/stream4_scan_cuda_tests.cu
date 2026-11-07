#include "../cuda/stream4_scan.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

__global__ void block_prefix_kernel(const std::uint32_t* flags, std::uint32_t* prefixes, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const auto prefix = beam::detail::stream4_block_prefix(i < n ? flags[i] : 0U);
    if (i < n) prefixes[i] = prefix;
}

void exercise(int gpu) {
    check(cudaSetDevice(gpu));
    constexpr unsigned max_count = 65537;
    std::uint32_t *input = nullptr, *output = nullptr, *total = nullptr;
    check(cudaMalloc(&input, max_count * sizeof(*input)));
    check(cudaMalloc(&output, max_count * sizeof(*output)));
    check(cudaMalloc(&total, sizeof(*total)));
    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::size_t temp_bytes = 0;
    check(beam::detail::stream4_scan_temp_bytes(max_count, temp_bytes));
    if (!temp_bytes) throw std::runtime_error("nonempty scan has no temporary storage");
    void* temp = nullptr; check(cudaMalloc(&temp, temp_bytes));
    if (beam::detail::stream4_scan_counts(input, output, total, max_count,
                                          temp, temp_bytes - 1U, stream) == cudaSuccess)
        throw std::runtime_error("scan accepted insufficient temporary storage");
    (void)cudaGetLastError();
    for (unsigned n : {0U, 1U, 2U, 255U, 256U, 257U, 1025U, 4097U, max_count}) {
        std::size_t required = 0;
        check(beam::detail::stream4_scan_temp_bytes(n, required));
        if (required > temp_bytes) throw std::runtime_error("maximum scan scratch is insufficient");
        std::vector<std::uint32_t> source(n), observed(n);
        for (unsigned i = 0; i < n; ++i) source[i] = (i * 17U + n) % 5U;
        if (n) check(cudaMemcpyAsync(input, source.data(), n * sizeof(*input), cudaMemcpyHostToDevice, stream));
        cudaGraph_t graph; cudaGraphExec_t executable;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        check(beam::detail::stream4_scan_counts(input, output, total, n, temp, temp_bytes, stream));
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (unsigned replay = 0; replay < 2; ++replay) {
            if (n) {
                for (unsigned i = 0; i < n; ++i) source[i] = (i * 17U + n + replay) % 5U;
                check(cudaMemcpyAsync(input, source.data(), n * sizeof(*input), cudaMemcpyHostToDevice, stream));
            }
            check(cudaMemsetAsync(total, 0xff, sizeof(*total), stream));
            check(cudaGraphLaunch(executable, stream));
            std::uint32_t observed_total = 0;
            if (n) check(cudaMemcpyAsync(observed.data(), output, n * sizeof(*output), cudaMemcpyDeviceToHost, stream));
            check(cudaMemcpyAsync(&observed_total, total, sizeof(*total), cudaMemcpyDeviceToHost, stream));
            check(cudaStreamSynchronize(stream));
            std::uint32_t expected = 0;
            for (unsigned i = 0; i < n; ++i) {
                if (observed[i] != expected) throw std::runtime_error("device exclusive scan mismatch");
                expected += source[i];
            }
            if (observed_total != expected) throw std::runtime_error("device total mismatch");
        }
        check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
    }
    for (unsigned n : {1U, 255U, 256U, 257U, 4097U, max_count}) {
        std::vector<std::uint32_t> flags(n), observed(n);
        for (unsigned i = 0; i < n; ++i) flags[i] = ((i * 13U + n) % 7U) < 3U;
        check(cudaMemcpyAsync(input, flags.data(), n * sizeof(*input), cudaMemcpyHostToDevice, stream));
        block_prefix_kernel<<<(n + 255U) / 256U, 256, 0, stream>>>(input, output, n);
        check(cudaGetLastError());
        check(cudaMemcpyAsync(observed.data(), output, n * sizeof(*output), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        std::uint32_t expected = 0;
        for (unsigned i = 0; i < n; ++i) {
            if (i % 256U == 0) expected = 0;
            if (observed[i] != expected) throw std::runtime_error("block prefix mismatch");
            expected += flags[i];
        }
    }
    check(cudaFree(temp)); check(cudaStreamDestroy(stream));
    check(cudaFree(total)); check(cudaFree(output)); check(cudaFree(input));
    std::cout << "gpu=" << gpu << " scan_graph_replay_and_block_prefix=pass\n";
}
}

int main() {
    try {
        int devices = 0; check(cudaGetDeviceCount(&devices));
        if (devices != 2) throw std::runtime_error("requires two physical GPUs");
        for (int gpu = 0; gpu < devices; ++gpu) exercise(gpu);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
}
