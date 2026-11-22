// Compare an ordinary graph, one conditional node, and three chained nodes.
// All alternatives execute the same kernels and borrow the same scratch slot.
#include "cuda/radix128.cuh"
#include <cub/device/device_reduce.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
__global__ void select_body(cudaGraphConditionalHandle handle) {
    if (threadIdx.x == 0 && blockIdx.x == 0) cudaGraphSetConditional(handle, 1);
}
__global__ void select_three(cudaGraphConditionalHandle first,
                             cudaGraphConditionalHandle second,
                             cudaGraphConditionalHandle third,
                             unsigned selected) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cudaGraphSetConditional(first, selected == 0);
        cudaGraphSetConditional(second, selected == 1);
        cudaGraphSetConditional(third, selected == 2);
    }
}
__global__ void marker(std::uint32_t* output) {
    __shared__ std::uint32_t flags[256];
    const unsigned tid = threadIdx.x;
    flags[tid] = tid & 1U;
    __syncthreads();
    for (unsigned stride = 128; stride; stride >>= 1) {
        if (tid < stride) flags[tid] += flags[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = flags[0];
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: chained-diagnostic plain|conditional|first|last count");
        const std::string requested_mode = argv[1];
        const std::string suffix = "-prefix";
        const bool prefix = requested_mode.size() >= suffix.size() &&
            requested_mode.compare(requested_mode.size() - suffix.size(), suffix.size(), suffix) == 0;
        const std::string mode = prefix ? requested_mode.substr(0, requested_mode.size() - suffix.size()) : requested_mode;
        const int n = std::atoi(argv[2]);
        if ((mode != "plain" && mode != "conditional" && mode != "first" && mode != "last") || n <= 0 || n > 1025)
            throw std::runtime_error("invalid arguments");
        beam::Hash128 *a, *b, *c;
        std::uint32_t *x, *y, *z, *runs, *markers;
        check(cudaMalloc(&a, n*sizeof(*a))); check(cudaMalloc(&b, n*sizeof(*b)));
        check(cudaMalloc(&c, n*sizeof(*c))); check(cudaMalloc(&x, n*sizeof(*x)));
        check(cudaMalloc(&y, n*sizeof(*y))); check(cudaMalloc(&z, n*sizeof(*z)));
        check(cudaMalloc(&runs, sizeof(*runs)));
        const unsigned blocks = (n + 255) / 256;
        check(cudaMalloc(&markers, blocks*sizeof(*markers)));
        std::vector<beam::Hash128> input(n);
        std::vector<std::uint32_t> values(n, 1);
        for (int i = 0; i < n; ++i) input[i] = {static_cast<std::uint64_t>(n-i), 0};
        check(cudaMemcpy(a, input.data(), n*sizeof(*a), cudaMemcpyHostToDevice));
        check(cudaMemcpy(x, values.data(), n*sizeof(*x), cudaMemcpyHostToDevice));
        std::size_t sort_bytes=0, reduce_bytes=0;
        check(beam::radix128_sort_pairs(nullptr, sort_bytes, a, b, x, y, n, nullptr));
        check(cub::DeviceReduce::ReduceByKey(nullptr, reduce_bytes, b, c, y, z, runs,
                                             cub::Sum{}, n, nullptr));
        const auto bytes = std::max(sort_bytes, reduce_bytes);
        void* temp = nullptr; check(cudaMalloc(&temp, bytes));
        auto emit = [&](cudaStream_t stream) {
            auto scratch = bytes;
            check(beam::radix128_sort_pairs(temp, scratch, a, b, x, y, n, stream));
            scratch = bytes;
            check(cub::DeviceReduce::ReduceByKey(temp, scratch, b, c, y, z, runs,
                                                 cub::Sum{}, n, stream));
            marker<<<blocks, 256, 0, stream>>>(markers);
            check(cudaGetLastError());
        };
        cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cudaGraph_t graph = nullptr;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        if (prefix) {
            // Match the production graph's block-synchronizing work before
            // the GPU selector and conditional graph nodes.
            marker<<<blocks, 256, 0, stream>>>(markers);
            check(cudaGetLastError());
        }
        if (mode == "plain") {
            emit(stream);
        } else {
            cudaStreamCaptureStatus status;
            cudaGraph_t parent = nullptr;
            check(cudaStreamGetCaptureInfo_v2(stream, &status, nullptr, &parent, nullptr, nullptr));
            const unsigned alternatives = mode == "conditional" ? 1 : 3;
            cudaGraphConditionalHandle handles[3];
            for (unsigned i = 0; i < alternatives; ++i)
                check(cudaGraphConditionalHandleCreate(&handles[i], parent, 0, cudaGraphCondAssignDefault));
            if (alternatives == 1) select_body<<<1,1,0,stream>>>(handles[0]);
            else select_three<<<1,1,0,stream>>>(handles[0], handles[1], handles[2], mode == "first" ? 0 : 2);
            check(cudaGetLastError());
            for (unsigned alternative = 0; alternative < alternatives; ++alternative) {
            const cudaGraphNode_t* dependencies = nullptr;
            std::size_t dependency_count = 0;
            check(cudaStreamGetCaptureInfo_v2(stream, &status, nullptr, &parent, &dependencies,
                                               &dependency_count));
            cudaGraphNodeParams params{};
            params.type = cudaGraphNodeTypeConditional;
            params.conditional.handle = handles[alternative];
            params.conditional.type = cudaGraphCondTypeIf;
            params.conditional.size = 1;
            cudaGraphNode_t conditional;
            check(cudaGraphAddNode(&conditional, parent, dependencies, dependency_count, &params));
            cudaStream_t body;
            check(cudaStreamCreateWithFlags(&body, cudaStreamNonBlocking));
            check(cudaStreamBeginCaptureToGraph(body, params.conditional.phGraph_out[0], nullptr,
                                               nullptr, 0, cudaStreamCaptureModeThreadLocal));
            emit(body);
            cudaGraph_t body_graph = nullptr;
            check(cudaStreamEndCapture(body, &body_graph));
            check(cudaStreamDestroy(body));
            check(cudaStreamUpdateCaptureDependencies(stream, &conditional, 1,
                                                      cudaStreamSetCaptureDependencies));
            }
        }
        check(cudaStreamEndCapture(stream, &graph));
        cudaGraphExec_t executable;
        check(cudaGraphInstantiate(&executable, graph, 0));
        for (unsigned repeat = 0; repeat < 2; ++repeat) {
            check(cudaGraphLaunch(executable, stream));
            check(cudaStreamSynchronize(stream));
            std::uint32_t actual_runs = 0;
            check(cudaMemcpy(&actual_runs, runs, sizeof(actual_runs), cudaMemcpyDeviceToHost));
            if (actual_runs != static_cast<unsigned>(n)) throw std::runtime_error("wrong run count");
            std::vector<beam::Hash128> actual_keys(n);
            std::vector<std::uint32_t> actual_values(n);
            check(cudaMemcpy(actual_keys.data(), c, n*sizeof(*c), cudaMemcpyDeviceToHost));
            check(cudaMemcpy(actual_values.data(), z, n*sizeof(*z), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i)
                if (actual_keys[i].lo != static_cast<std::uint64_t>(i + 1) ||
                    actual_keys[i].hi != 0 || actual_values[i] != 1)
                    throw std::runtime_error("wrong reduced key/value");
            std::vector<std::uint32_t> counts(blocks);
            check(cudaMemcpy(counts.data(), markers, blocks*sizeof(*markers), cudaMemcpyDeviceToHost));
            if (!std::all_of(counts.begin(), counts.end(), [](auto count){return count==128;}))
                throw std::runtime_error("wrong shared-memory reduction");
        }
        check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
        check(cudaStreamDestroy(stream));
        void* allocations[] = {a, b, c, x, y, z, runs, markers, temp};
        for (void* ptr : allocations) check(cudaFree(ptr));
        std::printf("DIAGNOSTIC_OK mode=%s n=%d repeats=2\n", requested_mode.c_str(), n);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what()); return 1;
    }
}
