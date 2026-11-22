#include "stream1.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <array>
#include <iostream>
#include <limits>
#include <string>
#include <cstdlib>

namespace beam {
__global__ void stream1_transformer_score_quantize_kernel(
    const half*, const half*, const std::uint32_t*, std::uint32_t*,
    std::uint32_t, std::uint32_t, Stream1TransformerDims, std::uint32_t*);
__global__ void stream1_transformer_score_quantize_graph_job_kernel(
    const half*, const half*, const std::uint32_t*, const std::uint32_t*, std::uint32_t*,
    std::uint32_t, std::uint32_t, std::uint32_t, Stream1TransformerDims, std::uint32_t*);
}

int main(int argc, char** argv) {
    const std::string mode = argc >= 2 ? argv[1] : "finite";
    const bool graph_job = argc == 3 && std::string(argv[2]) == "--graph-job";
    if (argc > 3 || (argc == 3 && !graph_job)) return 2;
    if (mode != "finite" && mode != "nan" && mode != "posinf" && mode != "neginf") return 2;
    auto check = [](cudaError_t status) {
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    };
    try {
        std::array<half, beam::MOVE_COUNT> logits{}, bias{};
        for (auto& value : logits) value = __float2half(1.0f);
        if (mode == "nan") logits[0] = __float2half(std::numeric_limits<float>::quiet_NaN());
        if (mode == "posinf") logits[0] = __float2half(std::numeric_limits<float>::infinity());
        if (mode == "neginf") logits[0] = __float2half(-std::numeric_limits<float>::infinity());
        half *device_logits = nullptr, *device_bias = nullptr;
        std::uint32_t *device_count = nullptr, *device_scores = nullptr, *device_error = nullptr;
        std::uint32_t* device_job = nullptr;
        std::cerr << "score_test_stage=alloc mode=" << mode << std::endl;
        check(cudaMalloc(&device_logits, sizeof(logits)));
        check(cudaMalloc(&device_bias, sizeof(bias)));
        check(cudaMalloc(&device_count, sizeof(std::uint32_t)));
        check(cudaMalloc(&device_scores, beam::MOVE_COUNT * sizeof(std::uint32_t)));
        check(cudaMalloc(&device_error, sizeof(std::uint32_t)));
        check(cudaMemset(device_error, 0, sizeof(std::uint32_t)));
        check(cudaMalloc(&device_job, sizeof(std::uint32_t)));
        check(cudaMemset(device_job, 0, sizeof(std::uint32_t)));
        const std::uint32_t count = 1;
        check(cudaMemcpy(device_logits, logits.data(), sizeof(logits), cudaMemcpyHostToDevice));
        check(cudaMemcpy(device_bias, bias.data(), sizeof(bias), cudaMemcpyHostToDevice));
        check(cudaMemcpy(device_count, &count, sizeof(count), cudaMemcpyHostToDevice));
        beam::Stream1TransformerDims dims{};
        dims.output_dim = beam::MOVE_COUNT;
        dims.dtype = beam::STREAM1_DTYPE_FP16;
        cudaStream_t stream = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        if (graph_job) check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::cerr << "score_test_stage=launch mode=" << mode << std::endl;
        auto launch = [&]() {
            if (graph_job) {
                beam::stream1_transformer_score_quantize_graph_job_kernel<<<1, 128, 0, stream>>>(
                    device_logits, device_bias, device_count, device_job, device_scores, 1, 1, 0, dims, device_error);
            } else {
                beam::stream1_transformer_score_quantize_kernel<<<1, 128, 0, stream>>>(
                    device_logits, device_bias, device_count, device_scores, 1, 0, dims, device_error);
            }
        };
        if (graph_job) {
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            launch();
            check(cudaStreamEndCapture(stream, &graph));
            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        }
        auto submit = [&]() {
            if (graph_job) check(cudaGraphLaunch(executable, stream));
            else launch();
        };
        submit();
        check(cudaGetLastError());
        std::cerr << "score_test_stage=synchronize mode=" << mode << std::endl;
        const auto status = cudaDeviceSynchronize();
        std::cerr << "score_test_stage=synchronized status=" << static_cast<int>(status) << std::endl;
        check(status);
        std::uint32_t numeric_error = 0;
        check(cudaMemcpy(&numeric_error, device_error, sizeof(numeric_error), cudaMemcpyDeviceToHost));
        if (mode != "finite") {
            if (numeric_error != 1) {
                std::cerr << "nonfinite score did not set sticky numeric error\n";
                return 1;
            }
            // A subsequent finite invocation must NOT erase the failure.
            logits[0] = __float2half(1.0f);
            check(cudaMemcpy(device_logits, logits.data(), sizeof(logits), cudaMemcpyHostToDevice));
            submit();
            check(cudaGetLastError()); check(cudaDeviceSynchronize());
            check(cudaMemcpy(&numeric_error, device_error, sizeof(numeric_error), cudaMemcpyDeviceToHost));
            if (numeric_error != 1) return 1;
            std::cout << "sticky_nonfinite_error=" << mode << std::endl;
        } else {
            if (numeric_error != 0) return 1;
        }
        std::array<std::uint32_t, beam::MOVE_COUNT> scores{};
        check(cudaMemcpy(scores.data(), device_scores, sizeof(scores), cudaMemcpyDeviceToHost));
        for (auto value : scores) if (value != beam::SCORE_SCALE) return 1;
        if (graph_job) {
            check(cudaGraphExecDestroy(executable));
            check(cudaGraphDestroy(graph));
            check(cudaStreamDestroy(stream));
            std::cout << "captured_graph_replay=pass\n";
        }
        check(cudaFree(device_logits)); check(cudaFree(device_bias));
        check(cudaFree(device_count)); check(cudaFree(device_scores));
        check(cudaFree(device_error));
        check(cudaFree(device_job));
        if (mode == "finite") std::cout << "finite_scores_unchanged=pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
