#include "stream1_transformer_sm120_fp8.hpp"
#include "../cuda/stream1.hpp"

#include "cuda_check.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct DeviceBuffer {
    void* pointer = nullptr;
    explicit DeviceBuffer(std::size_t bytes) {
        if (bytes != 0U) {
            BEAM_CUDA_CHECK(cudaMalloc(&pointer, bytes));
        }
    }
    ~DeviceBuffer() { cudaFree(pointer); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

void benchmark_shape(std::uint32_t rows, std::uint32_t input_cols, std::uint32_t output_cols) {
    constexpr int warmups = 5;
    constexpr int iterations = 30;
    const std::size_t input_elements = static_cast<std::size_t>(rows) * input_cols;
    const std::size_t weight_elements = static_cast<std::size_t>(input_cols) * output_cols;
    const std::size_t output_elements = static_cast<std::size_t>(rows) * output_cols;
    DeviceBuffer input(input_elements * sizeof(half));
    DeviceBuffer weight(weight_elements * sizeof(half));
    DeviceBuffer output(output_elements * sizeof(half));
    DeviceBuffer fp16_output(output_elements * sizeof(half));
    DeviceBuffer quantized_input(input_elements);
    DeviceBuffer quantized_weight(weight_elements);
    DeviceBuffer input_scales(
        stream1_transformer_sm120_fp8_input_scale_elements(rows, input_cols) * sizeof(float));
    DeviceBuffer weight_scales(
        stream1_transformer_sm120_fp8_weight_scale_elements(input_cols, output_cols) * sizeof(float));
    const std::size_t workspace_bytes =
        stream1_transformer_sm120_fp8_workspace_bytes(rows, input_cols, output_cols);
    DeviceBuffer workspace(workspace_bytes);
    BEAM_CUDA_CHECK(cudaMemset(input.pointer, 0x35, input_elements * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMemset(weight.pointer, 0x31, weight_elements * sizeof(half)));
    stream1_transformer_sm120_fp8_quantize_weight_cuda(
        static_cast<const half*>(weight.pointer),
        static_cast<std::uint8_t*>(quantized_weight.pointer),
        static_cast<float*>(weight_scales.pointer),
        input_cols, output_cols, 1.0f, nullptr);
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());

    auto run_once = [&]() {
        stream1_transformer_sm120_fp8_linear_cuda(
            static_cast<const half*>(input.pointer),
            static_cast<std::uint8_t*>(quantized_input.pointer),
            static_cast<float*>(input_scales.pointer),
            static_cast<const std::uint8_t*>(quantized_weight.pointer),
            static_cast<const float*>(weight_scales.pointer),
            static_cast<half*>(output.pointer),
            rows, input_cols, output_cols, 1.0f,
            workspace.pointer, workspace_bytes, nullptr);
    };
    for (int i = 0; i < warmups; ++i) {
        run_once();
    }
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    BEAM_CUDA_CHECK(cudaEventCreate(&start));
    BEAM_CUDA_CHECK(cudaEventCreate(&stop));
    BEAM_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        run_once();
    }
    BEAM_CUDA_CHECK(cudaEventRecord(stop));
    BEAM_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    BEAM_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const double average_ms = static_cast<double>(elapsed_ms) / iterations;
    const double tflops = 2.0 * static_cast<double>(rows) * input_cols * output_cols /
        (average_ms * 1.0e9);

    auto run_fp16_once = [&]() {
        stream1_cutlass_linear_cuda(
            static_cast<const half*>(input.pointer),
            static_cast<const half*>(weight.pointer),
            static_cast<half*>(fp16_output.pointer),
            rows, input_cols, output_cols, STREAM1_DTYPE_FP16, nullptr);
    };
    for (int i = 0; i < warmups; ++i) {
        run_fp16_once();
    }
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    BEAM_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        run_fp16_once();
    }
    BEAM_CUDA_CHECK(cudaEventRecord(stop));
    BEAM_CUDA_CHECK(cudaEventSynchronize(stop));
    float fp16_elapsed_ms = 0.0f;
    BEAM_CUDA_CHECK(cudaEventElapsedTime(&fp16_elapsed_ms, start, stop));
    const double fp16_average_ms = static_cast<double>(fp16_elapsed_ms) / iterations;
    std::cout << "sm120_fp8_shape"
              << " m=" << rows << " n=" << output_cols << " k=" << input_cols
              << " end_to_end_ms=" << std::fixed << std::setprecision(6) << average_ms
              << " effective_tflops=" << std::setprecision(3) << tflops
              << " fp16_ms=" << std::setprecision(6) << fp16_average_ms
              << " fp8_speedup=" << std::setprecision(3) << fp16_average_ms / average_ms
              << " workspace_bytes=" << workspace_bytes << "\n";
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
}

}  // namespace

int main(int argc, char** argv) {
    if (!stream1_transformer_sm120_fp8_supported()) {
        std::cerr << "SM120 FP8 benchmark requires a physical SM120 GPU and sm_120a build\n";
        return EXIT_FAILURE;
    }
    const std::uint32_t rows = argc > 1 ? static_cast<std::uint32_t>(std::stoul(argv[1])) : 51072U;
    benchmark_shape(rows, 256U, 768U);
    benchmark_shape(rows, 256U, 1024U);
    benchmark_shape(rows, 256U, 256U);
    benchmark_shape(rows, 1024U, 256U);
    return EXIT_SUCCESS;
}
