#include "stream1_transformer_sm120_fp8.hpp"

#include "cuda_check.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

float host_value(std::uint32_t row, std::uint32_t col) {
    const int signed_bucket = static_cast<int>((row * 17U + col * 13U) % 31U) - 15;
    const float row_scale = (row % 3U == 0U) ? 0.03125f : ((row % 3U == 1U) ? 0.25f : 1.0f);
    return static_cast<float>(signed_bucket) * row_scale * (1.0f / 16.0f);
}

}  // namespace

int main() {
    constexpr std::uint32_t rows = 128;
    constexpr std::uint32_t input_cols = 128;
    constexpr std::uint32_t output_cols = 128;

    require(
        stream1_transformer_sm120_fp8_supported(),
        "SM120 FP8 test requires an SM120 device and an sm_120a build");
    require(
        stream1_transformer_sm120_fp8_input_scale_elements(rows, input_cols) == rows,
        "one dynamic activation scale is required per row for K=128");
    require(
        stream1_transformer_sm120_fp8_weight_scale_elements(input_cols, output_cols) == 1U,
        "one weight scale is required for a 128x128 weight tile");

    std::vector<half> input(static_cast<std::size_t>(rows) * input_cols);
    std::vector<half> weight(static_cast<std::size_t>(input_cols) * output_cols, __float2half(0.0f));
    std::vector<float> weight_fp32(static_cast<std::size_t>(input_cols) * output_cols, 0.0f);
    std::vector<half> expected(static_cast<std::size_t>(rows) * output_cols, __float2half(0.0f));
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t col = 0; col < input_cols; ++col) {
            const float value = host_value(row, col);
            input[static_cast<std::size_t>(row) * input_cols + col] = __float2half(value);
            expected[static_cast<std::size_t>(row) * output_cols + col] = __float2half(value);
        }
    }
    for (std::uint32_t col = 0; col < input_cols; ++col) {
        weight[static_cast<std::size_t>(col) * output_cols + col] = __float2half(1.0f);
        weight_fp32[static_cast<std::size_t>(col) * output_cols + col] = 1.0f;
    }

    half* d_input = nullptr;
    half* d_weight = nullptr;
    float* d_weight_fp32 = nullptr;
    half* d_output = nullptr;
    std::uint8_t* d_quantized_input = nullptr;
    std::uint8_t* d_quantized_weight = nullptr;
    std::uint8_t* d_quantized_weight_fp32 = nullptr;
    float* d_input_scales = nullptr;
    float* d_weight_scales = nullptr;
    float* d_weight_scales_fp32 = nullptr;
    void* d_workspace = nullptr;

    BEAM_CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_weight, weight.size() * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_weight_fp32, weight_fp32.size() * sizeof(float)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_output, expected.size() * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_quantized_input, input.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&d_quantized_weight, weight.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&d_quantized_weight_fp32, weight_fp32.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&d_input_scales, rows * sizeof(float)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_weight_scales, sizeof(float)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_weight_scales_fp32, sizeof(float)));
    const std::size_t workspace_bytes =
        stream1_transformer_sm120_fp8_workspace_bytes(rows, input_cols, output_cols);
    if (workspace_bytes != 0U) {
        BEAM_CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));
    }
    BEAM_CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(half), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(half), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(
        d_weight_fp32, weight_fp32.data(), weight_fp32.size() * sizeof(float), cudaMemcpyHostToDevice));

    stream1_transformer_sm120_fp8_quantize_weight_cuda(
        d_weight,
        d_quantized_weight,
        d_weight_scales,
        input_cols,
        output_cols,
        1.0f,
        nullptr);
    stream1_transformer_sm120_fp8_quantize_weight_from_fp32_cuda(
        d_weight_fp32,
        d_quantized_weight_fp32,
        d_weight_scales_fp32,
        input_cols,
        output_cols,
        1.0f,
        nullptr);
    std::vector<std::uint8_t> encoded_half(weight.size());
    std::vector<std::uint8_t> encoded_fp32(weight_fp32.size());
    float encoded_half_scale = 0.0f;
    float encoded_fp32_scale = 0.0f;
    BEAM_CUDA_CHECK(cudaMemcpy(
        encoded_half.data(), d_quantized_weight, encoded_half.size(), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(
        encoded_fp32.data(), d_quantized_weight_fp32, encoded_fp32.size(), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(
        &encoded_half_scale, d_weight_scales, sizeof(float), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(
        &encoded_fp32_scale, d_weight_scales_fp32, sizeof(float), cudaMemcpyDeviceToHost));
    require(encoded_half == encoded_fp32, "exact FP32 weights must encode identically to FP16 values");
    require(encoded_half_scale == encoded_fp32_scale, "FP32 and FP16 exact weight scales differ");
    stream1_transformer_sm120_fp8_linear_cuda(
        d_input,
        d_quantized_input,
        d_input_scales,
        d_quantized_weight,
        d_weight_scales,
        d_output,
        rows,
        input_cols,
        output_cols,
        1.0f,
        d_workspace,
        workspace_bytes,
        nullptr);
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> output(expected.size());
    std::vector<std::uint8_t> quantized_input_probe(8);
    std::vector<std::uint8_t> quantized_weight_probe(8);
    float input_scale_probe = 0.0f;
    float weight_scale_probe = 0.0f;
    BEAM_CUDA_CHECK(cudaMemcpy(
        quantized_input_probe.data(), d_quantized_input,
        quantized_input_probe.size(), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(
        quantized_weight_probe.data(), d_quantized_weight,
        quantized_weight_probe.size(), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(
        &input_scale_probe, d_input_scales, sizeof(float), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(
        &weight_scale_probe, d_weight_scales, sizeof(float), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size() * sizeof(half), cudaMemcpyDeviceToHost));
    double squared_error = 0.0;
    double squared_reference = 0.0;
    float max_abs_error = 0.0f;
    for (std::size_t index = 0; index < output.size(); ++index) {
        const float got = __half2float(output[index]);
        const float want = __half2float(expected[index]);
        const float error = got - want;
        require(std::isfinite(got), "SM120 FP8 GEMM produced a non-finite value");
        squared_error += static_cast<double>(error) * error;
        squared_reference += static_cast<double>(want) * want;
        max_abs_error = std::max(max_abs_error, std::abs(error));
    }
    const double nmse = squared_error / squared_reference;
    std::cerr << "stream1_transformer_sm120_fp8_metrics"
              << " nmse=" << nmse
              << " max_abs_error=" << max_abs_error
              << " input_scale0=" << input_scale_probe
              << " weight_scale0=" << weight_scale_probe
              << " qinput0=" << static_cast<unsigned>(quantized_input_probe[0])
              << " qweight0=" << static_cast<unsigned>(quantized_weight_probe[0])
              << " output0=" << __half2float(output[0])
              << "\n";
    require(nmse < 1.0e-3, "SM120 block-scaled FP8 identity GEMM NMSE exceeded 1e-3");
    require(max_abs_error < 0.08f, "SM120 block-scaled FP8 identity GEMM max error exceeded 0.08");

    BEAM_CUDA_CHECK(cudaMemset(d_input, 0, input.size() * sizeof(half)));
    stream1_transformer_sm120_fp8_linear_cuda(
        d_input,
        d_quantized_input,
        d_input_scales,
        d_quantized_weight,
        d_weight_scales,
        d_output,
        rows,
        input_cols,
        output_cols,
        1.0f,
        d_workspace,
        workspace_bytes,
        nullptr);
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    BEAM_CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size() * sizeof(half), cudaMemcpyDeviceToHost));
    require(
        std::all_of(output.begin(), output.end(), [](half value) { return __half2float(value) == 0.0f; }),
        "zero activation tiles must remain exactly zero after dynamic FP8 quantization");

    std::cout << "stream1_transformer_sm120_fp8_cuda_tests=pass"
              << " nmse=" << nmse
              << " max_abs_error=" << max_abs_error
              << " workspace_bytes=" << workspace_bytes
              << "\n";

    cudaFree(d_workspace);
    cudaFree(d_weight_scales_fp32);
    cudaFree(d_weight_scales);
    cudaFree(d_input_scales);
    cudaFree(d_quantized_weight);
    cudaFree(d_quantized_weight_fp32);
    cudaFree(d_quantized_input);
    cudaFree(d_output);
    cudaFree(d_weight);
    cudaFree(d_weight_fp32);
    cudaFree(d_input);
    return 0;
}
