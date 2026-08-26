#include <cuda_runtime.h>

#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr int kSfVector = 16;
constexpr float kE2m1Max = 6.0f;
constexpr int kWarpsPerBlock = 8;

void check(cudaError_t result, const char* what) {
  if (result != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(result) << '\n';
    std::exit(1);
  }
}

// One warp owns one contiguous K16 vector. This is the exact numerical
// operation needed by the FF1(ReLU) -> FF2 shared-memory handoff. Only the
// destination changes when this is embedded into the fused kernel.
__global__ void relu_quantize_k16(
    const float* input,
    std::uint8_t* packed,
    std::uint8_t* scales,
    int vectors) {
  const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp_global >= vectors) {
    return;
  }

  float value = 0.0f;
  if (lane < kSfVector) {
    value = fmaxf(input[warp_global * kSfVector + lane], 0.0f);
  }
  float amax = value;
  constexpr unsigned kHalfWarpMask = 0x0000ffffu;
#pragma unroll
  for (int delta = 8; delta >= 1; delta >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(kHalfWarpMask, amax, delta));
  }

  std::uint8_t scale_raw = 0;
  std::uint8_t quant_raw = 0;
  if (lane < kSfVector && amax > 0.0f) {
    const cutlass::float_ue4m3_t scale(amax / kE2m1Max);
    const float scale_fp32 = static_cast<float>(scale);
    const float reciprocal = scale_fp32 > 0.0f ? 1.0f / scale_fp32 : FLT_MAX;
    const cutlass::float_e2m1_t quant(value * reciprocal);
    scale_raw = scale.raw();
    quant_raw = quant.raw() & 0x0fu;
  }
  if (lane == 0) {
    scales[warp_global] = scale_raw;
  }
  std::uint8_t lo = 0;
  std::uint8_t hi = 0;
  if (lane < kSfVector) {
    const int pair_lane = lane & 7;
    lo = static_cast<std::uint8_t>(
        __shfl_sync(kHalfWarpMask, static_cast<unsigned>(quant_raw), pair_lane * 2));
    hi = static_cast<std::uint8_t>(
        __shfl_sync(kHalfWarpMask, static_cast<unsigned>(quant_raw), pair_lane * 2 + 1));
  }
  if (lane < 8) {
    packed[warp_global * 8 + lane] = static_cast<std::uint8_t>(lo | (hi << 4));
  }
}

__global__ void scalar_device_reference(
    const float* input,
    const std::uint8_t* scales,
    std::uint8_t* quant_raw,
    std::size_t elements) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const float value = fmaxf(input[index], 0.0f);
  const auto scale = cutlass::float_ue4m3_t::bitcast(scales[index / kSfVector]);
  const float scale_fp32 = static_cast<float>(scale);
  const float reciprocal = scale_fp32 > 0.0f ? 1.0f / scale_fp32 : FLT_MAX;
  quant_raw[index] = static_cast<std::uint8_t>(
      cutlass::float_e2m1_t(value * reciprocal).raw() & 0x0fu);
}

}  // namespace

int main(int argc, char** argv) {
  const int rows = argc > 1 ? std::atoi(argv[1]) : 21888;
  const int cols = argc > 2 ? std::atoi(argv[2]) : 1024;
  const int iterations = argc > 3 ? std::atoi(argv[3]) : 100;
  if (rows <= 0 || cols <= 0 || cols % kSfVector != 0 || iterations <= 0) {
    std::cerr << "usage: rows>0 cols>0,multiple-of-16 iterations>0\n";
    return 2;
  }
  const int vectors = rows * (cols / kSfVector);
  const std::size_t elements = static_cast<std::size_t>(rows) * cols;
  std::vector<float> host_input(elements);
  std::mt19937 rng(0x120);
  std::normal_distribution<float> normal(0.0f, 2.0f);
  for (auto& value : host_input) {
    value = normal(rng);
  }
  // Exercise exact zero and extreme-value groups as correctness sentinels.
  std::fill_n(host_input.begin(), kSfVector, -1.0f);
  host_input[kSfVector] = 100.0f;

  float* device_input = nullptr;
  std::uint8_t* device_packed = nullptr;
  std::uint8_t* device_scales = nullptr;
  std::uint8_t* device_reference = nullptr;
  check(cudaMalloc(&device_input, elements * sizeof(float)), "cudaMalloc input");
  check(cudaMalloc(&device_packed, static_cast<std::size_t>(vectors) * 8), "cudaMalloc packed");
  check(cudaMalloc(&device_scales, vectors), "cudaMalloc scales");
  check(cudaMalloc(&device_reference, elements), "cudaMalloc reference");
  check(cudaMemcpy(device_input, host_input.data(), elements * sizeof(float), cudaMemcpyHostToDevice),
        "copy input");

  const int threads = kWarpsPerBlock * 32;
  const int blocks = (vectors + kWarpsPerBlock - 1) / kWarpsPerBlock;
  relu_quantize_k16<<<blocks, threads>>>(device_input, device_packed, device_scales, vectors);
  scalar_device_reference<<<static_cast<int>((elements + 255) / 256), 256>>>(
      device_input, device_scales, device_reference, elements);
  check(cudaDeviceSynchronize(), "warmup");

  cudaEvent_t start{}, stop{};
  check(cudaEventCreate(&start), "event start");
  check(cudaEventCreate(&stop), "event stop");
  check(cudaEventRecord(start), "record start");
  for (int i = 0; i < iterations; ++i) {
    relu_quantize_k16<<<blocks, threads>>>(device_input, device_packed, device_scales, vectors);
  }
  check(cudaEventRecord(stop), "record stop");
  check(cudaEventSynchronize(stop), "sync stop");
  float elapsed_ms = 0.0f;
  check(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed");

  std::vector<std::uint8_t> host_packed(static_cast<std::size_t>(vectors) * 8);
  std::vector<std::uint8_t> host_scales(vectors);
  std::vector<std::uint8_t> host_device_reference(elements);
  check(cudaMemcpy(host_packed.data(), device_packed, host_packed.size(), cudaMemcpyDeviceToHost),
        "copy packed");
  check(cudaMemcpy(host_scales.data(), device_scales, host_scales.size(), cudaMemcpyDeviceToHost),
        "copy scales");
  check(cudaMemcpy(host_device_reference.data(), device_reference, host_device_reference.size(),
                   cudaMemcpyDeviceToHost), "copy reference");

  std::size_t mismatches = 0;
  std::size_t scale_mismatches = 0;
  std::size_t value_mismatches = 0;
  std::size_t device_reference_mismatches = 0;
  double squared_error = 0.0;
  double squared_signal = 0.0;
  for (int vector = 0; vector < vectors; ++vector) {
    float amax = 0.0f;
    for (int i = 0; i < kSfVector; ++i) {
      amax = std::max(amax, std::max(host_input[vector * kSfVector + i], 0.0f));
    }
    const cutlass::float_ue4m3_t expected_scale(amax > 0.0f ? amax / kE2m1Max : 0.0f);
    if (host_scales[vector] != expected_scale.raw()) {
      ++mismatches;
      ++scale_mismatches;
    }
    const float scale = static_cast<float>(expected_scale);
    for (int i = 0; i < kSfVector; ++i) {
      const float value = std::max(host_input[vector * kSfVector + i], 0.0f);
      const float reciprocal = scale > 0.0f ? 1.0f / scale : FLT_MAX;
      const cutlass::float_e2m1_t expected(value * reciprocal);
      const auto byte = host_packed[static_cast<std::size_t>(vector) * 8 + i / 2];
      const auto actual_raw = static_cast<std::uint8_t>((byte >> ((i & 1) * 4)) & 0x0fu);
      if (actual_raw != expected.raw()) {
        ++mismatches;
        ++value_mismatches;
      }
      if (actual_raw != host_device_reference[static_cast<std::size_t>(vector) * kSfVector + i]) {
        ++device_reference_mismatches;
      }
      const cutlass::float_e2m1_t actual = cutlass::float_e2m1_t::bitcast(actual_raw);
      const float reconstructed = static_cast<float>(actual) * scale;
      const double error = static_cast<double>(reconstructed) - value;
      squared_error += error * error;
      squared_signal += static_cast<double>(value) * value;
    }
  }

  const double per_iteration_ms = elapsed_ms / iterations;
  const double input_gib_s = elements * sizeof(float) / (per_iteration_ms * 1.0e6);
  const double output_gib_s = (host_packed.size() + host_scales.size()) / (per_iteration_ms * 1.0e6);
  const double nmse = squared_signal > 0.0 ? squared_error / squared_signal : 0.0;
  std::cout << std::fixed << std::setprecision(6)
            << "rows=" << rows
            << " cols=" << cols
            << " vectors=" << vectors
            << " blocks=" << blocks
            << " iteration_ms=" << per_iteration_ms
            << " input_gib_s=" << input_gib_s
            << " packed_output_gib_s=" << output_gib_s
            << " nmse=" << nmse
            << " scale_mismatches=" << scale_mismatches
            << " value_mismatches=" << value_mismatches
            << " device_reference_mismatches=" << device_reference_mismatches
            << " mismatches=" << mismatches << '\n';

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device_input);
  cudaFree(device_packed);
  cudaFree(device_scales);
  cudaFree(device_reference);
  return scale_mismatches == 0 && device_reference_mismatches == 0 ? 0 : 1;
}
