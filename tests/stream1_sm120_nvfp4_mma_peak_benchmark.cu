#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "cute/arch/mma_sm120.hpp"

namespace {

using Mma = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t,
    cutlass::float_e2m1_t,
    float,
    cutlass::float_ue8m0_t,
    32>;

constexpr int kWarpsPerBlock = 8;
constexpr int kIndependentAccumulators = 8;
constexpr int kInstructionsPerAccumulator = 32;
constexpr std::uint64_t kFlopsPerMma = 2ull * 16ull * 8ull * 64ull;
constexpr int kFfnSlices = 8;
constexpr int kMmaStepsPerFfnStage = 8;
constexpr int kPackedHiddenWords = (128 * 128 / 2 + 128 * (128 / 16)) / sizeof(std::uint32_t);

__global__ void nvfp4_mma_peak_kernel(float* sink) {
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const std::uint32_t seed = 0x11111111u + lane + warp * 17u;
  const std::uint32_t a0 = seed;
  const std::uint32_t a1 = seed ^ 0x01010101u;
  const std::uint32_t a2 = seed ^ 0x10101010u;
  const std::uint32_t a3 = seed ^ 0x00110011u;
  const std::uint32_t b0 = seed ^ 0x00010001u;
  const std::uint32_t b1 = seed ^ 0x10001000u;
  const Mma::RegTypeSF sf = static_cast<Mma::RegTypeSF>(0x7f7fu);

  float acc[kIndependentAccumulators][4]{};
#pragma unroll 1
  for (int step = 0; step < kInstructionsPerAccumulator; ++step) {
#pragma unroll
    for (int slot = 0; slot < kIndependentAccumulators; ++slot) {
      Mma::fma(
          acc[slot][0], acc[slot][1], acc[slot][2], acc[slot][3],
          a0, a1, a2, a3, b0, b1,
          acc[slot][0], acc[slot][1], acc[slot][2], acc[slot][3], sf, sf);
    }
  }

  if (lane == 0) {
    float sum = 0.0f;
#pragma unroll
    for (int slot = 0; slot < kIndependentAccumulators; ++slot) {
      sum += acc[slot][0] + acc[slot][1] + acc[slot][2] + acc[slot][3];
    }
    sink[warp] = sum;
  }
}

__global__ void nvfp4_b2b_shared_handoff_kernel(float* sink) {
  __shared__ volatile std::uint32_t hidden_pingpong[2][kPackedHiddenWords];
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp_in_block = threadIdx.x >> 5;
  const std::uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  std::uint32_t a0 = 0x11111111u + lane + warp * 17u;
  std::uint32_t a1 = a0 ^ 0x01010101u;
  std::uint32_t a2 = a0 ^ 0x10101010u;
  std::uint32_t a3 = a0 ^ 0x00110011u;
  std::uint32_t b0 = a0 ^ 0x00010001u;
  std::uint32_t b1 = a0 ^ 0x10001000u;
  const Mma::RegTypeSF sf = static_cast<Mma::RegTypeSF>(0x7f7fu);
  float acc[kIndependentAccumulators][4]{};

#pragma unroll 1
  for (int slice = 0; slice < kFfnSlices; ++slice) {
#pragma unroll 1
    for (int step = 0; step < kMmaStepsPerFfnStage; ++step) {
#pragma unroll
      for (int slot = 0; slot < kIndependentAccumulators; ++slot) {
        Mma::fma(
            acc[slot][0], acc[slot][1], acc[slot][2], acc[slot][3],
            a0, a1, a2, a3, b0, b1,
            acc[slot][0], acc[slot][1], acc[slot][2], acc[slot][3], sf, sf);
      }
    }

    const int buffer = slice & 1;
    const std::uint32_t packed = __float_as_uint(acc[(lane + slice) & 7][0]) ^
                                 (warp_in_block << 24) ^ static_cast<std::uint32_t>(slice);
    for (int word = threadIdx.x; word < kPackedHiddenWords; word += blockDim.x) {
      hidden_pingpong[buffer][word] = packed ^ static_cast<std::uint32_t>(word);
    }
    __syncthreads();
    const std::uint32_t handoff = hidden_pingpong[buffer][
        (threadIdx.x * 9 + slice * 131) % kPackedHiddenWords];
    a0 ^= handoff;
    b0 ^= (handoff << 1) | (handoff >> 31);

#pragma unroll 1
    for (int step = 0; step < kMmaStepsPerFfnStage; ++step) {
#pragma unroll
      for (int slot = 0; slot < kIndependentAccumulators; ++slot) {
        Mma::fma(
            acc[slot][0], acc[slot][1], acc[slot][2], acc[slot][3],
            a0, a1, a2, a3, b0, b1,
            acc[slot][0], acc[slot][1], acc[slot][2], acc[slot][3], sf, sf);
      }
    }
    __syncthreads();
  }
  if (lane == 0) {
    sink[warp] = acc[0][0] + acc[1][1] + acc[2][2] + acc[3][3];
  }
}

void check(cudaError_t result, const char* what) {
  if (result != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(result) << '\n';
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  int blocks_per_sm = argc > 1 ? std::atoi(argv[1]) : 4;
  int iterations = argc > 2 ? std::atoi(argv[2]) : 2000;

  cudaDeviceProp props{};
  check(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");
  if (props.major != 12 || props.minor != 0) {
    std::cerr << "requires SM120, found " << props.major << '.' << props.minor << '\n';
    return 2;
  }
  const int blocks = props.multiProcessorCount * blocks_per_sm;
  const int warps = blocks * kWarpsPerBlock;
  float* sink = nullptr;
  check(cudaMalloc(&sink, static_cast<std::size_t>(warps) * sizeof(float)), "cudaMalloc");

  nvfp4_mma_peak_kernel<<<blocks, 32 * kWarpsPerBlock>>>(sink);
  check(cudaDeviceSynchronize(), "warmup");

  cudaEvent_t start{}, stop{};
  check(cudaEventCreate(&start), "event start");
  check(cudaEventCreate(&stop), "event stop");
  check(cudaEventRecord(start), "record start");
  for (int i = 0; i < iterations; ++i) {
    nvfp4_mma_peak_kernel<<<blocks, 32 * kWarpsPerBlock>>>(sink);
  }
  check(cudaEventRecord(stop), "record stop");
  check(cudaEventSynchronize(stop), "sync stop");
  float elapsed_ms = 0.0f;
  check(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed");

  const auto mma_per_warp =
      static_cast<std::uint64_t>(kIndependentAccumulators) * kInstructionsPerAccumulator;
  const long double flops = static_cast<long double>(iterations) * warps * mma_per_warp * kFlopsPerMma;
  const long double tflops = flops / (static_cast<long double>(elapsed_ms) * 1.0e9L);
  std::cout << std::fixed << std::setprecision(3)
            << "sm_count=" << props.multiProcessorCount
            << " blocks_per_sm=" << blocks_per_sm
            << " warps=" << warps
            << " mma_per_warp=" << mma_per_warp
            << " elapsed_ms=" << elapsed_ms
            << " nvfp4_mma_tflops=" << static_cast<double>(tflops) << '\n';

  for (int tile_m : {128, 64}) {
    const int rows = 21888;
    const int b2b_blocks = (rows + tile_m - 1) / tile_m;
    const int b2b_warps = b2b_blocks * kWarpsPerBlock;
    if (b2b_warps > warps) {
      cudaFree(sink);
      sink = nullptr;
      check(cudaMalloc(&sink, static_cast<std::size_t>(b2b_warps) * sizeof(float)), "cudaMalloc b2b");
    }
    nvfp4_b2b_shared_handoff_kernel<<<b2b_blocks, 32 * kWarpsPerBlock>>>(sink);
    check(cudaDeviceSynchronize(), "b2b warmup");
    check(cudaEventRecord(start), "b2b start");
    for (int i = 0; i < iterations; ++i) {
      nvfp4_b2b_shared_handoff_kernel<<<b2b_blocks, 32 * kWarpsPerBlock>>>(sink);
    }
    check(cudaEventRecord(stop), "b2b stop");
    check(cudaEventSynchronize(stop), "b2b sync");
    elapsed_ms = 0.0f;
    check(cudaEventElapsedTime(&elapsed_ms, start, stop), "b2b elapsed");
    const auto b2b_mma_per_warp = static_cast<std::uint64_t>(
        kFfnSlices * 2 * kMmaStepsPerFfnStage * kIndependentAccumulators);
    const long double b2b_flops = static_cast<long double>(iterations) * b2b_warps *
                                  b2b_mma_per_warp * kFlopsPerMma;
    const long double b2b_tflops = b2b_flops / (static_cast<long double>(elapsed_ms) * 1.0e9L);
    std::cout << "b2b_tile_m=" << tile_m
              << " blocks=" << b2b_blocks
              << " warps=" << b2b_warps
              << " shared_handoff_bytes_per_slice=" << kPackedHiddenWords * sizeof(std::uint32_t)
              << " elapsed_ms=" << elapsed_ms
              << " nvfp4_b2b_tflops=" << static_cast<double>(b2b_tflops) << '\n';
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(sink);
  return 0;
}
