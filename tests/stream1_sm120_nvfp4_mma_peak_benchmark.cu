#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cstdint>
#include <cfloat>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "cute/arch/mma_sm120.hpp"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

namespace {

namespace cg = cooperative_groups;

using Mma = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t,
    cutlass::float_e2m1_t,
    float,
    cutlass::float_ue4m3_t,
    32>;

constexpr int kWarpsPerBlock = 8;
constexpr int kIndependentAccumulators = 8;
constexpr int kInstructionsPerAccumulator = 32;
constexpr std::uint64_t kFlopsPerMma = 2ull * 16ull * 8ull * 64ull;
constexpr int kFfnSlices = 8;
constexpr int kMmaStepsPerFfnStage = 8;
constexpr int kPackedHiddenWords = (128 * 128 / 2 + 128 * (128 / 16)) / sizeof(std::uint32_t);
template <int TileM>
constexpr int kPackedHiddenWordsFor =
    (TileM * 128 / 2 + TileM * (128 / 16)) / sizeof(std::uint32_t);

__device__ Mma::RegTypeSF unit_nvfp4_scale() {
  const auto one = cutlass::float_ue4m3_t(1.0f).raw();
  return static_cast<Mma::RegTypeSF>(one | (static_cast<std::uint16_t>(one) << 8));
}

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
  const Mma::RegTypeSF sf = unit_nvfp4_scale();

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
  const Mma::RegTypeSF sf = unit_nvfp4_scale();
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

template <int TileM>
__global__ void nvfp4_b2b_numeric_handoff_kernel(float* sink) {
  constexpr int kPackedWords = kPackedHiddenWordsFor<TileM>;
  constexpr int kMmaSteps = TileM == 64 ? 4 : kMmaStepsPerFfnStage;
  __shared__ volatile std::uint32_t hidden_pingpong[2][kPackedWords];
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp_in_block = threadIdx.x >> 5;
  const std::uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  std::uint32_t a0 = 0x11111111u + lane + warp * 17u;
  std::uint32_t a1 = a0 ^ 0x01010101u;
  std::uint32_t a2 = a0 ^ 0x10101010u;
  std::uint32_t a3 = a0 ^ 0x00110011u;
  std::uint32_t b0 = a0 ^ 0x00010001u;
  std::uint32_t b1 = a0 ^ 0x10001000u;
  const Mma::RegTypeSF sf = unit_nvfp4_scale();
  float acc[kIndependentAccumulators][4]{};
  constexpr unsigned kFullWarpMask = 0xffffffffu;
  constexpr int kVectorsPerTile = TileM * 128 / 16;
  constexpr int kVectorsPerWarp = kVectorsPerTile / kWarpsPerBlock;
  constexpr int kVectorsPerWarpStep = 8;
  constexpr int kPackedValueBytes = TileM * 128 / 2;

#pragma unroll 1
  for (int slice = 0; slice < kFfnSlices; ++slice) {
#pragma unroll 1
    for (int step = 0; step < kMmaSteps; ++step) {
#pragma unroll
      for (int slot = 0; slot < kIndependentAccumulators; ++slot) {
        Mma::fma(
            acc[slot][0], acc[slot][1], acc[slot][2], acc[slot][3],
            a0, a1, a2, a3, b0, b1,
            acc[slot][0], acc[slot][1], acc[slot][2], acc[slot][3], sf, sf);
      }
    }

    const int buffer = slice & 1;
    auto* hidden_bytes = reinterpret_cast<volatile std::uint8_t*>(hidden_pingpong[buffer]);
#pragma unroll 1
    for (int local_step = 0; local_step < kVectorsPerWarp / kVectorsPerWarpStep;
         ++local_step) {
      const int quad = lane >> 2;
      const int lane_in_quad = lane & 3;
      const int local_vector = local_step * kVectorsPerWarpStep + quad;
      const int vector = static_cast<int>(warp_in_block) * kVectorsPerWarp + local_vector;
      float values[4];
      float amax = 0.0f;
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        values[item] = fmaxf(
            acc[(local_vector + lane_in_quad + item) & 7][item] +
                static_cast<float>((local_vector ^ lane_in_quad ^ item) & 3) * 0.125f,
            0.0f);
        amax = fmaxf(amax, values[item]);
      }
#pragma unroll
      for (int delta = 1; delta <= 2; delta <<= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(kFullWarpMask, amax, delta));
      }
      std::uint8_t scale_raw = 0;
      cutlass::Array<float, 4> scaled_values{};
      cutlass::Array<cutlass::float_e2m1_t, 4> quantized{};
      if (amax > 0.0f) {
        const cutlass::float_ue4m3_t scale(amax / 6.0f);
        const float scale_fp32 = static_cast<float>(scale);
        const float reciprocal = scale_fp32 > 0.0f ? 1.0f / scale_fp32 : FLT_MAX;
        scale_raw = scale.raw();
#pragma unroll
        for (int item = 0; item < 4; ++item) {
          scaled_values[item] = values[item] * reciprocal;
        }
        quantized = cutlass::NumericArrayConverter<cutlass::float_e2m1_t, float, 4>{}(
            scaled_values);
      }
      if (lane_in_quad == 0) {
        hidden_bytes[kPackedValueBytes + vector] = scale_raw;
      }
      const auto packed_quantized = reinterpret_cast<std::uint16_t const&>(quantized);
      hidden_bytes[vector * 8 + lane_in_quad * 2] =
          static_cast<std::uint8_t>(packed_quantized & 0xffu);
      hidden_bytes[vector * 8 + lane_in_quad * 2 + 1] =
          static_cast<std::uint8_t>(packed_quantized >> 8);
    }
    __syncthreads();
    const std::uint32_t handoff = hidden_pingpong[buffer][
        (threadIdx.x * 9 + slice * 131) % kPackedWords];
    a0 ^= handoff;
    b0 ^= (handoff << 1) | (handoff >> 31);

#pragma unroll 1
    for (int step = 0; step < kMmaSteps; ++step) {
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

__global__ void nvfp4_b2b_numeric_overlap_kernel(float* sink) {
  __shared__ volatile std::uint32_t hidden_ring[3][kPackedHiddenWords];
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp_in_block = threadIdx.x >> 5;
  const std::uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  std::uint32_t a0 = 0x11111111u + lane + warp * 17u;
  std::uint32_t a1 = a0 ^ 0x01010101u;
  std::uint32_t a2 = a0 ^ 0x10101010u;
  std::uint32_t a3 = a0 ^ 0x00110011u;
  std::uint32_t b0 = a0 ^ 0x00010001u;
  std::uint32_t b1 = a0 ^ 0x10001000u;
  const Mma::RegTypeSF sf = unit_nvfp4_scale();
  float ff1_acc[2][kIndependentAccumulators][4]{};
  float ff2_acc[kIndependentAccumulators][4]{};
  constexpr unsigned kFullWarpMask = 0xffffffffu;
  constexpr int kVectorsPerTile = 128 * 128 / 16;
  constexpr int kVectorsPerWarp = kVectorsPerTile / kWarpsPerBlock;
  constexpr int kVectorsPerWarpStep = 8;
  constexpr int kPackedValueBytes = 128 * 128 / 2;

  // Pipeline stage t produces slice t, quantizes t-1 and consumes t-2.
#pragma unroll 1
  for (int stage = 0; stage < kFfnSlices + 2; ++stage) {
    const int produce_slice = stage;
    const int quantize_slice = stage - 1;
    const int consume_slice = stage - 2;
    const int produce_buffer = produce_slice & 1;
    if (produce_slice < kFfnSlices) {
#pragma unroll
      for (int slot = 0; slot < kIndependentAccumulators; ++slot) {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
          ff1_acc[produce_buffer][slot][item] = 0.0f;
        }
      }
    }

#pragma unroll 1
    for (int step = 0; step < kMmaStepsPerFfnStage; ++step) {
      if (produce_slice < kFfnSlices) {
#pragma unroll
        for (int slot = 0; slot < kIndependentAccumulators; ++slot) {
          Mma::fma(
              ff1_acc[produce_buffer][slot][0], ff1_acc[produce_buffer][slot][1],
              ff1_acc[produce_buffer][slot][2], ff1_acc[produce_buffer][slot][3],
              a0, a1, a2, a3, b0, b1,
              ff1_acc[produce_buffer][slot][0], ff1_acc[produce_buffer][slot][1],
              ff1_acc[produce_buffer][slot][2], ff1_acc[produce_buffer][slot][3], sf, sf);
        }
      }

      if (quantize_slice >= 0 && quantize_slice < kFfnSlices) {
        auto* hidden_bytes = reinterpret_cast<volatile std::uint8_t*>(
            hidden_ring[quantize_slice % 3]);
#pragma unroll
        for (int quant_substep = 0; quant_substep < 2; ++quant_substep) {
          const int local_step = step * 2 + quant_substep;
          const int quad = lane >> 2;
          const int lane_in_quad = lane & 3;
          const int local_vector = local_step * kVectorsPerWarpStep + quad;
          const int vector = static_cast<int>(warp_in_block) * kVectorsPerWarp + local_vector;
          float values[4];
          float amax = 0.0f;
#pragma unroll
          for (int item = 0; item < 4; ++item) {
            values[item] = fmaxf(
                ff1_acc[quantize_slice & 1][(local_vector + lane_in_quad + item) & 7][item] +
                    static_cast<float>((local_vector ^ lane_in_quad ^ item) & 3) * 0.125f,
                0.0f);
            amax = fmaxf(amax, values[item]);
          }
#pragma unroll
          for (int delta = 1; delta <= 2; delta <<= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(kFullWarpMask, amax, delta));
          }
          std::uint8_t scale_raw = 0;
          cutlass::Array<float, 4> scaled_values{};
          cutlass::Array<cutlass::float_e2m1_t, 4> quantized{};
          if (amax > 0.0f) {
            const cutlass::float_ue4m3_t scale(amax / 6.0f);
            const float scale_fp32 = static_cast<float>(scale);
            const float reciprocal = scale_fp32 > 0.0f ? 1.0f / scale_fp32 : FLT_MAX;
            scale_raw = scale.raw();
#pragma unroll
            for (int item = 0; item < 4; ++item) {
              scaled_values[item] = values[item] * reciprocal;
            }
            quantized = cutlass::NumericArrayConverter<cutlass::float_e2m1_t, float, 4>{}(
                scaled_values);
          }
          if (lane_in_quad == 0) {
            hidden_bytes[kPackedValueBytes + vector] = scale_raw;
          }
          const auto packed_quantized = reinterpret_cast<std::uint16_t const&>(quantized);
          hidden_bytes[vector * 8 + lane_in_quad * 2] =
              static_cast<std::uint8_t>(packed_quantized & 0xffu);
          hidden_bytes[vector * 8 + lane_in_quad * 2 + 1] =
              static_cast<std::uint8_t>(packed_quantized >> 8);
        }
      }

      if (consume_slice >= 0 && consume_slice < kFfnSlices) {
        const std::uint32_t handoff = hidden_ring[consume_slice % 3][
            (threadIdx.x * 9 + consume_slice * 131 + step * 17) % kPackedHiddenWords];
        const std::uint32_t ca0 = a0 ^ handoff;
        const std::uint32_t cb0 = b0 ^ ((handoff << 1) | (handoff >> 31));
#pragma unroll
        for (int slot = 0; slot < kIndependentAccumulators; ++slot) {
          Mma::fma(
              ff2_acc[slot][0], ff2_acc[slot][1], ff2_acc[slot][2], ff2_acc[slot][3],
              ca0, a1, a2, a3, cb0, b1,
              ff2_acc[slot][0], ff2_acc[slot][1], ff2_acc[slot][2], ff2_acc[slot][3], sf, sf);
        }
      }
    }
    __syncthreads();
  }
  if (lane == 0) {
    sink[warp] = ff2_acc[0][0] + ff2_acc[1][1] + ff2_acc[2][2] + ff2_acc[3][3];
  }
}

__global__ __cluster_dims__(2, 1, 1) void nvfp4_cluster_dsm_handoff_kernel(float* sink) {
  __shared__ std::uint32_t local_hidden[4][2 * kPackedHiddenWords];
  const auto cluster = cg::this_cluster();
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  std::uint32_t a0 = 0x11111111u + lane + warp * 17u;
  std::uint32_t a1 = a0 ^ 0x01010101u;
  std::uint32_t a2 = a0 ^ 0x10101010u;
  std::uint32_t a3 = a0 ^ 0x00110011u;
  std::uint32_t b0 = a0 ^ 0x00010001u;
  std::uint32_t b1 = a0 ^ 0x10001000u;
  const Mma::RegTypeSF sf = unit_nvfp4_scale();
  float acc[kIndependentAccumulators][4]{};

  // Four producer GEMMs: each M128 x N128 x K256.
#pragma unroll 1
  for (int producer_slice = 0; producer_slice < 4; ++producer_slice) {
#pragma unroll 1
    for (int row_half = 0; row_half < 2; ++row_half) {
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
      const std::uint32_t packed = __float_as_uint(acc[lane & 7][lane & 3]) ^
                                   blockIdx.x ^ static_cast<std::uint32_t>(producer_slice) ^
                                   static_cast<std::uint32_t>(row_half << 16);
      for (int word = threadIdx.x; word < kPackedHiddenWords; word += blockDim.x) {
        local_hidden[producer_slice][row_half * kPackedHiddenWords + word] =
            packed ^ static_cast<std::uint32_t>(word);
      }
    }
  }
  cluster.sync();

  // One consumer output slice: M128 x N128 x K1024. Each K128 slice is
  // fetched from the producer CTA that owns it through distributed SMEM.
#pragma unroll 1
  for (int source_rank = 0; source_rank < 2; ++source_rank) {
#pragma unroll 1
    for (int source_slice = 0; source_slice < 4; ++source_slice) {
#pragma unroll 1
      for (int row_half = 0; row_half < 2; ++row_half) {
        auto* remote_hidden = cluster.map_shared_rank(local_hidden[source_slice], source_rank);
        const std::uint32_t handoff = remote_hidden[row_half * kPackedHiddenWords +
            (threadIdx.x * 9 + (source_rank * 4 + source_slice) * 131) % kPackedHiddenWords];
        a0 ^= handoff;
        b0 ^= (handoff << 1) | (handoff >> 31);
#pragma unroll 1
        for (int step = 0; step < 4; ++step) {
#pragma unroll
          for (int slot = 0; slot < kIndependentAccumulators; ++slot) {
            Mma::fma(
                acc[slot][0], acc[slot][1], acc[slot][2], acc[slot][3],
                a0, a1, a2, a3, b0, b1,
                acc[slot][0], acc[slot][1], acc[slot][2], acc[slot][3], sf, sf);
          }
        }
      }
    }
  }
  cluster.sync();
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

    if (tile_m == 128) {
      nvfp4_b2b_numeric_handoff_kernel<128><<<b2b_blocks, 32 * kWarpsPerBlock>>>(sink);
    } else {
      nvfp4_b2b_numeric_handoff_kernel<64><<<b2b_blocks, 32 * kWarpsPerBlock>>>(sink);
    }
    check(cudaDeviceSynchronize(), "numeric b2b warmup");
    check(cudaEventRecord(start), "numeric b2b start");
    for (int i = 0; i < iterations; ++i) {
      if (tile_m == 128) {
        nvfp4_b2b_numeric_handoff_kernel<128><<<b2b_blocks, 32 * kWarpsPerBlock>>>(sink);
      } else {
        nvfp4_b2b_numeric_handoff_kernel<64><<<b2b_blocks, 32 * kWarpsPerBlock>>>(sink);
      }
    }
    check(cudaEventRecord(stop), "numeric b2b stop");
    check(cudaEventSynchronize(stop), "numeric b2b sync");
    elapsed_ms = 0.0f;
    check(cudaEventElapsedTime(&elapsed_ms, start, stop), "numeric b2b elapsed");
    const long double numeric_b2b_flops = b2b_flops / (128 / tile_m);
    const long double numeric_b2b_tflops = numeric_b2b_flops /
        (static_cast<long double>(elapsed_ms) * 1.0e9L);
    std::cout << "numeric_b2b_tile_m=" << tile_m
              << " blocks=" << b2b_blocks
              << " warps=" << b2b_warps
              << " k16_vectors_per_slice=1024"
              << " elapsed_ms=" << elapsed_ms
              << " nvfp4_numeric_b2b_tflops=" << static_cast<double>(numeric_b2b_tflops)
              << '\n';
    if (tile_m == 128) {
      nvfp4_b2b_numeric_overlap_kernel<<<b2b_blocks, 32 * kWarpsPerBlock>>>(sink);
      check(cudaDeviceSynchronize(), "overlap b2b warmup");
      check(cudaEventRecord(start), "overlap b2b start");
      for (int i = 0; i < iterations; ++i) {
        nvfp4_b2b_numeric_overlap_kernel<<<b2b_blocks, 32 * kWarpsPerBlock>>>(sink);
      }
      check(cudaEventRecord(stop), "overlap b2b stop");
      check(cudaEventSynchronize(stop), "overlap b2b sync");
      elapsed_ms = 0.0f;
      check(cudaEventElapsedTime(&elapsed_ms, start, stop), "overlap b2b elapsed");
      const long double overlap_tflops = b2b_flops /
          (static_cast<long double>(elapsed_ms) * 1.0e9L);
      std::cout << "numeric_overlap_tile_m=128"
                << " blocks=" << b2b_blocks
                << " elapsed_ms=" << elapsed_ms
                << " nvfp4_numeric_overlap_tflops=" << static_cast<double>(overlap_tflops)
                << '\n';
    }
  }

  constexpr int kClusterSize = 2;
  const int cluster_blocks = ((21888 + 255) / 256) * kClusterSize;
  const int cluster_warps = cluster_blocks * kWarpsPerBlock;
  if (cluster_warps > warps) {
    cudaFree(sink);
    sink = nullptr;
    check(cudaMalloc(&sink, static_cast<std::size_t>(cluster_warps) * sizeof(float)),
          "cudaMalloc cluster");
  }
  nvfp4_cluster_dsm_handoff_kernel<<<cluster_blocks, 32 * kWarpsPerBlock>>>(sink);
  check(cudaDeviceSynchronize(), "cluster DSM warmup");
  check(cudaEventRecord(start), "cluster DSM start");
  for (int i = 0; i < iterations; ++i) {
    nvfp4_cluster_dsm_handoff_kernel<<<cluster_blocks, 32 * kWarpsPerBlock>>>(sink);
  }
  check(cudaEventRecord(stop), "cluster DSM stop");
  check(cudaEventSynchronize(stop), "cluster DSM sync");
  elapsed_ms = 0.0f;
  check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cluster DSM elapsed");
  const auto cluster_mma_per_warp = static_cast<std::uint64_t>(
      2 * (4 * kMmaStepsPerFfnStage + kFfnSlices * 4) * kIndependentAccumulators);
  const long double cluster_flops = static_cast<long double>(iterations) * cluster_warps *
                                    cluster_mma_per_warp * kFlopsPerMma;
  const long double cluster_tflops = cluster_flops /
      (static_cast<long double>(elapsed_ms) * 1.0e9L);
  std::cout << "cluster_dsm_tile_m=256 cluster_size=" << kClusterSize
            << " blocks=" << cluster_blocks
            << " warps=" << cluster_warps
            << " mma_per_warp=" << cluster_mma_per_warp
            << " elapsed_ms=" << elapsed_ms
            << " nvfp4_cluster_dsm_tflops=" << static_cast<double>(cluster_tflops) << '\n';

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(sink);
  return 0;
}
