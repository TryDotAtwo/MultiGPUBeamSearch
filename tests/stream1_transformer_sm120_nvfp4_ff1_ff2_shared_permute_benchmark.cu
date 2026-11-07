#include "stream1_transformer_sm120_nvfp4_cutlass_persistent_dsm.hpp"

#include <array>
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>

using namespace stream1_sm120_nvfp4_cutlass_dsm;

template <class Policy>
struct EpiloguePolicyTraits;

template <int StagesC, int StagesD, int FragmentSize, bool Reuse, bool Delay>
struct EpiloguePolicyTraits<
    cutlass::epilogue::Sm90TmaWarpSpecialized<
        StagesC, StagesD, FragmentSize, Reuse, Delay>> {
    static constexpr int kStagesC = StagesC;
    static constexpr int kStagesD = StagesD;
    static constexpr bool kReuse = Reuse;
};

using Ff1Policy = EpiloguePolicyTraits<typename Ff1Epilogue::DispatchPolicy>;
using Ff1SmemLayoutD = decltype(cute::tile_to_shape(
    typename Ff1Epilogue::SmemLayoutAtomD{},
    cute::make_shape(
        cute::size<0>(typename Ff1Epilogue::EpilogueTile{}),
        cute::size<1>(typename Ff1Epilogue::EpilogueTile{}),
        cute::Int<Ff1Policy::kReuse ? Ff1Policy::kStagesC : Ff1Policy::kStagesD>{}),
    cute::conditional_t<
        cutlass::epilogue::collective::detail::is_m_major<
            typename Ff1Epilogue::StrideD>(),
        cute::Step<cute::_2, cute::_1, cute::_3>,
        cute::Step<cute::_1, cute::_2, cute::_3>>{}));

struct Run {
  std::uint16_t source;
  std::uint16_t dest;
  std::uint8_t bytes;
};

struct RunTable {
  std::array<Run, 768> runs{};
  std::array<std::uint16_t, 8192> dest_for_source{};
  int count = 0;
  bool aligned = true;
};

RunTable make_run_table() {
  RunTable table{};
  for (int q = 0; q < 4; ++q) {
    for (int m = 0; m < 128; ++m) {
      for (int n = 0; n < 32; ++n) {
        const int source_nibble = Ff1SmemLayoutD{}(m, n, 0) + q * 4096;
        const int dest_nibble = Ff2StageLayoutA{}(m, q * 32 + n);
        table.dest_for_source[source_nibble >> 1] =
            static_cast<std::uint16_t>(dest_nibble >> 1);
      }
    }
  }
  int source = 0;
  while (source < 8192) {
    const int dest = table.dest_for_source[source];
    int bytes = 1;
    while (source + bytes < 8192 && bytes < 16 &&
           table.dest_for_source[source + bytes] == dest + bytes) {
      ++bytes;
    }
    table.runs[table.count++] = Run{
        static_cast<std::uint16_t>(source),
        static_cast<std::uint16_t>(dest),
        static_cast<std::uint8_t>(bytes)};
    table.aligned = table.aligned && (bytes == 8 || bytes == 16) &&
                    (source % bytes == 0) && (dest % bytes == 0);
    source += bytes;
  }
  return table;
}

__device__ __forceinline__ void copy_run(
    std::uint8_t* dest, const std::uint8_t* source, Run run) {
  if (run.bytes == 16) {
    *reinterpret_cast<uint4*>(dest + run.dest) =
        *reinterpret_cast<const uint4*>(source + run.source);
  } else {
    *reinterpret_cast<uint2*>(dest + run.dest) =
        *reinterpret_cast<const uint2*>(source + run.source);
  }
}

template <bool Copy>
__global__ void benchmark_kernel(
    std::uint64_t* clocks, std::uint32_t* errors, int iterations,
    const RunTable* table) {
  __shared__ alignas(16) std::uint8_t source[8192];
  __shared__ alignas(16) std::uint8_t dest[8192];
  __shared__ Run runs[768];
  for (int i = threadIdx.x; i < table->count; i += blockDim.x) {
    runs[i] = table->runs[i];
  }
  for (int i = threadIdx.x; i < 8192; i += blockDim.x) {
    source[i] = static_cast<std::uint8_t>((i * 131 + 17) & 255);
    dest[i] = 0;
  }
  __syncthreads();
  const std::uint64_t begin = clock64();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if constexpr (Copy) {
      for (int i = threadIdx.x; i < table->count; i += blockDim.x) {
        copy_run(dest, source, runs[i]);
      }
    }
    __syncthreads();
  }
  const std::uint64_t end = clock64();
  if (threadIdx.x == 0) {
    *clocks = end - begin;
  }
  __syncthreads();
  if constexpr (Copy) {
    for (int source_byte = threadIdx.x; source_byte < 8192;
         source_byte += blockDim.x) {
      const int dest_byte = table->dest_for_source[source_byte];
      if (dest[dest_byte] != source[source_byte]) atomicAdd(errors, 1U);
    }
  }
}

int main() {
  std::uint64_t* clocks = nullptr;
  std::uint32_t* errors = nullptr;
  RunTable* table = nullptr;
  cudaMallocManaged(&clocks, sizeof(*clocks));
  cudaMallocManaged(&errors, sizeof(*errors));
  cudaMallocManaged(&table, sizeof(*table));
  *table = make_run_table();
  if (table->count != 768 || !table->aligned) {
    std::cerr << "invalid run table count=" << table->count
              << " aligned=" << table->aligned << '\n';
    return 3;
  }
  std::cout << "run_prefix";
  for (int i = 0; i < 32; ++i) {
    std::cout << ' ' << table->runs[i].source << ':' << table->runs[i].dest
              << ':' << static_cast<int>(table->runs[i].bytes);
  }
  std::cout << '\n';
  constexpr int iterations = 1024;
  *errors = 0;
  benchmark_kernel<false><<<1, 384>>>(clocks, errors, iterations, table);
  auto status = cudaDeviceSynchronize();
  const std::uint64_t baseline = *clocks;
  benchmark_kernel<true><<<1, 384>>>(clocks, errors, iterations, table);
  status = cudaDeviceSynchronize();
  const std::uint64_t copied = *clocks;
  int clock_rate_khz = 0;
  cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0);
  const double overhead_cycles =
      static_cast<double>(copied - baseline) / iterations;
  const double overhead_us = overhead_cycles * 1000.0 / clock_rate_khz;
  const double gbps = 8192.0 / (overhead_us * 1000.0);
  std::cout << "ff1_ff2_shared_permute correctness="
            << (*errors == 0 && status == cudaSuccess ? "pass" : "fail")
            << " runs=" << table->count
            << " baseline_cycles=" << baseline
            << " copied_cycles=" << copied
            << " overhead_cycles_per_tile=" << overhead_cycles
            << " overhead_us_per_tile=" << overhead_us
            << " effective_GBps=" << gbps
            << " errors=" << *errors
            << " cuda=" << cudaGetErrorString(status) << '\n';
  return *errors == 0 && status == cudaSuccess ? 0 : 2;
}
