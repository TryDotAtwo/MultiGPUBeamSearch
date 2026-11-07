#include "stream1_transformer_sm120_nvfp4_cutlass_persistent_dsm.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>

using namespace stream1_sm120_nvfp4_cutlass_dsm;

constexpr int kRepeats = 128;

template <bool Store>
__global__ void lane_owned_writer(std::uint32_t* sink) {
    __shared__ DsmASlot slot;
    const std::uint32_t subgroup = threadIdx.x / 4U;
    const std::uint32_t lane = threadIdx.x & 3U;
    std::uint32_t local = 0U;
    #pragma unroll 1
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
        for (std::uint32_t work = subgroup; work < 128U * 8U;
             work += blockDim.x / 4U) {
            const std::uint32_t row = work / 8U;
            const std::uint32_t group = work % 8U;
            const std::uint8_t a = (row + group + lane * 2U + repeat) & 15U;
            const std::uint8_t b = (a + 1U) & 15U;
            const std::uint8_t c = (a + 8U) & 15U;
            const std::uint8_t d = (c + 1U) & 15U;
            const std::uint8_t pair_lo = a | (b << 4U);
            const std::uint8_t pair_hi = c | (d << 4U);
            local += pair_lo + pair_hi;
            if constexpr (Store) {
                Ff1HiddenRingWriter::store_group_from_lane_pairs(
                    slot, row, group, pair_lo, pair_hi,
                    cutlass::float_ue4m3_t(1.0F).raw());
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0U) {
        if constexpr (Store) local += slot.values[0] + slot.scales[0];
        atomicAdd(sink, local);
    }
}

template <bool Store>
double measure(std::uint32_t* sink) {
    for (int i = 0; i < 20; ++i) lane_owned_writer<Store><<<1, 128>>>(sink);
    if (cudaDeviceSynchronize() != cudaSuccess) return -1.0;
    cudaEvent_t begin{}, end{};
    cudaEventCreate(&begin); cudaEventCreate(&end);
    cudaEventRecord(begin);
    constexpr int iterations = 1000;
    for (int i = 0; i < iterations; ++i) lane_owned_writer<Store><<<1, 128>>>(sink);
    cudaEventRecord(end);
    if (cudaEventSynchronize(end) != cudaSuccess) return -1.0;
    float ms = 0.0F;
    cudaEventElapsedTime(&ms, begin, end);
    cudaEventDestroy(begin); cudaEventDestroy(end);
    return ms * 1000.0 / iterations;
}

int main() {
    std::uint32_t* sink = nullptr;
    cudaMallocManaged(&sink, sizeof(*sink));
    *sink = 0U;
    const double baseline_us = measure<false>(sink);
    const double writer_us = measure<true>(sink);
    const auto status = cudaGetLastError();
    const double overhead_us = writer_us - baseline_us;
    const double bytes = static_cast<double>(Ff1HiddenRingWriter::kSlotBytes) * kRepeats;
    const double gbps = bytes / (overhead_us * 1000.0);
    const bool ok = baseline_us > 0.0 && writer_us > baseline_us &&
        status == cudaSuccess && *sink != 0U;
    std::cout << "ff1_native_writer_in_kernel correctness=" << (ok ? "pass" : "fail")
              << " baseline_us=" << baseline_us
              << " writer_us=" << writer_us
              << " overhead_us=" << overhead_us
              << " effective_GBps=" << gbps
              << " cuda=" << cudaGetErrorString(status) << '\n';
    cudaFree(sink);
    return ok ? 0 : 2;
}
