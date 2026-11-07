#pragma once

#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>

namespace beam::detail {

// Every thread in a 256-thread block participates, including partial tails.
__device__ __forceinline__ std::uint32_t stream4_block_prefix(std::uint32_t keep) {
    using Scan = cub::BlockScan<std::uint32_t, 256>;
    __shared__ typename Scan::TempStorage storage;
    std::uint32_t prefix = 0;
    Scan(storage).ExclusiveSum(keep, prefix);
    return prefix;
}

__global__ static void stream4_scan_total_kernel(const std::uint32_t* counts,
                                                  const std::uint32_t* offsets,
                                                  std::uint32_t* total,
                                                  std::uint32_t count) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *total = offsets[count - 1U] + counts[count - 1U];
}

inline cudaError_t stream4_scan_temp_bytes(std::uint32_t count, std::size_t& bytes) {
    bytes = 0;
    if (!count) return cudaSuccess;
    if (count > static_cast<std::uint32_t>(INT_MAX)) return cudaErrorInvalidValue;
    const std::uint32_t* counts = nullptr;
    std::uint32_t* offsets = nullptr;
    return cub::DeviceScan::ExclusiveSum(nullptr, bytes, counts, offsets,
                                          static_cast<int>(count));
}

// Inputs, offsets and total are disjoint. Scratch belongs to the caller and
// remains in use until all work enqueued on stream has completed.
inline cudaError_t stream4_scan_counts(const std::uint32_t* counts, std::uint32_t* offsets,
                                       std::uint32_t* total, std::uint32_t count,
                                       void* temp, std::size_t temp_bytes, cudaStream_t stream) {
    if (!total) return cudaErrorInvalidValue;
    if (!count) return cudaMemsetAsync(total, 0, sizeof(*total), stream);
    if (!counts || !offsets || !temp || count > static_cast<std::uint32_t>(INT_MAX))
        return cudaErrorInvalidValue;
    auto bytes = temp_bytes;
    const auto status = cub::DeviceScan::ExclusiveSum(temp, bytes, counts, offsets,
                                                       static_cast<int>(count), stream);
    if (status != cudaSuccess) return status;
    stream4_scan_total_kernel<<<1, 1, 0, stream>>>(counts, offsets, total, count);
    return cudaPeekAtLastError();
}

} // namespace beam::detail
