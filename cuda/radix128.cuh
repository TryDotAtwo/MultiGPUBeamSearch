#pragma once

#include "../src/types.hpp"

#include <cub/version.cuh>
#if CUB_VERSION < 200700
#error "Experimental Hash128 radix sorting requires CUB 2.7 (CUDA 12.8) or newer"
#endif
#include <cub/device/device_radix_sort.cuh>
#include <cuda/std/tuple>

namespace beam {

// Hash128 is stored as {lo, hi}; CUB treats the leftmost tuple field as most
// significant. Both halves participate in every sort and identity comparison.
struct Hash128RadixDecomposer {
    __host__ __device__ cuda::std::tuple<std::uint64_t&, std::uint64_t&>
    operator()(Hash128& key) const {
        return {key.hi, key.lo};
    }
};

// The four input/output ranges must be disjoint. A null scratch pointer queries
// CUB's actual temporary-storage requirement without launching GPU work.
template <typename Value>
cudaError_t radix128_sort_pairs(void* scratch, std::size_t& scratch_bytes,
                                const Hash128* input_keys, Hash128* output_keys,
                                const Value* input_values, Value* output_values,
                                int count, cudaStream_t stream) {
    return cub::DeviceRadixSort::SortPairs(scratch, scratch_bytes, input_keys, output_keys,
                                          input_values, output_values, count,
                                          Hash128RadixDecomposer{}, 0, 128, stream);
}

} // namespace beam
