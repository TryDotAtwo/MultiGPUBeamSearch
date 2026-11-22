#pragma once

#include "../src/types.hpp"

#include <cub/device/device_merge_sort.cuh>

#ifndef BEAM_EXPERIMENTAL_RADIX128
#define BEAM_EXPERIMENTAL_RADIX128 0
#endif
#if BEAM_EXPERIMENTAL_RADIX128
#include "radix128.cuh"
#endif

namespace beam {

template <typename Value>
struct HashSortBuffers {
    Hash128* key_a;
    Hash128* key_b;
    Value* value_a;
    Value* value_b;
};

template <typename Value>
struct HashReductionViews {
    Hash128* sorted_keys;
    Hash128* reduced_keys;
    Value* sorted_values;
    Value* reduced_values;
};

// MergeSort is in-place; the experimental radix variant writes B and the
// following ReduceByKey writes back to A. Both preserve disjoint ranges.
template <typename Value>
HashReductionViews<Value> hash_reduction_views(HashSortBuffers<Value> buffers) {
#if BEAM_EXPERIMENTAL_RADIX128
    return {buffers.key_b, buffers.key_a, buffers.value_b, buffers.value_a};
#else
    return {buffers.key_a, buffers.key_b, buffers.value_a, buffers.value_b};
#endif
}

// This same entry point is used for real capacity-specific scratch queries and
// stream work, so selecting the sort cannot underallocate the CUB arena.
template <typename Value, typename Compare>
cudaError_t hash_sort_pairs(void* scratch, std::size_t& scratch_bytes,
                            HashSortBuffers<Value> buffers, int count,
                            Compare compare, cudaStream_t stream) {
#if BEAM_EXPERIMENTAL_RADIX128
    (void)compare;
    return radix128_sort_pairs(scratch, scratch_bytes, buffers.key_a, buffers.key_b,
                               buffers.value_a, buffers.value_b, count, stream);
#else
    return cub::DeviceMergeSort::SortPairs(scratch, scratch_bytes, buffers.key_a,
                                           buffers.value_a, count, compare, stream);
#endif
}

} // namespace beam
