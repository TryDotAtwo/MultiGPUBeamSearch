#include "cnot_domain.hpp"
#include "../src/cnot_domain.hpp"

#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <algorithm>

namespace beam::cnot {
namespace {

// The sorted keys are ordinals into the two input tensors. Closed ordinals
// precede all child ordinals, and earlier children win within any equal run.
struct RowOrder {
    const std::uint64_t* closed;
    const std::uint64_t* children;
    int closed_count;
    int words;
    int stride;
    std::uint64_t mask;

    __device__ const std::uint64_t* row(int ordinal) const {
        if (ordinal < closed_count) return closed + static_cast<std::int64_t>(ordinal) * stride;
        return children + static_cast<std::int64_t>(ordinal - closed_count) * stride;
    }
    __device__ int compare(int a, int b) const {
        const auto* left = row(a);
        const auto* right = row(b);
        for (int i = 0; i < stride; ++i) {
            std::uint64_t x = left[i], y = right[i];
            if (i % words == words - 1) { x &= mask; y &= mask; }
            if (x != y) return x < y ? -1 : 1;
        }
        return 0;
    }
    __device__ bool operator()(int a, int b) const {
        const int order = compare(a, b);
        return order < 0 || (order == 0 && a < b);
    }
};

__global__ void initialize_ordinals(int* ordinals, int total) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += gridDim.x * blockDim.x) ordinals[i] = static_cast<int>(i);
}

__global__ void mark_first_child(const int* ordinals, int total, RowOrder order, std::uint8_t* flags) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += gridDim.x * blockDim.x) {
        const int ordinal = ordinals[i];
        if (ordinal >= order.closed_count) {
            flags[ordinal - order.closed_count] = i == 0 || order.compare(ordinals[i - 1], ordinal) != 0;
        }
    }
}

} // namespace

cudaError_t first_unseen(void* temp, std::size_t& temp_bytes, const std::int64_t* closed,
                         int closed_count, const std::int64_t* children, int child_count, int n,
                         int* ordinals, std::uint8_t* flags, std::int64_t* selected, int* count,
                         cudaStream_t stream) {
    const int total = closed_count + child_count;
    const int words = words_per_row(n);
    const RowOrder order{reinterpret_cast<const std::uint64_t*>(closed),
                         reinterpret_cast<const std::uint64_t*>(children),
                         closed_count, words, n * words, tail_mask(n)};
    const auto child_ordinals = thrust::make_counting_iterator<std::int64_t>(0);
    if (temp == nullptr) {
        std::size_t sort_bytes = 0, select_bytes = 0;
        auto status = cub::DeviceMergeSort::SortKeys(nullptr, sort_bytes, ordinals, total, order, stream);
        if (status != cudaSuccess) return status;
        status = cub::DeviceSelect::Flagged(nullptr, select_bytes, child_ordinals, flags,
                                            selected, count, child_count, stream);
        temp_bytes = std::max(sort_bytes, select_bytes);
        return status;
    }
    // Every invocation owns one fixed temporary buffer, reused sequentially
    // on the caller's stream for sort and selection. No full matrix is moved.
    constexpr int threads = 128;
    const int blocks = std::min(65535, (total - 1) / threads + 1);
    initialize_ordinals<<<blocks, threads, 0, stream>>>(ordinals, total);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    std::size_t sort_bytes = temp_bytes;
    status = cub::DeviceMergeSort::SortKeys(temp, sort_bytes, ordinals, total, order, stream);
    if (status != cudaSuccess) return status;
    mark_first_child<<<blocks, threads, 0, stream>>>(ordinals, total, order, flags);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    std::size_t select_bytes = temp_bytes;
    return cub::DeviceSelect::Flagged(temp, select_bytes, child_ordinals, flags,
                                      selected, count, child_count, stream);
}

} // namespace beam::cnot
