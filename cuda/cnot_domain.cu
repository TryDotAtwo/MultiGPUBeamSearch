#include "cnot_domain.hpp"
#include "../src/cnot_domain.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

namespace beam::cnot {
namespace {

constexpr int threads = 128;
constexpr int warps_per_block = threads / 32;

template <typename DescriptorMove>
__global__ void children_kernel(const std::uint64_t* parents, std::int64_t parent_count, int n,
                                const std::int64_t* indices, const DescriptorMove* moves,
                                std::int64_t count, std::uint64_t* children) {
    const int words = words_per_row(n);
    const int stride = n * words;
    for (std::int64_t child = blockIdx.x; child < count; child += gridDim.x) {
        const std::int64_t parent = indices[child];
        const std::int64_t move = moves[child];
        if (parent < 0 || parent >= parent_count || move < 0 || move >= move_count(n)) {
            assert(false && "CNOT descriptor has an invalid parent or physical move");
            __trap();  // Also reject invalid descriptors in NDEBUG builds.
            return;
        }
        const Gate gate = decode_move(n, static_cast<MoveId>(move));
        const std::uint64_t* source = parents + parent * stride;
        std::uint64_t* target = children + child * stride;
        for (int i = threadIdx.x; i < stride; i += blockDim.x) {
            const int word = i % words;
            std::uint64_t value = source[i];
            if (i / words == gate.target) {
                value ^= source[gate.control * words + word];
            }
            target[i] = word == words - 1 ? value & tail_mask(n) : value;
        }
    }
}

__global__ void owners_kernel(const std::uint64_t* states, std::int64_t count, int n,
                              int world, std::int64_t* owners) {
    const int lane = threadIdx.x % 32;
    const int words = words_per_row(n);
    const int stride = n * words;
    const std::int64_t first = static_cast<std::int64_t>(blockIdx.x) * warps_per_block
                               + threadIdx.x / 32;
    for (std::int64_t state = first; state < count; state += gridDim.x * warps_per_block) {
        Hash128 hash{0, 0};
        for (int i = lane; i < stride; i += 32) {
            std::uint64_t value = states[state * stride + i];
            if (i % words == words - 1) value &= tail_mask(n);
            const auto part = hash_word(value, i);
            hash.lo ^= part.lo;
            hash.hi ^= part.hi;
        }
        for (int shift = 16; shift; shift /= 2) {
            hash.lo ^= __shfl_down_sync(0xffffffffU, hash.lo, shift);
            hash.hi ^= __shfl_down_sync(0xffffffffU, hash.hi, shift);
        }
        if (lane == 0) owners[state] = owner_key(hash, n) % world;
    }
}

int grid_size(std::int64_t blocks) {
    return static_cast<int>(std::min<std::int64_t>(blocks, 65535));
}

} // namespace

cudaError_t launch_children(const std::int64_t* parents, std::int64_t parent_count, int n,
                            const std::int64_t* indices, const void* moves, bool moves_are_u16,
                            std::int64_t count, std::int64_t* children, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    const auto* source = reinterpret_cast<const std::uint64_t*>(parents);
    auto* output = reinterpret_cast<std::uint64_t*>(children);
    if (moves_are_u16) {
        children_kernel<<<grid_size(count), threads, 0, stream>>>(
            source, parent_count, n, indices, static_cast<const MoveId*>(moves), count, output);
    } else {
        children_kernel<<<grid_size(count), threads, 0, stream>>>(
            source, parent_count, n, indices, static_cast<const std::int64_t*>(moves), count, output);
    }
    return cudaGetLastError();
}

cudaError_t launch_owners(const std::int64_t* states, std::int64_t count, int n, int world,
                          std::int64_t* owners, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    const auto* source = reinterpret_cast<const std::uint64_t*>(states);
    const auto blocks = (count - 1) / warps_per_block + 1;
    owners_kernel<<<grid_size(blocks), threads, 0, stream>>>(source, count, n, world, owners);
    return cudaGetLastError();
}

} // namespace beam::cnot
