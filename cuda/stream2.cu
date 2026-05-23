#include "stream2.hpp"

#include "config.hpp"
#include "hash.hpp"
#include "nvtx_ranges.hpp"
#include "state.hpp"

#include <cuda_runtime.h>

namespace beam {

__device__ Hash128 hash_xor(Hash128 a, Hash128 b) {
    return Hash128{a.lo ^ b.lo, a.hi ^ b.hi};
}

__global__ void stream2_hash_goal_kernel(
    const State128* current_frontier_states,
    const std::uint64_t* parent_base,
    const std::uint32_t* count,
    const std::uint8_t* generators,
    const State128* central_state,
    const Hash128* zobrist,
    Hash128* hash_ring,
    std::uint32_t ring,
    std::uint32_t ring_slot,
    std::uint32_t ring_slot_count,
    std::uint32_t b_micro,
    std::uint32_t depth,
    std::uint32_t local_rank,
    Stream2SolvedBuffers solved) {
    const std::uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = b_micro * static_cast<std::uint32_t>(MOVE_COUNT);
    if (candidate >= total) {
        return;
    }

    const std::uint32_t parent_local = candidate / static_cast<std::uint32_t>(MOVE_COUNT);
    const std::uint32_t move = candidate % static_cast<std::uint32_t>(MOVE_COUNT);
    if (parent_local >= count[ring * ring_slot_count + ring_slot]) {
        return;
    }

    const std::uint64_t parent_idx = parent_base[ring * ring_slot_count + ring_slot] + parent_local;
    const State128 parent = current_frontier_states[parent_idx];
    Hash128 hash{0, 0};
    bool found = true;

    for (std::uint32_t p = 0; p < STATE_STORAGE_LEN; ++p) {
        const std::uint8_t source = generators[move * STATE_STORAGE_LEN + p];
        const std::uint8_t value = parent.v[source];
        if (value != central_state->v[p]) {
            found = false;
        }
        const Hash128 h = zobrist[p * STATE_VALUE_PAD + value];
        hash = hash_xor(hash, h);
    }

    const std::uint64_t hash_offset =
        (((static_cast<std::uint64_t>(ring) * ring_slot_count + ring_slot) * b_micro + parent_local) * MOVE_COUNT) + move;
    hash_ring[hash_offset] = hash;

    if (found && solved.solved_count != nullptr) {
        const std::uint32_t idx = atomicAdd(solved.solved_count, 1U);
        if (idx < solved.solved_result_capacity) {
            const std::uint32_t solved_depth =
                solved.current_depth == nullptr ? depth : *solved.current_depth;
            CandidateMeta meta{};
            meta.hash = hash;
            meta.parent_idx = parent_idx;
            meta.score_key = GOAL_SCORE_KEY;
            meta.route_packed =
                (static_cast<std::uint32_t>(local_rank) << 16) |
                (static_cast<std::uint32_t>(local_rank) << 8) |
                move;
            solved.solved_meta_list[idx] = meta;
            solved.solved_depth_list[idx] = solved_depth;
        } else {
            atomicExch(solved.solved_overflow, 1U);
        }
        __threadfence_system();
        if (atomicCAS(solved.solved_flag, 0U, 1U) == 0U) {
            atomicExch(solved.stop_flag, 1U);
        }
    }
}

void stream2_hash_goal_cuda(
    const State128* current_frontier_states,
    const std::uint64_t* parent_base,
    const std::uint32_t* count,
    const std::uint8_t* generators,
    const State128* central_state,
    const Hash128* zobrist,
    Hash128* hash_ring,
    std::uint32_t ring,
    std::uint32_t ring_slot,
    std::uint32_t b_micro,
    std::uint32_t depth,
    std::uint32_t local_rank,
    Stream2SolvedBuffers solved,
    cudaStream_t stream) {
    NvtxRange range("Stream2_hash_goal_launch");
    const std::uint32_t ring_slot_count = 1;
    const std::uint32_t total = b_micro * static_cast<std::uint32_t>(MOVE_COUNT);
    const dim3 block(128);
    const dim3 grid((total + block.x - 1) / block.x);
    stream2_hash_goal_kernel<<<grid, block, 0, stream>>>(
        current_frontier_states,
        parent_base,
        count,
        generators,
        central_state,
        zobrist,
        hash_ring,
        ring,
        ring_slot,
        ring_slot_count,
        b_micro,
        depth,
        local_rank,
        solved);
}

} // namespace beam
