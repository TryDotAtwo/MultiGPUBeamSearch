#include <cuda_runtime.h>
#include <stdint.h>

#include "beam_types.hpp"

namespace beam_v6 {

__device__ __forceinline__ State128 apply_move_state128(const State128& parent, const uint8_t* generators, uint8_t move) {
    State128 child;
    const uint8_t* gen = generators + int(move) * STATE_STORAGE_LEN;
    #pragma unroll
    for (int p = 0; p < STATE_STORAGE_LEN; ++p) {
        child.v[p] = parent.v[gen[p]];
    }
    return child;
}

__device__ __forceinline__ bool is_goal_state128(const State128& child, const uint8_t* central_state) {
    #pragma unroll
    for (int p = 0; p < STATE_STORAGE_LEN; ++p) {
        if (child.v[p] != central_state[p]) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ Hash128 hash_state128_zobrist(const State128& child, const Hash128* zobrist) {
    Hash128 out{0ull, 0ull};
    #pragma unroll
    for (int p = 0; p < STATE_STORAGE_LEN; ++p) {
        const uint8_t v = child.v[p];
        const Hash128 h = zobrist[p * STATE_VALUE_PAD + int(v)];
        out.lo ^= h.lo;
        out.hi ^= h.hi;
    }
    return out;
}

extern "C" __global__ void kernel_v6_stream2_hash_goal(
    const State128* __restrict__ current_frontier_states,
    const uint64_t* __restrict__ parent_base,
    const uint32_t* __restrict__ count,
    const uint32_t* __restrict__ score_ring,
    Hash128* __restrict__ hash_ring,
    const uint8_t* __restrict__ generators,
    const uint8_t* __restrict__ central_state,
    const Hash128* __restrict__ zobrist,
    uint32_t* __restrict__ solved_flag,
    uint32_t* __restrict__ stop_flag,
    uint32_t* __restrict__ solved_count,
    uint32_t* __restrict__ solved_overflow,
    CandidateMeta* __restrict__ solved_meta_list,
    uint32_t* __restrict__ solved_depth_list,
    uint32_t solved_result_capacity,
    uint32_t depth,
    uint32_t local_rank,
    uint32_t ring,
    uint32_t ring_slot,
    uint32_t ring_slot_count,
    uint32_t b_micro
) {
    const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = b_micro * MOVE_COUNT;
    if (lane >= total) {
        return;
    }

    const uint32_t parent_local = lane / MOVE_COUNT;
    const uint8_t move = uint8_t(lane - parent_local * MOVE_COUNT);
    const uint32_t slot_idx = ring * ring_slot_count + ring_slot;
    if (parent_local >= count[slot_idx]) {
        return;
    }

    const uint64_t parent_idx = parent_base[slot_idx] + uint64_t(parent_local);
    const State128 parent = current_frontier_states[parent_idx];
    const State128 child = apply_move_state128(parent, generators, move);
    const Hash128 hash = hash_state128_zobrist(child, zobrist);

    const uint64_t ring_offset = (uint64_t(slot_idx) * uint64_t(b_micro) * uint64_t(MOVE_COUNT)) + uint64_t(lane);
    hash_ring[ring_offset] = hash;

    if (is_goal_state128(child, central_state)) {
        const uint32_t idx = atomicAdd(solved_count, 1u);
        if (idx < solved_result_capacity) {
            CandidateMeta meta;
            meta.hash = hash;
            meta.parent_idx = parent_idx;
            meta.score_key = GOAL_SCORE_KEY;
            meta.route_packed = pack_route(uint16_t(local_rank), uint8_t(local_rank), move);
            solved_meta_list[idx] = meta;
            solved_depth_list[idx] = depth;
        } else {
            atomicExch(solved_overflow, 1u);
        }
        __threadfence_system();
        if (atomicCAS(solved_flag, 0u, 1u) == 0u) {
            atomicExch(stop_flag, 1u);
        }
    }
}

} // namespace beam_v6

extern "C" void launch_v6_stream2_hash_goal(
    const beam_v6::State128* current_frontier_states,
    const uint64_t* parent_base,
    const uint32_t* count,
    const uint32_t* score_ring,
    beam_v6::Hash128* hash_ring,
    const uint8_t* generators,
    const uint8_t* central_state,
    const beam_v6::Hash128* zobrist,
    uint32_t* solved_flag,
    uint32_t* stop_flag,
    uint32_t* solved_count,
    uint32_t* solved_overflow,
    beam_v6::CandidateMeta* solved_meta_list,
    uint32_t* solved_depth_list,
    uint32_t solved_result_capacity,
    uint32_t depth,
    uint32_t local_rank,
    uint32_t ring,
    uint32_t ring_slot,
    uint32_t ring_slot_count,
    uint32_t b_micro,
    cudaStream_t stream
) {
    const uint32_t total = b_micro * beam_v6::MOVE_COUNT;
    const int threads = 256;
    const int blocks = int((total + threads - 1) / threads);
    beam_v6::kernel_v6_stream2_hash_goal<<<blocks, threads, 0, stream>>>(
        current_frontier_states,
        parent_base,
        count,
        score_ring,
        hash_ring,
        generators,
        central_state,
        zobrist,
        solved_flag,
        stop_flag,
        solved_count,
        solved_overflow,
        solved_meta_list,
        solved_depth_list,
        solved_result_capacity,
        depth,
        local_rank,
        ring,
        ring_slot,
        ring_slot_count,
        b_micro);
}
