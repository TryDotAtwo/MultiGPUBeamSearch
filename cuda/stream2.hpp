#pragma once

#include "types.hpp"

#include <cstdint>

namespace beam {

inline constexpr std::uint32_t SOLVED_NEIGHBORHOOD_BUCKET_SIZE = 4;

struct SolvedNeighborhoodDeviceTable {
    const std::uint32_t* fingerprint_slots = nullptr;
    const Hash128* hash_slots = nullptr;
    std::uint32_t bucket_mask = 0;
    std::uint32_t enabled = 0;
};

struct Stream2SolvedBuffers {
    std::uint32_t* solved_flag = nullptr;
    std::uint32_t* stop_flag = nullptr;
    std::uint32_t* solved_count = nullptr;
    std::uint32_t* solved_overflow = nullptr;
    CandidateMeta* solved_meta_list = nullptr;
    std::uint32_t* solved_depth_list = nullptr;
    std::uint32_t solved_result_capacity = 0;
    const std::uint32_t* current_depth = nullptr;
    SolvedNeighborhoodDeviceTable solved_neighborhood;
};

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
    cudaStream_t stream);

} // namespace beam
