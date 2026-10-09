#pragma once

#include "types.hpp"

#include <vector>

namespace beam {

struct Stream3CandidateInput {
    Hash128 hash;
    std::uint32_t score_key;
    std::uint32_t payload_id;
    std::uint64_t parent_idx;
    std::uint8_t move;
};

struct Stream3SplitResult {
    std::vector<CandidateMeta> local_pending;
    std::vector<CandidateMeta> remote_send;
    std::vector<std::uint32_t> send_count;
    std::vector<std::uint32_t> send_offset;
};

std::uint64_t pack_stream3_val(std::uint32_t score_key, std::uint32_t payload_id);
Stream3SplitResult stream3_threshold_dedup_split(
    const std::vector<Stream3CandidateInput>& input,
    std::uint32_t current_threshold,
    std::uint16_t local_rank,
    std::uint32_t world_size);

} // namespace beam

