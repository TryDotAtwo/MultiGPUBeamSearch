#pragma once

#include "types.hpp"

#include <cstdint>
#include <vector>

namespace beam {

struct HistoryDepth {
    std::vector<CandidateMeta> frontier_meta;
};

struct SolutionPath {
    std::vector<std::uint8_t> moves;
    std::vector<std::uint64_t> parent_indices;
};

class CpuHistoryStore {
public:
    void clear();
    void append_depth(std::vector<CandidateMeta> frontier_meta);
    std::size_t depth_count() const;
    const HistoryDepth& depth(std::size_t depth_index) const;
    SolutionPath reconstruct_solution(const CandidateMeta& solved_meta, std::uint32_t solved_depth) const;

private:
    std::vector<HistoryDepth> depths_;
};

} // namespace beam
