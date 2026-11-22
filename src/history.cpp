#include "history.hpp"

#include <stdexcept>
#include <utility>

namespace beam {

void CpuHistoryStore::clear() {
    depths_.clear();
}

void CpuHistoryStore::append_depth(std::vector<CandidateMeta> frontier_meta) {
    depths_.push_back(HistoryDepth{std::move(frontier_meta)});
}

std::size_t CpuHistoryStore::depth_count() const {
    return depths_.size();
}

const HistoryDepth& CpuHistoryStore::depth(std::size_t depth_index) const {
    if (depth_index >= depths_.size()) {
        throw std::out_of_range("history depth index out of range");
    }
    return depths_[depth_index];
}

SolutionPath CpuHistoryStore::reconstruct_solution(const CandidateMeta& solved_meta, std::uint32_t solved_depth) const {
    if (solved_depth > depths_.size() + 1U) {
        throw std::out_of_range("solved depth exceeds stored history depth count");
    }

    SolutionPath path;
    path.moves.resize(solved_depth);
    path.parent_indices.resize(solved_depth);

    CandidateMeta cursor = solved_meta;
    std::uint64_t parent_idx = cursor.parent_idx;
    for (std::uint32_t depth = solved_depth; depth > 0; --depth) {
        const std::uint32_t out = depth - 1U;
        path.moves[out] = unpack_move(cursor.route_packed);
        path.parent_indices[out] = parent_idx;
        if (out == 0) {
            break;
        }
        const HistoryDepth& previous_depth = depths_[out - 1U];
        if (parent_idx >= previous_depth.frontier_meta.size()) {
            throw std::out_of_range("history parent index out of range");
        }
        cursor = previous_depth.frontier_meta[static_cast<std::size_t>(parent_idx)];
        parent_idx = cursor.parent_idx;
    }
    return path;
}

} // namespace beam
