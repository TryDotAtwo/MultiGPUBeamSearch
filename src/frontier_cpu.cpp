#include "frontier_cpu.hpp"

#include <algorithm>

namespace beam {

CpuDepthResult expand_depth_cpu_reference(
    const std::vector<State128>& current_frontier,
    const std::vector<Generator>& generators,
    const State128& central_state,
    const ZobristTable& zobrist,
    std::uint32_t threshold,
    std::uint64_t beam_width) {
    CpuDepthResult result;
    std::vector<CandidateMeta> candidates;
    std::vector<State128> materialized;

    for (std::uint64_t parent_idx = 0; parent_idx < current_frontier.size(); ++parent_idx) {
        for (std::uint8_t move = 0; move < generators.size(); ++move) {
            State128 child = apply_move(current_frontier[parent_idx], generators[move]);
            clear_state_padding(child);
            if (is_goal_state(child, central_state)) {
                result.solved = true;
            }
            const Hash128 hash = hash_state(child, zobrist);
            const std::uint32_t score_key = static_cast<std::uint32_t>(parent_idx + move) % SCORE_BIN_COUNT;
            if (score_key <= threshold) {
                candidates.push_back(CandidateMeta{hash, parent_idx, score_key, pack_route(0, 0, move)});
                materialized.push_back(child);
            }
        }
    }

    auto survivors = stream4_threshold_sort_dedup(candidates, threshold);
    std::sort(survivors.begin(), survivors.end(), [](const CandidateMeta& a, const CandidateMeta& b) {
        return candidate_better(a, b);
    });
    if (survivors.size() > beam_width) {
        survivors.resize(static_cast<std::size_t>(beam_width));
    }
    result.survivors = survivors;
    result.next_frontier.reserve(survivors.size());
    for (const auto& survivor : survivors) {
        result.next_frontier.push_back(apply_move(current_frontier[survivor.parent_idx], generators[unpack_move(survivor.route_packed)]));
        clear_state_padding(result.next_frontier.back());
    }
    return result;
}

} // namespace beam
