#include "stream4.hpp"

#include "config.hpp"

#include <algorithm>
#include <stdexcept>

namespace beam {

bool candidate_better(const CandidateMeta& a, const CandidateMeta& b) {
    if (a.score_key != b.score_key) {
        return a.score_key < b.score_key;
    }
    if (a.parent_idx != b.parent_idx) {
        return a.parent_idx < b.parent_idx;
    }
    return a.route_packed < b.route_packed;
}

std::vector<CandidateMeta> stream4_threshold_sort_dedup(
    const std::vector<CandidateMeta>& clean_and_dirty,
    std::uint32_t threshold) {
    std::vector<CandidateMeta> filtered;
    filtered.reserve(clean_and_dirty.size());
    for (const auto& candidate : clean_and_dirty) {
        if (candidate.score_key <= threshold) {
            filtered.push_back(candidate);
        }
    }
    std::sort(filtered.begin(), filtered.end(), [](const CandidateMeta& a, const CandidateMeta& b) {
        if (!(a.hash == b.hash)) {
            return hash_less(a.hash, b.hash);
        }
        return candidate_better(a, b);
    });

    std::vector<CandidateMeta> output;
    output.reserve(filtered.size());
    for (const auto& candidate : filtered) {
        if (output.empty() || !(output.back().hash == candidate.hash)) {
            output.push_back(candidate);
        }
    }
    return output;
}

std::uint32_t histogram_threshold(
    const std::vector<std::uint64_t>& global_score_hist,
    std::uint64_t global_beam_width_effective) {
    if (global_score_hist.size() != SCORE_BIN_COUNT) {
        throw std::invalid_argument("global_score_hist size must equal SCORE_BIN_COUNT");
    }
    std::uint64_t cumulative = 0;
    for (std::uint32_t score = 0; score < SCORE_BIN_COUNT; ++score) {
        cumulative += global_score_hist[score];
        if (cumulative >= global_beam_width_effective) {
            return score;
        }
    }
    return UINT32_THRESHOLD_MAX;
}

} // namespace beam
