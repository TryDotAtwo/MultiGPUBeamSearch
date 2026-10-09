#pragma once

#include "types.hpp"

#include <vector>

namespace beam {

bool candidate_better(const CandidateMeta& a, const CandidateMeta& b);
std::vector<CandidateMeta> stream4_threshold_sort_dedup(
    const std::vector<CandidateMeta>& clean_and_dirty,
    std::uint32_t threshold);
std::uint32_t histogram_threshold(
    const std::vector<std::uint64_t>& global_score_hist,
    std::uint64_t global_beam_width_effective);

} // namespace beam
