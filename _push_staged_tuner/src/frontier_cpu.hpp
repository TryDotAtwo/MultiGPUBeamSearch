#pragma once

#include "hash.hpp"
#include "state.hpp"
#include "stream3.hpp"
#include "stream4.hpp"

namespace beam {

struct CpuDepthResult {
    std::vector<State128> next_frontier;
    std::vector<CandidateMeta> survivors;
    bool solved = false;
};

CpuDepthResult expand_depth_cpu_reference(
    const std::vector<State128>& current_frontier,
    const std::vector<Generator>& generators,
    const State128& central_state,
    const ZobristTable& zobrist,
    std::uint32_t threshold,
    std::uint64_t beam_width);

} // namespace beam
