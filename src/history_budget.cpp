#include "history_budget.hpp"
#include "config.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace beam {
namespace {
std::uint64_t expand_bound(std::uint64_t previous, std::uint64_t beam, std::uint32_t moves) {
    return previous > beam / moves ? beam : std::min(beam, previous * moves);
}

HistoryBudgetEstimate estimate(std::uint32_t depths, std::uint64_t beam, std::uint32_t moves) {
    if (moves == 0) throw std::invalid_argument("history move count must be nonzero");
    HistoryBudgetEstimate result{};
    result.effective_depth = depths;
    if (depths == 0 || beam == 0) return result;
    std::uint64_t bound = 1;
    for (std::uint32_t depth = 0; depth < depths; ++depth) {
        bound = expand_bound(bound, beam, moves);
        if (bound == beam) {
            result.target_beam_depth = depth;
            const auto remaining = static_cast<std::uint64_t>(depths - depth);
            if (remaining > (std::numeric_limits<std::uint64_t>::max() -
                             result.states_before_target_beam) / beam) {
                throw std::overflow_error("static hybrid history required entries overflow");
            }
            result.required_entries = result.states_before_target_beam + remaining * beam;
            return result;
        }
        if (bound > std::numeric_limits<std::uint64_t>::max() - result.states_before_target_beam)
            throw std::overflow_error("static hybrid history prefull entries overflow");
        result.states_before_target_beam += bound;
    }
    result.target_beam_depth = depths;
    result.required_entries = result.states_before_target_beam;
    return result;
}
} // namespace

HistoryBudgetEstimate estimate_history_budget_entries(
    std::uint32_t depth_limit, std::uint32_t /*solved_neighborhood_radius*/,
    std::uint32_t /*stream2_suffix_radius*/, std::uint64_t beam_entries) {
    return estimate(depth_limit, beam_entries, static_cast<std::uint32_t>(MOVE_COUNT));
}

const HistoryDepthReservation& HistoryPlan::reservation(
    std::uint32_t depth, std::uint64_t actual_count) const {
    const auto& entry = depths.at(depth);
    if (actual_count > entry.entry_bound)
        throw std::runtime_error("candidate history count exceeds planned depth bound");
    return entry;
}

HistoryPlan plan_history(
    std::uint32_t expansion_depths, std::uint64_t local_beam,
    std::uint32_t move_count, std::uint64_t ram_arena_entries,
    std::uint64_t disk_arena_entries) {
    HistoryPlan plan;
    plan.required_entries = estimate(expansion_depths, local_beam, move_count).required_entries;
    // Do not add two externally supplied capacities: that sum can overflow.
    if (plan.required_entries > ram_arena_entries &&
        plan.required_entries - ram_arena_entries > disk_arena_entries) {
        throw std::runtime_error("static hybrid history total budget too small");
    }
    plan.depths.reserve(expansion_depths);
    std::uint64_t bound = local_beam == 0 ? 0 : 1;
    for (std::uint32_t depth = 0; depth < expansion_depths; ++depth) {
        bound = expand_bound(bound, local_beam, move_count);
        if (bound <= disk_arena_entries - plan.disk_reserved_entries) {
            plan.depths.push_back({bound, plan.disk_reserved_entries, HistoryStorageLocation::Disk});
            plan.disk_reserved_entries += bound;
        } else if (bound <= ram_arena_entries - plan.ram_reserved_entries) {
            plan.depths.push_back({bound, plan.ram_reserved_entries, HistoryStorageLocation::Ram});
            plan.ram_reserved_entries += bound;
        } else {
            throw std::runtime_error("static hybrid history cannot place whole depth=" +
                                     std::to_string(depth) + " entries=" + std::to_string(bound));
        }
    }
    return plan;
}

std::uint64_t reserve_history_fallback_ram(
    std::uint64_t count, std::uint64_t capacity, std::uint64_t& reserved_end) {
    if (reserved_end > capacity || count > capacity - reserved_end)
        throw std::runtime_error("static hybrid history RAM fallback exhausted");
    const auto offset = reserved_end;
    reserved_end += count;
    return offset;
}
} // namespace beam
