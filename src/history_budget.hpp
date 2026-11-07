#pragma once

#include <cstdint>
#include <vector>

namespace beam {

enum class HistoryStorageLocation : std::uint8_t { None, Ram, Disk };

struct HistoryBudgetEstimate {
    std::uint32_t effective_depth = 0;
    std::uint32_t target_beam_depth = 0;
    std::uint64_t states_before_target_beam = 0;
    std::uint64_t required_entries = 0;
};

// The suffix arguments are retained for callers/log compatibility. They do not
// reduce the number of expansion iterations committed by production_runner.
HistoryBudgetEstimate estimate_history_budget_entries(
    std::uint32_t depth_limit, std::uint32_t solved_neighborhood_radius,
    std::uint32_t stream2_suffix_radius, std::uint64_t beam_entries);

struct HistoryDepthReservation {
    std::uint64_t entry_bound = 0;
    std::uint64_t offset_entries = 0;
    HistoryStorageLocation location = HistoryStorageLocation::None;
};

struct HistoryPlan {
    std::vector<HistoryDepthReservation> depths;
    std::uint64_t required_entries = 0;
    std::uint64_t ram_reserved_entries = 0;
    std::uint64_t disk_reserved_entries = 0;

    const HistoryDepthReservation& reservation(
        std::uint32_t depth, std::uint64_t actual_count) const;
};

// Immutable disk-first whole-layer placement. Smaller actual layers never move
// later reservations; unused tail RAM is the only disk-failure fallback space.
HistoryPlan plan_history(
    std::uint32_t expansion_depths, std::uint64_t local_beam,
    std::uint32_t move_count, std::uint64_t ram_arena_entries,
    std::uint64_t disk_arena_entries);

std::uint64_t reserve_history_fallback_ram(
    std::uint64_t count, std::uint64_t capacity, std::uint64_t& reserved_end);

} // namespace beam
