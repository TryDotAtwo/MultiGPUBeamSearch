#include "config.hpp"
#include "history_budget.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
using namespace beam;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <class F> void rejects(F operation, const char* message) {
    bool threw = false;
    try { operation(); } catch (const std::exception&) { threw = true; }
    require(threw, message);
}
}

int main() {
    try {
        const auto full = estimate_history_budget_entries(60, 4, 0, 32047104);
        require(full.effective_depth == 60, "BFS suffix must not remove history for executed depths");
        if constexpr (MOVE_COUNT == 24)
            require(full.required_entries == 1770899544ULL, "60-depth saturated history bound");
        const auto short_run = estimate_history_budget_entries(3, 4, 0, 1000);
        require(short_run.effective_depth == 3, "suffix larger than limit still needs history");
        if constexpr (MOVE_COUNT == 24)
            require(short_run.required_entries == 1600, "three depth bounds: 24 + 576 + 1000");
        require(estimate_history_budget_entries(60, UINT32_MAX, UINT32_MAX, 1).required_entries == 60,
                "large suffix cannot erase sixty single-entry layers");
        const auto empty = estimate_history_budget_entries(0, 4, 0, 1000);
        require(empty.required_entries == 0, "zero-depth history");
        rejects([] { plan_history(1, 10, 24, 6, 6); }, "whole layer cannot fit across fragmented arenas");
        const auto plan = plan_history(4, 10, 2, 20, 8);
        require(plan.required_entries == 24, "bounds are 2 + 4 + 8 + 10");
        require(plan.disk_reserved_entries == 6 && plan.ram_reserved_entries == 18, "whole-depth placement");
        require(plan.reservation(0, 1).offset_entries == 0, "first small actual depth");
        require(plan.reservation(1, 0).offset_entries == 2, "empty depth does not change reservations");
        require(plan.reservation(2, 7).location == HistoryStorageLocation::Ram, "third layer in RAM");
        require(plan.reservation(3, 8).offset_entries == 8, "later layer retains reserved RAM offset");
        rejects([&] { plan.reservation(0, 3); }, "actual count must not exceed its bound");
        rejects([&] { plan.reservation(4, 0); }, "depth outside plan");
        auto ram_end = plan.ram_reserved_entries;
        require(reserve_history_fallback_ram(2, 20, ram_end) == 18, "fallback follows ALL future RAM layers");
        rejects([&] { reserve_history_fallback_ram(1, 20, ram_end); }, "fallback capacity enforced");
        require(ram_end == 20, "failed fallback does not mutate cursor");
        rejects([] { plan_history(60, UINT64_MAX, 24, UINT64_MAX, UINT64_MAX); }, "entry total overflow");
        rejects([] { plan_history(64, (1ULL << 63) + 1, 2, UINT64_MAX, UINT64_MAX); },
                "prefull plus one saturated layer must not overflow");
        rejects([] { plan_history(1, 10, 0, 10, 10); }, "zero moves rejected");
        require(plan_history(0, 10, 24, 0, 0).depths.empty(), "zero iterations need no reservations");
        require(plan_history(3, 0, 24, 0, 0).required_entries == 0, "empty beam needs no entries");
        std::cout << "history_budget_tests=pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "history_budget_tests=fail reason=" << error.what() << '\n';
        return 1;
    }
}
