#include "history.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace beam;

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
} // namespace

int main() {
    std::filesystem::create_directories("test_results");
    std::ofstream report("test_results/history_tests_2026-05-22.md");
    report << "# History Tests 2026-05-22\n\n";

    CpuHistoryStore history;
    history.append_depth(std::vector<CandidateMeta>{
        CandidateMeta{Hash128{1, 1}, 0, 10, pack_route(0, 0, 4)},
        CandidateMeta{Hash128{2, 2}, 0, 11, pack_route(0, 0, 7)},
    });
    history.append_depth(std::vector<CandidateMeta>{
        CandidateMeta{Hash128{3, 3}, 1, 9, pack_route(0, 0, 2)},
    });

    const CandidateMeta solved{Hash128{4, 4}, 0, GOAL_SCORE_KEY, pack_route(0, 0, 5)};
    const SolutionPath path = history.reconstruct_solution(solved, 3);
    require(path.moves.size() == 3, "solution move count failed");
    require(path.moves[0] == 7 && path.moves[1] == 2 && path.moves[2] == 5, "solution moves failed");
    require(path.parent_indices[0] == 0 && path.parent_indices[1] == 1 && path.parent_indices[2] == 0, "solution parent indices failed");
    require(history.depth_count() == 2, "history depth count failed");

    report << "- append_depth=pass\n";
    report << "- reconstruct_solution=pass\n";
    report << "- move_order_root_to_goal=pass\n";
    report << "\nstatus=pass\n";
    std::cout << "history_tests=pass\n";
    return 0;
}
