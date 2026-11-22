#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace beam::rank_status {
inline void publish(const std::filesystem::path& directory, std::uint32_t rank,
                    std::uint64_t puzzle_id, bool solved, std::uint32_t completed) {
    std::filesystem::create_directories(directory);
    // Reserve a rank-owned directory, refusing reused/stale publications.
    const auto reservation = directory / ("rank-" + std::to_string(rank) + ".owned");
    if (!std::filesystem::create_directory(reservation))
        throw std::runtime_error("rank status already reserved");
    const auto target = directory / ("rank-" + std::to_string(rank) + ".json");
    if (std::filesystem::exists(target))
        throw std::runtime_error("rank status already exists");
    const auto temporary = directory / ("rank-" + std::to_string(rank) + ".tmp");
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot open rank status");
    stream << "{\"rank\":" << rank << ",\"puzzle_id\":" << puzzle_id
           << ",\"exit_code\":0,\"status\":\"" << (solved ? "solved" : "unsolved")
           << "\",\"completed_depths\":" << completed << "}\n";
    stream.flush();
    if (!stream) throw std::runtime_error("cannot write rank status");
    stream.close();
    if (!stream) throw std::runtime_error("cannot close rank status");
    std::filesystem::rename(temporary, target);
}
} // namespace beam::rank_status
