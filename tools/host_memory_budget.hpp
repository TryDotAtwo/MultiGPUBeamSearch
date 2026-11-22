#pragma once
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace beam::host_memory {
inline void prefault_pages(void* buffer, std::size_t bytes) {
    if (bytes == 0) return;
    if (buffer == nullptr) throw std::invalid_argument("nonempty history RAM prefault requires buffer");
    // Startup only. Volatile stores prevent malloc+memset from becoming lazy
    // calloc. 4KiB stride also covers larger host pages; touch the final byte
    // so an unaligned buffer's partial trailing page is included.
    auto* storage = static_cast<volatile unsigned char*>(buffer);
    std::size_t offset = 0;
    for (;;) {
        storage[offset] = 0;
        if (bytes - offset <= 4096) break;
        offset += 4096;
    }
    storage[bytes - 1] = 0;
}
inline std::uint64_t decimal(const std::string& text) {
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::runtime_error("malformed host memory byte count");
    return value;
}
inline std::string token_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::string token, extra;
    if (!(input >> token) || (input >> extra))
        throw std::runtime_error("cannot parse host memory file: " + path.string());
    return token;
}
inline std::uint64_t available_bytes(
    const std::filesystem::path& meminfo,
    const std::filesystem::path& cgroup_root,
    const std::filesystem::path& membership) {
    std::ifstream info(meminfo);
    std::string row;
    std::uint64_t available = 0;
    bool found = false;
    while (std::getline(info, row)) {
        std::istringstream fields(row);
        std::string key, value, unit, extra;
        if (!(fields >> key) || key != "MemAvailable:") continue;
        if (!(fields >> value >> unit) || unit != "kB" || (fields >> extra))
            throw std::runtime_error("malformed MemAvailable");
        const auto kib = decimal(value);
        if (kib > std::numeric_limits<std::uint64_t>::max() / 1024)
            throw std::overflow_error("MemAvailable byte count overflow");
        available = kib * 1024;
        found = true;
        break;
    }
    if (!found) throw std::runtime_error("MemAvailable unavailable for native host preflight");
    const auto root = cgroup_root.lexically_normal();
    std::vector<std::pair<std::filesystem::path, bool>> locations{{root, true}, {root / "memory", false}};
    if (std::filesystem::exists(membership)) {
        std::ifstream groups(membership);
        if (!groups) throw std::runtime_error("cannot read native cgroup membership");
        while (std::getline(groups, row)) {
            const auto first = row.find(':');
            const auto second = first == std::string::npos ? first : row.find(':', first + 1);
            if (second == std::string::npos) continue;
            const auto names = row.substr(first + 1, second - first - 1);
            const bool v2 = names.empty();
            if (!v2 && ("," + names + ",").find(",memory,") == std::string::npos) continue;
            const auto member = row.substr(second + 1);
            if (member.empty() || member.front() != '/') throw std::runtime_error("invalid native memory cgroup path");
            const std::filesystem::path relative(member.substr(1));
            for (const auto& part : relative)
                if (part == ".." || part == ".") throw std::runtime_error("invalid native memory cgroup path");
            const auto base = v2 ? root : root / "memory";
            auto current = (base / relative).lexically_normal();
            while (current != base) {
                locations.emplace_back(current, v2);
                const auto parent = current.parent_path();
                if (parent == current) throw std::runtime_error("native cgroup path escaped controller root");
                current = parent;
            }
        }
    }
    for (const auto& location : locations) {
        const auto limit_path = location.first / (location.second ? "memory.max" : "memory.limit_in_bytes");
        const auto usage_path = location.first / (location.second ? "memory.current" : "memory.usage_in_bytes");
        if (!std::filesystem::exists(limit_path) && !std::filesystem::exists(usage_path)) continue;
        const auto text = token_file(limit_path);
        if (location.second && text == "max") continue;
        const auto limit = decimal(text);
        const auto usage = decimal(token_file(usage_path));
        if (!location.second && limit >= (std::uint64_t{1} << 60)) continue;
        available = std::min(available, usage >= limit ? 0 : limit - usage);
    }
    return available;
}
inline void require_budget(std::uint64_t available, std::uint64_t history, std::uint64_t headroom) {
    if (history > available || headroom > available - history)
        throw std::runtime_error("insufficient container RAM for native history plus explicit headroom");
}
} // namespace beam::host_memory
