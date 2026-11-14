// Uninstrumented timing of generator loading. Allocation-count measurements
// live in a separate executable so counting hooks do not distort these times.
#include "../src/native_input.hpp"
#include <chrono>
#include <cstdint>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::filesystem::path path(argv[1]);
    constexpr int repeats = 100;
    std::uint64_t checksum = 0;
    auto load = [&] {
#ifdef BEAM_BENCH_COMBINED
        const auto moves = beam::input::load_p900_moves(path);
        const auto& data = moves.data;
        const auto& names = moves.names;
#else
        const auto data = beam::input::load_p900_generators(path);
        const auto names = beam::input::load_p900_move_names(path);
#endif
        for (auto value : data) checksum += value;
        for (const auto& name : names) checksum += name.size();
    };
    load();
    checksum = 0;
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) load();
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    std::cout << "{\"repeats\":" << repeats << ",\"seconds\":" << seconds
              << ",\"checksum\":" << checksum << "}\n";
}
