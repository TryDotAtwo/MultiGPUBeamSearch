#include "config.hpp"
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template <class F> void rejects(F fn, const char* message) {
    bool threw = false;
    try { fn(); } catch (const std::overflow_error&) { threw = true; }
    require(threw, message);
}
}

int main() {
    using namespace beam;
    try {
        rejects([] { round_up(UINT64_MAX, 1024); }, "round_up must reject overflow, not wrap to zero");
        require(round_up(UINT64_MAX - 1023, 1024) == UINT64_MAX - 1023, "aligned maximum stays unchanged");
        require(round_up(256000000, 8 * 64 * 1024) == 256376832, "measured 256M alignment unchanged");
        require(round_up(7, 0) == 7, "legacy zero-alignment bypass preserved");
        RuntimeConfig config{};
        config.b_micro = static_cast<std::uint32_t>(UINT32_MAX / MOVE_COUNT + 1);
        config.stream3_batch_candidates = static_cast<std::uint32_t>(
            static_cast<std::uint64_t>(config.b_micro) * MOVE_COUNT);
        config.inference_parallelism = 1;
        if constexpr (MOVE_COUNT > 1)
            rejects([&] { derive_config(config); }, "candidate-count multiplication must not wrap");
        config.b_micro = 1;
        config.stream3_batch_candidates = static_cast<std::uint32_t>(MOVE_COUNT);
        config.world_size = UINT32_MAX;
        config.shard_count = UINT32_MAX;
        config.stream4_batch_alignment = UINT32_MAX;
        rejects([&] { derive_config(config); }, "beam alignment multiplication must not wrap");
        std::cout << "config_limits_tests=pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "config_limits_tests=fail reason=" << error.what() << '\n';
        return 1;
    }
}
