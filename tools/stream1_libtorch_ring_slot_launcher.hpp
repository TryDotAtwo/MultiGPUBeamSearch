#pragma once

#include "../cuda/dispatcher.hpp"
#include "../cuda/stream2.hpp"
#include "../src/types.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace beam::stream1_libtorch {

struct RingSlotLauncherConfig {
    std::filesystem::path weight_dir;
    int device_index = 0;
    std::uint32_t inference_parallelism = 0;
    std::uint32_t local_rank = 0;
    std::uint32_t b_micro = 0;
    std::uint32_t move_count = 0;
    const State128* current_frontier_states = nullptr;
    const std::uint64_t* parent_base = nullptr;
    const std::uint32_t* count = nullptr;
    std::uint32_t* score_ring = nullptr;
    Hash128* hash_ring = nullptr;
    const std::uint8_t* generators = nullptr;
    const State128* central_state = nullptr;
    const Hash128* zobrist = nullptr;
    Stream2SolvedBuffers solved{};
};

class RingSlotLauncher {
public:
    explicit RingSlotLauncher(RingSlotLauncherConfig config);
    ~RingSlotLauncher();

    RingSlotLauncher(const RingSlotLauncher&) = delete;
    RingSlotLauncher& operator=(const RingSlotLauncher&) = delete;
    RingSlotLauncher(RingSlotLauncher&&) noexcept;
    RingSlotLauncher& operator=(RingSlotLauncher&&) noexcept;

    const DispatcherRingSlotLauncher& dispatcher_launcher() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace beam::stream1_libtorch