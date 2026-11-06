#pragma once

#include <cstdint>
#include <stdexcept>

namespace beam::distributed {

// The final exchange owns one fixed host slot per communicator rank.
inline constexpr std::uint32_t max_world_size = 128;

struct Rank {
    std::uint32_t world_size;
    std::uint32_t global_rank;
    std::uint32_t local_world_size;
    std::uint32_t local_rank;

    Rank(std::uint32_t world, std::uint32_t global, std::uint32_t local_size, std::uint32_t local)
        : world_size(world), global_rank(global), local_world_size(local_size), local_rank(local) {
        if (world == 0 || world > max_world_size || global >= world)
            throw std::invalid_argument("WORLD_SIZE must be in [1,128] and RANK in [0,WORLD_SIZE)");
        if (local_size == 0 || local_size > world || local >= local_size)
            throw std::invalid_argument("invalid LOCAL_RANK or LOCAL_WORLD_SIZE");
    }

    int cuda_ordinal(int visible_devices) const {
        // A scheduler can give a task a private one-device CUDA namespace.
        if (visible_devices == 1) return 0;
        if (visible_devices <= 0 || local_rank >= static_cast<std::uint32_t>(visible_devices))
            throw std::invalid_argument("LOCAL_RANK exceeds the CUDA devices visible to this task");
        return static_cast<int>(local_rank);
    }

    std::uint64_t node_share(std::uint64_t bytes) const { return bytes / local_world_size; }
    std::uint64_t default_host_headroom() const {
        return (static_cast<std::uint64_t>(local_world_size) + 2) << 30;
    }
};

} // namespace beam::distributed
