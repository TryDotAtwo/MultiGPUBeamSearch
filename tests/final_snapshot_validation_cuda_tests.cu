#include "../cuda/dispatcher.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Deliberately leave device memory/streams unallocated. Invalid host snapshots
// must be rejected before any CUDA submission can dereference these buffers.
int main() {
    try {
        beam::StaticMemoryPlan plan{};
        plan.config.world_size = 1;
        plan.config.local_rank = 0;
        plan.config.shard_capacity_candidates = 256;
        plan.storage_shard_count = 2;
        beam::StaticDeviceMemory memory{};
        beam::DispatcherStreams streams{};
        beam::DispatcherDeviceTables tables{};
        std::uint8_t generator_placeholder = 0;
        tables.generators = &generator_placeholder;
        auto reject = [&](std::vector<std::uint32_t> snapshot, const char* expected) {
            try {
                beam::finalize_depth_single_gpu(plan, memory, tables, streams, 0,
                    nullptr, 0, nullptr, nullptr, nullptr, nullptr, &snapshot);
            } catch (const std::invalid_argument& error) {
                if (std::string(error.what()) != expected) throw;
                std::cout << "rejected=" << expected << '\n';
                return;
            }
            throw std::runtime_error("malformed snapshot accepted");
        };
        reject({}, "final clean snapshot has wrong storage shard count");
        reject({0}, "final clean snapshot has wrong storage shard count");
        reject({0, 0, 0}, "final clean snapshot has wrong storage shard count");
        reject({257, 0}, "final clean snapshot exceeds physical shard capacity");
        reject({0, 257}, "final clean snapshot exceeds physical shard capacity");
        reject({UINT32_MAX, UINT32_MAX}, "final clean snapshot exceeds physical shard capacity");
        std::cout << "PASS: six malformed snapshots rejected before CUDA submission\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
