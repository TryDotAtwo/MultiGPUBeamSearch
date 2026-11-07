#include "stream1_chunk_plan.hpp"
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using Chunk = std::pair<std::uint32_t, std::uint32_t>;
std::vector<Chunk> observe(std::uint32_t outer, std::uint32_t inner) {
    std::vector<Chunk> result;
    beam::for_each_stream1_transformer_chunk(outer, inner,
        [&](std::uint32_t offset, std::uint32_t count) { result.emplace_back(offset, count); });
    return result;
}
int main() {
    // A save deferred until after the next launch loses the previous scratch.
    std::vector<Chunk> saved;
    Chunk scratch{99, 99};
    beam::for_each_stream1_transformer_chunk_observed(8, 3,
        [&](std::uint32_t offset, std::uint32_t count) { scratch = {offset, count}; },
        [&](std::uint32_t offset, std::uint32_t count) {
            if (scratch != Chunk{offset, count}) throw std::runtime_error("stale scratch");
            saved.push_back(scratch);
        });
    if (saved != std::vector<Chunk>{{0, 3}, {3, 3}, {6, 2}}) return 1;
    if (observe(2048, 1024) != std::vector<Chunk>{{0, 1024}, {1024, 1024}}) return 1;
    if (observe(1024, 1024) != std::vector<Chunk>{{0, 1024}}) return 1;
    if (observe(1000, 384) != std::vector<Chunk>{{0, 384}, {384, 384}, {768, 232}}) return 1;
    if (observe(UINT32_MAX, UINT32_MAX - 1) != std::vector<Chunk>{{0, UINT32_MAX - 1}, {UINT32_MAX - 1, 1}}) return 1;
    for (auto invalid : std::vector<Chunk>{{0, 1}, {1, 0}, {2, 3}}) {
        bool called = false, rejected = false;
        try {
            beam::for_each_stream1_transformer_chunk(invalid.first, invalid.second,
                [&](std::uint32_t, std::uint32_t) { called = true; });
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || called) return 1;
    }
    std::cout << "shared_chunk_plan=pass\n";
}
