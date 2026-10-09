#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace beam {
template<class Function>
inline void for_each_stream1_transformer_chunk(
    std::uint32_t outer, std::uint32_t inner, Function&& function) {
    if (outer == 0 || inner == 0 || inner > outer)
        throw std::invalid_argument("Transformer micro must be in [1, outer batch]");
    for (std::uint32_t offset = 0; offset < outer;) {
        const auto count = std::min(inner, outer - offset);
        function(offset, count);
        offset += count; // Last increment reaches outer exactly, without wrapping.
    }
}
template<class Launch, class Observe>
inline void for_each_stream1_transformer_chunk_observed(
    std::uint32_t outer, std::uint32_t inner, Launch&& launch, Observe&& observe) {
    for_each_stream1_transformer_chunk(outer, inner,
        [&](std::uint32_t offset, std::uint32_t count) {
            launch(offset, count);
            observe(offset, count);
        });
}
}
