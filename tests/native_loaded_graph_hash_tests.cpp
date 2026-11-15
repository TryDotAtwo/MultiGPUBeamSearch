#include "../src/native_input.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int main() {
    using namespace beam;
    State128 center{};
    for (std::size_t i = 0; i < STATE_LEN; ++i)
        center.v[i] = static_cast<std::uint8_t>(i);
    if constexpr (STATE_LEN > 128) center.v[STATE_LEN - 1] = 255;
    std::vector<std::uint8_t> generators(MOVE_COUNT * STATE_STORAGE_LEN);
    for (std::size_t move = 0; move < MOVE_COUNT; ++move) {
        for (std::size_t i = 0; i < STATE_LEN; ++i)
            generators[move * STATE_STORAGE_LEN + i] = static_cast<std::uint8_t>((i + move) % STATE_LEN);
        for (std::size_t i = STATE_LEN; i < STATE_STORAGE_LEN; ++i)
            generators[move * STATE_STORAGE_LEN + i] = static_cast<std::uint8_t>(i);
    }
    const auto digest = input::loaded_permutation_graph_sha256(center, generators);
#if BEAM_STATE_LOGICAL_BYTES == 120 && BEAM_MOVE_COUNT == 18
    if (digest != "1d41acb692d04eae581378bf3477f2a880e61bbeafdce20a8d161e1934b698eb")
        throw std::runtime_error("120-position graph hash mismatch");
#elif BEAM_STATE_LOGICAL_BYTES == 150 && BEAM_MOVE_COUNT == 255
    if (digest != "a80ac3205e883881dde14e355f3c54663d8259a5be6f04bf9c4559a1b0f70eba")
        throw std::runtime_error("150-position graph hash mismatch");
#elif BEAM_STATE_LOGICAL_BYTES == 96 && BEAM_MOVE_COUNT == 24
    if (digest != "5b30b66c842c6b4d405e5765702d8d46f9b352d7c5d853d87f2778ecc48cb3a6")
        throw std::runtime_error("96-position graph hash mismatch");
#endif
    if constexpr (STATE_LEN < STATE_STORAGE_LEN) {
        center.v[STATE_LEN] = 77;
        generators[STATE_LEN] ^= 17;
    }
    if (input::loaded_permutation_graph_sha256(center, generators) != digest)
        throw std::runtime_error("padding must not affect logical graph hash");
    center.v[0] ^= 1;
    if (input::loaded_permutation_graph_sha256(center, generators) == digest)
        throw std::runtime_error("center change was not bound");
    center.v[0] ^= 1;
    if constexpr (STATE_LEN >= 2) {
        std::swap(generators[0], generators[1]);
        if (input::loaded_permutation_graph_sha256(center, generators) == digest)
            throw std::runtime_error("generator change was not bound");
    }
    generators.pop_back();
    try {
        (void)input::loaded_permutation_graph_sha256(center, generators);
    } catch (const std::runtime_error&) {
        std::cout << "NATIVE_LOADED_GRAPH_HASH_OK\n";
        return 0;
    }
    throw std::runtime_error("truncated generator buffer was accepted");
}
