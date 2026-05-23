#pragma once

#include "types.hpp"

#include <cstddef>
#include <cstdint>

namespace beam {

BEAM_HOST_DEVICE inline std::uint64_t rotl64(std::uint64_t value, unsigned shift) {
    shift &= 63U;
    if (shift == 0U) {
        return value;
    }
    return (value << shift) | (value >> (64U - shift));
}

BEAM_HOST_DEVICE inline std::uint64_t hash128_distribution_key(Hash128 hash) {
    std::uint64_t x = hash.lo ^ rotl64(hash.hi, 32U) ^ 0x9E3779B97F4A7C15ULL;
    x ^= x >> 30U;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27U;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31U;
    return x;
}

Hash128 hash_state(const State128& state, const ZobristTable& zobrist);
ZobristTable make_deterministic_zobrist(std::uint64_t seed);
std::uint8_t owner_from_hash128(const Hash128& hash, std::uint32_t world_size);
std::uint32_t shard_from_hash128(const Hash128& hash, std::uint32_t shard_count);

} // namespace beam
