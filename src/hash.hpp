#pragma once

#include "types.hpp"

#include <cstddef>

namespace beam {

Hash128 hash_state(const State128& state, const ZobristTable& zobrist);
ZobristTable make_deterministic_zobrist(std::uint64_t seed);
std::uint8_t owner_from_hash128(const Hash128& hash, std::uint32_t world_size);
std::uint32_t shard_from_hash128(const Hash128& hash, std::uint32_t shard_count);

BEAM_HOST_DEVICE inline std::uint64_t hash128_rotl64(std::uint64_t value, unsigned shift) {
    return (value << shift) | (value >> (64U - shift));
}

BEAM_HOST_DEVICE inline std::uint64_t hash128_mix64(std::uint64_t value) {
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return value;
}

BEAM_HOST_DEVICE inline std::uint64_t hash128_distribution_key(Hash128 hash, std::uint64_t domain_salt) {
    std::uint64_t value = hash.lo ^ hash128_rotl64(hash.hi, 32U) ^ domain_salt;
    value ^= hash128_mix64(hash.hi + 0x9e3779b97f4a7c15ULL);
    return hash128_mix64(value);
}

BEAM_HOST_DEVICE inline std::uint64_t hash128_owner_distribution_key(Hash128 hash) {
    return hash128_distribution_key(hash, 0x243f6a8885a308d3ULL);
}

BEAM_HOST_DEVICE inline std::uint64_t hash128_shard_distribution_key(Hash128 hash) {
    return hash128_distribution_key(hash, 0x13198a2e03707344ULL);
}

} // namespace beam
