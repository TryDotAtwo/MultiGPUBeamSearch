#include "hash.hpp"

#include <random>
#include <stdexcept>

namespace beam {

bool hash_less(const Hash128& a, const Hash128& b) {
    if (a.hi != b.hi) {
        return a.hi < b.hi;
    }
    return a.lo < b.lo;
}

std::uint32_t pack_route(std::uint16_t source_rank, std::uint8_t owner, std::uint8_t move) {
    return (static_cast<std::uint32_t>(source_rank) << 16) |
           (static_cast<std::uint32_t>(owner) << 8) |
           static_cast<std::uint32_t>(move);
}

std::uint16_t unpack_source_rank(std::uint32_t route_packed) {
    return static_cast<std::uint16_t>(route_packed >> 16);
}

std::uint8_t unpack_owner(std::uint32_t route_packed) {
    return static_cast<std::uint8_t>((route_packed >> 8) & 0xff);
}

std::uint8_t unpack_move(std::uint32_t route_packed) {
    return static_cast<std::uint8_t>(route_packed & 0xff);
}

Hash128 hash_state(const State128& state, const ZobristTable& zobrist) {
    Hash128 hash{0, 0};
    for (std::size_t p = 0; p < STATE_STORAGE_LEN; ++p) {
        const Hash128 h = zobrist[p][state.v[p]];
        hash.lo ^= h.lo;
        hash.hi ^= h.hi;
    }
    return hash;
}

ZobristTable make_deterministic_zobrist(std::uint64_t seed) {
    ZobristTable table{};
    std::mt19937_64 rng(seed);
    for (std::size_t p = 0; p < STATE_LEN; ++p) {
        for (std::size_t v = 0; v < STATE_VALUE_PAD; ++v) {
            table[p][v] = Hash128{rng(), rng()};
        }
    }
    for (std::size_t p = STATE_LEN; p < STATE_STORAGE_LEN; ++p) {
        for (std::size_t v = 0; v < STATE_VALUE_PAD; ++v) {
            table[p][v] = Hash128{0, 0};
        }
    }
    return table;
}

std::uint8_t owner_from_hash128(const Hash128& hash, std::uint32_t world_size) {
    if (world_size == 0 || world_size > 256) {
        throw std::invalid_argument("world_size must be in [1, 256]");
    }
    return static_cast<std::uint8_t>(hash128_owner_distribution_key(hash) % world_size);
}

std::uint32_t shard_from_hash128(const Hash128& hash, std::uint32_t shard_count) {
    if (shard_count == 0) {
        throw std::invalid_argument("shard_count must be positive");
    }
    return static_cast<std::uint32_t>(hash128_shard_distribution_key(hash) % shard_count);
}

} // namespace beam
