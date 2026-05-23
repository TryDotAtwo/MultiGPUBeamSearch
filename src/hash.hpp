#pragma once

#include "types.hpp"

#include <cstddef>

namespace beam {

Hash128 hash_state(const State128& state, const ZobristTable& zobrist);
ZobristTable make_deterministic_zobrist(std::uint64_t seed);
std::uint8_t owner_from_hash128(const Hash128& hash, std::uint32_t world_size);
std::uint32_t shard_from_hash128(const Hash128& hash, std::uint32_t shard_count);

} // namespace beam
