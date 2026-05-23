#pragma once

#include "config.hpp"

#include <array>
#include <cstdint>

namespace beam {

using StateValue = std::uint8_t;

struct alignas(16) State128 {
    StateValue v[STATE_STORAGE_LEN];
};

struct alignas(16) Hash128 {
    std::uint64_t lo;
    std::uint64_t hi;
};

struct alignas(32) CandidateMeta {
    Hash128 hash;
    std::uint64_t parent_idx;
    std::uint32_t score_key;
    std::uint32_t route_packed;
};

struct alignas(16) FinalRequest {
    std::uint64_t parent_idx;
    std::uint32_t target_local_idx;
    std::uint16_t return_rank;
    std::uint8_t move;
    std::uint8_t pad;
};

using FinalResponse = State128;
using Generator = std::array<std::uint8_t, STATE_STORAGE_LEN>;
using State = std::array<std::uint8_t, STATE_STORAGE_LEN>;
using ZobristTable = std::array<std::array<Hash128, STATE_VALUE_PAD>, STATE_STORAGE_LEN>;

#ifdef __CUDACC__
#define BEAM_HOST_DEVICE __host__ __device__
#else
#define BEAM_HOST_DEVICE
#endif

static_assert(sizeof(State128) == 128);
static_assert(alignof(State128) == 16);
static_assert(sizeof(Hash128) == 16);
static_assert(alignof(Hash128) == 16);
static_assert(sizeof(CandidateMeta) == 32);
static_assert(alignof(CandidateMeta) == 32);
static_assert(sizeof(FinalRequest) == 16);
static_assert(alignof(FinalRequest) == 16);
static_assert(sizeof(FinalResponse) == 128);
static_assert(alignof(FinalResponse) == 16);

BEAM_HOST_DEVICE inline bool operator==(Hash128 a, Hash128 b) {
    return a.lo == b.lo && a.hi == b.hi;
}

bool hash_less(const Hash128& a, const Hash128& b);

std::uint32_t pack_route(std::uint16_t source_rank, std::uint8_t owner, std::uint8_t move);
std::uint16_t unpack_source_rank(std::uint32_t route_packed);
std::uint8_t unpack_owner(std::uint32_t route_packed);
std::uint8_t unpack_move(std::uint32_t route_packed);

} // namespace beam
