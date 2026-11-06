#pragma once

#include "hash.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace beam::cnot {

using MoveId = std::uint16_t;
inline constexpr int max_qubits = 256;

BEAM_HOST_DEVICE constexpr int words_per_row(int n) { return (n + 63) / 64; }
BEAM_HOST_DEVICE constexpr int move_count(int n) { return n * (n - 1); }
BEAM_HOST_DEVICE constexpr std::uint64_t tail_mask(int n) {
    const int remainder = n % 64;
    return remainder == 0 ? ~std::uint64_t{0} : (std::uint64_t{1} << remainder) - 1;
}

struct Gate { int control; int target; };

// Preconditions: 2 <= n <= 256, 0 <= control,target < n, control != target.
BEAM_HOST_DEVICE constexpr MoveId encode_move(int n, int control, int target) {
    return static_cast<MoveId>(control * (n - 1) + target - (target > control));
}
// Preconditions: 2 <= n <= 256 and id < move_count(n).
BEAM_HOST_DEVICE constexpr Gate decode_move(int n, MoveId id) {
    const int control = id / (n - 1);
    const int offset = id % (n - 1);
    return {control, offset + (offset >= control)};
}

// CNOT interpretation of CandidateMeta.route_packed. The receiver is the
// owner, so all 16 low bits are available for the physical generator ID.
BEAM_HOST_DEVICE constexpr std::uint32_t pack_route(std::uint16_t source_rank, MoveId move) {
    return (static_cast<std::uint32_t>(source_rank) << 16) | move;
}
BEAM_HOST_DEVICE constexpr std::uint16_t route_source(std::uint32_t route) {
    return static_cast<std::uint16_t>(route >> 16);
}
BEAM_HOST_DEVICE constexpr MoveId route_move(std::uint32_t route) {
    return static_cast<MoveId>(route);
}

struct alignas(16) FinalRequest {
    std::uint64_t parent_idx;
    std::uint32_t target_local_idx;
    std::uint16_t return_rank;
    MoveId move;
};
static_assert(sizeof(FinalRequest) == 16 && alignof(FinalRequest) == 16);
static_assert(offsetof(FinalRequest, move) == 14);
static_assert(move_count(max_qubits) - 1 <= std::numeric_limits<MoveId>::max());
static_assert(encode_move(144, 143, 142) == 20591);

// A position-dependent contribution from one canonical word. Combining by
// XOR makes the 128-bit hash identical for serial and warp reductions.
BEAM_HOST_DEVICE inline Hash128 hash_word(std::uint64_t word, int position) {
    const auto k = static_cast<std::uint64_t>(position + 1);
    return {
        hash128_mix64(word ^ (k * 0x9e3779b97f4a7c15ULL)),
        hash128_mix64(word + k * 0xd6e8feb86659fd93ULL)
    };
}
BEAM_HOST_DEVICE inline std::uint64_t owner_key(Hash128 hash, int n) {
    hash.lo ^= static_cast<std::uint64_t>(n);
    hash.hi ^= static_cast<std::uint64_t>(n) << 32;
    return hash128_owner_distribution_key(hash);
}

} // namespace beam::cnot
