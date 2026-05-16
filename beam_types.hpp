#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace beam_v6 {

#ifdef __CUDACC__
#define BEAM_V6_HD __host__ __device__
#else
#define BEAM_V6_HD
#endif

using StateValue = uint8_t;

static constexpr int STATE_LEN = 120;
static constexpr int STATE_STORAGE_LEN = 128;
static constexpr int STATE_VALUE_PAD = 128;
static constexpr int MOVE_COUNT = 24;

static constexpr float SCORE_MAX_Q = 300.0f;
static constexpr uint32_t SCORE_SCALE = 256u;
static constexpr uint32_t SCORE_MAX_KEY = static_cast<uint32_t>(SCORE_MAX_Q * static_cast<float>(SCORE_SCALE));
static constexpr uint32_t SCORE_BIN_COUNT = SCORE_MAX_KEY + 1u;
static constexpr uint32_t GOAL_SCORE_KEY = 0u;
static constexpr uint32_t THRESHOLD_UNINITIALIZED_VALUE = std::numeric_limits<uint32_t>::max();

struct alignas(16) State128 {
    StateValue v[STATE_STORAGE_LEN];
};

struct alignas(16) Hash128 {
    uint64_t lo;
    uint64_t hi;
};

struct alignas(32) CandidateMeta {
    Hash128 hash;
    uint64_t parent_idx;
    uint32_t score_key;
    uint32_t route_packed;
};

struct alignas(16) FinalRequest {
    uint64_t parent_idx;
    uint32_t target_local_idx;
    uint16_t return_rank;
    uint8_t move;
    uint8_t pad;
};

using FinalResponse = State128;

static_assert(sizeof(State128) == 128, "State128 must be 128 bytes");
static_assert(alignof(State128) == 16, "State128 must be 16-byte aligned");
static_assert(sizeof(Hash128) == 16, "Hash128 must be 16 bytes");
static_assert(alignof(Hash128) == 16, "Hash128 must be 16-byte aligned");
static_assert(sizeof(CandidateMeta) == 32, "CandidateMeta must be 32 bytes");
static_assert(alignof(CandidateMeta) == 32, "CandidateMeta must be 32-byte aligned");
static_assert(sizeof(FinalRequest) == 16, "FinalRequest must be 16 bytes");
static_assert(alignof(FinalRequest) == 16, "FinalRequest must be 16-byte aligned");
static_assert(sizeof(FinalResponse) == 128, "FinalResponse must be 128 bytes");
static_assert(alignof(FinalResponse) == 16, "FinalResponse must be 16-byte aligned");
static_assert(SCORE_MAX_KEY == 76800u, "Default SCORE_MAX_KEY must be 76800");
static_assert(SCORE_BIN_COUNT == 76801u, "Default SCORE_BIN_COUNT must be 76801");

BEAM_V6_HD inline uint32_t pack_route(uint16_t source_rank, uint8_t owner, uint8_t move) {
    return (uint32_t(source_rank) << 16) | (uint32_t(owner) << 8) | uint32_t(move);
}

BEAM_V6_HD inline uint16_t unpack_source_rank(uint32_t route_packed) {
    return uint16_t(route_packed >> 16);
}

BEAM_V6_HD inline uint8_t unpack_owner(uint32_t route_packed) {
    return uint8_t((route_packed >> 8) & 0xffu);
}

BEAM_V6_HD inline uint8_t unpack_move(uint32_t route_packed) {
    return uint8_t(route_packed & 0xffu);
}

BEAM_V6_HD inline void final_response_set_target_local_idx(FinalResponse& response, uint32_t target_local_idx) {
    response.v[120] = uint8_t(target_local_idx);
    response.v[121] = uint8_t(target_local_idx >> 8);
    response.v[122] = uint8_t(target_local_idx >> 16);
    response.v[123] = uint8_t(target_local_idx >> 24);
}

BEAM_V6_HD inline uint32_t final_response_get_target_local_idx(const FinalResponse& response) {
    return uint32_t(response.v[120]) |
           (uint32_t(response.v[121]) << 8) |
           (uint32_t(response.v[122]) << 16) |
           (uint32_t(response.v[123]) << 24);
}

BEAM_V6_HD inline void clear_state_padding(State128& state) {
    for (int i = STATE_LEN; i < STATE_STORAGE_LEN; ++i) {
        state.v[i] = 0;
    }
}

inline uint32_t q_to_score_key_host(float q) {
    q = std::min(std::max(q, 0.0f), SCORE_MAX_Q);
    return static_cast<uint32_t>(q * float(SCORE_SCALE) + 0.5f);
}

BEAM_V6_HD inline uint64_t pack_stream3_val(uint32_t score_key, uint32_t payload_id) {
    return (uint64_t(score_key) << 32) | uint64_t(payload_id);
}

BEAM_V6_HD inline uint32_t stream3_val_score_key(uint64_t stream3_val) {
    return uint32_t(stream3_val >> 32);
}

BEAM_V6_HD inline uint32_t stream3_val_payload_id(uint64_t stream3_val) {
    return uint32_t(stream3_val & 0xffffffffu);
}

BEAM_V6_HD inline uint32_t payload_id(uint32_t ring_slot, uint32_t parent_local, uint8_t move, uint32_t b_micro) {
    return ring_slot * (b_micro * MOVE_COUNT) + parent_local * MOVE_COUNT + uint32_t(move);
}

inline uint32_t threshold_update(uint32_t current_threshold, bool& threshold_initialized, uint64_t total_survivors, uint64_t global_beam_width_effective, uint32_t new_threshold) {
    if (!threshold_initialized && total_survivors < global_beam_width_effective) {
        return THRESHOLD_UNINITIALIZED_VALUE;
    }
    if (total_survivors >= global_beam_width_effective) {
        threshold_initialized = true;
        return std::min(current_threshold, new_threshold);
    }
    return current_threshold;
}

} // namespace beam_v6
