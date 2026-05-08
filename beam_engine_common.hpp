#pragma once

#include <cstdint>
#include <cstddef>

namespace beam_engine {

// BeamMeta describes one state in next_state_pool. State bytes are stored out-of-line.
struct alignas(32) BeamMeta {
    uint64_t hash;
    uint64_t fingerprint;
    uint32_t parent_idx;
    uint16_t score_q;
    uint8_t  action;
    uint8_t  parent_rank;
    uint16_t flags;
    uint16_t pad0;
};
static_assert(sizeof(BeamMeta) == 32, "BeamMeta must be 32 bytes");

// HashSlot is publication-safe: hash is EMPTY, BUSY, or a published hash.
// best_key = (score_q << 16) | tie16, updated with atomicMax.
struct alignas(32) HashSlot {
    uint64_t hash;
    uint64_t fingerprint;
    uint32_t pool_idx;
    uint32_t best_key;
    uint32_t flags;      // 0 = not committed; 1 = committed
    uint32_t pad0;
};
static_assert(sizeof(HashSlot) == 32, "HashSlot must be 32 bytes");

struct alignas(16) CandidateRecord {
    uint8_t  state[120];
    uint64_t hash;
    uint64_t fingerprint;
    uint32_t parent_idx;
    uint16_t score_q;
    uint8_t  action;
    uint8_t  parent_rank;
    uint8_t  valid;
    uint8_t  pad[15];
};
static_assert(sizeof(CandidateRecord) == 160, "CandidateRecord must be 160 bytes");

enum CounterIndex : int {
    COUNTER_NEXT_POOL_SIZE = 0,
    COUNTER_LOCAL_INSERTED = 1,
    COUNTER_LOCAL_DUPLICATE = 2,
    COUNTER_REMOTE_PACKED = 3,
    COUNTER_BUCKET_OVERFLOW = 4,
    COUNTER_HASH_OVERFLOW = 5,
    COUNTER_PRUNED = 6,
    COUNTER_LOCAL_UPDATED = 7,
    COUNTER_RESERVED = 8
};

enum BeamStatusIndex : int {
    STATUS_CURRENT_SIZE = 0,
    STATUS_COMPACTED_SIZE = 1,
    STATUS_FOUND = 2,
    STATUS_FOUND_LOCAL_INDEX = 3,
    STATUS_FOUND_ACTION = 4,
    STATUS_GRAPH_CAPTURED = 5,
    STATUS_MAX_ACTIVE_SCORE = 6,
    STATUS_LOCAL_FOUND = 7,
    STATUS_RESERVED = 8
};

static constexpr uint64_t HASH_EMPTY = 0ull;
static constexpr uint64_t HASH_BUSY = 1ull;
static constexpr uint64_t HASH_TOMBSTONE = 2ull;

static constexpr int STATE_SIZE_BYTES_FIXED = 120;
static constexpr int FANOUT_FIXED = 24;
static constexpr int SCORE_BINS = 65536;

} // namespace beam_engine
