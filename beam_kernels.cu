#include <cuda_runtime.h>
#include <stdint.h>
#include "beam_engine_common.hpp"

namespace beam_engine {

__constant__ uint8_t c_action_permutation[FANOUT_FIXED][STATE_SIZE_BYTES_FIXED];
__constant__ uint8_t c_action_permutation_loaded;
__constant__ uint8_t c_central_state[STATE_SIZE_BYTES_FIXED];
__constant__ uint8_t c_central_state_loaded;

__device__ __forceinline__ uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

__device__ __forceinline__ uint64_t hash_state_120(const uint8_t* s) {
    uint64_t h = 1469598103934665603ULL;
    #pragma unroll 4
    for (int i = 0; i < STATE_SIZE_BYTES_FIXED; ++i) {
        h ^= static_cast<uint64_t>(s[i]);
        h *= 1099511628211ULL;
    }
    h = mix64(h);
    if (h == HASH_EMPTY || h == HASH_BUSY || h == HASH_TOMBSTONE) h += 4ULL;
    return h;
}

__device__ __forceinline__ uint64_t fingerprint_state_120(const uint8_t* s) {
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    #pragma unroll 4
    for (int i = 0; i < STATE_SIZE_BYTES_FIXED; ++i) {
        h ^= static_cast<uint64_t>(s[i]) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    h = mix64(h ^ 0xd6e8feb86659fd93ULL);
    if (h == 0ULL) h = 4ULL;
    return h;
}

__device__ __forceinline__ uint16_t tie16(uint64_t fingerprint) {
    return static_cast<uint16_t>((fingerprint ^ (fingerprint >> 32)) & 0xFFFFu);
}

__device__ __forceinline__ uint32_t make_best_key(uint16_t score_q, uint64_t fingerprint) {
    return (static_cast<uint32_t>(score_q) << 16) | static_cast<uint32_t>(tie16(fingerprint));
}

__device__ __forceinline__ void apply_move_table(const uint8_t* src, uint8_t action, uint8_t* dst) {
    if (action < FANOUT_FIXED && c_action_permutation_loaded) {
        #pragma unroll 4
        for (int i = 0; i < STATE_SIZE_BYTES_FIXED; ++i) {
            dst[i] = src[c_action_permutation[action][i]];
        }
        return;
    }
    #pragma unroll 4
    for (int i = 0; i < STATE_SIZE_BYTES_FIXED; ++i) dst[i] = src[i];
    int p0 = action % STATE_SIZE_BYTES_FIXED;
    int p1 = (p0 + 17) % STATE_SIZE_BYTES_FIXED;
    uint8_t tmp = dst[p0];
    dst[p0] = static_cast<uint8_t>(dst[p1] ^ action);
    dst[p1] = tmp;
}

__device__ __forceinline__ int hamming_to_central(const uint8_t* s) {
    int d = 0;
    if (!c_central_state_loaded) return STATE_SIZE_BYTES_FIXED;
    #pragma unroll 4
    for (int i = 0; i < STATE_SIZE_BYTES_FIXED; ++i) {
        d += (s[i] != c_central_state[i]);
    }
    return d;
}

__device__ __forceinline__ bool is_central_state(const uint8_t* s) {
    if (!c_central_state_loaded) return false;
    #pragma unroll 4
    for (int i = 0; i < STATE_SIZE_BYTES_FIXED; ++i) {
        if (s[i] != c_central_state[i]) return false;
    }
    return true;
}

// Test/default inference: score(action(parent)) by distance to uploaded central state.
// This makes shallow generated correctness cases deterministic.
extern "C" __global__ void kernel_dummy_inference(
    const uint8_t* __restrict__ beam_current,
    const uint8_t* __restrict__ current_active_flags,
    uint16_t* __restrict__ score_ring,
    int slot,
    int state_size_bytes,
    int fanout,
    int b_micro,
    int64_t start_state,
    int micro_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= micro_size) return;

    const int64_t parent_idx = start_state + idx;
    uint16_t* slot_scores = score_ring + static_cast<int64_t>(slot) * b_micro * fanout;
    if (!current_active_flags[parent_idx]) {
        for (int a = 0; a < fanout; ++a) slot_scores[a * b_micro + idx] = 0;
        return;
    }

    const uint8_t* s = beam_current + parent_idx * state_size_bytes;
    uint64_t ph = hash_state_120(s);
    for (int a = 0; a < fanout; ++a) {
        uint8_t cand[STATE_SIZE_BYTES_FIXED];
        apply_move_table(s, static_cast<uint8_t>(a), cand);
        int ham = hamming_to_central(cand);
        uint16_t q;
        if (c_central_state_loaded) {
            int raw = 65535 - ham * 512;
            if (raw < 1) raw = 1;
            if (ham == 0) raw = 65535;
            q = static_cast<uint16_t>(raw);
        } else {
            q = static_cast<uint16_t>(mix64(ph ^ static_cast<uint64_t>(a + 0x1009)) & 0xFFFFu);
        }
        slot_scores[a * b_micro + idx] = q;
    }
}


extern "C" __global__ void kernel_copy_i16_scores_to_ring(
    const int16_t* __restrict__ model_scores,
    uint16_t* __restrict__ score_ring,
    int slot,
    int b_micro,
    int fanout,
    int micro_size
) {
    int64_t lane = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(micro_size) * fanout;
    if (lane >= total) return;
    int idx = static_cast<int>(lane / fanout);
    int action = static_cast<int>(lane - static_cast<int64_t>(idx) * fanout);
    const int16_t v = model_scores[static_cast<int64_t>(idx) * fanout + action];
    uint16_t* slot_scores = score_ring + static_cast<int64_t>(slot) * b_micro * fanout;
    slot_scores[static_cast<int64_t>(action) * b_micro + idx] = static_cast<uint16_t>(v);
}

extern "C" __global__ void kernel_quantize_f32_scores_to_ring(
    const float* __restrict__ model_scores,
    uint16_t* __restrict__ score_ring,
    int slot,
    int b_micro,
    int fanout,
    int micro_size,
    float scale,
    float bias
) {
    int64_t lane = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(micro_size) * fanout;
    if (lane >= total) return;
    int idx = static_cast<int>(lane / fanout);
    int action = static_cast<int>(lane - static_cast<int64_t>(idx) * fanout);
    float y = model_scores[static_cast<int64_t>(idx) * fanout + action] * scale + bias;
    if (y < 0.0f) y = 0.0f;
    if (y > 65535.0f) y = 65535.0f;
    uint16_t q = static_cast<uint16_t>(y + 0.5f);
    uint16_t* slot_scores = score_ring + static_cast<int64_t>(slot) * b_micro * fanout;
    slot_scores[static_cast<int64_t>(action) * b_micro + idx] = q;
}

extern "C" __global__ void kernel_reset_net_slot(
    CandidateRecord* __restrict__ send_buckets,
    CandidateRecord* __restrict__ recv_buckets,
    int32_t* __restrict__ send_counts,
    int32_t* __restrict__ recv_counts,
    int net_slot,
    int world_size,
    int bucket_cap
) {
    (void)send_buckets;
    (void)recv_buckets;
    (void)bucket_cap;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < world_size) {
        send_counts[net_slot * world_size + i] = 0;
        recv_counts[net_slot * world_size + i] = 0;
    }
}

struct InsertResult {
    uint32_t pool_idx;
    int status;              // 0 failed, 1 inserted, 2 updated, 3 duplicate noop
    uint16_t old_score;
};

__device__ InsertResult hash_insert_or_update(
    const uint8_t* cand_state,
    uint64_t hash,
    uint64_t fingerprint,
    uint16_t score_q,
    uint32_t parent_idx,
    uint8_t action,
    uint8_t parent_rank,
    uint8_t* __restrict__ next_state_pool,
    BeamMeta* __restrict__ next_meta,
    HashSlot* __restrict__ hash_table,
    uint8_t* __restrict__ active_flags,
    int32_t* __restrict__ free_indices,
    int32_t* __restrict__ free_count,
    int32_t* __restrict__ counters,
    int32_t* __restrict__ beam_status,
    int state_size_bytes,
    int hash_capacity,
    int k_work,
    int probe_limit
) {
    uint64_t pos0 = hash % static_cast<uint64_t>(hash_capacity);
    uint32_t key_new = make_best_key(score_q, fingerprint);
    int64_t reusable_pos = -1;
    for (int probe = 0; probe < probe_limit; ++probe) {
        uint64_t pos = (pos0 + probe) % static_cast<uint64_t>(hash_capacity);
        unsigned long long* key_ptr = reinterpret_cast<unsigned long long*>(&hash_table[pos].hash);
        unsigned long long old = static_cast<unsigned long long>(hash_table[pos].hash);
        if (old == HASH_TOMBSTONE) {
            if (reusable_pos < 0) reusable_pos = static_cast<int64_t>(pos);
            continue;
        }
        if (old == HASH_EMPTY) {
            uint64_t insert_pos = reusable_pos >= 0 ? static_cast<uint64_t>(reusable_pos) : pos;
            unsigned long long expected = reusable_pos >= 0
                ? static_cast<unsigned long long>(HASH_TOMBSTONE)
                : static_cast<unsigned long long>(HASH_EMPTY);
            key_ptr = reinterpret_cast<unsigned long long*>(&hash_table[insert_pos].hash);
            old = atomicCAS(key_ptr, expected, static_cast<unsigned long long>(HASH_BUSY));
            if (old != expected) continue;
            pos = insert_pos;
        }

        if (old == HASH_EMPTY || old == HASH_TOMBSTONE) {
            int new_idx = atomicAdd(&counters[COUNTER_NEXT_POOL_SIZE], 1);
            if (new_idx >= k_work) {
                int free_pos = atomicSub(free_count, 1) - 1;
                if (free_pos >= 0) {
                    new_idx = free_indices[free_pos];
                } else {
                    atomicAdd(free_count, 1);
                    atomicAdd(&counters[COUNTER_HASH_OVERFLOW], 1);
                    // Leave BUSY slot unpublished; no state pool entry exists. Safer than publishing bad pointer.
                    atomicExch(key_ptr, static_cast<unsigned long long>(HASH_TOMBSTONE));
                    return {UINT32_MAX, 0, 0};
                }
            }

            uint8_t* dst = next_state_pool + static_cast<int64_t>(new_idx) * state_size_bytes;
            #pragma unroll 4
            for (int i = 0; i < STATE_SIZE_BYTES_FIXED; ++i) dst[i] = cand_state[i];

            BeamMeta meta;
            meta.hash = hash;
            meta.fingerprint = fingerprint;
            meta.parent_idx = parent_idx;
            meta.score_q = score_q;
            meta.action = action;
            meta.parent_rank = parent_rank;
            meta.flags = 0;
            meta.pad0 = 0;
            next_meta[new_idx] = meta;
            active_flags[new_idx] = 1;

            hash_table[pos].fingerprint = fingerprint;
            hash_table[pos].pool_idx = static_cast<uint32_t>(new_idx);
            hash_table[pos].best_key = key_new;
            hash_table[pos].flags = 1;
            __threadfence();
            atomicExch(key_ptr, static_cast<unsigned long long>(hash));

            if (is_central_state(cand_state)) {
                if (atomicCAS(&beam_status[STATUS_FOUND], 0, 1) == 0) {
                    beam_status[STATUS_FOUND_LOCAL_INDEX] = new_idx;
                    beam_status[STATUS_FOUND_ACTION] = static_cast<int>(action);
                }
            }
            atomicAdd(&counters[COUNTER_LOCAL_INSERTED], 1);
            return {static_cast<uint32_t>(new_idx), 1, 0};
        }

        if (old == HASH_BUSY) {
            #pragma unroll 1
            for (int spin = 0; spin < 64 && old == HASH_BUSY; ++spin) {
                old = static_cast<unsigned long long>(hash_table[pos].hash);
            }
            if (old == HASH_BUSY) {
                continue;
            }
        }

        if (old == hash) {
            uint8_t slot_ready = hash_table[pos].flags;
            #pragma unroll 1
            for (int spin = 0; spin < 64 && slot_ready == 0; ++spin) {
                slot_ready = hash_table[pos].flags;
            }
            if (!slot_ready) {
                continue;
            }

            if (hash_table[pos].fingerprint == fingerprint) {
                uint32_t pool_idx = hash_table[pos].pool_idx;
                uint32_t old_key = hash_table[pos].best_key;
                uint16_t old_score = static_cast<uint16_t>(old_key >> 16);
                if (pool_idx < static_cast<uint32_t>(k_work) && key_new > old_key) {
                    uint32_t prev = atomicMax(&hash_table[pos].best_key, key_new);
                    if (key_new > prev) {
                        next_meta[pool_idx].score_q = score_q;
                        next_meta[pool_idx].parent_idx = parent_idx;
                        next_meta[pool_idx].action = action;
                        next_meta[pool_idx].parent_rank = parent_rank;
                        if (is_central_state(cand_state)) {
                            if (atomicCAS(&beam_status[STATUS_FOUND], 0, 1) == 0) {
                                beam_status[STATUS_FOUND_LOCAL_INDEX] = static_cast<int32_t>(pool_idx);
                                beam_status[STATUS_FOUND_ACTION] = static_cast<int>(action);
                            }
                        }
                        atomicAdd(&counters[COUNTER_LOCAL_UPDATED], 1);
                        return {pool_idx, 2, old_score};
                    }
                }
                atomicAdd(&counters[COUNTER_LOCAL_DUPLICATE], 1);
                return {pool_idx, 3, old_score};
            }
        }
    }
    atomicAdd(&counters[COUNTER_HASH_OVERFLOW], 1);
    return {UINT32_MAX, 0, 0};
}

__device__ __forceinline__ void update_hist_for_insert_result(
    uint32_t* __restrict__ local_hist,
    InsertResult r,
    uint16_t new_score
) {
    if (r.status == 1) {
        atomicAdd(&local_hist[new_score], 1u);
    } else if (r.status == 2) {
        if (r.old_score != new_score && r.old_score > 0) atomicSub(&local_hist[r.old_score], 1u);
        atomicAdd(&local_hist[new_score], 1u);
    }
}

extern "C" __global__ void kernel_process_score_slot(
    const uint8_t* __restrict__ beam_current,
    const uint8_t* __restrict__ current_active_flags,
    const uint16_t* __restrict__ score_ring,
    uint8_t* __restrict__ next_state_pool,
    BeamMeta* __restrict__ next_meta,
    HashSlot* __restrict__ hash_table,
    uint8_t* __restrict__ active_flags,
    int32_t* __restrict__ free_indices,
    int32_t* __restrict__ free_count,
    uint32_t* __restrict__ local_hist,
    int32_t* __restrict__ counters,
    int32_t* __restrict__ beam_status,
    CandidateRecord* __restrict__ send_buckets,
    int32_t* __restrict__ send_counts,
    const int32_t* __restrict__ threshold_cell,
    int slot,
    int net_slot,
    int world_size,
    int rank,
    int state_size_bytes,
    int fanout,
    int b_micro,
    int64_t start_state,
    int micro_size,
    int64_t candidate_lane_offset,
    int candidate_lanes,
    int bucket_cap,
    int hash_capacity,
    int k_work,
    int probe_limit
) {
    int64_t local_lane = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local_lane >= candidate_lanes) return;
    int64_t lane = candidate_lane_offset + local_lane;
    int64_t total = static_cast<int64_t>(micro_size) * fanout;
    if (lane >= total) return;

    int action = static_cast<int>(lane / micro_size);
    int local_idx = static_cast<int>(lane - static_cast<int64_t>(action) * micro_size);
    int64_t parent_global_idx = start_state + local_idx;
    if (!current_active_flags[parent_global_idx]) return;

    const uint16_t* slot_scores = score_ring + static_cast<int64_t>(slot) * b_micro * fanout;
    uint16_t score_q = slot_scores[action * b_micro + local_idx];
    int threshold_valid = threshold_cell[0];
    int threshold_q = threshold_cell[1];
    if (threshold_valid && static_cast<int>(score_q) <= threshold_q) return;

    const uint8_t* parent_state = beam_current + parent_global_idx * state_size_bytes;
    uint8_t cand_state[STATE_SIZE_BYTES_FIXED];
    apply_move_table(parent_state, static_cast<uint8_t>(action), cand_state);
    uint64_t h = hash_state_120(cand_state);
    uint64_t fp = fingerprint_state_120(cand_state);
    int owner = static_cast<int>(h % static_cast<uint64_t>(world_size));

    if (owner == rank) {
        InsertResult r = hash_insert_or_update(
            cand_state, h, fp, score_q,
            static_cast<uint32_t>(parent_global_idx), static_cast<uint8_t>(action), static_cast<uint8_t>(rank),
            next_state_pool, next_meta, hash_table, active_flags,
            free_indices, free_count, counters, beam_status, state_size_bytes, hash_capacity, k_work, probe_limit);
        if (r.pool_idx != UINT32_MAX) update_hist_for_insert_result(local_hist, r, score_q);
    } else {
        int pos = atomicAdd(&send_counts[net_slot * world_size + owner], 1);
        if (pos < bucket_cap) {
            CandidateRecord* rec = send_buckets + (static_cast<int64_t>(net_slot) * world_size + owner) * bucket_cap + pos;
            #pragma unroll 4
            for (int i = 0; i < STATE_SIZE_BYTES_FIXED; ++i) rec->state[i] = cand_state[i];
            rec->hash = h;
            rec->fingerprint = fp;
            rec->parent_idx = static_cast<uint32_t>(parent_global_idx);
            rec->score_q = score_q;
            rec->action = static_cast<uint8_t>(action);
            rec->parent_rank = static_cast<uint8_t>(rank);
            __threadfence();
            rec->valid = 1;
            atomicAdd(&counters[COUNTER_REMOTE_PACKED], 1);
        } else {
            atomicAdd(&counters[COUNTER_BUCKET_OVERFLOW], 1);
        }
    }
}

extern "C" __global__ void kernel_ingest_recv_slot(
    const CandidateRecord* __restrict__ recv_buckets,
    const int32_t* __restrict__ recv_counts,
    uint8_t* __restrict__ next_state_pool,
    BeamMeta* __restrict__ next_meta,
    HashSlot* __restrict__ hash_table,
    uint8_t* __restrict__ active_flags,
    int32_t* __restrict__ free_indices,
    int32_t* __restrict__ free_count,
    uint32_t* __restrict__ local_hist,
    int32_t* __restrict__ counters,
    int32_t* __restrict__ beam_status,
    const int32_t* __restrict__ threshold_cell,
    int net_slot,
    int world_size,
    int state_size_bytes,
    int bucket_cap,
    int hash_capacity,
    int k_work,
    int probe_limit
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(world_size) * bucket_cap;
    if (idx >= total) return;

    int peer = static_cast<int>(idx / bucket_cap);
    int pos = static_cast<int>(idx - static_cast<int64_t>(peer) * bucket_cap);
    int count = recv_counts[net_slot * world_size + peer];
    if (count < 0) count = 0;
    if (count > bucket_cap) count = bucket_cap;
    if (pos >= count) return;

    const CandidateRecord* rec = recv_buckets + static_cast<int64_t>(net_slot) * world_size * bucket_cap + idx;
    int threshold_valid = threshold_cell[0];
    int threshold_q = threshold_cell[1];
    if (threshold_valid && static_cast<int>(rec->score_q) <= threshold_q) return;

    InsertResult r = hash_insert_or_update(
        rec->state, rec->hash, rec->fingerprint, rec->score_q,
        rec->parent_idx, rec->action, rec->parent_rank,
        next_state_pool, next_meta, hash_table, active_flags, free_indices, free_count,
        counters, beam_status, state_size_bytes, hash_capacity, k_work, probe_limit);
    if (r.pool_idx != UINT32_MAX) update_hist_for_insert_result(local_hist, r, rec->score_q);
}

extern "C" __global__ void kernel_compute_threshold(
    const uint32_t* __restrict__ global_hist,
    int32_t* __restrict__ threshold_cell,
    int score_bins,
    int64_t target_global_beam
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int64_t acc = 0;
    int threshold = 0;
    for (int b = score_bins - 1; b >= 0; --b) {
        acc += static_cast<int64_t>(global_hist[b]);
        if (acc >= target_global_beam) {
            threshold = b;
            break;
        }
    }
    if (acc >= target_global_beam) {
        threshold_cell[0] = 1;
        threshold_cell[1] = threshold;
    } else {
        threshold_cell[0] = 0;
        threshold_cell[1] = 0;
    }
}

extern "C" __global__ void kernel_prune_by_threshold(
    BeamMeta* __restrict__ next_meta,
    HashSlot* __restrict__ hash_table,
    uint8_t* __restrict__ active_flags,
    int32_t* __restrict__ free_indices,
    int32_t* __restrict__ free_count,
    uint32_t* __restrict__ local_hist,
    int32_t* __restrict__ counters,
    const int32_t* __restrict__ threshold_cell,
    int k_work,
    int hash_capacity,
    int probe_limit
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k_work) return;
    if (!active_flags[idx]) return;
    if (!threshold_cell[0]) return;
    uint16_t score_q = next_meta[idx].score_q;
    if (static_cast<int>(score_q) <= threshold_cell[1]) {
        active_flags[idx] = 0;
        uint64_t h = next_meta[idx].hash;
        uint64_t pos0 = h % static_cast<uint64_t>(hash_capacity);
        for (int probe = 0; probe < probe_limit; ++probe) {
            uint64_t pos = (pos0 + probe) % static_cast<uint64_t>(hash_capacity);
            if (hash_table[pos].hash == h && hash_table[pos].pool_idx == static_cast<uint32_t>(idx)) {
                hash_table[pos].flags = 0;
                unsigned long long* key_ptr = reinterpret_cast<unsigned long long*>(&hash_table[pos].hash);
                atomicExch(key_ptr, static_cast<unsigned long long>(HASH_TOMBSTONE));
                break;
            }
            if (hash_table[pos].hash == HASH_EMPTY) break;
        }
        int free_pos = atomicAdd(free_count, 1);
        if (free_pos < k_work) free_indices[free_pos] = idx;
        atomicAdd(&counters[COUNTER_PRUNED], 1);
        if (score_q > 0) atomicSub(&local_hist[score_q], 1u);
    }
}

extern "C" __global__ void kernel_compact_next_to_current(
    const uint8_t* __restrict__ next_state_pool,
    const BeamMeta* __restrict__ next_meta,
    const uint8_t* __restrict__ active_flags,
    uint8_t* __restrict__ beam_current,
    uint8_t* __restrict__ current_active_flags,
    int32_t* __restrict__ history_parent_idx,
    uint8_t* __restrict__ history_parent_rank,
    uint8_t* __restrict__ history_action,
    uint8_t* __restrict__ history_valid,
    const int32_t* __restrict__ history_depth_cell,
    int32_t* __restrict__ beam_status,
    int state_size_bytes,
    int k_work,
    int n_local
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k_work) return;
    if (!active_flags[idx]) return;
    int pos = atomicAdd(&beam_status[STATUS_COMPACTED_SIZE], 1);
    if (pos >= n_local) return;
    const uint8_t* src = next_state_pool + static_cast<int64_t>(idx) * state_size_bytes;
    uint8_t* dst = beam_current + static_cast<int64_t>(pos) * state_size_bytes;
    #pragma unroll 4
    for (int i = 0; i < STATE_SIZE_BYTES_FIXED; ++i) dst[i] = src[i];
    current_active_flags[pos] = 1;
    int history_depth = history_depth_cell[0];
    if (history_depth >= 0) {
        int64_t hpos = static_cast<int64_t>(history_depth) * n_local + pos;
        BeamMeta meta = next_meta[idx];
        history_parent_idx[hpos] = static_cast<int32_t>(meta.parent_idx);
        history_parent_rank[hpos] = meta.parent_rank;
        history_action[hpos] = meta.action;
        history_valid[hpos] = 1;
    }
    if (is_central_state(src)) {
        beam_status[STATUS_FOUND] = 1;
        beam_status[STATUS_LOCAL_FOUND] = 1;
        beam_status[STATUS_FOUND_LOCAL_INDEX] = pos;
        beam_status[STATUS_FOUND_ACTION] = static_cast<int>(history_action[static_cast<int64_t>(history_depth) * n_local + pos]);
    }
}

extern "C" __global__ void kernel_increment_i32(int32_t* value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) value[0] += 1;
}

extern "C" __global__ void kernel_finalize_compaction(int32_t* beam_status, int n_local) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int sz = beam_status[STATUS_COMPACTED_SIZE];
    if (sz > n_local) sz = n_local;
    beam_status[STATUS_CURRENT_SIZE] = sz;
}

extern "C" __global__ void kernel_clear_hash_table(HashSlot* table, int hash_capacity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hash_capacity) return;
    table[idx].hash = HASH_EMPTY;
    table[idx].fingerprint = 0;
    table[idx].pool_idx = 0;
    table[idx].best_key = 0;
    table[idx].flags = 0;
    table[idx].pad0 = 0;
}

extern "C" __global__ void kernel_rebuild_hash_from_active(
    BeamMeta* __restrict__ next_meta,
    HashSlot* __restrict__ hash_table,
    const uint8_t* __restrict__ active_flags,
    int32_t* __restrict__ counters,
    int k_work,
    int hash_capacity,
    int probe_limit
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k_work) return;
    if (!active_flags[idx]) return;
    BeamMeta meta = next_meta[idx];
    uint64_t pos0 = meta.hash % static_cast<uint64_t>(hash_capacity);
    uint32_t key = make_best_key(meta.score_q, meta.fingerprint);
    for (int probe = 0; probe < probe_limit; ++probe) {
        uint64_t pos = (pos0 + probe) % static_cast<uint64_t>(hash_capacity);
        unsigned long long* key_ptr = reinterpret_cast<unsigned long long*>(&hash_table[pos].hash);
        unsigned long long old = atomicCAS(
            key_ptr,
            static_cast<unsigned long long>(HASH_EMPTY),
            static_cast<unsigned long long>(HASH_BUSY));
        if (old == HASH_EMPTY) {
            hash_table[pos].fingerprint = meta.fingerprint;
            hash_table[pos].pool_idx = static_cast<uint32_t>(idx);
            hash_table[pos].best_key = key;
            hash_table[pos].flags = 1;
            __threadfence();
            atomicExch(key_ptr, static_cast<unsigned long long>(meta.hash));
            return;
        }
        if (old == HASH_BUSY) {
            for (int spin = 0; spin < 64 && old == HASH_BUSY; ++spin) {
                old = static_cast<unsigned long long>(hash_table[pos].hash);
            }
        }
        if (old == meta.hash && hash_table[pos].fingerprint == meta.fingerprint) return;
    }
    atomicAdd(&counters[COUNTER_HASH_OVERFLOW], 1);
}

extern "C" __global__ void kernel_clear_step_state(
    int32_t* counters,
    uint32_t* local_hist,
    int32_t* threshold_cell,
    uint8_t* active_flags,
    int32_t* free_count,
    int32_t* beam_status,
    int score_bins,
    int k_work
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < COUNTER_RESERVED) counters[idx] = 0;
    if (idx < score_bins) local_hist[idx] = 0;
    if (idx < 2) threshold_cell[idx] = 0;
    if (idx < k_work) active_flags[idx] = 0;
    if (idx == 0) free_count[0] = 0;
    if (idx == 0) {
        beam_status[STATUS_COMPACTED_SIZE] = 0;
        beam_status[STATUS_MAX_ACTIVE_SCORE] = 0;
    }
}

extern "C" __global__ void kernel_check_current_solved(
    const uint8_t* __restrict__ beam_current,
    const uint8_t* __restrict__ current_active_flags,
    int32_t* __restrict__ beam_status,
    int state_size_bytes,
    int n_local
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_local) return;
    if (!current_active_flags[idx]) return;
    const uint8_t* s = beam_current + static_cast<int64_t>(idx) * state_size_bytes;
    if (is_central_state(s)) {
        if (atomicCAS(&beam_status[STATUS_FOUND], 0, 1) == 0) {
            beam_status[STATUS_LOCAL_FOUND] = 1;
            beam_status[STATUS_FOUND_LOCAL_INDEX] = idx;
            beam_status[STATUS_FOUND_ACTION] = -1;
        }
    }
}

} // namespace beam_engine

extern "C" void launch_dummy_inference(
    const uint8_t* beam_current,
    const uint8_t* current_active_flags,
    uint16_t* score_ring,
    int slot,
    int state_size_bytes,
    int fanout,
    int b_micro,
    int64_t start_state,
    int micro_size,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (micro_size + threads - 1) / threads;
    beam_engine::kernel_dummy_inference<<<blocks, threads, 0, stream>>>(
        beam_current, current_active_flags, score_ring, slot, state_size_bytes, fanout, b_micro, start_state, micro_size);
}


extern "C" void launch_copy_i16_scores_to_ring(
    const int16_t* model_scores,
    uint16_t* score_ring,
    int slot,
    int b_micro,
    int fanout,
    int micro_size,
    cudaStream_t stream
) {
    const int threads = 256;
    const int64_t total = static_cast<int64_t>(micro_size) * fanout;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    beam_engine::kernel_copy_i16_scores_to_ring<<<blocks, threads, 0, stream>>>(
        model_scores, score_ring, slot, b_micro, fanout, micro_size);
}

extern "C" void launch_quantize_f32_scores_to_ring(
    const float* model_scores,
    uint16_t* score_ring,
    int slot,
    int b_micro,
    int fanout,
    int micro_size,
    float scale,
    float bias,
    cudaStream_t stream
) {
    const int threads = 256;
    const int64_t total = static_cast<int64_t>(micro_size) * fanout;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    beam_engine::kernel_quantize_f32_scores_to_ring<<<blocks, threads, 0, stream>>>(
        model_scores, score_ring, slot, b_micro, fanout, micro_size, scale, bias);
}

extern "C" void launch_reset_net_slot(
    beam_engine::CandidateRecord* send_buckets,
    beam_engine::CandidateRecord* recv_buckets,
    int32_t* send_counts,
    int32_t* recv_counts,
    int net_slot,
    int world_size,
    int bucket_cap,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (world_size + threads - 1) / threads;
    beam_engine::kernel_reset_net_slot<<<blocks, threads, 0, stream>>>(
        send_buckets, recv_buckets, send_counts, recv_counts, net_slot, world_size, bucket_cap);
}

extern "C" void launch_process_score_slot(
    const uint8_t* beam_current,
    const uint8_t* current_active_flags,
    const uint16_t* score_ring,
    uint8_t* next_state_pool,
    beam_engine::BeamMeta* next_meta,
    beam_engine::HashSlot* hash_table,
    uint8_t* active_flags,
    int32_t* free_indices,
    int32_t* free_count,
    uint32_t* local_hist,
    int32_t* counters,
    int32_t* beam_status,
    beam_engine::CandidateRecord* send_buckets,
    int32_t* send_counts,
    const int32_t* threshold_cell,
    int slot,
    int net_slot,
    int world_size,
    int rank,
    int state_size_bytes,
    int fanout,
    int b_micro,
    int64_t start_state,
    int micro_size,
    int64_t candidate_lane_offset,
    int candidate_lanes,
    int bucket_cap,
    int hash_capacity,
    int k_work,
    int probe_limit,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = static_cast<int>((static_cast<int64_t>(candidate_lanes) + threads - 1) / threads);
    beam_engine::kernel_process_score_slot<<<blocks, threads, 0, stream>>>(
        beam_current, current_active_flags, score_ring, next_state_pool, next_meta, hash_table,
        active_flags, free_indices, free_count, local_hist, counters, beam_status, send_buckets, send_counts,
        threshold_cell, slot, net_slot, world_size, rank, state_size_bytes,
        fanout, b_micro, start_state, micro_size, candidate_lane_offset, candidate_lanes, bucket_cap,
        hash_capacity, k_work, probe_limit);
}

extern "C" void launch_ingest_recv_slot(
    const beam_engine::CandidateRecord* recv_buckets,
    const int32_t* recv_counts,
    uint8_t* next_state_pool,
    beam_engine::BeamMeta* next_meta,
    beam_engine::HashSlot* hash_table,
    uint8_t* active_flags,
    int32_t* free_indices,
    int32_t* free_count,
    uint32_t* local_hist,
    int32_t* counters,
    int32_t* beam_status,
    const int32_t* threshold_cell,
    int net_slot,
    int world_size,
    int state_size_bytes,
    int bucket_cap,
    int hash_capacity,
    int k_work,
    int probe_limit,
    cudaStream_t stream
) {
    const int threads = 256;
    const int64_t total = static_cast<int64_t>(world_size) * bucket_cap;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    beam_engine::kernel_ingest_recv_slot<<<blocks, threads, 0, stream>>>(
        recv_buckets, recv_counts, next_state_pool, next_meta, hash_table, active_flags, free_indices, free_count,
        local_hist, counters, beam_status, threshold_cell, net_slot, world_size,
        state_size_bytes, bucket_cap, hash_capacity, k_work, probe_limit);
}

extern "C" void launch_compute_threshold(const uint32_t* global_hist, int32_t* threshold_cell, int score_bins, int64_t target_global_beam, cudaStream_t stream) {
    beam_engine::kernel_compute_threshold<<<1, 1, 0, stream>>>(global_hist, threshold_cell, score_bins, target_global_beam);
}

extern "C" void launch_prune_by_threshold(
    beam_engine::BeamMeta* next_meta,
    beam_engine::HashSlot* hash_table,
    uint8_t* active_flags,
    int32_t* free_indices,
    int32_t* free_count,
    uint32_t* local_hist,
    int32_t* counters,
    const int32_t* threshold_cell,
    int k_work,
    int hash_capacity,
    int probe_limit,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (k_work + threads - 1) / threads;
    beam_engine::kernel_prune_by_threshold<<<blocks, threads, 0, stream>>>(
        next_meta, hash_table, active_flags, free_indices, free_count, local_hist, counters,
        threshold_cell, k_work, hash_capacity, probe_limit);
}

extern "C" void launch_compact_next_to_current(
    const uint8_t* next_state_pool,
    const beam_engine::BeamMeta* next_meta,
    const uint8_t* active_flags,
    uint8_t* beam_current,
    uint8_t* current_active_flags,
    int32_t* history_parent_idx,
    uint8_t* history_parent_rank,
    uint8_t* history_action,
    uint8_t* history_valid,
    const int32_t* history_depth_cell,
    int32_t* beam_status,
    int state_size_bytes,
    int k_work,
    int n_local,
    cudaStream_t stream
) {
    const int threads = 256;
    const int n = k_work > n_local ? k_work : n_local;
    const int blocks = (n + threads - 1) / threads;
    beam_engine::kernel_compact_next_to_current<<<blocks, threads, 0, stream>>>(
        next_state_pool, next_meta, active_flags, beam_current, current_active_flags,
        history_parent_idx, history_parent_rank, history_action, history_valid, history_depth_cell,
        beam_status, state_size_bytes, k_work, n_local);
    beam_engine::kernel_finalize_compaction<<<1, 1, 0, stream>>>(beam_status, n_local);
    beam_engine::kernel_increment_i32<<<1, 1, 0, stream>>>(const_cast<int32_t*>(history_depth_cell));
}

extern "C" void launch_clear_hash_table(beam_engine::HashSlot* table, int hash_capacity, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (hash_capacity + threads - 1) / threads;
    beam_engine::kernel_clear_hash_table<<<blocks, threads, 0, stream>>>(table, hash_capacity);
}

extern "C" void launch_rebuild_hash_from_active(
    beam_engine::BeamMeta* next_meta,
    beam_engine::HashSlot* hash_table,
    const uint8_t* active_flags,
    int32_t* counters,
    int k_work,
    int hash_capacity,
    int probe_limit,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (k_work + threads - 1) / threads;
    beam_engine::kernel_rebuild_hash_from_active<<<blocks, threads, 0, stream>>>(
        next_meta, hash_table, active_flags, counters, k_work, hash_capacity, probe_limit);
}

extern "C" void launch_clear_step_state(
    int32_t* counters,
    uint32_t* local_hist,
    int32_t* threshold_cell,
    uint8_t* active_flags,
    int32_t* free_count,
    int32_t* beam_status,
    int score_bins,
    int k_work,
    cudaStream_t stream
) {
    const int threads = 256;
    int n = score_bins;
    if (k_work > n) n = k_work;
    if (beam_engine::COUNTER_RESERVED > n) n = beam_engine::COUNTER_RESERVED;
    const int blocks = (n + threads - 1) / threads;
    beam_engine::kernel_clear_step_state<<<blocks, threads, 0, stream>>>(
        counters, local_hist, threshold_cell, active_flags, free_count, beam_status, score_bins, k_work);
}

extern "C" void launch_check_current_solved(
    const uint8_t* beam_current,
    const uint8_t* current_active_flags,
    int32_t* beam_status,
    int state_size_bytes,
    int n_local,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (n_local + threads - 1) / threads;
    beam_engine::kernel_check_current_solved<<<blocks, threads, 0, stream>>>(
        beam_current, current_active_flags, beam_status, state_size_bytes, n_local);
}

extern "C" void upload_action_permutation_table(const uint8_t* host_table, int actions, int state_size_bytes) {
    if (actions != beam_engine::FANOUT_FIXED || state_size_bytes != beam_engine::STATE_SIZE_BYTES_FIXED) return;
    cudaMemcpyToSymbol(beam_engine::c_action_permutation, host_table, beam_engine::FANOUT_FIXED * beam_engine::STATE_SIZE_BYTES_FIXED * sizeof(uint8_t));
    uint8_t loaded = 1;
    cudaMemcpyToSymbol(beam_engine::c_action_permutation_loaded, &loaded, sizeof(uint8_t));
}

extern "C" void upload_central_state(const uint8_t* host_state, int state_size_bytes) {
    if (state_size_bytes != beam_engine::STATE_SIZE_BYTES_FIXED) return;
    cudaMemcpyToSymbol(beam_engine::c_central_state, host_state, beam_engine::STATE_SIZE_BYTES_FIXED * sizeof(uint8_t));
    uint8_t loaded = 1;
    cudaMemcpyToSymbol(beam_engine::c_central_state_loaded, &loaded, sizeof(uint8_t));
}
