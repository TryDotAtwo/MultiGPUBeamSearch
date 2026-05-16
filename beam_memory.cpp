#include "beam_memory.hpp"

#include <algorithm>
#include <stdexcept>

namespace beam_v6 {

size_t align_up_size(size_t value, size_t alignment) {
    if (alignment == 0) {
        throw std::runtime_error("alignment must be positive");
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

static Region place(size_t& cursor, size_t bytes, size_t alignment) {
    cursor = align_up_size(cursor, alignment);
    Region r{cursor, bytes};
    cursor += bytes;
    return r;
}

ScratchLayouts derive_scratch_layouts(const TargetConfig& input_cfg) {
    TargetConfig cfg = derive_target_config(input_cfg);
    ScratchLayouts out;
    const int64_t ring_candidates = int64_t(cfg.ring_count) * cfg.ring_slot_count * cfg.b_micro * MOVE_COUNT;
    const int64_t ring_slots = int64_t(cfg.ring_count) * cfg.ring_slot_count;
    const int64_t stream3 = cfg.stream3_batch_candidates;
    const int64_t peers = cfg.world_size;
    const int64_t remote_cap = cfg.stream3_batch_candidates;
    const int64_t survivor_count = int64_t(cfg.shard_count) * 2LL * cfg.stream4_batch_candidates;
    const int64_t n_effective_local = (cfg.global_beam_width_effective + cfg.world_size - 1) / cfg.world_size;

    size_t s = 0;
    out.streams.score_ring = place(s, size_t(ring_candidates) * sizeof(uint32_t), alignof(uint32_t));
    out.streams.hash_ring = place(s, size_t(ring_candidates) * sizeof(Hash128), alignof(Hash128));
    out.streams.parent_base = place(s, size_t(ring_slots) * sizeof(uint64_t), alignof(uint64_t));
    out.streams.count = place(s, size_t(ring_slots) * sizeof(uint32_t), alignof(uint32_t));
    out.streams.stream3_key_a = place(s, size_t(stream3) * sizeof(Hash128), alignof(Hash128));
    out.streams.stream3_key_b = place(s, size_t(stream3) * sizeof(Hash128), alignof(Hash128));
    out.streams.stream3_val_a = place(s, size_t(stream3) * sizeof(uint64_t), alignof(uint64_t));
    out.streams.stream3_val_b = place(s, size_t(stream3) * sizeof(uint64_t), alignof(uint64_t));
    out.streams.unique_key = place(s, size_t(stream3) * sizeof(Hash128), alignof(Hash128));
    out.streams.unique_val = place(s, size_t(stream3) * sizeof(uint64_t), alignof(uint64_t));
    out.streams.unique_count = place(s, sizeof(uint32_t), alignof(uint32_t));
    out.streams.local_pending_buffer = place(s, size_t(remote_cap) * sizeof(CandidateMeta), alignof(CandidateMeta));
    out.streams.remote_send_buffer = place(s, size_t(remote_cap) * sizeof(CandidateMeta), alignof(CandidateMeta));
    out.streams.remote_recv_buffer = place(s, size_t(remote_cap) * sizeof(CandidateMeta), alignof(CandidateMeta));
    out.streams.send_count = place(s, size_t(peers) * sizeof(uint32_t), alignof(uint32_t));
    out.streams.send_offset = place(s, size_t(peers + 1) * sizeof(uint32_t), alignof(uint32_t));
    out.streams.recv_count = place(s, size_t(peers) * sizeof(uint32_t), alignof(uint32_t));
    out.streams.recv_offset = place(s, size_t(peers + 1) * sizeof(uint32_t), alignof(uint32_t));
    out.streams.survivor_shard = place(s, size_t(survivor_count) * sizeof(CandidateMeta), alignof(CandidateMeta));
    out.streams.clean_count = place(s, size_t(cfg.shard_count) * sizeof(uint32_t), alignof(uint32_t));
    out.streams.dirty_count = place(s, size_t(cfg.shard_count) * sizeof(uint32_t), alignof(uint32_t));
    out.streams.processing_flag = place(s, size_t(cfg.shard_count) * sizeof(uint8_t), alignof(uint8_t));
    out.streams.global_spill_buffer = place(s, size_t(cfg.global_spill_capacity) * sizeof(CandidateMeta), alignof(CandidateMeta));
    out.streams.local_score_hist = place(s, size_t(SCORE_BIN_COUNT) * sizeof(uint64_t), alignof(uint64_t));
    out.streams.global_score_hist = place(s, size_t(SCORE_BIN_COUNT) * sizeof(uint64_t), alignof(uint64_t));
    out.streams.current_threshold = place(s, sizeof(uint32_t), alignof(uint32_t));
    out.streams.bytes = align_up_size(s, 256);

    size_t f = 0;
    out.final.next_frontier_states_tmp = place(f, size_t(n_effective_local) * sizeof(State128), alignof(State128));
    out.final.final_request_buffer = place(f, size_t(n_effective_local) * sizeof(FinalRequest), alignof(FinalRequest));
    out.final.final_response_buffer = place(f, size_t(n_effective_local) * sizeof(FinalResponse), alignof(FinalResponse));
    out.final.final_send_count = place(f, size_t(peers) * sizeof(uint32_t), alignof(uint32_t));
    out.final.final_send_offset = place(f, size_t(peers + 1) * sizeof(uint32_t), alignof(uint32_t));
    out.final.final_recv_count = place(f, size_t(peers) * sizeof(uint32_t), alignof(uint32_t));
    out.final.final_recv_offset = place(f, size_t(peers + 1) * sizeof(uint32_t), alignof(uint32_t));
    out.final.bytes = align_up_size(f, 256);

    out.scratch_pool_bytes = std::max(out.streams.bytes, out.final.bytes);
    out.current_frontier_bytes = size_t(n_effective_local) * sizeof(State128);
    out.solved_buffers_bytes =
        sizeof(uint32_t) * 4u +
        size_t(cfg.solved_result_capacity) * (sizeof(CandidateMeta) + sizeof(uint32_t));
    return out;
}

} // namespace beam_v6
