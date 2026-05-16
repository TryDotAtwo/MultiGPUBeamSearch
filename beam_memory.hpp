#pragma once

#include <cstddef>
#include <cstdint>

#include "beam_config.hpp"
#include "beam_types.hpp"

namespace beam_v6 {

struct Region {
    size_t offset = 0;
    size_t bytes = 0;
};

struct LayoutStreams {
    Region score_ring;
    Region hash_ring;
    Region parent_base;
    Region count;
    Region stream3_key_a;
    Region stream3_key_b;
    Region stream3_val_a;
    Region stream3_val_b;
    Region unique_key;
    Region unique_val;
    Region unique_count;
    Region local_pending_buffer;
    Region remote_send_buffer;
    Region remote_recv_buffer;
    Region send_count;
    Region send_offset;
    Region recv_count;
    Region recv_offset;
    Region survivor_shard;
    Region clean_count;
    Region dirty_count;
    Region processing_flag;
    Region global_spill_buffer;
    Region local_score_hist;
    Region global_score_hist;
    Region current_threshold;
    size_t bytes = 0;
};

struct LayoutFinal {
    Region next_frontier_states_tmp;
    Region final_request_buffer;
    Region final_response_buffer;
    Region final_send_count;
    Region final_send_offset;
    Region final_recv_count;
    Region final_recv_offset;
    size_t bytes = 0;
};

struct ScratchLayouts {
    LayoutStreams streams;
    LayoutFinal final;
    size_t scratch_pool_bytes = 0;
    size_t current_frontier_bytes = 0;
    size_t solved_buffers_bytes = 0;
};

size_t align_up_size(size_t value, size_t alignment);
ScratchLayouts derive_scratch_layouts(const TargetConfig& cfg);

} // namespace beam_v6
