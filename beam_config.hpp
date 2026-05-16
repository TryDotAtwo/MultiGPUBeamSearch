#pragma once

#include <cstdint>
#include <string>

#include "beam_types.hpp"

namespace beam_v6 {

struct TargetConfig {
    int state_len = STATE_LEN;
    int state_storage_len = STATE_STORAGE_LEN;
    int state_value_pad = STATE_VALUE_PAD;
    int move_count = MOVE_COUNT;

    int b_micro = 32768;
    int inference_parallelism = 1;
    int64_t stream3_batch_candidates = 0;
    int64_t stream4_batch_candidates = 0;
    int64_t stream4_batch_candidates_per_shard_unit = 0;

    int ring_count = 2;
    int64_t ring_slot_count = 0;

    int world_size = 1;
    int local_rank = 0;
    int shard_count = 1;
    int64_t global_spill_capacity = 0;
    int64_t solved_result_capacity = 256;

    int64_t user_global_beam_width = 1LL << 16;
    int64_t global_beam_width_effective = 0;
    int64_t global_beam_width_max_safe = 0;
    int64_t beam_width_alignment = 0;

    uint32_t score_scale = SCORE_SCALE;
    uint32_t score_max_key = SCORE_MAX_KEY;
    uint32_t score_bin_count = SCORE_BIN_COUNT;

    int global_threshold_update_period_shards = 16;
    bool threshold_initialized = false;
    uint32_t current_threshold = THRESHOLD_UNINITIALIZED_VALUE;
};

int64_t round_up_i64(int64_t value, int64_t alignment);
TargetConfig derive_target_config(const TargetConfig& input);
std::string target_config_log(const TargetConfig& cfg);

} // namespace beam_v6
