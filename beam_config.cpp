#include "beam_config.hpp"

#include <sstream>
#include <stdexcept>

namespace beam_v6 {

int64_t round_up_i64(int64_t value, int64_t alignment) {
    if (alignment <= 0) {
        throw std::runtime_error("alignment must be positive");
    }
    if (value <= 0) {
        return 0;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

TargetConfig derive_target_config(const TargetConfig& input) {
    TargetConfig cfg = input;
    if (cfg.state_len != STATE_LEN || cfg.state_storage_len != STATE_STORAGE_LEN || cfg.state_value_pad != STATE_VALUE_PAD || cfg.move_count != MOVE_COUNT) {
        throw std::runtime_error("Target architecture v6 requires STATE_LEN=120, STATE_STORAGE_LEN=128, STATE_VALUE_PAD=128, MOVE_COUNT=24");
    }
    if (cfg.b_micro <= 0 || cfg.world_size <= 0 || cfg.shard_count <= 0 || cfg.ring_count <= 0) {
        throw std::runtime_error("Target architecture v6 requires positive B_MICRO, WORLD_SIZE, SHARD_COUNT, RING_COUNT");
    }
    if (cfg.stream3_batch_candidates <= 0) {
        cfg.stream3_batch_candidates = int64_t(cfg.b_micro) * MOVE_COUNT;
    }
    const int64_t candidates_per_slot = int64_t(cfg.b_micro) * MOVE_COUNT;
    if (cfg.stream3_batch_candidates % candidates_per_slot != 0) {
        throw std::runtime_error("STREAM3_BATCH_CANDIDATES must be divisible by B_MICRO * MOVE_COUNT");
    }
    cfg.ring_slot_count = cfg.stream3_batch_candidates / candidates_per_slot;
    if (cfg.ring_slot_count <= 0) {
        throw std::runtime_error("RING_SLOT_COUNT must be positive");
    }
    if (cfg.stream4_batch_candidates <= 0) {
        cfg.stream4_batch_candidates = cfg.stream3_batch_candidates;
    }
    if (cfg.stream4_batch_candidates_per_shard_unit <= 0) {
        cfg.stream4_batch_candidates_per_shard_unit = cfg.stream4_batch_candidates;
    }
    if (cfg.global_spill_capacity <= 0) {
        cfg.global_spill_capacity = cfg.stream4_batch_candidates;
    }
    cfg.beam_width_alignment = int64_t(cfg.world_size) * int64_t(cfg.shard_count) * cfg.stream4_batch_candidates_per_shard_unit;
    if (cfg.beam_width_alignment <= 0) {
        throw std::runtime_error("BEAM_WIDTH_ALIGNMENT must be positive");
    }
    const int64_t rounded = round_up_i64(cfg.user_global_beam_width, cfg.beam_width_alignment);
    if (cfg.global_beam_width_max_safe <= 0) {
        cfg.global_beam_width_max_safe = rounded;
    }
    cfg.global_beam_width_effective = rounded;
    if (cfg.global_beam_width_effective > cfg.global_beam_width_max_safe) {
        cfg.global_beam_width_effective = cfg.global_beam_width_max_safe;
    }
    cfg.score_scale = SCORE_SCALE;
    cfg.score_max_key = SCORE_MAX_KEY;
    cfg.score_bin_count = SCORE_BIN_COUNT;
    return cfg;
}

std::string target_config_log(const TargetConfig& cfg) {
    std::ostringstream os;
    os << "USER_GLOBAL_BEAM_WIDTH=" << cfg.user_global_beam_width << "\n"
       << "GLOBAL_BEAM_WIDTH_EFFECTIVE=" << cfg.global_beam_width_effective << "\n"
       << "GLOBAL_BEAM_WIDTH_MAX_SAFE=" << cfg.global_beam_width_max_safe << "\n"
       << "BEAM_WIDTH_ALIGNMENT=" << cfg.beam_width_alignment << "\n"
       << "SCORE_SCALE=" << cfg.score_scale << "\n"
       << "SCORE_MAX_KEY=" << cfg.score_max_key << "\n"
       << "SCORE_BIN_COUNT=" << cfg.score_bin_count;
    return os.str();
}

} // namespace beam_v6
