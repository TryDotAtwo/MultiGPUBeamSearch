#include "config.hpp"

#include <algorithm>
#include <cmath>

namespace beam {

std::uint64_t round_up(std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0) {
        return value;
    }
    const std::uint64_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

DerivedConfig derive_config(const RuntimeConfig& config) {
    DerivedConfig derived;
    derived.ring_slot_count = config.stream3_batch_candidates / (config.b_micro * static_cast<std::uint32_t>(MOVE_COUNT));
    derived.beam_width_alignment =
        static_cast<std::uint64_t>(config.world_size) *
        static_cast<std::uint64_t>(config.shard_count) *
        static_cast<std::uint64_t>(config.stream4_batch_candidates_per_shard_unit);
    derived.global_beam_width_effective =
        std::min(round_up(config.user_global_beam_width, derived.beam_width_alignment),
                 config.global_beam_width_max_safe);
    return derived;
}

std::uint32_t q_to_score_key(float q) {
    const float clamped = std::min(std::max(q, 0.0f), SCORE_MAX_Q);
    return static_cast<std::uint32_t>(std::lround(clamped * static_cast<float>(SCORE_SCALE)));
}

} // namespace beam

