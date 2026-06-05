#include "state.hpp"

#include <algorithm>
#include <stdexcept>

namespace beam {

void clear_state_padding(State128& state) {
    std::fill(state.v + STATE_LEN, state.v + STATE_STORAGE_LEN, 0);
}

bool padding_is_zero(const State128& state) {
    return std::all_of(state.v + STATE_LEN, state.v + STATE_STORAGE_LEN, [](std::uint8_t value) { return value == 0; });
}

void final_response_set_target_local_idx(FinalResponse& response, std::uint32_t target_local_idx) {
    response.v[120] = static_cast<std::uint8_t>(target_local_idx);
    response.v[121] = static_cast<std::uint8_t>(target_local_idx >> 8);
    response.v[122] = static_cast<std::uint8_t>(target_local_idx >> 16);
    response.v[123] = static_cast<std::uint8_t>(target_local_idx >> 24);
}

std::uint32_t final_response_get_target_local_idx(const FinalResponse& response) {
    return static_cast<std::uint32_t>(response.v[120]) |
           (static_cast<std::uint32_t>(response.v[121]) << 8) |
           (static_cast<std::uint32_t>(response.v[122]) << 16) |
           (static_cast<std::uint32_t>(response.v[123]) << 24);
}

State128 make_state128(const std::vector<std::uint8_t>& logical_state) {
    if (logical_state.size() != STATE_LEN) {
        throw std::invalid_argument("logical_state must contain exactly STATE_LEN values");
    }
    State128 state{};
    std::copy(logical_state.begin(), logical_state.end(), state.v);
    clear_state_padding(state);
    return state;
}

State128 apply_move(const State128& parent, const Generator& generator) {
    State128 child{};
    for (std::size_t p = 0; p < STATE_STORAGE_LEN; ++p) {
        child.v[p] = parent.v[generator[p]];
    }
    return child;
}

bool is_goal_state(const State128& state, const State128& central_state) {
    for (std::size_t p = 0; p < STATE_STORAGE_LEN; ++p) {
        if (state.v[p] != central_state.v[p]) {
            return false;
        }
    }
    return true;
}

} // namespace beam
