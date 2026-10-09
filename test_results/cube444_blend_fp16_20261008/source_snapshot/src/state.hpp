#pragma once

#include "types.hpp"

#include <vector>

namespace beam {

void clear_state_padding(State128& state);
bool padding_is_zero(const State128& state);
void final_response_set_target_local_idx(FinalResponse& response, std::uint32_t target_local_idx);
std::uint32_t final_response_get_target_local_idx(const FinalResponse& response);
State128 make_state128(const std::vector<std::uint8_t>& logical_state);
State128 apply_move(const State128& parent, const Generator& generator);
bool is_goal_state(const State128& state, const State128& central_state);

} // namespace beam

