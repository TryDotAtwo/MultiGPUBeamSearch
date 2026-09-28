#include "state.hpp"
#include "hash.hpp"
#include <cassert>
#include <numeric>
#include <vector>
#include <iostream>

int main() {
    using namespace beam;
    static_assert(sizeof(CandidateMeta) == 32);
    static_assert(STATE_STORAGE_LEN >= STATE_LEN + 4);
    std::vector<std::uint8_t> values(STATE_LEN);
    std::iota(values.begin(), values.end(), 0);
    auto state = make_state128(values);
    assert(padding_is_zero(state));
    final_response_set_target_local_idx(state, 0xfedcba98U);
    assert(final_response_get_target_local_idx(state) == 0xfedcba98U);
    for (std::size_t i = 0; i < STATE_LEN; ++i) assert(state.v[i] == i);
    clear_state_padding(state);
    Generator move{};
    std::iota(move.begin(), move.end(), 0);
    std::swap(move[0], move[STATE_LEN - 1]);
    auto child = apply_move(state, move);
    assert(child.v[0] == STATE_LEN - 1);
    assert(padding_is_zero(child));
    assert(is_goal_state(apply_move(child, move), state));
    // Check all values fit the compiled Zobrist row, including Cube555 128..149.
    assert(STATE_VALUE_PAD >= STATE_LEN);
    std::cout << "state=" << STATE_LEN << " storage=" << sizeof(state)
              << " alphabet=" << STATE_VALUE_PAD << " padding/replay=PASS\n";
}
