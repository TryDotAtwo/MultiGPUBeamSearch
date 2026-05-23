#include "config.hpp"
#include "frontier_cpu.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace beam;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Generator identity_generator() {
    Generator generator{};
    for (std::size_t i = 0; i < STATE_STORAGE_LEN; ++i) {
        generator[i] = static_cast<std::uint8_t>(i);
    }
    return generator;
}

Generator swap01_generator() {
    Generator generator = identity_generator();
    generator[0] = 1;
    generator[1] = 0;
    return generator;
}

} // namespace

int main() {
    std::filesystem::create_directories("test_results");
    std::ofstream report("test_results/contract_tests_2026-05-20.md");
    report << "# Contract Tests 2026-05-20\n\n";

    RuntimeConfig config;
    config.user_global_beam_width = 1000;
    config.world_size = 2;
    config.shard_count = 8;
    config.stream4_batch_candidates_per_shard_unit = 16;
    config.global_beam_width_max_safe = 4096;
    const DerivedConfig derived = derive_config(config);
    require(derived.ring_slot_count == config.stream3_batch_candidates / (config.b_micro * MOVE_COUNT), "ring_slot_count formula failed");
    require(derived.beam_width_alignment == 256, "beam_width_alignment formula failed");
    require(derived.global_beam_width_effective == 1024, "effective beam width formula failed");
    report << "- config_derivation=pass\n";

    require(q_to_score_key(-5.0f) == 0, "negative score clamp failed");
    require(q_to_score_key(300.5f) == SCORE_MAX_KEY, "max score clamp failed");
    require(q_to_score_key(1.5f) == 192, "score rounding failed");
    report << "- score_quantization=pass\n";

    State128 state = make_state128(std::vector<std::uint8_t>(STATE_LEN, 7));
    require(padding_is_zero(state), "state padding must be zero");
    final_response_set_target_local_idx(state, 0x44332211);
    require(final_response_get_target_local_idx(state) == 0x44332211, "final response byte-pack failed");
    clear_state_padding(state);
    require(padding_is_zero(state), "padding cleanup failed");
    report << "- final_response_padding=pass\n";

    const auto zobrist = make_deterministic_zobrist(42);
    Hash128 h1 = hash_state(state, zobrist);
    state.v[120] = 99;
    state.v[127] = 123;
    Hash128 h2 = hash_state(state, zobrist);
    require(h1 == h2, "padding must not influence hash");
    clear_state_padding(state);
    report << "- hash_padding_invariant=pass\n";

    CandidateMeta a{Hash128{1, 2}, 5, 10, pack_route(0, 0, 3)};
    CandidateMeta b{Hash128{1, 2}, 4, 10, pack_route(0, 0, 4)};
    CandidateMeta c{Hash128{3, 2}, 6, 9, pack_route(0, 0, 5)};
    auto deduped = stream4_threshold_sort_dedup({a, b, c}, 10);
    require(deduped.size() == 2, "stream4 dedup size failed");
    require(deduped[0].parent_idx == 4, "stream4 deterministic tie-break failed");
    report << "- stream4_threshold_sort_dedup=pass\n";

    std::vector<Stream3CandidateInput> s3input{
        {Hash128{7, 1}, 5, 10, 100, 1},
        {Hash128{7, 1}, 3, 99, 101, 2},
        {Hash128{8, 1}, 11, 12, 102, 3},
    };
    auto split = stream3_threshold_dedup_split(s3input, 10, 0, 2);
    require(split.local_pending.size() + split.remote_send.size() == 1, "stream3 threshold/dedup failed");
    const CandidateMeta s3meta = split.local_pending.empty() ? split.remote_send.front() : split.local_pending.front();
    require(s3meta.score_key == 3, "stream3 min stream3_val score failed");
    require(s3meta.parent_idx == 101, "stream3 payload restore failed");
    report << "- stream3_threshold_dedup_split=pass\n";

    State128 central{};
    for (std::size_t i = 0; i < STATE_LEN; ++i) {
        central.v[i] = static_cast<std::uint8_t>(i);
    }
    clear_state_padding(central);
    State128 start = central;
    start.v[0] = 1;
    start.v[1] = 0;
    std::vector<Generator> generators{identity_generator(), swap01_generator()};
    CpuDepthResult depth = expand_depth_cpu_reference({start}, generators, central, zobrist, UINT32_THRESHOLD_MAX, 8);
    require(depth.solved, "cpu reference must detect solved child");
    require(!depth.next_frontier.empty(), "cpu reference next frontier empty");
    require(padding_is_zero(depth.next_frontier.front()), "cpu reference persistent padding failed");
    report << "- cpu_reference_depth=pass\n";

    report << "\nstatus=pass\n";
    std::cout << "contract_tests=pass\n";
    return 0;
}
