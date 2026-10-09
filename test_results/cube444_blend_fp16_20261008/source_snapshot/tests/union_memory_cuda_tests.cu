#include <cuda_runtime.h>
#include "../cuda/static_memory.hpp"
#include "../cuda/stream4.hpp"
#include <vector>
#include <iostream>
#include <stdexcept>
using namespace beam;
int main() {
    try {
        RuntimeConfig config;
        config.b_micro = 1;
        config.stream3_batch_candidates = 2 * MOVE_COUNT;
        config.stream4_batch_candidates = 4;
        config.stream4_active_sort_slots = 1;
        config.shard_count = 1;
        config.shard_buffer_count = 2;
        config.shard_capacity_candidates = 4096;
        config.stream4_batch_alignment = 1;
        config.user_global_beam_width = 2;
        config.global_spill_capacity = 0;
        const auto plan = make_static_memory_plan(config);
        // Live union arrays alone: two Hash128, two CandidateMeta, two score
        // keys, two uint64 counts and flags. CUB temp/metadata add more.
        const std::size_t array_floor = 2ULL * config.shard_capacity_candidates *
            (2 * sizeof(Hash128) + 2 * sizeof(CandidateMeta) + 3 * sizeof(std::uint32_t) +
             2 * sizeof(std::uint64_t));
        std::cout << "union_array_floor=" << array_floor
                  << " phase_budget=" << plan.layout_final_budget_bytes << '\n';
        if (plan.layout_final_budget_bytes < array_floor) {
            std::cerr << "FAIL: final-phase budget does not reserve logical union scratch\n";
            return 1;
        }
        StaticDeviceMemory memory;
        allocate_static_device_memory(plan, memory);
        const auto begin = reinterpret_cast<std::uintptr_t>(memory.scratch_pool);
        const auto end = begin + plan.layout_union_bytes;
        for (const void* p : {static_cast<void*>(memory.streams.survivor_shard),
                              static_cast<void*>(memory.streams.clean_count),
                              static_cast<void*>(memory.streams.shard_score_hist_a),
                              static_cast<void*>(memory.streams.shard_score_hist_b)}) {
            if (reinterpret_cast<std::uintptr_t>(p) < end) throw std::runtime_error("union overlaps persistent state");
        }
        const auto temp_end = reinterpret_cast<std::uintptr_t>(memory.final_union.cub_temp) + plan.union_cub_temp_bytes;
        if (temp_end > end || !memory.final_union.count) throw std::runtime_error("bad union scratch allocation");
        auto check = [](cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); };
        check(cudaMemset(memory.allocation, 0, memory.allocation_bytes));
        // Exercise the primitive with the exact planner-provided pointers/temp,
        // not a separately overprovisioned test arena.
        std::vector<CandidateMeta> input(2ULL * config.shard_capacity_candidates);
        input[0] = CandidateMeta{Hash128{1, 0}, 9, 3, 1};
        input[1] = CandidateMeta{Hash128{2, 0}, 8, 7, 2};
        input[config.shard_capacity_candidates] = CandidateMeta{Hash128{1, 0}, 4, 1, 3};
        const std::uint32_t counts[2]{2, 1};
        check(cudaMemcpy(memory.streams.survivor_shard, input.data(), input.size() * sizeof(input[0]), cudaMemcpyHostToDevice));
        check(cudaMemcpy(memory.streams.clean_count, counts, sizeof(counts), cudaMemcpyHostToDevice));
        auto& u = memory.final_union;
        stream4_finalize_logical_shard_union_cuda(memory.streams.survivor_shard, memory.streams.clean_count,
            u.dirty, u.processing, SCORE_MAX_KEY, config.shard_capacity_candidates,
            u.key_a, u.key_b, u.value_a, u.value_b, u.score_a, u.score_b, u.score_count_a, u.score_count_b,
            u.keep, u.blocks, u.offsets, u.count, memory.streams.shard_score_hist_a,
            memory.streams.shard_score_hist_b, memory.streams.shard_score_hist_active_index,
            u.cub_temp, plan.union_cub_temp_bytes, 0);
        check(cudaDeviceSynchronize());
        std::uint32_t final_counts[2]{};
        CandidateMeta result[2]{};
        check(cudaMemcpy(final_counts, memory.streams.clean_count, sizeof(final_counts), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(result, memory.streams.survivor_shard, sizeof(result), cudaMemcpyDeviceToHost));
        if (final_counts[0] != 2 || final_counts[1] != 0 || result[0].hash.lo != 1 ||
            result[0].score_key != 1 || result[0].parent_idx != 4 || result[0].route_packed != 3 ||
            result[1].hash.lo != 2 || result[1].score_key != 7)
            throw std::runtime_error("planned arena union corrupts survivors or counts");
        free_static_device_memory(memory);
        std::cout << "PASS: logical union scratch budget\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 2; }
}
