#include "cuda_check.hpp"
#include "../cuda/static_memory.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
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

std::uintptr_t addr(const void* ptr) {
    return reinterpret_cast<std::uintptr_t>(ptr);
}
} // namespace

int main() {
    std::filesystem::create_directories("test_results");
    std::ofstream report("test_results/static_memory_cuda_tests_2026-05-22.md");
    report << "# Static Memory CUDA Tests 2026-05-22\n\n";

    BEAM_CUDA_CHECK(cudaSetDevice(0));
    RuntimeConfig config;
    config.b_micro = 8;
    config.stream3_batch_candidates = 8 * static_cast<std::uint32_t>(MOVE_COUNT) * 2;
    config.stream4_batch_candidates = 64;
    config.stream4_active_sort_slots = 2;
    config.ring_count = 2;
    config.shard_count = 4;
    config.shard_capacity_candidates = 128;
    config.global_spill_capacity = 128;
    config.stream4_batch_alignment = 1;
    config.user_global_beam_width = 256;
    config.solved_result_capacity = 16;
    const StaticMemoryPlan plan = make_static_memory_plan(config);
    require(plan.derived.ring_slot_count == 2, "ring slot count must follow stream3 batch formula");
    require(plan.scratch_pool_bytes >= plan.layout_streams_bytes, "scratch pool must fit streams layout");
    require(plan.scratch_pool_bytes >= plan.layout_final_bytes, "scratch pool must fit final layout");

    StaticDeviceMemory memory;
    allocate_static_device_memory(plan, memory);
    require(memory.allocation != nullptr, "static allocation must exist");
    require(memory.current_frontier_states != nullptr, "current frontier must exist outside scratch");
    require(memory.scratch_pool != nullptr, "scratch pool must exist");
    require(memory.final.next_frontier_states_tmp == memory.scratch_pool, "final layout must overlay scratch pool start");
    require(addr(memory.current_frontier_states) < addr(memory.scratch_pool), "current frontier must be before scratch pool");
    require(addr(memory.solved_flag) < addr(memory.scratch_pool), "solved buffers must be outside scratch pool");
    require(memory.current_depth != nullptr && addr(memory.current_depth) < addr(memory.scratch_pool), "current depth must be outside scratch pool");
    require(addr(memory.streams.score_ring) == addr(memory.scratch_pool), "streams layout must start at scratch pool");
    require(addr(memory.streams.hash_ring) % alignof(Hash128) == 0, "hash ring alignment failed");
    require(memory.streams.stream3_keep_flags != nullptr, "stream3 keep flags missing");
    require(memory.streams.stream3_block_counts != nullptr, "stream3 block counts missing");
    require(memory.streams.stream3_block_offsets != nullptr, "stream3 block offsets missing");
    require(memory.streams.stream3_owner != nullptr, "stream3 owner scratch missing");
    require(memory.streams.stream3_shard_counts != nullptr, "stream3 shard counts missing");
    require(memory.streams.stream3_shard_offsets != nullptr, "stream3 shard offsets missing");
    require(memory.streams.stream3_spill_counts != nullptr, "stream3 spill counts missing");
    require(memory.streams.stream3_spill_offsets != nullptr, "stream3 spill offsets missing");
    require(memory.streams.stream3_ready_flag != nullptr, "stream3 ready flag missing");
    require(memory.streams.stream3_ready_shard_list != nullptr, "stream3 ready shard list missing");
    require(memory.streams.stream3_ready_count != nullptr, "stream3 ready count missing");
    require(memory.streams.stream3_partition_key_a != nullptr, "stream3 partition key a missing");
    require(memory.streams.stream3_partition_key_b != nullptr, "stream3 partition key b missing");
    require(memory.streams.stream3_partition_val_a != nullptr, "stream3 partition value a missing");
    require(memory.streams.stream3_partition_val_b != nullptr, "stream3 partition value b missing");
    require(memory.streams.stream3_partition_unique_shard != nullptr, "stream3 partition unique shard missing");
    require(memory.streams.stream3_partition_unique_counts != nullptr, "stream3 partition unique counts missing");
    require(memory.streams.stream3_partition_unique_count != nullptr, "stream3 partition unique count missing");
    require(addr(memory.streams.stream3_partition_val_a) % alignof(CandidateMeta) == 0, "stream3 partition value alignment failed");
    require(memory.streams.stream3_cub_temp != nullptr, "stream3 CUB fixed temp storage missing");
    require(memory.streams.stream3_cub_temp_bytes > 0, "stream3 CUB fixed temp storage bytes missing");
    require(memory.streams.local_pending_buffer != nullptr, "stream3 local pending buffer missing");
    require(memory.streams.local_pending_count != nullptr, "stream3 local pending count missing");
    require(memory.streams.remote_send_buffer != nullptr, "stream3 remote send buffer missing");
    require(memory.streams.remote_recv_buffer != nullptr, "stream3 remote recv buffer missing");
    require(memory.streams.send_count != nullptr, "stream3 send count missing");
    require(memory.streams.send_offset != nullptr, "stream3 send offset missing");
    require(memory.streams.recv_count != nullptr, "stream3 recv count missing");
    require(memory.streams.recv_offset != nullptr, "stream3 recv offset missing");
    require(memory.streams.global_spill_buffer_a != nullptr, "global spill buffer a missing");
    require(memory.streams.global_spill_buffer_b != nullptr, "global spill buffer b missing");
    require(memory.streams.global_spill_count != nullptr, "global spill count missing");
    require(memory.streams.global_spill_active_index != nullptr, "global spill active index missing");
    require(memory.streams.shard_score_hist_a != nullptr, "shard score histogram a missing");
    require(memory.streams.shard_score_hist_b != nullptr, "shard score histogram b missing");
    require(memory.streams.shard_score_hist_active_index != nullptr, "shard score histogram active index missing");
    require(memory.streams.threshold_hist_active_snapshot != nullptr, "threshold histogram active snapshot missing");
    require(memory.streams.threshold_initialized != nullptr, "threshold initialized flag missing");
    require(addr(memory.streams.survivor_shard) % alignof(CandidateMeta) == 0, "candidate alignment failed");
    require(addr(memory.streams.stream4_key_a) % alignof(Hash128) == 0, "stream4 sort key alignment failed");
    require(addr(memory.streams.stream4_val_a) % alignof(CandidateMeta) == 0, "stream4 sort value alignment failed");
    require(memory.streams.stream4_key_b != nullptr, "stream4 reduce key missing");
    require(memory.streams.stream4_val_b != nullptr, "stream4 reduce value missing");
    require(memory.streams.stream4_score_key_a != nullptr, "stream4 score histogram key a missing");
    require(memory.streams.stream4_score_key_b != nullptr, "stream4 score histogram key b missing");
    require(memory.streams.stream4_score_count_a != nullptr, "stream4 score histogram count a missing");
    require(memory.streams.stream4_score_count_b != nullptr, "stream4 score histogram count b missing");
    require(memory.streams.stream4_keep_flags != nullptr, "stream4 keep flags missing");
    require(memory.streams.stream4_block_counts != nullptr, "stream4 block counts missing");
    require(memory.streams.stream4_block_offsets != nullptr, "stream4 block offsets missing");
    require(memory.streams.stream4_count != nullptr, "stream4 scratch count missing");
    require(memory.streams.stream4_cub_temp != nullptr, "stream4 CUB fixed temp storage missing");
    require(memory.streams.stream4_cub_temp_bytes > 0, "stream4 CUB fixed temp storage bytes missing");
    require(memory.final.final_candidate_buffer != nullptr, "final candidate buffer missing");
    require(memory.final.final_candidate_count != nullptr, "final candidate count missing");
    require(memory.final.final_keep_flags != nullptr, "final keep flags missing");
    require(memory.final.final_block_counts != nullptr, "final block counts missing");
    require(memory.final.final_block_offsets != nullptr, "final block offsets missing");
    require(memory.final.final_request_buffer != nullptr, "single rank final request buffer missing");
    require(addr(memory.final.final_request_buffer) % alignof(FinalRequest) == 0, "final request alignment failed");
    require(memory.final.final_request_count != nullptr, "final request count missing");
    require(memory.final.final_validation_error != nullptr, "final validation error storage missing");
    require(
        addr(memory.final.final_validation_error) % alignof(FinalRequestValidationError) == 0,
        "final validation error alignment failed");
    BEAM_CUDA_CHECK(cudaMemset(memory.allocation, 0, memory.allocation_bytes));
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    free_static_device_memory(memory);
    require(memory.allocation == nullptr, "free must reset memory struct");

    RuntimeConfig multi_config = config;
    multi_config.world_size = 2;
    multi_config.local_rank = 0;
    multi_config.user_global_beam_width = 512;
    const StaticMemoryPlan multi_plan = make_static_memory_plan(multi_config);
    require(multi_plan.final_materialize_slot_count == 3, "final materialize must use 3 slots");
    require(multi_plan.final_materialize_chunk_capacity > 0, "final materialize chunk capacity missing");
    require(
        multi_plan.final_materialize_exchange_capacity ==
            multi_plan.final_materialize_chunk_capacity * multi_config.world_size,
        "final materialize exchange capacity mismatch");
    require(
        multi_plan.final_selected_candidate_capacity ==
            std::min<std::uint64_t>(multi_plan.derived.global_beam_width_effective, multi_plan.survivor_count),
        "final selected candidate capacity must be min(global beam, survivors)");
    require(multi_plan.final_materialize_cub_temp_bytes > 0, "final materialize CUB temp missing");

    StaticDeviceMemory multi_memory;
    allocate_static_device_memory(multi_plan, multi_memory);
    require(multi_memory.final.final_selected_buffer != nullptr, "layout3 selected buffer missing");
    require(multi_memory.final.final_request_buffer == nullptr, "multi-rank full final request buffer must be omitted");
    require(multi_memory.final.final_response_buffer == nullptr, "multi-rank full final response buffer must be omitted");
    require(multi_memory.final.final_validation_error != nullptr, "layout3 validation error storage missing");
    require(multi_memory.final.final_mat_key_a != nullptr, "layout3 key a missing");
    require(multi_memory.final.final_mat_key_b != nullptr, "layout3 key b missing");
    require(multi_memory.final.final_mat_request_a != nullptr, "layout3 request a missing");
    require(multi_memory.final.final_mat_request_b != nullptr, "layout3 request b missing");
    require(multi_memory.final.final_mat_request_recv != nullptr, "layout3 request recv missing");
    require(multi_memory.final.final_mat_response_send != nullptr, "layout3 response send missing");
    require(multi_memory.final.final_mat_response_recv != nullptr, "layout3 response recv missing");
    require(multi_memory.final.final_mat_history_send != nullptr, "layout3 history send missing");
    require(multi_memory.final.final_mat_history_recv != nullptr, "layout3 history recv missing");
    require(multi_memory.final.final_mat_cub_temp != nullptr, "layout3 CUB temp missing");
    require(
        multi_memory.final.final_mat_cub_temp_bytes == multi_plan.final_materialize_cub_temp_bytes,
        "layout3 CUB temp size mismatch");
    require(
        addr(multi_memory.final.final_selected_buffer) % alignof(CandidateMeta) == 0,
        "layout3 selected buffer alignment failed");
    require(
        addr(multi_memory.final.final_validation_error) % alignof(FinalRequestValidationError) == 0,
        "layout3 validation error alignment failed");
    require(addr(multi_memory.final.final_mat_key_a) % alignof(std::uint32_t) == 0, "layout3 key a alignment failed");
    require(addr(multi_memory.final.final_mat_key_b) % alignof(std::uint32_t) == 0, "layout3 key b alignment failed");
    require(addr(multi_memory.final.final_mat_request_a) % alignof(FinalRequest) == 0, "layout3 request a alignment failed");
    require(addr(multi_memory.final.final_mat_request_b) % alignof(FinalRequest) == 0, "layout3 request b alignment failed");
    require(
        addr(multi_memory.final.final_mat_request_recv) % alignof(FinalRequest) == 0,
        "layout3 request recv alignment failed");
    require(
        addr(multi_memory.final.final_mat_response_send) % alignof(CandidateMeta) == 0,
        "layout3 response send alignment failed");
    require(
        addr(multi_memory.final.final_mat_response_recv) % alignof(CandidateMeta) == 0,
        "layout3 response recv alignment failed");
    require(
        addr(multi_memory.final.final_mat_history_send) % alignof(FinalHistoryRecord) == 0,
        "layout3 history send alignment failed");
    require(
        addr(multi_memory.final.final_mat_history_recv) % alignof(FinalHistoryRecord) == 0,
        "layout3 history recv alignment failed");
    require(addr(multi_memory.final.final_mat_cub_temp) % 256 == 0, "layout3 CUB temp alignment failed");
    BEAM_CUDA_CHECK(cudaMemset(multi_memory.allocation, 0, multi_memory.allocation_bytes));
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    free_static_device_memory(multi_memory);
    require(multi_memory.allocation == nullptr, "free must reset multi-rank memory struct");

    report << "- pre_start_single_allocation=pass\n";
    report << "- current_frontier_outside_scratch=pass\n";
    report << "- solved_buffers_outside_scratch=pass\n";
    report << "- current_depth_outside_scratch=pass\n";
    report << "- streams_final_overlay=pass\n";
    report << "- alignment=pass\n";
    report << "- stream4_active_sort_slots_scratch=pass\n";
    report << "- stream4_cub_fixed_temp=pass\n";
    report << "- stream3_scan_scratch=pass\n";
    report << "- stream3_cub_fixed_temp=pass\n";
    report << "- stream3_split_buffers=pass\n";
    report << "- global_spill_pingpong=pass\n";
    report << "- threshold_initialized=pass\n";
    report << "- threshold_fixed_shard_histogram=pass\n";
    report << "- stream4_score_histogram_scratch=pass\n";
    report << "- final_candidate_buffers=pass\n";
    report << "- final_filter_scan_scratch=pass\n";
    report << "- final_validation_error_static_storage=pass\n";
    report << "- final_materialize_layout3_slots=pass\n";
    report << "- final_materialize_layout3_alignment=pass\n";
    report << "\nstatus=pass\n";
    std::cout << "static_memory_cuda_tests=pass\n";
    return 0;
}
