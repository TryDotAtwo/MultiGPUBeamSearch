#include "cuda_check.hpp"
#include "../cuda/static_memory.hpp"
#include "../cuda/threshold.hpp"

#include <cuda_runtime.h>
#include <nccl.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace beam;

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_nccl(ncclResult_t status, const char* message) {
    if (status != ncclSuccess) {
        throw std::runtime_error(std::string(message) + ": " + ncclGetErrorString(status));
    }
}
} // namespace

int main() {
    std::filesystem::create_directories("test_results");
    std::ofstream report("test_results/threshold_cuda_tests_2026-05-22.md");
    report << "# Threshold CUDA Tests 2026-05-22\n\n";

    BEAM_CUDA_CHECK(cudaSetDevice(0));
    constexpr std::uint32_t shard_count = 2;
    constexpr std::uint32_t stream4_batch = 4;
    constexpr std::uint32_t shard_capacity = 2 * stream4_batch;
    RuntimeConfig config;
    config.b_micro = 8;
    config.stream3_batch_candidates = 8 * static_cast<std::uint32_t>(MOVE_COUNT) * 2;
    config.stream4_batch_candidates = stream4_batch;
    config.stream4_active_sort_slots = 1;
    config.ring_count = 2;
    config.shard_count = shard_count;
    config.global_spill_capacity = 16;
    config.user_global_beam_width = 4;
    config.global_beam_width_max_safe = 16;
    config.solved_result_capacity = 4;
    const StaticMemoryPlan plan = make_static_memory_plan(config);
    StaticDeviceMemory memory;
    allocate_static_device_memory(plan, memory);
    BEAM_CUDA_CHECK(cudaMemset(memory.allocation, 0, memory.allocation_bytes));

    std::vector<CandidateMeta> survivor(shard_count * shard_capacity);
    survivor[0] = CandidateMeta{Hash128{1, 1}, 10, 5, pack_route(0, 0, 3)};
    survivor[1] = CandidateMeta{Hash128{2, 2}, 11, 7, pack_route(0, 0, 4)};
    survivor[shard_capacity] = CandidateMeta{Hash128{3, 3}, 12, 7, pack_route(0, 0, 5)};
    survivor[shard_capacity + 1] = CandidateMeta{Hash128{4, 4}, 13, 9, pack_route(0, 0, 6)};
    std::uint32_t clean_count[shard_count]{2, 2};
    std::vector<std::uint32_t> shard_hist_a(static_cast<std::uint64_t>(shard_count) * SCORE_BIN_COUNT);
    std::uint32_t hist_active[shard_count]{};
    shard_hist_a[5] = 1;
    shard_hist_a[7] = 1;
    shard_hist_a[static_cast<std::uint64_t>(SCORE_BIN_COUNT) + 7] = 1;
    shard_hist_a[static_cast<std::uint64_t>(SCORE_BIN_COUNT) + 9] = 1;

    std::uint32_t* d_all_counts = nullptr;
    BEAM_CUDA_CHECK(cudaMalloc(&d_all_counts, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemcpy(memory.streams.survivor_shard, survivor.data(), survivor.size() * sizeof(CandidateMeta), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(memory.streams.clean_count, clean_count, sizeof(clean_count), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(
        memory.streams.shard_score_hist_a,
        shard_hist_a.data(),
        shard_hist_a.size() * sizeof(std::uint32_t),
        cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(
        memory.streams.shard_score_hist_active_index,
        hist_active,
        sizeof(hist_active),
        cudaMemcpyHostToDevice));

    threshold_build_local_histogram_cuda(
        memory.streams.shard_score_hist_a,
        memory.streams.shard_score_hist_b,
        memory.streams.shard_score_hist_active_index,
        memory.streams.threshold_hist_active_snapshot,
        memory.streams.local_score_hist,
        shard_count,
        0);
    ncclUniqueId id{};
    ncclComm_t comm{};
    require_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId failed");
    require_nccl(ncclCommInitRank(&comm, 1, id, 0), "ncclCommInitRank failed");
    threshold_allreduce_histogram_nccl_cuda(memory.streams.local_score_hist, memory.streams.global_score_hist, comm, 0);
    threshold_select_cuda(memory.streams.local_score_hist, memory.streams.current_threshold, 3, 0);
    final_filter_load_balance_cuda(
        memory.streams.survivor_shard,
        memory.streams.clean_count,
        memory.final.final_keep_flags,
        memory.final.final_block_counts,
        memory.final.final_block_offsets,
        memory.final.final_candidate_buffer,
        memory.final.final_candidate_count,
        memory.final.final_request_buffer,
        memory.final.final_request_count,
        memory.final.final_send_count,
        memory.final.final_send_offset,
        7,
        0,
        1,
        0,
        3,
        static_cast<std::uint32_t>(survivor.size()),
        shard_count,
        stream4_batch,
        0);
    final_allgather_counts_nccl_cuda(memory.final.final_candidate_count, d_all_counts, comm, 0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    require_nccl(ncclCommDestroy(comm), "ncclCommDestroy failed");

    std::vector<std::uint64_t> hist(SCORE_BIN_COUNT);
    std::uint32_t threshold = 0;
    std::uint32_t final_candidate_count = 0;
    std::uint32_t request_count = 0;
    std::uint32_t send_count[2]{};
    std::uint32_t send_offset[3]{};
    std::uint32_t all_counts = 0;
    std::vector<CandidateMeta> final_candidates(survivor.size());
    std::vector<FinalRequest> requests(survivor.size());
    BEAM_CUDA_CHECK(cudaMemcpy(hist.data(), memory.streams.local_score_hist, hist.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&threshold, memory.streams.current_threshold, sizeof(threshold), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&final_candidate_count, memory.final.final_candidate_count, sizeof(final_candidate_count), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&request_count, memory.final.final_request_count, sizeof(request_count), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(send_count, memory.final.final_send_count, sizeof(send_count), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(send_offset, memory.final.final_send_offset, sizeof(send_offset), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&all_counts, d_all_counts, sizeof(all_counts), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(final_candidates.data(), memory.final.final_candidate_buffer, final_candidates.size() * sizeof(CandidateMeta), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(requests.data(), memory.final.final_request_buffer, requests.size() * sizeof(FinalRequest), cudaMemcpyDeviceToHost));

    require(hist[5] == 1 && hist[7] == 2 && hist[9] == 1, "histogram bins failed");
    require(threshold == 7, "histogram threshold failed");
    require(final_candidate_count == 3, "final candidate count failed");
    require(request_count == 3, "final request count failed");
    require(send_count[0] == 3 && send_offset[0] == 0 && send_offset[1] == 3, "final send range failed");
    require(final_candidates[0].score_key == 5 && final_candidates[2].score_key == 7, "final candidate payload failed");
    require(requests[0].parent_idx == 10 && requests[0].target_local_idx == 0 && requests[0].move == 3, "final request zero failed");
    require(requests[2].parent_idx == 12 && requests[2].target_local_idx == 2 && requests[2].move == 5, "final request two failed");
    require(all_counts == 3, "final allgather count failed");

    survivor.assign(survivor.size(), CandidateMeta{});
    survivor[0] = CandidateMeta{Hash128{11, 11}, 20, 7, pack_route(0, 0, 1)};
    survivor[1] = CandidateMeta{Hash128{12, 12}, 21, 7, pack_route(0, 0, 2)};
    survivor[2] = CandidateMeta{Hash128{13, 13}, 22, 5, pack_route(0, 0, 3)};
    survivor[3] = CandidateMeta{Hash128{14, 14}, 23, 9, pack_route(0, 0, 4)};
    clean_count[0] = 4;
    clean_count[1] = 0;
    BEAM_CUDA_CHECK(cudaMemcpy(memory.streams.survivor_shard, survivor.data(), survivor.size() * sizeof(CandidateMeta), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(memory.streams.clean_count, clean_count, sizeof(clean_count), cudaMemcpyHostToDevice));
    final_filter_load_balance_cuda(
        memory.streams.survivor_shard,
        memory.streams.clean_count,
        memory.final.final_keep_flags,
        memory.final.final_block_counts,
        memory.final.final_block_offsets,
        memory.final.final_candidate_buffer,
        memory.final.final_candidate_count,
        memory.final.final_request_buffer,
        memory.final.final_request_count,
        memory.final.final_send_count,
        memory.final.final_send_offset,
        7,
        0,
        1,
        0,
        2,
        static_cast<std::uint32_t>(survivor.size()),
        shard_count,
        stream4_batch,
        0);
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    BEAM_CUDA_CHECK(cudaMemcpy(&final_candidate_count, memory.final.final_candidate_count, sizeof(final_candidate_count), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(final_candidates.data(), memory.final.final_candidate_buffer, final_candidates.size() * sizeof(CandidateMeta), cudaMemcpyDeviceToHost));
    require(final_candidate_count == 2, "final exact score cap count failed");
    require(final_candidates[0].score_key == 5 && final_candidates[1].score_key == 7, "final exact score cap order failed");

    report << "- local_histogram=pass\n";
    report << "- threshold_select=pass\n";
    report << "- final_filter_load_balance=pass\n";
    report << "- final_exact_score_cap=pass\n";
    report << "- final_request_build=pass\n";
    report << "- nccl_histogram_allreduce=pass\n";
    report << "- nccl_count_allgather=pass\n";
    report << "\nstatus=pass\n";

    cudaFree(d_all_counts);
    free_static_device_memory(memory);
    std::cout << "threshold_cuda_tests=pass\n";
    return 0;
}
