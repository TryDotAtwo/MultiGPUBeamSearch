#include "cuda_check.hpp"
#include "../cuda/final_materialize.hpp"
#include "../cuda/stream1.hpp"
#include "../cuda/stream2.hpp"
#include "../cuda/stream3.hpp"
#include "../cuda/stream4.hpp"
#include "hash.hpp"
#include "state.hpp"

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

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
    std::ofstream report("test_results/stitched_cuda_tests_2026-05-20.md");
    report << "# Stitched CUDA Tests 2026-05-20\n\n";

    BEAM_CUDA_CHECK(cudaSetDevice(0));
    State128 central{};
    for (std::size_t i = 0; i < STATE_LEN; ++i) {
        central.v[i] = static_cast<std::uint8_t>(i);
    }
    clear_state_padding(central);
    State128 start = central;
    start.v[0] = 1;
    start.v[1] = 0;

    std::vector<Generator> generators(MOVE_COUNT, identity_generator());
    generators[1] = swap01_generator();
    std::vector<std::uint8_t> flat_generators(MOVE_COUNT * STATE_STORAGE_LEN);
    for (std::size_t move = 0; move < MOVE_COUNT; ++move) {
        for (std::size_t p = 0; p < STATE_STORAGE_LEN; ++p) {
            flat_generators[move * STATE_STORAGE_LEN + p] = generators[move][p];
        }
    }
    const auto zobrist = make_deterministic_zobrist(77);
    std::vector<Hash128> flat_zobrist(STATE_STORAGE_LEN * STATE_VALUE_PAD);
    for (std::size_t p = 0; p < STATE_STORAGE_LEN; ++p) {
        for (std::size_t v = 0; v < STATE_VALUE_PAD; ++v) {
            flat_zobrist[p * STATE_VALUE_PAD + v] = zobrist[p][v];
        }
    }

    State128* d_frontier = nullptr;
    State128* d_central = nullptr;
    std::uint8_t* d_generators = nullptr;
    Hash128* d_zobrist = nullptr;
    std::uint32_t* d_score = nullptr;
    Hash128* d_hash = nullptr;
    std::uint64_t* d_parent_base = nullptr;
    std::uint32_t* d_count = nullptr;
    std::uint32_t* d_flags = nullptr;
    CandidateMeta* d_solved_meta = nullptr;
    std::uint32_t* d_solved_depth = nullptr;
    Hash128* d_compact_key = nullptr;
    Hash128* d_reduce_key = nullptr;
    Hash128* d_unique_key = nullptr;
    std::uint64_t* d_compact_val = nullptr;
    std::uint64_t* d_reduce_val = nullptr;
    std::uint64_t* d_unique_val = nullptr;
    std::uint32_t* d_stream3_keep_flags = nullptr;
    std::uint32_t* d_stream3_block_counts = nullptr;
    std::uint32_t* d_stream3_block_offsets = nullptr;
    std::uint32_t* d_compact_count = nullptr;
    CandidateMeta* d_survivors = nullptr;
    Hash128* d_stream4_sort_key = nullptr;
    Hash128* d_stream4_reduce_key = nullptr;
    CandidateMeta* d_stream4_sort_value = nullptr;
    CandidateMeta* d_stream4_reduce_value = nullptr;
    std::uint32_t* d_stream4_score_key_a = nullptr;
    std::uint32_t* d_stream4_score_key_b = nullptr;
    std::uint64_t* d_stream4_score_count_a = nullptr;
    std::uint64_t* d_stream4_score_count_b = nullptr;
    std::uint32_t* d_stream4_keep_flags = nullptr;
    std::uint32_t* d_stream4_block_counts = nullptr;
    std::uint32_t* d_stream4_block_offsets = nullptr;
    std::uint32_t* d_survivor_count = nullptr;
    std::uint32_t* d_stream4_dirty_count = nullptr;
    std::uint32_t* d_stream4_processing_flag = nullptr;
    std::uint32_t* d_stream4_hist_a = nullptr;
    std::uint32_t* d_stream4_hist_b = nullptr;
    std::uint32_t* d_stream4_hist_active = nullptr;
    FinalRequest* d_requests = nullptr;
    FinalResponse* d_responses = nullptr;
    State128* d_next = nullptr;
    void* d_stream3_cub_temp = nullptr;
    void* d_stream4_cub_temp = nullptr;
    constexpr std::size_t stream3_cub_temp_bytes = 8ULL * 1024ULL * 1024ULL;
    constexpr std::size_t stream4_cub_temp_bytes = 8ULL * 1024ULL * 1024ULL;

    BEAM_CUDA_CHECK(cudaMalloc(&d_frontier, sizeof(State128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_central, sizeof(State128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_generators, flat_generators.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&d_zobrist, flat_zobrist.size() * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_score, MOVE_COUNT * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_hash, MOVE_COUNT * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_parent_base, sizeof(std::uint64_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_count, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_flags, 4 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_solved_meta, 4 * sizeof(CandidateMeta)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_solved_depth, 4 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_compact_key, MOVE_COUNT * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_reduce_key, MOVE_COUNT * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_unique_key, MOVE_COUNT * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_compact_val, MOVE_COUNT * sizeof(std::uint64_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_reduce_val, MOVE_COUNT * sizeof(std::uint64_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_unique_val, MOVE_COUNT * sizeof(std::uint64_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream3_keep_flags, MOVE_COUNT * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream3_block_counts, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream3_block_offsets, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_compact_count, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_survivors, 2 * sizeof(CandidateMeta)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream4_sort_key, 2 * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream4_reduce_key, 2 * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream4_sort_value, 2 * sizeof(CandidateMeta)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream4_reduce_value, 2 * sizeof(CandidateMeta)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream4_score_key_a, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream4_score_key_b, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream4_score_count_a, 2 * sizeof(std::uint64_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream4_score_count_b, 2 * sizeof(std::uint64_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream4_keep_flags, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream4_block_counts, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream4_block_offsets, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_survivor_count, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream4_dirty_count, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream4_processing_flag, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream4_hist_a, SCORE_BIN_COUNT * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream4_hist_b, SCORE_BIN_COUNT * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream4_hist_active, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_requests, sizeof(FinalRequest)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_responses, sizeof(FinalResponse)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_next, sizeof(State128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream3_cub_temp, stream3_cub_temp_bytes));
    BEAM_CUDA_CHECK(cudaMalloc(&d_stream4_cub_temp, stream4_cub_temp_bytes));
    BEAM_CUDA_CHECK(cudaMemset(d_stream4_hist_a, 0, SCORE_BIN_COUNT * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_stream4_hist_b, 0, SCORE_BIN_COUNT * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_stream4_hist_active, 0, sizeof(std::uint32_t)));

    const std::uint64_t parent_base = 0;
    const std::uint32_t count = 1;
    BEAM_CUDA_CHECK(cudaMemcpy(d_frontier, &start, sizeof(State128), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_central, &central, sizeof(State128), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_generators, flat_generators.data(), flat_generators.size(), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_zobrist, flat_zobrist.data(), flat_zobrist.size() * sizeof(Hash128), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_parent_base, &parent_base, sizeof(parent_base), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_count, &count, sizeof(count), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemset(d_flags, 0, 4 * sizeof(std::uint32_t)));

    Stream2SolvedBuffers solved{d_flags, d_flags + 1, d_flags + 2, d_flags + 3, d_solved_meta, d_solved_depth, 4};
    stream1_score_contract_cuda(d_frontier, d_parent_base, d_count, d_score, 0, 0, 1, 0);
    stream2_hash_goal_cuda(d_frontier, d_parent_base, d_count, d_generators, d_central, d_zobrist, d_hash, 0, 0, 1, 1, 0, solved, 0);
    BEAM_CUDA_CHECK(cudaMemset(d_compact_count, 0, sizeof(std::uint32_t)));
    stream3_pack_threshold_cuda(
        d_score,
        d_hash,
        d_parent_base,
        d_count,
        d_compact_key,
        d_compact_val,
        d_reduce_key,
        d_reduce_val,
        d_unique_key,
        d_unique_val,
        d_stream3_keep_flags,
        d_stream3_block_counts,
        d_stream3_block_offsets,
        d_compact_count,
        d_stream3_cub_temp,
        stream3_cub_temp_bytes,
        SCORE_MAX_KEY,
        1,
        MOVE_COUNT,
        0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());

    std::uint32_t flags[4]{};
    std::uint32_t compact_count = 0;
    BEAM_CUDA_CHECK(cudaMemcpy(flags, d_flags, sizeof(flags), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&compact_count, d_compact_count, sizeof(compact_count), cudaMemcpyDeviceToHost));
    require(flags[0] == 1 && flags[1] == 1, "stitched solved path failed");
    require(compact_count == 2, "stitched stream3 dedup count failed");

    CandidateMeta host_candidates[2]{
        CandidateMeta{Hash128{1, 2}, 0, 5, pack_route(0, 0, 1)},
        CandidateMeta{Hash128{1, 2}, 0, 4, pack_route(0, 0, 1)},
    };
    const std::uint32_t stream4_clean_count = 0;
    const std::uint32_t stream4_dirty_count = 2;
    const std::uint32_t stream4_processing_flag = 1;
    BEAM_CUDA_CHECK(cudaMemcpy(d_survivors, host_candidates, sizeof(host_candidates), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_survivor_count, &stream4_clean_count, sizeof(stream4_clean_count), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_stream4_dirty_count, &stream4_dirty_count, sizeof(stream4_dirty_count), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_stream4_processing_flag, &stream4_processing_flag, sizeof(stream4_processing_flag), cudaMemcpyHostToDevice));
    stream4_shard_job_cuda(
        d_survivors,
        d_survivor_count,
        d_stream4_dirty_count,
        d_stream4_processing_flag,
        10,
        2,
        d_stream4_sort_key,
        d_stream4_reduce_key,
        d_stream4_sort_value,
        d_stream4_reduce_value,
        d_stream4_score_key_a,
        d_stream4_score_key_b,
        d_stream4_score_count_a,
        d_stream4_score_count_b,
        d_stream4_keep_flags,
        d_stream4_block_counts,
        d_stream4_block_offsets,
        d_compact_count,
        d_stream4_hist_a,
        d_stream4_hist_b,
        d_stream4_hist_active,
        d_stream4_cub_temp,
        stream4_cub_temp_bytes,
        0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    std::uint32_t survivor_count = 0;
    BEAM_CUDA_CHECK(cudaMemcpy(&survivor_count, d_survivor_count, sizeof(survivor_count), cudaMemcpyDeviceToHost));
    require(survivor_count == 1, "stitched stream4 dedup failed");

    FinalRequest request{};
    request.parent_idx = 0;
    request.target_local_idx = 0;
    request.return_rank = 0;
    request.move = 1;
    BEAM_CUDA_CHECK(cudaMemcpy(d_requests, &request, sizeof(request), cudaMemcpyHostToDevice));
    final_materialize_cuda(d_frontier, d_requests, d_generators, d_responses, d_next, 1, 0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    State128 next{};
    BEAM_CUDA_CHECK(cudaMemcpy(&next, d_next, sizeof(State128), cudaMemcpyDeviceToHost));
    require(next.v[0] == 0 && next.v[1] == 1 && padding_is_zero(next), "stitched final materialize failed");

    report << "- stream1_score=pass\n";
    report << "- stream2_goal=pass\n";
    report << "- stream3_threshold_sort_dedup=pass\n";
    report << "- stream4_dedup=pass\n";
    report << "- final_materialize=pass\n";
    report << "\nstatus=pass\n";

    cudaFree(d_frontier); cudaFree(d_central); cudaFree(d_generators); cudaFree(d_zobrist);
    cudaFree(d_score); cudaFree(d_hash); cudaFree(d_parent_base); cudaFree(d_count); cudaFree(d_flags);
    cudaFree(d_solved_meta); cudaFree(d_solved_depth); cudaFree(d_compact_key); cudaFree(d_reduce_key); cudaFree(d_unique_key); cudaFree(d_compact_val);
    cudaFree(d_reduce_val); cudaFree(d_unique_val); cudaFree(d_stream3_keep_flags); cudaFree(d_stream3_block_counts); cudaFree(d_stream3_block_offsets);
    cudaFree(d_compact_count); cudaFree(d_survivors); cudaFree(d_stream4_sort_key); cudaFree(d_stream4_reduce_key);
    cudaFree(d_stream4_sort_value); cudaFree(d_stream4_reduce_value); cudaFree(d_stream4_score_key_a);
    cudaFree(d_stream4_score_key_b); cudaFree(d_stream4_score_count_a); cudaFree(d_stream4_score_count_b); cudaFree(d_stream4_keep_flags);
    cudaFree(d_stream4_block_counts); cudaFree(d_stream4_block_offsets); cudaFree(d_survivor_count);
    cudaFree(d_stream4_dirty_count); cudaFree(d_stream4_processing_flag); cudaFree(d_stream4_hist_a);
    cudaFree(d_stream4_hist_b); cudaFree(d_stream4_hist_active);
    cudaFree(d_requests); cudaFree(d_responses); cudaFree(d_next); cudaFree(d_stream3_cub_temp); cudaFree(d_stream4_cub_temp);
    std::cout << "stitched_cuda_tests=pass\n";
    return 0;
}
