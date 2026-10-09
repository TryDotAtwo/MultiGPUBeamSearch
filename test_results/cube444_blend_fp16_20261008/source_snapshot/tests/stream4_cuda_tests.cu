#include "cuda_check.hpp"
#include "config.hpp"
#include "../cuda/stream4.hpp"

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
} // namespace

int main() {
    std::filesystem::create_directories("test_results");
    std::ofstream report("test_results/stream4_cuda_tests_2026-05-20.md");
    report << "# Stream4 CUDA Tests 2026-05-20\n\n";

    BEAM_CUDA_CHECK(cudaSetDevice(0));
    std::vector<CandidateMeta> input{
        CandidateMeta{Hash128{1, 2}, 9, 5, pack_route(0, 0, 3)},
        CandidateMeta{Hash128{1, 2}, 8, 5, pack_route(0, 0, 4)},
        CandidateMeta{Hash128{3, 4}, 7, 11, pack_route(0, 0, 5)},
        CandidateMeta{Hash128{5, 6}, 6, 2, pack_route(0, 0, 6)},
    };
    const std::uint32_t capacity = static_cast<std::uint32_t>(input.size());
    const std::uint32_t block_count = 1;
    constexpr std::size_t cub_temp_bytes = 8ULL * 1024ULL * 1024ULL;

    CandidateMeta* d_survivor = nullptr;
    Hash128* d_sort_key = nullptr;
    Hash128* d_reduce_key = nullptr;
    CandidateMeta* d_sort_value = nullptr;
    CandidateMeta* d_reduce_value = nullptr;
    std::uint32_t* d_score_key_a = nullptr;
    std::uint32_t* d_score_key_b = nullptr;
    std::uint64_t* d_score_count_a = nullptr;
    std::uint64_t* d_score_count_b = nullptr;
    std::uint32_t* d_keep_flags = nullptr;
    std::uint32_t* d_block_counts = nullptr;
    std::uint32_t* d_block_offsets = nullptr;
    std::uint32_t* d_scratch_count = nullptr;
    std::uint32_t* d_clean_count = nullptr;
    std::uint32_t* d_dirty_count = nullptr;
    std::uint32_t* d_processing_flag = nullptr;
    std::uint32_t* d_hist_a = nullptr;
    std::uint32_t* d_hist_b = nullptr;
    std::uint32_t* d_hist_active = nullptr;
    void* d_cub_temp = nullptr;

    BEAM_CUDA_CHECK(cudaMalloc(&d_survivor, input.size() * sizeof(CandidateMeta)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_sort_key, input.size() * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_reduce_key, input.size() * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_sort_value, input.size() * sizeof(CandidateMeta)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_reduce_value, input.size() * sizeof(CandidateMeta)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_score_key_a, input.size() * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_score_key_b, input.size() * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_score_count_a, input.size() * sizeof(std::uint64_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_score_count_b, input.size() * sizeof(std::uint64_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_keep_flags, input.size() * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_block_counts, block_count * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_block_offsets, block_count * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_scratch_count, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_clean_count, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_dirty_count, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_processing_flag, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_hist_a, SCORE_BIN_COUNT * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_hist_b, SCORE_BIN_COUNT * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_hist_active, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_bytes));
    BEAM_CUDA_CHECK(cudaMemset(d_hist_a, 0, SCORE_BIN_COUNT * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_hist_b, 0, SCORE_BIN_COUNT * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_hist_active, 0, sizeof(std::uint32_t)));

    const std::uint32_t clean_count_in = 1;
    const std::uint32_t dirty_count_in = 3;
    const std::uint32_t processing_flag_in = 1;
    BEAM_CUDA_CHECK(cudaMemcpy(d_survivor, input.data(), input.size() * sizeof(CandidateMeta), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_clean_count, &clean_count_in, sizeof(clean_count_in), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_dirty_count, &dirty_count_in, sizeof(dirty_count_in), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_processing_flag, &processing_flag_in, sizeof(processing_flag_in), cudaMemcpyHostToDevice));

    stream4_shard_job_cuda(
        d_survivor,
        d_clean_count,
        d_dirty_count,
        d_processing_flag,
        10,
        capacity,
        d_sort_key,
        d_reduce_key,
        d_sort_value,
        d_reduce_value,
        d_score_key_a,
        d_score_key_b,
        d_score_count_a,
        d_score_count_b,
        d_keep_flags,
        d_block_counts,
        d_block_offsets,
        d_scratch_count,
        d_hist_a,
        d_hist_b,
        d_hist_active,
        d_cub_temp,
        cub_temp_bytes,
        0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());

    std::uint32_t clean_count_out = 0;
    std::uint32_t dirty_count_out = 0;
    std::uint32_t processing_flag_out = 0;
    std::vector<CandidateMeta> output(input.size());
    std::vector<std::uint32_t> hist_a(SCORE_BIN_COUNT);
    std::vector<std::uint32_t> hist_b(SCORE_BIN_COUNT);
    std::uint32_t hist_active = 0;
    BEAM_CUDA_CHECK(cudaMemcpy(&clean_count_out, d_clean_count, sizeof(clean_count_out), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&dirty_count_out, d_dirty_count, sizeof(dirty_count_out), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&processing_flag_out, d_processing_flag, sizeof(processing_flag_out), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(output.data(), d_survivor, output.size() * sizeof(CandidateMeta), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(hist_a.data(), d_hist_a, hist_a.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(hist_b.data(), d_hist_b, hist_b.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&hist_active, d_hist_active, sizeof(hist_active), cudaMemcpyDeviceToHost));

    require(clean_count_out == 2, "stream4 shard job clean count failed");
    require(dirty_count_out == 0, "stream4 shard job dirty count must reset");
    require(processing_flag_out == 0, "stream4 shard job processing flag must reset");
    bool saw_tiebreak = false;
    bool saw_low_score = false;
    for (std::uint32_t i = 0; i < clean_count_out; ++i) {
        if (output[i].hash == Hash128{1, 2} && output[i].parent_idx == 8 && unpack_move(output[i].route_packed) == 4) {
            saw_tiebreak = true;
        }
        if (output[i].hash == Hash128{5, 6} && output[i].score_key == 2) {
            saw_low_score = true;
        }
    }
    require(saw_tiebreak, "stream4 tie-break failed");
    require(saw_low_score, "stream4 threshold keep failed");
    const std::vector<std::uint32_t>& active_hist = (hist_active & 1U) == 0U ? hist_a : hist_b;
    require(active_hist[2] == 1 && active_hist[5] == 1, "stream4 shard histogram failed");

    report << "- threshold_compact=pass\n";
    report << "- cub_sort_reduce_by_hash=pass\n";
    report << "- tie_parent_route=pass\n";
    report << "- shard_clean_dirty_update=pass\n";
    report << "\nstatus=pass\n";

    cudaFree(d_survivor);
    cudaFree(d_sort_key);
    cudaFree(d_reduce_key);
    cudaFree(d_sort_value);
    cudaFree(d_reduce_value);
    cudaFree(d_score_key_a);
    cudaFree(d_score_key_b);
    cudaFree(d_score_count_a);
    cudaFree(d_score_count_b);
    cudaFree(d_keep_flags);
    cudaFree(d_block_counts);
    cudaFree(d_block_offsets);
    cudaFree(d_scratch_count);
    cudaFree(d_clean_count);
    cudaFree(d_dirty_count);
    cudaFree(d_processing_flag);
    cudaFree(d_hist_a);
    cudaFree(d_hist_b);
    cudaFree(d_hist_active);
    cudaFree(d_cub_temp);
    std::cout << "stream4_cuda_tests=pass\n";
    return 0;
}
