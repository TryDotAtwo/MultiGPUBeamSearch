#include "cuda_check.hpp"
#include "../cuda/stream3.hpp"

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <array>
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
    std::ofstream report("test_results/stream3_cuda_tests_2026-05-20.md");
    report << "# Stream3 CUDA Tests 2026-05-20\n\n";

    BEAM_CUDA_CHECK(cudaSetDevice(0));
    constexpr std::uint32_t b_micro = 2;
    constexpr std::uint32_t total = b_micro * static_cast<std::uint32_t>(MOVE_COUNT);
    std::vector<std::uint32_t> score(total, 100);
    std::vector<Hash128> hash(total);
    for (std::uint32_t i = 0; i < total; ++i) {
        hash[i] = Hash128{i, i + 10};
    }
    score[5] = 7;
    score[17] = 3;
    score[29] = 2;
    hash[5] = Hash128{1, 1};
    hash[17] = Hash128{5, 205};
    hash[29] = hash[17];

    std::uint32_t* d_score = nullptr;
    Hash128* d_hash = nullptr;
    Hash128* d_key = nullptr;
    Hash128* d_reduce_key = nullptr;
    Hash128* d_unique_key = nullptr;
    std::uint64_t* d_val = nullptr;
    std::uint64_t* d_reduce_val = nullptr;
    std::uint64_t* d_unique_val = nullptr;
    std::uint32_t* d_keep_flags = nullptr;
    std::uint32_t* d_owner_scratch = nullptr;
    std::uint32_t* d_block_counts = nullptr;
    std::uint32_t* d_block_offsets = nullptr;
    std::uint32_t* d_count = nullptr;
    CandidateMeta* d_local_pending = nullptr;
    std::uint32_t* d_local_pending_count = nullptr;
    CandidateMeta* d_remote_send = nullptr;
    CandidateMeta* d_remote_recv = nullptr;
    std::uint32_t* d_send_count = nullptr;
    std::uint32_t* d_send_offset = nullptr;
    std::uint32_t* d_recv_count = nullptr;
    std::uint32_t* d_recv_offset = nullptr;
    CandidateMeta* d_survivor_shard = nullptr;
    std::uint32_t* d_clean_count = nullptr;
    std::uint32_t* d_dirty_count = nullptr;
    std::uint32_t* d_processing_flag = nullptr;
    CandidateMeta* d_global_spill_a = nullptr;
    CandidateMeta* d_global_spill_b = nullptr;
    std::uint32_t* d_global_spill_count = nullptr;
    std::uint32_t* d_global_spill_active_index = nullptr;
    std::uint32_t* d_fatal_error_flag = nullptr;
    std::uint64_t* d_fatal_error_trace = nullptr;
    std::uint32_t* d_shard_counts = nullptr;
    std::uint32_t* d_shard_offsets = nullptr;
    std::uint32_t* d_spill_counts = nullptr;
    std::uint32_t* d_spill_offsets = nullptr;
    std::uint32_t* d_ready_flag = nullptr;
    std::uint32_t* d_ready_shard_list = nullptr;
    std::uint32_t* d_ready_count = nullptr;
    std::uint32_t* d_partition_key_a = nullptr;
    std::uint32_t* d_partition_key_b = nullptr;
    CandidateMeta* d_partition_val_a = nullptr;
    CandidateMeta* d_partition_val_b = nullptr;
    std::uint32_t* d_partition_unique_shard = nullptr;
    std::uint32_t* d_partition_unique_counts = nullptr;
    std::uint32_t* d_partition_unique_count = nullptr;
    void* d_cub_temp = nullptr;
    constexpr std::size_t cub_temp_bytes = 8ULL * 1024ULL * 1024ULL;
    BEAM_CUDA_CHECK(cudaMalloc(&d_score, score.size() * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_hash, hash.size() * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_key, hash.size() * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_reduce_key, hash.size() * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_unique_key, hash.size() * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_val, hash.size() * sizeof(std::uint64_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_reduce_val, hash.size() * sizeof(std::uint64_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_unique_val, hash.size() * sizeof(std::uint64_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_keep_flags, hash.size() * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_owner_scratch, hash.size() * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_block_counts, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_block_offsets, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_count, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_local_pending, hash.size() * sizeof(CandidateMeta)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_local_pending_count, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_remote_send, hash.size() * sizeof(CandidateMeta)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_remote_recv, hash.size() * sizeof(CandidateMeta)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_send_count, 3 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_send_offset, 4 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_recv_count, 3 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_recv_offset, 4 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_survivor_shard, 8 * sizeof(CandidateMeta)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_clean_count, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_dirty_count, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_processing_flag, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_global_spill_a, hash.size() * sizeof(CandidateMeta)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_global_spill_b, hash.size() * sizeof(CandidateMeta)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_global_spill_count, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_global_spill_active_index, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_fatal_error_flag, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_fatal_error_trace, STREAM_FATAL_TRACE_WORDS * sizeof(std::uint64_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_shard_counts, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_shard_offsets, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_spill_counts, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_spill_offsets, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_ready_flag, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_ready_shard_list, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_ready_count, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_partition_key_a, total * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_partition_key_b, total * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_partition_val_a, total * sizeof(CandidateMeta)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_partition_val_b, total * sizeof(CandidateMeta)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_partition_unique_shard, 3 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_partition_unique_counts, 3 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_partition_unique_count, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_bytes));
    BEAM_CUDA_CHECK(cudaMemcpy(d_score, score.data(), score.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_hash, hash.data(), hash.size() * sizeof(Hash128), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemset(d_count, 0, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_survivor_shard, 0, 8 * sizeof(CandidateMeta)));
    BEAM_CUDA_CHECK(cudaMemset(d_clean_count, 0, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_dirty_count, 0, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_processing_flag, 0, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_global_spill_count, 0, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_global_spill_active_index, 0, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_fatal_error_flag, 0, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_fatal_error_trace, 0, STREAM_FATAL_TRACE_WORDS * sizeof(std::uint64_t)));

    stream3_pack_threshold_cuda(
        d_score,
        d_hash,
        nullptr,
        nullptr,
        d_key,
        d_val,
        d_reduce_key,
        d_reduce_val,
        d_unique_key,
        d_unique_val,
        d_keep_flags,
        d_block_counts,
        d_block_offsets,
        d_count,
        d_cub_temp,
        cub_temp_bytes,
        10,
        b_micro,
        total,
        0);
    stream3_restore_owner_split_cuda(
        d_unique_key,
        d_unique_val,
        d_count,
        nullptr,
        d_local_pending,
        d_local_pending_count,
        d_remote_send,
        d_send_count,
        d_send_offset,
        d_owner_scratch,
        0,
        3,
        b_micro,
        total,
        0);
    stream3_collect_local_pending_cuda(
        d_local_pending,
        d_local_pending_count,
        d_survivor_shard,
        d_clean_count,
        d_dirty_count,
        d_processing_flag,
        d_global_spill_a,
        d_global_spill_b,
        d_global_spill_count,
        d_global_spill_active_index,
        nullptr,
        d_shard_counts,
        d_shard_offsets,
        d_spill_counts,
        d_spill_offsets,
        d_partition_key_a,
        d_partition_key_b,
        d_partition_val_a,
        d_partition_val_b,
        d_partition_unique_shard,
        d_partition_unique_counts,
        d_partition_unique_count,
        d_cub_temp,
        cub_temp_bytes,
        total,
        2,
        1,
        4,
        2,
        total,
        0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());

    std::uint32_t compact_count = 0;
    std::uint32_t local_pending_count = 0;
    std::uint32_t send_count[3]{};
    std::uint32_t send_offset[4]{};
    std::uint32_t dirty_count[2]{};
    std::uint32_t global_spill_count[2]{};
    std::uint32_t global_spill_active_index = 0;
    std::vector<Hash128> key(total);
    std::vector<std::uint64_t> val(total);
    std::vector<CandidateMeta> local_pending(total);
    std::vector<CandidateMeta> remote_send(total);
    std::vector<CandidateMeta> survivor_shard(8);
    BEAM_CUDA_CHECK(cudaMemcpy(&compact_count, d_count, sizeof(compact_count), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&local_pending_count, d_local_pending_count, sizeof(local_pending_count), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(send_count, d_send_count, sizeof(send_count), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(send_offset, d_send_offset, sizeof(send_offset), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(key.data(), d_unique_key, key.size() * sizeof(Hash128), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(val.data(), d_unique_val, val.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(local_pending.data(), d_local_pending, local_pending.size() * sizeof(CandidateMeta), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(remote_send.data(), d_remote_send, remote_send.size() * sizeof(CandidateMeta), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(dirty_count, d_dirty_count, sizeof(dirty_count), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(global_spill_count, d_global_spill_count, sizeof(global_spill_count), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&global_spill_active_index, d_global_spill_active_index, sizeof(global_spill_active_index), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(survivor_shard.data(), d_survivor_shard, survivor_shard.size() * sizeof(CandidateMeta), cudaMemcpyDeviceToHost));

    require(compact_count == 2, "stream3 compact count failed");
    bool saw5 = false;
    bool saw29 = false;
    for (std::uint32_t i = 0; i < compact_count; ++i) {
        const std::uint32_t payload_id = static_cast<std::uint32_t>(val[i] & 0xffffffffULL);
        const std::uint32_t score_key = static_cast<std::uint32_t>(val[i] >> 32);
        if (payload_id == 5 && score_key == 7) {
            saw5 = true;
        }
        if (payload_id == 29 && score_key == 2) {
            saw29 = true;
        }
    }
    require(saw5 && saw29, "stream3 payload id must be original index and dedup must keep best value");
    require(local_pending_count == 1, "stream3 local pending count failed");
    require(local_pending[0].parent_idx == 0 && local_pending[0].score_key == 7, "stream3 local pending meta failed");
    require(unpack_owner(local_pending[0].route_packed) == 0 && unpack_move(local_pending[0].route_packed) == 5, "stream3 local route failed");
    require(send_count[0] == 0 && send_count[1] == 1 && send_count[2] == 0, "stream3 send count failed");
    require(send_offset[0] == 0 && send_offset[1] == 0 && send_offset[2] == 1 && send_offset[3] == 1, "stream3 send offset failed");
    require(remote_send[0].parent_idx == 1 && remote_send[0].score_key == 2, "stream3 remote send meta failed");
    require(unpack_owner(remote_send[0].route_packed) == 1 && unpack_move(remote_send[0].route_packed) == 5, "stream3 remote route failed");
    require(dirty_count[0] == 0 && dirty_count[1] == 1, "stream3 collector dirty count failed");
    require(global_spill_count[global_spill_active_index] == 0, "stream3 collector unexpected spill failed");
    require(survivor_shard[4].hash == Hash128{1, 1} && survivor_shard[4].score_key == 7, "stream3 collector survivor write failed");

    const std::uint32_t processing_busy[2]{0, 1};
    BEAM_CUDA_CHECK(cudaMemcpy(d_processing_flag, processing_busy, sizeof(processing_busy), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemset(d_global_spill_count, 0, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_global_spill_active_index, 0, sizeof(std::uint32_t)));
    stream3_collect_local_pending_cuda(
        d_local_pending,
        d_local_pending_count,
        d_survivor_shard,
        d_clean_count,
        d_dirty_count,
        d_processing_flag,
        d_global_spill_a,
        d_global_spill_b,
        d_global_spill_count,
        d_global_spill_active_index,
        nullptr,
        d_shard_counts,
        d_shard_offsets,
        d_spill_counts,
        d_spill_offsets,
        d_partition_key_a,
        d_partition_key_b,
        d_partition_val_a,
        d_partition_val_b,
        d_partition_unique_shard,
        d_partition_unique_counts,
        d_partition_unique_count,
        d_cub_temp,
        cub_temp_bytes,
        total,
        2,
        1,
        4,
        2,
        total,
        0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    BEAM_CUDA_CHECK(cudaMemcpy(global_spill_count, d_global_spill_count, sizeof(global_spill_count), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&global_spill_active_index, d_global_spill_active_index, sizeof(global_spill_active_index), cudaMemcpyDeviceToHost));
    require(global_spill_count[global_spill_active_index] == 1, "stream3 collector busy shard spill failed");
    const std::uint32_t processing_free[2]{0, 0};
    BEAM_CUDA_CHECK(cudaMemcpy(d_processing_flag, processing_free, sizeof(processing_free), cudaMemcpyHostToDevice));
    stream3_drain_global_spill_cuda(
        d_global_spill_a,
        d_global_spill_b,
        d_global_spill_count,
        d_global_spill_active_index,
        d_survivor_shard,
        d_clean_count,
        d_dirty_count,
        d_processing_flag,
        d_shard_counts,
        d_shard_offsets,
        d_spill_counts,
        d_spill_offsets,
        d_partition_key_a,
        d_partition_key_b,
        d_partition_val_a,
        d_partition_val_b,
        d_partition_unique_shard,
        d_partition_unique_counts,
        d_partition_unique_count,
        d_cub_temp,
        cub_temp_bytes,
        2,
        total,
        4,
        2,
        0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    BEAM_CUDA_CHECK(cudaMemcpy(dirty_count, d_dirty_count, sizeof(dirty_count), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(global_spill_count, d_global_spill_count, sizeof(global_spill_count), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&global_spill_active_index, d_global_spill_active_index, sizeof(global_spill_active_index), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(survivor_shard.data(), d_survivor_shard, survivor_shard.size() * sizeof(CandidateMeta), cudaMemcpyDeviceToHost));
    require(global_spill_count[global_spill_active_index] == 0, "stream3 global spill drain count failed");
    require(dirty_count[1] == 2, "stream3 global spill drain dirty count failed");
    require(survivor_shard[5].hash == Hash128{1, 1} && survivor_shard[5].score_key == 7, "stream3 global spill drain write failed");

    BEAM_CUDA_CHECK(cudaMemcpy(d_processing_flag, processing_busy, sizeof(processing_busy), cudaMemcpyHostToDevice));
    const std::uint32_t full_spill_count[2]{1, 0};
    BEAM_CUDA_CHECK(cudaMemcpy(d_global_spill_count, full_spill_count, sizeof(full_spill_count), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemset(d_global_spill_active_index, 0, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_fatal_error_flag, 0, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_fatal_error_trace, 0, STREAM_FATAL_TRACE_WORDS * sizeof(std::uint64_t)));
    stream3_collect_local_pending_cuda(
        d_local_pending,
        d_local_pending_count,
        d_survivor_shard,
        d_clean_count,
        d_dirty_count,
        d_processing_flag,
        d_global_spill_a,
        d_global_spill_b,
        d_global_spill_count,
        d_global_spill_active_index,
        nullptr,
        d_shard_counts,
        d_shard_offsets,
        d_spill_counts,
        d_spill_offsets,
        d_partition_key_a,
        d_partition_key_b,
        d_partition_val_a,
        d_partition_val_b,
        d_partition_unique_shard,
        d_partition_unique_counts,
        d_partition_unique_count,
        d_cub_temp,
        cub_temp_bytes,
        total,
        2,
        1,
        4,
        2,
        1,
        0,
        d_fatal_error_flag,
        d_fatal_error_trace);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    std::uint32_t fatal_error_flag = 0;
    std::array<std::uint64_t, STREAM_FATAL_TRACE_WORDS> fatal_error_trace{};
    BEAM_CUDA_CHECK(cudaMemcpy(&fatal_error_flag, d_fatal_error_flag, sizeof(fatal_error_flag), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(fatal_error_trace.data(), d_fatal_error_trace, fatal_error_trace.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    require(fatal_error_flag == STREAM_FATAL_STREAM3_SPILL_OVERFLOW, "stream3 spill overflow fatal flag failed");
    require(fatal_error_trace[FatalTraceSpillEnd] > fatal_error_trace[FatalTraceSpillCapacity],
            "stream3 spill overflow fatal trace failed");
    BEAM_CUDA_CHECK(cudaMemcpy(d_processing_flag, processing_free, sizeof(processing_free), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemset(d_global_spill_count, 0, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_global_spill_active_index, 0, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_fatal_error_flag, 0, sizeof(std::uint32_t)));

    stream3_build_ready_shard_queue_cuda(
        d_clean_count,
        d_dirty_count,
        d_processing_flag,
        nullptr,
        d_ready_flag,
        d_ready_shard_list,
        d_ready_count,
        2,
        1,
        4,
        4,
        2,
        false,
        false,
        0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    std::uint32_t ready_count = 0;
    std::uint32_t ready_shard_list[2]{};
    std::uint32_t processing_after_ready[2]{};
    BEAM_CUDA_CHECK(cudaMemcpy(&ready_count, d_ready_count, sizeof(ready_count), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(ready_shard_list, d_ready_shard_list, sizeof(ready_shard_list), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(processing_after_ready, d_processing_flag, sizeof(processing_after_ready), cudaMemcpyDeviceToHost));
    require(ready_count == 1 && ready_shard_list[0] == 1, "stream3 ready shard queue failed");
    require(processing_after_ready[1] == 1, "stream3 ready shard queue must set processing flag");

    std::uint32_t* d_db_clean_count = nullptr;
    std::uint32_t* d_db_dirty_count = nullptr;
    std::uint32_t* d_db_processing_flag = nullptr;
    std::uint32_t* d_db_write_buffer_index = nullptr;
    std::uint32_t* d_db_ready_flag = nullptr;
    std::uint32_t* d_db_ready_shard_list = nullptr;
    std::uint32_t* d_db_ready_count = nullptr;
    BEAM_CUDA_CHECK(cudaMalloc(&d_db_clean_count, 4 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_db_dirty_count, 4 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_db_processing_flag, 4 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_db_write_buffer_index, 2 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_db_ready_flag, 4 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_db_ready_shard_list, 4 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_db_ready_count, sizeof(std::uint32_t)));
    const std::uint32_t db_clean_count[4]{0, 0, 0, 0};
    const std::uint32_t db_dirty_count[4]{3, 3, 3, 0};
    const std::uint32_t db_processing_free[4]{0, 0, 0, 0};
    const std::uint32_t db_write_index_start[2]{0, 0};
    BEAM_CUDA_CHECK(cudaMemcpy(d_db_clean_count, db_clean_count, sizeof(db_clean_count), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_db_dirty_count, db_dirty_count, sizeof(db_dirty_count), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_db_processing_flag, db_processing_free, sizeof(db_processing_free), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_db_write_buffer_index, db_write_index_start, sizeof(db_write_index_start), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemset(d_db_ready_flag, 0, 4 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_db_ready_shard_list, 0, 4 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_db_ready_count, 0, sizeof(std::uint32_t)));
    stream3_build_ready_shard_queue_cuda(
        d_db_clean_count,
        d_db_dirty_count,
        d_db_processing_flag,
        d_db_write_buffer_index,
        d_db_ready_flag,
        d_db_ready_shard_list,
        d_db_ready_count,
        2,
        2,
        8,
        4,
        2,
        false,
        false,
        0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    std::uint32_t db_ready_count = 0;
    std::uint32_t db_ready_shards[4]{};
    std::uint32_t db_processing_after[4]{};
    std::uint32_t db_write_index_after[2]{};
    BEAM_CUDA_CHECK(cudaMemcpy(&db_ready_count, d_db_ready_count, sizeof(db_ready_count), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(db_ready_shards, d_db_ready_shard_list, sizeof(db_ready_shards), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(db_processing_after, d_db_processing_flag, sizeof(db_processing_after), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(db_write_index_after, d_db_write_buffer_index, sizeof(db_write_index_after), cudaMemcpyDeviceToHost));
    require(db_ready_count == 2, "double-buffer ready queue must emit at most one physical shard per logical shard");
    require(db_ready_shards[0] == 1 && db_ready_shards[1] == 2,
            "double-buffer ready queue must keep one logical sibling writable");
    require(db_processing_after[0] == 0 && db_processing_after[1] == 1,
            "double-buffer ready queue must not mark both buffers for logical shard 0");
    require(db_processing_after[2] == 1 && db_processing_after[3] == 0,
            "double-buffer ready queue must leave logical shard 1 sibling writable");
    require(db_write_index_after[0] == 0 && db_write_index_after[1] == 1,
            "double-buffer ready queue must point writes at non-processing siblings");

    const std::uint32_t db_processing_busy[4]{1, 0, 0, 0};
    BEAM_CUDA_CHECK(cudaMemcpy(d_db_processing_flag, db_processing_busy, sizeof(db_processing_busy), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_db_write_buffer_index, db_write_index_start, sizeof(db_write_index_start), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemset(d_db_ready_flag, 0, 4 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_db_ready_shard_list, 0, 4 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_db_ready_count, 0, sizeof(std::uint32_t)));
    stream3_build_ready_shard_queue_cuda(
        d_db_clean_count,
        d_db_dirty_count,
        d_db_processing_flag,
        d_db_write_buffer_index,
        d_db_ready_flag,
        d_db_ready_shard_list,
        d_db_ready_count,
        2,
        2,
        8,
        4,
        2,
        false,
        false,
        0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    BEAM_CUDA_CHECK(cudaMemcpy(&db_ready_count, d_db_ready_count, sizeof(db_ready_count), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(db_ready_shards, d_db_ready_shard_list, sizeof(db_ready_shards), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(db_processing_after, d_db_processing_flag, sizeof(db_processing_after), cudaMemcpyDeviceToHost));
    require(db_ready_count == 1 && db_ready_shards[0] == 2,
            "double-buffer ready queue must skip logical shards with active Stream4 processing");
    require(db_processing_after[0] == 1 && db_processing_after[1] == 0,
            "double-buffer ready queue must not process sibling while logical shard is busy");
    require(db_processing_after[2] == 1 && db_processing_after[3] == 0,
            "double-buffer ready queue must still process independent logical shards");

    const std::uint32_t db_clean_dirty_priority[4]{7, 0, 0, 0};
    const std::uint32_t db_dirty_dirty_priority[4]{0, 3, 0, 0};
    BEAM_CUDA_CHECK(cudaMemcpy(d_db_clean_count, db_clean_dirty_priority, sizeof(db_clean_dirty_priority), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_db_dirty_count, db_dirty_dirty_priority, sizeof(db_dirty_dirty_priority), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_db_processing_flag, db_processing_free, sizeof(db_processing_free), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_db_write_buffer_index, db_write_index_start, sizeof(db_write_index_start), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemset(d_db_ready_flag, 0, 4 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_db_ready_shard_list, 0, 4 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_db_ready_count, 0, sizeof(std::uint32_t)));
    stream3_build_ready_shard_queue_cuda(
        d_db_clean_count,
        d_db_dirty_count,
        d_db_processing_flag,
        d_db_write_buffer_index,
        d_db_ready_flag,
        d_db_ready_shard_list,
        d_db_ready_count,
        2,
        2,
        8,
        4,
        2,
        true,
        true,
        0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    BEAM_CUDA_CHECK(cudaMemcpy(&db_ready_count, d_db_ready_count, sizeof(db_ready_count), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(db_ready_shards, d_db_ready_shard_list, sizeof(db_ready_shards), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(db_processing_after, d_db_processing_flag, sizeof(db_processing_after), cudaMemcpyDeviceToHost));
    require(db_ready_count == 1 && db_ready_shards[0] == 1,
            "double-buffer ready queue must prioritize dirty sibling over clean-only sibling");
    require(db_processing_after[0] == 0 && db_processing_after[1] == 1,
            "double-buffer dirty priority must leave the clean sibling writable");

    cudaFree(d_db_clean_count);
    cudaFree(d_db_dirty_count);
    cudaFree(d_db_processing_flag);
    cudaFree(d_db_write_buffer_index);
    cudaFree(d_db_ready_flag);
    cudaFree(d_db_ready_shard_list);
    cudaFree(d_db_ready_count);

    const CandidateMeta remote_candidate{Hash128{0, 3}, 77, 4, pack_route(1, 0, 3)};
    const std::uint32_t recv_count_host[3]{0, 0, 1};
    const std::uint32_t recv_offset_host[4]{0, 0, 0, 1};
    BEAM_CUDA_CHECK(cudaMemcpy(d_remote_recv, &remote_candidate, sizeof(remote_candidate), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_recv_count, recv_count_host, sizeof(recv_count_host), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_recv_offset, recv_offset_host, sizeof(recv_offset_host), cudaMemcpyHostToDevice));
    stream3_collect_remote_recv_cuda(
        d_remote_recv,
        d_recv_count,
        d_recv_offset,
        d_survivor_shard,
        d_clean_count,
        d_dirty_count,
        d_processing_flag,
        d_global_spill_a,
        d_global_spill_b,
        d_global_spill_count,
        d_global_spill_active_index,
        nullptr,
        d_shard_counts,
        d_shard_offsets,
        d_spill_counts,
        d_spill_offsets,
        d_partition_key_a,
        d_partition_key_b,
        d_partition_val_a,
        d_partition_val_b,
        d_partition_unique_shard,
        d_partition_unique_counts,
        d_partition_unique_count,
        d_cub_temp,
        cub_temp_bytes,
        total,
        3,
        2,
        1,
        4,
        2,
        total,
        0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    BEAM_CUDA_CHECK(cudaMemcpy(dirty_count, d_dirty_count, sizeof(dirty_count), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(survivor_shard.data(), d_survivor_shard, survivor_shard.size() * sizeof(CandidateMeta), cudaMemcpyDeviceToHost));
    require(dirty_count[0] == 1, "stream3 remote recv collector dirty count failed");
    require(survivor_shard[0].parent_idx == 77 && survivor_shard[0].score_key == 4, "stream3 remote recv collector write failed");
    report << "- threshold_compact=pass\n";
    report << "- hash_sort_dedup=pass\n";
    report << "- payload_id_original_index=pass\n";
    report << "- restore_owner_split=pass\n";
    report << "- local_pending_collector=pass\n";
    report << "- busy_shard_spill=pass\n";
    report << "- global_spill_drain=pass\n";
    report << "- spill_overflow_fatal_guard=pass\n";
    report << "- ready_shard_queue=pass\n";
    report << "- double_buffer_ready_queue_logical_mutex=pass\n";
    report << "- remote_recv_collector=pass\n";
    report << "\nstatus=pass\n";

    cudaFree(d_score);
    cudaFree(d_hash);
    cudaFree(d_key);
    cudaFree(d_reduce_key);
    cudaFree(d_unique_key);
    cudaFree(d_val);
    cudaFree(d_reduce_val);
    cudaFree(d_unique_val);
    cudaFree(d_keep_flags);
    cudaFree(d_owner_scratch);
    cudaFree(d_block_counts);
    cudaFree(d_block_offsets);
    cudaFree(d_count);
    cudaFree(d_local_pending);
    cudaFree(d_local_pending_count);
    cudaFree(d_remote_send);
    cudaFree(d_remote_recv);
    cudaFree(d_send_count);
    cudaFree(d_send_offset);
    cudaFree(d_recv_count);
    cudaFree(d_recv_offset);
    cudaFree(d_survivor_shard);
    cudaFree(d_clean_count);
    cudaFree(d_dirty_count);
    cudaFree(d_processing_flag);
    cudaFree(d_global_spill_a);
    cudaFree(d_global_spill_b);
    cudaFree(d_global_spill_count);
    cudaFree(d_global_spill_active_index);
    cudaFree(d_fatal_error_flag);
    cudaFree(d_fatal_error_trace);
    cudaFree(d_shard_counts);
    cudaFree(d_shard_offsets);
    cudaFree(d_spill_counts);
    cudaFree(d_spill_offsets);
    cudaFree(d_ready_flag);
    cudaFree(d_ready_shard_list);
    cudaFree(d_ready_count);
    cudaFree(d_partition_key_a);
    cudaFree(d_partition_key_b);
    cudaFree(d_partition_val_a);
    cudaFree(d_partition_val_b);
    cudaFree(d_partition_unique_shard);
    cudaFree(d_partition_unique_counts);
    cudaFree(d_partition_unique_count);
    cudaFree(d_cub_temp);
    std::cout << "stream3_cuda_tests=pass\n";
    return 0;
}
