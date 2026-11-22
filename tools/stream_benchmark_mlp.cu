#include "stream_benchmark_common.hpp"
#include "stream1_transformer_score_dump.hpp"
#include <fstream>

namespace beam::bench {
constexpr std::uint64_t MLP_FNV64_OFFSET = 1469598103934665603ULL;
constexpr std::uint64_t MLP_FNV64_PRIME = 1099511628211ULL;

struct MlpScoreSummary {
    std::uint64_t checksum = 0;
    std::uint64_t digest = MLP_FNV64_OFFSET;
};

MlpScoreSummary summarize_mlp_scores(
    const std::vector<std::uint32_t*>& score_buffers,
    std::uint32_t parents_per_lane,
    std::uint32_t concurrent) {
    std::ofstream dump;
    if (const char* path = std::getenv("BEAM_STREAM1_MLP_SCORE_DUMP"); path != nullptr && path[0] != '\0') {
        dump.open(path, std::ios::binary | std::ios::trunc);
        if (!dump) throw std::runtime_error("failed to open Stream1 MLP score dump");
        const Stream1TransformerScoreDumpHeader header = make_stream1_transformer_score_dump_header(
            concurrent, static_cast<std::uint64_t>(parents_per_lane) * MOVE_COUNT);
        dump.write(reinterpret_cast<const char*>(&header), sizeof(header));
    }
    MlpScoreSummary summary;
    std::vector<std::uint32_t> host(static_cast<std::uint64_t>(parents_per_lane) * MOVE_COUNT);
    for (std::uint32_t lane = 0; lane < concurrent; ++lane) {
        BEAM_CUDA_CHECK(cudaMemcpy(
            host.data(), score_buffers[lane], host.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
        if (dump.is_open()) {
            dump.write(reinterpret_cast<const char*>(host.data()),
                       static_cast<std::streamsize>(host.size() * sizeof(std::uint32_t)));
        }
        for (std::uint32_t value : host) {
            summary.checksum += value;
            for (int shift = 0; shift < 32; shift += 8) {
                summary.digest ^= static_cast<std::uint64_t>((value >> shift) & 0xFFU);
                summary.digest *= MLP_FNV64_PRIME;
            }
        }
    }
    if (dump.is_open() && !dump) throw std::runtime_error("failed to write Stream1 MLP score dump");
    return summary;
}


std::vector<Stream1Result> benchmark_stream1_mlp(
    const stream1_weights::DeviceWeights& weights,
    const Stream1ModelConfig& model,
    const State128* states,
    const std::uint8_t* generators,
    std::uint32_t max_states,
    std::ofstream& report) {
    std::vector<Stream1Result> results;
    const Stream1NetworkDims dims = stream1_weights::network_dims(model);
    Stream1NetworkView network{
        weights.input_weight,
        weights.input_bias,
        weights.input_ln_gamma,
        weights.input_ln_beta,
        weights.hidden_weight,
        weights.hidden_bias,
        weights.hidden_ln_gamma,
        weights.hidden_ln_beta,
        weights.residual_fc1_weight.data(),
        weights.residual_fc1_bias.data(),
        weights.residual_fc1_ln_gamma.data(),
        weights.residual_fc1_ln_beta.data(),
        weights.residual_fc2_weight.data(),
        weights.residual_fc2_bias.data(),
        weights.residual_fc2_ln_gamma.data(),
        weights.residual_fc2_ln_beta.data(),
        weights.output_weight,
        weights.output_bias,
        dims};

    const char* only_b_micro_env = std::getenv("BEAM_STREAM1_MLP_B_MICRO");
    const char* only_concurrency_env = std::getenv("BEAM_STREAM1_MLP_CONCURRENCY");
    const std::uint32_t only_b_micro = only_b_micro_env != nullptr && only_b_micro_env[0] != '\0'
        ? static_cast<std::uint32_t>(parse_u64(only_b_micro_env, "BEAM_STREAM1_MLP_B_MICRO"))
        : 0U;
    const std::uint32_t only_concurrency = only_concurrency_env != nullptr && only_concurrency_env[0] != '\0'
        ? static_cast<std::uint32_t>(parse_u64(only_concurrency_env, "BEAM_STREAM1_MLP_CONCURRENCY"))
        : 0U;

    report << "## Stream1 TensorOp CUTLASS\n\n";
    if (only_b_micro != 0U || only_concurrency != 0U) {
        report << "- filter_b_micro=" << only_b_micro << "\n";
        report << "- filter_concurrency=" << only_concurrency << "\n\n";
    }
    report << "| B_MICRO | concurrent_inference | rows_per_launch_group | ms_per_launch_group | parents_per_sec | candidates_per_sec | scratch_bytes |\n";
    report << "|---:|---:|---:|---:|---:|---:|---:|\n";
    for (std::uint32_t b_micro : B_MICRO_SWEEP) {
        const std::uint32_t parent_batch = stream1_parent_batch_from_row_budget(b_micro, model);
        for (std::uint32_t concurrent : STREAM1_CONCURRENCY_SWEEP) {
            if ((only_b_micro != 0U && b_micro != only_b_micro) ||
                (only_concurrency != 0U && concurrent != only_concurrency)) {
                continue;
            }
            const std::uint64_t rows_per_launch_group = static_cast<std::uint64_t>(parent_batch) * concurrent;
            if (rows_per_launch_group > max_states) {
                report << "|" << b_micro << "|" << concurrent << "|" << rows_per_launch_group
                       << "|skip|skip|skip|0: exceeds prepared state batch|\n";
                std::cout << "stream1_micro_skip"
                          << " b_micro=" << b_micro
                          << " concurrent=" << concurrent
                          << " reason=exceeds_prepared_state_batch\n";
                continue;
            }
            const std::uint64_t scratch_bytes = stream1_weights::stream1_scratch_bytes(model, parent_batch, concurrent);
            const std::uint64_t io_bytes = rows_per_launch_group *
                (sizeof(std::uint64_t) + sizeof(std::uint32_t) + static_cast<std::uint64_t>(MOVE_COUNT) * sizeof(std::uint32_t));
            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;
            BEAM_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            constexpr std::uint64_t safety_margin_bytes = 256ULL * 1024ULL * 1024ULL;
            if (scratch_bytes + io_bytes + safety_margin_bytes > free_bytes) {
                report << "|" << b_micro << "|" << concurrent << "|" << rows_per_launch_group
                       << "|skip|skip|skip|" << scratch_bytes
                       << ": estimated allocation exceeds available GPU memory"
                       << " free_bytes=" << free_bytes
                       << " io_bytes=" << io_bytes << "|\n";
                std::cout << "stream1_micro_skip"
                          << " b_micro=" << b_micro
                          << " concurrent=" << concurrent
                          << " rows_per_launch_group=" << rows_per_launch_group
                          << " scratch_bytes=" << scratch_bytes
                          << " free_bytes=" << free_bytes
                          << " reason=estimated_allocation_exceeds_available_memory\n";
                continue;
            }
            std::vector<cudaStream_t> streams = create_streams(concurrent);
            std::vector<stream1_weights::ScratchAllocation> scratch_sets;
            std::vector<Stream1CutlassScratch> scratch_views;
            std::vector<std::uint64_t*> parent_base(concurrent, nullptr);
            std::vector<std::uint32_t*> count(concurrent, nullptr);
            std::vector<std::uint32_t*> score(concurrent, nullptr);
            scratch_sets.reserve(concurrent);
            scratch_views.reserve(concurrent);
            for (std::uint32_t i = 0; i < concurrent; ++i) {
                scratch_sets.push_back(stream1_weights::alloc_stream1_scratch(model, parent_batch, 1));
                scratch_views.push_back(Stream1CutlassScratch{
                    scratch_sets.back().hidden1,
                    scratch_sets.back().hidden2,
                    scratch_sets.back().residual,
                    scratch_sets.back().output});
                parent_base[i] = device_alloc<std::uint64_t>(1);
                count[i] = device_alloc<std::uint32_t>(1);
                score[i] = device_alloc<std::uint32_t>(static_cast<std::uint64_t>(parent_batch) * MOVE_COUNT);
                const std::uint64_t base = static_cast<std::uint64_t>(i) * parent_batch;
                BEAM_CUDA_CHECK(cudaMemcpy(parent_base[i], &base, sizeof(base), cudaMemcpyHostToDevice));
                BEAM_CUDA_CHECK(cudaMemcpy(count[i], &parent_batch, sizeof(parent_batch), cudaMemcpyHostToDevice));
            }
            const std::uint32_t iterations = b_micro >= 4096 ? 6U : 10U;
            const float ms = time_gpu_ms(streams, iterations, [&]() {
                for (std::uint32_t i = 0; i < concurrent; ++i) {
                    stream1_inference_cutlass_cuda(
                        states,
                        parent_base[i],
                        count[i],
                        generators,
                        network,
                        scratch_views[i],
                        score[i],
                        parent_batch,
                        streams[i]);
                }
            });
            const MlpScoreSummary score_summary = summarize_mlp_scores(score, parent_batch, concurrent);
            const double parents = static_cast<double>(rows_per_launch_group);
            const double parent_per_sec = parents * 1000.0 / static_cast<double>(ms);
            const double candidate_per_sec = parents * static_cast<double>(MOVE_COUNT) * 1000.0 / static_cast<double>(ms);
            results.push_back(Stream1Result{b_micro, concurrent, ms, parent_per_sec, candidate_per_sec});
            report << "|" << b_micro
                   << "|" << concurrent
                   << "|" << rows_per_launch_group
                   << "|" << std::fixed << std::setprecision(4) << ms
                   << "|" << std::setprecision(1) << parent_per_sec
                   << "|" << candidate_per_sec
                   << "|" << scratch_bytes << "|\n";
            std::cout << "stream1_micro"
                      << " b_micro=" << b_micro
                      << " concurrent=" << concurrent
                      << " rows_per_launch_group=" << rows_per_launch_group
                      << " ms_per_launch_group=" << std::fixed << std::setprecision(4) << ms
                      << " parents_per_sec=" << std::setprecision(1) << parent_per_sec
                      << " candidates_per_sec=" << candidate_per_sec
                      << " scratch_bytes=" << scratch_bytes
                      << " checksum=" << score_summary.checksum
                      << " score_key_digest=" << score_summary.digest
                      << "\n";
            for (std::uint32_t i = 0; i < concurrent; ++i) {
                cudaFree(parent_base[i]);
                cudaFree(count[i]);
                cudaFree(score[i]);
                stream1_weights::free_stream1_scratch(scratch_sets[i]);
            }
            destroy_streams(streams);
        }
    }
    report << "\n";
    return results;
}
std::vector<StreamResult> benchmark_stream2(
    const State128* states,
    const std::uint8_t* generators,
    const State128* central,
    const Hash128* zobrist,
    std::ofstream& report) {
    std::vector<StreamResult> results;
    report << "## Stream2 Hash Goal\n\n";
    report << "| B_MICRO | ms_per_job | candidates_per_sec |\n";
    report << "|---:|---:|---:|\n";
    for (std::uint32_t b_micro : B_MICRO_SWEEP) {
        std::vector<cudaStream_t> streams = create_streams(1);
        std::uint64_t* parent_base = device_alloc<std::uint64_t>(1);
        std::uint32_t* count = device_alloc<std::uint32_t>(1);
        Hash128* hash_ring = device_alloc<Hash128>(static_cast<std::uint64_t>(b_micro) * MOVE_COUNT);
        constexpr std::uint64_t base = 0;
        BEAM_CUDA_CHECK(cudaMemcpy(parent_base, &base, sizeof(base), cudaMemcpyHostToDevice));
        BEAM_CUDA_CHECK(cudaMemcpy(count, &b_micro, sizeof(b_micro), cudaMemcpyHostToDevice));
        const std::uint32_t iterations = b_micro >= 4096 ? 20U : 40U;
        Stream2SolvedBuffers solved{};
        const float ms = time_gpu_ms(streams, iterations, [&]() {
            stream2_hash_goal_cuda(
                states,
                parent_base,
                count,
                generators,
                central,
                zobrist,
                hash_ring,
                0,
                0,
                b_micro,
                0,
                0,
                solved,
                streams[0]);
        });
        const double candidates = static_cast<double>(b_micro) * MOVE_COUNT;
        const double candidate_per_sec = candidates * 1000.0 / static_cast<double>(ms);
        results.push_back(StreamResult{"Stream2", b_micro, 0, ms, candidate_per_sec});
        report << "|" << b_micro << "|" << std::fixed << std::setprecision(4) << ms << "|" << std::setprecision(1) << candidate_per_sec << "|\n";
        cudaFree(parent_base);
        cudaFree(count);
        cudaFree(hash_ring);
        destroy_streams(streams);
    }
    report << "\n";
    return results;
}

std::vector<StreamResult> benchmark_stream3(std::ofstream& report) {
    std::vector<StreamResult> results;
    report << "## Stream3 Threshold Sort Dedup Restore Collect\n\n";
    report << "| B_MICRO | ring_slot_count | candidates | ms_per_job | candidates_per_sec |\n";
    report << "|---:|---:|---:|---:|---:|\n";
    for (std::uint32_t b_micro : B_MICRO_SWEEP) {
        constexpr std::uint32_t ring_slot_count = 4;
        const std::uint32_t candidate_count = b_micro * static_cast<std::uint32_t>(MOVE_COUNT) * ring_slot_count;
        std::vector<cudaStream_t> streams = create_streams(1);
        RuntimeConfig config;
        config.b_micro = b_micro;
        config.stream3_batch_candidates = candidate_count;
        config.stream4_batch_candidates = 65536;
        config.stream4_active_sort_slots = 1;
        config.ring_count = 1;
        config.shard_count = 64;
        config.global_spill_capacity = candidate_count;
        config.user_global_beam_width = 4194304;
        StaticMemoryPlan plan = make_static_memory_plan(config);
        StaticDeviceMemory memory;
        allocate_static_device_memory(plan, memory);
        BenchmarkThresholdBuffers threshold_buffers = alloc_benchmark_threshold_buffers();
        attach_benchmark_threshold_buffers(memory, threshold_buffers);
        BEAM_CUDA_CHECK(cudaMemset(memory.allocation, 0, memory.allocation_bytes));
        std::vector<std::uint32_t> host_score(candidate_count);
        std::vector<Hash128> host_hash(candidate_count);
        for (std::uint32_t i = 0; i < candidate_count; ++i) {
            host_score[i] = i % SCORE_BIN_COUNT;
            host_hash[i] = Hash128{static_cast<std::uint64_t>(i) * 0x9E3779B185EBCA87ULL, static_cast<std::uint64_t>(i) ^ 0xD1B54A32D192ED03ULL};
        }
        std::vector<std::uint64_t> host_parent_base(ring_slot_count);
        std::vector<std::uint32_t> host_count(ring_slot_count, b_micro);
        for (std::uint32_t slot = 0; slot < ring_slot_count; ++slot) {
            host_parent_base[slot] = static_cast<std::uint64_t>(slot) * b_micro;
        }
        BEAM_CUDA_CHECK(cudaMemcpy(memory.streams.score_ring, host_score.data(), candidate_count * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
        BEAM_CUDA_CHECK(cudaMemcpy(memory.streams.hash_ring, host_hash.data(), candidate_count * sizeof(Hash128), cudaMemcpyHostToDevice));
        BEAM_CUDA_CHECK(cudaMemcpy(memory.streams.parent_base, host_parent_base.data(), ring_slot_count * sizeof(std::uint64_t), cudaMemcpyHostToDevice));
        BEAM_CUDA_CHECK(cudaMemcpy(memory.streams.count, host_count.data(), ring_slot_count * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
        const std::uint32_t threshold = SCORE_MAX_KEY;
        publish_threshold_for_benchmark(memory, threshold);
        const float ms = time_gpu_ms(streams, 3, [&]() {
            BEAM_CUDA_CHECK(cudaMemsetAsync(memory.streams.clean_count, 0, config.shard_count * sizeof(std::uint32_t), streams[0]));
            BEAM_CUDA_CHECK(cudaMemsetAsync(memory.streams.dirty_count, 0, config.shard_count * sizeof(std::uint32_t), streams[0]));
            BEAM_CUDA_CHECK(cudaMemsetAsync(memory.streams.processing_flag, 0, config.shard_count * sizeof(std::uint32_t), streams[0]));
            BEAM_CUDA_CHECK(cudaMemsetAsync(memory.streams.global_spill_count, 0, 2 * sizeof(std::uint32_t), streams[0]));
            BEAM_CUDA_CHECK(cudaMemsetAsync(memory.streams.global_spill_active_index, 0, sizeof(std::uint32_t), streams[0]));
            stream3_pack_threshold_device_threshold_cuda(
                memory.streams.score_ring,
                memory.streams.hash_ring,
                memory.streams.parent_base,
                memory.streams.count,
                memory.streams.stream3_key_a,
                memory.streams.stream3_val_a,
                memory.streams.stream3_key_b,
                memory.streams.stream3_val_b,
                memory.streams.unique_key,
                memory.streams.unique_val,
                memory.streams.stream3_keep_flags,
                memory.streams.stream3_block_counts,
                memory.streams.stream3_block_offsets,
                memory.streams.unique_count,
                memory.streams.stream3_cub_temp,
                memory.streams.stream3_cub_temp_bytes,
                memory.streams.current_threshold,
                memory.streams.current_threshold_active_index,
                b_micro,
                candidate_count,
                streams[0]);
            stream3_restore_collect_single_owner_cuda(
                memory.streams.unique_key,
                memory.streams.unique_val,
                memory.streams.unique_count,
                memory.streams.parent_base,
                memory.streams.local_pending_count,
                memory.streams.send_count,
                memory.streams.send_offset,
                memory.streams.survivor_shard,
                memory.streams.clean_count,
                memory.streams.dirty_count,
                memory.streams.processing_flag,
                memory.streams.global_spill_buffer_a,
                memory.streams.global_spill_buffer_b,
                memory.streams.global_spill_count,
                memory.streams.global_spill_active_index,
                memory.streams.stream3_write_buffer_index,
                memory.streams.stream3_shard_counts,
                memory.streams.stream3_shard_offsets,
                memory.streams.stream3_spill_counts,
                memory.streams.stream3_spill_offsets,
                memory.streams.stream3_partition_key_a,
                memory.streams.stream3_partition_key_b,
                memory.streams.stream3_partition_val_a,
                memory.streams.stream3_partition_val_b,
                memory.streams.stream3_partition_unique_shard,
                memory.streams.stream3_partition_unique_counts,
                memory.streams.stream3_partition_unique_count,
                memory.streams.stream3_cub_temp,
                memory.streams.stream3_cub_temp_bytes,
                0,
                b_micro,
                candidate_count,
                config.shard_count,
                config.shard_buffer_count,
                config.shard_capacity_candidates,
                config.stream4_batch_candidates,
                config.global_spill_capacity,
                streams[0]);
        });
        const double candidate_per_sec = static_cast<double>(candidate_count) * 1000.0 / static_cast<double>(ms);
        results.push_back(StreamResult{"Stream3", b_micro, ring_slot_count, ms, candidate_per_sec});
        report << "|" << b_micro << "|" << ring_slot_count << "|" << candidate_count << "|" << std::fixed << std::setprecision(4) << ms << "|" << std::setprecision(1) << candidate_per_sec << "|\n";
        free_benchmark_threshold_buffers(threshold_buffers);
        free_static_device_memory(memory);
        destroy_streams(streams);
    }
    report << "\n";
    return results;
}

std::vector<StreamResult> benchmark_stream4(std::ofstream& report) {
    std::vector<StreamResult> results;
    report << "## Stream4 Threshold Compact Sort Reduce\n\n";
    report << "| shard_capacity | STREAM4_BATCH_CANDIDATES | input_count | ms_per_job | shard_items_per_sec | batch_candidates_per_sec | allocation_bytes |\n";
    report << "|---:|---:|---:|---:|---:|---:|---:|\n";
    for (std::uint32_t capacity : STREAM4_SHARD_CAPACITY_SWEEP) {
        std::vector<CandidateMeta> host(capacity);
        for (std::uint32_t i = 0; i < capacity; ++i) {
            host[i].hash = Hash128{static_cast<std::uint64_t>(i), static_cast<std::uint64_t>(capacity - i)};
            host[i].parent_idx = i;
            host[i].score_key = i % SCORE_BIN_COUNT;
            host[i].route_packed = i % MOVE_COUNT;
        }
        for (std::uint32_t batch : STREAM4_BATCH_SWEEP) {
            if (batch > capacity) {
                continue;
            }
            RuntimeConfig config;
            config.b_micro = 8192;
            config.stream3_batch_candidates = config.b_micro * static_cast<std::uint32_t>(MOVE_COUNT);
            config.stream4_batch_candidates = batch;
            config.stream4_active_sort_slots = 1;
            config.ring_count = 1;
            config.shard_count = 1;
            config.shard_capacity_candidates = capacity;
            config.global_spill_capacity = config.stream3_batch_candidates;
            config.user_global_beam_width = batch;
            StaticMemoryPlan plan = make_static_memory_plan(config);
            StaticDeviceMemory memory;
            allocate_static_device_memory(plan, memory);
            BenchmarkThresholdBuffers threshold_buffers = alloc_benchmark_threshold_buffers();
            attach_benchmark_threshold_buffers(memory, threshold_buffers);
            BEAM_CUDA_CHECK(cudaMemset(memory.allocation, 0, memory.allocation_bytes));
            std::vector<cudaStream_t> streams = create_streams(1);
            const std::uint32_t clean_count = 0;
            const std::uint32_t dirty_count = batch;
            const std::uint32_t threshold = SCORE_MAX_KEY;
            const std::uint32_t processing_flag = 0;
            BEAM_CUDA_CHECK(cudaMemcpy(memory.streams.survivor_shard, host.data(), capacity * sizeof(CandidateMeta), cudaMemcpyHostToDevice));
            BEAM_CUDA_CHECK(cudaMemcpy(memory.streams.clean_count, &clean_count, sizeof(clean_count), cudaMemcpyHostToDevice));
            BEAM_CUDA_CHECK(cudaMemcpy(memory.streams.dirty_count, &dirty_count, sizeof(dirty_count), cudaMemcpyHostToDevice));
            publish_threshold_for_benchmark(memory, threshold);
            BEAM_CUDA_CHECK(cudaMemcpy(memory.streams.processing_flag, &processing_flag, sizeof(processing_flag), cudaMemcpyHostToDevice));
            auto launch_stream4 = [&]() {
                stream4_shard_job_device_threshold_cuda(
                    memory.streams.survivor_shard,
                    memory.streams.clean_count,
                    memory.streams.dirty_count,
                    memory.streams.processing_flag,
                    memory.streams.current_threshold,
                    memory.streams.current_threshold_active_index,
                    capacity,
                    memory.streams.stream4_key_a,
                    memory.streams.stream4_key_b,
                    memory.streams.stream4_val_a,
                    memory.streams.stream4_val_b,
                    memory.streams.stream4_score_key_a,
                    memory.streams.stream4_score_key_b,
                    memory.streams.stream4_score_count_a,
                    memory.streams.stream4_score_count_b,
                    memory.streams.stream4_keep_flags,
                    memory.streams.stream4_block_counts,
                    memory.streams.stream4_block_offsets,
                    memory.streams.stream4_count,
                    memory.streams.shard_score_hist_a,
                    memory.streams.shard_score_hist_b,
                    memory.streams.shard_score_hist_active_index,
                    memory.streams.stream4_cub_temp,
                    memory.streams.stream4_cub_temp_bytes,
                    streams[0]);
            };
            launch_stream4();
            BEAM_CUDA_CHECK(cudaStreamSynchronize(streams[0]));
            BEAM_CUDA_CHECK(cudaMemcpy(memory.streams.survivor_shard, host.data(), capacity * sizeof(CandidateMeta), cudaMemcpyHostToDevice));
            BEAM_CUDA_CHECK(cudaMemcpy(memory.streams.clean_count, &clean_count, sizeof(clean_count), cudaMemcpyHostToDevice));
            BEAM_CUDA_CHECK(cudaMemcpy(memory.streams.dirty_count, &dirty_count, sizeof(dirty_count), cudaMemcpyHostToDevice));
            publish_threshold_for_benchmark(memory, threshold);
            BEAM_CUDA_CHECK(cudaMemcpy(memory.streams.processing_flag, &processing_flag, sizeof(processing_flag), cudaMemcpyHostToDevice));
            const float ms = time_single_stream_ms(streams[0], launch_stream4);
            const double shard_items_per_sec =
                static_cast<double>(capacity) * 1000.0 / static_cast<double>(ms);
            const double batch_candidates_per_sec =
                static_cast<double>(batch) * 1000.0 / static_cast<double>(ms);
            results.push_back(StreamResult{"Stream4", batch, capacity, ms, shard_items_per_sec});
            report << "|" << capacity
                   << "|" << batch
                   << "|" << dirty_count
                   << "|" << std::fixed << std::setprecision(4) << ms
                   << "|" << std::setprecision(1) << shard_items_per_sec
                   << "|" << batch_candidates_per_sec
                   << "|" << plan.total_device_bytes
                   << "|\n";
            std::cout << "stream4_micro"
                      << " shard_capacity=" << capacity
                      << " stream4_batch=" << batch
                      << " input_count=" << dirty_count
                      << " ms_per_job=" << std::fixed << std::setprecision(4) << ms
                      << " shard_items_per_sec=" << std::setprecision(1) << shard_items_per_sec
                      << " batch_candidates_per_sec=" << batch_candidates_per_sec
                      << " allocation_bytes=" << plan.total_device_bytes
                      << "\n";
            destroy_streams(streams);
            free_benchmark_threshold_buffers(threshold_buffers);
            free_static_device_memory(memory);
        }
    }
    report << "\n";
    return results;
}


} // namespace beam::bench