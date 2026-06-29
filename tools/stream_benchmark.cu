#include "cuda_check.hpp"
#include "stream1_weight_io.hpp"
#include "../cuda/stream1.hpp"
#include "../cuda/stream2.hpp"
#include "../cuda/stream3.hpp"
#include "../cuda/stream4.hpp"
#include "../cuda/static_memory.hpp"
#include "../src/hash.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace beam;

namespace {

inline constexpr std::array<std::uint32_t, 6> B_MICRO_SWEEP{2048, 4096, 8192, 16384, 32768, 65536};
inline constexpr std::array<std::uint32_t, 6> STREAM1_CONCURRENCY_SWEEP{1, 2, 4, 8, 16, 32};
inline constexpr std::array<std::uint32_t, 6> STREAM4_BATCH_SWEEP{196608, 262144, 393216, 524288, 699392, 1048576};
inline constexpr std::array<std::uint32_t, 4> STREAM4_SHARD_CAPACITY_SWEEP{1048576, 2621440, 5242880, 10485760};

void publish_threshold_for_benchmark(StaticDeviceMemory& memory, std::uint32_t threshold) {
    const std::uint32_t thresholds[2]{threshold, threshold};
    const std::uint32_t initialized[2]{1U, 1U};
    const std::uint32_t active = 0;
    BEAM_CUDA_CHECK(cudaMemcpy(
        memory.streams.current_threshold,
        thresholds,
        sizeof(thresholds),
        cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(
        memory.streams.threshold_initialized,
        initialized,
        sizeof(initialized),
        cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(
        memory.streams.current_threshold_active_index,
        &active,
        sizeof(active),
        cudaMemcpyHostToDevice));
}

struct BenchmarkThresholdBuffers {
    std::uint32_t* current_threshold = nullptr;
    std::uint32_t* threshold_initialized = nullptr;
    std::uint32_t* active_index = nullptr;
};

BenchmarkThresholdBuffers alloc_benchmark_threshold_buffers() {
    BenchmarkThresholdBuffers buffers;
    BEAM_CUDA_CHECK(cudaMalloc(&buffers.current_threshold, 2ULL * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&buffers.threshold_initialized, 2ULL * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&buffers.active_index, sizeof(std::uint32_t)));
    return buffers;
}

void attach_benchmark_threshold_buffers(StaticDeviceMemory& memory, const BenchmarkThresholdBuffers& buffers) {
    memory.streams.current_threshold = buffers.current_threshold;
    memory.streams.threshold_initialized = buffers.threshold_initialized;
    memory.streams.current_threshold_active_index = buffers.active_index;
}

void free_benchmark_threshold_buffers(BenchmarkThresholdBuffers& buffers) {
    cudaFree(buffers.current_threshold);
    cudaFree(buffers.threshold_initialized);
    cudaFree(buffers.active_index);
    buffers = {};
}

std::uint64_t parse_u64(const char* text, const char* name) {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid numeric argument: ") + name);
    }
    return static_cast<std::uint64_t>(value);
}

std::uint32_t parse_next_u32(const std::string& text, std::size_t& pos, const char* context) {
    while (pos < text.size() && (text[pos] < '0' || text[pos] > '9')) {
        ++pos;
    }
    if (pos >= text.size()) {
        throw std::runtime_error(std::string("missing integer in ") + context);
    }
    std::uint64_t value = 0;
    while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
        value = value * 10ULL + static_cast<std::uint64_t>(text[pos] - '0');
        if (value > std::numeric_limits<std::uint8_t>::max()) {
            throw std::runtime_error(std::string("integer out of uint8 range in ") + context);
        }
        ++pos;
    }
    return static_cast<std::uint32_t>(value);
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open required text file: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

std::vector<std::uint8_t> load_p900_generators(const std::filesystem::path& path) {
    const std::string text = read_text_file(path);
    std::size_t pos = text.find("\"actions\"");
    if (pos == std::string::npos) {
        throw std::runtime_error("p900 generator json missing actions");
    }
    pos = text.find('[', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("p900 generator json malformed actions");
    }
    std::vector<std::uint8_t> generators(MOVE_COUNT * STATE_STORAGE_LEN);
    for (std::uint32_t move = 0; move < MOVE_COUNT; ++move) {
        pos = text.find('[', pos + 1);
        if (pos == std::string::npos) {
            throw std::runtime_error("p900 generator json missing move array");
        }
        for (std::uint32_t p = 0; p < STATE_LEN; ++p) {
            generators[move * STATE_STORAGE_LEN + p] =
                static_cast<std::uint8_t>(parse_next_u32(text, pos, "p900 generator"));
        }
        for (std::uint32_t p = STATE_LEN; p < STATE_STORAGE_LEN; ++p) {
            generators[move * STATE_STORAGE_LEN + p] = static_cast<std::uint8_t>(p);
        }
    }
    return generators;
}

State128 load_central_state(const std::filesystem::path& path) {
    const std::string text = read_text_file(path);
    std::size_t pos = text.find("\"central_state\"");
    if (pos == std::string::npos) {
        throw std::runtime_error("puzzle info json missing central_state");
    }
    pos = text.find('[', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("puzzle info json malformed central_state");
    }
    State128 state{};
    for (std::uint32_t p = 0; p < STATE_LEN; ++p) {
        state.v[p] = static_cast<std::uint8_t>(parse_next_u32(text, pos, "central_state"));
    }
    for (std::uint32_t p = STATE_LEN; p < STATE_STORAGE_LEN; ++p) {
        state.v[p] = 0;
    }
    return state;
}

State128 load_initial_state_from_test_csv(const std::filesystem::path& path, std::uint64_t puzzle_id) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open required csv file: " + path.string());
    }
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        const std::size_t comma = line.find(',');
        if (comma == std::string::npos) {
            continue;
        }
        const std::uint64_t row_id = parse_u64(line.substr(0, comma).c_str(), "initial_state_id");
        if (row_id != puzzle_id) {
            continue;
        }
        const std::size_t first_quote = line.find('"', comma);
        const std::size_t last_quote = line.rfind('"');
        if (first_quote == std::string::npos || last_quote == std::string::npos || last_quote <= first_quote) {
            throw std::runtime_error("test csv malformed initial_state row");
        }
        const std::string state_text = line.substr(first_quote + 1, last_quote - first_quote - 1);
        std::size_t pos = 0;
        State128 state{};
        for (std::uint32_t p = 0; p < STATE_LEN; ++p) {
            state.v[p] = static_cast<std::uint8_t>(parse_next_u32(state_text, pos, "initial_state"));
        }
        for (std::uint32_t p = STATE_LEN; p < STATE_STORAGE_LEN; ++p) {
            state.v[p] = 0;
        }
        return state;
    }
    throw std::runtime_error("requested puzzle_id not found in test csv");
}

void require_aligned(const void* ptr, std::uintptr_t alignment, const char* name) {
    if (reinterpret_cast<std::uintptr_t>(ptr) % alignment != 0) {
        throw std::runtime_error(std::string("device pointer alignment failed: ") + name);
    }
}

template <typename T>
T* device_alloc(std::uint64_t count) {
    T* ptr = nullptr;
    BEAM_CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

struct Timer {
    cudaStream_t control = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    std::vector<cudaEvent_t> done;

    explicit Timer(std::uint32_t stream_count) {
        BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&control, cudaStreamNonBlocking));
        BEAM_CUDA_CHECK(cudaEventCreate(&start));
        BEAM_CUDA_CHECK(cudaEventCreate(&stop));
        done.resize(stream_count);
        for (cudaEvent_t& event : done) {
            BEAM_CUDA_CHECK(cudaEventCreate(&event));
        }
    }

    ~Timer() {
        for (cudaEvent_t event : done) {
            cudaEventDestroy(event);
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(control);
    }
};

template <typename Fn>
float time_gpu_ms(const std::vector<cudaStream_t>& streams, std::uint32_t iterations, Fn launch) {
    Timer timer(static_cast<std::uint32_t>(streams.size()));
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    launch();
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    BEAM_CUDA_CHECK(cudaEventRecord(timer.start, timer.control));
    for (cudaStream_t stream : streams) {
        BEAM_CUDA_CHECK(cudaStreamWaitEvent(stream, timer.start, 0));
    }
    for (std::uint32_t i = 0; i < iterations; ++i) {
        launch();
    }
    for (std::uint32_t i = 0; i < streams.size(); ++i) {
        BEAM_CUDA_CHECK(cudaEventRecord(timer.done[i], streams[i]));
        BEAM_CUDA_CHECK(cudaStreamWaitEvent(timer.control, timer.done[i], 0));
    }
    BEAM_CUDA_CHECK(cudaEventRecord(timer.stop, timer.control));
    BEAM_CUDA_CHECK(cudaEventSynchronize(timer.stop));
    float elapsed_ms = 0.0f;
    BEAM_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer.start, timer.stop));
    return elapsed_ms / static_cast<float>(iterations);
}

template <typename Fn>
float time_single_stream_ms(cudaStream_t stream, Fn launch) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    BEAM_CUDA_CHECK(cudaEventCreate(&start));
    BEAM_CUDA_CHECK(cudaEventCreate(&stop));
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    BEAM_CUDA_CHECK(cudaEventRecord(start, stream));
    launch();
    BEAM_CUDA_CHECK(cudaEventRecord(stop, stream));
    BEAM_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    BEAM_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsed_ms;
}

std::vector<cudaStream_t> create_streams(std::uint32_t count) {
    std::vector<cudaStream_t> streams(count);
    for (cudaStream_t& stream : streams) {
        BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    return streams;
}

void destroy_streams(std::vector<cudaStream_t>& streams) {
    for (cudaStream_t stream : streams) {
        cudaStreamDestroy(stream);
    }
    streams.clear();
}

struct Stream1Result {
    std::uint32_t b_micro = 0;
    std::uint32_t concurrent = 0;
    float ms = 0.0f;
    double parent_per_sec = 0.0;
    double candidate_per_sec = 0.0;
};

struct StreamResult {
    std::string name;
    std::uint32_t param_a = 0;
    std::uint32_t param_b = 0;
    float ms = 0.0f;
    double candidate_per_sec = 0.0;
};

std::vector<State128> make_state_batch(
    State128 seed,
    std::uint32_t count,
    std::uint32_t num_classes) {
    std::vector<State128> states(count, seed);
    for (std::uint32_t i = 0; i < count; ++i) {
        states[i].v[0] = static_cast<std::uint8_t>((states[i].v[0] + i) % num_classes);
        states[i].v[STATE_LEN] = 0;
    }
    return states;
}

std::vector<Stream1Result> benchmark_stream1(
    const stream1_weights::DeviceWeights& weights,
    const Stream1ModelConfig& model,
    const State128* states,
    const std::uint8_t* generators,
    std::uint32_t max_states,
    std::ofstream& report) {
    std::vector<Stream1Result> results;
    const Stream1NetworkDims dims = stream1_weights::network_dims(model);
    const std::vector<const half*> residual_fc1_weight =
        stream1_weights::const_pointer_vector(weights.residual_fc1_weight);
    const std::vector<const half*> residual_fc1_bias =
        stream1_weights::const_pointer_vector(weights.residual_fc1_bias);
    const std::vector<const half*> residual_fc2_weight =
        stream1_weights::const_pointer_vector(weights.residual_fc2_weight);
    const std::vector<const half*> residual_fc2_bias =
        stream1_weights::const_pointer_vector(weights.residual_fc2_bias);
    Stream1NetworkView network{
        weights.input_weight,
        weights.input_bias,
        weights.hidden_weight,
        weights.hidden_bias,
        residual_fc1_weight.data(),
        residual_fc1_bias.data(),
        residual_fc2_weight.data(),
        residual_fc2_bias.data(),
        weights.output_weight,
        weights.output_bias,
        dims};

    report << "## Stream1 TensorOp CUTLASS\n\n";
    report << "| B_MICRO | concurrent_inference | ms_per_launch_group | parents_per_sec | candidates_per_sec |\n";
    report << "|---:|---:|---:|---:|---:|\n";
    for (std::uint32_t b_micro : B_MICRO_SWEEP) {
        const std::uint32_t parent_batch = stream1_parent_batch_from_row_budget(b_micro, model);
        for (std::uint32_t concurrent : STREAM1_CONCURRENCY_SWEEP) {
            if (parent_batch * concurrent > max_states) {
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
            const double parents = static_cast<double>(parent_batch) * concurrent;
            const double parent_per_sec = parents * 1000.0 / static_cast<double>(ms);
            const double candidate_per_sec = parents * static_cast<double>(MOVE_COUNT) * 1000.0 / static_cast<double>(ms);
            results.push_back(Stream1Result{b_micro, concurrent, ms, parent_per_sec, candidate_per_sec});
            report << "|" << b_micro
                   << "|" << concurrent
                   << "|" << std::fixed << std::setprecision(4) << ms
                   << "|" << std::setprecision(1) << parent_per_sec
                   << "|" << candidate_per_sec << "|\n";
            std::cout << "stream1_micro"
                      << " b_micro=" << b_micro
                      << " concurrent=" << concurrent
                      << " ms_per_launch_group=" << std::fixed << std::setprecision(4) << ms
                      << " parents_per_sec=" << std::setprecision(1) << parent_per_sec
                      << " candidates_per_sec=" << candidate_per_sec
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

} // namespace

int main(int argc, char** argv) {
    if (argc != 1 && argc != 2) {
        std::cerr << "usage: stream_benchmark [puzzle_id]\n";
        return 2;
    }
    std::cout << std::unitbuf;
    const std::uint64_t puzzle_id = argc == 2 ? parse_u64(argv[1], "puzzle_id") : 0;
    BEAM_CUDA_CHECK(cudaSetDevice(0));

    std::size_t free_before = 0;
    std::size_t total_before = 0;
    BEAM_CUDA_CHECK(cudaMemGetInfo(&free_before, &total_before));

    const std::filesystem::path generator_path = "FullBeamNice/generators/p900.json";
    const std::filesystem::path puzzle_info_path = "data/puzzle_info.json";
    const std::filesystem::path test_csv_path = "data/test.csv";
    const char* weight_dir_env = std::getenv("BEAM_WEIGHT_DIR");
    const std::filesystem::path weight_dir =
        weight_dir_env != nullptr && weight_dir_env[0] != '\0'
            ? std::filesystem::path(weight_dir_env)
            : std::filesystem::path("build-docker/stream1_weights");
    const std::vector<std::uint8_t> host_generators = load_p900_generators(generator_path);
    const State128 host_central = load_central_state(puzzle_info_path);
    const State128 host_initial = load_initial_state_from_test_csv(test_csv_path, puzzle_id);
    const ZobristTable host_zobrist = make_deterministic_zobrist(0xC0DEC0DEULL);
    const stream1_weights::HostWeightBytes host_weights =
        stream1_weights::load_stream1_weights(weight_dir);
    const Stream1ModelConfig& stream1_model = host_weights.model;
    if (stream1_model.backend == STREAM1_BACKEND_PIECE_TRANSFORMER) {
        throw std::runtime_error("piece_transformer Stream1 forward is not wired yet");
    }

    const std::uint32_t max_states =
        stream1_parent_batch_from_row_budget(B_MICRO_SWEEP.back(), stream1_model) *
        STREAM1_CONCURRENCY_SWEEP.back();
    const std::vector<State128> host_states = make_state_batch(host_initial, max_states, stream1_model.num_classes);
    State128* d_states = device_alloc<State128>(max_states);
    std::uint8_t* d_generators = device_alloc<std::uint8_t>(MOVE_COUNT * STATE_STORAGE_LEN);
    State128* d_central = device_alloc<State128>(1);
    Hash128* d_zobrist = device_alloc<Hash128>(STATE_STORAGE_LEN * STATE_VALUE_PAD);
    BEAM_CUDA_CHECK(cudaMemcpy(d_states, host_states.data(), host_states.size() * sizeof(State128), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_generators, host_generators.data(), host_generators.size(), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_central, &host_central, sizeof(State128), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_zobrist, &host_zobrist[0][0], STATE_STORAGE_LEN * STATE_VALUE_PAD * sizeof(Hash128), cudaMemcpyHostToDevice));
    require_aligned(d_states, alignof(State128), "states");
    require_aligned(d_generators, 16, "generators");
    require_aligned(d_central, alignof(State128), "central");
    require_aligned(d_zobrist, alignof(Hash128), "zobrist");
    stream1_weights::DeviceWeights weights = stream1_weights::upload_weights(host_weights);

    std::filesystem::create_directories("test_results");
    const char* report_env = std::getenv("BEAM_STREAM_BENCH_REPORT");
    const std::string report_path =
        report_env != nullptr ? std::string(report_env) : std::string("test_results/per_stream_benchmark_2026-05-24.md");
    const bool stream_micro_only = std::getenv("BEAM_STREAM_MICRO_ONLY") != nullptr;
    std::ofstream report(report_path);
    report << "# Per Stream Benchmark 2026-05-22\n\n";
    report << "- puzzle_id=" << puzzle_id << "\n";
    report << "- gpu_total_bytes=" << total_before << "\n";
    report << "- gpu_free_before_bytes=" << free_before << "\n";
    report << "- generator_path=" << generator_path.string() << "\n";
    report << "- puzzle_info_path=" << puzzle_info_path.string() << "\n";
    report << "- test_csv_path=" << test_csv_path.string() << "\n";
    report << "- weight_dir=" << weight_dir.string() << "\n";
    report << "- stream1_model_hidden1=" << stream1_model.hidden1 << "\n";
    report << "- stream1_model_hidden2=" << stream1_model.hidden2 << "\n";
    report << "- stream1_model_residual_count=" << stream1_model.residual_count << "\n";
    report << "- stream1_model_output_dim=" << stream1_model.output_dim << "\n";
    report << "- stream1_model_weight_bytes=" << stream1_weights::total_host_weight_bytes(host_weights) << "\n";
    report << "- cuda_architectures=75,86\n";
    report << "- stream1_gemm=TensorOp_Sm75_common_for_T4_and_RTX3070\n\n";
    std::cout << "stream_benchmark_start=1\n";
    benchmark_stream1(weights, stream1_model, d_states, d_generators, max_states, report);
    std::cout << "stream1_benchmark_done=1\n";
    if (!stream_micro_only) {
        benchmark_stream2(d_states, d_generators, d_central, d_zobrist, report);
        std::cout << "stream2_benchmark_done=1\n";
        benchmark_stream3(report);
        std::cout << "stream3_benchmark_done=1\n";
    }
    benchmark_stream4(report);
    std::cout << "stream4_benchmark_done=1\n";

    report << "## Status\n\n";
    report << "- status=pass\n";
    report.close();

    stream1_weights::free_weights(weights);
    cudaFree(d_states);
    cudaFree(d_generators);
    cudaFree(d_central);
    cudaFree(d_zobrist);
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "stream_benchmark_report=" << report_path << "\n";
    return 0;
}
