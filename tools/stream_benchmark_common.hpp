#pragma once

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

namespace beam::bench {

inline constexpr std::array<std::uint32_t, 6> B_MICRO_SWEEP{2048, 4096, 8192, 16384, 32768, 65536};
inline constexpr std::array<std::uint32_t, 6> STREAM1_CONCURRENCY_SWEEP{1, 2, 4, 8, 16, 32};
inline constexpr std::array<std::uint32_t, 18> TRANSFORMER_B_MICRO_SWEEP{128, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 1536, 2048, 3072, 4096, 8192, 12288, 16384, 24576};
inline constexpr std::array<std::uint32_t, 8> TRANSFORMER_STREAM1_CONCURRENCY_SWEEP{1, 2, 3, 4, 5, 6, 8, 12};
inline constexpr std::array<std::uint32_t, 6> STREAM4_BATCH_SWEEP{196608, 262144, 393216, 524288, 699392, 1048576};
inline constexpr std::array<std::uint32_t, 4> STREAM4_SHARD_CAPACITY_SWEEP{1048576, 2621440, 5242880, 10485760};

inline void publish_threshold_for_benchmark(StaticDeviceMemory& memory, std::uint32_t threshold) {
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
    std::uint32_t* request_local = nullptr;
    std::uint32_t* request_global = nullptr;
};

inline BenchmarkThresholdBuffers alloc_benchmark_threshold_buffers() {
    BenchmarkThresholdBuffers buffers;
    BEAM_CUDA_CHECK(cudaMalloc(&buffers.current_threshold, 2ULL * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&buffers.threshold_initialized, 2ULL * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&buffers.active_index, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&buffers.request_local, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&buffers.request_global, sizeof(std::uint32_t)));
    return buffers;
}

inline void attach_benchmark_threshold_buffers(StaticDeviceMemory& memory, const BenchmarkThresholdBuffers& buffers) {
    memory.streams.current_threshold = buffers.current_threshold;
    memory.streams.threshold_initialized = buffers.threshold_initialized;
    memory.streams.current_threshold_active_index = buffers.active_index;
    memory.streams.threshold_request_local = buffers.request_local;
    memory.streams.threshold_request_global = buffers.request_global;
}

inline void free_benchmark_threshold_buffers(BenchmarkThresholdBuffers& buffers) {
    cudaFree(buffers.current_threshold);
    cudaFree(buffers.threshold_initialized);
    cudaFree(buffers.active_index);
    cudaFree(buffers.request_local);
    cudaFree(buffers.request_global);
    buffers = {};
}

inline std::uint64_t parse_u64(const char* text, const char* name) {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid numeric argument: ") + name);
    }
    return static_cast<std::uint64_t>(value);
}

inline std::uint32_t parse_next_u32(const std::string& text, std::size_t& pos, const char* context) {
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

inline std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open required text file: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

inline std::vector<std::uint8_t> load_p900_generators(const std::filesystem::path& path) {
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

inline State128 load_central_state(const std::filesystem::path& path) {
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

inline State128 load_initial_state_from_test_csv(const std::filesystem::path& path, std::uint64_t puzzle_id) {
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

inline void require_aligned(const void* ptr, std::uintptr_t alignment, const char* name) {
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

inline std::vector<cudaStream_t> create_streams(std::uint32_t count) {
    std::vector<cudaStream_t> streams(count);
    for (cudaStream_t& stream : streams) {
        BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    return streams;
}

inline void destroy_streams(std::vector<cudaStream_t>& streams) {
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

inline std::vector<State128> make_state_batch(
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

inline std::vector<State128> make_synthetic_state_batch(
    std::uint32_t count,
    std::uint32_t state_len,
    std::uint32_t num_classes) {
    if (state_len > STATE_LEN) {
        throw std::runtime_error("synthetic state_len exceeds logical State128 bytes");
    }
    std::vector<State128> states(count);
    for (std::uint32_t row = 0; row < count; ++row) {
        for (std::uint32_t p = 0; p < STATE_STORAGE_LEN; ++p) {
            states[row].v[p] = 0;
        }
        for (std::uint32_t p = 0; p < state_len; ++p) {
            const std::uint64_t value =
                (static_cast<std::uint64_t>(row) * state_len + p) * 17ULL + 23ULL;
            states[row].v[p] = static_cast<std::uint8_t>(value % num_classes);
        }
    }
    return states;
}


std::vector<Stream1Result> benchmark_stream1_mlp(
    const stream1_weights::DeviceWeights& weights,
    const Stream1ModelConfig& model,
    const State128* states,
    const std::uint8_t* generators,
    std::uint32_t max_states,
    std::ofstream& report);

std::vector<Stream1Result> benchmark_stream1_transformer(
    const stream1_weights::DeviceWeights& weights,
    const Stream1ModelConfig& model,
    const State128* states,
    std::uint32_t max_states,
    std::ofstream& report);

std::vector<StreamResult> benchmark_stream2(
    const State128* states,
    const std::uint8_t* generators,
    const State128* central,
    const Hash128* zobrist,
    std::ofstream& report);

std::vector<StreamResult> benchmark_stream3(std::ofstream& report);

std::vector<StreamResult> benchmark_stream4(std::ofstream& report);

} // namespace beam::bench