#include "../cuda/stream3.hpp"
#include "../src/hash.hpp"
#include <cuda_runtime.h>
#ifdef BEAM_TEST_OWNER_RADIX
#include <cub/device/device_radix_sort.cuh>
#endif
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace beam;
namespace {
void cu(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> T* allocate(std::size_t count) {
    T* pointer = nullptr;
    cu(cudaMalloc(&pointer, count * sizeof(T)));
    return pointer;
}
template<class T> void upload(T* device, const T* host, std::size_t count) {
    cu(cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice));
}
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
}

// One physical GPU simulates up to 256 owner domains. No NCCL or neural inference.
// Allocation occurs once before the cases; eager and captured public calls share it.
int main() {
    try {
        cu(cudaSetDevice(0));
        constexpr unsigned capacity = 513, b_micro = 64;
        cudaStream_t stream = nullptr;
        cu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        auto* key = allocate<Hash128>(capacity);
        auto* value = allocate<std::uint64_t>(capacity);
        auto* count_device = allocate<std::uint32_t>(1);
        auto* parent_device = allocate<std::uint64_t>(4);
        auto* local = allocate<CandidateMeta>(capacity);
        auto* local_count_device = allocate<std::uint32_t>(1);
        auto* remote = allocate<CandidateMeta>(capacity);
        auto* send_count_device = allocate<std::uint32_t>(256);
        auto* send_offset_device = allocate<std::uint32_t>(257);
        auto* owners_device = allocate<std::uint32_t>(capacity);
#ifdef BEAM_TEST_OWNER_RADIX
        auto* key_a = allocate<std::uint32_t>(capacity);
        auto* key_b = allocate<std::uint32_t>(capacity);
        auto* val_a = allocate<CandidateMeta>(capacity);
        auto* val_b = allocate<CandidateMeta>(capacity);
        auto* fatal_flag = allocate<std::uint32_t>(1);
        auto* fatal_trace = allocate<std::uint64_t>(STREAM_FATAL_TRACE_WORDS);
        cu(cudaMemset(fatal_flag, 0, sizeof(*fatal_flag)));
        cu(cudaMemset(fatal_trace, 0, STREAM_FATAL_TRACE_WORDS * sizeof(*fatal_trace)));
        std::size_t owner_temp_bytes = 0;
        cu(cub::DeviceRadixSort::SortPairs(nullptr, owner_temp_bytes, key_a, key_b,
                                            val_a, val_b, capacity, 0, 9, stream));
        void* owner_temp = nullptr; cu(cudaMalloc(&owner_temp, owner_temp_bytes));
        std::size_t required_for_two = 0;
        cu(cub::DeviceRadixSort::SortPairs(nullptr, required_for_two, key_a, key_b,
                                            val_a, val_b, capacity, 0, 2, stream));
        if (!required_for_two || required_for_two > owner_temp_bytes)
            throw std::runtime_error("owner radix scratch query exceeds allocated maximum");
        const std::uint32_t zero = 0; upload(count_device, &zero, 1);
        for (std::size_t available : {std::size_t{0}, required_for_two - 1U}) {
            bool rejected = false;
            try {
                stream3_restore_owner_radix_cuda(key, value, count_device, parent_device,
                    local, local_count_device, remote, send_count_device, send_offset_device,
                    key_a, key_b, val_a, val_b, owner_temp, available,
                    0, 2, b_micro, capacity, stream, fatal_flag, fatal_trace);
            } catch (const std::runtime_error& error) {
                const std::string message = error.what();
                rejected = message.find("required=" + std::to_string(required_for_two)) != std::string::npos &&
                           message.find("available=" + std::to_string(available)) != std::string::npos;
            }
            if (!rejected) throw std::runtime_error("undersized owner scratch did not report exact sizes");
        }
        (void)cudaGetLastError();
        cu(cudaStreamSynchronize(stream));
#endif
        const std::array<std::uint64_t, 4> parents{1000, 10000, 100000, 1000000};
        upload(parent_device, parents.data(), parents.size());
#ifdef BEAM_TEST_OWNER_RADIX
        // Deliberately corrupt the device count. Eager execution and graph replay
        // must report exact sizes, preserve the context, and emit no output.
        for (unsigned world : {1U, 2U}) {
            for (unsigned bad_count : {capacity + 1U, UINT32_MAX}) {
                for (bool captured : {false, true}) {
                    const unsigned zero = 0;
                    upload(fatal_flag, &zero, 1);
                    upload(count_device, &bad_count, 1);
                    cu(cudaMemset(fatal_trace, 0, STREAM_FATAL_TRACE_WORDS * sizeof(*fatal_trace)));
                    cu(cudaMemset(local, 0xa5, capacity * sizeof(*local)));
                    cu(cudaMemset(remote, 0xa5, capacity * sizeof(*remote)));
                    cu(cudaMemset(send_count_device, 0xff, world * sizeof(unsigned)));
                    cu(cudaMemset(send_offset_device, 0xff, (world + 1) * sizeof(unsigned)));
                    cu(cudaDeviceSynchronize());
                    auto enqueue_bad = [&] {
                        stream3_restore_owner_radix_cuda(key, value, count_device, parent_device,
                            local, local_count_device, remote, send_count_device, send_offset_device,
                            key_a, key_b, val_a, val_b, owner_temp, owner_temp_bytes,
                            0, world, b_micro, capacity, stream, fatal_flag, fatal_trace);
                    };
                    cudaGraph_t graph = nullptr; cudaGraphExec_t executable = nullptr;
                    if (captured) {
                        cu(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                        enqueue_bad();
                        cu(cudaStreamEndCapture(stream, &graph));
                        cu(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                        cu(cudaGraphLaunch(executable, stream));
                    } else enqueue_bad();
                    cu(cudaStreamSynchronize(stream));
                    unsigned flag = 0, local_count = UINT32_MAX;
                    std::array<std::uint64_t, STREAM_FATAL_TRACE_WORDS> trace{};
                    std::vector<unsigned> counts(world, UINT32_MAX), offsets(world + 1, UINT32_MAX);
                    std::array<unsigned char, sizeof(CandidateMeta)> local_canary{}, remote_canary{};
                    cu(cudaMemcpy(&flag, fatal_flag, sizeof(flag), cudaMemcpyDeviceToHost));
                    cu(cudaMemcpy(trace.data(), fatal_trace, sizeof(trace), cudaMemcpyDeviceToHost));
                    cu(cudaMemcpy(&local_count, local_count_device, sizeof(local_count), cudaMemcpyDeviceToHost));
                    cu(cudaMemcpy(counts.data(), send_count_device, world * sizeof(unsigned), cudaMemcpyDeviceToHost));
                    cu(cudaMemcpy(offsets.data(), send_offset_device, (world + 1) * sizeof(unsigned), cudaMemcpyDeviceToHost));
                    cu(cudaMemcpy(local_canary.data(), local, local_canary.size(), cudaMemcpyDeviceToHost));
                    cu(cudaMemcpy(remote_canary.data(), remote, remote_canary.size(), cudaMemcpyDeviceToHost));
                    require(flag == STREAM_FATAL_STREAM3_OWNER_COUNT_OVERFLOW, "owner count overflow code mismatch");
                    require(trace[FatalTraceCode] == flag && trace[FatalTraceRawCount] == bad_count &&
                            trace[FatalTraceAvailable] == capacity, "owner count overflow sizes mismatch");
                    require(local_count == 0 && std::all_of(counts.begin(), counts.end(), [](unsigned x) { return x == 0; }) &&
                            std::all_of(offsets.begin(), offsets.end(), [](unsigned x) { return x == 0; }),
                            "owner count overflow emitted nonzero counts");
                    require(std::all_of(local_canary.begin(), local_canary.end(), [](unsigned char x) { return x == 0xa5; }) &&
                            std::all_of(remote_canary.begin(), remote_canary.end(), [](unsigned char x) { return x == 0xa5; }),
                            "owner count overflow overwrote outputs");
                    if (captured) { cu(cudaGraphExecDestroy(executable)); cu(cudaGraphDestroy(graph)); }
                }
            }
        }
        // A prior failure must retain its trace even if the count is corrupt.
        const unsigned prior = STREAM_FATAL_STREAM3_DOUBLE_BUFFER_OVERFLOW;
        const unsigned bad_count = capacity + 1U;
        std::array<std::uint64_t, STREAM_FATAL_TRACE_WORDS> marker{};
        marker.fill(0x123456789abcdef0ULL);
        upload(fatal_flag, &prior, 1); upload(fatal_trace, marker.data(), marker.size());
        upload(count_device, &bad_count, 1);
        stream3_restore_owner_radix_cuda(key, value, count_device, parent_device,
            local, local_count_device, remote, send_count_device, send_offset_device,
            key_a, key_b, val_a, val_b, owner_temp, owner_temp_bytes,
            0, 2, b_micro, capacity, stream, fatal_flag, fatal_trace);
        cu(cudaStreamSynchronize(stream));
        unsigned observed_prior = 0;
        std::array<std::uint64_t, STREAM_FATAL_TRACE_WORDS> observed_marker{};
        cu(cudaMemcpy(&observed_prior, fatal_flag, sizeof(observed_prior), cudaMemcpyDeviceToHost));
        cu(cudaMemcpy(observed_marker.data(), fatal_trace, sizeof(observed_marker), cudaMemcpyDeviceToHost));
        require(observed_prior == prior && observed_marker == marker, "earlier fatal trace was overwritten");
        cu(cudaMemset(fatal_flag, 0, sizeof(*fatal_flag)));
        std::cout << "owner_count_overflow eager_and_graph=PASS prior_error_preserved=PASS\n";
#endif
        unsigned checks = 0;
        for (unsigned world : {1U, 2U, 3U, 5U, 8U, 256U}) {
            for (unsigned rank = 0; rank < world; ++rank) {
                if (world == 256U && rank != 0U && rank != 127U && rank != 255U) continue;
                for (unsigned count : {0U, 1U, 2U, 255U, 256U, 257U, 513U}) {
                    for (unsigned pattern = 0; pattern < 3; ++pattern) {
                        std::vector<Hash128> keys(count);
                        std::vector<std::uint64_t> values(count);
                        std::vector<std::vector<CandidateMeta>> expected(world);
                        std::uint64_t nonce = 1;
                        const unsigned per_slot = b_micro * MOVE_COUNT;
                        for (unsigned i = 0; i < count; ++i) {
                            // Mixed starts remote then local; also all-local/all-remote.
                            const unsigned owner = pattern == 1 ? rank : pattern == 2 ?
                                (rank + 1) % world : (rank + 1 + i) % world;
                            Hash128 hash{};
                            do { hash = Hash128{nonce++, 0x8123456789abcdefULL}; }
                            while (hash128_owner_distribution_key(hash) % world != owner);
                            keys[i] = hash;
                            const unsigned payload = (i * 37U + 11U) % (4 * per_slot);
                            const unsigned score = i % 13;
                            values[i] = (std::uint64_t(score) << 32U) | payload;
                            const unsigned slot = payload / per_slot;
                            const unsigned within = payload % per_slot;
                            expected[owner].push_back(CandidateMeta{hash,
                                parents[slot] + within / MOVE_COUNT, score,
                                static_cast<std::uint32_t>((rank << 16U) | (owner << 8U) | (within % MOVE_COUNT))});
                        }
                        if (count) {
                            upload(key, keys.data(), count);
                            upload(value, values.data(), count);
                        }
                        upload(count_device, &count, 1);
                        auto enqueue = [&] {
#ifdef BEAM_TEST_OWNER_RADIX
                            stream3_restore_owner_radix_cuda(key, value, count_device,
                                parent_device, local, local_count_device, remote,
                                send_count_device, send_offset_device, key_a, key_b,
                                val_a, val_b, owner_temp, owner_temp_bytes,
                                static_cast<std::uint16_t>(rank), world, b_micro,
                                capacity, stream, fatal_flag, fatal_trace);
#else
                            stream3_restore_owner_split_cuda(key, value, count_device,
                                parent_device, local, local_count_device, remote,
                                send_count_device, send_offset_device, owners_device,
                                static_cast<std::uint16_t>(rank), world, b_micro,
                                capacity, stream);
#endif
                            cu(cudaPeekAtLastError());
                        };
                        auto verify = [&] {
                            cu(cudaStreamSynchronize(stream));
                            unsigned local_count = 0;
                            std::vector<unsigned> counts(world), offsets(world + 1);
                            cu(cudaMemcpy(&local_count, local_count_device, sizeof(unsigned), cudaMemcpyDeviceToHost));
                            cu(cudaMemcpy(counts.data(), send_count_device, world * sizeof(unsigned), cudaMemcpyDeviceToHost));
                            cu(cudaMemcpy(offsets.data(), send_offset_device, (world + 1) * sizeof(unsigned), cudaMemcpyDeviceToHost));
                            require(local_count == expected[rank].size(), "local count mismatch");
                            require(offsets[0] == 0, "offset origin mismatch");
                            for (unsigned peer = 0; peer < world; ++peer) {
                                const auto wanted = peer == rank ? 0U : expected[peer].size();
                                require(counts[peer] == wanted, "remote count mismatch");
                                require(offsets[peer + 1] == offsets[peer] + counts[peer], "offset mismatch");
                                std::vector<CandidateMeta> actual(expected[peer].size());
                                if (!actual.empty()) cu(cudaMemcpy(actual.data(), peer == rank ? local : remote + offsets[peer],
                                    actual.size() * sizeof(CandidateMeta), cudaMemcpyDeviceToHost));
                                for (std::size_t i = 0; i < actual.size(); ++i)
                                    require(std::memcmp(&actual[i], &expected[peer][i], sizeof(CandidateMeta)) == 0,
                                        "hash/score/parent/route/order mismatch");
                            }
                            ++checks;
                        };
                        enqueue(); verify();
                        cudaGraph_t graph = nullptr;
                        cudaGraphExec_t executable = nullptr;
                        cu(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                        enqueue();
                        cu(cudaStreamEndCapture(stream, &graph));
                        cu(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                        for (unsigned repeat = 0; repeat < 3; ++repeat) {
                            cu(cudaGraphLaunch(executable, stream)); verify();
                        }
                        cu(cudaGraphExecDestroy(executable));
                        cu(cudaGraphDestroy(graph));
                    }
                }
            }
        }
        cu(cudaFree(key)); cu(cudaFree(value)); cu(cudaFree(count_device));
        cu(cudaFree(parent_device)); cu(cudaFree(local)); cu(cudaFree(local_count_device));
        cu(cudaFree(remote)); cu(cudaFree(send_count_device)); cu(cudaFree(send_offset_device));
        cu(cudaFree(owners_device)); cu(cudaStreamDestroy(stream));
#ifdef BEAM_TEST_OWNER_RADIX
        cu(cudaFree(owner_temp)); cu(cudaFree(key_a)); cu(cudaFree(key_b));
        cu(cudaFree(val_a)); cu(cudaFree(val_b));
        cu(cudaFree(fatal_flag)); cu(cudaFree(fatal_trace));
#endif
        std::cout << "owner_scatter checks=" << checks << " eager_and_graph=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "owner_scatter FAIL " << error.what() << '\n';
        return 1;
    }
}
