#include "final_materialize.hpp"

#include "config.hpp"
#include "nvtx_ranges.hpp"

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#ifndef BEAM_DEBUG_FINAL_VALIDATE
#define BEAM_DEBUG_FINAL_VALIDATE 0
#endif

namespace beam {

namespace {

#if BEAM_DEBUG_FINAL_VALIDATE
void check_final_materialize_cuda(cudaError_t status, const char* op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
    }
}

__global__ void validate_final_requests_kernel(
    const FinalRequest* requests,
    FinalRequestValidationError* error,
    std::uint32_t request_count,
    std::uint64_t current_frontier_size,
    std::uint32_t target_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    std::uint32_t invalid_count = 0;
    for (std::uint32_t i = 0; i < request_count; ++i) {
        const FinalRequest request = requests[i];
        std::uint32_t reason = 0;
        if (request.parent_idx >= current_frontier_size) {
            reason |= FinalRequestInvalidParent;
        }
        if (request.target_local_idx >= target_count) {
            reason |= FinalRequestInvalidTarget;
        }
        if (static_cast<std::uint32_t>(request.move) >= MOVE_COUNT) {
            reason |= FinalRequestInvalidMove;
        }
        if (request.target_local_idx != i) {
            reason |= FinalRequestInvalidSlot;
        }
        if (reason == 0U) {
            continue;
        }
        if (invalid_count == 0U) {
            error->first_index = i;
            error->parent_idx = request.parent_idx;
            error->target_local_idx = request.target_local_idx;
            error->request_count = request_count;
            error->current_frontier_size = current_frontier_size;
            error->target_count = target_count;
            error->reason_mask = reason;
            error->move = static_cast<std::uint32_t>(request.move);
        }
        ++invalid_count;
    }
    error->invalid_count = invalid_count;
}
#endif

} // namespace

__device__ void final_response_set_target_local_idx_device(FinalResponse& response, std::uint32_t target_local_idx) {
    response.v[120] = static_cast<std::uint8_t>(target_local_idx);
    response.v[121] = static_cast<std::uint8_t>(target_local_idx >> 8);
    response.v[122] = static_cast<std::uint8_t>(target_local_idx >> 16);
    response.v[123] = static_cast<std::uint8_t>(target_local_idx >> 24);
}

__device__ std::uint32_t final_response_get_target_local_idx_device(const FinalResponse& response) {
    return static_cast<std::uint32_t>(response.v[120]) |
           (static_cast<std::uint32_t>(response.v[121]) << 8) |
           (static_cast<std::uint32_t>(response.v[122]) << 16) |
           (static_cast<std::uint32_t>(response.v[123]) << 24);
}

__device__ void clear_state_padding_device(State128& state) {
    for (std::uint32_t p = STATE_LEN; p < STATE_STORAGE_LEN; ++p) {
        state.v[p] = 0;
    }
}

__device__ std::uint64_t ceil_div_u64_final_device(std::uint64_t a, std::uint64_t b) {
    return b == 0ULL ? 0ULL : (a + b - 1ULL) / b;
}

__device__ std::uint32_t target_rank_for_global_final_device(
    std::uint64_t global_idx,
    std::uint64_t global_count,
    std::uint32_t world_size) {
    std::uint32_t target =
        static_cast<std::uint32_t>((global_idx * static_cast<std::uint64_t>(world_size)) / global_count);
    if (target >= world_size) {
        target = world_size - 1U;
    }
    return target;
}

__device__ std::uint16_t unpack_source_rank_final_device(std::uint32_t route_packed) {
    return static_cast<std::uint16_t>(route_packed >> 16);
}

__device__ std::uint8_t unpack_move_final_device(std::uint32_t route_packed) {
    return static_cast<std::uint8_t>(route_packed & 0xffU);
}

__global__ void final_build_materialize_chunk_kernel(
    const CandidateMeta* candidates,
    std::uint32_t chunk_begin,
    std::uint32_t chunk_count,
    std::uint32_t local_less_count,
    std::uint64_t less_prefix_for_rank,
    std::uint64_t global_less,
    std::uint64_t equal_prefix_for_rank,
    std::uint64_t global_keep_count,
    std::uint32_t world_size,
    std::uint32_t* source_rank_keys,
    FinalRequest* requests,
    FinalHistoryRecord* history_records) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= chunk_count || global_keep_count == 0ULL || world_size == 0U) {
        return;
    }
    const std::uint32_t local_out = chunk_begin + i;
    const CandidateMeta candidate = candidates[local_out];
    const bool less_phase = local_out < local_less_count;
    const std::uint64_t phase_local =
        less_phase ? static_cast<std::uint64_t>(local_out) :
                     static_cast<std::uint64_t>(local_out - local_less_count);
    const std::uint64_t global_idx =
        less_phase ? less_prefix_for_rank + phase_local : global_less + equal_prefix_for_rank + phase_local;
    const std::uint32_t target_rank =
        target_rank_for_global_final_device(global_idx, global_keep_count, world_size);
    const std::uint64_t target_begin =
        ceil_div_u64_final_device(static_cast<std::uint64_t>(target_rank) * global_keep_count, world_size);
    const std::uint32_t target_local_idx = static_cast<std::uint32_t>(global_idx - target_begin);
    const std::uint32_t source_rank = unpack_source_rank_final_device(candidate.route_packed);

    source_rank_keys[i] = source_rank;
    FinalRequest request{};
    request.parent_idx = candidate.parent_idx;
    request.target_local_idx = target_local_idx;
    request.return_rank = static_cast<std::uint16_t>(target_rank);
    request.move = unpack_move_final_device(candidate.route_packed);
    requests[i] = request;
    history_records[i] = FinalHistoryRecord{candidate, target_local_idx, 0U, {0ULL, 0ULL, 0ULL}};
}

__device__ std::uint32_t lower_bound_rank_key_device(
    const std::uint32_t* sorted_keys,
    std::uint32_t item_count,
    std::uint32_t key) {
    std::uint32_t first = 0;
    std::uint32_t count = item_count;
    while (count != 0U) {
        const std::uint32_t step = count >> 1U;
        const std::uint32_t mid = first + step;
        if (sorted_keys[mid] < key) {
            first = mid + 1U;
            count -= step + 1U;
        } else {
            count = step;
        }
    }
    return first;
}

__global__ void final_count_sorted_rank_keys_kernel(
    const std::uint32_t* sorted_keys,
    std::uint32_t item_count,
    std::uint32_t* counts,
    std::uint32_t* offsets,
    std::uint32_t world_size) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    std::uint32_t running = 0;
    offsets[0] = 0;
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
        const std::uint32_t begin = lower_bound_rank_key_device(sorted_keys, item_count, rank);
        const std::uint32_t end = lower_bound_rank_key_device(sorted_keys, item_count, rank + 1U);
        const std::uint32_t count = end - begin;
        counts[rank] = count;
        running += count;
        offsets[rank + 1U] = running;
    }
}

__global__ void final_build_return_rank_keys_kernel(
    const FinalRequest* requests,
    std::uint32_t* return_rank_keys,
    std::uint32_t request_count) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= request_count) {
        return;
    }
    return_rank_keys[i] = static_cast<std::uint32_t>(requests[i].return_rank);
}

__global__ void final_scatter_history_records_kernel(
    const FinalHistoryRecord* records,
    CandidateMeta* history_candidates,
    std::uint32_t record_count) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= record_count) {
        return;
    }
    const FinalHistoryRecord record = records[i];
    history_candidates[record.target_local_idx] = record.meta;
}

__global__ void final_materialize_kernel(
    const State128* current_frontier_states,
    const FinalRequest* requests,
    const std::uint8_t* generators,
    FinalResponse* responses,
    State128* next_frontier_states_tmp,
    std::uint32_t request_count) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= request_count) {
        return;
    }
    const FinalRequest request = requests[i];
    const State128 parent = current_frontier_states[request.parent_idx];
    FinalResponse response{};
    for (std::uint32_t p = 0; p < STATE_STORAGE_LEN; ++p) {
        const std::uint8_t source = generators[static_cast<std::uint32_t>(request.move) * STATE_STORAGE_LEN + p];
        response.v[p] = parent.v[source];
    }
    final_response_set_target_local_idx_device(response, request.target_local_idx);
    if (responses != nullptr) {
        responses[i] = response;
    }

    const std::uint32_t target_local_idx = final_response_get_target_local_idx_device(response);
    clear_state_padding_device(response);
    next_frontier_states_tmp[target_local_idx] = response;
}

__global__ void final_materialize_responses_kernel(
    const State128* current_frontier_states,
    const FinalRequest* requests,
    const std::uint8_t* generators,
    FinalResponse* responses,
    std::uint32_t request_count) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= request_count) {
        return;
    }
    const FinalRequest request = requests[i];
    const State128 parent = current_frontier_states[request.parent_idx];
    FinalResponse response{};
    for (std::uint32_t p = 0; p < STATE_STORAGE_LEN; ++p) {
        const std::uint8_t source = generators[static_cast<std::uint32_t>(request.move) * STATE_STORAGE_LEN + p];
        response.v[p] = parent.v[source];
    }
    final_response_set_target_local_idx_device(response, request.target_local_idx);
    responses[i] = response;
}

__global__ void final_scatter_responses_kernel(
    const FinalResponse* responses,
    State128* next_frontier_states_tmp,
    std::uint32_t response_count) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= response_count) {
        return;
    }
    State128 state = responses[i];
    const std::uint32_t target_local_idx = final_response_get_target_local_idx_device(state);
    clear_state_padding_device(state);
    next_frontier_states_tmp[target_local_idx] = state;
}

void final_materialize_cuda(
    const State128* current_frontier_states,
    const FinalRequest* requests,
    const std::uint8_t* generators,
    FinalResponse* responses,
    State128* next_frontier_states_tmp,
    std::uint32_t request_count,
    cudaStream_t stream) {
    NvtxRange range("Final_materialize_launch");
    const dim3 block(128);
    const dim3 grid((request_count + block.x - 1) / block.x);
    final_materialize_kernel<<<grid, block, 0, stream>>>(
        current_frontier_states,
        requests,
        generators,
        responses,
        next_frontier_states_tmp,
        request_count);
}

void final_materialize_responses_cuda(
    const State128* current_frontier_states,
    const FinalRequest* requests,
    const std::uint8_t* generators,
    FinalResponse* responses,
    std::uint32_t request_count,
    cudaStream_t stream) {
    NvtxRange range("Final_materialize_responses_launch");
    const dim3 block(128);
    const dim3 grid((request_count + block.x - 1) / block.x);
    if (grid.x != 0U) {
        final_materialize_responses_kernel<<<grid, block, 0, stream>>>(
            current_frontier_states,
            requests,
            generators,
            responses,
            request_count);
    }
}

void final_scatter_responses_cuda(
    const FinalResponse* responses,
    State128* next_frontier_states_tmp,
    std::uint32_t response_count,
    cudaStream_t stream) {
    NvtxRange range("Final_scatter_responses_launch");
    const dim3 block(128);
    const dim3 grid((response_count + block.x - 1) / block.x);
    if (grid.x != 0U) {
        final_scatter_responses_kernel<<<grid, block, 0, stream>>>(
            responses,
            next_frontier_states_tmp,
            response_count);
    }
}

void final_build_materialize_chunk_cuda(
    const CandidateMeta* candidates,
    std::uint32_t chunk_begin,
    std::uint32_t chunk_count,
    std::uint32_t local_less_count,
    std::uint64_t less_prefix_for_rank,
    std::uint64_t global_less,
    std::uint64_t equal_prefix_for_rank,
    std::uint64_t global_keep_count,
    std::uint32_t local_rank,
    std::uint32_t world_size,
    std::uint32_t* source_rank_keys,
    FinalRequest* requests,
    FinalHistoryRecord* history_records,
    cudaStream_t stream) {
    (void)local_rank;
    NvtxRange range("Final_build_materialize_chunk_launch");
    const dim3 block(128);
    const dim3 grid((chunk_count + block.x - 1) / block.x);
    if (grid.x != 0U) {
        final_build_materialize_chunk_kernel<<<grid, block, 0, stream>>>(
            candidates,
            chunk_begin,
            chunk_count,
            local_less_count,
            less_prefix_for_rank,
            global_less,
            equal_prefix_for_rank,
            global_keep_count,
            world_size,
            source_rank_keys,
            requests,
            history_records);
    }
}

void final_sort_requests_by_key_cuda(
    const std::uint32_t* keys_in,
    std::uint32_t* keys_out,
    const FinalRequest* requests_in,
    FinalRequest* requests_out,
    std::uint32_t request_count,
    void* cub_temp,
    std::size_t cub_temp_bytes,
    cudaStream_t stream) {
    NvtxRange range("Final_sort_requests_by_key_launch");
    if (request_count == 0U) {
        return;
    }
    cudaError_t status = cub::DeviceRadixSort::SortPairs(
        cub_temp,
        cub_temp_bytes,
        keys_in,
        keys_out,
        requests_in,
        requests_out,
        static_cast<int>(request_count),
        0,
        32,
        stream);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("final request radix sort: ") + cudaGetErrorString(status));
    }
}

void final_count_sorted_rank_keys_cuda(
    const std::uint32_t* sorted_keys,
    std::uint32_t item_count,
    std::uint32_t* counts,
    std::uint32_t* offsets,
    std::uint32_t world_size,
    cudaStream_t stream) {
    NvtxRange range("Final_count_sorted_rank_keys_launch");
    final_count_sorted_rank_keys_kernel<<<1, 1, 0, stream>>>(sorted_keys, item_count, counts, offsets, world_size);
}

void final_build_return_rank_keys_cuda(
    const FinalRequest* requests,
    std::uint32_t* return_rank_keys,
    std::uint32_t request_count,
    cudaStream_t stream) {
    NvtxRange range("Final_build_return_rank_keys_launch");
    const dim3 block(128);
    const dim3 grid((request_count + block.x - 1) / block.x);
    if (grid.x != 0U) {
        final_build_return_rank_keys_kernel<<<grid, block, 0, stream>>>(requests, return_rank_keys, request_count);
    }
}

void final_scatter_history_records_cuda(
    const FinalHistoryRecord* records,
    CandidateMeta* history_candidates,
    std::uint32_t record_count,
    cudaStream_t stream) {
    NvtxRange range("Final_scatter_history_records_launch");
    const dim3 block(128);
    const dim3 grid((record_count + block.x - 1) / block.x);
    if (grid.x != 0U) {
        final_scatter_history_records_kernel<<<grid, block, 0, stream>>>(records, history_candidates, record_count);
    }
}

void validate_final_requests_cuda(
    const FinalRequest* requests,
    std::uint32_t request_count,
    std::uint64_t current_frontier_size,
    std::uint32_t target_count,
    FinalRequestValidationError* device_error,
    cudaStream_t stream) {
#if BEAM_DEBUG_FINAL_VALIDATE
    if (request_count == 0U) {
        return;
    }
    if (device_error == nullptr) {
        throw std::invalid_argument("final request validation requires static device error storage");
    }
    FinalRequestValidationError host_error{};
    host_error.first_index = UINT32_MAX;
    check_final_materialize_cuda(
        cudaMemcpyAsync(
            device_error,
            &host_error,
            sizeof(FinalRequestValidationError),
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync init final request validation error");
    validate_final_requests_kernel<<<1, 1, 0, stream>>>(
        requests,
        device_error,
        request_count,
        current_frontier_size,
        target_count);
    check_final_materialize_cuda(cudaGetLastError(), "validate_final_requests_kernel launch");
    check_final_materialize_cuda(
        cudaMemcpyAsync(
            &host_error,
            device_error,
            sizeof(FinalRequestValidationError),
            cudaMemcpyDeviceToHost,
            stream),
        "cudaMemcpyAsync final request validation error to host");
    check_final_materialize_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize final request validation");
    if (host_error.invalid_count != 0U) {
        throw std::runtime_error(
            "final request validation failed: invalid_count=" +
            std::to_string(host_error.invalid_count) +
            " first_index=" + std::to_string(host_error.first_index) +
            " parent_idx=" + std::to_string(host_error.parent_idx) +
            " current_frontier_size=" + std::to_string(host_error.current_frontier_size) +
            " target_local_idx=" + std::to_string(host_error.target_local_idx) +
            " target_count=" + std::to_string(host_error.target_count) +
            " move=" + std::to_string(host_error.move) +
            " move_count=" + std::to_string(MOVE_COUNT) +
            " request_count=" + std::to_string(host_error.request_count) +
            " reason_mask=" + std::to_string(host_error.reason_mask) +
            " reason_parent=" + std::to_string((host_error.reason_mask & FinalRequestInvalidParent) != 0U ? 1U : 0U) +
            " reason_target=" + std::to_string((host_error.reason_mask & FinalRequestInvalidTarget) != 0U ? 1U : 0U) +
            " reason_move=" + std::to_string((host_error.reason_mask & FinalRequestInvalidMove) != 0U ? 1U : 0U) +
            " reason_slot=" + std::to_string((host_error.reason_mask & FinalRequestInvalidSlot) != 0U ? 1U : 0U));
    }
#else
    (void)requests;
    (void)request_count;
    (void)current_frontier_size;
    (void)target_count;
    (void)device_error;
    (void)stream;
#endif
}

} // namespace beam
