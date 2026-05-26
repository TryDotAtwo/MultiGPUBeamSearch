#include "final_materialize.hpp"

#include "config.hpp"
#include "nvtx_ranges.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#ifndef BEAM_DEBUG_FINAL_VALIDATE
#define BEAM_DEBUG_FINAL_VALIDATE 0
#endif

namespace beam {

namespace {

#if BEAM_DEBUG_FINAL_VALIDATE
constexpr std::uint32_t FinalRequestInvalidParent = 1U << 0U;
constexpr std::uint32_t FinalRequestInvalidTarget = 1U << 1U;
constexpr std::uint32_t FinalRequestInvalidMove = 1U << 2U;
constexpr std::uint32_t FinalRequestInvalidSlot = 1U << 3U;

struct alignas(16) FinalRequestValidationError {
    std::uint32_t invalid_count;
    std::uint32_t first_index;
    std::uint64_t parent_idx;
    std::uint32_t target_local_idx;
    std::uint32_t request_count;
    std::uint64_t current_frontier_size;
    std::uint32_t target_count;
    std::uint32_t reason_mask;
    std::uint32_t move;
};

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
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= request_count) {
        return;
    }
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
        return;
    }

    atomicAdd(&error->invalid_count, 1U);
    if (atomicCAS(&error->first_index, UINT32_MAX, i) == UINT32_MAX) {
        error->parent_idx = request.parent_idx;
        error->target_local_idx = request.target_local_idx;
        error->request_count = request_count;
        error->current_frontier_size = current_frontier_size;
        error->target_count = target_count;
        error->reason_mask = reason;
        error->move = static_cast<std::uint32_t>(request.move);
    }
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

void validate_final_requests_cuda(
    const FinalRequest* requests,
    std::uint32_t request_count,
    std::uint64_t current_frontier_size,
    std::uint32_t target_count,
    cudaStream_t stream) {
#if BEAM_DEBUG_FINAL_VALIDATE
    if (request_count == 0U) {
        return;
    }
    FinalRequestValidationError host_error{};
    host_error.first_index = UINT32_MAX;
    FinalRequestValidationError* device_error = nullptr;
    check_final_materialize_cuda(
        cudaMalloc(&device_error, sizeof(FinalRequestValidationError)),
        "cudaMalloc final request validation error");
    struct DeviceErrorCleanup {
        FinalRequestValidationError*& ptr;
        ~DeviceErrorCleanup() {
            if (ptr != nullptr) {
                cudaFree(ptr);
            }
        }
    } cleanup{device_error};
    check_final_materialize_cuda(
        cudaMemcpyAsync(
            device_error,
            &host_error,
            sizeof(FinalRequestValidationError),
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync init final request validation error");
    const dim3 block(128);
    const dim3 grid((request_count + block.x - 1U) / block.x);
    validate_final_requests_kernel<<<grid, block, 0, stream>>>(
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
    (void)stream;
#endif
}

} // namespace beam
