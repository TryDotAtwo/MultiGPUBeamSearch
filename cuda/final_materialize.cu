#include "final_materialize.hpp"

#include "config.hpp"
#include "nvtx_ranges.hpp"

#include <cuda_runtime.h>

namespace beam {

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
    responses[i] = response;

    const std::uint32_t target_local_idx = final_response_get_target_local_idx_device(response);
    clear_state_padding_device(response);
    next_frontier_states_tmp[target_local_idx] = response;
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

} // namespace beam
