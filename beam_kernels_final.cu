#include <cuda_runtime.h>
#include <stdint.h>

#include "beam_types.hpp"

namespace beam_v6 {

__device__ __forceinline__ State128 final_apply_move_state128(const State128& parent, const uint8_t* generators, uint8_t move) {
    State128 child;
    const uint8_t* gen = generators + int(move) * STATE_STORAGE_LEN;
    #pragma unroll
    for (int p = 0; p < STATE_STORAGE_LEN; ++p) {
        child.v[p] = parent.v[gen[p]];
    }
    return child;
}

extern "C" __global__ void kernel_v6_final_materialize(
    const State128* __restrict__ current_frontier_states,
    const FinalRequest* __restrict__ final_request_buffer,
    const uint8_t* __restrict__ generators,
    FinalResponse* __restrict__ final_response_buffer,
    uint32_t request_count
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= request_count) {
        return;
    }

    const FinalRequest req = final_request_buffer[idx];
    const State128 parent = current_frontier_states[req.parent_idx];
    FinalResponse response = final_apply_move_state128(parent, generators, req.move);
    final_response_set_target_local_idx(response, req.target_local_idx);
    response.v[124] = 0;
    response.v[125] = 0;
    response.v[126] = 0;
    response.v[127] = 0;
    final_response_buffer[idx] = response;
}

extern "C" __global__ void kernel_v6_final_scatter_responses(
    const FinalResponse* __restrict__ final_response_buffer,
    State128* __restrict__ next_frontier_states_tmp,
    uint32_t response_count
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= response_count) {
        return;
    }

    FinalResponse response = final_response_buffer[idx];
    const uint32_t target_local_idx = final_response_get_target_local_idx(response);
    clear_state_padding(response);
    next_frontier_states_tmp[target_local_idx] = response;
}

} // namespace beam_v6

extern "C" void launch_v6_final_materialize(
    const beam_v6::State128* current_frontier_states,
    const beam_v6::FinalRequest* final_request_buffer,
    const uint8_t* generators,
    beam_v6::FinalResponse* final_response_buffer,
    uint32_t request_count,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = int((request_count + threads - 1) / threads);
    beam_v6::kernel_v6_final_materialize<<<blocks, threads, 0, stream>>>(
        current_frontier_states,
        final_request_buffer,
        generators,
        final_response_buffer,
        request_count);
}

extern "C" void launch_v6_final_scatter_responses(
    const beam_v6::FinalResponse* final_response_buffer,
    beam_v6::State128* next_frontier_states_tmp,
    uint32_t response_count,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = int((response_count + threads - 1) / threads);
    beam_v6::kernel_v6_final_scatter_responses<<<blocks, threads, 0, stream>>>(
        final_response_buffer,
        next_frontier_states_tmp,
        response_count);
}
