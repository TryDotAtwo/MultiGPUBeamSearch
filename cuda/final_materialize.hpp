#pragma once

#include "types.hpp"

namespace beam {

void final_materialize_cuda(
    const State128* current_frontier_states,
    const FinalRequest* requests,
    const std::uint8_t* generators,
    FinalResponse* responses,
    State128* next_frontier_states_tmp,
    std::uint32_t request_count,
    cudaStream_t stream);

void validate_final_requests_cuda(
    const FinalRequest* requests,
    std::uint32_t request_count,
    std::uint64_t current_frontier_size,
    std::uint32_t target_count,
    cudaStream_t stream);

} // namespace beam
