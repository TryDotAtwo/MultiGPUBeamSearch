#pragma once
#include "stream1_transformer_layernorm_policy.hpp"
#include <cstdint>
#include <limits>
namespace beam {
struct Stream1LayerNormOccupancy {
    int sm=0;
    int multiprocessors=0;
    int copy_blocks_per_sm=0;
    int bias_round_blocks_per_sm=0;
};
struct Stream1LayerNormResidentPlan {
    std::uint32_t copy_grid=0;
    std::uint32_t bias_round_grid=0;
};
Stream1LayerNormOccupancy observe_stream1_layernorm_occupancy_cuda();
inline Stream1LayerNormResidentPlan resolve_stream1_layernorm_resident_plan(
    const Stream1LayerNormOccupancy& observed,int sm,const char* requested_copy_limit) {
    if(observed.sm!=sm || observed.multiprocessors<=0 ||
       observed.copy_blocks_per_sm<=0 || observed.bias_round_blocks_per_sm<=0)
        throw std::invalid_argument("persistent LayerNorm requires actual matching device occupancy");
    const auto copy=stream1_transformer_layernorm_persistent_blocks_per_sm(
        requested_copy_limit,observed.copy_blocks_per_sm);
    const auto grid=[&](int blocks) {
        const auto count=static_cast<std::uint64_t>(blocks)*observed.multiprocessors;
        if(count>std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("persistent LayerNorm resident grid overflows");
        return static_cast<std::uint32_t>(count);
    };
    // The existing bias-round launch always uses maximum occupancy; the user
    // copy limit applies only to the ordinary copy kernel, not both kernels.
    return {grid(copy),grid(observed.bias_round_blocks_per_sm)};
}
}
