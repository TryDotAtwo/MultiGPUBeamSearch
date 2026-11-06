#pragma once

#include "../cuda/cuda_check.hpp"
#include "../src/distributed_rank.hpp"

namespace beam::distributed {
inline int select_cuda_device(std::uint32_t local_rank) {
    int visible = 0;
    BEAM_CUDA_CHECK(cudaGetDeviceCount(&visible));
    const int ordinal = cuda_ordinal(local_rank, visible);
    BEAM_CUDA_CHECK(cudaSetDevice(ordinal));
    return ordinal;
}
} // namespace beam::distributed
