#pragma once
#include "cuda_check.hpp"
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace beam {
// Startup only: compare the actual uploaded bytes with the prepared host blob,
// including its packed physical representation. Never called per inference job.
inline void verify_device_weight_bytes(const void* device,
                                      const std::vector<std::byte>& prepared,
                                      const char* name) {
    if (device == nullptr || prepared.empty())
        throw std::invalid_argument("invalid device weight verification input");
    std::vector<std::byte> observed(prepared.size());
    BEAM_CUDA_CHECK(cudaMemcpy(observed.data(),device,observed.size(),cudaMemcpyDeviceToHost));
    if (observed != prepared)
        throw std::runtime_error(std::string("uploaded device weight bytes differ: ")+
                                 (name ? name : "unnamed"));
}
} // namespace beam
