#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace beam {

inline void cuda_check(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error: ") + expression +
            " file=" + file +
            " line=" + std::to_string(line) +
            " message=" + cudaGetErrorString(status));
    }
}

} // namespace beam

#define BEAM_CUDA_CHECK(expr) ::beam::cuda_check((expr), #expr, __FILE__, __LINE__)
