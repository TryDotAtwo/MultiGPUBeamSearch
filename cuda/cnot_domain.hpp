#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace beam::cnot {

// All arrays reside on the stream's device. parents is contiguous row-major
// [B,n,ceil(n/64)]. Each child uses exactly one (parent index, physical move).
// The caller owns the output and sizes it for K rows before launching.
cudaError_t launch_children(const std::int64_t* parents, std::int64_t parent_count, int n,
                            const std::int64_t* indices, const void* moves, bool moves_are_u16,
                            std::int64_t count, std::int64_t* children, cudaStream_t stream);

cudaError_t launch_owners(const std::int64_t* states, std::int64_t count, int n, int world,
                          std::int64_t* owners, cudaStream_t stream);

// Passing temp=nullptr only queries the shared sort/compact workspace size.
cudaError_t first_unseen(void* temp, std::size_t& temp_bytes, const std::int64_t* closed,
                         int closed_count, const std::int64_t* children, int child_count, int n,
                         int* ordinals, std::uint8_t* flags, std::int64_t* selected, int* count,
                         cudaStream_t stream);

} // namespace beam::cnot
