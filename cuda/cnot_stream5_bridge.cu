#include "stream5.hpp"

#include <cuda_runtime.h>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// A small C boundary lets a tensor owner supply its existing device pointers
// and current stream. NCCL and CUDA types never cross the Python ABI by value.
namespace {
thread_local std::string last_error;

void require_nccl(ncclResult_t result) {
    if (result != ncclSuccess) throw std::runtime_error(ncclGetErrorString(result));
}

template<typename Operation>
int checked(Operation op) {
    try { op(); return 0; }
    catch (const std::exception& e) { last_error = e.what(); return 1; }
}

std::vector<std::uint32_t> offsets(const std::uint32_t* counts, int world) {
    std::vector<std::uint32_t> result(world + 1, 0);
    for (int i = 0; i < world; ++i) {
        if (counts[i] > std::numeric_limits<std::uint32_t>::max() - result[i])
            throw std::overflow_error("Stream 5 peer offsets exceed 32 bits");
        result[i + 1] = result[i] + counts[i];
    }
    return result;
}
} // namespace

extern "C" {
int cnot_s5_abi() { return 1; }
const char* cnot_s5_error() { return last_error.c_str(); }
int cnot_s5_id_bytes() { return sizeof(ncclUniqueId); }
int cnot_s5_id(ncclUniqueId* out) {
    return checked([&] { require_nccl(ncclGetUniqueId(out)); });
}
int cnot_s5_init(ncclComm_t* comm, const ncclUniqueId* id, int rank, int world) {
    return checked([&] { require_nccl(ncclCommInitRank(comm, world, *id, rank)); });
}
int cnot_s5_close(ncclComm_t comm, int abort) {
    return checked([&] { require_nccl(abort ? ncclCommAbort(comm) : ncclCommDestroy(comm)); });
}
int cnot_s5_counts(ncclComm_t comm, cudaStream_t stream, int rank, int world,
                   const std::uint32_t* send, std::uint32_t* recv) {
    return checked([&] {
        beam::stream5_exchange_counts_nccl_cuda(send, recv, rank, world, comm, stream);
    });
}
int cnot_s5_rows(ncclComm_t comm, cudaStream_t stream, int rank, int world,
                 const void* send, void* recv, const std::uint32_t* send_counts,
                 const std::uint32_t* recv_counts, std::uint32_t stride) {
    return checked([&] {
        auto send_offsets = offsets(send_counts, world), recv_offsets = offsets(recv_counts, world);
        beam::stream5_exchange_u64_payload_nccl_cuda(send, recv, send_counts, send_offsets.data(),
            recv_counts, recv_offsets.data(), stride, rank, world, comm, stream);
    });
}
} // extern "C"
