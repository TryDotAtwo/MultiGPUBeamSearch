#include "stream5.hpp"

#include "nvtx_ranges.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace beam {

namespace {

void check_nccl(ncclResult_t status, const char* op) {
    if (status != ncclSuccess) {
        throw std::runtime_error(std::string(op) + ": " + ncclGetErrorString(status));
    }
}

__global__ void stream5_self_count_and_offsets_kernel(
    const std::uint32_t* send_count,
    std::uint32_t* recv_count,
    std::uint32_t local_rank) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        recv_count[local_rank] = send_count[local_rank];
    }
}

__global__ void stream5_self_copy_kernel(
    const CandidateMeta* remote_send_buffer,
    const std::uint32_t* send_count,
    const std::uint32_t* send_offset,
    CandidateMeta* remote_recv_buffer,
    const std::uint32_t* recv_offset,
    std::uint32_t local_rank) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count = send_count[local_rank];
    if (i >= count) {
        return;
    }
    remote_recv_buffer[recv_offset[local_rank] + i] = remote_send_buffer[send_offset[local_rank] + i];
}

__global__ void stream5_self_copy_words_kernel(
    const std::uint64_t* send_buffer,
    std::uint64_t* recv_buffer,
    std::uint32_t send_offset_items,
    std::uint32_t recv_offset_items,
    std::uint32_t item_count,
    std::uint32_t words_per_item) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t word_count = static_cast<std::uint64_t>(item_count) * words_per_item;
    if (i >= word_count) {
        return;
    }
    recv_buffer[static_cast<std::uint64_t>(recv_offset_items) * words_per_item + i] =
        send_buffer[static_cast<std::uint64_t>(send_offset_items) * words_per_item + i];
}

} // namespace

void stream5_exchange_counts_nccl_cuda(
    const std::uint32_t* device_send_count,
    std::uint32_t* device_recv_count,
    std::uint32_t local_rank,
    std::uint32_t world_size,
    ncclComm_t comm,
    cudaStream_t stream) {
    NvtxRange range("Stream5_NCCL_count_exchange_launch");
    if (world_size == 0 || local_rank >= world_size) {
        throw std::invalid_argument("stream5 NCCL exchange rank parameters are invalid");
    }

    check_nccl(ncclGroupStart(), "ncclGroupStart count exchange");
    for (std::uint32_t peer = 0; peer < world_size; ++peer) {
        if (peer == local_rank) {
            continue;
        }
        check_nccl(ncclSend(device_send_count + peer, 1, ncclUint32, static_cast<int>(peer), comm, stream), "ncclSend send_count");
        check_nccl(ncclRecv(device_recv_count + peer, 1, ncclUint32, static_cast<int>(peer), comm, stream), "ncclRecv recv_count");
    }
    check_nccl(ncclGroupEnd(), "ncclGroupEnd count exchange");

    stream5_self_count_and_offsets_kernel<<<1, 1, 0, stream>>>(device_send_count, device_recv_count, local_rank);
}

void stream5_write_recv_offsets_cuda(
    std::uint32_t* device_recv_offset,
    const std::uint32_t* host_recv_offset,
    std::uint32_t world_size,
    cudaStream_t stream) {
    NvtxRange range("Stream5_write_recv_offsets_launch");
    cudaError_t status = cudaMemcpyAsync(
        device_recv_offset,
        host_recv_offset,
        (static_cast<std::size_t>(world_size) + 1U) * sizeof(std::uint32_t),
        cudaMemcpyHostToDevice,
        stream);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync stream5 recv offsets: ") + cudaGetErrorString(status));
    }
}

void stream5_exchange_payload_nccl_cuda(
    const CandidateMeta* remote_send_buffer,
    const std::uint32_t* device_send_count,
    const std::uint32_t* device_send_offset,
    CandidateMeta* remote_recv_buffer,
    const std::uint32_t* device_recv_offset,
    const std::uint32_t* host_send_count,
    const std::uint32_t* host_send_offset,
    const std::uint32_t* host_recv_count,
    const std::uint32_t* host_recv_offset,
    std::uint32_t local_rank,
    std::uint32_t world_size,
    ncclComm_t comm,
    cudaStream_t stream) {
    NvtxRange range("Stream5_NCCL_payload_exchange_launch");
    if (world_size == 0 || local_rank >= world_size) {
        throw std::invalid_argument("stream5 NCCL payload exchange rank parameters are invalid");
    }

    check_nccl(ncclGroupStart(), "ncclGroupStart payload exchange");
    for (std::uint32_t peer = 0; peer < world_size; ++peer) {
        if (peer == local_rank) {
            continue;
        }
        check_nccl(ncclSend(
                       reinterpret_cast<const std::uint64_t*>(remote_send_buffer) + static_cast<std::uint64_t>(host_send_offset[peer]) * 4ULL,
                       static_cast<std::size_t>(host_send_count[peer]) * 4ULL,
                       ncclUint64,
                       static_cast<int>(peer),
                       comm,
                       stream),
                   "ncclSend payload");
        check_nccl(ncclRecv(
                       reinterpret_cast<std::uint64_t*>(remote_recv_buffer) + static_cast<std::uint64_t>(host_recv_offset[peer]) * 4ULL,
                       static_cast<std::size_t>(host_recv_count[peer]) * 4ULL,
                       ncclUint64,
                       static_cast<int>(peer),
                       comm,
                       stream),
                   "ncclRecv payload");
    }
    check_nccl(ncclGroupEnd(), "ncclGroupEnd payload exchange");

    const std::uint32_t block = 128;
    const std::uint32_t grid = (host_send_count[local_rank] + block - 1U) / block;
    if (grid != 0) {
        stream5_self_copy_kernel<<<grid, block, 0, stream>>>(
            remote_send_buffer,
            device_send_count,
            device_send_offset,
            remote_recv_buffer,
            device_recv_offset,
            local_rank);
    }
}

void stream5_exchange_u64_payload_nccl_cuda(
    const void* send_buffer,
    void* recv_buffer,
    const std::uint32_t* host_send_count,
    const std::uint32_t* host_send_offset,
    const std::uint32_t* host_recv_count,
    const std::uint32_t* host_recv_offset,
    std::uint32_t words_per_item,
    std::uint32_t local_rank,
    std::uint32_t world_size,
    ncclComm_t comm,
    cudaStream_t stream) {
    NvtxRange range("Stream5_NCCL_u64_payload_exchange_launch");
    if (world_size == 0 || local_rank >= world_size || words_per_item == 0U) {
        throw std::invalid_argument("stream5 generic NCCL payload exchange parameters are invalid");
    }

    const auto* send_words = reinterpret_cast<const std::uint64_t*>(send_buffer);
    auto* recv_words = reinterpret_cast<std::uint64_t*>(recv_buffer);
    check_nccl(ncclGroupStart(), "ncclGroupStart generic payload exchange");
    for (std::uint32_t peer = 0; peer < world_size; ++peer) {
        if (peer == local_rank) {
            continue;
        }
        check_nccl(ncclSend(
                       send_words + static_cast<std::uint64_t>(host_send_offset[peer]) * words_per_item,
                       static_cast<std::size_t>(host_send_count[peer]) * words_per_item,
                       ncclUint64,
                       static_cast<int>(peer),
                       comm,
                       stream),
                   "ncclSend generic payload");
        check_nccl(ncclRecv(
                       recv_words + static_cast<std::uint64_t>(host_recv_offset[peer]) * words_per_item,
                       static_cast<std::size_t>(host_recv_count[peer]) * words_per_item,
                       ncclUint64,
                       static_cast<int>(peer),
                       comm,
                       stream),
                   "ncclRecv generic payload");
    }
    check_nccl(ncclGroupEnd(), "ncclGroupEnd generic payload exchange");

    const std::uint32_t self_count = host_send_count[local_rank];
    const std::uint64_t self_words = static_cast<std::uint64_t>(self_count) * words_per_item;
    const std::uint32_t block = 128;
    const std::uint32_t grid = static_cast<std::uint32_t>((self_words + block - 1ULL) / block);
    if (grid != 0) {
        stream5_self_copy_words_kernel<<<grid, block, 0, stream>>>(
            send_words,
            recv_words,
            host_send_offset[local_rank],
            host_recv_offset[local_rank],
            self_count,
            words_per_item);
    }
}

void stream5_exchange_nccl_cuda(
    const CandidateMeta* remote_send_buffer,
    const std::uint32_t* device_send_count,
    const std::uint32_t* device_send_offset,
    CandidateMeta* remote_recv_buffer,
    std::uint32_t* device_recv_count,
    std::uint32_t* device_recv_offset,
    const std::uint32_t* host_send_count,
    const std::uint32_t* host_send_offset,
    const std::uint32_t* host_recv_count,
    const std::uint32_t* host_recv_offset,
    std::uint32_t local_rank,
    std::uint32_t world_size,
    ncclComm_t comm,
    cudaStream_t stream) {
    NvtxRange range("Stream5_NCCL_exchange_launch");
    stream5_exchange_counts_nccl_cuda(
        device_send_count,
        device_recv_count,
        local_rank,
        world_size,
        comm,
        stream);
    stream5_write_recv_offsets_cuda(device_recv_offset, host_recv_offset, world_size, stream);
    stream5_exchange_payload_nccl_cuda(
        remote_send_buffer,
        device_send_count,
        device_send_offset,
        remote_recv_buffer,
        device_recv_offset,
        host_send_count,
        host_send_offset,
        host_recv_count,
        host_recv_offset,
        local_rank,
        world_size,
        comm,
        stream);
}

} // namespace beam
