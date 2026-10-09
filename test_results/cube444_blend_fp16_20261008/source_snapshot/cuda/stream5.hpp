#pragma once

#include "types.hpp"

#include <nccl.h>

namespace beam {

void stream5_exchange_counts_nccl_cuda(
    const std::uint32_t* device_send_count,
    std::uint32_t* device_recv_count,
    std::uint32_t local_rank,
    std::uint32_t world_size,
    ncclComm_t comm,
    cudaStream_t stream);

void stream5_write_recv_offsets_cuda(
    std::uint32_t* device_recv_offset,
    const std::uint32_t* host_recv_offset,
    std::uint32_t world_size,
    cudaStream_t stream);

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
    cudaStream_t stream);

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
    cudaStream_t stream);

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
    cudaStream_t stream);

} // namespace beam
