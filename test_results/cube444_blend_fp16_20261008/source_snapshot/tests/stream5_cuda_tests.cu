#include "cuda_check.hpp"
#include "../cuda/stream5.hpp"

#include <cuda_runtime.h>
#include <nccl.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace beam;

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_nccl(ncclResult_t status, const char* message) {
    if (status != ncclSuccess) {
        throw std::runtime_error(std::string(message) + ": " + ncclGetErrorString(status));
    }
}
} // namespace

int main() {
    std::filesystem::create_directories("test_results");
    std::ofstream report("test_results/stream5_cuda_tests_2026-05-20.md");
    report << "# Stream5 CUDA Tests 2026-05-20\n\n";

    BEAM_CUDA_CHECK(cudaSetDevice(0));
    std::vector<CandidateMeta> send{
        CandidateMeta{Hash128{1, 2}, 10, 3, pack_route(0, 0, 1)},
        CandidateMeta{Hash128{3, 4}, 11, 4, pack_route(0, 0, 2)},
    };
    std::uint32_t send_count[1]{2};
    std::uint32_t send_offset[2]{0, 2};
    std::uint32_t recv_count_host[1]{2};
    std::uint32_t recv_offset_host[2]{0, 2};

    CandidateMeta* d_send = nullptr;
    CandidateMeta* d_recv = nullptr;
    std::uint32_t* d_send_count = nullptr;
    std::uint32_t* d_send_offset = nullptr;
    std::uint32_t* d_recv_count = nullptr;
    std::uint32_t* d_recv_offset = nullptr;
    BEAM_CUDA_CHECK(cudaMalloc(&d_send, send.size() * sizeof(CandidateMeta)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_recv, send.size() * sizeof(CandidateMeta)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_send_count, sizeof(send_count)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_send_offset, sizeof(send_offset)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_recv_count, sizeof(send_count)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_recv_offset, sizeof(send_offset)));
    BEAM_CUDA_CHECK(cudaMemcpy(d_send, send.data(), send.size() * sizeof(CandidateMeta), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_send_count, send_count, sizeof(send_count), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_send_offset, send_offset, sizeof(send_offset), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemset(d_recv_count, 0, sizeof(send_count)));
    BEAM_CUDA_CHECK(cudaMemset(d_recv_offset, 0, sizeof(send_offset)));

    ncclUniqueId id{};
    ncclComm_t comm{};
    require_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId failed");
    require_nccl(ncclCommInitRank(&comm, 1, id, 0), "ncclCommInitRank failed");
    stream5_exchange_counts_nccl_cuda(d_send_count, d_recv_count, 0, 1, comm, 0);
    stream5_write_recv_offsets_cuda(d_recv_offset, recv_offset_host, 1, 0);
    stream5_exchange_payload_nccl_cuda(
        d_send,
        d_send_count,
        d_send_offset,
        d_recv,
        d_recv_offset,
        send_count,
        send_offset,
        recv_count_host,
        recv_offset_host,
        0,
        1,
        comm,
        0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    require_nccl(ncclCommDestroy(comm), "ncclCommDestroy failed");

    std::vector<CandidateMeta> recv(send.size());
    std::uint32_t recv_count[1]{};
    std::uint32_t recv_offset[2]{};
    BEAM_CUDA_CHECK(cudaMemcpy(recv.data(), d_recv, recv.size() * sizeof(CandidateMeta), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(recv_count, d_recv_count, sizeof(recv_count), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(recv_offset, d_recv_offset, sizeof(recv_offset), cudaMemcpyDeviceToHost));

    require(recv_count[0] == 2, "stream5 recv_count failed");
    require(recv_offset[0] == 0 && recv_offset[1] == 2, "stream5 recv_offset failed");
    require(recv[0].hash == send[0].hash && recv[1].hash == send[1].hash, "stream5 payload copy failed");
    report << "- nccl_exchange_single_rank=pass\n";
    report << "\nstatus=pass\n";

    cudaFree(d_send);
    cudaFree(d_recv);
    cudaFree(d_send_count);
    cudaFree(d_send_offset);
    cudaFree(d_recv_count);
    cudaFree(d_recv_offset);
    std::cout << "stream5_cuda_tests=pass\n";
    return 0;
}
