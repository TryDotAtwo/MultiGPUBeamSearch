#include "cuda_check.hpp"
#include "../cuda/final_materialize.hpp"
#include "state.hpp"

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace beam;

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Generator identity_generator() {
    Generator generator{};
    for (std::size_t i = 0; i < STATE_STORAGE_LEN; ++i) {
        generator[i] = static_cast<std::uint8_t>(i);
    }
    return generator;
}

Generator swap01_generator() {
    Generator generator = identity_generator();
    generator[0] = 1;
    generator[1] = 0;
    return generator;
}
} // namespace

int main() {
    std::filesystem::create_directories("test_results");
    std::ofstream report("test_results/final_cuda_tests_2026-05-20.md");
    report << "# Final CUDA Tests 2026-05-20\n\n";

    BEAM_CUDA_CHECK(cudaSetDevice(0));
    State128 state{};
    for (std::size_t i = 0; i < STATE_LEN; ++i) {
        state.v[i] = static_cast<std::uint8_t>(i);
    }
    clear_state_padding(state);
    state.v[0] = 1;
    state.v[1] = 0;

    std::vector<Generator> generators(MOVE_COUNT, identity_generator());
    generators[1] = swap01_generator();
    std::vector<std::uint8_t> flat_generators(MOVE_COUNT * STATE_STORAGE_LEN);
    for (std::size_t move = 0; move < MOVE_COUNT; ++move) {
        for (std::size_t p = 0; p < STATE_STORAGE_LEN; ++p) {
            flat_generators[move * STATE_STORAGE_LEN + p] = generators[move][p];
        }
    }
    FinalRequest request{};
    request.parent_idx = 0;
    request.target_local_idx = 0;
    request.return_rank = 0;
    request.move = 1;

    State128* d_frontier = nullptr;
    FinalRequest* d_requests = nullptr;
    std::uint8_t* d_generators = nullptr;
    FinalResponse* d_responses = nullptr;
    State128* d_next = nullptr;
    BEAM_CUDA_CHECK(cudaMalloc(&d_frontier, sizeof(State128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_requests, sizeof(FinalRequest)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_generators, flat_generators.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&d_responses, sizeof(FinalResponse)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_next, sizeof(State128)));
    BEAM_CUDA_CHECK(cudaMemcpy(d_frontier, &state, sizeof(State128), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_requests, &request, sizeof(FinalRequest), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_generators, flat_generators.data(), flat_generators.size(), cudaMemcpyHostToDevice));

    final_materialize_cuda(d_frontier, d_requests, d_generators, d_responses, d_next, 1, 0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());

    FinalResponse response{};
    State128 next{};
    BEAM_CUDA_CHECK(cudaMemcpy(&response, d_responses, sizeof(FinalResponse), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&next, d_next, sizeof(State128), cudaMemcpyDeviceToHost));
    require(final_response_get_target_local_idx(response) == 0, "final response target idx failed");
    require(next.v[0] == 0 && next.v[1] == 1, "final apply_move failed");
    require(padding_is_zero(next), "final persistent padding cleanup failed");
    report << "- response_target_idx_pack=pass\n";
    report << "- write_next_frontier=pass\n";
    report << "- padding_cleanup=pass\n";
    report << "\nstatus=pass\n";

    cudaFree(d_frontier);
    cudaFree(d_requests);
    cudaFree(d_generators);
    cudaFree(d_responses);
    cudaFree(d_next);
    std::cout << "final_cuda_tests=pass\n";
    return 0;
}
