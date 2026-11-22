#include "cuda_check.hpp"
#include "hash.hpp"
#include "state.hpp"
#include "stream2.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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

Generator swap23_generator() {
    Generator generator = identity_generator();
    generator[2] = 3;
    generator[3] = 2;
    return generator;
}

} // namespace

int main() {
    std::filesystem::create_directories("test_results");
    std::ofstream report("test_results/stream2_cuda_tests_2026-05-20.md");
    report << "# Stream2 CUDA Tests 2026-05-20\n\n";

    int device_count = 0;
    BEAM_CUDA_CHECK(cudaGetDeviceCount(&device_count));
    require(device_count > 0, "CUDA device required");
    BEAM_CUDA_CHECK(cudaSetDevice(0));

    State128 central{};
    for (std::size_t i = 0; i < STATE_LEN; ++i) {
        central.v[i] = static_cast<std::uint8_t>(i);
    }
    clear_state_padding(central);
    State128 start = central;
    start.v[0] = 1;
    start.v[1] = 0;

    std::vector<Generator> generators(MOVE_COUNT, identity_generator());
    generators[1] = swap01_generator();
    std::vector<std::uint8_t> flat_generators(MOVE_COUNT * STATE_STORAGE_LEN);
    for (std::size_t move = 0; move < MOVE_COUNT; ++move) {
        for (std::size_t p = 0; p < STATE_STORAGE_LEN; ++p) {
            flat_generators[move * STATE_STORAGE_LEN + p] = generators[move][p];
        }
    }

    const auto zobrist = make_deterministic_zobrist(123);
    std::vector<Hash128> flat_zobrist(STATE_STORAGE_LEN * STATE_VALUE_PAD);
    for (std::size_t p = 0; p < STATE_STORAGE_LEN; ++p) {
        for (std::size_t v = 0; v < STATE_VALUE_PAD; ++v) {
            flat_zobrist[p * STATE_VALUE_PAD + v] = zobrist[p][v];
        }
    }

    State128* d_frontier = nullptr;
    State128* d_central = nullptr;
    std::uint8_t* d_generators = nullptr;
    Hash128* d_zobrist = nullptr;
    Hash128* d_hash_ring = nullptr;
    std::uint64_t* d_parent_base = nullptr;
    std::uint32_t* d_count = nullptr;
    std::uint32_t* d_flags = nullptr;
    CandidateMeta* d_solved_meta = nullptr;
    std::uint32_t* d_solved_depth = nullptr;
    std::uint32_t* d_solved_suffix = nullptr;

    BEAM_CUDA_CHECK(cudaMalloc(&d_frontier, sizeof(State128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_central, sizeof(State128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_generators, flat_generators.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&d_zobrist, flat_zobrist.size() * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_hash_ring, MOVE_COUNT * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_parent_base, sizeof(std::uint64_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_count, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_flags, 4 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_solved_meta, 4 * sizeof(CandidateMeta)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_solved_depth, 4 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_solved_suffix, 4 * sizeof(std::uint32_t)));

    const std::uint64_t parent_base = 0;
    const std::uint32_t count = 1;
    BEAM_CUDA_CHECK(cudaMemcpy(d_frontier, &start, sizeof(State128), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_central, &central, sizeof(State128), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_generators, flat_generators.data(), flat_generators.size(), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_zobrist, flat_zobrist.data(), flat_zobrist.size() * sizeof(Hash128), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_parent_base, &parent_base, sizeof(parent_base), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_count, &count, sizeof(count), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemset(d_flags, 0, 4 * sizeof(std::uint32_t)));

    Stream2SolvedBuffers solved;
    solved.solved_flag = d_flags;
    solved.stop_flag = d_flags + 1;
    solved.solved_count = d_flags + 2;
    solved.solved_overflow = d_flags + 3;
    solved.solved_meta_list = d_solved_meta;
    solved.solved_depth_list = d_solved_depth;
    solved.solved_result_capacity = 4;
    solved.solved_suffix_list = d_solved_suffix;

    stream2_hash_goal_cuda(
        d_frontier,
        d_parent_base,
        d_count,
        d_generators,
        d_central,
        d_zobrist,
        d_hash_ring,
        0,
        0,
        1,
        7,
        0,
        solved,
        0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<Hash128> hashes(MOVE_COUNT);
    std::uint32_t flags[4]{};
    CandidateMeta solved_meta{};
    std::uint32_t solved_depth = 0;
    std::uint32_t solved_suffix = 0;
    BEAM_CUDA_CHECK(cudaMemcpy(hashes.data(), d_hash_ring, hashes.size() * sizeof(Hash128), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(flags, d_flags, sizeof(flags), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&solved_meta, d_solved_meta, sizeof(CandidateMeta), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&solved_depth, d_solved_depth, sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&solved_suffix, d_solved_suffix, sizeof(std::uint32_t), cudaMemcpyDeviceToHost));

    State128 child = apply_move(start, generators[1]);
    clear_state_padding(child);
    require(hashes[1] == hash_state(child, zobrist), "stream2 hash must match CPU reference");
    require(flags[0] == 1 && flags[1] == 1 && flags[2] == 1 && flags[3] == 0, "stream2 solved flags failed");
    require(solved_meta.score_key == GOAL_SCORE_KEY, "stream2 goal score key failed");
    require(solved_depth == 7, "stream2 solved depth failed");
    require(solved_suffix == 0U, "stream2 direct suffix id failed");
    report << "- hash_matches_cpu=pass\n";
    report << "- goal_sets_flags=pass\n";

    BEAM_CUDA_CHECK(cudaMemset(d_flags, 0, 4 * sizeof(std::uint32_t)));
    std::vector<Generator> identity_generators(MOVE_COUNT, swap01_generator());
    identity_generators[0] = identity_generator();
    std::vector<std::uint8_t> flat_identity_generators(MOVE_COUNT * STATE_STORAGE_LEN);
    for (std::size_t move = 0; move < MOVE_COUNT; ++move) {
        for (std::size_t p = 0; p < STATE_STORAGE_LEN; ++p) {
            flat_identity_generators[move * STATE_STORAGE_LEN + p] = identity_generators[move][p];
        }
    }
    BEAM_CUDA_CHECK(cudaMemcpy(
        d_generators,
        flat_identity_generators.data(),
        flat_identity_generators.size(),
        cudaMemcpyHostToDevice));

    const Hash128 neighborhood_hash = hash_state(start, zobrist);
    constexpr std::uint32_t bucket_count = 4;
    std::vector<std::uint32_t> neighborhood_fingerprints(
        bucket_count * SOLVED_NEIGHBORHOOD_BUCKET_SIZE,
        0U);
    std::vector<Hash128> neighborhood_hashes(
        bucket_count * SOLVED_NEIGHBORHOOD_BUCKET_SIZE);
    const std::uint32_t bucket =
        static_cast<std::uint32_t>(hash128_bucket_key_0(neighborhood_hash)) & (bucket_count - 1U);
    const std::uint32_t slot = bucket * SOLVED_NEIGHBORHOOD_BUCKET_SIZE;
    neighborhood_fingerprints[slot] = hash128_fingerprint32(neighborhood_hash);
    neighborhood_hashes[slot] = neighborhood_hash;
    std::uint32_t* d_neighborhood_fingerprints = nullptr;
    Hash128* d_neighborhood_hashes = nullptr;
    BEAM_CUDA_CHECK(cudaMalloc(
        &d_neighborhood_fingerprints,
        neighborhood_fingerprints.size() * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMalloc(
        &d_neighborhood_hashes,
        neighborhood_hashes.size() * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMemcpy(
        d_neighborhood_fingerprints,
        neighborhood_fingerprints.data(),
        neighborhood_fingerprints.size() * sizeof(std::uint32_t),
        cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(
        d_neighborhood_hashes,
        neighborhood_hashes.data(),
        neighborhood_hashes.size() * sizeof(Hash128),
        cudaMemcpyHostToDevice));
    solved.solved_neighborhood = SolvedNeighborhoodDeviceTable{
        d_neighborhood_fingerprints,
        d_neighborhood_hashes,
        bucket_count - 1U,
        1U};
    stream2_hash_goal_cuda(
        d_frontier,
        d_parent_base,
        d_count,
        d_generators,
        d_central,
        d_zobrist,
        d_hash_ring,
        0,
        0,
        1,
        9,
        0,
        solved,
        0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    BEAM_CUDA_CHECK(cudaMemcpy(flags, d_flags, sizeof(flags), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&solved_meta, d_solved_meta, sizeof(CandidateMeta), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&solved_suffix, d_solved_suffix, sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    require(flags[0] == 1 && flags[1] == 1 && flags[2] != 0 && flags[3] == 0, "stream2 neighborhood flags failed");
    require(solved_meta.hash == neighborhood_hash, "stream2 neighborhood hash failed");
    require(solved_suffix == 0U, "stream2 neighborhood suffix id failed");
    report << "- solved_neighborhood_lookup=pass\n";

    State128 start_k2 = central;
    start_k2.v[0] = 1;
    start_k2.v[1] = 0;
    start_k2.v[2] = 3;
    start_k2.v[3] = 2;
    clear_state_padding(start_k2);
    std::vector<Generator> k2_generators(MOVE_COUNT, identity_generator());
    k2_generators[1] = swap01_generator();
    k2_generators[2] = swap23_generator();
    std::vector<std::uint8_t> flat_k2_generators(MOVE_COUNT * STATE_STORAGE_LEN);
    for (std::size_t move = 0; move < MOVE_COUNT; ++move) {
        for (std::size_t p = 0; p < STATE_STORAGE_LEN; ++p) {
            flat_k2_generators[move * STATE_STORAGE_LEN + p] = k2_generators[move][p];
        }
    }
    BEAM_CUDA_CHECK(cudaMemcpy(d_frontier, &start_k2, sizeof(State128), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(
        d_generators,
        flat_k2_generators.data(),
        flat_k2_generators.size(),
        cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemset(d_flags, 0, 4 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_solved_suffix, 0, 4 * sizeof(std::uint32_t)));

    const Hash128 central_hash = hash_state(central, zobrist);
    std::fill(neighborhood_fingerprints.begin(), neighborhood_fingerprints.end(), 0U);
    std::fill(neighborhood_hashes.begin(), neighborhood_hashes.end(), Hash128{});
    const std::uint32_t central_bucket =
        static_cast<std::uint32_t>(hash128_bucket_key_0(central_hash)) & (bucket_count - 1U);
    const std::uint32_t central_slot = central_bucket * SOLVED_NEIGHBORHOOD_BUCKET_SIZE;
    neighborhood_fingerprints[central_slot] = hash128_fingerprint32(central_hash);
    neighborhood_hashes[central_slot] = central_hash;
    BEAM_CUDA_CHECK(cudaMemcpy(
        d_neighborhood_fingerprints,
        neighborhood_fingerprints.data(),
        neighborhood_fingerprints.size() * sizeof(std::uint32_t),
        cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(
        d_neighborhood_hashes,
        neighborhood_hashes.data(),
        neighborhood_hashes.size() * sizeof(Hash128),
        cudaMemcpyHostToDevice));

    std::uint64_t* d_suffix_moves = nullptr;
    std::uint8_t* d_suffix_lengths = nullptr;
    std::uint8_t* d_suffix_permutations = nullptr;
    const std::uint64_t suffix_moves[2]{0ULL, 2ULL};
    const std::uint8_t suffix_lengths[2]{0U, 1U};
    BEAM_CUDA_CHECK(cudaMalloc(&d_suffix_moves, sizeof(suffix_moves)));
    BEAM_CUDA_CHECK(cudaMalloc(&d_suffix_lengths, sizeof(suffix_lengths)));
    BEAM_CUDA_CHECK(cudaMemcpy(d_suffix_moves, suffix_moves, sizeof(suffix_moves), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(d_suffix_lengths, suffix_lengths, sizeof(suffix_lengths), cudaMemcpyHostToDevice));
    solved.stream2_suffix = Stream2SuffixDeviceTable{
        d_suffix_moves,
        d_suffix_lengths,
        nullptr,
        2U,
        STREAM2_SUFFIX_BACKEND_BASE_GENERATORS,
        1U};
    stream2_hash_goal_cuda(
        d_frontier,
        d_parent_base,
        d_count,
        d_generators,
        d_central,
        d_zobrist,
        d_hash_ring,
        0,
        0,
        1,
        11,
        0,
        solved,
        0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    BEAM_CUDA_CHECK(cudaMemcpy(flags, d_flags, sizeof(flags), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&solved_meta, d_solved_meta, sizeof(CandidateMeta), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&solved_suffix, d_solved_suffix, sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    require(flags[0] == 1 && flags[1] == 1 && flags[2] != 0 && flags[3] == 0, "stream2 k2 base flags failed");
    require(solved_meta.hash == central_hash, "stream2 k2 base hash failed");
    require(solved_suffix == 1U, "stream2 k2 base suffix id failed");
    report << "- stream2_suffix_base_generators=pass\n";

    std::vector<std::uint8_t> suffix_permutations(2ULL * STATE_STORAGE_LEN);
    for (std::uint32_t p = 0; p < STATE_STORAGE_LEN; ++p) {
        suffix_permutations[p] = static_cast<std::uint8_t>(p);
        suffix_permutations[STATE_STORAGE_LEN + p] =
            flat_k2_generators[2ULL * STATE_STORAGE_LEN + p];
    }
    BEAM_CUDA_CHECK(cudaMalloc(&d_suffix_permutations, suffix_permutations.size()));
    BEAM_CUDA_CHECK(cudaMemcpy(
        d_suffix_permutations,
        suffix_permutations.data(),
        suffix_permutations.size(),
        cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemset(d_flags, 0, 4 * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(d_solved_suffix, 0, 4 * sizeof(std::uint32_t)));
    solved.stream2_suffix = Stream2SuffixDeviceTable{
        d_suffix_moves,
        d_suffix_lengths,
        d_suffix_permutations,
        2U,
        STREAM2_SUFFIX_BACKEND_COMPOSED_PERMUTATIONS,
        1U};
    stream2_hash_goal_cuda(
        d_frontier,
        d_parent_base,
        d_count,
        d_generators,
        d_central,
        d_zobrist,
        d_hash_ring,
        0,
        0,
        1,
        11,
        0,
        solved,
        0);
    BEAM_CUDA_CHECK(cudaGetLastError());
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    BEAM_CUDA_CHECK(cudaMemcpy(flags, d_flags, sizeof(flags), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&solved_meta, d_solved_meta, sizeof(CandidateMeta), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&solved_suffix, d_solved_suffix, sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    require(flags[0] == 1 && flags[1] == 1 && flags[2] != 0 && flags[3] == 0, "stream2 k2 composed flags failed");
    require(solved_meta.hash == central_hash, "stream2 k2 composed hash failed");
    require(solved_suffix == 1U, "stream2 k2 composed suffix id failed");
    report << "- stream2_suffix_composed_permutations=pass\n";

    report << "\nstatus=pass\n";

    cudaFree(d_suffix_moves);
    cudaFree(d_suffix_lengths);
    cudaFree(d_suffix_permutations);
    cudaFree(d_neighborhood_fingerprints);
    cudaFree(d_neighborhood_hashes);
    cudaFree(d_frontier);
    cudaFree(d_central);
    cudaFree(d_generators);
    cudaFree(d_zobrist);
    cudaFree(d_hash_ring);
    cudaFree(d_parent_base);
    cudaFree(d_count);
    cudaFree(d_flags);
    cudaFree(d_solved_meta);
    cudaFree(d_solved_depth);
    cudaFree(d_solved_suffix);
    std::cout << "stream2_cuda_tests=pass\n";
    return 0;
}
