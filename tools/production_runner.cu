#include "cuda_check.hpp"
#include "stream1_weight_io.hpp"
#include "../cuda/dispatcher.hpp"
#include "../cuda/runtime_config.hpp"
#include "../src/hash.hpp"
#include "../src/state.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <new>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace beam;

#ifndef BEAM_ENABLE_DEPTH_LOGS
#define BEAM_ENABLE_DEPTH_LOGS 0
#endif

#ifndef BEAM_ENABLE_DEBUG_LOGS
#define BEAM_ENABLE_DEBUG_LOGS 0
#endif

#ifndef BEAM_DEBUG_STREAM_TIMING
#define BEAM_DEBUG_STREAM_TIMING 0
#endif

#ifndef BEAM_DEBUG_INFERENCE_TRACE
#define BEAM_DEBUG_INFERENCE_TRACE 0
#endif

#ifndef BEAM_DEBUG_PATH_TRACE
#define BEAM_DEBUG_PATH_TRACE 0
#endif

#ifndef BEAM_DEBUG_DEPTH_FLOW_TRACE
#define BEAM_DEBUG_DEPTH_FLOW_TRACE 0
#endif

namespace {

std::uint64_t parse_u64(const char* text, const char* name) {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid numeric argument: ") + name);
    }
    return static_cast<std::uint64_t>(value);
}

std::uint32_t env_u32(const char* name, std::uint32_t default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    const std::uint64_t parsed = parse_u64(value, name);
    if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string("env value exceeds uint32: ") + name);
    }
    return static_cast<std::uint32_t>(parsed);
}

std::uint64_t env_u64(const char* name, std::uint64_t default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return parse_u64(value, name);
}

bool env_present(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
}

std::uint32_t env_or_default_u32(const char* name, std::uint32_t default_value) {
    return env_present(name) ? env_u32(name, default_value) : default_value;
}

bool env_bool(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "ON") == 0;
}

std::filesystem::path env_path(const char* name, const char* default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return std::filesystem::path(default_value);
    }
    return std::filesystem::path(value);
}

std::uint32_t parse_next_u32(const std::string& text, std::size_t& pos, const char* context) {
    while (pos < text.size() && (text[pos] < '0' || text[pos] > '9')) {
        ++pos;
    }
    if (pos >= text.size()) {
        throw std::runtime_error(std::string("missing integer in ") + context);
    }
    std::uint64_t value = 0;
    while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
        value = value * 10ULL + static_cast<std::uint64_t>(text[pos] - '0');
        if (value > std::numeric_limits<std::uint8_t>::max()) {
            throw std::runtime_error(std::string("integer out of uint8 range in ") + context);
        }
        ++pos;
    }
    return static_cast<std::uint32_t>(value);
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open required text file: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

std::vector<std::uint8_t> load_p900_generators(const std::filesystem::path& path) {
    const std::string text = read_text_file(path);
    std::size_t pos = text.find("\"actions\"");
    if (pos == std::string::npos) {
        throw std::runtime_error("p900 generator json missing actions");
    }
    pos = text.find('[', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("p900 generator json malformed actions");
    }
    std::vector<std::uint8_t> generators(MOVE_COUNT * STATE_STORAGE_LEN);
    for (std::uint32_t move = 0; move < MOVE_COUNT; ++move) {
        pos = text.find('[', pos + 1);
        if (pos == std::string::npos) {
            throw std::runtime_error("p900 generator json missing move array");
        }
        for (std::uint32_t p = 0; p < STATE_LEN; ++p) {
            generators[move * STATE_STORAGE_LEN + p] =
                static_cast<std::uint8_t>(parse_next_u32(text, pos, "p900 generator"));
        }
        for (std::uint32_t p = STATE_LEN; p < STATE_STORAGE_LEN; ++p) {
            generators[move * STATE_STORAGE_LEN + p] = static_cast<std::uint8_t>(p);
        }
    }
    return generators;
}

std::vector<std::string> load_p900_move_names(const std::filesystem::path& path) {
    const std::string text = read_text_file(path);
    std::size_t pos = text.find("\"names\"");
    if (pos == std::string::npos) {
        throw std::runtime_error("p900 generator json missing names");
    }
    pos = text.find('[', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("p900 generator json malformed names");
    }
    std::vector<std::string> names;
    while (names.size() < MOVE_COUNT) {
        const std::size_t begin = text.find('"', pos + 1);
        if (begin == std::string::npos) {
            throw std::runtime_error("p900 generator json missing move name");
        }
        const std::size_t end = text.find('"', begin + 1);
        if (end == std::string::npos) {
            throw std::runtime_error("p900 generator json malformed move name");
        }
        names.push_back(text.substr(begin + 1, end - begin - 1));
        pos = end;
    }
    return names;
}

State128 load_central_state(const std::filesystem::path& path) {
    const std::string text = read_text_file(path);
    std::size_t pos = text.find("\"central_state\"");
    if (pos == std::string::npos) {
        throw std::runtime_error("puzzle info json missing central_state");
    }
    pos = text.find('[', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("puzzle info json malformed central_state");
    }
    State128 state{};
    for (std::uint32_t p = 0; p < STATE_LEN; ++p) {
        state.v[p] = static_cast<std::uint8_t>(parse_next_u32(text, pos, "central_state"));
    }
    for (std::uint32_t p = STATE_LEN; p < STATE_STORAGE_LEN; ++p) {
        state.v[p] = 0;
    }
    return state;
}

State128 load_initial_state_from_test_csv(const std::filesystem::path& path, std::uint64_t puzzle_id) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open required csv file: " + path.string());
    }
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        const std::size_t comma = line.find(',');
        if (comma == std::string::npos) {
            continue;
        }
        const std::uint64_t row_id = parse_u64(line.substr(0, comma).c_str(), "initial_state_id");
        if (row_id != puzzle_id) {
            continue;
        }
        const std::size_t first_quote = line.find('"', comma);
        const std::size_t last_quote = line.rfind('"');
        if (first_quote == std::string::npos || last_quote == std::string::npos || last_quote <= first_quote) {
            throw std::runtime_error("test csv malformed initial_state row");
        }
        const std::string state_text = line.substr(first_quote + 1, last_quote - first_quote - 1);
        std::size_t pos = 0;
        State128 state{};
        for (std::uint32_t p = 0; p < STATE_LEN; ++p) {
            state.v[p] = static_cast<std::uint8_t>(parse_next_u32(state_text, pos, "initial_state"));
        }
        for (std::uint32_t p = STATE_LEN; p < STATE_STORAGE_LEN; ++p) {
            state.v[p] = 0;
        }
        return state;
    }
    throw std::runtime_error("requested puzzle_id not found in test csv");
}

std::string timestamp_id() {
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    return std::to_string(seconds);
}

std::filesystem::path make_history_dir(
    std::uint64_t puzzle_id,
    std::uint32_t depth_limit,
    std::uint64_t beam,
    std::uint32_t rank,
    std::uint32_t world_size) {
    std::filesystem::path dir = "test_results";
    std::string name = "candidate_history_p" + std::to_string(puzzle_id) +
        "_d" + std::to_string(depth_limit) +
        "_b" + std::to_string(beam);
    if (world_size > 1U) {
        name += "_r" + std::to_string(rank);
    }
    name += "_" + timestamp_id();
    dir /= name;
    std::filesystem::create_directories(dir);
    return dir;
}

std::string state120_to_text(const State128& state) {
    std::ostringstream out;
    for (std::uint32_t i = 0; i < STATE_LEN; ++i) {
        if (i != 0U) {
            out << ',';
        }
        out << static_cast<std::uint32_t>(state.v[i]);
    }
    return out.str();
}

std::string submit_move_name(const std::string& raw_name) {
    if (!raw_name.empty() && raw_name.back() == '\'') {
        return "-" + raw_name.substr(0, raw_name.size() - 1U);
    }
    return raw_name;
}

std::string moves_to_path_text(const std::vector<std::uint8_t>& moves, const std::vector<std::string>& names) {
    std::ostringstream out;
    for (std::size_t i = 0; i < moves.size(); ++i) {
        if (i != 0U) {
            out << '.';
        }
        const std::uint32_t move = moves[i];
        if (move >= names.size()) {
            throw std::runtime_error("solution move index exceeds names table");
        }
        out << submit_move_name(names[move]);
    }
    return out.str();
}

std::vector<std::uint8_t> parse_solution_path_text(
    const std::string& path_text,
    const std::vector<std::string>& names) {
    std::unordered_map<std::string, std::uint8_t> move_by_name;
    for (std::uint32_t move = 0; move < names.size(); ++move) {
        move_by_name.emplace(submit_move_name(names[move]), static_cast<std::uint8_t>(move));
    }
    std::vector<std::uint8_t> moves;
    std::size_t begin = 0;
    while (begin <= path_text.size()) {
        const std::size_t end = path_text.find('.', begin);
        const std::string token = path_text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        const auto found = move_by_name.find(token);
        if (found == move_by_name.end()) {
            throw std::runtime_error("tracked solution path contains unknown move token: " + token);
        }
        moves.push_back(found->second);
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1U;
    }
    return moves;
}

State128 apply_move_flat_host(const State128& parent, const std::vector<std::uint8_t>& generators, std::uint8_t move) {
    if (move >= MOVE_COUNT) {
        throw std::runtime_error("solution move exceeds MOVE_COUNT");
    }
    State128 child{};
    const std::uint64_t base = static_cast<std::uint64_t>(move) * STATE_STORAGE_LEN;
    for (std::uint32_t p = 0; p < STATE_STORAGE_LEN; ++p) {
        child.v[p] = parent.v[generators[base + p]];
    }
    clear_state_padding(child);
    return child;
}

bool states_equal_storage(const State128& a, const State128& b) {
    for (std::uint32_t p = 0; p < STATE_STORAGE_LEN; ++p) {
        if (a.v[p] != b.v[p]) {
            return false;
        }
    }
    return true;
}

struct Hash128Hasher {
    std::size_t operator()(Hash128 hash) const noexcept {
        return static_cast<std::size_t>(hash128_distribution_key(hash, 0x7f4a7c159e3779b9ULL));
    }
};

struct PackedSuffix {
    std::uint64_t moves = 0;
    std::uint8_t len = 0;
};

State128 apply_inverse_move_flat_host(
    const State128& child,
    const std::vector<std::uint8_t>& generators,
    std::uint8_t move) {
    if (move >= MOVE_COUNT) {
        throw std::runtime_error("inverse move exceeds MOVE_COUNT");
    }
    State128 parent{};
    const std::uint64_t base = static_cast<std::uint64_t>(move) * STATE_STORAGE_LEN;
    for (std::uint32_t p = 0; p < STATE_STORAGE_LEN; ++p) {
        parent.v[generators[base + p]] = child.v[p];
    }
    clear_state_padding(parent);
    return parent;
}

PackedSuffix prepend_suffix_move(PackedSuffix suffix, std::uint8_t move) {
    if (move >= MOVE_COUNT) {
        throw std::runtime_error("suffix move exceeds MOVE_COUNT");
    }
    if (suffix.len >= 12U) {
        throw std::runtime_error("solved neighborhood suffix packing supports radius <= 12");
    }
    suffix.moves = (suffix.moves << 5U) | static_cast<std::uint64_t>(move);
    ++suffix.len;
    return suffix;
}

PackedSuffix append_suffix_move(PackedSuffix suffix, std::uint8_t move) {
    if (move >= MOVE_COUNT) {
        throw std::runtime_error("suffix move exceeds MOVE_COUNT");
    }
    if (suffix.len >= 12U) {
        throw std::runtime_error("stream2 suffix packing supports radius <= 12");
    }
    suffix.moves |= static_cast<std::uint64_t>(move) << (5U * suffix.len);
    ++suffix.len;
    return suffix;
}

void append_packed_suffix(std::vector<std::uint8_t>& moves, PackedSuffix suffix) {
    for (std::uint32_t i = 0; i < suffix.len; ++i) {
        moves.push_back(static_cast<std::uint8_t>((suffix.moves >> (5U * i)) & 31ULL));
    }
}

std::uint8_t packed_suffix_move_host(PackedSuffix suffix, std::uint32_t index) {
    return static_cast<std::uint8_t>((suffix.moves >> (5U * index)) & 31ULL);
}

std::uint64_t next_power_of_two_u64(std::uint64_t value) {
    if (value <= 1ULL) {
        return 1ULL;
    }
    --value;
    for (std::uint32_t shift = 1U; shift < 64U; shift <<= 1U) {
        value |= value >> shift;
    }
    return value + 1ULL;
}

struct SolvedNeighborhoodRuntime {
    std::uint32_t radius = 0;
    std::uint32_t bucket_count = 0;
    std::uint64_t entry_count = 0;
    std::uint64_t slot_count = 0;
    std::uint64_t device_bytes = 0;
    std::uint32_t* device_fingerprints = nullptr;
    Hash128* device_hashes = nullptr;
    std::unordered_map<Hash128, PackedSuffix, Hash128Hasher> suffix_by_hash;

    bool enabled() const {
        return radius != 0U && device_fingerprints != nullptr && device_hashes != nullptr;
    }

    SolvedNeighborhoodDeviceTable device_table() const {
        SolvedNeighborhoodDeviceTable table{};
        table.fingerprint_slots = device_fingerprints;
        table.hash_slots = device_hashes;
        table.bucket_mask = bucket_count == 0U ? 0U : bucket_count - 1U;
        table.enabled = enabled() ? 1U : 0U;
        return table;
    }

    PackedSuffix suffix_for(Hash128 hash) const {
        if (!enabled()) {
            return PackedSuffix{};
        }
        const auto found = suffix_by_hash.find(hash);
        if (found == suffix_by_hash.end()) {
            throw std::runtime_error("solved neighborhood hit is missing CPU suffix");
        }
        return found->second;
    }

    void destroy() {
        if (device_fingerprints != nullptr) {
            cudaFree(device_fingerprints);
        }
        if (device_hashes != nullptr) {
            cudaFree(device_hashes);
        }
        device_fingerprints = nullptr;
        device_hashes = nullptr;
        device_bytes = 0;
        slot_count = 0;
        bucket_count = 0;
        entry_count = 0;
        suffix_by_hash.clear();
    }
};

struct SolvedNeighborhoodNode {
    State128 state{};
    Hash128 hash{};
    PackedSuffix suffix{};
};

bool place_solved_neighborhood_entry(
    Hash128 hash,
    std::vector<std::uint32_t>& fingerprints,
    std::vector<Hash128>& hashes,
    std::uint32_t bucket_count) {
    const std::uint32_t mask = bucket_count - 1U;
    const std::uint32_t fingerprint = hash128_fingerprint32(hash);
    const std::uint32_t buckets[2]{
        static_cast<std::uint32_t>(hash128_bucket_key_0(hash)) & mask,
        static_cast<std::uint32_t>(hash128_bucket_key_1(hash)) & mask,
    };
    for (std::uint32_t b = 0; b < 2U; ++b) {
        const std::uint32_t base = buckets[b] * SOLVED_NEIGHBORHOOD_BUCKET_SIZE;
        for (std::uint32_t i = 0; i < SOLVED_NEIGHBORHOOD_BUCKET_SIZE; ++i) {
            const std::uint32_t index = base + i;
            if (fingerprints[index] == 0U) {
                fingerprints[index] = fingerprint;
                hashes[index] = hash;
                return true;
            }
        }
    }
    return false;
}

SolvedNeighborhoodRuntime build_solved_neighborhood_runtime(
    const State128& central_state,
    const std::vector<std::uint8_t>& generators,
    const ZobristTable& zobrist) {
    SolvedNeighborhoodRuntime runtime;
    runtime.radius = env_u32("BEAM_SOLVED_NEIGHBORHOOD_RADIUS", 0);
    if (runtime.radius == 0U) {
        return runtime;
    }
    if (runtime.radius > 12U) {
        throw std::runtime_error("BEAM_SOLVED_NEIGHBORHOOD_RADIUS must be <= 12");
    }
    if (generators.size() != MOVE_COUNT * STATE_STORAGE_LEN) {
        throw std::runtime_error("solved neighborhood builder requires flat p900 generators");
    }
    const std::uint64_t max_entries = env_u64("BEAM_SOLVED_NEIGHBORHOOD_MAX_ENTRIES", 0);
    std::vector<SolvedNeighborhoodNode> all_nodes;
    std::vector<SolvedNeighborhoodNode> frontier;
    const Hash128 central_hash = hash_state(central_state, zobrist);
    all_nodes.push_back(SolvedNeighborhoodNode{central_state, central_hash, PackedSuffix{}});
    frontier.push_back(all_nodes.front());
    runtime.suffix_by_hash.emplace(central_hash, PackedSuffix{});

    for (std::uint32_t depth = 0; depth < runtime.radius && !frontier.empty(); ++depth) {
        std::vector<SolvedNeighborhoodNode> next;
        next.reserve(frontier.size() * MOVE_COUNT);
        for (const SolvedNeighborhoodNode& node : frontier) {
            for (std::uint8_t move = 0; move < MOVE_COUNT; ++move) {
                State128 predecessor = apply_inverse_move_flat_host(node.state, generators, move);
                const Hash128 hash = hash_state(predecessor, zobrist);
                if (runtime.suffix_by_hash.find(hash) != runtime.suffix_by_hash.end()) {
                    continue;
                }
                const PackedSuffix suffix = prepend_suffix_move(node.suffix, move);
                if (max_entries != 0ULL && runtime.suffix_by_hash.size() >= max_entries) {
                    throw std::runtime_error("solved neighborhood exceeded BEAM_SOLVED_NEIGHBORHOOD_MAX_ENTRIES");
                }
                runtime.suffix_by_hash.emplace(hash, suffix);
                next.push_back(SolvedNeighborhoodNode{predecessor, hash, suffix});
            }
        }
        all_nodes.insert(all_nodes.end(), next.begin(), next.end());
        frontier = std::move(next);
    }

    runtime.entry_count = all_nodes.size();
    std::uint64_t bucket_count64 = next_power_of_two_u64(std::max<std::uint64_t>(1ULL, runtime.entry_count));
    while (true) {
        if (bucket_count64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::runtime_error("solved neighborhood bucket count exceeds uint32 range");
        }
        const std::uint32_t bucket_count = static_cast<std::uint32_t>(bucket_count64);
        std::vector<std::uint32_t> fingerprints(
            bucket_count64 * SOLVED_NEIGHBORHOOD_BUCKET_SIZE,
            0U);
        std::vector<Hash128> hashes(bucket_count64 * SOLVED_NEIGHBORHOOD_BUCKET_SIZE);
        bool packed = true;
        for (const SolvedNeighborhoodNode& node : all_nodes) {
            if (!place_solved_neighborhood_entry(node.hash, fingerprints, hashes, bucket_count)) {
                packed = false;
                break;
            }
        }
        if (!packed) {
            bucket_count64 *= 2ULL;
            continue;
        }
        runtime.bucket_count = bucket_count;
        runtime.slot_count = fingerprints.size();
        runtime.device_bytes =
            runtime.slot_count * (sizeof(std::uint32_t) + sizeof(Hash128));
        BEAM_CUDA_CHECK(cudaMalloc(&runtime.device_fingerprints, fingerprints.size() * sizeof(std::uint32_t)));
        BEAM_CUDA_CHECK(cudaMalloc(&runtime.device_hashes, hashes.size() * sizeof(Hash128)));
        BEAM_CUDA_CHECK(cudaMemcpy(
            runtime.device_fingerprints,
            fingerprints.data(),
            fingerprints.size() * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice));
        BEAM_CUDA_CHECK(cudaMemcpy(
            runtime.device_hashes,
            hashes.data(),
            hashes.size() * sizeof(Hash128),
            cudaMemcpyHostToDevice));
        return runtime;
    }
}

struct Stream2SuffixRuntime {
    std::uint32_t radius = 0;
    std::uint32_t backend = STREAM2_SUFFIX_BACKEND_BASE_GENERATORS;
    std::string backend_name = "base_generators";
    std::uint64_t entry_count = 0;
    std::uint64_t device_bytes = 0;
    std::vector<PackedSuffix> suffixes;
    std::uint64_t* device_packed_moves = nullptr;
    std::uint8_t* device_lengths = nullptr;
    std::uint8_t* device_composed_permutations = nullptr;

    bool enabled() const {
        return radius != 0U &&
            device_packed_moves != nullptr &&
            device_lengths != nullptr &&
            !suffixes.empty();
    }

    Stream2SuffixDeviceTable device_table() const {
        Stream2SuffixDeviceTable table{};
        table.packed_moves = device_packed_moves;
        table.lengths = device_lengths;
        table.composed_permutations = device_composed_permutations;
        table.suffix_count = static_cast<std::uint32_t>(suffixes.size());
        table.backend = backend;
        table.enabled = enabled() ? 1U : 0U;
        return table;
    }

    PackedSuffix suffix_for(std::uint32_t suffix_id) const {
        if (suffix_id >= suffixes.size()) {
            throw std::runtime_error("stream2 suffix id exceeds suffix table");
        }
        return suffixes[suffix_id];
    }

    std::uint8_t suffix_len(std::uint32_t suffix_id) const {
        if (suffix_id == 0U && suffixes.empty()) {
            return 0U;
        }
        return suffix_for(suffix_id).len;
    }

    void destroy() {
        if (device_packed_moves != nullptr) {
            cudaFree(device_packed_moves);
        }
        if (device_lengths != nullptr) {
            cudaFree(device_lengths);
        }
        if (device_composed_permutations != nullptr) {
            cudaFree(device_composed_permutations);
        }
        device_packed_moves = nullptr;
        device_lengths = nullptr;
        device_composed_permutations = nullptr;
        device_bytes = 0;
        entry_count = 0;
        suffixes.clear();
    }
};

std::uint32_t parse_stream2_suffix_backend(const std::string& backend) {
    if (backend == "base_generators") {
        return STREAM2_SUFFIX_BACKEND_BASE_GENERATORS;
    }
    if (backend == "composed_permutations") {
        return STREAM2_SUFFIX_BACKEND_COMPOSED_PERMUTATIONS;
    }
    throw std::runtime_error(
        "BEAM_STREAM2_SUFFIX_BACKEND must be base_generators or composed_permutations");
}

std::vector<PackedSuffix> build_stream2_suffix_list(std::uint32_t radius, std::uint64_t max_count) {
    std::vector<PackedSuffix> suffixes;
    std::vector<PackedSuffix> frontier;
    suffixes.push_back(PackedSuffix{});
    frontier.push_back(PackedSuffix{});
    for (std::uint32_t depth = 0; depth < radius; ++depth) {
        std::vector<PackedSuffix> next;
        next.reserve(frontier.size() * MOVE_COUNT);
        for (const PackedSuffix& suffix : frontier) {
            for (std::uint8_t move = 0; move < MOVE_COUNT; ++move) {
                const PackedSuffix child = append_suffix_move(suffix, move);
                if (max_count != 0ULL && suffixes.size() >= max_count) {
                    throw std::runtime_error("stream2 suffix table exceeded BEAM_STREAM2_SUFFIX_MAX_COUNT");
                }
                suffixes.push_back(child);
                next.push_back(child);
            }
        }
        frontier = std::move(next);
    }
    return suffixes;
}

std::vector<std::uint8_t> build_stream2_composed_permutations(
    const std::vector<PackedSuffix>& suffixes,
    const std::vector<std::uint8_t>& generators) {
    std::vector<std::uint8_t> permutations(
        static_cast<std::uint64_t>(suffixes.size()) * STATE_STORAGE_LEN);
    for (std::size_t suffix_id = 0; suffix_id < suffixes.size(); ++suffix_id) {
        std::uint8_t perm[STATE_STORAGE_LEN]{};
        for (std::uint32_t p = 0; p < STATE_STORAGE_LEN; ++p) {
            perm[p] = static_cast<std::uint8_t>(p);
        }
        const PackedSuffix suffix = suffixes[suffix_id];
        for (std::uint32_t step = 0; step < suffix.len; ++step) {
            const std::uint8_t move = packed_suffix_move_host(suffix, step);
            std::uint8_t next_perm[STATE_STORAGE_LEN]{};
            const std::uint64_t base = static_cast<std::uint64_t>(move) * STATE_STORAGE_LEN;
            for (std::uint32_t p = 0; p < STATE_STORAGE_LEN; ++p) {
                next_perm[p] = perm[generators[base + p]];
            }
            for (std::uint32_t p = 0; p < STATE_STORAGE_LEN; ++p) {
                perm[p] = next_perm[p];
            }
        }
        for (std::uint32_t p = 0; p < STATE_STORAGE_LEN; ++p) {
            permutations[static_cast<std::uint64_t>(suffix_id) * STATE_STORAGE_LEN + p] = perm[p];
        }
    }
    return permutations;
}

Stream2SuffixRuntime build_stream2_suffix_runtime(const std::vector<std::uint8_t>& generators) {
    Stream2SuffixRuntime runtime;
    runtime.radius = env_u32("BEAM_STREAM2_SUFFIX_RADIUS", 0);
    const char* backend_env = std::getenv("BEAM_STREAM2_SUFFIX_BACKEND");
    runtime.backend_name =
        (backend_env == nullptr || backend_env[0] == '\0') ? "base_generators" : backend_env;
    runtime.backend = parse_stream2_suffix_backend(runtime.backend_name);
    if (runtime.radius == 0U) {
        return runtime;
    }
    if (runtime.radius > 3U) {
        throw std::runtime_error("BEAM_STREAM2_SUFFIX_RADIUS must be <= 3 for direct Stream2 suffix scan");
    }
    if (generators.size() != MOVE_COUNT * STATE_STORAGE_LEN) {
        throw std::runtime_error("stream2 suffix builder requires flat p900 generators");
    }
    const std::uint64_t max_count = env_u64("BEAM_STREAM2_SUFFIX_MAX_COUNT", 0);
    runtime.suffixes = build_stream2_suffix_list(runtime.radius, max_count);
    runtime.entry_count = runtime.suffixes.size();
    if (runtime.entry_count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error("stream2 suffix table exceeds uint32 range");
    }
    std::vector<std::uint64_t> packed_moves(runtime.suffixes.size());
    std::vector<std::uint8_t> lengths(runtime.suffixes.size());
    for (std::size_t i = 0; i < runtime.suffixes.size(); ++i) {
        packed_moves[i] = runtime.suffixes[i].moves;
        lengths[i] = runtime.suffixes[i].len;
    }
    BEAM_CUDA_CHECK(cudaMalloc(
        &runtime.device_packed_moves,
        packed_moves.size() * sizeof(std::uint64_t)));
    BEAM_CUDA_CHECK(cudaMalloc(&runtime.device_lengths, lengths.size() * sizeof(std::uint8_t)));
    BEAM_CUDA_CHECK(cudaMemcpy(
        runtime.device_packed_moves,
        packed_moves.data(),
        packed_moves.size() * sizeof(std::uint64_t),
        cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(
        runtime.device_lengths,
        lengths.data(),
        lengths.size() * sizeof(std::uint8_t),
        cudaMemcpyHostToDevice));
    runtime.device_bytes =
        packed_moves.size() * sizeof(std::uint64_t) +
        lengths.size() * sizeof(std::uint8_t);
    if (runtime.backend == STREAM2_SUFFIX_BACKEND_COMPOSED_PERMUTATIONS) {
        const std::vector<std::uint8_t> permutations =
            build_stream2_composed_permutations(runtime.suffixes, generators);
        BEAM_CUDA_CHECK(cudaMalloc(
            &runtime.device_composed_permutations,
            permutations.size() * sizeof(std::uint8_t)));
        BEAM_CUDA_CHECK(cudaMemcpy(
            runtime.device_composed_permutations,
            permutations.data(),
            permutations.size() * sizeof(std::uint8_t),
            cudaMemcpyHostToDevice));
        runtime.device_bytes += permutations.size() * sizeof(std::uint8_t);
    }
    return runtime;
}

const char* track_location_name(std::uint32_t location) {
    switch (location) {
        case 1U:
            return "clean";
        case 2U:
            return "dirty";
        case 3U:
            return "active_spill";
        case 4U:
            return "inactive_spill";
        case 0U:
            return "none";
        default:
            return "unknown";
    }
}

const char* track_stream4_phase_name(std::uint32_t phase) {
    switch (phase) {
        case 1U:
            return "after_stream3";
        case 2U:
            return "stream4_input";
        case 3U:
            return "stream4_output";
        default:
            return "unknown";
    }
}

#if BEAM_DEBUG_PATH_TRACE || BEAM_DEBUG_INFERENCE_TRACE
struct TrackedSolutionPrefix {
    std::vector<std::uint8_t> moves;
    std::vector<Hash128> prefix_hashes;
    std::vector<bool> survived;
    std::vector<std::uint64_t> final_indices;
    bool enabled = false;
    bool stop_after_path = false;
    std::uint32_t missing_stop_extra_depths = UINT32_MAX;
    bool has_first_missing_depth = false;
    std::uint32_t first_missing_depth = UINT32_MAX;

    void initialize(
        std::uint64_t puzzle_id,
        const State128& initial,
        const std::vector<std::uint8_t>& generators,
        const ZobristTable& zobrist,
        const std::vector<std::string>& move_names) {
        const char* path_env = std::getenv("BEAM_TRACK_SOLUTION_PATH");
        if (path_env == nullptr || path_env[0] == '\0') {
            return;
        }
        enabled = true;
        stop_after_path = env_bool("BEAM_STOP_AFTER_TRACKED_PATH", false);
        if (env_present("BEAM_STOP_AFTER_TRACKED_MISSING_EXTRA_DEPTHS")) {
            missing_stop_extra_depths =
                env_u32("BEAM_STOP_AFTER_TRACKED_MISSING_EXTRA_DEPTHS", UINT32_MAX);
        }
        moves = parse_solution_path_text(path_env, move_names);
        prefix_hashes.reserve(moves.size());
        survived.assign(moves.size(), false);
        final_indices.assign(moves.size(), UINT64_MAX);
        State128 state = initial;
        for (std::uint8_t move : moves) {
            state = apply_move_flat_host(state, generators, move);
            prefix_hashes.push_back(hash_state(state, zobrist));
        }
        std::cout << "track_solution_start=1"
                  << " puzzle_id=" << puzzle_id
                  << " path_len=" << moves.size()
                  << " stop_after_path=" << (stop_after_path ? 1 : 0)
                  << " missing_extra_depths=" << missing_stop_extra_depths
                  << " path=" << path_env << "\n";
    }

    const Hash128* hash_for_depth(std::uint32_t depth) const {
        if (!enabled || depth >= prefix_hashes.size()) {
            return nullptr;
        }
        return &prefix_hashes[depth];
    }

    GeneratedTrackRequest generated_request_for_depth(std::uint32_t depth) const {
        GeneratedTrackRequest request{};
        if (!enabled || depth >= moves.size()) {
            return request;
        }
        if (depth == 0U) {
            request.enabled = true;
            request.parent_idx = 0;
            request.move = moves[depth];
            return request;
        }
        const std::uint64_t parent_idx = final_indices[depth - 1U];
        if (parent_idx == UINT64_MAX) {
            return request;
        }
        request.enabled = true;
        request.parent_idx = parent_idx;
        request.move = moves[depth];
        return request;
    }

    void log_generated(
        std::uint64_t puzzle_id,
        std::uint32_t depth,
        const GeneratedTrackResult& result,
        const std::vector<std::string>& move_names) const {
        if (!enabled || depth >= moves.size()) {
            return;
        }
        const std::int64_t threshold_margin =
            result.found && result.current_threshold != UINT32_THRESHOLD_MAX
                ? static_cast<std::int64_t>(result.score_key) - static_cast<std::int64_t>(result.current_threshold)
                : 0;
        std::cout << "track_solution_generated"
                  << " puzzle_id=" << puzzle_id
                  << " depth=" << depth
                  << " prefix_len=" << depth + 1U
                  << " expected_parent_idx="
                  << (result.enabled ? result.request_parent_idx : UINT64_MAX)
                  << " expected_move=" << move_names[moves[depth]]
                  << " request_enabled=" << (result.enabled ? 1 : 0)
                  << " found=" << (result.found ? 1 : 0)
                  << " ring=" << result.ring
                  << " ring_slot=" << result.ring_slot
                  << " job=" << result.job
                  << " parent_base=" << result.parent_base
                  << " parent_local=" << result.parent_local
                  << " count=" << result.count
                  << " payload_id=" << result.payload_id
                  << " score_ring_offset=" << result.score_ring_offset
                  << " score_key=" << (result.found ? result.score_key : UINT32_THRESHOLD_MAX)
                  << " current_threshold=" << result.current_threshold
                  << " threshold_pass="
                  << (result.found && result.score_key <= result.current_threshold ? 1 : 0)
                  << " threshold_margin=" << threshold_margin
                  << " hash_lo=" << (result.found ? result.hash.lo : UINT64_MAX)
                  << " hash_hi=" << (result.found ? result.hash.hi : UINT64_MAX)
                  << " owner=" << static_cast<std::uint32_t>(result.owner)
                  << " shard=" << result.shard
                  << "\n";
    }

    void log_stream3(
        std::uint64_t puzzle_id,
        std::uint32_t depth,
        const Stream3TrackResult& result,
        const std::vector<std::string>& move_names) const {
        if (!enabled || depth >= moves.size() || !result.enabled) {
            return;
        }
        std::cout << "track_solution_stream3_path"
                  << " puzzle_id=" << puzzle_id
                  << " depth=" << depth
                  << " prefix_len=" << depth + 1U
                  << " expected_move=" << move_names[moves[depth]]
                  << " scanned=" << (result.scanned ? 1 : 0)
                  << " unique_found=" << (result.unique_found ? 1 : 0)
                  << " unique_local=" << result.unique_local
                  << " unique_count=" << result.unique_count
                  << " unique_score_key=" << result.unique_score_key
                  << " unique_payload_id=" << result.unique_payload_id
                  << " unique_parent_idx=" << result.unique_parent_idx
                  << " unique_move=" << static_cast<std::uint32_t>(result.unique_move)
                  << " partition_found=" << (result.partition_found ? 1 : 0)
                  << " partition_local=" << result.partition_local
                  << " local_pending_count=" << result.local_pending_count
                  << " partition_unique_count=" << result.partition_unique_count
                  << " group_offset=" << result.group_offset
                  << " group_raw_count=" << result.group_raw_count
                  << " local_in_group=" << result.local_in_group
                  << " shard_write_count=" << result.shard_write_count
                  << " shard_spill_count=" << result.shard_spill_count
                  << " shard_spill_offset=" << result.shard_spill_offset
                  << " spill_idx=" << result.spill_idx
                  << " spill_capacity=" << result.spill_capacity
                  << " spill_capacity_drop=" << (result.spill_capacity_drop ? 1 : 0)
                  << " clean_count=" << result.clean_count
                  << " dirty_count=" << result.dirty_count
                  << " processing_flag=" << result.processing_flag
                  << " active_spill_count=" << result.active_spill_count
                  << " inactive_spill_count=" << result.inactive_spill_count
                  << "\n";
    }

    void log_stream4(
        std::uint64_t puzzle_id,
        std::uint32_t depth,
        const Stream4TrackResult& result,
        const std::vector<Stream4TrackEvent>& events,
        const std::vector<std::string>& move_names) const {
        if (!enabled || depth >= moves.size() || !result.enabled) {
            return;
        }
        const std::int64_t input_threshold_margin =
            result.input_found && result.input_threshold != UINT32_THRESHOLD_MAX
                ? static_cast<std::int64_t>(result.score_key) - static_cast<std::int64_t>(result.input_threshold)
                : 0;
        std::cout << "track_solution_stream4"
                  << " puzzle_id=" << puzzle_id
                  << " depth=" << depth
                  << " prefix_len=" << depth + 1U
                  << " expected_move=" << move_names[moves[depth]]
                  << " score_key=" << result.score_key
                  << " shard=" << result.shard
                  << " hash_lo=" << result.hash.lo
                  << " hash_hi=" << result.hash.hi
                  << " after_stream3_scanned=" << (result.after_stream3_scanned ? 1 : 0)
                  << " after_stream3_found=" << (result.after_stream3_found ? 1 : 0)
                  << " after_stream3_location=" << track_location_name(result.after_stream3_location)
                  << " after_stream3_local=" << result.after_stream3_local
                  << " after_stream3_clean_count=" << result.after_stream3_clean_count
                  << " after_stream3_dirty_count=" << result.after_stream3_dirty_count
                  << " after_stream3_active_spill_count=" << result.after_stream3_active_spill_count
                  << " after_stream3_inactive_spill_count=" << result.after_stream3_inactive_spill_count
                  << " after_stream3_threshold=" << result.after_stream3_threshold
                  << " input_scan_count=" << result.input_scan_count
                  << " input_found=" << (result.input_found ? 1 : 0)
                  << " input_slot=" << result.input_slot
                  << " input_job=" << result.input_job
                  << " input_location=" << track_location_name(result.input_location)
                  << " input_local=" << result.input_local
                  << " input_clean_count=" << result.input_clean_count
                  << " input_dirty_count=" << result.input_dirty_count
                  << " input_threshold=" << result.input_threshold
                  << " input_threshold_pass="
                  << (result.input_found && result.score_key <= result.input_threshold ? 1 : 0)
                  << " input_threshold_margin=" << input_threshold_margin
                  << " output_scan_count=" << result.output_scan_count
                  << " output_found=" << (result.output_found ? 1 : 0)
                  << " output_slot=" << result.output_slot
                  << " output_job=" << result.output_job
                  << " output_local=" << result.output_local
                  << " output_clean_count=" << result.output_clean_count
                  << " output_dirty_count=" << result.output_dirty_count
                  << " output_threshold=" << result.output_threshold
                  << "\n";
        for (std::uint32_t event_index = 0; event_index < events.size(); ++event_index) {
            const Stream4TrackEvent& event = events[event_index];
            const std::int64_t threshold_margin =
                event.found && event.threshold != UINT32_THRESHOLD_MAX
                    ? static_cast<std::int64_t>(event.score_key) - static_cast<std::int64_t>(event.threshold)
                    : 0;
            std::cout << "track_solution_stream4_event"
                      << " puzzle_id=" << puzzle_id
                      << " depth=" << depth
                      << " prefix_len=" << depth + 1U
                      << " event_index=" << event_index
                      << " phase=" << track_stream4_phase_name(event.phase)
                      << " found=" << (event.found ? 1 : 0)
                      << " shard=" << event.shard
                      << " slot=" << event.slot
                      << " job=" << event.job
                      << " location=" << track_location_name(event.location)
                      << " local=" << event.local
                      << " clean_count=" << event.clean_count
                      << " dirty_count=" << event.dirty_count
                      << " active_spill_count=" << event.active_spill_count
                      << " inactive_spill_count=" << event.inactive_spill_count
                      << " threshold=" << event.threshold
                      << " score_key=" << event.score_key
                      << " threshold_pass=" << (event.found && event.score_key <= event.threshold ? 1 : 0)
                      << " threshold_margin=" << threshold_margin
                      << "\n";
        }
    }

    void log_candidate_fields(
        const char* prefix,
        std::uint64_t puzzle_id,
        std::uint32_t depth,
        std::uint32_t prefix_len,
        bool found,
        std::uint32_t matches,
        std::uint64_t first_index,
        std::uint64_t best_index,
        std::uint32_t best_score_key,
        std::uint32_t final_threshold,
        std::uint64_t parent_idx,
        std::uint32_t route_packed,
        std::uint32_t shard,
        std::uint32_t local,
        std::uint32_t final_candidate_count,
        const std::vector<std::string>& move_names) {
        const std::uint8_t move = found ? unpack_move(route_packed) : 0U;
        const std::uint16_t source_rank = found ? unpack_source_rank(route_packed) : 0U;
        const std::uint8_t owner = found ? unpack_owner(route_packed) : 0U;
        const std::int64_t threshold_margin =
            found && final_threshold != UINT32_THRESHOLD_MAX
                ? static_cast<std::int64_t>(best_score_key) - static_cast<std::int64_t>(final_threshold)
                : 0;
        std::cout << prefix
                  << " puzzle_id=" << puzzle_id
                  << " depth=" << depth
                  << " prefix_len=" << prefix_len
                  << " expected_move="
                  << (depth < moves.size() ? move_names[moves[depth]] : "")
                  << " found=" << (found ? 1 : 0)
                  << " matches=" << matches
                  << " first_index=" << first_index
                  << " best_index=" << best_index
                  << " best_score_key=" << (found ? best_score_key : UINT32_THRESHOLD_MAX)
                  << " final_threshold=" << final_threshold
                  << " threshold_pass="
                  << (found && best_score_key <= final_threshold ? 1 : 0)
                  << " threshold_margin=" << threshold_margin
                  << " parent_idx=" << (found ? parent_idx : UINT64_MAX)
                  << " route_packed=" << (found ? route_packed : UINT32_MAX)
                  << " source_rank=" << static_cast<std::uint32_t>(source_rank)
                  << " owner=" << static_cast<std::uint32_t>(owner)
                  << " move=" << static_cast<std::uint32_t>(move)
                  << " move_name=" << (found ? move_names[move] : "")
                  << " shard=" << shard
                  << " local=" << local
                  << " final_candidate_count=" << final_candidate_count
                  << "\n";
    }

    void log_prefinal(
        std::uint64_t puzzle_id,
        std::uint32_t depth,
        const FinalizeDepthState& final_state,
        const std::vector<std::string>& move_names) {
        if (!enabled || depth >= prefix_hashes.size() || !final_state.tracked_prefinal_enabled) {
            return;
        }
        const bool found = final_state.tracked_prefinal_matches != 0U;
        log_candidate_fields(
            "track_solution_prefinal",
            puzzle_id,
            depth,
            depth + 1U,
            found,
            final_state.tracked_prefinal_matches,
            final_state.tracked_prefinal_first_index,
            final_state.tracked_prefinal_best_index,
            final_state.tracked_prefinal_best_score_key,
            final_state.final_threshold,
            final_state.tracked_prefinal_best_parent_idx,
            final_state.tracked_prefinal_best_route_packed,
            final_state.tracked_prefinal_best_shard,
            final_state.tracked_prefinal_best_local,
            final_state.final_candidate_count,
            move_names);
    }

    void scan_depth(
        std::uint64_t puzzle_id,
        std::uint32_t depth,
        const CandidateMeta* candidates,
        std::uint32_t count,
        std::uint32_t final_threshold,
        const std::vector<std::string>& move_names) {
        if (!enabled || depth >= prefix_hashes.size()) {
            return;
        }
        const Hash128 target = prefix_hashes[depth];
        std::uint32_t match_count = 0;
        std::uint32_t best_score = UINT32_THRESHOLD_MAX;
        std::uint64_t first_index = 0;
        std::uint64_t best_index = 0;
        CandidateMeta best_candidate{};
        for (std::uint32_t i = 0; i < count; ++i) {
            const CandidateMeta& candidate = candidates[i];
            if (candidate.hash == target) {
                if (match_count == 0U) {
                    first_index = i;
                    best_index = i;
                    best_candidate = candidate;
                    best_score = candidate.score_key;
                } else if (
                    candidate.score_key < best_score ||
                    (candidate.score_key == best_score && candidate.parent_idx < best_candidate.parent_idx) ||
                    (candidate.score_key == best_score && candidate.parent_idx == best_candidate.parent_idx &&
                        candidate.route_packed < best_candidate.route_packed)) {
                    best_index = i;
                    best_candidate = candidate;
                    best_score = candidate.score_key;
                }
                ++match_count;
            }
        }
        survived[depth] = match_count != 0U;
        final_indices[depth] = survived[depth] ? best_index : UINT64_MAX;
        if (!survived[depth] && !has_first_missing_depth) {
            has_first_missing_depth = true;
            first_missing_depth = depth;
            std::cout << "track_solution_first_missing"
                      << " puzzle_id=" << puzzle_id
                      << " depth=" << depth
                      << " prefix_len=" << depth + 1U
                      << " stop_after_depth="
                      << (missing_stop_extra_depths == UINT32_MAX
                              ? UINT32_MAX
                              : first_missing_depth + missing_stop_extra_depths)
                      << "\n";
        }
        log_candidate_fields(
            "track_solution_prefix",
            puzzle_id,
            depth,
            depth + 1U,
            survived[depth],
            match_count,
            first_index,
            best_index,
            best_score,
            final_threshold,
            best_candidate.parent_idx,
            best_candidate.route_packed,
            UINT32_MAX,
            UINT32_MAX,
            count,
            move_names);
    }

    bool should_stop_after_missing(std::uint32_t depth) const {
        return enabled &&
            has_first_missing_depth &&
            missing_stop_extra_depths != UINT32_MAX &&
            depth >= first_missing_depth + missing_stop_extra_depths;
    }
};
#else
struct TrackedSolutionPrefix {};
#endif

enum class CandidateHistoryMode : std::uint8_t {
    Ram,
    Disk,
    StaticHybrid
};

enum class HistoryStorageLocation : std::uint8_t {
    None,
    Ram,
    Disk
};

CandidateHistoryMode parse_history_mode() {
    const char* value = std::getenv("BEAM_HISTORY_MODE");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "ram") == 0) {
        return CandidateHistoryMode::Ram;
    }
    if (std::strcmp(value, "disk") == 0) {
        return CandidateHistoryMode::Disk;
    }
    if (std::strcmp(value, "static_hybrid") == 0 || std::strcmp(value, "hybrid") == 0) {
        return CandidateHistoryMode::StaticHybrid;
    }
    throw std::runtime_error("BEAM_HISTORY_MODE must be ram, disk, static_hybrid, or hybrid");
}

const char* history_mode_name(CandidateHistoryMode mode) {
    switch (mode) {
    case CandidateHistoryMode::Ram:
        return "ram";
    case CandidateHistoryMode::Disk:
        return "disk";
    case CandidateHistoryMode::StaticHybrid:
        return "static_hybrid";
    }
    return "unknown";
}

struct HistoryEntry {
    std::uint64_t parent_idx = 0;
    std::uint32_t route_packed = 0;
    std::uint32_t pad = 0;
};

static_assert(sizeof(HistoryEntry) == 16);
static_assert(alignof(HistoryEntry) == 8);

struct HistoryBudgetEstimate {
    std::uint32_t effective_depth = 0;
    std::uint32_t target_beam_depth = 0;
    std::uint64_t states_before_target_beam = 0;
    std::uint64_t required_entries = 0;
};

HistoryBudgetEstimate estimate_history_budget_entries(
    std::uint32_t depth_limit,
    std::uint32_t solved_neighborhood_radius,
    std::uint32_t stream2_suffix_radius,
    std::uint64_t beam_entries) {
    HistoryBudgetEstimate estimate{};
    const std::uint32_t suffix_radius =
        solved_neighborhood_radius > std::numeric_limits<std::uint32_t>::max() - stream2_suffix_radius
            ? std::numeric_limits<std::uint32_t>::max()
            : solved_neighborhood_radius + stream2_suffix_radius;
    estimate.effective_depth = depth_limit > suffix_radius ? depth_limit - suffix_radius : 0U;
    if (estimate.effective_depth == 0U || beam_entries == 0ULL) {
        return estimate;
    }

    std::uint64_t frontier_bound = 1ULL;
    for (std::uint32_t depth = 0; depth < estimate.effective_depth; ++depth) {
        if (frontier_bound < beam_entries) {
            if (frontier_bound > std::numeric_limits<std::uint64_t>::max() / MOVE_COUNT) {
                frontier_bound = beam_entries;
            } else {
                frontier_bound = std::min<std::uint64_t>(
                    beam_entries,
                    frontier_bound * static_cast<std::uint64_t>(MOVE_COUNT));
            }
        }

        if (frontier_bound >= beam_entries) {
            estimate.target_beam_depth = depth;
            const std::uint64_t full_depths =
                static_cast<std::uint64_t>(estimate.effective_depth - depth);
            if (full_depths > std::numeric_limits<std::uint64_t>::max() / beam_entries) {
                throw std::overflow_error("static hybrid history required entries overflow");
            }
            estimate.required_entries = estimate.states_before_target_beam + full_depths * beam_entries;
            return estimate;
        }

        if (estimate.states_before_target_beam >
            std::numeric_limits<std::uint64_t>::max() - frontier_bound) {
            throw std::overflow_error("static hybrid history prefull entries overflow");
        }
        estimate.states_before_target_beam += frontier_bound;
    }

    estimate.target_beam_depth = estimate.effective_depth;
    estimate.required_entries = estimate.states_before_target_beam;
    return estimate;
}

struct CpuCandidateHistory {
    static constexpr std::uint32_t kWriteChunkEntries = 1U << 20U;

    struct DepthLocation {
        HistoryStorageLocation location = HistoryStorageLocation::None;
        std::uint64_t offset_entries = 0;
    };

    struct MaterializeResult {
        std::uint32_t depth_index = 0;
        HistoryStorageLocation location = HistoryStorageLocation::None;
        std::uint64_t offset_entries = 0;
        std::vector<HistoryEntry> ram_data;
    };

    struct Slot {
        CandidateMeta* host = nullptr;
        std::vector<HistoryEntry> staging;
        std::uint32_t capacity = 0;
        std::uint32_t count = 0;
        std::uint32_t depth_index = 0;
        std::filesystem::path path;
        HistoryStorageLocation location = HistoryStorageLocation::None;
        std::uint64_t offset_entries = 0;
        cudaEvent_t copy_done = nullptr;
        bool copy_pending = false;
        bool free = true;
        std::future<MaterializeResult> writer;
    };

    struct PruneResult {
        std::uint32_t depth_index = 0;
        std::vector<HistoryEntry> previous;
        std::vector<HistoryEntry> current;
        std::uint64_t bytes_before = 0;
        std::uint64_t bytes_after = 0;
    };

    struct PruneJob {
        std::uint32_t depth_index = 0;
        std::future<PruneResult> worker;
    };

    CandidateHistoryMode mode = CandidateHistoryMode::Ram;
    std::filesystem::path dir;
    std::vector<std::filesystem::path> depth_files;
    std::vector<std::uint32_t> depth_counts;
    std::vector<DepthLocation> depth_locations;
    std::vector<std::vector<HistoryEntry>> ram_depths;
    std::vector<bool> ram_ready;
    std::vector<bool> prune_dirty;
    std::vector<PruneJob> prune_jobs;
    std::uint64_t bytes_received = 0;
    std::uint64_t bytes_stored = 0;
    std::uint64_t bytes_stored_ram = 0;
    std::uint64_t bytes_stored_disk = 0;
    std::uint64_t bytes_pruned = 0;
    std::uint64_t bytes_pinned_slots = 0;
    std::uint64_t bytes_slot_staging = 0;
    std::uint64_t bytes_static_ram_arena = 0;
    std::uint64_t bytes_static_disk_arena = 0;
    std::uint64_t history_required_bytes = 0;
    HistoryBudgetEstimate budget_estimate{};
    std::uint32_t worker_count = 1;
    bool prune_enabled = true;
    std::vector<Slot> slots;
    cudaStream_t copy_stream = nullptr;
    HistoryEntry* ram_arena = nullptr;
    std::uint64_t ram_arena_entries = 0;
    std::uint64_t ram_entries_used = 0;
    std::uint64_t disk_arena_entries = 0;
    std::uint64_t disk_entries_used = 0;
    std::filesystem::path static_disk_path;
    mutable std::mutex static_disk_mutex;
    mutable std::mutex storage_mutex;
    bool disk_write_failed = false;

    void initialize(
        CandidateHistoryMode selected_mode,
        std::uint32_t capacity,
        std::uint32_t slot_count,
        std::uint32_t selected_worker_count,
        std::uint32_t depth_limit,
        std::uint32_t solved_neighborhood_radius,
        std::uint32_t stream2_suffix_radius,
        std::uint64_t history_ram_budget_bytes = 0,
        std::uint64_t history_disk_budget_bytes = 0,
        const std::filesystem::path& history_disk_path = {}) {
        if (copy_stream != nullptr) {
            throw std::runtime_error("candidate history already initialized");
        }
        if (capacity == 0U || slot_count == 0U || selected_worker_count == 0U) {
            throw std::runtime_error("candidate history capacity, slot count, and worker count must be nonzero");
        }
        mode = selected_mode;
        worker_count = selected_worker_count;
        std::filesystem::create_directories(dir);

        const std::uint64_t capacity_entries = static_cast<std::uint64_t>(capacity);
        const std::uint64_t slot_entries = static_cast<std::uint64_t>(slot_count);
        if (slot_entries > std::numeric_limits<std::uint64_t>::max() / capacity_entries ||
            slot_entries * capacity_entries > std::numeric_limits<std::uint64_t>::max() / sizeof(CandidateMeta)) {
            throw std::overflow_error("candidate history pinned slot bytes overflow");
        }
        bytes_pinned_slots = slot_entries * capacity_entries * sizeof(CandidateMeta);
        const std::uint64_t staging_entries_per_slot =
            std::min<std::uint64_t>(capacity_entries, kWriteChunkEntries);
        if (slot_entries > std::numeric_limits<std::uint64_t>::max() / staging_entries_per_slot ||
            slot_entries * staging_entries_per_slot > std::numeric_limits<std::uint64_t>::max() / sizeof(HistoryEntry)) {
            throw std::overflow_error("candidate history staging bytes overflow");
        }
        bytes_slot_staging = slot_entries * staging_entries_per_slot * sizeof(HistoryEntry);

        if (selected_mode == CandidateHistoryMode::StaticHybrid) {
            if (history_ram_budget_bytes <= bytes_pinned_slots + bytes_slot_staging) {
                throw std::runtime_error(
                    "static hybrid history RAM budget cannot fit pinned slots and staging: ram_budget=" +
                    std::to_string(history_ram_budget_bytes) +
                    " pinned_slots=" + std::to_string(bytes_pinned_slots) +
                    " staging=" + std::to_string(bytes_slot_staging));
            }
            budget_estimate = estimate_history_budget_entries(
                depth_limit,
                solved_neighborhood_radius,
                stream2_suffix_radius,
                capacity_entries);
            if (budget_estimate.required_entries >
                std::numeric_limits<std::uint64_t>::max() / sizeof(HistoryEntry)) {
                throw std::overflow_error("static hybrid history required bytes overflow");
            }
            history_required_bytes = budget_estimate.required_entries * sizeof(HistoryEntry);
            bytes_static_ram_arena = history_ram_budget_bytes - bytes_pinned_slots - bytes_slot_staging;
            ram_arena_entries = bytes_static_ram_arena / sizeof(HistoryEntry);
            bytes_static_ram_arena = ram_arena_entries * sizeof(HistoryEntry);
            disk_arena_entries = history_disk_budget_bytes / sizeof(HistoryEntry);
            bytes_static_disk_arena = disk_arena_entries * sizeof(HistoryEntry);
            if (history_required_bytes > bytes_static_ram_arena + bytes_static_disk_arena) {
                throw std::runtime_error(
                    "static hybrid history budget too small: required=" +
                    std::to_string(history_required_bytes) +
                    " ram_entries=" + std::to_string(bytes_static_ram_arena) +
                    " disk_entries=" + std::to_string(bytes_static_disk_arena) +
                    " pinned_slots=" + std::to_string(bytes_pinned_slots) +
                    " staging=" + std::to_string(bytes_slot_staging) +
                    " effective_depth=" + std::to_string(budget_estimate.effective_depth) +
                    " target_beam_depth=" + std::to_string(budget_estimate.target_beam_depth) +
                    " states_before_target_beam=" +
                    std::to_string(budget_estimate.states_before_target_beam));
            }
            if (ram_arena_entries != 0ULL) {
                ram_arena = static_cast<HistoryEntry*>(std::malloc(bytes_static_ram_arena));
                if (ram_arena == nullptr) {
                    throw std::bad_alloc();
                }
            }
            if (bytes_static_disk_arena != 0ULL) {
                static_disk_path = history_disk_path.empty() ? dir / "history_static_arena.bin" : history_disk_path;
                const std::filesystem::path parent = static_disk_path.parent_path();
                if (!parent.empty()) {
                    std::filesystem::create_directories(parent);
                }
                {
                    std::ofstream create_file(static_disk_path, std::ios::binary | std::ios::trunc);
                    if (!create_file) {
                        throw std::runtime_error("cannot create static history disk arena: " + static_disk_path.string());
                    }
                }
                std::filesystem::resize_file(static_disk_path, bytes_static_disk_arena);
            }
            prune_enabled = false;
        }

        BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
        slots.resize(slot_count);
        for (Slot& slot : slots) {
            slot.capacity = capacity;
            slot.staging.resize(static_cast<std::size_t>(staging_entries_per_slot));
            BEAM_CUDA_CHECK(cudaHostAlloc(
                reinterpret_cast<void**>(&slot.host),
                static_cast<std::uint64_t>(capacity) * sizeof(CandidateMeta),
                cudaHostAllocPortable));
            BEAM_CUDA_CHECK(cudaEventCreateWithFlags(&slot.copy_done, cudaEventDisableTiming));
        }
    }

    static HistoryEntry compress_candidate(const CandidateMeta& candidate) {
        return HistoryEntry{candidate.parent_idx, candidate.route_packed, 0U};
    }

    static std::vector<HistoryEntry> materialize_depth_ram_vector(
        const CandidateMeta* host,
        std::uint32_t count) {
        std::vector<HistoryEntry> entries(static_cast<std::size_t>(count));
        for (std::uint32_t i = 0; i < count; ++i) {
            entries[static_cast<std::size_t>(i)] = compress_candidate(host[i]);
        }
        return entries;
    }

    static void fill_history_staging(
        HistoryEntry* staging,
        const CandidateMeta* host,
        std::uint64_t begin,
        std::uint32_t count) {
        for (std::uint32_t i = 0; i < count; ++i) {
            staging[i] = compress_candidate(host[begin + i]);
        }
    }

    static void write_depth_file(
        const std::filesystem::path& path,
        const CandidateMeta* host,
        std::uint32_t count,
        HistoryEntry* staging,
        std::uint32_t staging_capacity) {
        std::ofstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("cannot open candidate history file for write: " + path.string());
        }
        for (std::uint64_t begin = 0; begin < count; begin += staging_capacity) {
            const std::uint32_t chunk =
                static_cast<std::uint32_t>(std::min<std::uint64_t>(staging_capacity, count - begin));
            fill_history_staging(staging, host, begin, chunk);
            file.write(
                reinterpret_cast<const char*>(staging),
                static_cast<std::streamsize>(static_cast<std::uint64_t>(chunk) * sizeof(HistoryEntry)));
        }
        if (!file) {
            throw std::runtime_error("candidate history write failed: " + path.string());
        }
    }

    void write_static_disk(
        std::uint64_t offset_entries,
        const CandidateMeta* host,
        std::uint32_t count,
        HistoryEntry* staging,
        std::uint32_t staging_capacity) {
        std::lock_guard<std::mutex> lock(static_disk_mutex);
        std::fstream file(static_disk_path, std::ios::binary | std::ios::in | std::ios::out);
        if (!file) {
            throw std::runtime_error("cannot open static history disk arena: " + static_disk_path.string());
        }
        file.seekp(static_cast<std::streamoff>(offset_entries * sizeof(HistoryEntry)), std::ios::beg);
        for (std::uint64_t begin = 0; begin < count; begin += staging_capacity) {
            const std::uint32_t chunk =
                static_cast<std::uint32_t>(std::min<std::uint64_t>(staging_capacity, count - begin));
            fill_history_staging(staging, host, begin, chunk);
            file.write(
                reinterpret_cast<const char*>(staging),
                static_cast<std::streamsize>(static_cast<std::uint64_t>(chunk) * sizeof(HistoryEntry)));
        }
        if (!file) {
            throw std::runtime_error("static history disk arena write failed: " + static_disk_path.string());
        }
    }

    void write_static_ram(std::uint64_t offset_entries, const CandidateMeta* host, std::uint32_t count) {
        if (ram_arena == nullptr || offset_entries + count > ram_arena_entries) {
            throw std::runtime_error("static history RAM arena write exceeds capacity");
        }
        HistoryEntry* out = ram_arena + offset_entries;
        for (std::uint32_t i = 0; i < count; ++i) {
            out[i] = compress_candidate(host[i]);
        }
    }

    DepthLocation reserve_static_location(std::uint32_t count) {
        const std::uint64_t count_entries = static_cast<std::uint64_t>(count);
        std::lock_guard<std::mutex> lock(storage_mutex);
        if (!disk_write_failed && disk_entries_used + count_entries <= disk_arena_entries) {
            const std::uint64_t offset = disk_entries_used;
            disk_entries_used += count_entries;
            return DepthLocation{HistoryStorageLocation::Disk, offset};
        }
        if (ram_entries_used + count_entries <= ram_arena_entries) {
            const std::uint64_t offset = ram_entries_used;
            ram_entries_used += count_entries;
            return DepthLocation{HistoryStorageLocation::Ram, offset};
        }
        throw std::runtime_error(
            "static hybrid history arena exhausted: count=" + std::to_string(count) +
            " ram_used=" + std::to_string(ram_entries_used * sizeof(HistoryEntry)) +
            " ram_capacity=" + std::to_string(bytes_static_ram_arena) +
            " disk_used=" + std::to_string(disk_entries_used * sizeof(HistoryEntry)) +
            " disk_capacity=" + std::to_string(bytes_static_disk_arena));
    }

    DepthLocation reserve_static_ram_fallback(std::uint32_t count, const std::string& disk_error) {
        const std::uint64_t count_entries = static_cast<std::uint64_t>(count);
        std::lock_guard<std::mutex> lock(storage_mutex);
        disk_write_failed = true;
        if (ram_entries_used + count_entries <= ram_arena_entries) {
            const std::uint64_t offset = ram_entries_used;
            ram_entries_used += count_entries;
            return DepthLocation{HistoryStorageLocation::Ram, offset};
        }
        throw std::runtime_error(
            "static hybrid history disk write failed and RAM fallback is exhausted: disk_error=" + disk_error +
            " count=" + std::to_string(count) +
            " ram_used=" + std::to_string(ram_entries_used * sizeof(HistoryEntry)) +
            " ram_capacity=" + std::to_string(bytes_static_ram_arena));
    }

    MaterializeResult materialize_depth(
        CandidateHistoryMode selected_mode,
        const std::filesystem::path& path,
        const CandidateMeta* host,
        std::uint32_t count,
        std::uint32_t depth_index,
        HistoryStorageLocation location,
        std::uint64_t offset_entries,
        HistoryEntry* staging,
        std::uint32_t staging_capacity) {
        MaterializeResult result{depth_index, location, offset_entries, {}};
        if (selected_mode == CandidateHistoryMode::Ram) {
            result.ram_data = materialize_depth_ram_vector(host, count);
            return result;
        }
        if (selected_mode == CandidateHistoryMode::Disk) {
            write_depth_file(path, host, count, staging, staging_capacity);
            return result;
        }
        if (location == HistoryStorageLocation::Disk) {
            try {
                write_static_disk(offset_entries, host, count, staging, staging_capacity);
            } catch (const std::exception& ex) {
                const DepthLocation fallback = reserve_static_ram_fallback(count, ex.what());
                write_static_ram(fallback.offset_entries, host, count);
                result.location = fallback.location;
                result.offset_entries = fallback.offset_entries;
            }
            return result;
        }
        if (location == HistoryStorageLocation::Ram) {
            write_static_ram(offset_entries, host, count);
            return result;
        }
        throw std::runtime_error("static hybrid history materialize missing storage location");
    }

    static PruneResult prune_adjacent_depths(
        std::uint32_t depth_index,
        std::vector<HistoryEntry> previous,
        std::vector<HistoryEntry> current) {
        constexpr std::uint32_t invalid = std::numeric_limits<std::uint32_t>::max();
        const std::uint64_t bytes_before =
            (static_cast<std::uint64_t>(previous.size()) + static_cast<std::uint64_t>(current.size())) *
            sizeof(HistoryEntry);
        std::vector<std::uint32_t> remap(previous.size(), invalid);
        for (const HistoryEntry& entry : current) {
            if (entry.parent_idx >= previous.size()) {
                throw std::runtime_error("candidate history prune parent index out of range");
            }
            remap[static_cast<std::size_t>(entry.parent_idx)] = 1U;
        }

        std::vector<HistoryEntry> compact_previous;
        compact_previous.reserve(previous.size());
        for (std::size_t i = 0; i < previous.size(); ++i) {
            if (remap[i] != invalid) {
                remap[i] = static_cast<std::uint32_t>(compact_previous.size());
                compact_previous.push_back(previous[i]);
            }
        }
        for (HistoryEntry& entry : current) {
            entry.parent_idx = remap[static_cast<std::size_t>(entry.parent_idx)];
        }
        const std::uint64_t bytes_after =
            (static_cast<std::uint64_t>(compact_previous.size()) + static_cast<std::uint64_t>(current.size())) *
            sizeof(HistoryEntry);
        return PruneResult{depth_index, std::move(compact_previous), std::move(current), bytes_before, bytes_after};
    }

    bool prune_job_uses_depth(const PruneJob& job, std::uint32_t depth_index) const {
        return job.depth_index == depth_index || job.depth_index + 1U == depth_index;
    }

    bool any_prune_job_uses_pair(std::uint32_t depth_index) const {
        for (const PruneJob& job : prune_jobs) {
            if (prune_job_uses_depth(job, depth_index) || prune_job_uses_depth(job, depth_index + 1U)) {
                return true;
            }
        }
        return false;
    }

    void pump_prune_jobs(bool wait_all) {
        if (!prune_enabled) {
            return;
        }
        bool progressed = true;
        while (progressed) {
            progressed = false;
            for (std::size_t i = 0; i < prune_jobs.size();) {
                PruneJob& job = prune_jobs[i];
                const bool ready = wait_all ||
                    job.worker.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
                if (!ready) {
                    ++i;
                    continue;
                }
                PruneResult result = job.worker.get();
                if (result.depth_index + 1U >= ram_depths.size()) {
                    throw std::runtime_error("candidate history prune result depth index out of range");
                }
                ram_depths[result.depth_index] = std::move(result.previous);
                ram_depths[result.depth_index + 1U] = std::move(result.current);
                depth_counts[result.depth_index] =
                    static_cast<std::uint32_t>(ram_depths[result.depth_index].size());
                depth_counts[result.depth_index + 1U] =
                    static_cast<std::uint32_t>(ram_depths[result.depth_index + 1U].size());
                ram_ready[result.depth_index] = true;
                ram_ready[result.depth_index + 1U] = true;
                prune_dirty[result.depth_index] = false;
                if (result.bytes_before > result.bytes_after) {
                    bytes_stored -= (result.bytes_before - result.bytes_after);
                    bytes_pruned += (result.bytes_before - result.bytes_after);
                }
                if (result.depth_index > 0U) {
                    prune_dirty[result.depth_index - 1U] = true;
                }
                prune_jobs.erase(prune_jobs.begin() + static_cast<std::ptrdiff_t>(i));
                progressed = true;
            }

            while (mode == CandidateHistoryMode::Ram && prune_jobs.size() < worker_count) {
                bool launched = false;
                if (ram_depths.size() >= 2U) {
                    for (std::uint32_t depth = static_cast<std::uint32_t>(ram_depths.size() - 2U);
                         depth != std::numeric_limits<std::uint32_t>::max();
                         --depth) {
                        if (depth + 1U < ram_depths.size() &&
                            prune_dirty[depth] &&
                            ram_ready[depth] &&
                            ram_ready[depth + 1U] &&
                            !any_prune_job_uses_pair(depth)) {
                            std::vector<HistoryEntry> previous = std::move(ram_depths[depth]);
                            std::vector<HistoryEntry> current = std::move(ram_depths[depth + 1U]);
                            ram_ready[depth] = false;
                            ram_ready[depth + 1U] = false;
                            prune_jobs.push_back(PruneJob{
                                depth,
                                std::async(
                                    std::launch::async,
                                    &CpuCandidateHistory::prune_adjacent_depths,
                                    depth,
                                    std::move(previous),
                                    std::move(current))});
                            launched = true;
                            progressed = true;
                            break;
                        }
                        if (depth == 0U) {
                            break;
                        }
                    }
                }
                if (!launched) {
                    break;
                }
            }
            if (!wait_all) {
                break;
            }
        }
    }

    void pump_completed(bool wait_all) {
        bool progressed = true;
        while (progressed) {
            progressed = false;
            for (Slot& slot : slots) {
                if (slot.copy_pending) {
                    const cudaError_t status = wait_all ? cudaEventSynchronize(slot.copy_done) : cudaEventQuery(slot.copy_done);
                    if (status == cudaSuccess) {
                        slot.copy_pending = false;
                        const CandidateHistoryMode selected_mode = mode;
                        const std::filesystem::path path = slot.path;
                        const CandidateMeta* host = slot.host;
                        const std::uint32_t count = slot.count;
                        const std::uint32_t depth_index = slot.depth_index;
                        const HistoryStorageLocation location = slot.location;
                        const std::uint64_t offset_entries = slot.offset_entries;
                        HistoryEntry* staging = slot.staging.data();
                        const std::uint32_t staging_capacity =
                            static_cast<std::uint32_t>(slot.staging.size());
                        slot.writer = std::async(
                            std::launch::async,
                            [this,
                             selected_mode,
                             path,
                             host,
                             count,
                             depth_index,
                             location,
                             offset_entries,
                             staging,
                             staging_capacity]() {
                                return materialize_depth(
                                    selected_mode,
                                    path,
                                    host,
                                    count,
                                    depth_index,
                                    location,
                                    offset_entries,
                                    staging,
                                    staging_capacity);
                             });
                        progressed = true;
                    } else if (status != cudaErrorNotReady) {
                        BEAM_CUDA_CHECK(status);
                    }
                }
                if (!slot.copy_pending && !slot.free && slot.writer.valid()) {
                    const bool ready = wait_all ||
                        slot.writer.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
                    if (ready) {
                        MaterializeResult result = slot.writer.get();
                        if (mode == CandidateHistoryMode::Ram) {
                            if (slot.depth_index >= ram_depths.size()) {
                                throw std::runtime_error("candidate history RAM depth index missing");
                            }
                            ram_depths[slot.depth_index] = std::move(result.ram_data);
                            ram_ready[slot.depth_index] = true;
                            if (prune_enabled && slot.depth_index > 0U) {
                                prune_dirty[slot.depth_index - 1U] = true;
                            }
                        } else if (mode == CandidateHistoryMode::StaticHybrid) {
                            if (slot.depth_index >= ram_ready.size()) {
                                throw std::runtime_error("static history depth index missing");
                            }
                            if (result.depth_index >= depth_locations.size()) {
                                throw std::runtime_error("static history result depth index missing");
                            }
                            depth_locations[result.depth_index] =
                                DepthLocation{result.location, result.offset_entries};
                            depth_files[result.depth_index] =
                                result.location == HistoryStorageLocation::Disk ? static_disk_path : std::filesystem::path{};
                            const std::uint64_t stored_bytes =
                                static_cast<std::uint64_t>(slot.count) * sizeof(HistoryEntry);
                            if (result.location == HistoryStorageLocation::Disk) {
                                bytes_stored_disk += stored_bytes;
                            } else if (result.location == HistoryStorageLocation::Ram) {
                                bytes_stored_ram += stored_bytes;
                            }
                            ram_ready[slot.depth_index] = true;
                        }
                        slot.free = true;
                        progressed = true;
                    }
                }
            }
            if (!wait_all) {
                break;
            }
        }
        pump_prune_jobs(wait_all);
    }

    Slot& acquire_slot() {
        pump_completed(false);
        for (Slot& slot : slots) {
            if (slot.free) {
                slot.free = false;
                return slot;
            }
        }
        pump_completed(true);
        for (Slot& slot : slots) {
            if (slot.free) {
                slot.free = false;
                return slot;
            }
        }
        throw std::runtime_error("candidate history slot acquisition failed");
    }

    void commit_slot(Slot& slot, std::uint32_t depth_index, std::uint32_t count) {
        if (count > slot.capacity) {
            throw std::runtime_error("candidate history count exceeds slot capacity");
        }
        slot.count = count;
        slot.depth_index = depth_index;
        slot.path = dir / ("depth_" + std::to_string(depth_index) + ".candidate_meta.bin");
        slot.location = HistoryStorageLocation::None;
        slot.offset_entries = 0;
        slot.copy_pending = true;
        slot.free = false;
        if (depth_index != depth_counts.size()) {
            throw std::runtime_error("candidate history commits must be depth-ordered");
        }
        if (mode == CandidateHistoryMode::StaticHybrid) {
            const DepthLocation location = reserve_static_location(count);
            slot.location = location.location;
            slot.offset_entries = location.offset_entries;
            depth_locations.push_back(DepthLocation{slot.location, slot.offset_entries});
            depth_files.push_back(slot.location == HistoryStorageLocation::Disk ? static_disk_path : std::filesystem::path{});
        } else {
            depth_files.push_back(mode == CandidateHistoryMode::Disk ? slot.path : std::filesystem::path{});
        }
        depth_counts.push_back(count);
        if (mode == CandidateHistoryMode::Ram) {
            ram_depths.emplace_back();
            ram_ready.push_back(false);
            prune_dirty.push_back(false);
        } else if (mode == CandidateHistoryMode::StaticHybrid) {
            ram_ready.push_back(false);
            prune_dirty.push_back(false);
        }
        bytes_received += static_cast<std::uint64_t>(count) * sizeof(CandidateMeta);
        bytes_stored += static_cast<std::uint64_t>(count) * sizeof(HistoryEntry);
        if (mode == CandidateHistoryMode::Ram) {
            bytes_stored_ram += static_cast<std::uint64_t>(count) * sizeof(HistoryEntry);
        } else if (mode == CandidateHistoryMode::Disk) {
            bytes_stored_disk += static_cast<std::uint64_t>(count) * sizeof(HistoryEntry);
        }
    }

    void finish_all() {
        pump_completed(true);
        pump_prune_jobs(true);
    }

    void destroy() {
        finish_all();
        for (Slot& slot : slots) {
            if (slot.copy_done != nullptr) {
                cudaEventDestroy(slot.copy_done);
                slot.copy_done = nullptr;
            }
            if (slot.host != nullptr) {
                cudaFreeHost(slot.host);
                slot.host = nullptr;
            }
        }
        slots.clear();
        if (ram_arena != nullptr) {
            std::free(ram_arena);
            ram_arena = nullptr;
        }
        if (copy_stream != nullptr) {
            cudaStreamDestroy(copy_stream);
            copy_stream = nullptr;
        }
    }

    HistoryEntry read_entry(std::uint32_t depth_index, std::uint64_t index) const {
        if (depth_index >= depth_counts.size()) {
            throw std::out_of_range("candidate history depth index out of range");
        }
        if (index >= depth_counts[depth_index]) {
            throw std::out_of_range("candidate history parent index out of range");
        }
        if (mode == CandidateHistoryMode::Ram) {
            if (depth_index >= ram_depths.size() || !ram_ready[depth_index]) {
                throw std::runtime_error("candidate history RAM depth not materialized");
            }
            return ram_depths[depth_index][static_cast<std::size_t>(index)];
        }
        if (mode == CandidateHistoryMode::StaticHybrid) {
            if (depth_index >= depth_locations.size() || depth_index >= ram_ready.size() || !ram_ready[depth_index]) {
                throw std::runtime_error("static hybrid history depth not materialized");
            }
            const DepthLocation location = depth_locations[depth_index];
            if (location.location == HistoryStorageLocation::Ram) {
                if (ram_arena == nullptr || location.offset_entries + index >= ram_arena_entries) {
                    throw std::runtime_error("static hybrid history RAM read exceeds capacity");
                }
                return ram_arena[location.offset_entries + index];
            }
            if (location.location == HistoryStorageLocation::Disk) {
                std::ifstream file(static_disk_path, std::ios::binary);
                if (!file) {
                    throw std::runtime_error("cannot open static history disk arena for read: " + static_disk_path.string());
                }
                const std::uint64_t offset = (location.offset_entries + index) * sizeof(HistoryEntry);
                file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
                HistoryEntry entry{};
                file.read(reinterpret_cast<char*>(&entry), sizeof(entry));
                if (!file) {
                    throw std::runtime_error("static history disk arena read failed: " + static_disk_path.string());
                }
                return entry;
            }
            throw std::runtime_error("static hybrid history read missing storage location");
        }
        std::ifstream file(depth_files[depth_index], std::ios::binary);
        if (!file) {
            throw std::runtime_error("cannot open candidate history file for read: " + depth_files[depth_index].string());
        }
        const std::uint64_t offset = index * sizeof(HistoryEntry);
        file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        HistoryEntry entry{};
        file.read(reinterpret_cast<char*>(&entry), sizeof(entry));
        if (!file) {
            throw std::runtime_error("candidate history read failed: " + depth_files[depth_index].string());
        }
        return entry;
    }
};

struct ReconstructedSolution {
    std::vector<std::uint8_t> moves;
    std::vector<std::uint64_t> parent_indices;
};

ReconstructedSolution reconstruct_solution_from_history(
    const CpuCandidateHistory& history,
    const CandidateMeta& solved_meta,
    std::uint32_t solved_depth) {
    if (solved_depth == 0U) {
        throw std::runtime_error("solved depth must be positive");
    }
    if (solved_depth > history.depth_files.size() + 1U) {
        throw std::runtime_error("solved depth exceeds stored candidate history");
    }
    ReconstructedSolution solution;
    solution.moves.resize(solved_depth);
    solution.parent_indices.resize(solved_depth);

    HistoryEntry cursor{solved_meta.parent_idx, solved_meta.route_packed, 0U};
    std::uint64_t parent_idx = cursor.parent_idx;
    for (std::uint32_t depth = solved_depth; depth > 0; --depth) {
        const std::uint32_t out = depth - 1U;
        solution.moves[out] = unpack_move(cursor.route_packed);
        solution.parent_indices[out] = parent_idx;
        if (out == 0U) {
            break;
        }
        cursor = history.read_entry(out - 1U, parent_idx);
        parent_idx = cursor.parent_idx;
    }
    return solution;
}

State128 apply_solution_moves(
    const State128& initial,
    const std::vector<std::uint8_t>& moves,
    const std::vector<std::uint8_t>& generators) {
    State128 state = initial;
    clear_state_padding(state);
    for (std::uint8_t move : moves) {
        state = apply_move_flat_host(state, generators, move);
    }
    return state;
}

struct SolvedSnapshot {
    bool found = false;
    std::uint32_t count = 0;
    std::uint32_t overflow = 0;
    std::vector<CandidateMeta> meta;
    std::vector<std::uint32_t> depth;
    std::vector<std::uint32_t> suffix;
};

std::uint32_t solved_total_depth(
    const SolvedNeighborhoodRuntime& solved_neighborhood,
    const Stream2SuffixRuntime& stream2_suffix,
    const CandidateMeta& meta,
    std::uint32_t prefix_depth,
    std::uint32_t suffix_id) {
    const std::uint32_t stream2_suffix_depth =
        static_cast<std::uint32_t>(stream2_suffix.suffix_len(suffix_id));
    const PackedSuffix suffix = solved_neighborhood.suffix_for(meta.hash);
    return prefix_depth + stream2_suffix_depth + static_cast<std::uint32_t>(suffix.len);
}

SolvedSnapshot select_best_solved_snapshot(
    const SolvedSnapshot& snapshot,
    const SolvedNeighborhoodRuntime& solved_neighborhood,
    const Stream2SuffixRuntime& stream2_suffix) {
    if (!snapshot.found || snapshot.meta.empty() || snapshot.depth.empty()) {
        return snapshot;
    }
    std::uint32_t best = 0;
    const std::uint32_t first_suffix_id =
        snapshot.suffix.empty() ? 0U : snapshot.suffix[0];
    std::uint32_t best_total_depth =
        solved_total_depth(
            solved_neighborhood,
            stream2_suffix,
            snapshot.meta[0],
            snapshot.depth[0],
            first_suffix_id);
    for (std::uint32_t i = 1U; i < snapshot.meta.size() && i < snapshot.depth.size(); ++i) {
        const std::uint32_t suffix_id =
            i < snapshot.suffix.size() ? snapshot.suffix[i] : 0U;
        const std::uint32_t total_depth =
            solved_total_depth(
                solved_neighborhood,
                stream2_suffix,
                snapshot.meta[i],
                snapshot.depth[i],
                suffix_id);
        const CandidateMeta& candidate = snapshot.meta[i];
        const CandidateMeta& incumbent = snapshot.meta[best];
        const std::uint32_t incumbent_suffix_id =
            best < snapshot.suffix.size() ? snapshot.suffix[best] : 0U;
        const bool better =
            total_depth < best_total_depth ||
            (total_depth == best_total_depth && candidate.parent_idx < incumbent.parent_idx) ||
            (total_depth == best_total_depth && candidate.parent_idx == incumbent.parent_idx &&
             candidate.route_packed < incumbent.route_packed) ||
            (total_depth == best_total_depth && candidate.parent_idx == incumbent.parent_idx &&
             candidate.route_packed == incumbent.route_packed &&
             hash_less(candidate.hash, incumbent.hash)) ||
            (total_depth == best_total_depth && candidate.parent_idx == incumbent.parent_idx &&
             candidate.route_packed == incumbent.route_packed &&
             candidate.hash == incumbent.hash &&
             suffix_id < incumbent_suffix_id);
        if (better) {
            best = i;
            best_total_depth = total_depth;
        }
    }
    SolvedSnapshot selected;
    selected.found = true;
    selected.count = snapshot.count;
    selected.overflow = snapshot.overflow;
    selected.meta.push_back(snapshot.meta[best]);
    selected.depth.push_back(snapshot.depth[best]);
    selected.suffix.push_back(best < snapshot.suffix.size() ? snapshot.suffix[best] : 0U);
    return selected;
}

void append_solution_suffixes(
    ReconstructedSolution& solution,
    const Stream2SuffixRuntime& stream2_suffix,
    std::uint32_t suffix_id,
    const SolvedNeighborhoodRuntime& solved_neighborhood,
    Hash128 hash) {
    if (suffix_id != 0U) {
        append_packed_suffix(solution.moves, stream2_suffix.suffix_for(suffix_id));
    }
    if (!solved_neighborhood.enabled()) {
        return;
    }
    const PackedSuffix suffix = solved_neighborhood.suffix_for(hash);
    append_packed_suffix(solution.moves, suffix);
}

void require_nccl(ncclResult_t status, const char* op);

SolvedSnapshot read_solved_snapshot(const StaticDeviceMemory& memory, std::uint32_t capacity) {
    SolvedSnapshot snapshot;
    std::uint32_t flag = 0;
    BEAM_CUDA_CHECK(cudaMemcpy(&flag, memory.solved_flag, sizeof(flag), cudaMemcpyDeviceToHost));
    snapshot.found = flag != 0U;
    if (!snapshot.found) {
        return snapshot;
    }
    BEAM_CUDA_CHECK(cudaMemcpy(&snapshot.count, memory.solved_count, sizeof(snapshot.count), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(&snapshot.overflow, memory.solved_overflow, sizeof(snapshot.overflow), cudaMemcpyDeviceToHost));
    const std::uint32_t stored = std::min(snapshot.count, capacity);
    snapshot.meta.resize(stored);
    snapshot.depth.resize(stored);
    snapshot.suffix.resize(stored);
    if (stored != 0U) {
        BEAM_CUDA_CHECK(cudaMemcpy(
            snapshot.meta.data(),
            memory.solved_meta_list,
            static_cast<std::uint64_t>(stored) * sizeof(CandidateMeta),
            cudaMemcpyDeviceToHost));
        BEAM_CUDA_CHECK(cudaMemcpy(
            snapshot.depth.data(),
            memory.solved_depth_list,
            static_cast<std::uint64_t>(stored) * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost));
        BEAM_CUDA_CHECK(cudaMemcpy(
            snapshot.suffix.data(),
            memory.solved_suffix_list,
            static_cast<std::uint64_t>(stored) * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost));
    }
    return snapshot;
}

struct DistributedReconstructionResult {
    bool has_solution = false;
    bool controller_rank = false;
    std::uint32_t controller = 0;
    CandidateMeta solved_meta{};
    std::uint32_t solved_depth = 0;
    std::uint32_t solved_suffix_id = 0;
    ReconstructedSolution solution;
};

DistributedReconstructionResult reconstruct_solution_distributed(
    const CpuCandidateHistory& history,
    const SolvedSnapshot& local_solved,
    const SolvedNeighborhoodRuntime& solved_neighborhood,
    const Stream2SuffixRuntime& stream2_suffix,
    const StaticMemoryPlan& plan,
    StaticDeviceMemory& memory,
    const DispatcherStreams& streams,
    ncclComm_t comm,
    std::uint32_t world_size,
    std::uint32_t rank) {
    if (world_size <= 1U) {
        throw std::invalid_argument("distributed reconstruction requires WORLD_SIZE > 1");
    }
    if (comm == nullptr) {
        throw std::invalid_argument("distributed reconstruction requires NCCL communicator");
    }
    constexpr std::uint32_t packet_words = 8;
    constexpr std::uint32_t query_words = 4;
    constexpr std::uint32_t response_words = 2;
    const std::uint64_t scratch_words =
        (plan.frontier_states * sizeof(State128)) / sizeof(std::uint64_t);
    const std::uint64_t required_words =
        static_cast<std::uint64_t>(packet_words) * (static_cast<std::uint64_t>(world_size) + 1ULL) +
        query_words + response_words;
    if (scratch_words < required_words || memory.final.next_frontier_states_tmp == nullptr) {
        throw std::runtime_error("distributed reconstruction scratch buffer is too small");
    }

    std::uint64_t* scratch = reinterpret_cast<std::uint64_t*>(memory.final.next_frontier_states_tmp);
    std::uint64_t* packet_send_device = scratch;
    std::uint64_t* packet_recv_device = scratch + packet_words;
    std::uint64_t* query_device =
        packet_recv_device + static_cast<std::uint64_t>(packet_words) * world_size;
    std::uint64_t* response_device = query_device + query_words;

    const bool local_found = local_solved.found && !local_solved.meta.empty() && !local_solved.depth.empty();
    const std::uint32_t local_suffix_id =
        local_found && !local_solved.suffix.empty() ? local_solved.suffix.front() : 0U;
    std::array<std::uint64_t, packet_words> local_packet{
        local_found ? 1ULL : 0ULL,
        static_cast<std::uint64_t>(rank),
        local_found ? static_cast<std::uint64_t>(local_solved.depth.front()) : 0ULL,
        local_found ? local_solved.meta.front().parent_idx : 0ULL,
        local_found ? static_cast<std::uint64_t>(local_solved.meta.front().route_packed) : 0ULL,
        local_found ? local_solved.meta.front().hash.lo : 0ULL,
        local_found ? local_solved.meta.front().hash.hi : 0ULL,
        local_found ? static_cast<std::uint64_t>(local_suffix_id) : 0ULL};
    BEAM_CUDA_CHECK(cudaMemcpyAsync(
        packet_send_device,
        local_packet.data(),
        packet_words * sizeof(std::uint64_t),
        cudaMemcpyHostToDevice,
        streams.stream5));
    require_nccl(ncclAllGather(
        packet_send_device,
        packet_recv_device,
        packet_words,
        ncclUint64,
        comm,
        streams.stream5), "ncclAllGather solved packet");
    std::vector<std::uint64_t> packets(static_cast<std::uint64_t>(packet_words) * world_size);
    BEAM_CUDA_CHECK(cudaMemcpyAsync(
        packets.data(),
        packet_recv_device,
        packets.size() * sizeof(std::uint64_t),
        cudaMemcpyDeviceToHost,
        streams.stream5));
    BEAM_CUDA_CHECK(cudaStreamSynchronize(streams.stream5));

    DistributedReconstructionResult result;
    bool best_found = false;
    std::uint32_t best_total_depth = 0;
    for (std::uint32_t peer = 0; peer < world_size; ++peer) {
        const std::uint64_t* packet = packets.data() + static_cast<std::uint64_t>(peer) * packet_words;
        if (packet[0] == 0ULL) {
            continue;
        }
        CandidateMeta candidate{};
        candidate.parent_idx = packet[3];
        candidate.route_packed = static_cast<std::uint32_t>(packet[4]);
        candidate.hash = Hash128{packet[5], packet[6]};
        candidate.score_key = GOAL_SCORE_KEY;
        const std::uint32_t candidate_depth = static_cast<std::uint32_t>(packet[2]);
        const std::uint32_t candidate_suffix_id = static_cast<std::uint32_t>(packet[7]);
        const std::uint32_t candidate_total_depth =
            solved_total_depth(
                solved_neighborhood,
                stream2_suffix,
                candidate,
                candidate_depth,
                candidate_suffix_id);
        const bool better =
            !best_found ||
            candidate_total_depth < best_total_depth ||
            (candidate_total_depth == best_total_depth && candidate.parent_idx < result.solved_meta.parent_idx) ||
            (candidate_total_depth == best_total_depth && candidate.parent_idx == result.solved_meta.parent_idx &&
             candidate.route_packed < result.solved_meta.route_packed) ||
            (candidate_total_depth == best_total_depth && candidate.parent_idx == result.solved_meta.parent_idx &&
             candidate.route_packed == result.solved_meta.route_packed &&
             hash_less(candidate.hash, result.solved_meta.hash)) ||
            (candidate_total_depth == best_total_depth && candidate.parent_idx == result.solved_meta.parent_idx &&
             candidate.route_packed == result.solved_meta.route_packed &&
             candidate.hash == result.solved_meta.hash &&
             candidate_suffix_id < result.solved_suffix_id) ||
            (candidate_total_depth == best_total_depth && candidate.parent_idx == result.solved_meta.parent_idx &&
             candidate.route_packed == result.solved_meta.route_packed &&
             candidate.hash == result.solved_meta.hash &&
             candidate_suffix_id == result.solved_suffix_id &&
             peer < result.controller);
        if (better) {
            best_found = true;
            best_total_depth = candidate_total_depth;
            result.has_solution = true;
            result.controller = peer;
            result.solved_depth = candidate_depth;
            result.solved_suffix_id = candidate_suffix_id;
            result.solved_meta = candidate;
        }
    }
    if (!result.has_solution) {
        return result;
    }
    if (result.solved_depth == 0U) {
        throw std::runtime_error("distributed solved depth must be positive");
    }
    result.controller_rank = rank == result.controller;
    if (result.controller_rank) {
        result.solution.moves.resize(result.solved_depth);
        result.solution.parent_indices.resize(result.solved_depth);
    }

    HistoryEntry cursor{result.solved_meta.parent_idx, result.solved_meta.route_packed, 0U};
    for (std::uint32_t depth = result.solved_depth; depth > 0; --depth) {
        const std::uint32_t out = depth - 1U;
        if (result.controller_rank) {
            result.solution.moves[out] = unpack_move(cursor.route_packed);
            result.solution.parent_indices[out] = cursor.parent_idx;
        }
        if (out == 0U) {
            break;
        }

        const std::uint32_t target_rank = unpack_source_rank(cursor.route_packed);
        if (target_rank >= world_size) {
            throw std::runtime_error("distributed history source rank exceeds WORLD_SIZE");
        }
        std::array<std::uint64_t, query_words> query{
            1ULL,
            static_cast<std::uint64_t>(target_rank),
            static_cast<std::uint64_t>(out - 1U),
            cursor.parent_idx};
        if (result.controller_rank) {
            BEAM_CUDA_CHECK(cudaMemcpyAsync(
                query_device,
                query.data(),
                query_words * sizeof(std::uint64_t),
                cudaMemcpyHostToDevice,
                streams.stream5));
        }
        require_nccl(ncclBroadcast(
            query_device,
            query_device,
            query_words,
            ncclUint64,
            static_cast<int>(result.controller),
            comm,
            streams.stream5), "ncclBroadcast history query");
        BEAM_CUDA_CHECK(cudaMemcpyAsync(
            query.data(),
            query_device,
            query_words * sizeof(std::uint64_t),
            cudaMemcpyDeviceToHost,
            streams.stream5));
        BEAM_CUDA_CHECK(cudaStreamSynchronize(streams.stream5));

        const std::uint32_t query_target = static_cast<std::uint32_t>(query[1]);
        if (query_target >= world_size) {
            throw std::runtime_error("distributed history query rank exceeds WORLD_SIZE");
        }
        if (rank == query_target) {
            const HistoryEntry entry =
                history.read_entry(static_cast<std::uint32_t>(query[2]), query[3]);
            const std::array<std::uint64_t, response_words> response{
                entry.parent_idx,
                static_cast<std::uint64_t>(entry.route_packed)};
            BEAM_CUDA_CHECK(cudaMemcpyAsync(
                response_device,
                response.data(),
                response_words * sizeof(std::uint64_t),
                cudaMemcpyHostToDevice,
                streams.stream5));
        }
        require_nccl(ncclBroadcast(
            response_device,
            response_device,
            response_words,
            ncclUint64,
            static_cast<int>(query_target),
            comm,
            streams.stream5), "ncclBroadcast history response");
        std::array<std::uint64_t, response_words> response{};
        BEAM_CUDA_CHECK(cudaMemcpyAsync(
            response.data(),
            response_device,
            response_words * sizeof(std::uint64_t),
            cudaMemcpyDeviceToHost,
            streams.stream5));
        BEAM_CUDA_CHECK(cudaStreamSynchronize(streams.stream5));
        cursor.parent_idx = response[0];
        cursor.route_packed = static_cast<std::uint32_t>(response[1]);
    }
    if (result.controller_rank) {
        append_solution_suffixes(
            result.solution,
            stream2_suffix,
            result.solved_suffix_id,
            solved_neighborhood,
            result.solved_meta.hash);
    }
    return result;
}

void write_solution_artifacts(
    std::uint64_t puzzle_id,
    std::uint32_t depth_limit,
    std::uint64_t beam,
    const ReconstructedSolution& solution,
    const std::vector<std::string>& move_names,
    const State128& final_state,
    bool valid,
    const CpuCandidateHistory& history,
    const SolvedSnapshot& solved) {
    const std::string path_text = moves_to_path_text(solution.moves, move_names);
    const std::string state_text = state120_to_text(final_state);
    const std::filesystem::path solution_log =
        std::filesystem::path("test_results") /
        ("solution_p" + std::to_string(puzzle_id) +
         "_d" + std::to_string(depth_limit) +
         "_b" + std::to_string(beam) +
         "_" + timestamp_id() + ".log");
    std::ofstream log(solution_log);
    if (!log) {
        throw std::runtime_error("cannot open solution log for write: " + solution_log.string());
    }
    log << "solution_found=1\n";
    log << "solution_valid=" << (valid ? 1 : 0) << "\n";
    log << "solution_depth=" << solution.moves.size() << "\n";
    log << "solution_path=" << path_text << "\n";
    log << "solution_state120=\"" << state_text << "\"\n";
    log << "history_mode=" << history_mode_name(history.mode) << "\n";
    log << "history_dir=" << history.dir.string() << "\n";
    log << "history_depth_count=" << history.depth_counts.size() << "\n";
    log << "history_bytes_received=" << history.bytes_received << "\n";
    log << "history_bytes_stored=" << history.bytes_stored << "\n";
    log << "history_bytes_stored_ram=" << history.bytes_stored_ram << "\n";
    log << "history_bytes_stored_disk=" << history.bytes_stored_disk << "\n";
    log << "history_bytes_pruned=" << history.bytes_pruned << "\n";
    log << "solved_count=" << solved.count << "\n";
    log << "solved_overflow=" << solved.overflow << "\n";

    std::ofstream submit("submit.csv");
    if (!submit) {
        throw std::runtime_error("cannot open submit.csv for write");
    }
    submit << "initial_state_id,path\n";
    submit << puzzle_id << "," << path_text << "\n";

    const std::filesystem::path submit_copy =
        std::filesystem::path("test_results") /
        ("submit_p" + std::to_string(puzzle_id) +
         "_d" + std::to_string(depth_limit) +
         "_b" + std::to_string(beam) + ".csv");
    std::ofstream submit_copy_stream(submit_copy);
    if (!submit_copy_stream) {
        throw std::runtime_error("cannot open test_results submit copy for write: " + submit_copy.string());
    }
    submit_copy_stream << "initial_state_id,path\n";
    submit_copy_stream << puzzle_id << "," << path_text << "\n";

    const std::filesystem::path path_copy =
        std::filesystem::path("test_results") /
        ("solution_path_p" + std::to_string(puzzle_id) +
         "_d" + std::to_string(depth_limit) +
         "_b" + std::to_string(beam) + ".csv");
    std::ofstream submit_path_copy(path_copy);
    if (!submit_path_copy) {
        throw std::runtime_error("cannot open solution path copy for write: " + path_copy.string());
    }
    submit_path_copy << "initial_state_id,path\n";
    submit_path_copy << puzzle_id << "," << path_text << "\n";

#if BEAM_ENABLE_DEBUG_LOGS
    std::cout << "solution_log=" << solution_log.string() << "\n";
    std::cout << "submit_csv=submit.csv\n";
    std::cout << "submit_path_copy=" << submit_copy.string() << "\n";
    std::cout << "solution_path_copy=" << path_copy.string() << "\n";
    std::cout << "solution_found=1\n";
    std::cout << "solution_valid=" << (valid ? 1 : 0) << "\n";
    std::cout << "solution_depth=" << solution.moves.size() << "\n";
    std::cout << "solution_path=" << path_text << "\n";
    std::cout << "solution_state120=\"" << state_text << "\"\n";
#endif
}

#if BEAM_DEBUG_INFERENCE_TRACE
std::vector<half> stream1_reference_linear(
    const std::vector<half>& input,
    const half* weight,
    std::uint32_t input_cols,
    std::uint32_t output_cols) {
    std::vector<half> output(output_cols);
    for (std::uint32_t out = 0; out < output_cols; ++out) {
        float acc = 0.0f;
        for (std::uint32_t in = 0; in < input_cols; ++in) {
            acc += __half2float(input[in]) * __half2float(weight[in * output_cols + out]);
        }
        output[out] = __float2half(acc);
    }
    return output;
}

void stream1_reference_bias_relu(std::vector<half>& values, const half* bias) {
    for (std::uint32_t i = 0; i < values.size(); ++i) {
        const float x = __half2float(values[i]) + __half2float(bias[i]);
        values[i] = __float2half(x > 0.0f ? x : 0.0f);
    }
}

void stream1_reference_residual_add_bias_relu(
    std::vector<half>& values,
    const std::vector<half>& residual,
    const half* bias) {
    for (std::uint32_t i = 0; i < values.size(); ++i) {
        const float x = __half2float(values[i]) + __half2float(residual[i]) + __half2float(bias[i]);
        values[i] = __float2half(x > 0.0f ? x : 0.0f);
    }
}

std::array<std::uint32_t, MOVE_COUNT> stream1_reference_score_keys(
    const State128& state,
    const stream1_weights::HostWeightBytes& weights) {
    const Stream1ModelConfig& model = weights.model;
    const half* input_weight = stream1_weights::weight_half_data(weights.input_weight);
    const half* input_bias = stream1_weights::weight_half_data(weights.input_bias);
    const half* hidden_weight = stream1_weights::weight_half_data(weights.hidden_weight);
    const half* hidden_bias = stream1_weights::weight_half_data(weights.hidden_bias);
    const half* output_weight = stream1_weights::weight_half_data(weights.output_weight);
    const half* output_bias = stream1_weights::weight_half_data(weights.output_bias);

    std::vector<half> hidden1(model.hidden1);
    for (std::uint32_t h = 0; h < model.hidden1; ++h) {
        float acc = __half2float(input_bias[h]);
        for (std::uint32_t p = 0; p < model.state_len; ++p) {
            const std::uint32_t value = static_cast<std::uint32_t>(state.v[p]);
            const std::uint32_t idx = (p * model.num_classes + value) * model.hidden1 + h;
            acc += __half2float(input_weight[idx]);
        }
        hidden1[h] = __float2half(acc > 0.0f ? acc : 0.0f);
    }

    std::vector<half> hidden2 =
        stream1_reference_linear(hidden1, hidden_weight, model.hidden1, model.hidden2);
    stream1_reference_bias_relu(hidden2, hidden_bias);

    std::vector<half> residual = hidden2;
    for (std::uint32_t block = 0; block < model.residual_count; ++block) {
        const half* fc1_weight = stream1_weights::weight_half_data(weights.residual_fc1_weight[block]);
        const half* fc1_bias = stream1_weights::weight_half_data(weights.residual_fc1_bias[block]);
        const half* fc2_weight = stream1_weights::weight_half_data(weights.residual_fc2_weight[block]);
        const half* fc2_bias = stream1_weights::weight_half_data(weights.residual_fc2_bias[block]);
        std::vector<half> tmp =
            stream1_reference_linear(residual, fc1_weight, model.hidden2, model.hidden2);
        stream1_reference_bias_relu(tmp, fc1_bias);
        tmp = stream1_reference_linear(tmp, fc2_weight, model.hidden2, model.hidden2);
        stream1_reference_residual_add_bias_relu(tmp, residual, fc2_bias);
        residual = std::move(tmp);
    }

    const std::vector<half> output =
        stream1_reference_linear(residual, output_weight, model.hidden2, MOVE_COUNT);
    std::array<std::uint32_t, MOVE_COUNT> score_keys{};
    for (std::uint32_t move = 0; move < MOVE_COUNT; ++move) {
        const float q = __half2float(output[move]) + __half2float(output_bias[move]);
        score_keys[move] = q_to_score_key(q);
    }
    return score_keys;
}

std::array<std::uint32_t, MOVE_COUNT> score_ranks(const std::array<std::uint32_t, MOVE_COUNT>& score_keys) {
    std::array<std::uint32_t, MOVE_COUNT> order{};
    std::array<std::uint32_t, MOVE_COUNT> ranks{};
    for (std::uint32_t move = 0; move < MOVE_COUNT; ++move) {
        order[move] = move;
    }
    std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        if (score_keys[a] != score_keys[b]) {
            return score_keys[a] < score_keys[b];
        }
        return a < b;
    });
    for (std::uint32_t rank = 0; rank < MOVE_COUNT; ++rank) {
        ranks[order[rank]] = rank + 1U;
    }
    return ranks;
}

void log_stream1_move_score_comparison(
    std::uint64_t puzzle_id,
    std::uint32_t depth,
    const TrackedSolutionPrefix& tracked,
    const GeneratedTrackResult& generated,
    const stream1_weights::HostWeightBytes& weights,
    const std::vector<std::string>& move_names) {
    if (!tracked.enabled || depth >= tracked.moves.size()) {
        return;
    }
    if (!generated.found || !generated.parent_state_copied || !generated.all_move_scores_copied) {
        std::cout << "track_solution_move_score_summary"
                  << " puzzle_id=" << puzzle_id
                  << " depth=" << depth
                  << " prefix_len=" << depth + 1U
                  << " expected_move=" << move_names[tracked.moves[depth]]
                  << " available=0\n";
        return;
    }
    const std::array<std::uint32_t, MOVE_COUNT> ref_scores =
        stream1_reference_score_keys(generated.parent_state, weights);
    const std::array<std::uint32_t, MOVE_COUNT> cuda_ranks =
        score_ranks(generated.move_score_keys);
    const std::array<std::uint32_t, MOVE_COUNT> ref_ranks = score_ranks(ref_scores);
    const std::uint32_t expected_move = tracked.moves[depth];
    std::uint32_t best_cuda_move = 0;
    std::uint32_t best_ref_move = 0;
    for (std::uint32_t move = 1; move < MOVE_COUNT; ++move) {
        if (generated.move_score_keys[move] < generated.move_score_keys[best_cuda_move]) {
            best_cuda_move = move;
        }
        if (ref_scores[move] < ref_scores[best_ref_move]) {
            best_ref_move = move;
        }
    }
    std::cout << "track_solution_move_score_summary"
              << " puzzle_id=" << puzzle_id
              << " depth=" << depth
              << " prefix_len=" << depth + 1U
              << " expected_move=" << move_names[expected_move]
              << " expected_move_idx=" << expected_move
              << " cuda_expected_rank=" << cuda_ranks[expected_move]
              << " ref_expected_rank=" << ref_ranks[expected_move]
              << " cuda_expected_score_key=" << generated.move_score_keys[expected_move]
              << " ref_expected_score_key=" << ref_scores[expected_move]
              << " best_cuda_move=" << move_names[best_cuda_move]
              << " best_cuda_move_idx=" << best_cuda_move
              << " best_cuda_score_key=" << generated.move_score_keys[best_cuda_move]
              << " best_ref_move=" << move_names[best_ref_move]
              << " best_ref_move_idx=" << best_ref_move
              << " best_ref_score_key=" << ref_scores[best_ref_move]
              << "\n";
    for (std::uint32_t move = 0; move < MOVE_COUNT; ++move) {
        const std::int64_t score_delta =
            static_cast<std::int64_t>(generated.move_score_keys[move]) -
            static_cast<std::int64_t>(ref_scores[move]);
        const std::int64_t rank_delta =
            static_cast<std::int64_t>(cuda_ranks[move]) -
            static_cast<std::int64_t>(ref_ranks[move]);
        std::cout << "track_solution_move_score"
                  << " puzzle_id=" << puzzle_id
                  << " depth=" << depth
                  << " prefix_len=" << depth + 1U
                  << " move=" << move
                  << " move_name=" << move_names[move]
                  << " expected=" << (move == expected_move ? 1 : 0)
                  << " cuda_score_key=" << generated.move_score_keys[move]
                  << " cuda_q=" << (static_cast<double>(generated.move_score_keys[move]) / SCORE_SCALE)
                  << " cuda_rank=" << cuda_ranks[move]
                  << " ref_score_key=" << ref_scores[move]
                  << " ref_q=" << (static_cast<double>(ref_scores[move]) / SCORE_SCALE)
                  << " ref_rank=" << ref_ranks[move]
                  << " score_delta=" << score_delta
                  << " rank_delta=" << rank_delta
                  << "\n";
    }
}
#endif

void require_aligned(const void* ptr, std::uintptr_t alignment, const char* name) {
    if (reinterpret_cast<std::uintptr_t>(ptr) % alignment != 0) {
        throw std::runtime_error(std::string("device pointer alignment failed: ") + name);
    }
}

void require_nccl(ncclResult_t status, const char* op) {
    if (status != ncclSuccess) {
        throw std::runtime_error(std::string(op) + ": " + ncclGetErrorString(status));
    }
}

struct NcclRuntime {
    ncclComm_t comm = nullptr;

    NcclRuntime() = default;
    NcclRuntime(const NcclRuntime&) = delete;
    NcclRuntime& operator=(const NcclRuntime&) = delete;

    NcclRuntime(NcclRuntime&& other) noexcept : comm(other.comm) {
        other.comm = nullptr;
    }

    NcclRuntime& operator=(NcclRuntime&& other) noexcept {
        if (this != &other) {
            if (comm != nullptr) {
                ncclCommDestroy(comm);
            }
            comm = other.comm;
            other.comm = nullptr;
        }
        return *this;
    }

    ~NcclRuntime() {
        if (comm != nullptr) {
            ncclCommDestroy(comm);
        }
    }
};

ncclUniqueId read_nccl_id_file(const std::filesystem::path& path) {
    ncclUniqueId id{};
    for (std::uint32_t attempt = 0; attempt < 600U; ++attempt) {
        std::ifstream in(path, std::ios::binary);
        if (in) {
            in.read(reinterpret_cast<char*>(&id), sizeof(id));
            if (in.gcount() == static_cast<std::streamsize>(sizeof(id))) {
                return id;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error("timed out waiting for NCCL rendezvous file: " + path.string());
}

NcclRuntime create_nccl_runtime(std::uint32_t world_size, std::uint32_t rank) {
    NcclRuntime runtime;
    if (world_size == 1U) {
        return runtime;
    }
    const std::filesystem::path id_path = env_path("BEAM_NCCL_ID_FILE", "/tmp/beam_solver_nccl_id.bin");
    ncclUniqueId id{};
    if (rank == 0U) {
        require_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
        const std::filesystem::path tmp_path = id_path.string() + ".tmp";
        {
            std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
            if (!out) {
                throw std::runtime_error("failed to open NCCL rendezvous temp file: " + tmp_path.string());
            }
            out.write(reinterpret_cast<const char*>(&id), sizeof(id));
        }
        std::filesystem::rename(tmp_path, id_path);
    } else {
        id = read_nccl_id_file(id_path);
    }
    require_nccl(
        ncclCommInitRank(&runtime.comm, static_cast<int>(world_size), id, static_cast<int>(rank)),
        "ncclCommInitRank");
    return runtime;
}

std::uint32_t propagate_stop_flag(
    StaticDeviceMemory& memory,
    DispatcherStreams& streams,
    ncclComm_t comm,
    std::uint32_t world_size) {
    if (world_size == 1U) {
        std::uint32_t stop_value = 0;
        BEAM_CUDA_CHECK(cudaMemcpy(&stop_value, memory.stop_flag, sizeof(stop_value), cudaMemcpyDeviceToHost));
        return stop_value;
    }
    require_nccl(
        ncclAllReduce(
            memory.stop_flag,
            memory.global_stop_flag,
            1,
            ncclUint32,
            ncclMax,
            comm,
            streams.stream5),
        "ncclAllReduce stop flag");
    BEAM_CUDA_CHECK(cudaMemcpyAsync(
        memory.stop_flag,
        memory.global_stop_flag,
        sizeof(std::uint32_t),
        cudaMemcpyDeviceToDevice,
        streams.stream5));
    BEAM_CUDA_CHECK(cudaStreamSynchronize(streams.stream5));
    std::uint32_t stop_value = 0;
    BEAM_CUDA_CHECK(cudaMemcpy(&stop_value, memory.stop_flag, sizeof(stop_value), cudaMemcpyDeviceToHost));
    return stop_value;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4 && argc != 6) {
        std::cerr << "usage: production_runner <puzzle_id> <depth> <beam> [world_size] [local_rank]\n";
        return 2;
    }
    const std::uint64_t puzzle_id = parse_u64(argv[1], "puzzle_id");
    const std::uint32_t depth_limit = static_cast<std::uint32_t>(parse_u64(argv[2], "depth"));
    const std::uint64_t beam = parse_u64(argv[3], "beam");
    const std::uint32_t world_size_arg = argc == 6 ? static_cast<std::uint32_t>(parse_u64(argv[4], "world_size")) : 1U;
    const std::uint32_t rank_arg = argc == 6 ? static_cast<std::uint32_t>(parse_u64(argv[5], "local_rank")) : 0U;
    const std::uint32_t world_size = env_or_default_u32("WORLD_SIZE", world_size_arg);
    const std::uint32_t rank = env_or_default_u32("RANK", rank_arg);
    const std::uint32_t device_local_rank = env_or_default_u32("LOCAL_RANK", rank);
    if (world_size == 0 || world_size > 255 || rank >= world_size) {
        throw std::invalid_argument("world_size must be in [1,255] and rank must be less than world_size");
    }
    std::cout << std::unitbuf;

    BEAM_CUDA_CHECK(cudaSetDevice(static_cast<int>(device_local_rank)));
    NcclRuntime nccl_runtime = create_nccl_runtime(world_size, rank);
    DispatcherCollective collective{nccl_runtime.comm};
    const DispatcherCollective* collective_ptr = world_size > 1U ? &collective : nullptr;
    std::size_t free_before = 0;
    std::size_t total_before = 0;
    BEAM_CUDA_CHECK(cudaMemGetInfo(&free_before, &total_before));

    const std::filesystem::path weight_dir = env_path("BEAM_WEIGHT_DIR", "stream1_weights");
    const stream1_weights::HostWeightBytes host_weights =
        stream1_weights::load_stream1_weights(weight_dir);
    const Stream1ModelConfig& stream1_model = host_weights.model;
    const RuntimeConfigBuild config_build =
        build_runtime_config_from_budget(beam, world_size, rank, stream1_model, free_before);
    const RuntimeConfig config = config_build.config;
    const StaticMemoryPlan plan = config_build.plan;
#if BEAM_ENABLE_DEBUG_LOGS
    std::cout << "puzzle_id=" << puzzle_id << "\n";
    std::cout << "depth_limit=" << depth_limit << "\n";
    std::cout << "RUNTIME_CONFIG_MODE=" << (config_build.manual_config ? "manual" : "auto") << "\n";
    std::cout << "USER_GLOBAL_BEAM_WIDTH=" << beam << "\n";
    std::cout << "WORLD_SIZE=" << config.world_size << "\n";
    std::cout << "LOCAL_RANK=" << config.local_rank << "\n";
    std::cout << "CUDA_DEVICE_LOCAL_RANK=" << device_local_rank << "\n";
    std::cout << "B_MICRO=" << config.b_micro << "\n";
    std::cout << "STREAM1_CONCURRENCY=" << config.inference_parallelism << "\n";
    std::cout << "STREAM3_RING_SLOTS=" << config_build.stream3_ring_slots << "\n";
    std::cout << "RING_COUNT=" << config.ring_count << "\n";
    std::cout << "RING_SLOT_COUNT=" << plan.derived.ring_slot_count << "\n";
    std::cout << "STREAM3_BATCH_CANDIDATES=" << config.stream3_batch_candidates << "\n";
    std::cout << "STREAM5_BATCH_CANDIDATES=" << config.stream3_batch_candidates << "\n";
    std::cout << "SHARD_COUNT=" << config.shard_count << "\n";
    std::cout << "SHARD_BUFFER_COUNT=" << config.shard_buffer_count << "\n";
    std::cout << "STORAGE_SHARD_COUNT=" << plan.storage_shard_count << "\n";
    std::cout << "STREAM4_BATCH_CANDIDATES=" << config.stream4_batch_candidates << "\n";
    std::cout << "STREAM4_TRIGGER_CANDIDATES=" << config.stream4_trigger_candidates << "\n";
    std::cout << "STREAM4_BATCH_ALIGNMENT=" << config.stream4_batch_alignment << "\n";
    std::cout << "STREAM4_ACTIVE_SORT_SLOTS=" << config.stream4_active_sort_slots << "\n";
    std::cout << "SHARD_CAPACITY_CANDIDATES=" << config.shard_capacity_candidates << "\n";
    std::cout << "SHARD_CAPACITY_SCALE_PPM=" << config.shard_capacity_scale_ppm << "\n";
    std::cout << "GLOBAL_SPILL_CAPACITY=" << config.global_spill_capacity << "\n";
    std::cout << "GLOBAL_SPILL_SCALE_PPM=" << config.global_spill_scale_ppm << "\n";
    std::cout << "STREAM5_RECV_CAPACITY_SCALE_PPM=" << config.stream5_recv_capacity_scale_ppm << "\n";
    std::cout << "STREAM5_SLOT_COUNT=" << plan.stream5_slot_count << "\n";
    std::cout << "STREAM5_SEND_SLOT_CANDIDATES=" << plan.stream5_send_slot_capacity << "\n";
    std::cout << "STREAM5_RECV_SLOT_CANDIDATES=" << plan.stream5_recv_slot_capacity << "\n";
    std::cout << "N_LOCAL=" << local_frontier_capacity(config) << "\n";
    std::cout << "LOGICAL_SHARD_SIZE=" << logical_shard_size_for(config) << "\n";
    std::cout << "GROSS_CANDIDATES_PER_DEPTH_EST=" << config_build.gross_candidates_per_depth_est << "\n";
    std::cout << "STREAM3_JOBS_PER_DEPTH_EST=" << config_build.stream3_jobs_per_depth_est << "\n";
    std::cout << "STREAM3_SORT_WORK_UNITS_EST=" << config_build.stream3_sort_work_units_est << "\n";
    std::cout << "STREAM4_FLOW_SCALE_PPM=" << config_build.stream4_flow_scale_ppm << "\n";
    std::cout << "STREAM4_INPUT_CANDIDATES_PER_DEPTH_EST=" << config_build.stream4_input_candidates_per_depth_est << "\n";
    std::cout << "STREAM4_JOBS_PER_SHARD_EST=" << config_build.stream4_jobs_per_shard_est << "\n";
    std::cout << "STREAM4_JOBS_PER_DEPTH_EST=" << config_build.stream4_jobs_per_depth_est << "\n";
    std::cout << "STREAM4_WAVES_PER_DEPTH_EST=" << config_build.stream4_waves_per_depth_est << "\n";
    std::cout << "STREAM4_SORT_WORK_UNITS_EST=" << config_build.stream4_sort_work_units_est << "\n";
    std::cout << "GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS=" << config.global_threshold_update_period_shards << "\n";
    std::cout << "GLOBAL_BEAM_WIDTH_EFFECTIVE=" << plan.derived.global_beam_width_effective << "\n";
    std::cout << "BEAM_WIDTH_ALIGNMENT=" << plan.derived.beam_width_alignment << "\n";
    std::cout << "SCORE_SCALE=" << SCORE_SCALE << "\n";
    std::cout << "SCORE_MAX_KEY=" << SCORE_MAX_KEY << "\n";
    std::cout << "SCORE_BIN_COUNT=" << SCORE_BIN_COUNT << "\n";
    std::cout << "SOLVED_RESULT_CAPACITY=" << config.solved_result_capacity << "\n";
    std::cout << "gpu_total_bytes=" << total_before << "\n";
    std::cout << "gpu_free_before_bytes=" << free_before << "\n";
    std::cout << "gpu_headroom_bytes=" << config_build.gpu_headroom_bytes << "\n";
    std::cout << "gpu_budget_bytes=" << config_build.gpu_budget_bytes << "\n";
    std::cout << "static_allocation_bytes=" << plan.total_device_bytes << "\n";
    std::cout << "estimated_non_static_device_bytes=" << config_build.estimated_non_static_device_bytes << "\n";
    std::cout << "estimated_required_device_bytes=" << config_build.estimated_required_device_bytes << "\n";
    std::cout << "current_frontier_bytes=" << plan.current_frontier_bytes << "\n";
    std::cout << "solved_bytes=" << plan.solved_bytes << "\n";
    std::cout << "scratch_pool_bytes=" << plan.scratch_pool_bytes << "\n";
    std::cout << "layout_phase1_streams_bytes=" << plan.layout_phase1_streams_bytes << "\n";
    std::cout << "layout_phase2_select_bytes=" << plan.layout_phase2_select_bytes << "\n";
    std::cout << "layout_phase3_materialize_bytes=" << plan.layout_phase3_materialize_bytes << "\n";
    std::cout << "layout_streams_bytes=" << plan.layout_streams_bytes << "\n";
    std::cout << "layout_final_budget_bytes=" << plan.layout_final_budget_bytes << "\n";
    std::cout << "layout_final_bytes=" << plan.layout_final_bytes << "\n";
    std::cout << "frontier_state_capacity=" << plan.frontier_states << "\n";
#endif

    const std::filesystem::path generator_path = "FullBeamNice/generators/p900.json";
    const std::filesystem::path puzzle_info_path = "data/puzzle_info.json";
    const std::filesystem::path test_csv_path = "data/test.csv";
    const std::vector<std::uint8_t> host_generators = load_p900_generators(generator_path);
    const std::vector<std::string> host_move_names = load_p900_move_names(generator_path);
    const State128 host_central = load_central_state(puzzle_info_path);
    const State128 host_initial = load_initial_state_from_test_csv(test_csv_path, puzzle_id);
    const ZobristTable host_zobrist = make_deterministic_zobrist(0xC0DEC0DEULL);
    SolvedNeighborhoodRuntime solved_neighborhood =
        build_solved_neighborhood_runtime(host_central, host_generators, host_zobrist);
    Stream2SuffixRuntime stream2_suffix = build_stream2_suffix_runtime(host_generators);
#if BEAM_ENABLE_DEBUG_LOGS
    std::cout << "real_assets=enabled\n";
    std::cout << "generator_path=" << generator_path.string() << "\n";
    std::cout << "puzzle_info_path=" << puzzle_info_path.string() << "\n";
    std::cout << "test_csv_path=" << test_csv_path.string() << "\n";
    std::cout << "weight_dir=" << weight_dir.string() << "\n";
    std::cout << "stream1_model_state_len=" << stream1_model.state_len << "\n";
    std::cout << "stream1_model_num_classes=" << stream1_model.num_classes << "\n";
    std::cout << "stream1_model_hidden1=" << stream1_model.hidden1 << "\n";
    std::cout << "stream1_model_hidden2=" << stream1_model.hidden2 << "\n";
    std::cout << "stream1_model_residual_count=" << stream1_model.residual_count << "\n";
    std::cout << "stream1_model_output_dim=" << stream1_model.output_dim << "\n";
    std::cout << "stream1_model_weight_bytes=" << stream1_weights::total_host_weight_bytes(host_weights) << "\n";
    std::cout << "solved_neighborhood_radius=" << solved_neighborhood.radius << "\n";
    std::cout << "solved_neighborhood_entries=" << solved_neighborhood.entry_count << "\n";
    std::cout << "solved_neighborhood_bucket_count=" << solved_neighborhood.bucket_count << "\n";
    std::cout << "solved_neighborhood_device_bytes=" << solved_neighborhood.device_bytes << "\n";
    std::cout << "stream2_suffix_radius=" << stream2_suffix.radius << "\n";
    std::cout << "stream2_suffix_backend=" << stream2_suffix.backend_name << "\n";
    std::cout << "stream2_suffix_entries=" << stream2_suffix.entry_count << "\n";
    std::cout << "stream2_suffix_device_bytes=" << stream2_suffix.device_bytes << "\n";
#endif

    StaticDeviceMemory memory;
    allocate_static_device_memory(plan, memory);
    BEAM_CUDA_CHECK(cudaMemset(memory.allocation, 0, memory.allocation_bytes));
    BEAM_CUDA_CHECK(cudaMemset(memory.current_frontier_states, 0, plan.current_frontier_bytes));
    if (rank == 0U) {
        BEAM_CUDA_CHECK(cudaMemcpy(memory.current_frontier_states, &host_initial, sizeof(State128), cudaMemcpyHostToDevice));
    }
    BEAM_CUDA_CHECK(cudaMemset(memory.streams.current_threshold, 0xff, 2ULL * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(memory.streams.threshold_initialized, 0, 2ULL * sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(memory.streams.current_threshold_active_index, 0, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(memory.streams.threshold_request_local, 0, sizeof(std::uint32_t)));
    BEAM_CUDA_CHECK(cudaMemset(memory.streams.threshold_request_global, 0, sizeof(std::uint32_t)));

    std::size_t free_after = 0;
    std::size_t total_after = 0;
    BEAM_CUDA_CHECK(cudaMemGetInfo(&free_after, &total_after));
#if BEAM_ENABLE_DEBUG_LOGS
    std::cout << "gpu_free_after_static_alloc_bytes=" << free_after << "\n";
    std::cout << "runner_phase=allocations_done\n";
#endif

    std::uint8_t* generators = nullptr;
    State128* central_state = nullptr;
    Hash128* zobrist = nullptr;
    BEAM_CUDA_CHECK(cudaMalloc(&generators, host_generators.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&central_state, sizeof(State128)));
    BEAM_CUDA_CHECK(cudaMalloc(&zobrist, STATE_STORAGE_LEN * STATE_VALUE_PAD * sizeof(Hash128)));
    require_aligned(generators, 16, "generators");
    require_aligned(central_state, alignof(State128), "central_state");
    require_aligned(zobrist, alignof(Hash128), "zobrist");
    BEAM_CUDA_CHECK(cudaMemcpy(generators, host_generators.data(), host_generators.size(), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(central_state, &host_central, sizeof(State128), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(zobrist, &host_zobrist[0][0], STATE_STORAGE_LEN * STATE_VALUE_PAD * sizeof(Hash128), cudaMemcpyHostToDevice));
    stream1_weights::DeviceWeights device_weights = stream1_weights::upload_weights(host_weights);
    stream1_weights::ScratchAllocation stream1_scratch =
        stream1_weights::alloc_stream1_scratch(stream1_model, config.b_micro, config.inference_parallelism);
    TrackedSolutionPrefix tracked_solution;
#if BEAM_DEBUG_PATH_TRACE
    tracked_solution.initialize(puzzle_id, host_initial, host_generators, host_zobrist, host_move_names);
#endif
#if BEAM_ENABLE_DEBUG_LOGS
    std::size_t free_after_all_allocations = 0;
    std::size_t total_after_all_allocations = 0;
    BEAM_CUDA_CHECK(cudaMemGetInfo(&free_after_all_allocations, &total_after_all_allocations));
    std::cout << "gpu_free_after_all_allocations_bytes=" << free_after_all_allocations << "\n";
#endif

    DispatcherStreams streams;
    DispatcherEvents events;
    CudaGraphJobTemplates graphs;
    create_dispatcher_streams(streams);
    create_dispatcher_events(events);
    const std::vector<const half*> residual_fc1_weight =
        stream1_weights::const_pointer_vector(device_weights.residual_fc1_weight);
    const std::vector<const half*> residual_fc1_bias =
        stream1_weights::const_pointer_vector(device_weights.residual_fc1_bias);
    const std::vector<const half*> residual_fc2_weight =
        stream1_weights::const_pointer_vector(device_weights.residual_fc2_weight);
    const std::vector<const half*> residual_fc2_bias =
        stream1_weights::const_pointer_vector(device_weights.residual_fc2_bias);
    const Stream1NetworkDims dims = stream1_weights::network_dims(stream1_model);
    std::vector<Stream1CutlassScratch> stream1_scratch_lanes;
    stream1_scratch_lanes.reserve(config.inference_parallelism);
    for (std::uint32_t lane = 0; lane < config.inference_parallelism; ++lane) {
        stream1_scratch_lanes.push_back(Stream1CutlassScratch{
            stream1_scratch.hidden1 + static_cast<std::uint64_t>(lane) * config.b_micro * stream1_model.hidden1,
            stream1_scratch.hidden2 + static_cast<std::uint64_t>(lane) * config.b_micro * stream1_model.hidden2,
            stream1_scratch.residual + static_cast<std::uint64_t>(lane) * config.b_micro * stream1_model.hidden2,
            stream1_scratch.output + static_cast<std::uint64_t>(lane) * config.b_micro * stream1_model.output_dim});
    }
    DispatcherNetwork network{
        Stream1NetworkView{
            device_weights.input_weight,
            device_weights.input_bias,
            device_weights.hidden_weight,
            device_weights.hidden_bias,
            residual_fc1_weight.data(),
            residual_fc1_bias.data(),
            residual_fc2_weight.data(),
            residual_fc2_bias.data(),
            device_weights.output_weight,
            device_weights.output_bias,
            dims},
        stream1_scratch_lanes};
    DispatcherDeviceTables tables{generators, central_state, zobrist};
    Stream2SolvedBuffers solved{
        memory.solved_flag,
        memory.stop_flag,
        memory.solved_count,
        memory.solved_overflow,
        memory.solved_meta_list,
        memory.solved_depth_list,
        config.solved_result_capacity,
        memory.current_depth,
        solved_neighborhood.device_table(),
        stream2_suffix.device_table(),
        memory.solved_suffix_list};
    instantiate_cuda_graph_job_templates(plan, memory, tables, network, solved, streams, events, graphs);
#if BEAM_ENABLE_DEBUG_LOGS
    std::cout << "runner_phase=graphs_instantiated\n";
#endif

    const auto start = std::chrono::steady_clock::now();
    CpuCandidateHistory history;
    const CandidateHistoryMode history_mode = parse_history_mode();
    const std::uint32_t history_slot_count = env_u32("BEAM_HISTORY_SLOT_COUNT", 3);
    const std::uint32_t history_worker_count = env_u32("BEAM_HISTORY_WORKERS", 1);
    const std::uint64_t history_ram_budget_total = env_u64("BEAM_HISTORY_RAM_BYTES", 0);
    const std::uint64_t history_disk_budget_total = env_u64("BEAM_HISTORY_DISK_BYTES", 0);
    const std::uint64_t history_ram_budget_per_rank =
        history_ram_budget_total == 0ULL ? 0ULL : history_ram_budget_total / static_cast<std::uint64_t>(world_size);
    const std::uint64_t history_disk_budget_per_rank =
        history_disk_budget_total == 0ULL ? 0ULL : history_disk_budget_total / static_cast<std::uint64_t>(world_size);
    const std::filesystem::path history_disk_root =
        env_path("BEAM_HISTORY_DISK_PATH", "").empty()
            ? std::filesystem::path{}
            : env_path("BEAM_HISTORY_DISK_PATH", "");
    const std::uint32_t depth_log_every = env_u32("BEAM_DEPTH_LOG_EVERY", 1);
    history.dir = make_history_dir(puzzle_id, depth_limit, beam, rank, world_size);
    const std::filesystem::path history_disk_path =
        history_disk_root.empty()
            ? std::filesystem::path{}
            : history_disk_root / ("rank_" + std::to_string(rank) + "_history_static_arena.bin");
    history.initialize(
        history_mode,
        static_cast<std::uint32_t>(plan.frontier_states),
        history_slot_count,
        history_worker_count,
        depth_limit,
        solved_neighborhood.radius,
        stream2_suffix.radius,
        history_ram_budget_per_rank,
        history_disk_budget_per_rank,
        history_disk_path);
    if (world_size > 1U) {
        history.prune_enabled = false;
    }
#if BEAM_ENABLE_DEBUG_LOGS
    std::cout << "candidate_history_dir=" << history.dir.string() << "\n";
    std::cout << "candidate_history_mode=" << history_mode_name(history.mode) << "\n";
    std::cout << "candidate_history_prune_enabled=" << (history.prune_enabled ? 1 : 0) << "\n";
    std::cout << "candidate_history_slots=" << history_slot_count << "\n";
    std::cout << "candidate_history_workers=" << history_worker_count << "\n";
    std::cout << "candidate_history_ram_budget_total=" << history_ram_budget_total << "\n";
    std::cout << "candidate_history_disk_budget_total=" << history_disk_budget_total << "\n";
    std::cout << "candidate_history_ram_budget_per_rank=" << history_ram_budget_per_rank << "\n";
    std::cout << "candidate_history_disk_budget_per_rank=" << history_disk_budget_per_rank << "\n";
    std::cout << "candidate_history_pinned_slot_bytes=" << history.bytes_pinned_slots << "\n";
    std::cout << "candidate_history_slot_staging_bytes=" << history.bytes_slot_staging << "\n";
    std::cout << "candidate_history_static_ram_arena_bytes=" << history.bytes_static_ram_arena << "\n";
    std::cout << "candidate_history_static_disk_arena_bytes=" << history.bytes_static_disk_arena << "\n";
    std::cout << "candidate_history_budget_formula=effective_depth_minus_prefull_target\n";
    std::cout << "candidate_history_prefull_estimator=move_count_upper_bound\n";
    std::cout << "candidate_history_effective_depth=" << history.budget_estimate.effective_depth << "\n";
    std::cout << "candidate_history_target_beam_depth="
              << history.budget_estimate.target_beam_depth << "\n";
    std::cout << "candidate_history_states_before_target_beam="
              << history.budget_estimate.states_before_target_beam << "\n";
    std::cout << "candidate_history_required_entries="
              << history.budget_estimate.required_entries << "\n";
    std::cout << "candidate_history_required_bytes=" << history.history_required_bytes << "\n";
    if (!history.static_disk_path.empty()) {
        std::cout << "candidate_history_static_disk_path=" << history.static_disk_path.string() << "\n";
    }
#endif
    std::uint64_t frontier_size = rank == 0U ? 1ULL : 0ULL;
    [[maybe_unused]] std::uint64_t last_final_frontier_size = frontier_size;
    [[maybe_unused]] std::uint32_t last_final_threshold = UINT32_THRESHOLD_MAX;
    std::uint32_t total_threshold_updates = 0;
    std::uint32_t completed_depths = 0;
    bool solution_found = false;
    for (std::uint32_t depth = 0; depth < depth_limit; ++depth) {
        const std::uint32_t current_solution_depth = depth + 1U;
        BEAM_CUDA_CHECK(cudaMemcpy(
            memory.current_depth,
            &current_solution_depth,
            sizeof(current_solution_depth),
            cudaMemcpyHostToDevice));
        const auto depth_start = std::chrono::steady_clock::now();
        history.pump_completed(false);
        [[maybe_unused]] const bool emit_depth_log = (depth_log_every != 0U) && ((depth % depth_log_every) == 0U);
#if BEAM_ENABLE_DEPTH_LOGS
        if (emit_depth_log) {
        std::cout << "depth_start=" << depth
                  << " frontier_size=" << frontier_size
                  << " depth_limit=" << depth_limit << "\n";
        }
#endif
        GeneratedTrackRequest generated_track_request{};
#if BEAM_DEBUG_PATH_TRACE
        generated_track_request = tracked_solution.generated_request_for_depth(depth);
#endif
        const DepthDispatchState state =
            run_depth_cuda_graphs(plan, memory, graphs, streams, frontier_size, generated_track_request, collective_ptr);
        if (!state.depth_drained) {
            throw std::runtime_error("depth did not drain");
        }
#if BEAM_DEBUG_PATH_TRACE
        tracked_solution.log_generated(
            puzzle_id,
            depth,
            state.tracked_generated,
            host_move_names);
        tracked_solution.log_stream3(
            puzzle_id,
            depth,
            state.tracked_stream3,
            host_move_names);
#if BEAM_DEBUG_INFERENCE_TRACE
        log_stream1_move_score_comparison(
            puzzle_id,
            depth,
            tracked_solution,
            state.tracked_generated,
            host_weights,
            host_move_names);
#endif
        tracked_solution.log_stream4(
            puzzle_id,
            depth,
            state.tracked_stream4,
            state.tracked_stream4_events,
            host_move_names);
#endif
        total_threshold_updates += state.threshold_updates;
        ++completed_depths;

        const std::uint32_t global_stop_value =
            propagate_stop_flag(memory, streams, nccl_runtime.comm, world_size);
        const SolvedSnapshot solved_snapshot = read_solved_snapshot(memory, config.solved_result_capacity);
        const SolvedSnapshot selected_solved_snapshot =
            select_best_solved_snapshot(solved_snapshot, solved_neighborhood, stream2_suffix);
        if (world_size > 1U && global_stop_value != 0U) {
            history.finish_all();
            const DistributedReconstructionResult distributed_solution =
                reconstruct_solution_distributed(
                    history,
                    selected_solved_snapshot,
                    solved_neighborhood,
                    stream2_suffix,
                    plan,
                    memory,
                    streams,
                    nccl_runtime.comm,
                    world_size,
                    rank);
            if (!distributed_solution.has_solution) {
                throw std::runtime_error("global stop set but no solved rank reported metadata");
            }
            if (distributed_solution.controller_rank) {
                const State128 final_state =
                    apply_solution_moves(host_initial, distributed_solution.solution.moves, host_generators);
                const bool valid = states_equal_storage(final_state, host_central);
                const double solved_elapsed_sec =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                SolvedSnapshot global_solved = selected_solved_snapshot;
                global_solved.found = true;
                global_solved.count = std::max<std::uint32_t>(global_solved.count, 1U);
                if (global_solved.meta.empty()) {
                    global_solved.meta.push_back(distributed_solution.solved_meta);
                }
                if (global_solved.depth.empty()) {
                    global_solved.depth.push_back(distributed_solution.solved_depth);
                }
                if (global_solved.suffix.empty()) {
                    global_solved.suffix.push_back(distributed_solution.solved_suffix_id);
                }
                write_solution_artifacts(
                    puzzle_id,
                    depth_limit,
                    beam,
                    distributed_solution.solution,
                    host_move_names,
                    final_state,
                    valid,
                    history,
                    global_solved);
                if (!valid) {
                    throw std::runtime_error("CPU solution validation failed: generated state is not central_state");
                }
                std::cout << "puzzle_solved=1"
                          << " puzzle_id=" << puzzle_id
                          << " seconds=" << solved_elapsed_sec
                          << " solution_length=" << distributed_solution.solution.moves.size()
                          << " solution=" << moves_to_path_text(distributed_solution.solution.moves, host_move_names)
                          << "\n";
            }
            solution_found = true;
            const auto depth_end = std::chrono::steady_clock::now();
            const double depth_sec = std::chrono::duration<double>(depth_end - depth_start).count();
#if BEAM_ENABLE_DEPTH_LOGS
            if (emit_depth_log) {
            std::cout << "depth_global_solved=" << depth
                      << " depth_sec=" << depth_sec
                      << " controller_rank=" << distributed_solution.controller
                      << " local_rank=" << rank
                      << " solved_depth=" << distributed_solution.solved_depth
                      << " stop_requested=" << global_stop_value << "\n";
            }
#endif
            break;
        }
        if (selected_solved_snapshot.found) {
            if (selected_solved_snapshot.meta.empty() || selected_solved_snapshot.depth.empty()) {
                throw std::runtime_error("solved flag set but solved metadata list is empty");
            }
            const CandidateMeta solved_meta = selected_solved_snapshot.meta.front();
            const std::uint32_t solved_depth = selected_solved_snapshot.depth.front();
            const std::uint32_t solved_suffix_id =
                selected_solved_snapshot.suffix.empty() ? 0U : selected_solved_snapshot.suffix.front();
            history.finish_all();
            ReconstructedSolution solution =
                reconstruct_solution_from_history(history, solved_meta, solved_depth);
            append_solution_suffixes(
                solution,
                stream2_suffix,
                solved_suffix_id,
                solved_neighborhood,
                solved_meta.hash);
            const State128 final_state = apply_solution_moves(host_initial, solution.moves, host_generators);
            const bool valid = states_equal_storage(final_state, host_central);
            const double solved_elapsed_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            write_solution_artifacts(
                puzzle_id,
                depth_limit,
                beam,
                solution,
                host_move_names,
                final_state,
                valid,
                history,
                selected_solved_snapshot);
            if (!valid) {
                throw std::runtime_error("CPU solution validation failed: generated state is not central_state");
            }
            std::cout << "puzzle_solved=1"
                      << " puzzle_id=" << puzzle_id
                      << " seconds=" << solved_elapsed_sec
                      << " solution_length=" << solution.moves.size()
                      << " solution=" << moves_to_path_text(solution.moves, host_move_names)
                      << "\n";
            solution_found = true;
            const auto depth_end = std::chrono::steady_clock::now();
            const double depth_sec = std::chrono::duration<double>(depth_end - depth_start).count();
#if BEAM_ENABLE_DEPTH_LOGS
            if (emit_depth_log) {
            std::cout << "depth_solved=" << depth
                      << " depth_sec=" << depth_sec
                      << " solved_depth=" << solved_depth
                      << " solved_count=" << selected_solved_snapshot.count
                      << " solved_overflow=" << selected_solved_snapshot.overflow
                      << " stop_requested=" << (state.stop_requested ? 1 : 0) << "\n";
            }
#endif
            break;
        }
        if (global_stop_value != 0U) {
            history.finish_all();
#if BEAM_ENABLE_DEPTH_LOGS
            if (emit_depth_log) {
            std::cout << "depth_global_stop_without_local_solution=" << depth
                      << " rank=" << rank
                      << " stop_requested=" << global_stop_value << "\n";
            }
#endif
            solution_found = true;
            break;
        }

        CpuCandidateHistory::Slot& history_slot = history.acquire_slot();
        const FinalizeDepthState final_state = finalize_depth_single_gpu(
            plan,
            memory,
            tables,
            streams,
            frontier_size,
            history_slot.host,
            history_slot.capacity,
            history.copy_stream,
            history_slot.copy_done,
#if BEAM_DEBUG_PATH_TRACE
            tracked_solution.hash_for_depth(depth),
            collective_ptr);
#else
            nullptr,
            collective_ptr);
#endif
#if BEAM_DEBUG_PATH_TRACE
        if (tracked_solution.enabled) {
            BEAM_CUDA_CHECK(cudaEventSynchronize(history_slot.copy_done));
            tracked_solution.log_prefinal(
                puzzle_id,
                depth,
                final_state,
                host_move_names);
            tracked_solution.scan_depth(
                puzzle_id,
                depth,
                history_slot.host,
                final_state.final_candidate_count,
                final_state.final_threshold,
                host_move_names);
        }
#endif
        history.commit_slot(history_slot, depth, final_state.final_candidate_count);
        history.pump_completed(false);
        frontier_size = final_state.next_frontier_size;
        last_final_frontier_size = frontier_size;
        last_final_threshold = final_state.final_threshold;
#if BEAM_DEBUG_DEPTH_FLOW_TRACE
        std::cout << "depth_flow_trace"
                  << " rank=" << rank
                  << " depth=" << depth
                  << " phase=summary"
                  << " frontier_size=" << state.frontier_size
                  << " generated=" << state.generated_candidates_total
                  << " threshold_start=" << state.threshold_start
                  << " threshold_start_initialized=" << state.threshold_start_initialized
                  << " threshold_end=" << state.threshold_end
                  << " threshold_end_initialized=" << state.threshold_end_initialized
                  << " stream3_threshold_pass=" << state.stream3_threshold_pass_total
                  << " stream3_unique=" << state.stream3_unique_total
                  << " stream3_local_pending=" << state.stream3_local_pending_total
                  << " stream3_remote_send=" << state.stream3_remote_send_total
                  << " stream5_recv=" << state.stream5_recv_total
                  << " stream3_local_write=" << state.stream3_local_write_total
                  << " stream3_local_spill=" << state.stream3_local_spill_total
                  << " stream3_remote_write=" << state.stream3_remote_write_total
                  << " stream3_remote_spill=" << state.stream3_remote_spill_total
                  << " stream4_clean_after_drain=" << state.stream4_clean_after_drain_total
                  << " stream4_dirty_after_drain=" << state.stream4_dirty_after_drain_total
                  << " final_local_clean=" << final_state.local_clean_before_final
                  << " final_global_less=" << final_state.final_global_less
                  << " final_global_equal=" << final_state.final_global_equal
                  << " final_total_available=" << final_state.final_total_available
                  << " final_global_keep=" << final_state.final_global_keep_count
                  << " final_threshold=" << final_state.final_threshold
                  << " final_candidate_count=" << final_state.final_candidate_count
                  << " final_request_count=" << final_state.final_request_count
                  << " next_frontier_size=" << final_state.next_frontier_size
                  << "\n";
#endif
        const auto depth_end = std::chrono::steady_clock::now();
        const double depth_sec = std::chrono::duration<double>(depth_end - depth_start).count();
#if BEAM_ENABLE_DEPTH_LOGS
        if (emit_depth_log) {
#if BEAM_DEBUG_STREAM_TIMING
        const double stream4_avg_ms =
            state.stream4_jobs_launched == 0U ? 0.0 :
            state.stream4_ms_total / static_cast<double>(state.stream4_jobs_launched);
        const double stream3_batch_ms =
            state.stream3_jobs_launched == 0U ? 0.0 :
            state.stream3_ring_ms_total / static_cast<double>(state.stream3_jobs_launched);
        const double stream3_candidates_per_sec =
            state.stream3_ring_ms_total <= 0.0 ? 0.0 :
            (static_cast<double>(state.stream3_jobs_launched) *
             static_cast<double>(config.stream3_batch_candidates) * 1000.0) /
            state.stream3_ring_ms_total;
        const double stream4_candidates_per_sec =
            state.stream4_ms_total <= 0.0 ? 0.0 :
            (static_cast<double>(state.stream4_jobs_launched) *
             static_cast<double>(config.stream4_batch_candidates) * 1000.0) /
            state.stream4_ms_total;
        const std::uint64_t spill_peak_ratio_ppm =
            config.global_spill_capacity == 0U ? 0ULL :
            (static_cast<std::uint64_t>(state.global_spill_peak) * 1'000'000ULL) /
                static_cast<std::uint64_t>(config.global_spill_capacity);
        const double stream5_ms_total = state.stream5_ms_total + final_state.stream5_threshold_ms;
#endif
        std::cout << "depth_done=" << depth
                  << " depth_sec=" << depth_sec
                  << " ring_slot_jobs=" << state.ring_slot_jobs_launched
                  << " stream3_jobs=" << state.stream3_jobs_launched
                  << " stream4_jobs=" << state.stream4_jobs_launched
                  << " stream4_slots_used=" << state.stream4_active_sort_slots_used
#if BEAM_DEBUG_STREAM_TIMING
                  << " stream12_ms=" << state.stream12_ms_total
                  << " stream12_max_ms=" << state.stream12_ms_max
                  << " stream3_ring_ms=" << state.stream3_ring_ms_total
                  << " stream3_ring_max_ms=" << state.stream3_ring_ms_max
                  << " stream3_spill_drain_ms=" << state.stream3_spill_drain_ms_total
                  << " stream3_final_filter_ms=" << final_state.stream3_final_filter_ms
                  << " stream3_final_materialize_ms=" << final_state.stream3_final_materialize_ms
                  << " stream3_reset_ms=" << final_state.stream3_reset_ms
                  << " stream4_ms=" << state.stream4_ms_total
                  << " stream4_max_ms=" << state.stream4_ms_max
                  << " stream4_avg_ms=" << stream4_avg_ms
                  << " stream5_ms=" << stream5_ms_total
                  << " stream4_pending_max=" << state.stream4_pending_shards_max
                  << " stream4_busy_max=" << state.stream4_busy_slots_max
                  << " global_spill_peak=" << state.global_spill_peak
#endif
                  << " threshold_updates_depth=" << state.threshold_updates
                  << " final_threshold=" << final_state.final_threshold
                  << " final_candidate_count=" << final_state.final_candidate_count
                  << " final_request_count=" << final_state.final_request_count
                  << " next_frontier_size=" << frontier_size
                  << " history_bytes_received=" << history.bytes_received
                  << " history_bytes_stored=" << history.bytes_stored
                  << " history_bytes_stored_ram=" << history.bytes_stored_ram
                  << " history_bytes_stored_disk=" << history.bytes_stored_disk
                  << " history_bytes_pruned=" << history.bytes_pruned << "\n";
#if BEAM_DEBUG_STREAM_TIMING
        std::cout << "throughput_probe"
                  << " depth=" << depth
                  << " stream3_batch_ms=" << stream3_batch_ms
                  << " stream3_candidates_per_sec=" << stream3_candidates_per_sec
                  << " stream4_batch_ms=" << stream4_avg_ms
                  << " stream4_candidates_per_sec=" << stream4_candidates_per_sec
                  << " spill_peak=" << state.global_spill_peak
                  << " spill_capacity=" << config.global_spill_capacity
                  << " spill_peak_ratio_ppm=" << spill_peak_ratio_ppm << "\n";
#endif
        }
#endif
        if (frontier_size == 0) {
            break;
        }
#if BEAM_DEBUG_PATH_TRACE
        if (tracked_solution.should_stop_after_missing(depth)) {
            std::cout << "track_solution_stop=1"
                      << " puzzle_id=" << puzzle_id
                      << " depth=" << depth
                      << " reason=tracked_path_missing_extra_depths_elapsed"
                      << " first_missing_depth=" << tracked_solution.first_missing_depth
                      << " extra_depths=" << tracked_solution.missing_stop_extra_depths
                      << "\n";
            break;
        }
        if (tracked_solution.enabled &&
            tracked_solution.stop_after_path &&
            depth + 1U >= tracked_solution.moves.size()) {
            std::cout << "track_solution_stop=1"
                      << " puzzle_id=" << puzzle_id
                      << " depth=" << depth
                      << " reason=tracked_path_exhausted_without_solution\n";
            break;
        }
#endif
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_sec = std::chrono::duration<double>(end - start).count();
    const auto history_flush_start = std::chrono::steady_clock::now();
    history.finish_all();
    const auto history_flush_end = std::chrono::steady_clock::now();
    const double history_flush_sec = std::chrono::duration<double>(history_flush_end - history_flush_start).count();
    const double avg_depth_sec = completed_depths == 0 ? 0.0 : elapsed_sec / static_cast<double>(completed_depths);
#if BEAM_ENABLE_DEBUG_LOGS
    std::cout << "elapsed_sec=" << elapsed_sec << "\n";
    std::cout << "avg_depth_sec=" << avg_depth_sec << "\n";
    std::cout << "completed_depths=" << completed_depths << "\n";
    std::cout << "last_final_frontier_size=" << last_final_frontier_size << "\n";
    std::cout << "last_final_threshold=" << last_final_threshold << "\n";
    std::cout << "threshold_updates=" << total_threshold_updates << "\n";
    std::cout << "solution_found=" << (solution_found ? 1 : 0) << "\n";
    std::cout << "history_mode=" << history_mode_name(history.mode) << "\n";
    std::cout << "history_depth_count=" << history.depth_counts.size() << "\n";
    std::cout << "history_bytes_received=" << history.bytes_received << "\n";
    std::cout << "history_bytes_stored=" << history.bytes_stored << "\n";
    std::cout << "history_bytes_stored_ram=" << history.bytes_stored_ram << "\n";
    std::cout << "history_bytes_stored_disk=" << history.bytes_stored_disk << "\n";
    std::cout << "history_bytes_pruned=" << history.bytes_pruned << "\n";
    std::cout << "history_flush_sec=" << history_flush_sec << "\n";
#endif
    if (!solution_found) {
        const std::filesystem::path no_solution_log =
            std::filesystem::path("test_results") /
            ("no_solution_p" + std::to_string(puzzle_id) +
             "_d" + std::to_string(depth_limit) +
             "_b" + std::to_string(beam) + ".log");
        std::ofstream no_solution(no_solution_log);
        no_solution << "solution_found=0\n";
        no_solution << "completed_depths=" << completed_depths << "\n";
        no_solution << "history_mode=" << history_mode_name(history.mode) << "\n";
        no_solution << "history_dir=" << history.dir.string() << "\n";
        no_solution << "history_depth_count=" << history.depth_counts.size() << "\n";
        no_solution << "history_bytes_received=" << history.bytes_received << "\n";
        no_solution << "history_bytes_stored=" << history.bytes_stored << "\n";
        no_solution << "history_bytes_stored_ram=" << history.bytes_stored_ram << "\n";
        no_solution << "history_bytes_stored_disk=" << history.bytes_stored_disk << "\n";
        no_solution << "history_bytes_pruned=" << history.bytes_pruned << "\n";
        std::cout << "puzzle_solved=0"
                  << " puzzle_id=" << puzzle_id
                  << " seconds=" << elapsed_sec
                  << " solution_length=-1"
                  << " solution="
                  << "\n";
#if BEAM_ENABLE_DEBUG_LOGS
        std::cout << "no_solution_log=" << no_solution_log.string() << "\n";
#endif
    }

    history.destroy();
    destroy_cuda_graph_job_templates(graphs);
    destroy_dispatcher_events(events);
    destroy_dispatcher_streams(streams);
    free_static_device_memory(memory);
    solved_neighborhood.destroy();
    stream2_suffix.destroy();
    cudaFree(generators);
    cudaFree(central_state);
    cudaFree(zobrist);
    stream1_weights::free_weights(device_weights);
    stream1_weights::free_stream1_scratch(stream1_scratch);
    return 0;
}
