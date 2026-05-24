#include "cuda_check.hpp"
#include "../cuda/dispatcher.hpp"
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
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace beam;

#ifndef BEAM_ENABLE_DEPTH_LOGS
#define BEAM_ENABLE_DEPTH_LOGS 0
#endif

#ifndef BEAM_ENABLE_DEBUG_LOGS
#define BEAM_ENABLE_DEBUG_LOGS 0
#endif

namespace {

inline constexpr std::uint32_t STREAM1_MODEL_CLASSES = 120;
inline constexpr std::uint32_t STREAM1_HIDDEN1 = 1536;
inline constexpr std::uint32_t STREAM1_HIDDEN2 = 512;
inline constexpr std::uint32_t STREAM1_RESIDUAL_COUNT = 2;

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

std::uint64_t ceil_div_u64(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) {
        throw std::invalid_argument("ceil_div denominator is zero");
    }
    return (numerator + denominator - 1ULL) / denominator;
}

std::uint32_t checked_u32(std::uint64_t value, const char* name) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string("value exceeds uint32: ") + name);
    }
    return static_cast<std::uint32_t>(value);
}

std::uint32_t next_power_of_two_u32(std::uint32_t value) {
    if (value <= 1U) {
        return 1U;
    }
    --value;
    value |= value >> 1U;
    value |= value >> 2U;
    value |= value >> 4U;
    value |= value >> 8U;
    value |= value >> 16U;
    return value + 1U;
}

std::uint32_t round_up_u32(std::uint32_t value, std::uint32_t alignment) {
    return checked_u32(round_up(value, alignment), "round_up_u32");
}

std::uint64_t scaled_round_up(std::uint64_t value, std::uint64_t ppm) {
    constexpr std::uint64_t scale = 1'000'000ULL;
    return (value * ppm + scale - 1ULL) / scale;
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

std::vector<std::byte> read_binary_exact(const std::filesystem::path& path, std::size_t expected_bytes) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open required binary file: " + path.string());
    }
    std::vector<std::byte> bytes(expected_bytes);
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    const std::size_t actual = static_cast<std::size_t>(file.gcount());
    file.peek();
    if (bytes.size() != expected_bytes) {
        throw std::runtime_error(
            "binary file size mismatch: " + path.string() +
            " expected=" + std::to_string(expected_bytes) +
            " actual=" + std::to_string(bytes.size()));
    }
    if (actual != expected_bytes || !file.eof()) {
        throw std::runtime_error(
            "binary file size mismatch: " + path.string() +
            " expected=" + std::to_string(expected_bytes) +
            " actual_read=" + std::to_string(actual));
    }
    return bytes;
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

std::filesystem::path make_history_dir(std::uint64_t puzzle_id, std::uint32_t depth_limit, std::uint64_t beam) {
    std::filesystem::path dir = "test_results";
    dir /= "candidate_history_p" + std::to_string(puzzle_id) +
        "_d" + std::to_string(depth_limit) +
        "_b" + std::to_string(beam) +
        "_" + timestamp_id();
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

enum class CandidateHistoryMode : std::uint8_t {
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
    throw std::runtime_error("BEAM_HISTORY_MODE must be ram or disk");
}

const char* history_mode_name(CandidateHistoryMode mode) {
    return mode == CandidateHistoryMode::Ram ? "ram" : "disk";
}

struct HistoryEntry {
    std::uint64_t parent_idx = 0;
    std::uint32_t route_packed = 0;
    std::uint32_t pad = 0;
};

static_assert(sizeof(HistoryEntry) == 16);
static_assert(alignof(HistoryEntry) == 8);

struct CpuCandidateHistory {
    struct Slot {
        CandidateMeta* host = nullptr;
        std::uint32_t capacity = 0;
        std::uint32_t count = 0;
        std::uint32_t depth_index = 0;
        std::filesystem::path path;
        cudaEvent_t copy_done = nullptr;
        bool copy_pending = false;
        bool free = true;
        std::future<std::vector<HistoryEntry>> writer;
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
    std::vector<std::vector<HistoryEntry>> ram_depths;
    std::vector<bool> ram_ready;
    std::vector<bool> prune_dirty;
    std::vector<PruneJob> prune_jobs;
    std::uint64_t bytes_received = 0;
    std::uint64_t bytes_stored = 0;
    std::uint64_t bytes_pruned = 0;
    std::uint32_t worker_count = 1;
    std::vector<Slot> slots;
    cudaStream_t copy_stream = nullptr;

    void initialize(
        CandidateHistoryMode selected_mode,
        std::uint32_t capacity,
        std::uint32_t slot_count,
        std::uint32_t selected_worker_count) {
        if (copy_stream != nullptr) {
            throw std::runtime_error("candidate history already initialized");
        }
        if (capacity == 0U || slot_count == 0U || selected_worker_count == 0U) {
            throw std::runtime_error("candidate history capacity, slot count, and worker count must be nonzero");
        }
        mode = selected_mode;
        worker_count = selected_worker_count;
        BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
        slots.resize(slot_count);
        for (Slot& slot : slots) {
            slot.capacity = capacity;
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

    static std::vector<HistoryEntry> materialize_depth(
        CandidateHistoryMode mode,
        const std::filesystem::path& path,
        const CandidateMeta* host,
        std::uint32_t count) {
        std::vector<HistoryEntry> entries(static_cast<std::size_t>(count));
        for (std::uint32_t i = 0; i < count; ++i) {
            entries[static_cast<std::size_t>(i)] = compress_candidate(host[i]);
        }
        if (mode == CandidateHistoryMode::Ram) {
            return entries;
        }
        std::ofstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("cannot open candidate history file for write: " + path.string());
        }
        if (count != 0U) {
            file.write(
                reinterpret_cast<const char*>(entries.data()),
                static_cast<std::streamsize>(static_cast<std::uint64_t>(count) * sizeof(HistoryEntry)));
        }
        if (!file) {
            throw std::runtime_error("candidate history write failed: " + path.string());
        }
        return {};
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
                        slot.writer = std::async(
                            std::launch::async,
                            &CpuCandidateHistory::materialize_depth,
                            mode,
                            slot.path,
                            slot.host,
                            slot.count);
                        progressed = true;
                    } else if (status != cudaErrorNotReady) {
                        BEAM_CUDA_CHECK(status);
                    }
                }
                if (!slot.copy_pending && !slot.free && slot.writer.valid()) {
                    const bool ready = wait_all ||
                        slot.writer.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
                    if (ready) {
                        std::vector<HistoryEntry> ram_data = slot.writer.get();
                        if (mode == CandidateHistoryMode::Ram) {
                            if (slot.depth_index >= ram_depths.size()) {
                                throw std::runtime_error("candidate history RAM depth index missing");
                            }
                            ram_depths[slot.depth_index] = std::move(ram_data);
                            ram_ready[slot.depth_index] = true;
                            if (slot.depth_index > 0U) {
                                prune_dirty[slot.depth_index - 1U] = true;
                            }
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
        slot.copy_pending = true;
        slot.free = false;
        if (depth_index != depth_counts.size()) {
            throw std::runtime_error("candidate history commits must be depth-ordered");
        }
        depth_files.push_back(mode == CandidateHistoryMode::Disk ? slot.path : std::filesystem::path{});
        depth_counts.push_back(count);
        if (mode == CandidateHistoryMode::Ram) {
            ram_depths.emplace_back();
            ram_ready.push_back(false);
            prune_dirty.push_back(false);
        }
        bytes_received += static_cast<std::uint64_t>(count) * sizeof(CandidateMeta);
        bytes_stored += static_cast<std::uint64_t>(count) * sizeof(HistoryEntry);
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
};

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
    }
    return snapshot;
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

struct HostWeightBytes {
    std::vector<std::byte> input_weight;
    std::vector<std::byte> input_bias;
    std::vector<std::byte> hidden_weight;
    std::vector<std::byte> hidden_bias;
    std::vector<std::byte> residual0_fc1_weight;
    std::vector<std::byte> residual0_fc1_bias;
    std::vector<std::byte> residual0_fc2_weight;
    std::vector<std::byte> residual0_fc2_bias;
    std::vector<std::byte> residual1_fc1_weight;
    std::vector<std::byte> residual1_fc1_bias;
    std::vector<std::byte> residual1_fc2_weight;
    std::vector<std::byte> residual1_fc2_bias;
    std::vector<std::byte> output_weight;
    std::vector<std::byte> output_bias;
};

HostWeightBytes load_stream1_weights(const std::filesystem::path& dir) {
    constexpr std::size_t fp16 = sizeof(std::uint16_t);
    HostWeightBytes weights;
    weights.input_weight = read_binary_exact(dir / "input_weight_hxk.fp16", STATE_LEN * STREAM1_MODEL_CLASSES * STREAM1_HIDDEN1 * fp16);
    weights.input_bias = read_binary_exact(dir / "input_bias.fp16", STREAM1_HIDDEN1 * fp16);
    weights.hidden_weight = read_binary_exact(dir / "hidden_weight_hxk.fp16", STREAM1_HIDDEN1 * STREAM1_HIDDEN2 * fp16);
    weights.hidden_bias = read_binary_exact(dir / "hidden_bias.fp16", STREAM1_HIDDEN2 * fp16);
    weights.residual0_fc1_weight = read_binary_exact(dir / "residual0_fc1_weight_hxk.fp16", STREAM1_HIDDEN2 * STREAM1_HIDDEN2 * fp16);
    weights.residual0_fc1_bias = read_binary_exact(dir / "residual0_fc1_bias.fp16", STREAM1_HIDDEN2 * fp16);
    weights.residual0_fc2_weight = read_binary_exact(dir / "residual0_fc2_weight_hxk.fp16", STREAM1_HIDDEN2 * STREAM1_HIDDEN2 * fp16);
    weights.residual0_fc2_bias = read_binary_exact(dir / "residual0_fc2_bias.fp16", STREAM1_HIDDEN2 * fp16);
    weights.residual1_fc1_weight = read_binary_exact(dir / "residual1_fc1_weight_hxk.fp16", STREAM1_HIDDEN2 * STREAM1_HIDDEN2 * fp16);
    weights.residual1_fc1_bias = read_binary_exact(dir / "residual1_fc1_bias.fp16", STREAM1_HIDDEN2 * fp16);
    weights.residual1_fc2_weight = read_binary_exact(dir / "residual1_fc2_weight_hxk.fp16", STREAM1_HIDDEN2 * STREAM1_HIDDEN2 * fp16);
    weights.residual1_fc2_bias = read_binary_exact(dir / "residual1_fc2_bias.fp16", STREAM1_HIDDEN2 * fp16);
    weights.output_weight = read_binary_exact(dir / "output_weight_hxk.fp16", STREAM1_HIDDEN2 * MOVE_COUNT * fp16);
    weights.output_bias = read_binary_exact(dir / "output_bias.fp16", MOVE_COUNT * fp16);
    return weights;
}

const half* weight_half_data(const std::vector<std::byte>& bytes) {
    return reinterpret_cast<const half*>(bytes.data());
}

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
    const HostWeightBytes& weights) {
    const half* input_weight = weight_half_data(weights.input_weight);
    const half* input_bias = weight_half_data(weights.input_bias);
    const half* hidden_weight = weight_half_data(weights.hidden_weight);
    const half* hidden_bias = weight_half_data(weights.hidden_bias);
    const half* residual_fc1_weight[STREAM1_RESIDUAL_COUNT] = {
        weight_half_data(weights.residual0_fc1_weight),
        weight_half_data(weights.residual1_fc1_weight)};
    const half* residual_fc1_bias[STREAM1_RESIDUAL_COUNT] = {
        weight_half_data(weights.residual0_fc1_bias),
        weight_half_data(weights.residual1_fc1_bias)};
    const half* residual_fc2_weight[STREAM1_RESIDUAL_COUNT] = {
        weight_half_data(weights.residual0_fc2_weight),
        weight_half_data(weights.residual1_fc2_weight)};
    const half* residual_fc2_bias[STREAM1_RESIDUAL_COUNT] = {
        weight_half_data(weights.residual0_fc2_bias),
        weight_half_data(weights.residual1_fc2_bias)};
    const half* output_weight = weight_half_data(weights.output_weight);
    const half* output_bias = weight_half_data(weights.output_bias);

    std::vector<half> hidden1(STREAM1_HIDDEN1);
    for (std::uint32_t h = 0; h < STREAM1_HIDDEN1; ++h) {
        float acc = __half2float(input_bias[h]);
        for (std::uint32_t p = 0; p < STATE_LEN; ++p) {
            const std::uint32_t value = static_cast<std::uint32_t>(state.v[p]);
            const std::uint32_t idx = (p * STREAM1_MODEL_CLASSES + value) * STREAM1_HIDDEN1 + h;
            acc += __half2float(input_weight[idx]);
        }
        hidden1[h] = __float2half(acc > 0.0f ? acc : 0.0f);
    }

    std::vector<half> hidden2 =
        stream1_reference_linear(hidden1, hidden_weight, STREAM1_HIDDEN1, STREAM1_HIDDEN2);
    stream1_reference_bias_relu(hidden2, hidden_bias);

    std::vector<half> residual = hidden2;
    for (std::uint32_t block = 0; block < STREAM1_RESIDUAL_COUNT; ++block) {
        std::vector<half> tmp =
            stream1_reference_linear(residual, residual_fc1_weight[block], STREAM1_HIDDEN2, STREAM1_HIDDEN2);
        stream1_reference_bias_relu(tmp, residual_fc1_bias[block]);
        tmp = stream1_reference_linear(tmp, residual_fc2_weight[block], STREAM1_HIDDEN2, STREAM1_HIDDEN2);
        stream1_reference_residual_add_bias_relu(tmp, residual, residual_fc2_bias[block]);
        residual = std::move(tmp);
    }

    const std::vector<half> output =
        stream1_reference_linear(residual, output_weight, STREAM1_HIDDEN2, MOVE_COUNT);
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
    const HostWeightBytes& weights,
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

void copy_bytes_to_device(void* dst, const std::vector<std::byte>& bytes, const char* name) {
    if (dst == nullptr || bytes.empty()) {
        throw std::runtime_error(std::string("invalid device copy input: ") + name);
    }
    BEAM_CUDA_CHECK(cudaMemcpy(dst, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
}

void require_aligned(const void* ptr, std::uintptr_t alignment, const char* name) {
    if (reinterpret_cast<std::uintptr_t>(ptr) % alignment != 0) {
        throw std::runtime_error(std::string("device pointer alignment failed: ") + name);
    }
}

std::uint64_t estimate_stream1_weight_bytes() {
    constexpr std::uint64_t fp16 = sizeof(std::uint16_t);
    std::uint64_t total = 0;
    total += static_cast<std::uint64_t>(STATE_LEN) * STREAM1_MODEL_CLASSES * STREAM1_HIDDEN1 * fp16;
    total += static_cast<std::uint64_t>(STREAM1_HIDDEN1) * fp16;
    total += static_cast<std::uint64_t>(STREAM1_HIDDEN1) * STREAM1_HIDDEN2 * fp16;
    total += static_cast<std::uint64_t>(STREAM1_HIDDEN2) * fp16;
    total += 2ULL * static_cast<std::uint64_t>(STREAM1_HIDDEN2) * STREAM1_HIDDEN2 * fp16;
    total += 2ULL * static_cast<std::uint64_t>(STREAM1_HIDDEN2) * fp16;
    total += 2ULL * static_cast<std::uint64_t>(STREAM1_HIDDEN2) * STREAM1_HIDDEN2 * fp16;
    total += 2ULL * static_cast<std::uint64_t>(STREAM1_HIDDEN2) * fp16;
    total += static_cast<std::uint64_t>(STREAM1_HIDDEN2) * MOVE_COUNT * fp16;
    total += static_cast<std::uint64_t>(MOVE_COUNT) * fp16;
    return total;
}

std::uint64_t estimate_read_only_table_bytes() {
    return static_cast<std::uint64_t>(MOVE_COUNT) * STATE_STORAGE_LEN * sizeof(std::uint8_t) +
           sizeof(State128) +
           static_cast<std::uint64_t>(STATE_STORAGE_LEN) * STATE_VALUE_PAD * sizeof(Hash128);
}

std::uint64_t estimate_stream1_scratch_bytes(std::uint32_t b_micro) {
    return static_cast<std::uint64_t>(b_micro) *
           (STREAM1_HIDDEN1 + 2ULL * STREAM1_HIDDEN2 + MOVE_COUNT) *
           sizeof(half);
}

std::uint64_t estimate_non_static_device_bytes(const RuntimeConfig& config) {
    return estimate_read_only_table_bytes() +
           estimate_stream1_weight_bytes() +
           estimate_stream1_scratch_bytes(config.b_micro);
}

struct RuntimeConfigBuild {
    RuntimeConfig config;
    StaticMemoryPlan plan;
    std::uint32_t stream3_ring_slots = 0;
    std::uint64_t gpu_headroom_bytes = 0;
    std::uint64_t gpu_budget_bytes = 0;
    std::uint64_t estimated_non_static_device_bytes = 0;
    std::uint64_t estimated_required_device_bytes = 0;
};

std::uint64_t beam_alignment_for(const RuntimeConfig& config) {
    return static_cast<std::uint64_t>(config.world_size) *
           static_cast<std::uint64_t>(config.shard_count) *
           static_cast<std::uint64_t>(config.stream4_batch_candidates_per_shard_unit);
}

void set_exact_frontier_capacity(RuntimeConfig& config) {
    const std::uint64_t alignment = beam_alignment_for(config);
    const std::uint64_t effective = round_up(config.user_global_beam_width, alignment);
    config.global_beam_width_max_safe = effective;
    if (env_present("BEAM_GLOBAL_BEAM_WIDTH_MAX_SAFE")) {
        config.global_beam_width_max_safe = env_u64("BEAM_GLOBAL_BEAM_WIDTH_MAX_SAFE", config.global_beam_width_max_safe);
        if (config.global_beam_width_max_safe < effective) {
            throw std::invalid_argument("BEAM_GLOBAL_BEAM_WIDTH_MAX_SAFE is smaller than GLOBAL_BEAM_WIDTH_EFFECTIVE");
        }
    }
}

std::uint64_t ring_slot_candidate_count(const RuntimeConfig& config) {
    return static_cast<std::uint64_t>(config.b_micro) * static_cast<std::uint64_t>(MOVE_COUNT);
}

std::uint32_t derive_stream4_batch_candidates(const RuntimeConfig& config, std::uint64_t local_frontier_capacity) {
    if (env_present("BEAM_STREAM4_BATCH_CANDIDATES")) {
        const std::uint32_t batch = env_u32("BEAM_STREAM4_BATCH_CANDIDATES", 65536);
        if (batch == 0U) {
            throw std::invalid_argument("BEAM_STREAM4_BATCH_CANDIDATES must be nonzero");
        }
        if (batch % config.stream4_batch_candidates_per_shard_unit != 0U) {
            throw std::invalid_argument("BEAM_STREAM4_BATCH_CANDIDATES must be aligned to STREAM4_BATCH_CANDIDATES_PER_SHARD_UNIT");
        }
        return batch;
    }
    const std::uint64_t slot_candidates = ring_slot_candidate_count(config);
    const std::uint64_t min_batch = env_u64("BEAM_MIN_STREAM4_BATCH_CANDIDATES", slot_candidates);
    const std::uint64_t shard_limit = env_present("BEAM_SHARD_COUNT")
        ? std::max<std::uint64_t>(config.shard_count, 1ULL)
        : std::max<std::uint64_t>(env_u32("BEAM_MAX_SHARD_COUNT", 128), 1U);
    const std::uint64_t batch_by_shard_limit = ceil_div_u64(local_frontier_capacity, shard_limit);
    const std::uint64_t raw_batch = std::max({slot_candidates, min_batch, batch_by_shard_limit});
    return round_up_u32(checked_u32(raw_batch, "stream4_batch_candidates"),
                        config.stream4_batch_candidates_per_shard_unit);
}

std::uint32_t derive_shard_count_for_stream4_batch(
    const RuntimeConfig& config,
    std::uint64_t local_frontier_capacity) {
    if (env_present("BEAM_SHARD_COUNT")) {
        const std::uint32_t shard_count = env_u32("BEAM_SHARD_COUNT", 1);
        if (shard_count == 0U) {
            throw std::invalid_argument("BEAM_SHARD_COUNT must be nonzero");
        }
        return shard_count;
    }
    const std::uint64_t slot_candidates = ring_slot_candidate_count(config);
    std::uint64_t shard_count = std::max<std::uint64_t>(
        1ULL,
        ceil_div_u64(local_frontier_capacity, config.stream4_batch_candidates));
    while ((shard_count * static_cast<std::uint64_t>(config.stream4_batch_candidates)) % slot_candidates != 0ULL) {
        ++shard_count;
    }
    return checked_u32(shard_count, "shard_count");
}

std::uint32_t set_stream3_batch_from_stream4(RuntimeConfig& config) {
    const std::uint64_t slot_candidates = ring_slot_candidate_count(config);
    const std::uint64_t stream3_batch =
        static_cast<std::uint64_t>(config.shard_count) *
        static_cast<std::uint64_t>(config.stream4_batch_candidates);
    if (stream3_batch == 0ULL) {
        throw std::invalid_argument("STREAM3_BATCH_CANDIDATES must be nonzero");
    }
    if (stream3_batch % slot_candidates != 0ULL) {
        throw std::invalid_argument("STREAM3_BATCH_CANDIDATES must be divisible by B_MICRO * MOVE_COUNT");
    }
    config.stream3_batch_candidates = checked_u32(stream3_batch, "stream3_batch_candidates");
    return checked_u32(stream3_batch / slot_candidates, "ring_slot_count");
}

RuntimeConfigBuild build_runtime_config_from_budget(
    std::uint64_t beam,
    std::uint32_t world_size,
    std::uint32_t local_rank,
    std::uint64_t free_before_bytes) {
    RuntimeConfigBuild build;
    RuntimeConfig& config = build.config;
    config.user_global_beam_width = beam;
    config.world_size = world_size;
    config.local_rank = local_rank;
    config.b_micro = env_u32("BEAM_B_MICRO", 8192);
    config.ring_count = env_u32("BEAM_RING_COUNT", 2);
    config.stream4_batch_candidates_per_shard_unit =
        env_u32("BEAM_STREAM4_BATCH_CANDIDATES_PER_SHARD_UNIT", 1024);
    config.stream4_active_sort_slots = env_u32("BEAM_STREAM4_ACTIVE_SORT_SLOTS", 4);
    config.global_threshold_update_period_shards =
        env_u32("BEAM_GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS", 64);
    config.solved_result_capacity = env_u32("BEAM_SOLVED_RESULT_CAPACITY", 1024);
    build.gpu_headroom_bytes = env_u64("BEAM_GPU_HEADROOM_BYTES", 768ULL * 1024ULL * 1024ULL);
    build.gpu_budget_bytes =
        free_before_bytes > build.gpu_headroom_bytes ? free_before_bytes - build.gpu_headroom_bytes : 0ULL;
    if (config.b_micro == 0U || config.ring_count == 0U || config.stream4_batch_candidates_per_shard_unit == 0U) {
        throw std::invalid_argument("B_MICRO, RING_COUNT, and STREAM4_BATCH_CANDIDATES_PER_SHARD_UNIT must be nonzero");
    }

    if (env_present("BEAM_SHARD_COUNT") && env_present("BEAM_STREAM4_BATCH_CANDIDATES")) {
        config.shard_count = env_u32("BEAM_SHARD_COUNT", 1);
        config.stream4_batch_candidates = derive_stream4_batch_candidates(config, ceil_div_u64(beam, world_size));
        set_exact_frontier_capacity(config);
        build.stream3_ring_slots = set_stream3_batch_from_stream4(config);
    } else {
        config.shard_count = env_present("BEAM_SHARD_COUNT")
            ? env_u32("BEAM_SHARD_COUNT", 1)
            : env_u32("BEAM_INITIAL_SHARD_COUNT", 1);
        for (std::uint32_t iteration = 0; iteration < 16U; ++iteration) {
            set_exact_frontier_capacity(config);
            const std::uint64_t local_frontier_capacity =
                ceil_div_u64(config.global_beam_width_max_safe, world_size);
            config.stream4_batch_candidates =
                derive_stream4_batch_candidates(config, local_frontier_capacity);
            const std::uint32_t next_shard_count =
                derive_shard_count_for_stream4_batch(config, local_frontier_capacity);
            if (next_shard_count == config.shard_count) {
                break;
            }
            config.shard_count = next_shard_count;
            if (iteration == 15U) {
                throw std::runtime_error("runtime config fixed-point derivation did not converge");
            }
        }
        set_exact_frontier_capacity(config);
        build.stream3_ring_slots = set_stream3_batch_from_stream4(config);
    }
    const std::uint64_t local_frontier_capacity =
        ceil_div_u64(config.global_beam_width_max_safe, world_size);
    const std::uint64_t stream4_aggregate_batch =
        static_cast<std::uint64_t>(config.shard_count) *
        static_cast<std::uint64_t>(config.stream4_batch_candidates);
    if (stream4_aggregate_batch < local_frontier_capacity) {
        throw std::invalid_argument("STREAM3/STREAM4 aggregate batch is smaller than local frontier capacity");
    }

    auto refresh_spill_capacity = [&]() {
        if (env_present("BEAM_GLOBAL_SPILL_CAPACITY")) {
            config.global_spill_capacity = env_u32("BEAM_GLOBAL_SPILL_CAPACITY", 1U << 20);
        } else {
            config.global_spill_capacity =
                round_up_u32(config.stream3_batch_candidates, config.stream4_batch_candidates_per_shard_unit);
        }
    };

    refresh_spill_capacity();

    auto refresh_plan = [&]() {
        build.plan = make_static_memory_plan(config);
        build.estimated_non_static_device_bytes = estimate_non_static_device_bytes(config);
        build.estimated_required_device_bytes =
            static_cast<std::uint64_t>(build.plan.total_device_bytes) +
            build.estimated_non_static_device_bytes;
    };
    refresh_plan();

    while (build.estimated_required_device_bytes > build.gpu_budget_bytes &&
           !env_present("BEAM_STREAM4_ACTIVE_SORT_SLOTS") &&
           config.stream4_active_sort_slots > 1U) {
        config.stream4_active_sort_slots = std::max<std::uint32_t>(1U, config.stream4_active_sort_slots / 2U);
        refresh_spill_capacity();
        refresh_plan();
    }

    if (build.estimated_required_device_bytes > build.gpu_budget_bytes) {
        throw std::runtime_error(
            "derived GPU memory budget exceeded: required=" +
            std::to_string(build.estimated_required_device_bytes) +
            " budget=" + std::to_string(build.gpu_budget_bytes) +
            " free_before=" + std::to_string(free_before_bytes) +
            " headroom=" + std::to_string(build.gpu_headroom_bytes));
    }
    return build;
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
    const std::uint32_t world_size = argc == 6 ? static_cast<std::uint32_t>(parse_u64(argv[4], "world_size")) : 1U;
    const std::uint32_t local_rank = argc == 6 ? static_cast<std::uint32_t>(parse_u64(argv[5], "local_rank")) : 0U;
    if (world_size == 0 || world_size > 255 || local_rank >= world_size) {
        throw std::invalid_argument("world_size must be in [1,255] and local_rank must be less than world_size");
    }
    std::cout << std::unitbuf;

    BEAM_CUDA_CHECK(cudaSetDevice(0));
    std::size_t free_before = 0;
    std::size_t total_before = 0;
    BEAM_CUDA_CHECK(cudaMemGetInfo(&free_before, &total_before));

    const RuntimeConfigBuild config_build =
        build_runtime_config_from_budget(beam, world_size, local_rank, free_before);
    const RuntimeConfig config = config_build.config;
    const StaticMemoryPlan plan = config_build.plan;
#if BEAM_ENABLE_DEBUG_LOGS
    std::cout << "puzzle_id=" << puzzle_id << "\n";
    std::cout << "depth_limit=" << depth_limit << "\n";
    std::cout << "USER_GLOBAL_BEAM_WIDTH=" << beam << "\n";
    std::cout << "WORLD_SIZE=" << config.world_size << "\n";
    std::cout << "LOCAL_RANK=" << config.local_rank << "\n";
    std::cout << "B_MICRO=" << config.b_micro << "\n";
    std::cout << "STREAM3_RING_SLOTS=" << config_build.stream3_ring_slots << "\n";
    std::cout << "RING_COUNT=" << config.ring_count << "\n";
    std::cout << "RING_SLOT_COUNT=" << plan.derived.ring_slot_count << "\n";
    std::cout << "STREAM3_BATCH_CANDIDATES=" << config.stream3_batch_candidates << "\n";
    std::cout << "STREAM5_BATCH_CANDIDATES=" << config.stream3_batch_candidates << "\n";
    std::cout << "SHARD_COUNT=" << config.shard_count << "\n";
    std::cout << "STREAM4_BATCH_CANDIDATES=" << config.stream4_batch_candidates << "\n";
    std::cout << "STREAM4_BATCH_CANDIDATES_PER_SHARD_UNIT=" << config.stream4_batch_candidates_per_shard_unit << "\n";
    std::cout << "STREAM4_ACTIVE_SORT_SLOTS=" << config.stream4_active_sort_slots << "\n";
    std::cout << "GLOBAL_SPILL_CAPACITY=" << config.global_spill_capacity << "\n";
    std::cout << "GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS=" << config.global_threshold_update_period_shards << "\n";
    std::cout << "GLOBAL_BEAM_WIDTH_EFFECTIVE=" << plan.derived.global_beam_width_effective << "\n";
    std::cout << "GLOBAL_BEAM_WIDTH_MAX_SAFE=" << config.global_beam_width_max_safe << "\n";
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
    std::cout << "layout_streams_bytes=" << plan.layout_streams_bytes << "\n";
    std::cout << "layout_final_bytes=" << plan.layout_final_bytes << "\n";
    std::cout << "frontier_state_capacity=" << plan.frontier_states << "\n";
#endif

    const std::filesystem::path generator_path = "FullBeamNice/generators/p900.json";
    const std::filesystem::path puzzle_info_path = "data/puzzle_info.json";
    const std::filesystem::path test_csv_path = "data/test.csv";
    const std::filesystem::path weight_dir = env_path("BEAM_WEIGHT_DIR", "stream1_weights");
    const std::vector<std::uint8_t> host_generators = load_p900_generators(generator_path);
    const std::vector<std::string> host_move_names = load_p900_move_names(generator_path);
    const State128 host_central = load_central_state(puzzle_info_path);
    const State128 host_initial = load_initial_state_from_test_csv(test_csv_path, puzzle_id);
    const ZobristTable host_zobrist = make_deterministic_zobrist(0xC0DEC0DEULL);
    const HostWeightBytes host_weights = load_stream1_weights(weight_dir);
#if BEAM_ENABLE_DEBUG_LOGS
    std::cout << "real_assets=enabled\n";
    std::cout << "generator_path=" << generator_path.string() << "\n";
    std::cout << "puzzle_info_path=" << puzzle_info_path.string() << "\n";
    std::cout << "test_csv_path=" << test_csv_path.string() << "\n";
    std::cout << "weight_dir=" << weight_dir.string() << "\n";
#endif

    StaticDeviceMemory memory;
    allocate_static_device_memory(plan, memory);
    BEAM_CUDA_CHECK(cudaMemset(memory.allocation, 0, memory.allocation_bytes));
    BEAM_CUDA_CHECK(cudaMemset(memory.current_frontier_states, 0, plan.current_frontier_bytes));
    BEAM_CUDA_CHECK(cudaMemcpy(memory.current_frontier_states, &host_initial, sizeof(State128), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemset(memory.streams.current_threshold, 0xff, sizeof(std::uint32_t)));

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
    half* input_weight = nullptr;
    half* input_bias = nullptr;
    half* hidden_weight = nullptr;
    half* hidden_bias = nullptr;
    half* residual0_fc1_weight = nullptr;
    half* residual0_fc1_bias = nullptr;
    half* residual0_fc2_weight = nullptr;
    half* residual0_fc2_bias = nullptr;
    half* residual1_fc1_weight = nullptr;
    half* residual1_fc1_bias = nullptr;
    half* residual1_fc2_weight = nullptr;
    half* residual1_fc2_bias = nullptr;
    half* output_weight = nullptr;
    half* output_bias = nullptr;
    half* hidden1 = nullptr;
    half* hidden2 = nullptr;
    half* residual = nullptr;
    half* output = nullptr;
    constexpr std::uint32_t hidden1_cols = STREAM1_HIDDEN1;
    constexpr std::uint32_t hidden2_cols = STREAM1_HIDDEN2;
    constexpr std::uint32_t residual_count = STREAM1_RESIDUAL_COUNT;
    BEAM_CUDA_CHECK(cudaMalloc(&generators, host_generators.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&central_state, sizeof(State128)));
    BEAM_CUDA_CHECK(cudaMalloc(&zobrist, STATE_STORAGE_LEN * STATE_VALUE_PAD * sizeof(Hash128)));
    BEAM_CUDA_CHECK(cudaMalloc(&input_weight, host_weights.input_weight.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&input_bias, host_weights.input_bias.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&hidden_weight, host_weights.hidden_weight.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&hidden_bias, host_weights.hidden_bias.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&residual0_fc1_weight, host_weights.residual0_fc1_weight.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&residual0_fc1_bias, host_weights.residual0_fc1_bias.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&residual0_fc2_weight, host_weights.residual0_fc2_weight.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&residual0_fc2_bias, host_weights.residual0_fc2_bias.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&residual1_fc1_weight, host_weights.residual1_fc1_weight.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&residual1_fc1_bias, host_weights.residual1_fc1_bias.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&residual1_fc2_weight, host_weights.residual1_fc2_weight.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&residual1_fc2_bias, host_weights.residual1_fc2_bias.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&output_weight, host_weights.output_weight.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&output_bias, host_weights.output_bias.size()));
    BEAM_CUDA_CHECK(cudaMalloc(&hidden1, config.b_micro * hidden1_cols * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&hidden2, config.b_micro * hidden2_cols * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&residual, config.b_micro * hidden2_cols * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMalloc(&output, config.b_micro * MOVE_COUNT * sizeof(half)));
    require_aligned(generators, 16, "generators");
    require_aligned(central_state, alignof(State128), "central_state");
    require_aligned(zobrist, alignof(Hash128), "zobrist");
    require_aligned(input_weight, 16, "input_weight");
    require_aligned(hidden_weight, 16, "hidden_weight");
    require_aligned(output_weight, 16, "output_weight");
    BEAM_CUDA_CHECK(cudaMemcpy(generators, host_generators.data(), host_generators.size(), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(central_state, &host_central, sizeof(State128), cudaMemcpyHostToDevice));
    BEAM_CUDA_CHECK(cudaMemcpy(zobrist, &host_zobrist[0][0], STATE_STORAGE_LEN * STATE_VALUE_PAD * sizeof(Hash128), cudaMemcpyHostToDevice));
    copy_bytes_to_device(input_weight, host_weights.input_weight, "input_weight");
    copy_bytes_to_device(input_bias, host_weights.input_bias, "input_bias");
    copy_bytes_to_device(hidden_weight, host_weights.hidden_weight, "hidden_weight");
    copy_bytes_to_device(hidden_bias, host_weights.hidden_bias, "hidden_bias");
    copy_bytes_to_device(residual0_fc1_weight, host_weights.residual0_fc1_weight, "residual0_fc1_weight");
    copy_bytes_to_device(residual0_fc1_bias, host_weights.residual0_fc1_bias, "residual0_fc1_bias");
    copy_bytes_to_device(residual0_fc2_weight, host_weights.residual0_fc2_weight, "residual0_fc2_weight");
    copy_bytes_to_device(residual0_fc2_bias, host_weights.residual0_fc2_bias, "residual0_fc2_bias");
    copy_bytes_to_device(residual1_fc1_weight, host_weights.residual1_fc1_weight, "residual1_fc1_weight");
    copy_bytes_to_device(residual1_fc1_bias, host_weights.residual1_fc1_bias, "residual1_fc1_bias");
    copy_bytes_to_device(residual1_fc2_weight, host_weights.residual1_fc2_weight, "residual1_fc2_weight");
    copy_bytes_to_device(residual1_fc2_bias, host_weights.residual1_fc2_bias, "residual1_fc2_bias");
    copy_bytes_to_device(output_weight, host_weights.output_weight, "output_weight");
    copy_bytes_to_device(output_bias, host_weights.output_bias, "output_bias");
    TrackedSolutionPrefix tracked_solution;
    tracked_solution.initialize(puzzle_id, host_initial, host_generators, host_zobrist, host_move_names);
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
    const std::array<const half*, residual_count> residual_fc1_weight{residual0_fc1_weight, residual1_fc1_weight};
    const std::array<const half*, residual_count> residual_fc1_bias{residual0_fc1_bias, residual1_fc1_bias};
    const std::array<const half*, residual_count> residual_fc2_weight{residual0_fc2_weight, residual1_fc2_weight};
    const std::array<const half*, residual_count> residual_fc2_bias{residual0_fc2_bias, residual1_fc2_bias};
    Stream1NetworkDims dims{STATE_LEN, STREAM1_MODEL_CLASSES, hidden1_cols, hidden2_cols, residual_count};
    DispatcherNetwork network{
        Stream1NetworkView{
            input_weight,
            input_bias,
            hidden_weight,
            hidden_bias,
            residual_fc1_weight.data(),
            residual_fc1_bias.data(),
            residual_fc2_weight.data(),
            residual_fc2_bias.data(),
            output_weight,
            output_bias,
            dims},
        Stream1CutlassScratch{hidden1, hidden2, residual, output}};
    DispatcherDeviceTables tables{generators, central_state, zobrist};
    Stream2SolvedBuffers solved{
        memory.solved_flag,
        memory.stop_flag,
        memory.solved_count,
        memory.solved_overflow,
        memory.solved_meta_list,
        memory.solved_depth_list,
        config.solved_result_capacity,
        memory.current_depth};
    instantiate_cuda_graph_job_templates(plan, memory, tables, network, solved, streams, events, graphs);
#if BEAM_ENABLE_DEBUG_LOGS
    std::cout << "runner_phase=graphs_instantiated\n";
#endif

    const auto start = std::chrono::steady_clock::now();
    CpuCandidateHistory history;
    const CandidateHistoryMode history_mode = parse_history_mode();
    const std::uint32_t history_slot_count = env_u32("BEAM_HISTORY_SLOT_COUNT", 3);
    const std::uint32_t history_worker_count = env_u32("BEAM_HISTORY_WORKERS", 1);
    const std::uint32_t depth_log_every = env_u32("BEAM_DEPTH_LOG_EVERY", 1);
    history.dir = make_history_dir(puzzle_id, depth_limit, beam);
    history.initialize(
        history_mode,
        static_cast<std::uint32_t>(plan.frontier_states),
        history_slot_count,
        history_worker_count);
#if BEAM_ENABLE_DEBUG_LOGS
    std::cout << "candidate_history_dir=" << history.dir.string() << "\n";
    std::cout << "candidate_history_mode=" << history_mode_name(history.mode) << "\n";
    std::cout << "candidate_history_slots=" << history_slot_count << "\n";
    std::cout << "candidate_history_workers=" << history_worker_count << "\n";
#endif
    std::uint64_t frontier_size = 1;
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
        const GeneratedTrackRequest generated_track_request =
            tracked_solution.generated_request_for_depth(depth);
        const DepthDispatchState state =
            run_depth_cuda_graphs(plan, memory, graphs, streams, frontier_size, generated_track_request);
        if (!state.depth_drained) {
            throw std::runtime_error("depth did not drain");
        }
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
        log_stream1_move_score_comparison(
            puzzle_id,
            depth,
            tracked_solution,
            state.tracked_generated,
            host_weights,
            host_move_names);
        tracked_solution.log_stream4(
            puzzle_id,
            depth,
            state.tracked_stream4,
            state.tracked_stream4_events,
            host_move_names);
        total_threshold_updates += state.threshold_updates;
        ++completed_depths;

        const SolvedSnapshot solved_snapshot = read_solved_snapshot(memory, config.solved_result_capacity);
        if (solved_snapshot.found) {
            if (solved_snapshot.meta.empty() || solved_snapshot.depth.empty()) {
                throw std::runtime_error("solved flag set but solved metadata list is empty");
            }
            const CandidateMeta solved_meta = solved_snapshot.meta.front();
            const std::uint32_t solved_depth = solved_snapshot.depth.front();
            history.finish_all();
            const ReconstructedSolution solution =
                reconstruct_solution_from_history(history, solved_meta, solved_depth);
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
                solved_snapshot);
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
                      << " solved_count=" << solved_snapshot.count
                      << " solved_overflow=" << solved_snapshot.overflow
                      << " stop_requested=" << (state.stop_requested ? 1 : 0) << "\n";
            }
#endif
            break;
        }

        CpuCandidateHistory::Slot& history_slot = history.acquire_slot();
        const FinalizeDepthState final_state = finalize_depth_single_gpu(
            plan,
            memory,
            tables,
            streams,
            history_slot.host,
            history_slot.capacity,
            history.copy_stream,
            history_slot.copy_done,
            tracked_solution.hash_for_depth(depth));
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
        history.commit_slot(history_slot, depth, final_state.final_candidate_count);
        history.pump_completed(false);
        frontier_size = final_state.next_frontier_size;
        last_final_frontier_size = frontier_size;
        last_final_threshold = final_state.final_threshold;
        const auto depth_end = std::chrono::steady_clock::now();
        const double depth_sec = std::chrono::duration<double>(depth_end - depth_start).count();
#if BEAM_ENABLE_DEPTH_LOGS
        if (emit_depth_log) {
        std::cout << "depth_done=" << depth
                  << " depth_sec=" << depth_sec
                  << " ring_slot_jobs=" << state.ring_slot_jobs_launched
                  << " stream3_jobs=" << state.stream3_jobs_launched
                  << " stream4_jobs=" << state.stream4_jobs_launched
                  << " stream4_slots_used=" << state.stream4_active_sort_slots_used
                  << " threshold_updates_depth=" << state.threshold_updates
                  << " final_threshold=" << final_state.final_threshold
                  << " final_candidate_count=" << final_state.final_candidate_count
                  << " final_request_count=" << final_state.final_request_count
                  << " next_frontier_size=" << frontier_size
                  << " history_bytes_received=" << history.bytes_received
                  << " history_bytes_stored=" << history.bytes_stored
                  << " history_bytes_pruned=" << history.bytes_pruned << "\n";
        }
#endif
        if (frontier_size == 0) {
            break;
        }
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
    cudaFree(generators);
    cudaFree(central_state);
    cudaFree(zobrist);
    cudaFree(input_weight);
    cudaFree(input_bias);
    cudaFree(hidden_weight);
    cudaFree(hidden_bias);
    cudaFree(residual0_fc1_weight);
    cudaFree(residual0_fc1_bias);
    cudaFree(residual0_fc2_weight);
    cudaFree(residual0_fc2_bias);
    cudaFree(residual1_fc1_weight);
    cudaFree(residual1_fc1_bias);
    cudaFree(residual1_fc2_weight);
    cudaFree(residual1_fc2_bias);
    cudaFree(output_weight);
    cudaFree(output_bias);
    cudaFree(hidden1);
    cudaFree(hidden2);
    cudaFree(residual);
    cudaFree(output);
    return 0;
}
