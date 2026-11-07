#include "native_input.hpp"
#include "../third_party/nlohmann/json.hpp"
#include "../third_party/picosha2/picosha2.h"
#include <charconv>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>

namespace beam::input {
namespace {
using Json = nlohmann::ordered_json;
[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }
Json json(const std::string& text) {
    std::vector<std::set<std::string>> keys;
    auto callback = [&](int depth, Json::parse_event_t event, Json& value) {
        if (depth > 64) fail("JSON nesting exceeds 64");
        if (event == Json::parse_event_t::object_start) keys.emplace_back();
        if (event == Json::parse_event_t::key && !keys.back().insert(value.get<std::string>()).second)
            fail("duplicate JSON key");
        if (event == Json::parse_event_t::object_end) keys.pop_back();
        return true;
    };
    return Json::parse(text, callback);
}
std::uint64_t bounded(const Json& value, std::uint64_t limit) {
    if (!value.is_number_unsigned() || value.get<std::uint64_t>() >= limit)
        fail("expected bounded unsigned integer");
    return value.get<std::uint64_t>();
}
State128 state(const Json& value) {
    if (!value.is_array() || value.size() != STATE_LEN) fail("state length mismatch");
    State128 result{};
    for (unsigned i = 0; i < STATE_LEN; ++i)
        result.v[i] = static_cast<std::uint8_t>(bounded(value[i], STATE_VALUE_PAD));
    return result;
}
struct Moves { std::vector<std::uint8_t> data; std::vector<std::string> names; };
Moves moves(const std::filesystem::path& path) {
    const auto root = json(read_text_file(path));
    if (!root.is_object()) fail("generator root must be object");
    const Json* entries = nullptr;
    for (const char* key : {"actions", "moves", "generators"}) {
        if (!root.contains(key)) continue;
        if (entries) fail("ambiguous generator definitions");
        entries = &root.at(key);
    }
    if (!entries || (!entries->is_array() && !entries->is_object()) || entries->size() != MOVE_COUNT)
        fail("generator count/type mismatch");
    Moves result;
    for (auto it = entries->begin(); it != entries->end(); ++it) {
        if (!it->is_array() || it->size() != STATE_LEN) fail("permutation length mismatch");
        std::set<std::uint64_t> seen;
        for (const auto& value : *it) {
            const auto index = bounded(value, STATE_LEN);
            if (!seen.insert(index).second) fail("permutation is not bijective");
            result.data.push_back(static_cast<std::uint8_t>(index));
        }
        for (unsigned i = STATE_LEN; i < STATE_STORAGE_LEN; ++i)
            result.data.push_back(static_cast<std::uint8_t>(i));
        if (entries->is_object()) result.names.push_back(it.key());
    }
    const Json* names = nullptr;
    for (const char* key : {"names", "move_names"}) {
        if (!root.contains(key)) continue;
        if (names) fail("ambiguous move names");
        names = &root.at(key);
    }
    if (names) {
        if (!names->is_array() || names->size() != MOVE_COUNT) fail("move names mismatch");
        auto explicit_names = names->get<std::vector<std::string>>();
        if (!result.names.empty() && result.names != explicit_names) fail("move order mismatch");
        result.names = std::move(explicit_names);
    }
    std::set<std::string> unique;
    if (result.names.size() != MOVE_COUNT) fail("missing move names");
    for (const auto& name : result.names)
        if (name.empty() || !unique.insert(name).second) fail("empty/duplicate move name");
    return result;
}
// RFC4180 quoting, including escaped quotes and newlines within quoted fields.
std::vector<std::vector<std::string>> csv(const std::string& text) {
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    std::string field;
    bool quoted = false, closed = false, started = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (quoted) {
            if (c == '"') {
                if (i + 1 < text.size() && text[i + 1] == '"') { field += '"'; ++i; }
                else { quoted = false; closed = true; }
            } else field += c;
            continue;
        }
        if (c == ',' || c == '\n' || c == '\r') {
            row.push_back(field); field.clear(); closed = started = false;
            if (c != ',') {
                rows.push_back(std::move(row)); row.clear();
                if (c == '\r' && i + 1 < text.size() && text[i + 1] == '\n') ++i;
            }
        } else if (c == '"' && !started && !closed) { quoted = started = true; }
        else {
            if (closed || c == '"') fail("malformed CSV quoting");
            field += c; started = true;
        }
    }
    if (quoted) fail("unterminated CSV quote");
    if (started || closed || !row.empty()) { row.push_back(field); rows.push_back(std::move(row)); }
    return rows;
}
}
TensorBinding::TensorBinding(const std::string& manifest, bool required) {
    const auto root = json(manifest);
    if (!root.is_object()) fail("tensor manifest must be object");
    if (!root.contains("tensor_files")) {
        if (required) fail("missing tensor_files binding");
        return;
    }
    const auto& files = root.at("tensor_files");
    if (!files.is_object() || files.empty()) fail("invalid tensor_files binding");
    for (auto it = files.begin(); it != files.end(); ++it) {
        const auto& name = it.key();
        if (name.empty() || name == "." || name == ".." ||
            name.find_first_of("/\\:") != std::string::npos || name.find('\0') != std::string::npos)
            fail("invalid tensor filename");
        const auto& entry = it.value();
        if (!entry.is_object() || !entry.contains("size_bytes") || !entry.contains("sha256") ||
            !entry.at("size_bytes").is_number_unsigned() || !entry.at("sha256").is_string())
            fail("invalid tensor binding: " + name);
        const auto digest = entry.at("sha256").get<std::string>();
        if (digest.size() != 64 || digest.find_first_not_of("0123456789abcdef") != std::string::npos)
            fail("invalid tensor SHA-256: " + name);
        expected_.emplace(name, std::make_pair(entry.at("size_bytes").get<std::uint64_t>(), digest));
    }
    active_ = true;
}
void TensorBinding::verify(const std::string& name, const std::vector<std::byte>& bytes) {
    if (!active_) return;
    const auto it = expected_.find(name);
    if (it == expected_.end()) fail("unlisted tensor: " + name);
    if (it->second.first != bytes.size()) fail("tensor binding size mismatch: " + name);
    static const unsigned char empty = 0;
    const auto* begin = bytes.empty() ? &empty : reinterpret_cast<const unsigned char*>(bytes.data());
    if (picosha2::hash256_hex_string(begin, begin + bytes.size()) != it->second.second)
        fail("tensor SHA-256 mismatch: " + name);
    seen_.insert(name);
}
void TensorBinding::finish() const {
    if (active_ && seen_.size() != expected_.size()) fail("manifest contains unloaded tensors");
}
std::uint64_t parse_u64(const char* text, const char* context) {
    if (!text || !*text) fail(std::string(context) + ": empty integer");
    const char* end = text + std::strlen(text);
    for (const char* p = text; p != end; ++p)
        if (*p < '0' || *p > '9') fail(std::string(context) + ": invalid unsigned integer");
    std::uint64_t result{};
    const auto parsed = std::from_chars(text, end, result);
    if (parsed.ec != std::errc{} || parsed.ptr != end) fail(std::string(context) + ": integer overflow");
    return result;
}
std::uint32_t parse_u32(const char* text, const char* context) {
    const auto value = parse_u64(text, context);
    if (value > UINT32_MAX) fail(std::string(context) + ": uint32 overflow");
    return static_cast<std::uint32_t>(value);
}
bool json_has_field(const std::string& text, const char* key) try {
    const auto root = json(text);
    if (!root.is_object()) fail("JSON manifest root must be object");
    return root.contains(key);
} catch (const std::exception& error) {
    fail(std::string("manifest field ") + key + ": " + error.what());
}
std::uint32_t json_u32_field(const std::string& text, const char* key) try {
    const auto root = json(text);
    if (!root.is_object()) fail("JSON manifest root must be object");
    return static_cast<std::uint32_t>(bounded(root.at(key), std::uint64_t{UINT32_MAX} + 1));
} catch (const std::exception& error) {
    fail(std::string("manifest field ") + key + ": " + error.what());
}
std::string json_string_field(const std::string& text, const char* key) try {
    const auto root = json(text);
    if (!root.is_object()) fail("JSON manifest root must be object");
    return root.at(key).get<std::string>();
} catch (const std::exception& error) {
    fail(std::string("manifest field ") + key + ": " + error.what());
}
std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) fail("cannot open " + path.string());
    std::string result{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    if (file.bad()) fail("cannot read " + path.string());
    return result;
}
State128 parse_state_text(const std::string& text, const char* context) {
    try {
        const auto first = text.find_first_not_of(" \t\r\n");
        return state(json(first != std::string::npos && text[first] == '[' ? text : "[" + text + "]"));
    } catch (const std::exception& error) { fail(std::string(context) + ": " + error.what()); }
}
State128 load_central_state(const std::filesystem::path& path) {
    return state(json(read_text_file(path)).at("central_state"));
}
void validate_state(const State128& value, const State128& central, std::uint32_t classes, const char* context) {
    if (classes == 0 || classes > STATE_VALUE_PAD) fail(std::string(context) + ": invalid model alphabet");
    std::array<std::uint32_t, STATE_VALUE_PAD> actual{}, expected{};
    for (unsigned i = 0; i < STATE_LEN; ++i) {
        if (value.v[i] >= classes || central.v[i] >= classes)
            fail(std::string(context) + ": state exceeds model alphabet");
        ++actual[value.v[i]];
        ++expected[central.v[i]];
    }
    for (unsigned i = STATE_LEN; i < STATE_STORAGE_LEN; ++i)
        if (value.v[i] != 0 || central.v[i] != 0) fail(std::string(context) + ": nonzero state padding");
    if (actual != expected) fail(std::string(context) + ": state multiset differs from central state");
}
std::vector<std::uint8_t> load_p900_generators(const std::filesystem::path& path) { return moves(path).data; }
std::vector<std::string> load_p900_move_names(const std::filesystem::path& path) { return moves(path).names; }
std::map<std::uint64_t, State128> load_initial_states_from_test_csv(const std::filesystem::path& path) {
    const auto rows = csv(read_text_file(path));
    if (rows.empty()) fail("missing CSV header");
    std::map<std::string, std::size_t> columns;
    for (std::size_t i = 0; i < rows[0].size(); ++i)
        if (!columns.emplace(rows[0][i], i).second) fail("duplicate CSV column");
    if (!columns.count("initial_state_id") || !columns.count("initial_state")) fail("missing CSV columns");
    std::map<std::uint64_t, State128> result;
    for (std::size_t r = 1; r < rows.size(); ++r) {
        if (rows[r].size() != rows[0].size()) fail("CSV row width mismatch");
        const auto id = parse_u64(rows[r][columns.at("initial_state_id")].c_str(), "initial_state_id");
        const auto value = parse_state_text(rows[r][columns.at("initial_state")], "initial_state");
        if (!result.emplace(id, value).second) fail("duplicate puzzle ID");
    }
    return result;
}
State128 load_initial_state_from_test_csv(const std::filesystem::path& path, std::uint64_t id) {
    const auto states = load_initial_states_from_test_csv(path);
    const auto found = states.find(id);
    if (found == states.end()) fail("requested puzzle ID not found");
    return found->second;
}
}
