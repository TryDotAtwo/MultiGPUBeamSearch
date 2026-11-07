#pragma once
#include "types.hpp"
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace beam::input {
// Binds loaded bytes to a manifest; does not authenticate the manifest itself.
class TensorBinding {
public:
    TensorBinding(const std::string& manifest, bool required);
    void verify(const std::string& name, const std::vector<std::byte>& bytes);
    void finish() const;
private:
    bool active_ = false;
    std::map<std::string, std::pair<std::uint64_t, std::string>> expected_;
    std::set<std::string> seen_;
};
std::uint64_t parse_u64(const char*, const char*);
std::uint32_t parse_u32(const char*, const char*);
std::string read_text_file(const std::filesystem::path&);
bool json_has_field(const std::string&, const char*);
std::uint32_t json_u32_field(const std::string&, const char*);
std::string json_string_field(const std::string&, const char*);
State128 parse_state_text(const std::string&, const char*);
// A permutation search preserves the central state's multiset. Validate both
// states against the actual model alphabet before embedding/hash access.
void validate_state(const State128&, const State128& central, std::uint32_t classes, const char*);
State128 load_central_state(const std::filesystem::path&);
std::vector<std::uint8_t> load_p900_generators(const std::filesystem::path&);
std::vector<std::string> load_p900_move_names(const std::filesystem::path&);
std::map<std::uint64_t, State128> load_initial_states_from_test_csv(const std::filesystem::path&);
State128 load_initial_state_from_test_csv(const std::filesystem::path&, std::uint64_t);
}
