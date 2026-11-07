#include "../src/types.hpp"
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef BEAM_TEST_LEGACY_NATIVE_INPUT
namespace legacy {
using namespace beam;
#include "legacy_native_input.hpp"
std::uint32_t parse_u32(const char* s, const char* context) {
    return static_cast<std::uint32_t>(parse_u64(s, context));
}
}
namespace input = legacy;
#else
#include "../src/native_input.hpp"
namespace input = beam::input;
#endif

int main(int argc, char** argv) {
    using namespace beam;
#ifndef BEAM_TEST_LEGACY_NATIVE_INPUT
    if (argc == 6) {
        const auto generators = input::load_p900_generators(argv[1]);
        const auto names = input::load_p900_move_names(argv[1]);
        const auto central = input::load_central_state(argv[2]);
        const auto initial = input::load_initial_state_from_test_csv(argv[3], input::parse_u64(argv[4], "puzzle_id"));
        input::validate_state(initial, central, input::parse_u32(argv[5], "classes"), "real fixture");
        std::cout << "real_input PASS moves=" << names.size() << " generator_bytes=" << generators.size() << '\n';
        return 0;
    }
#endif
    unsigned failures = 0, checks = 0, serial = 0;
    auto reject = [&](const char* label, auto action) {
        ++checks;
        try { action(); ++failures; std::cerr << "FAIL accepted " << label << '\n'; }
        catch (const std::exception&) {}
    };
    auto require = [&](bool condition, const char* label) {
        ++checks;
        if (!condition) { ++failures; std::cerr << "FAIL " << label << '\n'; }
    };
    const auto directory = std::filesystem::path("test_results/native_input_fixture");
    std::filesystem::create_directories(directory);
    auto file = [&](const std::string& text) {
        auto path = directory / ("input_" + std::to_string(serial++) + ".txt");
        std::ofstream out(path, std::ios::binary); out << text; out.close();
        if (!out) throw std::runtime_error("fixture write failed");
        return path;
    };
    auto values = [&](const std::string& first) {
        std::string result = first;
        for (unsigned i = 1; i < STATE_LEN; ++i) result += "," + std::to_string(i);
        return result;
    };
    auto generators = [&](const std::string& first, bool extra = false) {
        std::string result = "{\"generators\":{";
        for (unsigned i = 0; i < MOVE_COUNT + (extra ? 1U : 0U); ++i) {
            if (i) result += ",";
            result += "\"m" + std::to_string(i) + "\":[" + values(i == 0 ? first : "0") + "]";
        }
        return result + "}}";
    };
    require(input::parse_u64("18446744073709551615", "test") == UINT64_MAX, "uint64 max");
    require(input::parse_u32("4294967295", "test") == UINT32_MAX, "uint32 max");
    for (const char* bad : {"-1", "+1", " 1", "1 ", "", "18446744073709551616", "9999999999999999999999999999999", "1x"})
        reject("unsigned number", [&] { input::parse_u64(bad, "test"); });
    reject("uint32 narrowing", [&] { input::parse_u32("4294967296", "test"); });
    auto state = input::parse_state_text(values("0"), "state");
    require(state.v[STATE_LEN - 1] == STATE_LEN - 1, "valid bare state");
#ifndef BEAM_TEST_LEGACY_NATIVE_INPUT
    input::validate_state(state, state, STATE_LEN, "valid");
    reject("model class bounds", [&] { input::validate_state(state, state, 6, "state"); });
    reject("zero model classes", [&] { input::validate_state(state, state, 0, "state"); });
    reject("too many model classes", [&] { input::validate_state(state, state, STATE_VALUE_PAD + 1, "state"); });
    auto altered = state;
    altered.v[0] = 1;
    reject("state multiset mismatch", [&] { input::validate_state(altered, state, STATE_LEN, "state"); });
    altered = state;
    altered.v[STATE_LEN] = 1;
    reject("state nonzero padding", [&] { input::validate_state(altered, state, STATE_LEN, "state"); });
#endif
    state = input::parse_state_text("[" + values("0") + "]", "state");
    for (unsigned i = STATE_LEN; i < STATE_STORAGE_LEN; ++i) require(state.v[i] == 0, "padding zero");
    for (const char* first : {"-1", "0.0", "true", "\"0\"", "128"})
        reject("state token", [&] { input::parse_state_text("[" + values(first) + "]", "state"); });
    reject("state trailing entry", [&] { input::parse_state_text("[" + values("0") + ",0]", "state"); });
    reject("state trailing junk", [&] { input::parse_state_text("[" + values("0") + "]junk", "state"); });
    auto valid_path = file(generators("0"));
    const auto valid_generators = input::load_p900_generators(valid_path);
    require(valid_generators.size() == MOVE_COUNT * STATE_STORAGE_LEN, "generator dimensions");
    require(input::load_p900_move_names(valid_path).size() == MOVE_COUNT, "move count");
    for (const char* first : {"-1", "1", "255", "0.0"})
        reject("invalid permutation", [&] { input::load_p900_generators(file(generators(first))); });
    reject("extra move", [&] { input::load_p900_generators(file(generators("0", true))); });
    reject("duplicate JSON key", [&] {
        input::load_central_state(file("{\"central_state\":[" + values("0") + "],\"central_state\":[" + values("0") + "]}"));
    });
    const std::string header = "initial_state_id,initial_state,comment\n";
    const std::string row = "1000,\"" + values("0") + "\",\"has , comma\"\n";
    const auto csv_state = input::load_initial_state_from_test_csv(file(header + row), 1000);
    require(csv_state.v[STATE_LEN - 1] == STATE_LEN - 1, "quoted CSV comment");
    reject("duplicate puzzle ID", [&] { input::load_initial_state_from_test_csv(file(header + row + row), 1000); });
    reject("CSV malformed state", [&] { input::load_initial_state_from_test_csv(file(header + "1000,\"" + values("-1") + "\",bad\n"), 1000); });
    std::cout << "native_input checks=" << checks << " failures=" << failures << '\n';
    return failures ? 1 : 0;
}
