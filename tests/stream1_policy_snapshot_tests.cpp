// Pending remote RED execution. This is snapshot behavior, not kernel admission.
#include "../cuda/stream1_policy_snapshot.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

static void require(bool value, const char* reason) {
    if (!value) throw std::runtime_error(reason);
}
static bool equals(const char* actual, const char* expected) {
    return actual && std::strcmp(actual, expected) == 0;
}
int main() {
    constexpr const char* selected = "BEAM_STREAM1_TRANSFORMER_FF2_POLICY";
    constexpr const char* absent = "BEAM_STREAM1_TRANSFORMER_FF1_POLICY";
    constexpr const char* empty = "BEAM_STREAM1_TRANSFORMER_QKV_POLICY";
    require(setenv(selected, "m128n128", 1) == 0, "set selected fixture");
    require(unsetenv(absent) == 0, "unset absent fixture");
    require(setenv(empty, "", 1) == 0, "set empty fixture");
    const auto snapshot = beam::Stream1PolicySnapshot::capture({selected, absent, empty});
    require(setenv(selected, "m64n64", 1) == 0, "mutate selected fixture");
    require(setenv(absent, "m128n64", 1) == 0, "introduce previously absent value");
    require(setenv(empty, "m64n64", 1) == 0, "mutate empty fixture");
    require(equals(snapshot.get(selected), "m128n128"), "snapshot reread changed environment");
    require(snapshot.get(absent) == nullptr, "absent must stay absent, not inherit later value");
    require(equals(snapshot.get(empty), ""), "empty must remain distinct from absent");
    {
        beam::Stream1PolicyScope scope(snapshot);
        require(equals(beam::stream1_policy_value(selected), "m128n128"), "launch read ignored frozen value");
        require(beam::stream1_policy_value(absent) == nullptr, "launch inherited absent value");
        bool rejected = false;
        try { (void)beam::stream1_policy_value("BEAM_STREAM1_TRANSFORMER_UNBOUND_POLICY"); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "unbound launch policy silently fell back to getenv");
        const auto nested = beam::Stream1PolicySnapshot::capture({selected, absent, empty});
        struct ExpectedUnwind {};
        try {
            beam::Stream1PolicyScope nested_scope(nested);
            require(equals(beam::stream1_policy_value(selected), "m64n64"), "nested snapshot not selected");
            throw ExpectedUnwind{};
        } catch (const ExpectedUnwind&) {}
        require(equals(beam::stream1_policy_value(selected), "m128n128"), "unwind did not restore outer scope");
    }
    require(equals(beam::stream1_policy_value(selected), "m64n64"), "scope outlived its execution");
    return 0;
}
