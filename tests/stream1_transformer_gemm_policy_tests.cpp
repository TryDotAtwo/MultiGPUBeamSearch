#include "stream1_transformer_gemm_policy.hpp"

#include <iostream>
#include <stdexcept>
#include <string_view>

using namespace beam;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main() {
    require(
        parse_stream1_transformer_ff1_policy(nullptr) == Stream1TransformerFf1Policy::Baseline,
        "unset FF1 policy must select baseline");
    require(
        parse_stream1_transformer_ff1_policy("") == Stream1TransformerFf1Policy::Baseline,
        "empty FF1 policy must select baseline");
    require(
        parse_stream1_transformer_ff1_policy("baseline") == Stream1TransformerFf1Policy::Baseline,
        "baseline FF1 policy parse failed");
    require(
        parse_stream1_transformer_ff1_policy("m64n128") == Stream1TransformerFf1Policy::M64N128,
        "m64n128 FF1 policy parse failed");
    require(
        parse_stream1_transformer_ff1_policy("m64n64") == Stream1TransformerFf1Policy::M64N64,
        "m64n64 FF1 policy parse failed");

    require(
        std::string_view(stream1_transformer_ff1_policy_name(Stream1TransformerFf1Policy::Baseline)) == "baseline",
        "baseline FF1 policy name failed");
    require(
        std::string_view(stream1_transformer_ff1_policy_name(Stream1TransformerFf1Policy::M64N128)) == "m64n128",
        "m64n128 FF1 policy name failed");
    require(
        std::string_view(stream1_transformer_ff1_policy_name(Stream1TransformerFf1Policy::M64N64)) == "m64n64",
        "m64n64 FF1 policy name failed");

    bool invalid_rejected = false;
    try {
        static_cast<void>(parse_stream1_transformer_ff1_policy("invalid-policy"));
    } catch (const std::invalid_argument&) {
        invalid_rejected = true;
    }
    require(invalid_rejected, "unknown FF1 policy must fail closed");

    std::cout << "stream1_transformer_gemm_policy_tests=pass\n";
    return 0;
}
