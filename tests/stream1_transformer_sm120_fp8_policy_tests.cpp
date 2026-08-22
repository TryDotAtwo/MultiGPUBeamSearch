#include "stream1_transformer_sm120_fp8_policy.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main() {
    if (parse_stream1_sm120_fp8_operator_policy(nullptr) != 0U) return EXIT_FAILURE;
    const std::uint32_t mask = parse_stream1_sm120_fp8_operator_policy(
        "blocks.0.attn.in_proj_weight,blocks.3.ff.0.weight");
    if ((mask & stream1_sm120_fp8_qkv_bit(0)) == 0U ||
        (mask & stream1_sm120_fp8_ff1_bit(3)) == 0U) return EXIT_FAILURE;
    bool rejected = false;
    try {
        (void)parse_stream1_sm120_fp8_operator_policy("blocks.0.ff.3.weight");
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) return EXIT_FAILURE;
    std::cout << "stream1_transformer_sm120_fp8_policy_tests=pass\n";
    return EXIT_SUCCESS;
}
