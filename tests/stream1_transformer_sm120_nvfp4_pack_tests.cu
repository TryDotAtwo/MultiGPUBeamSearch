#include "stream1_transformer_sm120_nvfp4_pack.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main() {
    constexpr std::uint32_t k = 16U;
    constexpr std::uint32_t n = 128U;
    const std::vector<float> source(static_cast<std::size_t>(k) * n, 1.0F);
    const auto packed = stream1_sm120_nvfp4_pack_weight_host(source, k, n);
    require(packed.values.size() == 1024U, "packed E2M1 byte count mismatch");
    require(packed.scales.size() == 512U, "tiled UE4M3 byte count mismatch");
    require(
        std::all_of(packed.values.begin(), packed.values.end(), [](std::uint8_t byte) {
            return byte == 0x77U;
        }),
        "unit weights must use positive max E2M1 code in both nibbles");

    const auto reconstructed = stream1_sm120_nvfp4_unpack_weight_host(packed, k, n);
    require(reconstructed.size() == source.size(), "round-trip shape mismatch");
    float max_abs_error = 0.0F;
    for (std::size_t index = 0; index < source.size(); ++index) {
        max_abs_error = std::max(max_abs_error, std::abs(reconstructed[index] - source[index]));
    }
    require(max_abs_error <= 0.05F, "unit-weight NVFP4 round-trip error is too large");

    const std::vector<float> zeros(source.size(), 0.0F);
    const auto packed_zeros = stream1_sm120_nvfp4_pack_weight_host(zeros, k, n);
    require(
        std::all_of(packed_zeros.values.begin(), packed_zeros.values.end(),
                    [](std::uint8_t byte) { return byte == 0U; }),
        "zero weights must have zero E2M1 payload");

    std::vector<float> hxk(source.size(), 0.0F);
    constexpr std::uint32_t output = 4U;
    constexpr std::uint32_t input = 2U;
    const std::size_t hxk_index = static_cast<std::size_t>(output) * k + input;
    const std::size_t cutlass_b_physical_index = hxk_index;
    hxk[hxk_index] = 1.0F;
    const auto packed_hxk = stream1_sm120_nvfp4_pack_weight_host(hxk, k, n);
    const auto hxk_nibble = static_cast<std::uint8_t>(
        (packed_hxk.values[cutlass_b_physical_index / 2U] >>
         ((cutlass_b_physical_index & 1U) * 4U)) & 0x0fU);
    require(
        hxk_nibble == 0x07U,
        "HxK weight must match CUTLASS KxN ColumnMajor B physical index n*K+k");
    const auto reconstructed_hxk = stream1_sm120_nvfp4_unpack_weight_host(
        packed_hxk, k, n);
    require(
        reconstructed_hxk[hxk_index] > 0.95F && reconstructed_hxk[hxk_index] < 1.05F,
        "HxK sparse weight round-trip mismatch");

    std::cout << "stream1_transformer_sm120_nvfp4_pack_tests=pass"
              << " max_abs_error=" << max_abs_error << '\n';
    return EXIT_SUCCESS;
}
