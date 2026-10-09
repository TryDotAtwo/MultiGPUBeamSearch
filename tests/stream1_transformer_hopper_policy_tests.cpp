#include "stream1_transformer_hopper_policy.hpp"

#include <stdexcept>

int main() {
    using beam::Stream1TransformerHopperMode;
    const auto require = [](bool condition) {
        if (!condition) throw std::runtime_error("Hopper Stream1 policy contract failed");
    };
    require(!beam::resolve_stream1_packed_fp16_hopper(false, false, 75, false));
    require(beam::resolve_stream1_packed_fp16_hopper(true, true, 90, true));
    for (unsigned invalid = 0; invalid < 3; ++invalid) {
        bool rejected_route = false;
        try { (void)beam::resolve_stream1_packed_fp16_hopper(
            true, invalid != 0, invalid == 1 ? 86 : 90, invalid != 2); }
        catch (const std::invalid_argument&) { rejected_route = true; }
        require(rejected_route);
    }

    require(beam::parse_stream1_transformer_hopper_mode(nullptr) == Stream1TransformerHopperMode::Off);
    require(beam::parse_stream1_transformer_hopper_mode("") == Stream1TransformerHopperMode::Off);
    require(beam::parse_stream1_transformer_hopper_mode("fp16_tma") == Stream1TransformerHopperMode::Fp16Tma);
    require(beam::parse_stream1_transformer_hopper_mode("fp8_e4m3") == Stream1TransformerHopperMode::Fp8E4m3);
    require(beam::select_stream1_transformer_hopper_mode("fp16_tma", nullptr) == Stream1TransformerHopperMode::Fp16Tma);
    require(beam::select_stream1_transformer_hopper_mode("fp16_tma", "off") == Stream1TransformerHopperMode::Off);
    // The H200 launcher sets global off but explicit per-family FP16 overrides.
    const auto h200_family = beam::select_stream1_transformer_hopper_mode("off", "fp16_tma");
    require(h200_family == Stream1TransformerHopperMode::Fp16Tma);
    require(beam::select_stream1_transformer_hopper_mode("off", "") == Stream1TransformerHopperMode::Off);
    require(!beam::stream1_transformer_hopper_large_gemm_allowed(h200_family, 90, 4095));
    require(beam::stream1_transformer_hopper_large_gemm_allowed(h200_family, 90, 4096));
    require(!beam::stream1_transformer_hopper_large_gemm_allowed(h200_family, 86, 4096));

    bool rejected = false;
    try { (void)beam::parse_stream1_transformer_hopper_mode("mxfp4"); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
    require(!beam::stream1_transformer_hopper_epilogue_128x64(nullptr));
    require(!beam::stream1_transformer_hopper_epilogue_128x64("auto"));
    require(beam::stream1_transformer_hopper_epilogue_128x64("128x64"));
    rejected = false;
    try { (void)beam::stream1_transformer_hopper_epilogue_128x64("64x64"); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);

    require(!beam::stream1_transformer_hopper_large_gemm_allowed(Stream1TransformerHopperMode::Off, 90, 43776));
    require(!beam::stream1_transformer_hopper_large_gemm_allowed(Stream1TransformerHopperMode::Fp16Tma, 89, 43776));
    require(beam::stream1_transformer_hopper_large_gemm_allowed(Stream1TransformerHopperMode::Fp16Tma, 90, 43776));
    require(beam::stream1_transformer_hopper_large_gemm_allowed(Stream1TransformerHopperMode::Fp8E4m3, 90, 43776));
    require(!beam::stream1_transformer_hopper_large_gemm_allowed(Stream1TransformerHopperMode::Fp8E4m3, 90, 768));
    require(!beam::stream1_transformer_hopper_large_gemm_allowed(Stream1TransformerHopperMode::Fp8E4m3, 100, 43776));
    require(beam::stream1_transformer_hopper_uses_packed_full_token_weight(
        Stream1TransformerHopperMode::Fp16Tma, 0, 4));
    require(beam::stream1_transformer_hopper_uses_packed_full_token_weight(
        Stream1TransformerHopperMode::Fp16Tma, 2, 4));
    require(!beam::stream1_transformer_hopper_uses_packed_full_token_weight(
        Stream1TransformerHopperMode::Fp16Tma, 3, 4));
    require(!beam::stream1_transformer_hopper_uses_packed_full_token_weight(
        Stream1TransformerHopperMode::Off, 0, 4));
    return 0;
}
