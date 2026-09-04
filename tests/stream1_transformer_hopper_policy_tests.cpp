#include "stream1_transformer_hopper_policy.hpp"

#include <stdexcept>

int main() {
    using beam::Stream1TransformerHopperMode;
    const auto require = [](bool condition) {
        if (!condition) throw std::runtime_error("Hopper Stream1 policy contract failed");
    };

    require(beam::parse_stream1_transformer_hopper_mode(nullptr) == Stream1TransformerHopperMode::Off);
    require(beam::parse_stream1_transformer_hopper_mode("") == Stream1TransformerHopperMode::Off);
    require(beam::parse_stream1_transformer_hopper_mode("fp16_tma") == Stream1TransformerHopperMode::Fp16Tma);
    require(beam::parse_stream1_transformer_hopper_mode("fp8_e4m3") == Stream1TransformerHopperMode::Fp8E4m3);
    require(beam::select_stream1_transformer_hopper_mode("fp16_tma", nullptr) == Stream1TransformerHopperMode::Fp16Tma);
    require(beam::select_stream1_transformer_hopper_mode("fp16_tma", "off") == Stream1TransformerHopperMode::Off);

    bool rejected = false;
    try { (void)beam::parse_stream1_transformer_hopper_mode("mxfp4"); }
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
