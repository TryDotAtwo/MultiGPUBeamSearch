#pragma once

#include "stream1_transformer_sm120_nvfp4_ffn.hpp"

#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

struct Stream1Sm120Nvfp4PackedWeight {
    std::vector<std::uint8_t> values;
    std::vector<std::uint8_t> scales;
};

namespace stream1_sm120_nvfp4_pack_detail {

using Scale = cutlass::float_ue4m3_t;
using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;

inline void validate_shape(
    std::size_t source_elements,
    std::uint32_t input_cols,
    std::uint32_t output_cols) {
    if (input_cols == 0U || output_cols == 0U || input_cols % 16U != 0U ||
        output_cols % 128U != 0U) {
        throw std::invalid_argument(
            "SM120 NVFP4 weight shape requires positive K16 and N128 dimensions");
    }
    if (source_elements != static_cast<std::size_t>(input_cols) * output_cols) {
        throw std::invalid_argument("SM120 NVFP4 source weight element count mismatch");
    }
}

inline auto scale_layout(std::uint32_t input_cols, std::uint32_t output_cols) {
    return ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(
        1, static_cast<int>(output_cols), static_cast<int>(input_cols), 1));
}

inline std::uint8_t e2m1_raw(float value) {
    return static_cast<std::uint8_t>(cutlass::float_e2m1_t(value).raw() & 0x0fU);
}

inline float e2m1_value(std::uint8_t raw) {
    return static_cast<float>(cutlass::float_e2m1_t::bitcast(raw & 0x0fU));
}

}  // namespace stream1_sm120_nvfp4_pack_detail

inline Stream1Sm120Nvfp4PackedWeight stream1_sm120_nvfp4_pack_weight_host(
    const std::vector<float>& source,
    std::uint32_t input_cols,
    std::uint32_t output_cols) {
    using namespace stream1_sm120_nvfp4_pack_detail;
    validate_shape(source.size(), input_cols, output_cols);
    Stream1Sm120Nvfp4PackedWeight packed;
    packed.values.assign(
        stream1_sm120_nvfp4_weight_value_bytes(input_cols, output_cols), 0U);
    packed.scales.assign(
        stream1_sm120_nvfp4_weight_scale_bytes(input_cols, output_cols), 0U);
    const auto layout = scale_layout(input_cols, output_cols);
    auto scale_tensor = cute::make_tensor(
        reinterpret_cast<Scale*>(packed.scales.data()), layout);

    for (std::uint32_t n = 0; n < output_cols; ++n) {
        for (std::uint32_t k_begin = 0; k_begin < input_cols; k_begin += 16U) {
            float max_abs = 0.0F;
            for (std::uint32_t k = k_begin; k < k_begin + 16U; ++k) {
                max_abs = std::max(
                    max_abs,
                    std::abs(source[static_cast<std::size_t>(n) * input_cols + k]));
            }
            const Scale encoded_scale(max_abs == 0.0F ? 1.0F : max_abs / 6.0F);
            const float scale = static_cast<float>(encoded_scale);
            if (!(scale > 0.0F) || !std::isfinite(scale)) {
                throw std::runtime_error("SM120 NVFP4 UE4M3 scale is not finite and positive");
            }
            scale_tensor(static_cast<int>(n), static_cast<int>(k_begin), 0) = encoded_scale;
            for (std::uint32_t k = k_begin; k < k_begin + 16U; ++k) {
                // Transformer artifacts are logical HxK (N x K). CUTLASS
                // receives B as logical KxN with ColumnMajor storage, whose
                // physical offset is k + n*K == n*K + k. SFB keeps its own
                // CUTLASS tiled layout below.
                const std::size_t hxk = static_cast<std::size_t>(n) * input_cols + k;
                const std::size_t physical = hxk;
                const std::uint8_t raw = e2m1_raw(source[hxk] / scale);
                const unsigned shift = static_cast<unsigned>((physical & 1U) * 4U);
                packed.values[physical / 2U] = static_cast<std::uint8_t>(
                    packed.values[physical / 2U] | (raw << shift));
            }
        }
    }
    return packed;
}

inline std::vector<float> stream1_sm120_nvfp4_unpack_weight_host(
    const Stream1Sm120Nvfp4PackedWeight& packed,
    std::uint32_t input_cols,
    std::uint32_t output_cols) {
    using namespace stream1_sm120_nvfp4_pack_detail;
    validate_shape(
        static_cast<std::size_t>(input_cols) * output_cols,
        input_cols,
        output_cols);
    if (packed.values.size() !=
            stream1_sm120_nvfp4_weight_value_bytes(input_cols, output_cols) ||
        packed.scales.size() !=
            stream1_sm120_nvfp4_weight_scale_bytes(input_cols, output_cols)) {
        throw std::invalid_argument("SM120 NVFP4 packed artifact byte count mismatch");
    }
    const auto layout = scale_layout(input_cols, output_cols);
    auto scale_tensor = cute::make_tensor(
        reinterpret_cast<const Scale*>(packed.scales.data()), layout);
    std::vector<float> result(static_cast<std::size_t>(input_cols) * output_cols);
    for (std::uint32_t k = 0; k < input_cols; ++k) {
        for (std::uint32_t n = 0; n < output_cols; ++n) {
            const std::size_t hxk = static_cast<std::size_t>(n) * input_cols + k;
            const std::size_t physical = hxk;
            const unsigned shift = static_cast<unsigned>((physical & 1U) * 4U);
            const std::uint8_t raw = static_cast<std::uint8_t>(
                (packed.values[physical / 2U] >> shift) & 0x0fU);
            result[hxk] = e2m1_value(raw) *
                static_cast<float>(scale_tensor(static_cast<int>(n), static_cast<int>(k), 0));
        }
    }
    return result;
}
