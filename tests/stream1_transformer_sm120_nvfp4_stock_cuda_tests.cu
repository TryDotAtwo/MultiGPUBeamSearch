#include "stream1_transformer_sm120_nvfp4_pack.hpp"
#include "stream1_transformer_sm120_nvfp4_stock.hpp"
#include "cuda_check.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

template <typename T>
T* copy_to_device(const std::vector<T>& host) {
    T* device = nullptr;
    BEAM_CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(T)));
    BEAM_CUDA_CHECK(cudaMemcpy(
        device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

std::uint8_t* copy_bytes_to_device(const std::vector<std::uint8_t>& host) {
    std::uint8_t* device = nullptr;
    BEAM_CUDA_CHECK(cudaMalloc(&device, host.size()));
    BEAM_CUDA_CHECK(cudaMemcpy(
        device, host.data(), host.size(), cudaMemcpyHostToDevice));
    return device;
}

std::vector<float> quantize_rowwise_nvfp4(
    const std::vector<float>& input,
    std::uint32_t rows,
    std::uint32_t columns) {
    std::vector<float> result(input.size());
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t begin = 0; begin < columns; begin += 16U) {
            float max_abs = 0.0F;
            for (std::uint32_t column = begin; column < begin + 16U; ++column) {
                max_abs = std::max(
                    max_abs,
                    std::abs(input[static_cast<std::size_t>(row) * columns + column]));
            }
            const cutlass::float_ue4m3_t encoded_scale(
                max_abs == 0.0F ? 1.0F : max_abs / 6.0F);
            const float scale = static_cast<float>(encoded_scale);
            for (std::uint32_t column = begin; column < begin + 16U; ++column) {
                const auto index = static_cast<std::size_t>(row) * columns + column;
                result[index] = static_cast<float>(
                    cutlass::float_e2m1_t(input[index] / scale)) * scale;
            }
        }
    }
    return result;
}

std::vector<float> dense_ffn_nvfp4_reference(
    const std::vector<float>& input,
    const std::vector<float>& residual,
    const std::vector<float>& ff1_weight_hxk,
    const std::vector<float>& ff1_bias,
    const std::vector<float>& ff2_weight_hxk,
    std::uint32_t rows,
    std::uint32_t d_model,
    std::uint32_t ff_dim,
    std::vector<float>* quantized_hidden_out) {
    const auto quantized_input = quantize_rowwise_nvfp4(input, rows, d_model);
    std::vector<float> hidden(static_cast<std::size_t>(rows) * ff_dim, 0.0F);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t output = 0; output < ff_dim; ++output) {
            float accumulator = ff1_bias[output];
            const auto input_base = static_cast<std::size_t>(row) * d_model;
            const auto weight_base = static_cast<std::size_t>(output) * d_model;
            for (std::uint32_t input_column = 0; input_column < d_model; ++input_column) {
                accumulator += quantized_input[input_base + input_column] *
                    ff1_weight_hxk[weight_base + input_column];
            }
            hidden[static_cast<std::size_t>(row) * ff_dim + output] =
                std::max(0.0F, accumulator);
        }
    }
    auto quantized_hidden = quantize_rowwise_nvfp4(hidden, rows, ff_dim);
    std::vector<float> result(static_cast<std::size_t>(rows) * d_model, 0.0F);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t output = 0; output < d_model; ++output) {
            float accumulator = residual[static_cast<std::size_t>(row) * d_model + output];
            const auto hidden_base = static_cast<std::size_t>(row) * ff_dim;
            const auto weight_base = static_cast<std::size_t>(output) * ff_dim;
            for (std::uint32_t hidden_column = 0; hidden_column < ff_dim; ++hidden_column) {
                accumulator += quantized_hidden[hidden_base + hidden_column] *
                    ff2_weight_hxk[weight_base + hidden_column];
            }
            result[static_cast<std::size_t>(row) * d_model + output] = accumulator;
        }
    }
    if (quantized_hidden_out != nullptr) {
        *quantized_hidden_out = std::move(quantized_hidden);
    }
    return result;
}

}  // namespace

int main() try {
    cudaDeviceProp properties{};
    BEAM_CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    if (properties.major != 12 || properties.minor != 0) {
        std::cerr << "stream1_transformer_sm120_nvfp4_stock_cuda_tests requires sm_120\n";
        return EXIT_FAILURE;
    }
    constexpr std::uint32_t rows = 128U;
    constexpr std::uint32_t d_model = 256U;
    constexpr std::uint32_t ff_dim = 1024U;
    std::vector<half> input(static_cast<std::size_t>(rows) * d_model);
    std::vector<half> residual(input.size(), __float2half(0.0F));
    std::vector<half> bias1(ff_dim, __float2half(0.0F));
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t column = 0; column < d_model; ++column) {
            const float sign = ((row + column) & 1U) ? -1.0F : 1.0F;
            const float magnitude = 0.5F * static_cast<float>(column % 7U);
            input[static_cast<std::size_t>(row) * d_model + column] =
                __float2half(sign * magnitude);
        }
    }
    std::vector<float> ff1(static_cast<std::size_t>(d_model) * ff_dim, 0.0F);
    std::vector<float> ff2(static_cast<std::size_t>(ff_dim) * d_model, 0.0F);
    for (std::uint32_t index = 0; index < d_model; ++index) {
        ff1[static_cast<std::size_t>(index) * d_model + index] = 1.0F;
        ff2[static_cast<std::size_t>(index) * ff_dim + index] = 1.0F;
    }
    const auto packed_ff1 = stream1_sm120_nvfp4_pack_weight_host(
        ff1, d_model, ff_dim);
    const auto packed_ff2 = stream1_sm120_nvfp4_pack_weight_host(
        ff2, ff_dim, d_model);
    const auto effective_ff1 = stream1_sm120_nvfp4_unpack_weight_host(
        packed_ff1, d_model, ff_dim);
    const auto effective_ff2 = stream1_sm120_nvfp4_unpack_weight_host(
        packed_ff2, ff_dim, d_model);
    half* device_input = copy_to_device(input);
    half* device_residual = copy_to_device(residual);
    half* device_bias1 = copy_to_device(bias1);
    auto* device_ff1 = copy_bytes_to_device(packed_ff1.values);
    auto* device_ff1_scales = copy_bytes_to_device(packed_ff1.scales);
    auto* device_ff2 = copy_bytes_to_device(packed_ff2.values);
    auto* device_ff2_scales = copy_bytes_to_device(packed_ff2.scales);
    void* input_scratch = nullptr;
    void* hidden_scratch = nullptr;
    const auto plan = stream1_sm120_nvfp4_stock_scratch_plan(rows);
    BEAM_CUDA_CHECK(cudaMalloc(&input_scratch, plan.input_total_bytes));
    BEAM_CUDA_CHECK(cudaMalloc(&hidden_scratch, plan.hidden_total_bytes));

    stream1_transformer_sm120_nvfp4_stock_ffn_cuda(
        device_input,
        device_residual,
        device_ff1,
        device_ff1_scales,
        device_bias1,
        device_ff2,
        device_ff2_scales,
        device_residual,
        input_scratch,
        plan.input_total_bytes,
        hidden_scratch,
        plan.hidden_total_bytes,
        rows,
        nullptr);
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<half> output(residual.size());
    std::vector<std::uint8_t> hidden_bytes(plan.hidden_total_bytes);
    BEAM_CUDA_CHECK(cudaMemcpy(
        output.data(), device_residual, output.size() * sizeof(half), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(
        hidden_bytes.data(), hidden_scratch, hidden_bytes.size(), cudaMemcpyDeviceToHost));
    using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;
    using OutputScaleConfig = cutlass::detail::Sm1xxBlockScaledOutputConfig<16>;
    const auto ff1_shape = cute::make_shape(
        static_cast<int>(rows), static_cast<int>(ff_dim),
        static_cast<int>(d_model), 1);
    const auto ff2_shape = cute::make_shape(
        static_cast<int>(rows), static_cast<int>(d_model),
        static_cast<int>(ff_dim), 1);
    const auto layout_sfd = OutputScaleConfig::tile_atom_to_shape_SFD(ff1_shape);
    const auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(ff2_shape);
    const auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(ff2_shape);
    auto* hidden_scale_data = reinterpret_cast<const cutlass::float_ue4m3_t*>(
        hidden_bytes.data() + plan.hidden_scale_offset);
    auto hidden_sfd = cute::make_tensor(hidden_scale_data, layout_sfd);
    auto hidden_sfa = cute::make_tensor(hidden_scale_data, layout_sfa);
    auto ff2_sfb = cute::make_tensor(
        reinterpret_cast<const cutlass::float_ue4m3_t*>(packed_ff2.scales.data()),
        layout_sfb);
    std::cout << "hidden_scale_layout_probe";
    for (std::uint32_t column = 0; column < 16U; column += 2U) {
        const auto sfd_offset = static_cast<std::size_t>(layout_sfd(0, column, 0));
        const auto sfa_offset = static_cast<std::size_t>(layout_sfa(0, column, 0));
        const std::size_t logical = column;
        const auto raw = static_cast<std::uint8_t>(
            (hidden_bytes[logical / 2U] >> ((logical & 1U) * 4U)) & 0x0fU);
        std::cout << " c" << column
                  << "=v" << static_cast<int>(raw)
                  << ",sfd@" << sfd_offset << ':' << static_cast<int>(hidden_scale_data[sfd_offset].raw())
                  << ",sfa@" << sfa_offset << ':' << static_cast<int>(hidden_scale_data[sfa_offset].raw())
                  << ",b@" << static_cast<std::size_t>(layout_sfb(column, column, 0))
                  << ':' << static_cast<int>(ff2_sfb(column, column, 0).raw());
    }
    std::cout << '\n';
    std::vector<float> input_fp32(input.size());
    std::transform(input.begin(), input.end(), input_fp32.begin(), [](half value) {
        return __half2float(value);
    });
    const auto quantized_input = quantize_rowwise_nvfp4(input_fp32, rows, d_model);
    std::vector<float> hidden(static_cast<std::size_t>(rows) * ff_dim, 0.0F);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t column = 0; column < d_model; ++column) {
            hidden[static_cast<std::size_t>(row) * ff_dim + column] = std::max(
                0.0F,
                quantized_input[static_cast<std::size_t>(row) * d_model + column] *
                    effective_ff1[static_cast<std::size_t>(column) * d_model + column]);
        }
    }
    const auto quantized_hidden = quantize_rowwise_nvfp4(hidden, rows, ff_dim);
    std::vector<float> nvfp4_reference(output.size(), 0.0F);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t column = 0; column < d_model; ++column) {
            nvfp4_reference[static_cast<std::size_t>(row) * d_model + column] =
                quantized_hidden[static_cast<std::size_t>(row) * ff_dim + column] *
                effective_ff2[static_cast<std::size_t>(column) * ff_dim + column];
        }
    }
    float max_abs_error = 0.0F;
    float max_fp32_error = 0.0F;
    std::size_t wrong = 0U;
    for (std::size_t index = 0; index < output.size(); ++index) {
        const float observed = __half2float(output[index]);
        const float error = std::abs(observed - nvfp4_reference[index]);
        max_abs_error = std::max(max_abs_error, error);
        max_fp32_error = std::max(
            max_fp32_error,
            std::abs(observed - std::max(0.0F, __half2float(input[index]))));
        wrong += error > 0.002F;
    }
    std::cout << "stream1_transformer_sm120_nvfp4_stock_identity"
              << " max_abs_error=" << max_abs_error
              << " max_fp32_error=" << max_fp32_error
              << " wrong=" << wrong
              << " values=";
    for (std::uint32_t column = 0; column < 16U; ++column) {
        if (column) std::cout << ',';
        std::cout << __half2float(output[column]);
    }
    std::cout << '\n';
    if (wrong != 0U || max_fp32_error > 0.5F) {
        return EXIT_FAILURE;
    }

    // A dense, non-symmetric fixture is required here.  The identity fixture
    // above proves the basic path, but cannot expose a transposed HxK weight,
    // a misplaced per-column bias, or a broken residual epilogue.
    std::vector<half> dense_input(input.size());
    std::vector<half> dense_residual(residual.size());
    std::vector<half> dense_bias1(bias1.size());
    std::vector<float> dense_input_fp32(input.size());
    std::vector<float> dense_residual_fp32(residual.size());
    std::vector<float> dense_bias1_fp32(bias1.size());
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t column = 0; column < d_model; ++column) {
            const int input_code = static_cast<int>((row * 17U + column * 13U) % 29U) - 14;
            const int residual_code = static_cast<int>((row * 7U + column * 11U) % 17U) - 8;
            const auto index = static_cast<std::size_t>(row) * d_model + column;
            dense_input_fp32[index] = static_cast<float>(input_code) * 0.0625F;
            dense_residual_fp32[index] = static_cast<float>(residual_code) * 0.03125F;
            dense_input[index] = __float2half(dense_input_fp32[index]);
            dense_residual[index] = __float2half(dense_residual_fp32[index]);
        }
    }
    for (std::uint32_t output_column = 0; output_column < ff_dim; ++output_column) {
        const int bias_code = static_cast<int>((output_column * 5U) % 13U) - 6;
        dense_bias1_fp32[output_column] = static_cast<float>(bias_code) * 0.015625F;
        dense_bias1[output_column] = __float2half(dense_bias1_fp32[output_column]);
    }
    std::vector<float> dense_ff1(static_cast<std::size_t>(ff_dim) * d_model);
    std::vector<float> dense_ff2(static_cast<std::size_t>(d_model) * ff_dim);
    for (std::uint32_t output_column = 0; output_column < ff_dim; ++output_column) {
        for (std::uint32_t input_column = 0; input_column < d_model; ++input_column) {
            const int code = static_cast<int>(
                (output_column * 7U + input_column * 11U) % 31U) - 15;
            dense_ff1[static_cast<std::size_t>(output_column) * d_model + input_column] =
                static_cast<float>(code) * 0.00390625F;
        }
    }
    for (std::uint32_t output_column = 0; output_column < d_model; ++output_column) {
        for (std::uint32_t hidden_column = 0; hidden_column < ff_dim; ++hidden_column) {
            const int code = static_cast<int>(
                (output_column * 13U + hidden_column * 3U) % 37U) - 18;
            dense_ff2[static_cast<std::size_t>(output_column) * ff_dim + hidden_column] =
                static_cast<float>(code) * 0.001953125F;
        }
    }
    const auto dense_packed_ff1 = stream1_sm120_nvfp4_pack_weight_host(
        dense_ff1, d_model, ff_dim);
    const auto dense_packed_ff2 = stream1_sm120_nvfp4_pack_weight_host(
        dense_ff2, ff_dim, d_model);
    const auto dense_effective_ff1 = stream1_sm120_nvfp4_unpack_weight_host(
        dense_packed_ff1, d_model, ff_dim);
    const auto dense_effective_ff2 = stream1_sm120_nvfp4_unpack_weight_host(
        dense_packed_ff2, ff_dim, d_model);
    std::vector<float> dense_quantized_hidden;
    const auto dense_reference = dense_ffn_nvfp4_reference(
        dense_input_fp32,
        dense_residual_fp32,
        dense_effective_ff1,
        dense_bias1_fp32,
        dense_effective_ff2,
        rows,
        d_model,
        ff_dim,
        &dense_quantized_hidden);

    half* dense_device_input = copy_to_device(dense_input);
    half* dense_device_residual = copy_to_device(dense_residual);
    half* dense_device_bias1 = copy_to_device(dense_bias1);
    auto* dense_device_ff1 = copy_bytes_to_device(dense_packed_ff1.values);
    auto* dense_device_ff1_scales = copy_bytes_to_device(dense_packed_ff1.scales);
    auto* dense_device_ff2 = copy_bytes_to_device(dense_packed_ff2.values);
    auto* dense_device_ff2_scales = copy_bytes_to_device(dense_packed_ff2.scales);
    void* dense_input_scratch = nullptr;
    void* dense_hidden_scratch = nullptr;
    BEAM_CUDA_CHECK(cudaMalloc(&dense_input_scratch, plan.input_total_bytes));
    BEAM_CUDA_CHECK(cudaMalloc(&dense_hidden_scratch, plan.hidden_total_bytes));
    stream1_transformer_sm120_nvfp4_stock_ffn_cuda(
        dense_device_input,
        dense_device_residual,
        dense_device_ff1,
        dense_device_ff1_scales,
        dense_device_bias1,
        dense_device_ff2,
        dense_device_ff2_scales,
        dense_device_residual,
        dense_input_scratch,
        plan.input_total_bytes,
        dense_hidden_scratch,
        plan.hidden_total_bytes,
        rows,
        nullptr);
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<half> dense_output(dense_residual.size());
    std::vector<std::uint8_t> dense_hidden_bytes(plan.hidden_total_bytes);
    BEAM_CUDA_CHECK(cudaMemcpy(
        dense_output.data(), dense_device_residual,
        dense_output.size() * sizeof(half), cudaMemcpyDeviceToHost));
    BEAM_CUDA_CHECK(cudaMemcpy(
        dense_hidden_bytes.data(), dense_hidden_scratch,
        dense_hidden_bytes.size(), cudaMemcpyDeviceToHost));

    auto* dense_hidden_scale_data = reinterpret_cast<const cutlass::float_ue4m3_t*>(
        dense_hidden_bytes.data() + plan.hidden_scale_offset);
    auto dense_hidden_sfd = cute::make_tensor(dense_hidden_scale_data, layout_sfd);
    float dense_hidden_max_abs_error = 0.0F;
    std::size_t dense_hidden_wrong = 0U;
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t column = 0; column < ff_dim; ++column) {
            const auto logical = static_cast<std::size_t>(row) * ff_dim + column;
            const unsigned shift = static_cast<unsigned>((logical & 1U) * 4U);
            const std::uint8_t raw = static_cast<std::uint8_t>(
                (dense_hidden_bytes[logical / 2U] >> shift) & 0x0fU);
            const float observed = static_cast<float>(
                cutlass::float_e2m1_t::bitcast(raw)) *
                static_cast<float>(dense_hidden_sfd(
                    static_cast<int>(row), static_cast<int>(column), 0));
            const float error = std::abs(observed - dense_quantized_hidden[logical]);
            dense_hidden_max_abs_error = std::max(dense_hidden_max_abs_error, error);
            dense_hidden_wrong += error > 0.0F;
        }
    }
    float dense_output_max_abs_error = 0.0F;
    std::size_t dense_output_wrong = 0U;
    for (std::size_t index = 0; index < dense_output.size(); ++index) {
        const float observed = __half2float(dense_output[index]);
        const float expected = __half2float(__float2half(dense_reference[index]));
        const float error = std::abs(observed - expected);
        dense_output_max_abs_error = std::max(dense_output_max_abs_error, error);
        dense_output_wrong += error > 0.0625F;
    }
    std::cout << "stream1_transformer_sm120_nvfp4_stock_dense"
              << " hidden_max_abs_error=" << dense_hidden_max_abs_error
              << " hidden_wrong=" << dense_hidden_wrong
              << " output_max_abs_error=" << dense_output_max_abs_error
              << " output_wrong=" << dense_output_wrong
              << '\n';
    if (dense_hidden_wrong != 0U || dense_output_wrong != 0U) {
        return EXIT_FAILURE;
    }
    std::cout << "stream1_transformer_sm120_nvfp4_stock_cuda_tests=pass\n";
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << "stream1_transformer_sm120_nvfp4_stock_cuda_tests_error="
              << error.what() << '\n';
    return EXIT_FAILURE;
}
