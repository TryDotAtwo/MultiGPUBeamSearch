#include "../cuda/stream1_transformer_sm120_fp8.hpp"
#include "cuda_check.hpp"
#include "sha256.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
struct OperatorSpec {
    std::string name;
    fs::path source_file;
    std::uint32_t input_cols;
    std::uint32_t output_cols;
};

static std::vector<std::string> split(const std::string& text) {
    std::vector<std::string> result;
    std::size_t begin = 0U;
    while (begin <= text.size()) {
        const std::size_t comma = text.find(',', begin);
        const std::string value = text.substr(begin, comma == std::string::npos ? comma : comma - begin);
        if (value.empty()) throw std::invalid_argument("empty operator in --operators");
        result.push_back(value);
        if (comma == std::string::npos) break;
        begin = comma + 1U;
    }
    return result;
}

static OperatorSpec resolve(const fs::path& weight_dir, const std::string& name) {
    for (std::uint32_t layer = 0; layer < 4U; ++layer) {
        const std::string prefix = "blocks." + std::to_string(layer) + ".";
        if (name == prefix + "attn.in_proj_weight") {
            return {name, weight_dir / ("block" + std::to_string(layer) + "_attn_qkv_weight_hxk.fp32"), 256U, 768U};
        }
        if (name == prefix + "ff.0.weight") {
            return {name, weight_dir / ("block" + std::to_string(layer) + "_ff1_weight_hxk.fp32"), 256U, 1024U};
        }
    }
    throw std::invalid_argument("offline encoder does not support operator: " + name);
}

static std::vector<std::byte> read_exact(const fs::path& path, std::size_t bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open FP32 source weight: " + path.string());
    std::vector<std::byte> data(bytes);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
    if (static_cast<std::size_t>(in.gcount()) != bytes || in.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("FP32 source weight size mismatch: " + path.string());
    }
    return data;
}

static void write_bytes(const fs::path& path, const void* data, std::size_t bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create immutable artifact: " + path.string());
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!out) throw std::runtime_error("cannot write immutable artifact: " + path.string());
}

static std::string slug(const std::string& name) {
    std::string value = name;
    std::replace(value.begin(), value.end(), '.', '_');
    return value;
}

int main(int argc, char** argv) try {
    if ((argc != 7 && argc != 9) || std::string(argv[1]) != "--weight-dir" ||
        std::string(argv[3]) != "--output-dir" || std::string(argv[5]) != "--operators" ||
        (argc == 9 && std::string(argv[7]) != "--weight-scale-policy")) {
        throw std::invalid_argument(
            "usage: exporter --weight-dir DIR --output-dir NEW_DIR --operators CSV "
            "[--weight-scale-policy max_abs|mse_grid]");
    }
    const fs::path weight_dir(argv[2]);
    const fs::path output_dir(argv[4]);
    const std::string operator_csv(argv[6]);
    const std::string weight_scale_policy = argc == 9 ? argv[8] : "max_abs";
    if (weight_scale_policy != "max_abs" && weight_scale_policy != "mse_grid") {
        throw std::invalid_argument("weight scale policy must be max_abs or mse_grid");
    }
    if (fs::exists(output_dir)) throw std::runtime_error("output directory already exists: " + output_dir.string());
    if (!fs::is_regular_file(weight_dir / "manifest.json")) throw std::runtime_error("missing source manifest.json");
    if (!stream1_transformer_sm120_fp8_supported()) {
        throw std::runtime_error("offline encoder requires a physical SM120 GPU and sm_120a build");
    }
    const auto names = split(operator_csv);
    if (names.empty()) throw std::invalid_argument("offline encoder requires at least one operator");
    fs::create_directories(output_dir / "weights");
    std::ostringstream manifest;
    manifest << "schema_version=2\n";
    manifest << "operators=" << operator_csv << "\n";
    manifest << "weight_scale_policy=" << weight_scale_policy << "\n";
    manifest << "encoding_source_manifest_sha256="
             << beam::sha256::file_hex(weight_dir / "manifest.json") << "\n";
    for (const std::string& name : names) {
        const OperatorSpec spec = resolve(weight_dir, name);
        const std::size_t elements = static_cast<std::size_t>(spec.input_cols) * spec.output_cols;
        const std::size_t scale_elements = stream1_transformer_sm120_fp8_weight_scale_elements(
            spec.input_cols, spec.output_cols);
        const auto source = read_exact(spec.source_file, elements * sizeof(float));
        float* d_source = nullptr;
        std::uint8_t* d_quantized = nullptr;
        float* d_scales = nullptr;
        BEAM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_source), source.size()));
        BEAM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_quantized), elements));
        BEAM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scales), scale_elements * sizeof(float)));
        BEAM_CUDA_CHECK(cudaMemcpy(d_source, source.data(), source.size(), cudaMemcpyHostToDevice));
        if (weight_scale_policy == "mse_grid") {
            stream1_transformer_sm120_fp8_quantize_weight_mse_from_fp32_cuda(
                d_source, d_quantized, d_scales, spec.input_cols, spec.output_cols, nullptr);
        } else {
            stream1_transformer_sm120_fp8_quantize_weight_from_fp32_cuda(
                d_source, d_quantized, d_scales, spec.input_cols, spec.output_cols, 1.0f, nullptr);
        }
        std::vector<std::uint8_t> quantized(elements);
        std::vector<float> scales(scale_elements);
        BEAM_CUDA_CHECK(cudaMemcpy(quantized.data(), d_quantized, elements, cudaMemcpyDeviceToHost));
        BEAM_CUDA_CHECK(cudaMemcpy(scales.data(), d_scales, scale_elements * sizeof(float), cudaMemcpyDeviceToHost));
        BEAM_CUDA_CHECK(cudaFree(d_scales));
        BEAM_CUDA_CHECK(cudaFree(d_quantized));
        BEAM_CUDA_CHECK(cudaFree(d_source));
        const std::string base = slug(name);
        const fs::path weight_relative = fs::path("weights") / (base + ".e4m3");
        const fs::path scale_relative = fs::path("weights") / (base + ".scales.fp32");
        write_bytes(output_dir / weight_relative, quantized.data(), quantized.size());
        write_bytes(output_dir / scale_relative, scales.data(), scales.size() * sizeof(float));
        const std::string prefix = "operator." + name + ".";
        manifest << prefix << "input_cols=" << spec.input_cols << "\n";
        manifest << prefix << "output_cols=" << spec.output_cols << "\n";
        manifest << prefix << "source_fp32_sha256=" << beam::sha256::file_hex(spec.source_file) << "\n";
        manifest << prefix << "weight_file=" << weight_relative.generic_string() << "\n";
        manifest << prefix << "weight_sha256=" << beam::sha256::file_hex(output_dir / weight_relative) << "\n";
        manifest << prefix << "scale_file=" << scale_relative.generic_string() << "\n";
        manifest << prefix << "scale_sha256=" << beam::sha256::file_hex(output_dir / scale_relative) << "\n";
    }
    const std::string runtime_manifest = manifest.str();
    write_bytes(output_dir / "runtime_manifest.txt", runtime_manifest.data(), runtime_manifest.size());
    std::cout << "sm120_offline_weight_export_done output_dir=" << output_dir
              << " operators=" << names.size()
              << " weight_scale_policy=" << weight_scale_policy << "\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "sm120_offline_weight_export_error=" << error.what() << "\n";
    return 1;
}
