#include "../cuda/stream1_transformer_sm120_nvfp4_pack.hpp"
#include "sha256.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::vector<std::uint32_t> parse_layers(const std::string& csv) {
    std::vector<std::uint32_t> result;
    std::size_t begin = 0U;
    while (begin <= csv.size()) {
        const std::size_t comma = csv.find(',', begin);
        const std::string item = csv.substr(
            begin, comma == std::string::npos ? comma : comma - begin);
        if (item.empty()) {
            throw std::invalid_argument("empty layer in --layers");
        }
        const unsigned long layer = std::stoul(item);
        if (layer > 2UL) {
            throw std::invalid_argument(
                "fused NVFP4 next-layer packing supports only non-final layers 0..2");
        }
        result.push_back(static_cast<std::uint32_t>(layer));
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1U;
    }
    std::sort(result.begin(), result.end());
    if (std::adjacent_find(result.begin(), result.end()) != result.end()) {
        throw std::invalid_argument("duplicate layer in --layers");
    }
    return result;
}

struct OperatorSpec {
    std::string name;
    fs::path source;
    std::uint32_t input_cols;
    std::uint32_t logical_output_cols;
    std::uint32_t physical_output_cols;
};

std::vector<OperatorSpec> full_stream1_operators(const fs::path& weight_dir) {
    std::vector<OperatorSpec> result;
    result.reserve(17U);
    for (std::uint32_t layer = 0U; layer < 4U; ++layer) {
        const std::string prefix = "block" + std::to_string(layer);
        result.push_back({
            prefix + "_attn_qkv", weight_dir / (prefix + "_attn_qkv_weight_hxk.fp32"),
            256U, 768U, 768U});
        result.push_back({
            prefix + "_attn_out", weight_dir / (prefix + "_attn_out_weight_hxk.fp32"),
            256U, 256U, 256U});
        result.push_back({
            prefix + "_ff1", weight_dir / (prefix + "_ff1_weight_hxk.fp32"),
            256U, 1024U, 1024U});
        result.push_back({
            prefix + "_ff2", weight_dir / (prefix + "_ff2_weight_hxk.fp32"),
            1024U, 256U, 256U});
    }
    result.push_back({
        "output_head", weight_dir / "output_weight_hxk.fp32",
        256U, 24U, 128U});
    return result;
}

std::vector<float> read_floats(const fs::path& path, std::size_t count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open FP32 weight: " + path.string());
    }
    std::vector<float> values(count);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(count * sizeof(float)));
    if (static_cast<std::size_t>(input.gcount()) != count * sizeof(float) ||
        input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("FP32 weight size mismatch: " + path.string());
    }
    return values;
}

void write_bytes(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create NVFP4 artifact: " + path.string());
    }
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error("cannot write NVFP4 artifact: " + path.string());
    }
}

float max_abs_error(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("NVFP4 round-trip vectors have different sizes");
    }
    float result = 0.0F;
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        result = std::max(result, std::abs(lhs[index] - rhs[index]));
    }
    return result;
}

void export_weight(
    const fs::path& source,
    const fs::path& output_dir,
    const std::string& base,
    std::uint32_t input_cols,
    std::uint32_t logical_output_cols,
    std::uint32_t physical_output_cols,
    std::ostringstream& manifest) {
    const auto logical_fp32 = read_floats(
        source, static_cast<std::size_t>(input_cols) * logical_output_cols);
    std::vector<float> physical_fp32(
        static_cast<std::size_t>(input_cols) * physical_output_cols, 0.0F);
    std::copy(logical_fp32.begin(), logical_fp32.end(), physical_fp32.begin());
    const auto packed = stream1_sm120_nvfp4_pack_weight_host(
        physical_fp32, input_cols, physical_output_cols);
    const auto reconstructed = stream1_sm120_nvfp4_unpack_weight_host(
        packed, input_cols, physical_output_cols);
    const fs::path values_relative = fs::path("weights") / (base + ".e2m1");
    const fs::path scales_relative = fs::path("weights") / (base + ".ue4m3_sfb");
    write_bytes(output_dir / values_relative, packed.values);
    write_bytes(output_dir / scales_relative, packed.scales);
    manifest << "operator." << base << ".input_cols=" << input_cols << '\n'
             << "operator." << base << ".logical_output_cols=" << logical_output_cols << '\n'
             << "operator." << base << ".physical_output_cols=" << physical_output_cols << '\n'
             << "operator." << base << ".source_fp32_sha256="
             << beam::sha256::file_hex(source) << '\n'
             << "operator." << base << ".value_file=" << values_relative.generic_string() << '\n'
             << "operator." << base << ".value_sha256="
             << beam::sha256::file_hex(output_dir / values_relative) << '\n'
             << "operator." << base << ".scale_file=" << scales_relative.generic_string() << '\n'
             << "operator." << base << ".scale_sha256="
             << beam::sha256::file_hex(output_dir / scales_relative) << '\n'
             << "operator." << base << ".max_abs_roundtrip_error="
             << std::setprecision(9) << max_abs_error(physical_fp32, reconstructed) << '\n';
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc != 7 || std::string(argv[1]) != "--weight-dir" ||
        std::string(argv[3]) != "--output-dir" ||
        (std::string(argv[5]) != "--layers" && std::string(argv[5]) != "--scope")) {
        throw std::invalid_argument(
            "usage: exporter --weight-dir FP32_DIR --output-dir NEW_DIR "
            "(--layers CSV | --scope full_stream1)");
    }
    const fs::path weight_dir(argv[2]);
    const fs::path output_dir(argv[4]);
    const bool full_stream1 = std::string(argv[5]) == "--scope";
    if (full_stream1 && std::string(argv[6]) != "full_stream1") {
        throw std::invalid_argument("only --scope full_stream1 is supported");
    }
    const auto layers = full_stream1
        ? std::vector<std::uint32_t>{}
        : parse_layers(argv[6]);
    const auto operators = full_stream1
        ? full_stream1_operators(weight_dir)
        : std::vector<OperatorSpec>{};
    if (!full_stream1 && layers.empty()) {
        throw std::invalid_argument("offline NVFP4 export requires at least one layer");
    }
    if (fs::exists(output_dir)) {
        throw std::runtime_error("output directory already exists: " + output_dir.string());
    }
    if (!fs::is_regular_file(weight_dir / "manifest.json")) {
        throw std::runtime_error("missing source manifest.json");
    }
    fs::create_directories(output_dir / "weights");
    std::ostringstream manifest;
    manifest << "schema_version=" << (full_stream1 ? 2 : 1) << "\n"
             << "encoding=sm120_nvfp4_e2m1_k16_ue4m3_sfb\n"
             << "weights=offline_immutable\n"
             << "source_manifest_sha256="
             << beam::sha256::file_hex(weight_dir / "manifest.json") << '\n';
    if (full_stream1) {
        manifest << "profile_scope=full_stream1\n"
                 << "operator_count=" << operators.size() << '\n';
        for (const auto& spec : operators) {
            export_weight(
                spec.source, output_dir, spec.name, spec.input_cols,
                spec.logical_output_cols, spec.physical_output_cols, manifest);
        }
    } else {
        for (const std::uint32_t layer : layers) {
            const std::string prefix = "block" + std::to_string(layer);
            export_weight(
                weight_dir / (prefix + "_ff1_weight_hxk.fp32"), output_dir,
                prefix + "_ff1", 256U, 1024U, 1024U, manifest);
            export_weight(
                weight_dir / (prefix + "_ff2_weight_hxk.fp32"), output_dir,
                prefix + "_ff2", 1024U, 256U, 256U, manifest);
        }
    }
    const std::string text = manifest.str();
    std::ofstream output(output_dir / "runtime_manifest.txt", std::ios::binary);
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output) {
        throw std::runtime_error("cannot write runtime_manifest.txt");
    }
    std::cout << "stream1_sm120_nvfp4_export_done output_dir=" << output_dir
              << " scope=" << (full_stream1 ? "full_stream1" : "ffn_only")
              << " operators=" << (full_stream1 ? operators.size() : layers.size() * 2U)
              << '\n';
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << "stream1_sm120_nvfp4_export_error=" << error.what() << '\n';
    return EXIT_FAILURE;
}
