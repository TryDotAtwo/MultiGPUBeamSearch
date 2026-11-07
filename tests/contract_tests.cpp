#define BEAM_STREAM1_WEIGHT_IO_MANIFEST_ONLY
#include "config.hpp"
#include "frontier_cpu.hpp"
#include "../tools/stream1_weight_io.hpp"

#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace beam;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Generator identity_generator() {
    Generator generator{};
    for (std::size_t i = 0; i < STATE_STORAGE_LEN; ++i) {
        generator[i] = static_cast<std::uint8_t>(i);
    }
    return generator;
}

Generator swap01_generator() {
    Generator generator = identity_generator();
    generator[0] = 1;
    generator[1] = 0;
    return generator;
}

std::string minimal_stream1_manifest(const std::string& backend_line) {
    std::string manifest = "{\n";
    if (!backend_line.empty()) {
        manifest += backend_line + ",\n";
    }
    manifest +=
        "  \"dtype\": \"fp16\",\n"
        "  \"state_len\": " + std::to_string(STATE_LEN) + ",\n"
        "  \"num_classes\": " + std::to_string(STATE_LEN) + ",\n"
        "  \"hidden1\": 1536,\n"
        "  \"hidden2\": 512,\n"
        "  \"residual_count\": 2,\n"
        "  \"output_dim\": " + std::to_string(MOVE_COUNT) + ",\n"
        "  \"normalization\": \"none\"\n"
        "}\n";
    return manifest;
}
std::string transformer_manifest(std::uint32_t seq_len = 51, std::uint32_t output_dim = static_cast<std::uint32_t>(MOVE_COUNT)) {
    return
        "{\n"
        "  \"backend\": \"piece_transformer\",\n"
        "  \"dtype\": \"fp16\",\n"
        "  \"state_len\": " + std::to_string(STATE_LEN) + ",\n"
        "  \"num_classes\": 120,\n"
        "  \"output_dim\": " + std::to_string(output_dim) + ",\n"
        "  \"num_pieces\": 50,\n"
        "  \"max_piece_size\": 3,\n"
        "  \"seq_len\": " + std::to_string(seq_len) + ",\n"
        "  \"d_model\": 256,\n"
        "  \"nhead\": 8,\n"
        "  \"head_dim\": 32,\n"
        "  \"num_layers\": 4,\n"
        "  \"ff_dim\": 1024,\n"
        "  \"activation\": \"silu\",\n"
        "  \"pooling\": \"cls\",\n"
        "  \"piece_layout\": \"p900\",\n"
        "  \"piece_embed_mode\": \"full_s120\"\n"
        "}\n";
}

std::string transformer_manifest_with_layer_fields(const std::string& layer_fields) {
    return
        "{\n"
        "  \"backend\": \"piece_transformer\",\n"
        "  \"dtype\": \"fp16\",\n"
        "  \"state_len\": " + std::to_string(STATE_LEN) + ",\n"
        "  \"num_classes\": 120,\n"
        "  \"output_dim\": " + std::to_string(MOVE_COUNT) + ",\n"
        "  \"num_pieces\": 50,\n"
        "  \"max_piece_size\": 3,\n"
        "  \"seq_len\": 51,\n"
        "  \"d_model\": 256,\n"
        "  \"nhead\": 8,\n"
        "  \"head_dim\": 32,\n" +
        layer_fields +
        "  \"ff_dim\": 1024,\n"
        "  \"activation\": \"silu\",\n"
        "  \"pooling\": \"cls\",\n"
        "  \"piece_layout\": \"p900\",\n"
        "  \"piece_embed_mode\": \"full_s120\"\n"
        "}\n";
}

std::filesystem::path write_manifest_fixture(const std::string& name, const std::string& backend_line) {
    const std::filesystem::path dir = std::filesystem::path("test_results") / name;
    std::filesystem::create_directories(dir);
    std::ofstream manifest(dir / "manifest.json");
    manifest << minimal_stream1_manifest(backend_line);
    return dir;
}
std::filesystem::path write_transformer_manifest_fixture(
    const std::string& name,
    std::uint32_t seq_len = 51,
    std::uint32_t output_dim = static_cast<std::uint32_t>(MOVE_COUNT)) {
    const std::filesystem::path dir = std::filesystem::path("test_results") / name;
    std::filesystem::create_directories(dir);
    std::ofstream manifest(dir / "manifest.json");
    manifest << transformer_manifest(seq_len, output_dim);
    return dir;
}

std::filesystem::path write_transformer_manifest_text_fixture(
    const std::string& name,
    const std::string& manifest_text) {
    const std::filesystem::path dir = std::filesystem::path("test_results") / name;
    std::filesystem::create_directories(dir);
    std::ofstream manifest(dir / "manifest.json");
    manifest << manifest_text;
    return dir;
}

void write_zero_file(const std::filesystem::path& path, std::uint64_t bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to create fixture file: " + path.string());
    }
    if (bytes > 0) {
        out.seekp(static_cast<std::streamoff>(bytes - 1));
        out.put('\0');
    }
}

void set_test_environment(const char* name, const std::string& value) {
#if defined(_WIN32)
    if (_putenv_s(name, value.c_str()) != 0) {
        throw std::runtime_error("failed to set test environment");
    }
#else
    if (setenv(name, value.c_str(), 1) != 0) {
        throw std::runtime_error("failed to set test environment");
    }
#endif
}

void write_transformer_weight_fixture_files(const std::filesystem::path& dir, bool bad_fast_static_size = false) {
    constexpr std::uint64_t elem = sizeof(std::uint16_t);
    const std::string suffix = ".fp16";
    write_zero_file(dir / ("fast_slot_projected" + suffix), 3ULL * 120ULL * 256ULL * elem);
    write_zero_file(dir / ("fast_piece_static" + suffix), 50ULL * 256ULL * elem + (bad_fast_static_size ? 1ULL : 0ULL));
    write_zero_file(dir / ("cls_token" + suffix), 256ULL * elem);
    write_zero_file(dir / ("input_ln_gamma" + suffix), 256ULL * elem);
    write_zero_file(dir / ("input_ln_beta" + suffix), 256ULL * elem);
    write_zero_file(dir / ("output_ln_gamma" + suffix), 256ULL * elem);
    write_zero_file(dir / ("output_ln_beta" + suffix), 256ULL * elem);
    for (std::uint32_t block = 0; block < 4; ++block) {
        const std::string prefix = "block" + std::to_string(block);
        write_zero_file(dir / (prefix + "_ln1_gamma" + suffix), 256ULL * elem);
        write_zero_file(dir / (prefix + "_ln1_beta" + suffix), 256ULL * elem);
        write_zero_file(dir / (prefix + "_attn_qkv_weight_hxk" + suffix), 768ULL * 256ULL * elem);
        write_zero_file(dir / (prefix + "_attn_qkv_bias" + suffix), 768ULL * elem);
        write_zero_file(dir / (prefix + "_attn_out_weight_hxk" + suffix), 256ULL * 256ULL * elem);
        write_zero_file(dir / (prefix + "_attn_out_bias" + suffix), 256ULL * elem);
        write_zero_file(dir / (prefix + "_ln2_gamma" + suffix), 256ULL * elem);
        write_zero_file(dir / (prefix + "_ln2_beta" + suffix), 256ULL * elem);
        write_zero_file(dir / (prefix + "_ff1_weight_hxk" + suffix), 1024ULL * 256ULL * elem);
        write_zero_file(dir / (prefix + "_ff1_bias" + suffix), 1024ULL * elem);
        write_zero_file(dir / (prefix + "_ff2_weight_hxk" + suffix), 256ULL * 1024ULL * elem);
        write_zero_file(dir / (prefix + "_ff2_bias" + suffix), 256ULL * elem);
    }
    write_zero_file(dir / ("output_weight_hxk" + suffix), 24ULL * 256ULL * elem);
    write_zero_file(dir / ("output_bias" + suffix), 24ULL * elem);
    write_zero_file(dir / "piece_positions.u16", 50ULL * 3ULL * sizeof(std::uint16_t));
    write_zero_file(dir / "piece_mask.u8", 50ULL * 3ULL);
    write_zero_file(dir / "piece_types.u8", 50ULL);
}


std::filesystem::path write_mlp_weight_fixture(const std::string& name) {
    const std::filesystem::path dir = std::filesystem::path("test_results") / name;
    std::filesystem::create_directories(dir);
    std::ofstream manifest(dir / "manifest.json");
    manifest << "{\n"
             << "  \"backend\": \"mlp\",\n"
             << "  \"dtype\": \"fp16\",\n"
             << "  \"state_len\": " << STATE_LEN << ",\n"
             << "  \"num_classes\": " << STATE_LEN << ",\n"
             << "  \"hidden1\": 8,\n"
             << "  \"hidden2\": 8,\n"
             << "  \"residual_count\": 1,\n"
             << "  \"output_dim\": " << MOVE_COUNT << ",\n"
             << "  \"normalization\": \"none\"\n"
             << "}\n";
    constexpr std::uint64_t elem = sizeof(std::uint16_t);
    const std::string suffix = ".fp16";
    write_zero_file(dir / ("input_weight_hxk" + suffix),
        static_cast<std::uint64_t>(STATE_LEN) * STATE_LEN * 8ULL * elem);
    write_zero_file(dir / ("input_bias" + suffix), 8ULL * elem);
    write_zero_file(dir / ("hidden_weight_hxk" + suffix), 8ULL * 8ULL * elem);
    write_zero_file(dir / ("hidden_bias" + suffix), 8ULL * elem);
    write_zero_file(dir / ("residual0_fc1_weight_hxk" + suffix), 8ULL * 8ULL * elem);
    write_zero_file(dir / ("residual0_fc1_bias" + suffix), 8ULL * elem);
    write_zero_file(dir / ("residual0_fc2_weight_hxk" + suffix), 8ULL * 8ULL * elem);
    write_zero_file(dir / ("residual0_fc2_bias" + suffix), 8ULL * elem);
    write_zero_file(dir / ("output_weight_hxk" + suffix), 8ULL * MOVE_COUNT * elem);
    write_zero_file(dir / ("output_bias" + suffix), MOVE_COUNT * elem);
    return dir;
}
bool error_contains(const std::runtime_error& error, const std::string& needle) {
    return std::string(error.what()).find(needle) != std::string::npos;
}

void test_stream1_manifest_backend_parsing() {
    const Stream1ModelConfig legacy = beam::stream1_weights::load_stream1_manifest(
        write_manifest_fixture("stream1_manifest_backend_legacy_mlp", ""));
    require(legacy.backend == STREAM1_BACKEND_MLP, "legacy Stream1 manifest without backend must load as MLP");

    const Stream1ModelConfig explicit_mlp = beam::stream1_weights::load_stream1_manifest(
        write_manifest_fixture("stream1_manifest_backend_explicit_mlp", "  \"backend\": \"mlp\""));
    require(explicit_mlp.backend == STREAM1_BACKEND_MLP, "explicit MLP Stream1 manifest must load as MLP");

    const Stream1ModelConfig transformer = beam::stream1_weights::load_stream1_manifest(
        write_transformer_manifest_fixture("stream1_manifest_backend_piece_transformer"));
    require(transformer.backend == STREAM1_BACKEND_PIECE_TRANSFORMER, "piece_transformer manifest must parse as transformer");

    bool unknown_threw = false;
    try {
        (void)beam::stream1_weights::load_stream1_manifest(
            write_manifest_fixture("stream1_manifest_backend_unknown", "  \"backend\": \"other\""));
    } catch (const std::runtime_error& error) {
        unknown_threw = error_contains(error, "accepted values: mlp, piece_transformer");
    }
    require(unknown_threw, "unknown Stream1 manifest backend must name accepted values");
}
void test_stream1_transformer_manifest_contract() {
    const Stream1ModelConfig model = beam::stream1_weights::load_stream1_manifest(
        write_transformer_manifest_fixture("stream1_transformer_manifest_ok"));
    require(model.backend == STREAM1_BACKEND_PIECE_TRANSFORMER, "transformer backend parse failed");
    require(model.state_len == STATE_LEN, "transformer state_len parse failed");
    require(model.num_classes == 120U, "transformer num_classes parse failed");
    require(model.output_dim == MOVE_COUNT, "transformer output_dim parse failed");
    require(model.num_pieces == 50U, "transformer num_pieces parse failed");
    require(model.max_piece_size == 3U, "transformer max_piece_size parse failed");
    require(model.seq_len == 51U, "transformer seq_len parse failed");
    require(model.d_model == 256U, "transformer d_model parse failed");
    require(model.nhead == 8U, "transformer nhead parse failed");
    require(model.head_dim == 32U, "transformer head_dim parse failed");
    require(model.transformer_layers == 4U, "transformer_layers parse failed");
    require(model.ff_dim == 1024U, "transformer ff_dim parse failed");

    bool bad_seq_len_threw = false;
    try {
        (void)beam::stream1_weights::load_stream1_manifest(
            write_transformer_manifest_fixture("stream1_transformer_manifest_bad_seq_len", 52));
    } catch (const std::runtime_error& error) {
        bad_seq_len_threw = error_contains(error, "seq_len");
    }
    require(bad_seq_len_threw, "wrong transformer seq_len must be rejected with a clear error");

    bool malformed_primary_threw = false;
    try {
        (void)beam::stream1_weights::load_stream1_manifest(
            write_transformer_manifest_text_fixture(
                "stream1_transformer_manifest_bad_primary_layer",
                transformer_manifest_with_layer_fields(
                    "  \"transformer_layers\": \"four\",\n"
                    "  \"num_layers\": 4,\n")));
    } catch (const std::runtime_error& error) {
        malformed_primary_threw = error_contains(error, "transformer_layers");
    }
    require(malformed_primary_threw, "malformed transformer_layers must not fall back to num_layers");

    bool mismatched_layers_threw = false;
    try {
        (void)beam::stream1_weights::load_stream1_manifest(
            write_transformer_manifest_text_fixture(
                "stream1_transformer_manifest_mismatched_layers",
                transformer_manifest_with_layer_fields(
                    "  \"transformer_layers\": 3,\n"
                    "  \"num_layers\": 4,\n")));
    } catch (const std::runtime_error& error) {
        mismatched_layers_threw =
            error_contains(error, "transformer_layers") && error_contains(error, "num_layers");
    }
    require(mismatched_layers_threw, "transformer_layers and num_layers mismatch must be rejected");
}

void test_stream1_transformer_weight_size_contract() {
    const std::filesystem::path ok_dir = write_transformer_manifest_fixture("stream1_transformer_weight_sizes_ok");
    write_transformer_weight_fixture_files(ok_dir);
    const auto weights = beam::stream1_weights::load_stream1_weights(ok_dir);
    require(weights.model.backend == STREAM1_BACKEND_PIECE_TRANSFORMER, "transformer weights must keep backend metadata");
    require(weights.transformer.blocks.size() == 4U, "transformer loader must allocate four block containers");
    require(weights.transformer.fast_slot_projected.size() == 3ULL * 120ULL * 256ULL * sizeof(std::uint16_t),
        "fast_slot_projected exact size failed");
    require(weights.transformer.blocks[0].ff1_weight.size() == 1024ULL * 256ULL * sizeof(std::uint16_t),
        "transformer block ff1_weight exact size failed");
    require(beam::stream1_weights::total_host_weight_bytes(weights) > 0, "host byte helper must count transformer tensors");

    const std::filesystem::path bad_dir = write_transformer_manifest_fixture("stream1_transformer_weight_sizes_bad");
    write_transformer_weight_fixture_files(bad_dir, true);
    bool bad_size_threw = false;
    try {
        (void)beam::stream1_weights::load_stream1_weights(bad_dir);
    } catch (const std::runtime_error& error) {
        bad_size_threw = error_contains(error, "fast_piece_static") && error_contains(error, "expected=");
    }
    require(bad_size_threw, "wrong transformer file size must be rejected with file name and expected size");
}

void test_stream1_mlp_weight_loading_still_works() {
    const auto weights = beam::stream1_weights::load_stream1_weights(
        write_mlp_weight_fixture("stream1_mlp_weight_loading_ok"));
    require(weights.model.backend == STREAM1_BACKEND_MLP, "fixture MLP manifest must load as MLP");
    require(weights.input_weight.size() == static_cast<std::uint64_t>(STATE_LEN) * STATE_LEN * 8ULL * sizeof(std::uint16_t),
        "MLP input_weight fixture size mismatch");
    require(weights.output_bias.size() == weights.model.output_dim * sizeof(std::uint16_t), "MLP output_bias size mismatch");
}

void test_sm120_nvfp4_runtime_artifact_contract() {
    namespace sw = beam::stream1_weights;
    const std::filesystem::path root =
        std::filesystem::path("test_results") / "sm120_nvfp4_runtime_artifact_contract";
    const std::filesystem::path base = root / "base";
    const std::filesystem::path profile = root / "profile";
    std::filesystem::create_directories(base);
    std::filesystem::create_directories(profile / "weights");
    const std::string checkpoint(64U, 'a');
    {
        std::ofstream manifest(base / "manifest.json");
        manifest << "{\"checkpoint_sha256\":\"" << checkpoint << "\"}\n";
    }
    {
        std::ofstream manifest(profile / "runtime_manifest.txt");
        manifest << "schema_version=4\n"
                 << "backend=sm120a_nvfp4_persistent\n"
                 << "encoding=sm120_nvfp4_e2m1_k16_ue4m3_sfb\n"
                 << "weights=offline_immutable\n"
                 << "target_sm=120\n"
                 << "target_arch=sm120a\n"
                 << "profile_scope=full_stream1\n"
                 << "promotion_state=accepted\n"
                 << "microbatch=896\n"
                 << "physical_slots=2\n"
                 << "operator_count=17\n"
                 << "checkpoint_sha256=" << checkpoint << "\n";
        auto record = [&](const char* name, std::uint32_t input_cols,
                          std::uint32_t logical_output_cols,
                          std::uint32_t physical_output_cols) {
            const auto values = profile / (std::string("weights/") + name + ".e2m1");
            const auto scales = profile / (std::string("weights/") + name + ".ue4m3_sfb");
            write_zero_file(values, stream1_sm120_nvfp4_weight_value_bytes(
                input_cols, physical_output_cols));
            write_zero_file(scales, stream1_sm120_nvfp4_weight_scale_bytes(
                input_cols, physical_output_cols));
            manifest << "operator." << name << ".input_cols=" << input_cols << "\n"
                     << "operator." << name << ".logical_output_cols="
                     << logical_output_cols << "\n"
                     << "operator." << name << ".physical_output_cols="
                     << physical_output_cols << "\n"
                     << "operator." << name << ".value_file="
                     << std::filesystem::relative(values, profile).generic_string() << "\n"
                     << "operator." << name << ".value_sha256=" << beam::sha256::file_hex(values) << "\n"
                     << "operator." << name << ".scale_file="
                     << std::filesystem::relative(scales, profile).generic_string() << "\n"
                     << "operator." << name << ".scale_sha256=" << beam::sha256::file_hex(scales) << "\n";
        };
        for (std::uint32_t layer = 0U; layer < 4U; ++layer) {
            const std::string base = "block" + std::to_string(layer);
            record((base + "_attn_qkv").c_str(), 256U, 768U, 768U);
            record((base + "_attn_out").c_str(), 256U, 256U, 256U);
            record((base + "_ff1").c_str(), 256U, 1024U, 1024U);
            record((base + "_ff2").c_str(), 1024U, 256U, 256U);
        }
        record("output_head", 256U, 24U, 128U);
    }
    sw::HostWeightBytes host;
    host.model.backend = STREAM1_BACKEND_PIECE_TRANSFORMER;
    host.model.dtype = STREAM1_DTYPE_FP16;
    host.model.output_dim = 24U;
    host.model.d_model = 256U;
    host.model.ff_dim = 1024U;
    host.model.transformer_layers = 4U;
    host.model.activation = STREAM1_ACTIVATION_RELU;
    host.transformer.blocks.resize(4U);
    set_test_environment("BEAM_STREAM1_SM120_PROFILE_DIR", profile.string());
    sw::load_sm120_offline_artifacts(base, host);
    sw::load_sm120_nvfp4_ffn_artifacts(base, host);
    set_test_environment("BEAM_STREAM1_SM120_PROFILE_DIR", "");
    require(host.sm120_nvfp4_ffn_layer_mask == 15U, "NVFP4 full-graph layer mask mismatch");
    require(
        host.sm120_nvfp4_execution_kind ==
            STREAM1_SM120_NVFP4_EXECUTION_PERSISTENT_FULL_STREAM1,
        "schema-4 profile must preserve persistent full-Stream1 execution kind");
    require(host.transformer.blocks[0].ff1_weight_nvfp4.size() == 131072U,
            "NVFP4 FF1 packed value size mismatch");
    require(host.transformer.blocks[0].ff2_weight_nvfp4_scales.size() == 65536U,
            "NVFP4 FF2 physical scale-factor size mismatch");
    require(host.transformer.blocks[3].attn_qkv_weight_nvfp4.size() == 98304U,
            "NVFP4 QKV packed value size mismatch");
    require(host.transformer.output_weight_nvfp4.size() == 16384U,
            "NVFP4 padded output-head value size mismatch");

    std::string component_manifest = sw::read_text_exact(profile / "runtime_manifest.txt");
    const std::string accepted_line = "promotion_state=accepted";
    const std::size_t accepted_position = component_manifest.find(accepted_line);
    require(accepted_position != std::string::npos, "accepted fixture promotion state missing");
    component_manifest.replace(
        accepted_position,
        accepted_line.size(),
        "promotion_state=component_candidate");
    {
        std::ofstream manifest(profile / "runtime_manifest.txt", std::ios::trunc);
        manifest << component_manifest;
    }
    sw::HostWeightBytes component;
    component.model = host.model;
    component.transformer.blocks.resize(4U);
    bool component_rejected = false;
    set_test_environment("BEAM_STREAM1_SM120_PROFILE_DIR", profile.string());
    try {
        sw::load_sm120_nvfp4_ffn_artifacts(base, component);
    } catch (const std::runtime_error& error) {
        component_rejected = error_contains(error, "component_candidate");
    }
    set_test_environment("BEAM_STREAM1_SM120_PROFILE_DIR", "");
    require(
        component_rejected,
        "component-candidate NVFP4 profile must never enter production dispatch");

    const std::string component_line = "promotion_state=component_candidate";
    const std::size_t component_position = component_manifest.find(component_line);
    require(component_position != std::string::npos, "component fixture state missing");
    component_manifest.replace(
        component_position,
        component_line.size(),
        "promotion_state=accepted");
    const std::string two_slots_line = "physical_slots=2";
    const std::size_t slots_position = component_manifest.find(two_slots_line);
    require(slots_position != std::string::npos, "two-slot fixture geometry missing");
    component_manifest.replace(slots_position, two_slots_line.size(), "physical_slots=3");
    {
        std::ofstream manifest(profile / "runtime_manifest.txt", std::ios::trunc);
        manifest << component_manifest;
    }
    sw::HostWeightBytes wrong_slots;
    wrong_slots.model = host.model;
    wrong_slots.transformer.blocks.resize(4U);
    bool wrong_slots_rejected = false;
    set_test_environment("BEAM_STREAM1_SM120_PROFILE_DIR", profile.string());
    try {
        sw::load_sm120_nvfp4_ffn_artifacts(base, wrong_slots);
    } catch (const std::runtime_error& error) {
        wrong_slots_rejected = error_contains(error, "physical_slots");
    }
    set_test_environment("BEAM_STREAM1_SM120_PROFILE_DIR", "");
    require(
        wrong_slots_rejected,
        "schema-4 production profile must require exactly two physical slots");

    sw::HostWeightBytes legacy;
    legacy.model = host.model;
    legacy.transformer.blocks.resize(4U);
    bool legacy_rejected = false;
    set_test_environment("BEAM_STREAM1_SM120_NVFP4_PROFILE_DIR", profile.string());
    try {
        sw::load_sm120_nvfp4_ffn_artifacts(base, legacy);
    } catch (const std::runtime_error& error) {
        legacy_rejected = error_contains(error, "BEAM_STREAM1_SM120_PROFILE_DIR");
    }
    set_test_environment("BEAM_STREAM1_SM120_NVFP4_PROFILE_DIR", "");
    require(
        legacy_rejected,
        "legacy NVFP4 profile environment must fail closed instead of silently falling back");
}

void test_stream1_backend_row_modes() {
    Stream1ModelConfig default_mlp;
    require(default_mlp.backend == STREAM1_BACKEND_MLP, "default Stream1 backend must be MLP");
    require(default_mlp.output_dim == MOVE_COUNT, "default MLP output_dim must remain MOVE_COUNT");
    require(!stream1_uses_child_rows(default_mlp), "default MLP MOVE_COUNT output must use parent rows");
    require(stream1_rows_per_parent(default_mlp) == 1U, "default MLP MOVE_COUNT output rows_per_parent must be 1");

    Stream1ModelConfig mlp;
    mlp.backend = STREAM1_BACKEND_MLP;
    mlp.output_dim = STREAM1_SINGLE_SCORE_OUTPUT_DIM;
    if (!stream1_uses_child_rows(mlp) || stream1_rows_per_parent(mlp) != MOVE_COUNT) {
        throw std::runtime_error("MLP one-output backend must use child rows");
    }

    Stream1ModelConfig transformer;
    transformer.backend = STREAM1_BACKEND_PIECE_TRANSFORMER;
    transformer.output_dim = static_cast<std::uint32_t>(MOVE_COUNT);
    transformer.num_pieces = 50;
    transformer.max_piece_size = 3;
    transformer.seq_len = 51;
    transformer.d_model = 256;
    transformer.nhead = 8;
    transformer.head_dim = 32;
    transformer.transformer_layers = 4;
    transformer.ff_dim = 1024;
    if (transformer.num_pieces != 50 || transformer.max_piece_size != 3 || transformer.seq_len != 51 ||
        transformer.d_model != 256 || transformer.nhead != 8 || transformer.head_dim != 32 ||
        transformer.transformer_layers != 4 || transformer.ff_dim != 1024) {
        throw std::runtime_error("piece_transformer sample config metadata mismatch");
    }
    if (stream1_uses_child_rows(transformer) || stream1_rows_per_parent(transformer) != 1U) {
        throw std::runtime_error("piece_transformer backend must use parent rows");
    }

    Stream1ModelConfig invalid;
    invalid.backend = 999U;
    bool threw = false;
    try {
        (void)stream1_rows_per_parent(invalid);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "invalid Stream1 backend must be rejected");
}

} // namespace

int run_contract_tests() {
    std::filesystem::create_directories("test_results");
    std::ofstream report("test_results/contract_tests_2026-05-20.md");
    report << "# Contract Tests 2026-05-20\n\n";

    RuntimeConfig config;
    config.user_global_beam_width = 1000;
    config.world_size = 2;
    config.shard_count = 8;
    config.stream4_batch_alignment = 16;
    const DerivedConfig derived = derive_config(config);
    require(derived.ring_slot_count == config.stream3_batch_candidates / (config.b_micro * MOVE_COUNT), "ring_slot_count formula failed");
    require(derived.beam_width_alignment == 256, "beam_width_alignment formula failed");
    require(derived.global_beam_width_effective == 1024, "effective beam width formula failed");
    report << "- config_derivation=pass\n";

    require(q_to_score_key(-5.0f) == 0, "negative score clamp failed");
    require(q_to_score_key(300.5f) == SCORE_MAX_KEY, "max score clamp failed");
    require(q_to_score_key(1.5f) == 1536, "score rounding failed");
    report << "- score_quantization=pass\n";

    test_stream1_backend_row_modes();
    report << "- stream1_backend_row_modes=pass\n";
    test_stream1_manifest_backend_parsing();
    report << "- stream1_manifest_backend_parsing=pass\n";
    test_stream1_transformer_manifest_contract();
    report << "- stream1_transformer_manifest_contract=pass\n";
    test_stream1_transformer_weight_size_contract();
    report << "- stream1_transformer_weight_size_contract=pass\n";
    test_stream1_mlp_weight_loading_still_works();
    report << "- stream1_mlp_weight_loading=pass\n";
    test_sm120_nvfp4_runtime_artifact_contract();
    report << "- sm120_nvfp4_runtime_artifact_contract=pass\n";

    State128 state = make_state128(std::vector<std::uint8_t>(STATE_LEN, 7));
    require(padding_is_zero(state), "state padding must be zero");
    final_response_set_target_local_idx(state, 0x44332211);
    require(final_response_get_target_local_idx(state) == 0x44332211, "final response byte-pack failed");
    clear_state_padding(state);
    require(padding_is_zero(state), "padding cleanup failed");
    report << "- final_response_padding=pass\n";

    const auto zobrist = make_deterministic_zobrist(42);
    Hash128 h1 = hash_state(state, zobrist);
    state.v[STATE_LEN] = 99;
    state.v[STATE_STORAGE_LEN - 1] = 123;
    Hash128 h2 = hash_state(state, zobrist);
    require(h1 == h2, "padding must not influence hash");
    clear_state_padding(state);
    report << "- hash_padding_invariant=pass\n";

    CandidateMeta a{Hash128{1, 2}, 5, 10, pack_route(0, 0, 3)};
    CandidateMeta b{Hash128{1, 2}, 4, 10, pack_route(0, 0, 4)};
    CandidateMeta c{Hash128{3, 2}, 6, 9, pack_route(0, 0, 5)};
    auto deduped = stream4_threshold_sort_dedup({a, b, c}, 10);
    require(deduped.size() == 2, "stream4 dedup size failed");
    require(deduped[0].parent_idx == 4, "stream4 deterministic tie-break failed");
    report << "- stream4_threshold_sort_dedup=pass\n";

    std::vector<Stream3CandidateInput> s3input{
        {Hash128{7, 1}, 5, 10, 100, 1},
        {Hash128{7, 1}, 3, 99, 101, 2},
        {Hash128{8, 1}, 11, 12, 102, 3},
    };
    auto split = stream3_threshold_dedup_split(s3input, 10, 0, 2);
    require(split.local_pending.size() + split.remote_send.size() == 1, "stream3 threshold/dedup failed");
    const CandidateMeta s3meta = split.local_pending.empty() ? split.remote_send.front() : split.local_pending.front();
    require(s3meta.score_key == 3, "stream3 min stream3_val score failed");
    require(s3meta.parent_idx == 101, "stream3 payload restore failed");
    report << "- stream3_threshold_dedup_split=pass\n";

    State128 central{};
    for (std::size_t i = 0; i < STATE_LEN; ++i) {
        central.v[i] = static_cast<std::uint8_t>(i);
    }
    clear_state_padding(central);
    State128 start = central;
    start.v[0] = 1;
    start.v[1] = 0;
    std::vector<Generator> generators{identity_generator(), swap01_generator()};
    CpuDepthResult depth = expand_depth_cpu_reference({start}, generators, central, zobrist, UINT32_THRESHOLD_MAX, 8);
    require(depth.solved, "cpu reference must detect solved child");
    require(!depth.next_frontier.empty(), "cpu reference next frontier empty");
    require(padding_is_zero(depth.next_frontier.front()), "cpu reference persistent padding failed");
    report << "- cpu_reference_depth=pass\n";

    report << "\nstatus=pass\n";
    std::cout << "contract_tests=pass\n";
    return 0;
}

int main() {
    try {
        return run_contract_tests();
    } catch (const std::exception& error) {
        std::cerr << "contract_tests=fail error=" << error.what() << "\n";
        return 1;
    }
}
