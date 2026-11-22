#define BEAM_STREAM1_WEIGHT_IO_MANIFEST_ONLY
#include "../tools/stream1_weight_io.hpp"
#include <iostream>

int main() {
    using namespace beam::stream1_weights;
    unsigned failures = 0, checks = 0;
    auto reject = [&](const char* label, auto action) {
        ++checks;
        try { action(); ++failures; std::cerr << "FAIL accepted " << label << '\n'; }
        catch (const std::exception&) {}
    };
    auto check = [&](bool condition, const char* label) {
        ++checks;
        if (!condition) { ++failures; std::cerr << "FAIL " << label << '\n'; }
    };
    for (const char* value : {"1.2", "1e2", "true", "-1", "4294967296", "\"1\""})
        reject("non uint32", [&] { parse_manifest_u32(std::string("{\"layers\":") + value + "}", "layers"); });
    reject("nested field", [&] { parse_manifest_u32("{\"nested\":{\"layers\":1}}", "layers"); });
    reject("duplicate integer", [&] { parse_manifest_u32("{\"layers\":1,\"layers\":2}", "layers"); });
    reject("truncated JSON", [&] { parse_manifest_u32("{\"layers\":1", "layers"); });
    reject("wrong string type", [&] { parse_manifest_string("{\"dtype\":0,\"x\":\"fp16\"}", "dtype"); });
    reject("malformed optional field", [&] { parse_manifest_string_default("{\"dtype\":0}", "dtype", "fp16"); });
    reject("duplicate string", [&] { parse_manifest_string("{\"dtype\":\"fp16\",\"dtype\":\"bf16\"}", "dtype"); });
    check(parse_manifest_u32("{\"layers\":4294967295}", "layers") == UINT32_MAX, "uint32 maximum");
    check(parse_manifest_string_default("{}", "dtype", "fp16") == "fp16", "absent optional default");
    check(!manifest_has_key("{\"nested\":{\"dtype\":0}}", "dtype"), "top-level membership");
    check(parse_manifest_string("{\"dtype\":\"fp\\u0031\\u0036\"}", "dtype") == "fp16", "JSON string decoding");
    const auto fixture = std::filesystem::path("test_results/native_weight_values");
    std::filesystem::create_directories(fixture);
    for (const auto& spec : std::vector<std::pair<std::string, std::uint16_t>>{
            {"fp16", 0x7c00}, {"fp16", 0xfc00}, {"fp16", 0x7e01},
            {"bf16", 0x7f80}, {"bf16", 0xff80}, {"bf16", 0x7fc1}}) {
        const auto path = fixture / ("nonfinite." + spec.first);
        { std::ofstream out(path, std::ios::binary); out.write(reinterpret_cast<const char*>(&spec.second), 2); }
        reject("nonfinite binary weight", [&] { read_binary_exact(path, 2); });
    }
    const auto finite_path = fixture / "finite.fp16";
    const std::uint16_t finite_bits[] = {0, 0x8000, 1, 0x7bff, 0xfbff};
    { std::ofstream out(finite_path, std::ios::binary); out.write(reinterpret_cast<const char*>(finite_bits), sizeof(finite_bits)); }
    check(read_binary_exact(finite_path, sizeof(finite_bits)).size() == sizeof(finite_bits), "finite binary weights");
    // Real loader, tiny host-only model: catches missing semantic validation
    // even when every tensor has the expected byte count.
    const auto pieces = fixture / "pieces";
    std::filesystem::create_directories(pieces);
    auto write = [&](const char* name, std::vector<unsigned char> bytes) {
        std::ofstream out(pieces / name, std::ios::binary);
        out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    { std::ofstream out(pieces / "manifest.json"); out << "{\"num_piece_types\":3}"; }
    beam::Stream1ModelConfig model{};
    model.state_len = 2; model.num_classes = 2; model.num_pieces = 1;
    model.max_piece_size = 3; model.d_model = 1; model.output_dim = 1;
    write("fast_slot_projected.fp16", std::vector<unsigned char>(12));
    for (const char* name : {"fast_piece_static.fp16", "cls_token.fp16", "input_ln_gamma.fp16",
             "input_ln_beta.fp16", "output_ln_gamma.fp16", "output_ln_beta.fp16",
             "output_weight_hxk.fp16", "output_bias.fp16"}) write(name, {0, 0});
    auto reset = [&] {
        write("piece_positions.u16", {0, 0, 1, 0, 255, 255});
        write("piece_mask.u8", {1, 1, 0}); write("piece_types.u8", {2});
    };
    reset();
    check(load_stream1_transformer_weights(pieces, model).transformer.piece_mask.size() == 3,
          "valid pieces with ignored masked position");
    for (const auto& bad : std::vector<std::vector<unsigned char>>{{2, 1, 0}, {0, 0, 0}, {1, 0, 0}}) {
        reset(); write("piece_mask.u8", bad);
        reject("mask binary/nonempty/coverage", [&] { load_stream1_transformer_weights(pieces, model); });
    }
    for (const auto& bad : std::vector<std::vector<unsigned char>>{{0, 0, 0, 0, 0, 0}, {0, 0, 2, 0, 0, 0}}) {
        reset(); write("piece_positions.u16", bad);
        reject("duplicate or out-of-range position", [&] { load_stream1_transformer_weights(pieces, model); });
    }
    reset(); write("piece_types.u8", {3});
    reject("out-of-range piece type", [&] { load_stream1_transformer_weights(pieces, model); });
    std::cout << "native_manifest checks=" << checks << " failures=" << failures << '\n';
    return failures ? 1 : 0;
}
