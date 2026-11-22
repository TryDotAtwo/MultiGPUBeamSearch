#pragma once
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include "config.hpp"
#include "cube444_blend_libtorch.hpp"

namespace beam::stream1_libtorch {
// Feature bodies retain Cube555 preprocessing. The existing ensemble owns the
// linear FP32 readout/weighted accumulation and performs clamp/round only once.
struct Cube555Native {
    torch::jit::script::Module backbone;
    torch::Tensor weight, bias;
    std::int64_t hidden = 0;

    Cube555Native(const fs::path& directory, const torch::Device& device,
                  const std::string& family) {
        const auto manifest = nlohmann::json::parse(read_text_exact(directory / "manifest.json"));
        if (STATE_LEN != 150 || MOVE_COUNT != 30 || manifest.at("schema_version") != 1 ||
            manifest.at("family") != family || manifest.at("state_len") != STATE_LEN ||
            manifest.at("num_classes") != 150 || manifest.at("output_dim") != MOVE_COUNT ||
            manifest.at("dtype") != "fp16" || manifest.at("feature_input_dtype") != "uint8" ||
            manifest.at("feature_output_dtype") != "fp16" ||
            manifest.at("output_weight_layout") != "hidden_by_output" ||
            manifest.at("readout_semantics") != "fp16_features_fp32_accumulation" ||
            manifest.at("score_semantics") != "distance_q")
            throw std::runtime_error("Cube555 feature/readout contract mismatch");
        hidden = manifest.at("feature_dim").get<std::int64_t>();
        if (hidden < 8 || hidden > 65536 || hidden % 8)
            throw std::runtime_error("invalid Cube555 feature width");
        for (const char* name : {"features.ts", "output_weight.fp16", "output_bias.fp32",
                                 "layout.json", "puzzle_info.json"}) {
            const auto bytes = read_text_exact(directory / name);
            if (picosha2::hash256_hex_string(bytes) != manifest.at("files").at(name).get<std::string>())
                throw std::runtime_error(std::string("Cube555 artifact checksum mismatch: ") + name);
        }
        backbone = torch::jit::load((directory / "features.ts").string(), device);
        backbone.eval();
        for (const auto& parameter : backbone.parameters())
            if (parameter.is_floating_point() && parameter.scalar_type() != torch::kFloat16)
                throw std::runtime_error("Cube555 feature parameters must be FP16");
        weight = load_tensor(directory / "output_weight.fp16", {hidden, MOVE_COUNT}, torch::kFloat16, device);
        bias = load_tensor(directory / "output_bias.fp32", {MOVE_COUNT}, torch::kFloat32, device);
        if (!torch::isfinite(weight).all().item<bool>() || !torch::isfinite(bias).all().item<bool>())
            throw std::runtime_error("nonfinite Cube555 readout");
    }

    torch::Tensor features(const torch::Tensor& states) const {
        // Preserve exported FP16 operation boundaries on the first and later calls.
        // Profiling JIT optimization changes real Transformer features after warmup.
        torch::jit::GraphOptimizerEnabledGuard stable_fp16(false);
        auto result = backbone.get_method("forward")({states.narrow(1, 0, STATE_LEN).contiguous()}).toTensor();
        if (result.scalar_type() != torch::kFloat16 || result.dim() != 2 ||
            result.size(0) != states.size(0) || result.size(1) != hidden || result.device() != states.device())
            throw std::runtime_error("Cube555 feature output shape/dtype/device mismatch");
        return result.contiguous();
    }
};
}
