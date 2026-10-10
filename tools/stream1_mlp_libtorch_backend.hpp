#pragma once
#include "stream1_transformer_libtorch_backend.hpp"

namespace beam::stream1_libtorch {
inline torch::Tensor scalar_children(const torch::Tensor& states, const std::uint8_t* generators,
                                     std::int64_t moves, std::int64_t logical, std::int64_t storage) {
    auto indices = torch::from_blob(const_cast<std::uint8_t*>(generators), {moves, storage}, states.options())
        .narrow(1, 0, logical).to(torch::kLong);
    return states.narrow(1, 0, logical).index_select(1, indices.flatten())
        .reshape({states.size(0) * moves, logical});
}
struct MlpLibTorch {
    struct Layer {
        torch::Tensor weight, bias, gamma, beta;
        torch::Tensor apply(const torch::Tensor& x) const {
            auto y = at::linear(x, weight, bias);
            if (gamma.defined()) {
                y = torch::layer_norm(y, {y.size(-1)}, gamma, beta, 1e-5);
            }
            return y;
        }
    };
    std::uint32_t state_len, num_classes, output_dim;
    torch::ScalarType dtype;
    Layer input, hidden, output;
    std::vector<std::pair<Layer, Layer>> residuals;

    MlpLibTorch(const fs::path& dir, const torch::Device& device) {
        const auto manifest = read_text_exact(dir / "manifest.json");
        state_len = manifest_u32(manifest, "state_len");
        num_classes = manifest_u32(manifest, "num_classes");
        output_dim = manifest_u32(manifest, "output_dim");
        const auto h1 = manifest_u32_any(manifest, "hidden1", "hd1");
        const auto h2 = manifest_u32_any(manifest, "hidden2", "hd2");
        const auto blocks = manifest_u32_any(manifest, "residual_count", "nrd");
        const auto format = manifest_string(manifest, "dtype");
        if (format != "fp16" && format != "bf16") throw std::runtime_error("unsupported MLP dtype");
        dtype = format == "fp16" ? torch::kFloat16 : torch::kBFloat16;
        const auto norm = manifest_string(manifest, "normalization");
        if (norm != "none" && norm != "batchnorm_folded" && norm != "layernorm") {
            throw std::runtime_error("unsupported MLP normalization");
        }
        auto load = [&](const std::string& name, std::int64_t in, std::int64_t out, bool normalize) {
            Layer layer;
            layer.weight = load_linear_weight_kxh(dir / (name + "_weight_hxk." + format), {in, out}, dtype, device);
            layer.bias = load_tensor(dir / (name + "_bias." + format), {out}, dtype, device);
            if (normalize && norm == "layernorm") {
                layer.gamma = load_tensor(dir / (name + "_ln_gamma." + format), {out}, dtype, device);
                layer.beta = load_tensor(dir / (name + "_ln_beta." + format), {out}, dtype, device);
            }
            return layer;
        };
        input = load("input", std::int64_t(state_len) * num_classes, h1, true);
        hidden = load("hidden", h1, h2, true);
        for (std::uint32_t i = 0; i < blocks; ++i) {
            const auto prefix = "residual" + std::to_string(i);
            residuals.emplace_back(load(prefix + "_fc1", h2, h2, true), load(prefix + "_fc2", h2, h2, true));
        }
        output = load("output", h2, output_dim, false);
    }
    torch::Tensor features(const torch::Tensor& states) const {
        auto labels = states.narrow(1, 0, state_len).to(torch::kLong);
        auto x = torch::one_hot(labels, num_classes).flatten(1).to(dtype);
        x = at::relu(input.apply(x));
        x = at::relu(hidden.apply(x));
        for (const auto& block : residuals) {
            x = at::relu(x + block.second.apply(at::relu(block.first.apply(x))));
        }
        return x.contiguous();
    }
    torch::Tensor forward(const torch::Tensor& states) const {
        return output.apply(features(states));
    }
};
} // namespace beam::stream1_libtorch
