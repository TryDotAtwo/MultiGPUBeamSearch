#pragma once
#include <ATen/ops/scaled_dot_product_attention.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace beam::stream1_libtorch {
namespace fs = std::filesystem;

inline constexpr double kScoreMaxQ = 300.0;
inline constexpr double kScoreScale = 1024.0;


inline std::string read_text_exact(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open text file: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

inline std::vector<std::uint8_t> read_binary_exact(const fs::path& path, std::uint64_t bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open binary file: " + path.string());
    }
    std::vector<std::uint8_t> out(static_cast<std::size_t>(bytes));
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
    if (in.gcount() != static_cast<std::streamsize>(out.size())) {
        throw std::runtime_error("short read: " + path.string());
    }
    char extra = 0;
    if (in.read(&extra, 1)) {
        throw std::runtime_error("file larger than expected: " + path.string());
    }
    return out;
}

inline std::string manifest_string(const std::string& text, const std::string& key) {
    const std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (!std::regex_search(text, match, re)) {
        throw std::runtime_error("manifest missing string key: " + key);
    }
    return match[1].str();
}

inline std::uint32_t manifest_u32(const std::string& text, const std::string& key) {
    const std::regex re("\"" + key + "\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(text, match, re)) {
        throw std::runtime_error("manifest missing integer key: " + key);
    }
    const unsigned long value = std::stoul(match[1].str());
    if (value > UINT32_MAX) {
        throw std::runtime_error("manifest integer key too large: " + key);
    }
    return static_cast<std::uint32_t>(value);
}

inline std::uint32_t manifest_u32_any(const std::string& text, const std::string& key, const std::string& alternate) {
    const std::regex re("\"" + key + "\"\\s*:\\s*([0-9]+)");
    if (std::regex_search(text, re)) {
        return manifest_u32(text, key);
    }
    return manifest_u32(text, alternate);
}

inline std::vector<std::int64_t> parse_csv_i64(const std::string& text) {
    std::vector<std::int64_t> values;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string token = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) {
            values.push_back(std::stoll(token));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (values.empty()) {
        throw std::runtime_error("empty integer csv list");
    }
    return values;
}

inline std::int64_t product(const std::vector<std::int64_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), std::int64_t{1}, std::multiplies<std::int64_t>());
}

inline torch::Tensor load_tensor(
    const fs::path& path,
    const std::vector<std::int64_t>& shape,
    torch::ScalarType dtype,
    const torch::Device& device) {
    const std::int64_t values = product(shape);
    const std::uint64_t element_bytes = static_cast<std::uint64_t>(torch::elementSize(dtype));
    std::vector<std::uint8_t> bytes = read_binary_exact(path, static_cast<std::uint64_t>(values) * element_bytes);
    torch::Tensor cpu = torch::from_blob(
        bytes.data(),
        shape,
        torch::TensorOptions().dtype(dtype).device(torch::kCPU));
    return cpu.clone().to(device, /*non_blocking=*/false);
}

inline torch::Tensor make_states(
    std::int64_t batch,
    std::int64_t state_len,
    std::int64_t num_classes,
    const torch::Device& device) {
    torch::Tensor values = torch::arange(batch * state_len, torch::TensorOptions().dtype(torch::kInt64).device(device));
    values = (values.view({batch, state_len}) * 17 + 23).remainder(num_classes);
    return values.to(torch::kUInt8);
}

inline torch::Tensor score_keys(const torch::Tensor& logits) {
    return torch::round(torch::clamp(logits.to(torch::kFloat32), 0.0, kScoreMaxQ) * kScoreScale).to(torch::kInt64);
}

struct TransformerBlock {
    torch::Tensor ln1_gamma;
    torch::Tensor ln1_beta;
    torch::Tensor qkv_weight;
    torch::Tensor qkv_bias;
    torch::Tensor attn_out_weight;
    torch::Tensor attn_out_bias;
    torch::Tensor ln2_gamma;
    torch::Tensor ln2_beta;
    torch::Tensor ff1_weight;
    torch::Tensor ff1_bias;
    torch::Tensor ff2_weight;
    torch::Tensor ff2_bias;
};

struct PieceTransformerLibTorch {
    fs::path weight_dir;
    torch::Device device;
    torch::ScalarType dtype = torch::kFloat16;
    std::string dtype_suffix;
    std::uint32_t state_len = 0;
    std::uint32_t num_classes = 0;
    std::uint32_t move_count = 0;
    std::uint32_t output_dim = 0;
    std::uint32_t num_pieces = 0;
    std::uint32_t max_piece_size = 0;
    std::uint32_t seq_len = 0;
    std::uint32_t d_model = 0;
    std::uint32_t nhead = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t num_layers = 0;
    std::uint32_t ff_dim = 0;

    torch::Tensor fast_slot_projected;
    torch::Tensor fast_piece_static;
    torch::Tensor cls_token;
    torch::Tensor input_ln_gamma;
    torch::Tensor input_ln_beta;
    torch::Tensor output_ln_gamma;
    torch::Tensor output_ln_beta;
    torch::Tensor output_weight;
    torch::Tensor output_bias;
    torch::Tensor piece_positions;
    torch::Tensor piece_mask;
    std::vector<torch::Tensor> slot_active_piece_indices;
    std::vector<torch::Tensor> slot_active_positions;
    std::vector<TransformerBlock> blocks;

    PieceTransformerLibTorch(fs::path dir, torch::Device target_device)
        : weight_dir(std::move(dir)), device(std::move(target_device)) {
        const std::string manifest = read_text_exact(weight_dir / "manifest.json");
        if (manifest_string(manifest, "backend") != "piece_transformer") {
            throw std::runtime_error("manifest backend must be piece_transformer");
        }
        const std::string manifest_dtype = manifest_string(manifest, "dtype");
        if (manifest_dtype == "fp16") {
            dtype = torch::kFloat16;
            dtype_suffix = ".fp16";
        } else if (manifest_dtype == "bf16") {
            dtype = torch::kBFloat16;
            dtype_suffix = ".bf16";
        } else {
            throw std::runtime_error("manifest dtype must be fp16 or bf16");
        }
        state_len = manifest_u32(manifest, "state_len");
        num_classes = manifest_u32(manifest, "num_classes");
        move_count = manifest_u32(manifest, "move_count");
        output_dim = manifest_u32(manifest, "output_dim");
        num_pieces = manifest_u32(manifest, "num_pieces");
        max_piece_size = manifest_u32(manifest, "max_piece_size");
        seq_len = manifest_u32(manifest, "seq_len");
        d_model = manifest_u32(manifest, "d_model");
        nhead = manifest_u32(manifest, "nhead");
        head_dim = manifest_u32(manifest, "head_dim");
        num_layers = manifest_u32_any(manifest, "num_layers", "transformer_layers");
        ff_dim = manifest_u32(manifest, "ff_dim");
        if (seq_len != num_pieces + 1U || d_model != nhead * head_dim || output_dim != move_count) {
            throw std::runtime_error("invalid piece_transformer manifest dimensions");
        }
        if (manifest_string(manifest, "activation") != "silu" ||
            manifest_string(manifest, "pooling") != "cls" ||
            manifest_string(manifest, "piece_layout") != "p900" ||
            manifest_string(manifest, "piece_embed_mode") != "full_s120") {
            throw std::runtime_error("unsupported piece_transformer manifest contract");
        }

        fast_slot_projected = load_tensor(
            weight_dir / ("fast_slot_projected" + dtype_suffix),
            {max_piece_size, num_classes, d_model},
            dtype,
            device);
        fast_piece_static =
            load_tensor(weight_dir / ("fast_piece_static" + dtype_suffix), {num_pieces, d_model}, dtype, device);
        cls_token = load_tensor(weight_dir / ("cls_token" + dtype_suffix), {d_model}, dtype, device);
        input_ln_gamma = load_tensor(weight_dir / ("input_ln_gamma" + dtype_suffix), {d_model}, dtype, device);
        input_ln_beta = load_tensor(weight_dir / ("input_ln_beta" + dtype_suffix), {d_model}, dtype, device);
        output_ln_gamma = load_tensor(weight_dir / ("output_ln_gamma" + dtype_suffix), {d_model}, dtype, device);
        output_ln_beta = load_tensor(weight_dir / ("output_ln_beta" + dtype_suffix), {d_model}, dtype, device);
        output_weight = load_tensor(weight_dir / ("output_weight_hxk" + dtype_suffix), {d_model, output_dim}, dtype, device);
        output_bias = load_tensor(weight_dir / ("output_bias" + dtype_suffix), {output_dim}, dtype, device);
        piece_positions =
            load_tensor(weight_dir / "piece_positions.u16", {num_pieces, max_piece_size}, torch::kInt16, device)
                .to(torch::kLong);
        piece_mask = load_tensor(weight_dir / "piece_mask.u8", {num_pieces, max_piece_size}, torch::kUInt8, device)
                         .to(torch::kBool);
        slot_active_piece_indices.reserve(max_piece_size);
        slot_active_positions.reserve(max_piece_size);
        for (std::uint32_t slot = 0; slot < max_piece_size; ++slot) {
            torch::Tensor active =
                torch::nonzero(piece_mask.index({torch::indexing::Slice(), static_cast<std::int64_t>(slot)}))
                    .flatten()
                    .to(torch::kLong);
            slot_active_piece_indices.push_back(active);
            if (active.numel() == 0) {
                slot_active_positions.push_back(
                    torch::empty({0}, torch::TensorOptions().dtype(torch::kLong).device(device)));
            } else {
                slot_active_positions.push_back(
                    piece_positions.index({active, static_cast<std::int64_t>(slot)}).to(torch::kLong));
            }
        }

        blocks.reserve(num_layers);
        for (std::uint32_t i = 0; i < num_layers; ++i) {
            const std::string prefix = "block" + std::to_string(i);
            blocks.push_back(TransformerBlock{
                load_tensor(weight_dir / (prefix + "_ln1_gamma" + dtype_suffix), {d_model}, dtype, device),
                load_tensor(weight_dir / (prefix + "_ln1_beta" + dtype_suffix), {d_model}, dtype, device),
                load_tensor(weight_dir / (prefix + "_attn_qkv_weight_hxk" + dtype_suffix), {d_model, 3U * d_model}, dtype, device),
                load_tensor(weight_dir / (prefix + "_attn_qkv_bias" + dtype_suffix), {3U * d_model}, dtype, device),
                load_tensor(weight_dir / (prefix + "_attn_out_weight_hxk" + dtype_suffix), {d_model, d_model}, dtype, device),
                load_tensor(weight_dir / (prefix + "_attn_out_bias" + dtype_suffix), {d_model}, dtype, device),
                load_tensor(weight_dir / (prefix + "_ln2_gamma" + dtype_suffix), {d_model}, dtype, device),
                load_tensor(weight_dir / (prefix + "_ln2_beta" + dtype_suffix), {d_model}, dtype, device),
                load_tensor(weight_dir / (prefix + "_ff1_weight_hxk" + dtype_suffix), {d_model, ff_dim}, dtype, device),
                load_tensor(weight_dir / (prefix + "_ff1_bias" + dtype_suffix), {ff_dim}, dtype, device),
                load_tensor(weight_dir / (prefix + "_ff2_weight_hxk" + dtype_suffix), {ff_dim, d_model}, dtype, device),
                load_tensor(weight_dir / (prefix + "_ff2_bias" + dtype_suffix), {d_model}, dtype, device),
            });
        }
    }

    torch::Tensor layer_norm(const torch::Tensor& x, const torch::Tensor& gamma, const torch::Tensor& beta) const {
        return torch::layer_norm(x, {static_cast<std::int64_t>(d_model)}, gamma, beta, 1.0e-5, false);
    }

    torch::Tensor build_tokens(const torch::Tensor& state_u8) const {
        const torch::Tensor states = state_u8.index({torch::indexing::Slice(), torch::indexing::Slice(0, state_len)})
                                         .to(torch::kLong);
        const std::int64_t batch = states.size(0);
        torch::Tensor pieces = fast_piece_static.unsqueeze(0).expand({batch, -1, -1}).clone();
        for (std::uint32_t slot = 0; slot < max_piece_size; ++slot) {
            const torch::Tensor& active = slot_active_piece_indices[slot];
            const std::int64_t active_count = active.numel();
            if (active_count == 0) {
                continue;
            }
            torch::Tensor state_values = states.index_select(1, slot_active_positions[slot]).reshape({-1});
            torch::Tensor gathered = fast_slot_projected.index({static_cast<std::int64_t>(slot)})
                                         .index_select(0, state_values)
                                         .view({batch, active_count, static_cast<std::int64_t>(d_model)});
            torch::Tensor current = pieces.index({torch::indexing::Slice(), active, torch::indexing::Slice()});
            pieces.index_put_({torch::indexing::Slice(), active, torch::indexing::Slice()}, current + gathered);
        }
        torch::Tensor cls = cls_token.view({1, 1, static_cast<std::int64_t>(d_model)}).expand({batch, 1, -1});
        return torch::cat({cls, pieces}, 1);
    }

    torch::Tensor forward(const torch::Tensor& state_u8) const {
        const std::int64_t batch = state_u8.size(0);
        torch::Tensor x = layer_norm(build_tokens(state_u8), input_ln_gamma, input_ln_beta);
        for (const TransformerBlock& block : blocks) {
            torch::Tensor y = layer_norm(x, block.ln1_gamma, block.ln1_beta);
            torch::Tensor qkv = torch::matmul(y, block.qkv_weight) + block.qkv_bias;
            qkv = qkv.view({batch,
                            static_cast<std::int64_t>(seq_len),
                            3,
                            static_cast<std::int64_t>(nhead),
                            static_cast<std::int64_t>(head_dim)});
            torch::Tensor q = qkv.select(2, 0).permute({0, 2, 1, 3}).contiguous();
            torch::Tensor k = qkv.select(2, 1).permute({0, 2, 1, 3}).contiguous();
            torch::Tensor v = qkv.select(2, 2).permute({0, 2, 1, 3}).contiguous();
            torch::Tensor attn = at::scaled_dot_product_attention(q, k, v, std::nullopt, 0.0, false, std::nullopt, false);
            torch::Tensor context = attn.permute({0, 2, 1, 3})
                                        .reshape({batch,
                                                  static_cast<std::int64_t>(seq_len),
                                                  static_cast<std::int64_t>(d_model)});
            x = x + torch::matmul(context, block.attn_out_weight) + block.attn_out_bias;

            y = layer_norm(x, block.ln2_gamma, block.ln2_beta);
            y = torch::matmul(y, block.ff1_weight) + block.ff1_bias;
            y = y * torch::sigmoid(y);
            x = x + torch::matmul(y, block.ff2_weight) + block.ff2_bias;
        }
        torch::Tensor cls = layer_norm(x.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}), output_ln_gamma, output_ln_beta);
        return torch::matmul(cls, output_weight) + output_bias;
    }
};


inline void synchronize_if_cuda(const torch::Device& device) {
    if (device.is_cuda()) {
        torch::cuda::synchronize(device.index());
    }
}

} // namespace beam::stream1_libtorch
