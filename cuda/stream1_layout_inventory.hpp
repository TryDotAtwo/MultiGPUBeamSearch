#pragma once
#include "stream1.hpp"
#include <cstddef>
#include <ostream>
#include <stdexcept>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>

namespace beam {
// Host block views only. This inventory observes physical FP16 layout flags;
// it does not attest device contents, resolved kernels or numerical acceptance.
inline void write_stream1_layout_inventory(
    std::ostream& out, const Stream1TransformerBlockView* blocks, std::size_t count) {
    if (count != 0 && blocks == nullptr)
        throw std::invalid_argument("missing host transformer blocks for layout inventory");
    out << '[';
    for (std::size_t i = 0; i < count; ++i) {
        if (i) out << ',';
        const auto& block = blocks[i];
        out << "{\"block\":" << i
            << ",\"qkv_packed_fp16\":" << (block.qkv_hopper_fp16 ? "true" : "false")
            << ",\"ff1_packed_fp16\":" << (block.ff1_hopper_fp16 ? "true" : "false")
            << ",\"ff2_packed_fp16\":" << (block.ff2_hopper_fp16 ? "true" : "false") << '}';
    }
    out << ']';
}

// Describe the actual host view, never reconstruct it from environment text.
// Construct privately so invalid scales cannot publish a partial JSON object.
inline void write_stream1_loaded_view_inventory(
    std::ostream& out, const Stream1TransformerNetworkView& view) {
    const auto& d = view.dims;
    if (d.transformer_layers && view.blocks == nullptr)
        throw std::invalid_argument("missing host transformer blocks for loaded view");
    for (std::size_t i = 0; i < d.transformer_layers; ++i) {
        const auto& b = view.blocks[i];
        if (!std::isfinite(b.qkv_e4m3_scale) || !std::isfinite(b.ff1_e4m3_scale) ||
            !std::isfinite(b.ff2_e4m3_scale))
            throw std::invalid_argument("nonfinite scale in loaded transformer view");
    }
    std::ostringstream json;
    json.imbue(std::locale::classic());
    json << std::setprecision(std::numeric_limits<float>::max_digits10);
    json << "{\"state_len\":" << d.state_len << ",\"num_classes\":" << d.num_classes
         << ",\"num_pieces\":" << d.num_pieces << ",\"max_piece_size\":" << d.max_piece_size
         << ",\"seq_len\":" << d.seq_len << ",\"padded_seq_len\":" << d.padded_seq_len
         << ",\"sequence_alignment\":" << d.sequence_alignment << ",\"d_model\":" << d.d_model
         << ",\"nhead\":" << d.nhead << ",\"head_dim\":" << d.head_dim
         << ",\"transformer_layers\":" << d.transformer_layers << ",\"ff_dim\":" << d.ff_dim
         << ",\"output_dim\":" << d.output_dim << ",\"dtype\":" << d.dtype
         << ",\"activation\":" << d.activation << ",\"fp16_layouts\":";
    write_stream1_layout_inventory(json, view.blocks, d.transformer_layers);
    json << ",\"block_scales\":[";
    for (std::size_t i = 0; i < d.transformer_layers; ++i) {
        if (i) json << ',';
        const auto& b = view.blocks[i];
        json << "{\"block\":" << i << ",\"qkv_e4m3_scale\":" << b.qkv_e4m3_scale
             << ",\"ff1_e4m3_scale\":" << b.ff1_e4m3_scale
             << ",\"ff2_e4m3_scale\":" << b.ff2_e4m3_scale << '}';
    }
    json << "]}";
    out << json.str();
}
} // namespace beam
