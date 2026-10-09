#include "stream1_transformer_attention_policy.hpp"

#include <cassert>
#include <stdexcept>

int main() {
    using beam::Stream1TransformerAttentionTilePolicy;
    using beam::Stream1TransformerClsAttentionPolicy;
    using beam::Stream1TransformerAttentionMaxKPolicy;
    const auto require = [](bool condition) {
        if (!condition) { throw std::runtime_error("attention policy contract failed"); }
    };
    require(beam::parse_stream1_transformer_attention_tile_policy(nullptr) == Stream1TransformerAttentionTilePolicy::Q64K64);
    require(beam::parse_stream1_transformer_attention_tile_policy("") == Stream1TransformerAttentionTilePolicy::Q64K64);
    require(beam::parse_stream1_transformer_attention_tile_policy("q64k64") == Stream1TransformerAttentionTilePolicy::Q64K64);
    require(beam::parse_stream1_transformer_attention_tile_policy("q32k64") == Stream1TransformerAttentionTilePolicy::Q32K64);
    require(beam::parse_stream1_transformer_attention_tile_policy("q64k64v4") == Stream1TransformerAttentionTilePolicy::Q64K64V4);
    bool rejected = false;
    try { (void)beam::parse_stream1_transformer_attention_tile_policy("unknown"); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
    require(beam::stream1_transformer_attention_tile_desc(Stream1TransformerAttentionTilePolicy::Q64K64).warps == 4);
    require(beam::stream1_transformer_attention_tile_desc(Stream1TransformerAttentionTilePolicy::Q32K64).warps == 2);
    const auto q64k64v4 = beam::stream1_transformer_attention_tile_desc(Stream1TransformerAttentionTilePolicy::Q64K64V4);
    require(q64k64v4.queries == 64 && q64k64v4.keys == 64 && q64k64v4.warps == 4 && q64k64v4.alignment_elements == 4);
    require(beam::parse_stream1_transformer_attention_max_k_policy(nullptr) == Stream1TransformerAttentionMaxKPolicy::Padded64);
    require(beam::parse_stream1_transformer_attention_max_k_policy("") == Stream1TransformerAttentionMaxKPolicy::Padded64);
    require(beam::parse_stream1_transformer_attention_max_k_policy("padded64") == Stream1TransformerAttentionMaxKPolicy::Padded64);
    require(beam::parse_stream1_transformer_attention_max_k_policy("exact32") == Stream1TransformerAttentionMaxKPolicy::Exact32);
    require(beam::stream1_transformer_attention_max_k(Stream1TransformerAttentionMaxKPolicy::Padded64) == 64);
    require(beam::stream1_transformer_attention_max_k(Stream1TransformerAttentionMaxKPolicy::Exact32) == 32);
    rejected = false;
    try { (void)beam::parse_stream1_transformer_attention_max_k_policy("unknown"); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
    require(beam::parse_stream1_transformer_cls_attention_policy(nullptr) == Stream1TransformerClsAttentionPolicy::Cutlass);
    require(beam::parse_stream1_transformer_cls_attention_policy("q32k64") == Stream1TransformerClsAttentionPolicy::Q32K64);
    rejected = false;
    try { (void)beam::parse_stream1_transformer_cls_attention_policy("unknown"); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
    return 0;
}