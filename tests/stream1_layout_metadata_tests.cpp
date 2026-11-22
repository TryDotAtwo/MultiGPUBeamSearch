#include "stream1.hpp"
#include "stream1_layout_inventory.hpp"
#include <cassert>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <sstream>
#include <stdexcept>
#include <limits>

int main() {
    using Block = beam::Stream1TransformerBlockView;
    static_assert(std::is_standard_layout_v<Block>);
    static_assert(sizeof(void*) == 8, "native host contract requires64-bit pointers");
    // Prior layout:12 pointers,3 floats,1 bool,padding to8-byte alignment.
    static_assert(sizeof(Block) == 112);
    static_assert(alignof(Block) == 8);
    static_assert(offsetof(Block, ff2_hopper_fp16) == 108);
    static_assert(offsetof(Block, qkv_hopper_fp16) == 109);
    static_assert(offsetof(Block, ff1_hopper_fp16) == 110);
    Block block{};
    assert(!block.ff2_hopper_fp16 && !block.qkv_hopper_fp16 && !block.ff1_hopper_fp16);
    block.qkv_hopper_fp16 = true;
    block.ff2_hopper_fp16 = true;
    std::ostringstream inventory;
    beam::write_stream1_layout_inventory(inventory, &block, 1);
    assert(inventory.str() == "[{\"block\":0,\"qkv_packed_fp16\":true,\"ff1_packed_fp16\":false,\"ff2_packed_fp16\":true}]");
    bool rejected = false;
    try { beam::write_stream1_layout_inventory(inventory, nullptr, 1); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    beam::Stream1TransformerNetworkView network{};
    network.blocks = &block;
    network.dims.transformer_layers = 1;
    network.dims.state_len = 96;
    network.dims.seq_len = 57;
    network.dims.padded_seq_len = 64;
    network.dims.d_model = 256;
    network.dims.ff_dim = 1024;
    network.dims.dtype = 1;
    network.dims.activation = 2;
    block.qkv_e4m3_scale = 0.5f;
    block.ff1_e4m3_scale = 0.25f;
    block.ff2_e4m3_scale = 0.125f;
    std::ostringstream loaded;
    beam::write_stream1_loaded_view_inventory(loaded, network);
    assert(loaded.str().find("\"state_len\":96") != std::string::npos);
    assert(loaded.str().find("\"seq_len\":57,\"padded_seq_len\":64") != std::string::npos);
    assert(loaded.str().find("\"dtype\":1,\"activation\":2") != std::string::npos);
    assert(loaded.str().find("\"qkv_e4m3_scale\":0.5,\"ff1_e4m3_scale\":0.25,\"ff2_e4m3_scale\":0.125") != std::string::npos);
    block.qkv_e4m3_scale = std::nextafter(1.f, 2.f);
    std::ostringstream precise;
    precise << std::hex << std::fixed << std::setprecision(1);
    beam::write_stream1_loaded_view_inventory(precise, network);
    const std::string scale_key = "\"qkv_e4m3_scale\":";
    const auto scale_at = precise.str().find(scale_key);
    assert(scale_at != std::string::npos);
    assert(std::stof(precise.str().substr(scale_at + scale_key.size())) == block.qkv_e4m3_scale);
    assert(precise.str().find("\"state_len\":96") != std::string::npos);
    block.qkv_e4m3_scale = std::numeric_limits<float>::infinity();
    std::ostringstream invalid;
    rejected = false;
    try { beam::write_stream1_loaded_view_inventory(invalid, network); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected && invalid.str().empty());
    std::cout << "block_view_bytes=112 alignment=8 new_flags_use_existing_padding=true\n";
}
