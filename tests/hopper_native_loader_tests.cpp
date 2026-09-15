#define BEAM_STREAM1_WEIGHT_IO_MANIFEST_ONLY
#include "../tools/stream1_weight_io.hpp"
#include <iostream>

int main(int argc,char**argv){try{
    if(argc!=2)throw std::runtime_error("usage: hopper_native_loader_tests artifact");
    unsetenv("BEAM_HOPPER_NATIVE_FP8");
    bool rejected=false;
    try{beam::stream1_weights::load_stream1_weights(argv[1]);}catch(const std::runtime_error&){rejected=true;}
    if(!rejected)throw std::runtime_error("experimental bundle accepted without opt-in");
    setenv("BEAM_HOPPER_NATIVE_FP8","1",1);
    auto weights=beam::stream1_weights::load_stream1_weights(argv[1]);
    bool ffn=beam::stream1_weights::read_text_exact(std::filesystem::path(argv[1])/"manifest.json").find("experimental_v2")!=std::string::npos;
    if(weights.transformer.blocks.size()!=4)throw std::runtime_error("block count");
    for(int i=0;i<4;++i){const auto&b=weights.transformer.blocks[i];
        if(b.attn_qkv_weight.size()!=size_t(256*768*(i==3?2:1)))throw std::runtime_error("wrong QKV byte count");
        if((b.qkv_e4m3_scale>0)!=(i<3))throw std::runtime_error("wrong QKV format selection");
        bool packed_ffn=ffn&&i<3;
        if(b.ff1_weight.size()!=size_t(256*1024*(packed_ffn?1:2)) || b.ff2_weight.size()!=b.ff1_weight.size())throw std::runtime_error("wrong FFN bytes");
        if((b.ff1_e4m3_scale>0)!=packed_ffn || (b.ff2_e4m3_scale>0)!=packed_ffn)throw std::runtime_error("wrong FFN format selection");}
    if(weights.transformer.piece_positions.size()!=56*3*2 || weights.transformer.piece_mask.size()!=56*3 || weights.transformer.piece_types.size()!=56)
        throw std::runtime_error("piece metadata");
    std::cout<<"PASS explicit opt-in, 3 E4M3 + 1 FP16 QKV, all piece metadata, bytes="<<beam::stream1_weights::total_host_weight_bytes(weights)<<std::endl;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
