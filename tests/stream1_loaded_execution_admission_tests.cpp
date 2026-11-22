#include "../cuda/stream1_loaded_execution_admission.hpp"
#include "../cuda/stream1_transformer_policy_snapshot.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace beam;
static void policy(const char* key,const char* value) {
#ifdef _WIN32
    const int status=_putenv_s(key,value);
#else
    const int status=setenv(key,value,1);
#endif
    if(status) throw std::runtime_error("cannot configure admission fixture");
}
template<class Fn> void reject(Fn fn) {
    bool bad=false;
    try { fn(); } catch(const std::invalid_argument&) {bad=true;}
    if(!bad) throw std::runtime_error("unsupported loaded execution admitted");
}
int main() {
    Stream1TransformerBlockView blocks[4]{};
    Stream1TransformerNetworkView v{};
    v.blocks=blocks;
    v.dims={96,6,56,3,57,64,16,256,8,32,4,1024,24,STREAM1_DTYPE_FP16,STREAM1_ACTIVATION_RELU};
    auto p=capture_stream1_transformer_launch_policies();
    auto check=[&](int sm=86, bool compiled=false, unsigned outer=32, unsigned inner=13, unsigned lanes=2) {
        validate_stream1_cube4_loaded_execution(v,sm,compiled,outer,inner,lanes,p);
    };
    check();
    for(const auto* key:{"BEAM_STREAM1_TRANSFORMER_ATTENTION_TILE_POLICY",
                        "BEAM_STREAM1_TRANSFORMER_ATTENTION_MAX_K_POLICY",
                        "BEAM_STREAM1_TRANSFORMER_CLS_ATTENTION_POLICY",
                        "BEAM_STREAM1_TRANSFORMER_LAYERNORM_ROWS_POLICY",
                        "BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM"}) {
        policy(key,"unsupported");p=capture_stream1_transformer_launch_policies();
        reject([&]{check();});
        policy(key,"");p=capture_stream1_transformer_launch_policies();check();
    }
    policy("BEAM_STREAM1_TRANSFORMER_LAYERNORM_ROWS_POLICY","block2");
    p=capture_stream1_transformer_launch_policies();reject([&]{check();});
    policy("BEAM_STREAM1_TRANSFORMER_LAYERNORM_ROWS_POLICY","row");
    p=capture_stream1_transformer_launch_policies();check();
    reject([&]{check(74);});
    v.dims.dtype=STREAM1_DTYPE_BF16; check(); reject([&]{check(75);});
    v.dims.dtype=7; reject([&]{check();}); v.dims.dtype=STREAM1_DTYPE_FP16;
    v.dims.activation=STREAM1_ACTIVATION_SILU; reject([&]{check();}); v.dims.activation=STREAM1_ACTIVATION_RELU;
    v.dims.head_dim=64; reject([&]{check();}); v.dims.head_dim=32;
    v.dims.padded_seq_len=63; reject([&]{check();}); v.dims.padded_seq_len=64;
    v.dims.sequence_alignment=1; v.dims.padded_seq_len=57; check();
    reject([&]{check(86,false,0);}); reject([&]{check(86,false,32,33);});
    reject([&]{check(86,false,32,13,0);});
    reject([&]{check(86,false,std::numeric_limits<unsigned>::max(),std::numeric_limits<unsigned>::max());});
    const auto* saved=v.blocks; v.blocks=nullptr; reject([&]{check();}); v.blocks=saved;
    blocks[0].qkv_e4m3_scale=-1; reject([&]{check();});
    blocks[0].qkv_e4m3_scale=std::numeric_limits<float>::quiet_NaN(); reject([&]{check();});
    blocks[0].qkv_e4m3_scale=1; reject([&]{check();});
    blocks[0].qkv_e4m3_scale=0;
    blocks[0].qkv_hopper_fp16=true; reject([&]{check();}); reject([&]{check(90,false);});
    // Packed routes require the final-CLS flags, not just a compatible SM.
    reject([&]{check(90,true);});
    blocks[0].qkv_hopper_fp16=false;
    policy("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY","1");
    policy("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION","1");
    policy("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV","1");
    policy("BEAM_STREAM1_TRANSFORMER_DUAL_INPUT_LN","0");
    p=capture_stream1_transformer_launch_policies();
    blocks[0].qkv_hopper_fp16=true; check(90,true);
    blocks[0].qkv_e4m3_scale=1; reject([&]{check(90,true);});
    blocks[0].qkv_hopper_fp16=false; check(90,true);
    blocks[0].qkv_e4m3_scale=0;
    blocks[3].ff1_hopper_fp16=true; reject([&]{check(90,true);});
    blocks[3].ff1_hopper_fp16=false;
    policy("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY","0");
    p=capture_stream1_transformer_launch_policies();
    reject([&]{check();}); // Attention/split cannot remain enabled without CLS.
    std::cout << "loaded_execution_admission=pass\n";
}
