#include "../tools/stream1_weight_io.hpp"
#include <iostream>
#include <cstring>
int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("artifact required");
    for(const char* flag:{"BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY","BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION","BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV","BEAM_STREAM1_TRANSFORMER_COMPACT57"})setenv(flag,"1",1);
    setenv("BEAM_STREAM1_TRANSFORMER_HOPPER_FF2","fp16_tma",1);
    auto h=beam::stream1_weights::load_stream1_weights(argv[1]);
    auto d=beam::stream1_weights::upload_weights(h);
    auto view=beam::stream1_weights::make_transformer_network_view(d.transformer,h.model);
    for(size_t l=0;l<h.transformer.blocks.size();++l){
        bool packed=l+1<h.transformer.blocks.size();
        if(view.blocks[l].ff2_hopper_fp16!=packed)throw std::runtime_error("FF2 layout flag missing or wrong");
        const auto& original=h.transformer.blocks[l].ff2_weight;
        std::vector<std::byte> actual(original.size());
        BEAM_CUDA_CHECK(cudaMemcpy(actual.data(),d.transformer.blocks[l].ff2_weight,actual.size(),cudaMemcpyDeviceToHost));
        for(unsigned k=0;k<h.model.ff_dim;++k)for(unsigned n=0;n<h.model.d_model;++n){
            size_t src=(k*h.model.d_model+n)*2;
            size_t dst=(packed?n*h.model.ff_dim+k:k*h.model.d_model+n)*2;
            if(std::memcmp(original.data()+src,actual.data()+dst,2))throw std::runtime_error("FF2 upload transpose mismatch");
        }
    }
    beam::stream1_weights::free_weights(d);
    std::cout<<"PASS one FF2 representation, layout bytes and view flags, final CLS unchanged\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
