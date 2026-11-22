#include "stream1_transformer_dual_ln.cuh"
#include "../tools/stream1_weight_io.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
namespace beam {
void stream1_transformer_layernorm_copy_launch(const half*,half*,const half*,const half*,std::uint32_t,std::uint32_t,std::uint32_t,cudaStream_t);
}
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Device {
    T* p=nullptr; std::size_t n;
    explicit Device(std::size_t n):n(n){check(cudaMalloc(&p,n*sizeof(T)));}
    ~Device(){cudaFree(p);}
    void put(const std::vector<T>& x){check(cudaMemcpy(p,x.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T>x(n);check(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return x;}
};
static void run(int batch,bool graph_job) {
    using namespace beam;
    constexpr int D=256,S=57;
    Device<State128> states(batch+8);std::vector<State128> hs(batch+8);
    for(int r=0;r<batch+8;++r)for(int k=0;k<96;++k)hs[r].v[k]=(r*3+k*5)%6;
    states.put(hs);
    Device<std::uint64_t> base(2);base.put({0,2});
    Device<std::uint32_t> count(2),job(1);count.put({std::uint32_t(batch-1),std::uint32_t(batch)});job.put({1});
    Device<std::uint16_t> positions(56);std::vector<std::uint16_t> hp(56);for(int p=0;p<56;++p)hp[p]=p;positions.put(hp);
    Device<std::uint8_t> mask(56);std::vector<std::uint8_t> hm(56,1);hm[3]=0;mask.put(hm);
    Device<half> table(6*D),stat(56*D),cls(D),gamma(D),beta(D),g1(D),b1(D);
    auto fill=[](auto& buf,int seed,float scale){std::vector<half>x(buf.n);for(int i=0;i<int(x.size());++i)x[i]=__float2half(((i*17+seed)%53-26)*scale);buf.put(x);};
    fill(table,1,.02f);fill(stat,3,.07f);fill(cls,5,.09f);fill(gamma,7,.06f);fill(beta,9,.01f);fill(g1,11,.08f);fill(b1,13,.02f);
    Stream1TransformerBlockView block{};block.ln1_gamma=g1.p;block.ln1_beta=b1.p;
    Stream1TransformerNetworkView net{};net.dims={96,6,56,1,S,S,1,D,8,32,4,1024,24,0,0};
    net.fast_slot_projected=table.p;net.fast_piece_static=stat.p;net.cls_token=cls.p;
    net.input_ln_gamma=gamma.p;net.input_ln_beta=beta.p;net.blocks=&block;
    net.piece_positions=positions.p;net.piece_mask=mask.p;
    Device<half> tokens(batch*S*D),norm(batch*S*D),ref_tokens(batch*S*D),ref_norm(batch*S*D);
    check(cudaMemset(norm.p,0xff,norm.n*sizeof(half)));
    const auto* index=graph_job?job.p:nullptr;
    stream1_transformer_build_input_layernorm256_generic_launch(states.p,base.p,count.p,index,net,ref_tokens.p,batch,1,nullptr);
    stream1_transformer_layernorm_copy_launch(ref_tokens.p,ref_norm.p,g1.p,b1.p,batch*S,D,0,nullptr);
    stream1_transformer_build_input_dual_ln_cuda(states.p,base.p,count.p,index,net,tokens.p,norm.p,batch,1,nullptr);
    check(cudaDeviceSynchronize());
    auto a=ref_tokens.get(),b=tokens.get(),c=ref_norm.get(),d=norm.get();
    if(std::memcmp(a.data(),b.data(),a.size()*sizeof(half)))throw std::runtime_error("persistent input-LN tokens changed");
    if(std::memcmp(c.data(),d.data(),c.size()*sizeof(half)))throw std::runtime_error("second LayerNorm output missing or non-exact");
    std::cout<<"dual_ln exact batch="<<batch<<" graph_job="<<graph_job<<" half_values="<<a.size()*2<<std::endl;
}
static void actual_weights() {
    using namespace beam;
    const char* dir=std::getenv("BEAM_STREAM1_TRANSFORMER_WEIGHTS_DIR");
    if(!dir)return;
    auto host=stream1_weights::load_stream1_weights(dir);
    auto weights=stream1_weights::upload_weights(host);
    auto holder=stream1_weights::transformer_network_view(weights.transformer,host.model);
    auto& net=holder.view;
    const int batch=97,S=net.dims.seq_len,D=net.dims.d_model;
    Device<State128> states(batch);std::vector<State128> hs(batch);
    unsigned rng=20260915;
    for(auto& state:hs)for(int k=0;k<96;++k){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;state.v[k]=rng%6;}
    states.put(hs);
    Device<std::uint64_t> base(1);base.put({0});Device<std::uint32_t> count(1);count.put({unsigned(batch)});
    Device<half> tokens(batch*S*D),norm(batch*S*D),ref_tokens(batch*S*D),ref_norm(batch*S*D);
    stream1_transformer_build_input_layernorm256_generic_launch(states.p,base.p,count.p,nullptr,net,ref_tokens.p,batch,0,nullptr);
    stream1_transformer_layernorm_copy_launch(ref_tokens.p,ref_norm.p,net.blocks[0].ln1_gamma,net.blocks[0].ln1_beta,batch*S,D,0,nullptr);
    stream1_transformer_build_input_dual_ln_cuda(states.p,base.p,count.p,nullptr,net,tokens.p,norm.p,batch,0,nullptr);
    auto a=ref_tokens.get(),b=tokens.get(),c=ref_norm.get(),d=norm.get();
    std::size_t nt=0,nn=0;
    for(std::size_t i=0;i<a.size();++i){nt+=__half2float(a[i])!=__half2float(b[i]);if(__half2float(c[i])!=__half2float(d[i])){if(nn<5)std::cout<<"norm_diff index="<<i<<" expected="<<__half2float(c[i])<<" actual="<<__half2float(d[i])<<std::endl;++nn;}}
    std::cout<<"actual_weights token_mismatch="<<nt<<" normalized_mismatch="<<nn<<std::endl;
    if(nt||nn)throw std::runtime_error("real-weight dual LN is not bitexact");
}
int main(){try{for(int batch:{2,5,17})for(bool graph:{false,true})run(batch,graph);actual_weights();}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}}
