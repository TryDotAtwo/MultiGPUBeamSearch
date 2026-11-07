// Captured typed Params mutations only, not trained-model numerical acceptance.
#include "../tools/graph_attention_members.hpp"
#include <iostream>
int main(int argc,char** argv) {
    try {
        BEAM_CUDA_CHECK(cudaSetDevice(argc>1?std::stoi(argv[1]):0));
        using A=AttentionKernel<cutlass::half_t,cutlass::arch::Sm80,true,64,64,64,false,false>;
        using P=typename A::Params;
        unsigned char* buffer=nullptr;BEAM_CUDA_CHECK(cudaMalloc(&buffer,32768));
        P p{};
        p.query_ptr=reinterpret_cast<cutlass::half_t*>(buffer);
        p.key_ptr=reinterpret_cast<cutlass::half_t*>(buffer+512);
        p.value_ptr=reinterpret_cast<cutlass::half_t*>(buffer+1024);
        p.output_ptr=reinterpret_cast<cutlass::half_t*>(buffer+16384);
        p.q_strideB=43776;p.scale=0.1767766952966369f;
        void* args[]={&p};cudaKernelNodeParams parameters{};
        parameters.func=reinterpret_cast<void*>(attention_kernel_batched_impl<A>);
        parameters.gridDim=dim3(1,8,1);parameters.blockDim=dim3(32,A::kNumWarpsPerBlock,1);
        parameters.sharedMemBytes=sizeof(A::SharedStorage);parameters.kernelParams=args;
        cudaGraph_t graph{};cudaGraphNode_t node{};
        BEAM_CUDA_CHECK(cudaGraphCreate(&graph,0));
        BEAM_CUDA_CHECK(cudaGraphAddKernelNode(&node,graph,nullptr,0,&parameters));
        const char* name=nullptr;BEAM_CUDA_CHECK(cudaFuncGetName(&name,parameters.func));
        beam::score_mode::GraphPointerRegistry registry;
        registry.add("lane_qkv",buffer,16384);registry.add("lane_context",buffer+16384,16384);
        auto extract=[&]() {
            cudaKernelNodeParams captured{};BEAM_CUDA_CHECK(cudaGraphKernelNodeGetParams(node,&captured));
            return beam::score_mode::captured_attention_members(captured,name,&registry,0,0);
        };
        auto good=extract();
        if(good["attention_signature_known"]!=true || good["pointers"]["key_ptr"]["offset"]!=512)
            throw std::runtime_error("typed attention extraction mismatch");
        if(good["optional_pointers"].size()!=4 || good["optional_scalars"].size()!=8 ||
           good["pytorch_rng_state_present"]!=false)
            throw std::runtime_error("optional Params metadata missing");
        auto check_scalar=[&](const char* field) {
            BEAM_CUDA_CHECK(cudaGraphKernelNodeSetParams(node,&parameters));
            if(extract()["optional_scalars"][field]==0)
                throw std::runtime_error("optional scalar mutation hidden");
        };
        p.causal_diagonal_offset=1;check_scalar("causal_diagonal_offset");p.causal_diagonal_offset=0;
        p.num_keys_absolute=1;check_scalar("num_keys_absolute");p.num_keys_absolute=0;
        p.bias_strideM=1;check_scalar("bias_strideM");p.bias_strideM=0;
        p.bias_strideH=1;check_scalar("bias_strideH");p.bias_strideH=0;
        p.bias_strideB=1;check_scalar("bias_strideB");p.bias_strideB=0;
        p.use_dropout=true;
        BEAM_CUDA_CHECK(cudaGraphKernelNodeSetParams(node,&parameters));
        if(extract()["optional_scalars"]["use_dropout"]!=true)
            throw std::runtime_error("dropout bool mutation hidden");
        p.use_dropout=false;
        p.dropout_batch_head_rng_offset=1;check_scalar("dropout_batch_head_rng_offset");p.dropout_batch_head_rng_offset=0;
        p.dropout_prob=0.5f;check_scalar("dropout_prob");p.dropout_prob=0;
        auto check_pointer=[&](const char* field) {
            BEAM_CUDA_CHECK(cudaGraphKernelNodeSetParams(node,&parameters));
            if(extract()["optional_pointers"][field]["role"]!="lane_qkv")
                throw std::runtime_error("optional pointer mutation hidden");
        };
        p.attn_bias_ptr=reinterpret_cast<cutlass::half_t*>(buffer);check_pointer("attn_bias_ptr");p.attn_bias_ptr=nullptr;
        p.seqstart_q_ptr=reinterpret_cast<int32_t*>(buffer);check_pointer("seqstart_q_ptr");p.seqstart_q_ptr=nullptr;
        p.seqstart_k_ptr=reinterpret_cast<int32_t*>(buffer);check_pointer("seqstart_k_ptr");p.seqstart_k_ptr=nullptr;
        p.seqlen_k_ptr=reinterpret_cast<int32_t*>(buffer);check_pointer("seqlen_k_ptr");p.seqlen_k_ptr=nullptr;
        p.q_strideB=768;BEAM_CUDA_CHECK(cudaGraphKernelNodeSetParams(node,&parameters));
        if(extract()["scalars"]["q_strideB"]!=768)throw std::runtime_error("stride mutation hidden");
        p.query_ptr=reinterpret_cast<cutlass::half_t*>(buffer+16384);
        BEAM_CUDA_CHECK(cudaGraphKernelNodeSetParams(node,&parameters));
        if(extract()["pointers"]["query_ptr"]["role"]!="lane_context")throw std::runtime_error("role mutation hidden");
        p.value_ptr=reinterpret_cast<cutlass::half_t*>(buffer+16383);
        BEAM_CUDA_CHECK(cudaGraphKernelNodeSetParams(node,&parameters));
        if(extract()["pointers"]["value_ptr"]["available_bytes"]!=1)throw std::runtime_error("short span hidden");
        p.key_ptr=reinterpret_cast<cutlass::half_t*>(buffer+32768);
        BEAM_CUDA_CHECK(cudaGraphKernelNodeSetParams(node,&parameters));
        bool rejected=false;try{extract();}catch(const std::runtime_error&){rejected=true;}
        if(!rejected)throw std::runtime_error("foreign pointer accepted");
        BEAM_CUDA_CHECK(cudaGraphDestroy(graph));BEAM_CUDA_CHECK(cudaFree(buffer));
        std::cout<<"PASS captured attention stride/role/short mutations preserved; foreign rejected\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
