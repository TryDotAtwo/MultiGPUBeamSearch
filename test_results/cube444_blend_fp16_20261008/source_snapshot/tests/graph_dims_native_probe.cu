// Synthetic typed NetworkView dimensions, not trained inference quality.
#include "../tools/stream1_graph_inventory.hpp"
#include "../cuda/stream1.hpp"
#include <iostream>
namespace beam {
template<bool Dual=false>
__global__ void stream1_transformer_build_input_layernorm256_generic_kernel(
    const State128* states,const std::uint64_t* base,const std::uint32_t* count,
    const std::uint32_t* job,Stream1TransformerNetworkView network,half* tokens,
    std::uint32_t batch,std::uint32_t offset,half* normalized,
    const half* gamma,const half* beta) {
    if(threadIdx.x==0)tokens[0]=__float2half_rn(network.dims.seq_len+network.dims.activation);
}
}
int main(int argc,char** argv) {
    try {
        BEAM_CUDA_CHECK(cudaSetDevice(argc>1?std::stoi(argv[1]):0));
        half* output=nullptr;BEAM_CUDA_CHECK(cudaMalloc(&output,sizeof(half)));
        beam::Stream1TransformerNetworkView network{};
        network.dims={96,6,56,3,57,57,1,256,8,32,4,1024,24,0,1};
        cudaStream_t stream{};cudaGraph_t graph{};cudaGraphExec_t exec{};
        BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        BEAM_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        beam::stream1_transformer_build_input_layernorm256_generic_kernel<<<1,128,0,stream>>>(
            nullptr,nullptr,nullptr,nullptr,network,output,13,26,nullptr,nullptr,nullptr);
        BEAM_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
        beam::score_mode::GraphKernelInventory observed;observed.observe(graph,0);
        const auto payload=observed.dims_values_payload();
        const auto& node=payload.at("lanes").at(0).at("kernels").at(0);
        const char* names[]={"state_len","num_classes","num_pieces","max_piece_size","seq_len",
            "padded_seq_len","sequence_alignment","d_model","nhead","head_dim","transformer_layers",
            "ff_dim","output_dim","dtype","activation"};
        const unsigned values[]={96,6,56,3,57,57,1,256,8,32,4,1024,24,0,1};
        if(node.at("dims_signature_known")!=true || node.at("dims_index")!=4)
            throw std::runtime_error("native embedded dimension coverage missing");
        for(unsigned i=0;i<15;++i)if(node.at("dims").at(names[i])!=values[i])
            throw std::runtime_error("native embedded dimension value mismatch");
        BEAM_CUDA_CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        BEAM_CUDA_CHECK(cudaGraphLaunch(exec,stream));BEAM_CUDA_CHECK(cudaStreamSynchronize(stream));
        half result{};BEAM_CUDA_CHECK(cudaMemcpy(&result,output,sizeof(result),cudaMemcpyDeviceToHost));
        if(__half2float(result)!=58)throw std::runtime_error("native dims replay mismatch");
        std::cout<<payload.dump()<<'\n';
        BEAM_CUDA_CHECK(cudaGraphExecDestroy(exec));BEAM_CUDA_CHECK(cudaGraphDestroy(graph));
        BEAM_CUDA_CHECK(cudaStreamDestroy(stream));BEAM_CUDA_CHECK(cudaFree(output));
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
