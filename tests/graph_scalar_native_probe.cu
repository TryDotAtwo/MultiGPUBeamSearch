// Signature extraction probe, not a trained-model or LayerNorm quality test.
#include "../tools/stream1_graph_inventory.hpp"
#include <iostream>
namespace beam {
__global__ void stream1_transformer_layernorm256_copy_kernel(
    const float* input,float* output,const float* gamma,const float* beta,
    std::uint32_t rows,std::uint32_t dtype) {
    if(threadIdx.x==0 && output)output[0]=static_cast<float>(rows+dtype);
}
}
int main(int argc,char** argv) {
    try {
        BEAM_CUDA_CHECK(cudaSetDevice(argc>1?std::stoi(argv[1]):0));
        float* output=nullptr;BEAM_CUDA_CHECK(cudaMalloc(&output,sizeof(float)));
        cudaStream_t stream{};cudaGraph_t graph{};
        BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        BEAM_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        beam::stream1_transformer_layernorm256_copy_kernel<<<1,128,0,stream>>>(nullptr,output,nullptr,nullptr,13,0);
        beam::stream1_transformer_layernorm256_copy_kernel<<<1,128,0,stream>>>(nullptr,output,nullptr,nullptr,6,1);
        BEAM_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
        beam::score_mode::GraphKernelInventory observed;observed.observe(graph,0);
        const auto values=observed.u32_values_payload();
        const auto& nodes=values.at("lanes").at(0).at("kernels");
        if(nodes.size()!=2)throw std::runtime_error("scalar node coverage missing");
        bool first=false,second=false;
        for(const auto& node:nodes) {
            const auto& scalars=node.at("u32_values");
            if(scalars.size()!=2 || scalars.at(0).at("index")!=4 || scalars.at(1).at("index")!=5)
                throw std::runtime_error("scalar argument index mismatch");
            first=first || (scalars.at(0).at("value")==13 && scalars.at(1).at("value")==0);
            second=second || (scalars.at(0).at("value")==6 && scalars.at(1).at("value")==1);
        }
        if(!first || !second)throw std::runtime_error("captured scalar values differ");
        std::cout<<values.dump()<<'\n';
        BEAM_CUDA_CHECK(cudaGraphDestroy(graph));
        BEAM_CUDA_CHECK(cudaStreamDestroy(stream));BEAM_CUDA_CHECK(cudaFree(output));
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
