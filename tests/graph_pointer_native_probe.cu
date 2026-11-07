// Signature extraction only; synthetic kernel is not real LayerNorm quality.
#include "../tools/graph_pointer_inventory.hpp"
#include "../cuda/cuda_check.hpp"
#include <iostream>
namespace beam {
__global__ void stream1_transformer_layernorm256_copy_kernel(
    const float* input,float* output,const float* gamma,const float* beta,
    std::uint32_t rows,std::uint32_t dtype) {
    if(threadIdx.x==0)output[0]=input[0]+gamma[0]+beta[0]+rows+dtype;
}
}
int main(int argc,char** argv) {
    try {
        BEAM_CUDA_CHECK(cudaSetDevice(argc>1?std::stoi(argv[1]):0));
        float* buffer=nullptr;BEAM_CUDA_CHECK(cudaMalloc(&buffer,64));
        BEAM_CUDA_CHECK(cudaMemset(buffer,0,64));
        cudaStream_t stream{};cudaGraph_t graph{};cudaGraphExec_t exec{};
        BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        BEAM_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        beam::stream1_transformer_layernorm256_copy_kernel<<<1,128,0,stream>>>(buffer,buffer+4,buffer+8,buffer+12,13,0);
        BEAM_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
        beam::score_mode::GraphPointerRegistry spans;
        spans.add("input",buffer,16);spans.add("output",buffer+4,16);
        spans.add("gamma",buffer+8,16);spans.add("beta",buffer+12,16);
        beam::score_mode::GraphPointerInventory inventory;inventory.observe(graph,0,spans);
        const auto payload=inventory.payload();
        const auto& pointers=payload.at("lanes").at(0).at("kernels").at(0).at("pointers");
        const char* expected[]={"input","output","gamma","beta"};
        if(pointers.size()!=4)throw std::runtime_error("pointer coverage missing");
        for(unsigned i=0;i<4;++i)
            if(pointers.at(i).at("index")!=i || pointers.at(i).at("role")!=expected[i] || pointers.at(i).at("offset")!=0)
                throw std::runtime_error("wrong pointer identity");
        beam::score_mode::GraphPointerRegistry incomplete;incomplete.add("input",buffer,16);
        bool rejected=false;
        try {beam::score_mode::GraphPointerInventory bad;bad.observe(graph,0,incomplete);}
        catch(const std::runtime_error&) {rejected=true;}
        if(!rejected)throw std::runtime_error("foreign pointer accepted");
        BEAM_CUDA_CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        BEAM_CUDA_CHECK(cudaGraphLaunch(exec,stream));BEAM_CUDA_CHECK(cudaStreamSynchronize(stream));
        float result=0;BEAM_CUDA_CHECK(cudaMemcpy(&result,buffer+4,4,cudaMemcpyDeviceToHost));
        if(result!=13)throw std::runtime_error("replay output mismatch");
        std::cout<<payload.dump()<<'\n';
        BEAM_CUDA_CHECK(cudaGraphExecDestroy(exec));BEAM_CUDA_CHECK(cudaGraphDestroy(graph));
        BEAM_CUDA_CHECK(cudaStreamDestroy(stream));BEAM_CUDA_CHECK(cudaFree(buffer));
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
