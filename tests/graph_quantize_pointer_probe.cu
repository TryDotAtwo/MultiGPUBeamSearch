// Synthetic ABI probe, not score quantization quality.
#include "../tools/graph_pointer_inventory.hpp"
#include <iostream>
namespace beam {
struct Stream1TransformerDims {std::uint32_t values[15];};
__global__ void stream1_transformer_score_quantize_graph_job_kernel(
    const float* logits,const float* bias,const std::uint32_t* count,
    const std::uint32_t* job,std::uint32_t* scores,std::uint32_t batch,
    std::uint32_t outer,std::uint32_t offset,Stream1TransformerDims dims,
    std::uint32_t* error) {
    if(threadIdx.x==0){scores[0]=count[*job]+batch+outer+offset;*error=0;}
}
}
int main(int argc,char** argv) {
    try {
        BEAM_CUDA_CHECK(cudaSetDevice(argc>1?std::stoi(argv[1]):0));
        unsigned char* allocation=nullptr;BEAM_CUDA_CHECK(cudaMalloc(&allocation,128));
        BEAM_CUDA_CHECK(cudaMemset(allocation,0,128));
        cudaStream_t stream{};cudaGraph_t graph{};cudaGraphExec_t exec{};
        BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        BEAM_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        beam::stream1_transformer_score_quantize_graph_job_kernel<<<1,256,0,stream>>>(
            reinterpret_cast<float*>(allocation),reinterpret_cast<float*>(allocation+16),
            reinterpret_cast<std::uint32_t*>(allocation+32),reinterpret_cast<std::uint32_t*>(allocation+48),
            reinterpret_cast<std::uint32_t*>(allocation+64),13,32,26,{},reinterpret_cast<std::uint32_t*>(allocation+80));
        BEAM_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
        beam::score_mode::GraphPointerRegistry registry;
        const char* roles[]={"logits","bias","count","job","scores","error"};
        for(unsigned i=0;i<6;++i)registry.add(roles[i],allocation+16*i,16);
        beam::score_mode::GraphPointerInventory observer;observer.observe(graph,0,registry);
        const auto payload=observer.payload();const auto& pointers=payload.at("lanes").at(0).at("kernels").at(0).at("pointers");
        const unsigned indices[]={0,1,2,3,4,9};
        if(pointers.size()!=6)throw std::runtime_error("quantize pointer coverage missing");
        for(unsigned i=0;i<6;++i)if(pointers.at(i).at("index")!=indices[i] || pointers.at(i).at("role")!=roles[i])
            throw std::runtime_error("quantize pointer identity mismatch");
        BEAM_CUDA_CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        BEAM_CUDA_CHECK(cudaGraphLaunch(exec,stream));BEAM_CUDA_CHECK(cudaStreamSynchronize(stream));
        std::uint32_t score=0;BEAM_CUDA_CHECK(cudaMemcpy(&score,allocation+64,4,cudaMemcpyDeviceToHost));
        if(score!=71)throw std::runtime_error("quantize probe replay mismatch");
        std::cout<<payload.dump()<<'\n';
        BEAM_CUDA_CHECK(cudaGraphExecDestroy(exec));BEAM_CUDA_CHECK(cudaGraphDestroy(graph));
        BEAM_CUDA_CHECK(cudaStreamDestroy(stream));BEAM_CUDA_CHECK(cudaFree(allocation));
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
