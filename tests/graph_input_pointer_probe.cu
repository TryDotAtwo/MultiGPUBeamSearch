// Synthetic input pointer ABI/null probe, not trained-model numerical quality.
#include "../tools/graph_pointer_inventory.hpp"
#include <iostream>
namespace beam {
struct ProbeNetwork {std::uint32_t values[15];};
template<bool Dual=false>
__global__ void stream1_transformer_build_input_layernorm256_generic_kernel(
    const float* states,const std::uint64_t* base,const std::uint32_t* count,
    const std::uint32_t* job,ProbeNetwork network,float* tokens,
    std::uint32_t batch,std::uint32_t offset,float* normalized,
    const float* gamma,const float* beta) {
    if(threadIdx.x==0){tokens[0]=batch+offset+count[*job]+base[*job];
        if constexpr(Dual)normalized[0]=tokens[0]+gamma[0]+beta[0];}
}
}
int main(int argc,char** argv) {
    try {
        BEAM_CUDA_CHECK(cudaSetDevice(argc>1?std::stoi(argv[1]):0));
        unsigned char* buffer=nullptr;BEAM_CUDA_CHECK(cudaMalloc(&buffer,128));
        BEAM_CUDA_CHECK(cudaMemset(buffer,0,128));
        cudaStream_t stream{};cudaGraph_t graph{};cudaGraphExec_t exec{};
        BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        BEAM_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        beam::stream1_transformer_build_input_layernorm256_generic_kernel<<<1,128,0,stream>>>(
            reinterpret_cast<float*>(buffer),reinterpret_cast<std::uint64_t*>(buffer+16),
            reinterpret_cast<std::uint32_t*>(buffer+32),reinterpret_cast<std::uint32_t*>(buffer+48),
            {},reinterpret_cast<float*>(buffer+64),13,26,nullptr,nullptr,nullptr);
        beam::stream1_transformer_build_input_layernorm256_generic_kernel<true><<<1,128,0,stream>>>(
            reinterpret_cast<float*>(buffer),reinterpret_cast<std::uint64_t*>(buffer+16),
            reinterpret_cast<std::uint32_t*>(buffer+32),reinterpret_cast<std::uint32_t*>(buffer+48),
            {},reinterpret_cast<float*>(buffer+64),13,26,reinterpret_cast<float*>(buffer+80),
            reinterpret_cast<float*>(buffer+96),reinterpret_cast<float*>(buffer+112));
        BEAM_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
        beam::score_mode::GraphPointerRegistry registry;
        const char* roles[]={"states","base","count","job","tokens","normalized","gamma","beta"};
        for(unsigned i=0;i<8;++i)registry.add(roles[i],buffer+16*i,16);
        beam::score_mode::GraphPointerInventory observer;observer.observe(graph,0,registry);
        const auto payload=observer.payload();
        const auto& nodes=payload.at("lanes").at(0).at("kernels");
        if(nodes.size()!=2)throw std::runtime_error("input template coverage missing");
        const unsigned indices[]={0,1,2,3,5,8,9,10};
        for(const auto& node:nodes){
        const auto& pointers=node.at("pointers");
        if(pointers.size()!=8)throw std::runtime_error("input pointer coverage missing");
        const bool dual=node.at("mangled_name").get<std::string>().find("ILb1E")!=std::string::npos;
        for(unsigned i=0;i<8;++i){
            if(pointers.at(i).at("index")!=indices[i])throw std::runtime_error("input pointer index mismatch");
            if(i<5 || dual){if(pointers.at(i).at("role")!=roles[i])throw std::runtime_error("input pointer role mismatch");}
            else if(pointers.at(i).at("role")!="null" || pointers.at(i).at("available_bytes")!=0 || pointers.at(i).at("offset")!=0)
                throw std::runtime_error("input optional null mismatch");
        }
        }
        BEAM_CUDA_CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        BEAM_CUDA_CHECK(cudaGraphLaunch(exec,stream));BEAM_CUDA_CHECK(cudaStreamSynchronize(stream));
        float result=0;BEAM_CUDA_CHECK(cudaMemcpy(&result,buffer+64,4,cudaMemcpyDeviceToHost));
        if(result!=39)throw std::runtime_error("input probe replay mismatch");
        BEAM_CUDA_CHECK(cudaMemcpy(&result,buffer+80,4,cudaMemcpyDeviceToHost));
        if(result!=39)throw std::runtime_error("dual input probe replay mismatch");
        cudaGraph_t invalid{};
        BEAM_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        beam::stream1_transformer_build_input_layernorm256_generic_kernel<true><<<1,128,0,stream>>>(
            reinterpret_cast<float*>(buffer),reinterpret_cast<std::uint64_t*>(buffer+16),
            reinterpret_cast<std::uint32_t*>(buffer+32),reinterpret_cast<std::uint32_t*>(buffer+48),
            {},reinterpret_cast<float*>(buffer+64),13,26,nullptr,
            reinterpret_cast<float*>(buffer+96),reinterpret_cast<float*>(buffer+112));
        BEAM_CUDA_CHECK(cudaStreamEndCapture(stream,&invalid));
        bool rejected=false;
        try {beam::score_mode::GraphPointerInventory bad;bad.observe(invalid,0,registry);}
        catch(const std::runtime_error&){rejected=true;}
        BEAM_CUDA_CHECK(cudaGraphDestroy(invalid));
        if(!rejected)throw std::runtime_error("dual required null pointer accepted");
        std::cout<<payload.dump()<<'\n';
        BEAM_CUDA_CHECK(cudaGraphExecDestroy(exec));BEAM_CUDA_CHECK(cudaGraphDestroy(graph));
        BEAM_CUDA_CHECK(cudaStreamDestroy(stream));BEAM_CUDA_CHECK(cudaFree(buffer));
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
