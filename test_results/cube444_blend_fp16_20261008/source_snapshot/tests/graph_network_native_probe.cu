// Synthetic typed capture extraction; not real transformer numerical admission.
#include "../tools/stream1_graph_inventory.hpp"
#include <iostream>
namespace beam {
__global__ void stream1_transformer_build_input_kernel_graph_job(
    const void*,const void*,const void*,const void*,Stream1TransformerNetworkView network,
    half* output,std::uint32_t batch,std::uint32_t offset) {
    if(threadIdx.x==0) output[0]=network.cls_token[0];
}
}
int main() {
    try {
        unsigned char* buffer=nullptr;
        BEAM_CUDA_CHECK(cudaMalloc(&buffer,4096));
        BEAM_CUDA_CHECK(cudaMemset(buffer,0,4096));
        beam::Stream1TransformerNetworkView network{};
        network.fast_slot_projected=reinterpret_cast<half*>(buffer);
        network.fast_piece_static=reinterpret_cast<half*>(buffer+512);
        network.cls_token=reinterpret_cast<half*>(buffer+1024);
        network.piece_positions=reinterpret_cast<std::uint16_t*>(buffer+1536);
        network.piece_mask=buffer+2048;
        cudaStream_t stream{};cudaGraph_t graph{};
        BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        BEAM_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        beam::stream1_transformer_build_input_kernel_graph_job<<<1,128,0,stream>>>(
            buffer,buffer,buffer,buffer,network,reinterpret_cast<half*>(buffer+3072),13,0);
        BEAM_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
        const char* names[]={"fast_slot_projected","fast_piece_static","cls_token","piece_positions","piece_mask"};
        auto observe=[&](bool swapped,bool short_span,bool foreign) {
            beam::score_mode::GraphPointerRegistry registry;
            for(unsigned i=0;i<5;++i) {
                if(foreign && i==4)continue;
                const auto role=swapped && i<2?names[1-i]:names[i];
                registry.add(role,buffer+512*i,short_span && i==2?1:512);
            }
            beam::score_mode::GraphKernelInventory inventory;
            inventory.observe(graph,0,&registry);
            return inventory.network_members_payload();
        };
        const auto good=observe(false,false,false);
        const auto& members=good.at("lanes").at(0).at("kernels").at(0).at("members");
        if(members.size()!=5)throw std::runtime_error("missing typed members");
        for(unsigned i=0;i<5;++i)
            if(members.at(i).at("member")!=names[i] || members.at(i).at("role")!=names[i])
                throw std::runtime_error("incorrect typed extraction");
        const auto swapped=observe(true,false,false);
        if(swapped.at("lanes").at(0).at("kernels").at(0).at("members").at(0).at("role")!=names[1])
            throw std::runtime_error("swapped role silently relabelled");
        const auto shortened=observe(false,true,false);
        if(shortened.at("lanes").at(0).at("kernels").at(0).at("members").at(2).at("available_bytes")!=1)
            throw std::runtime_error("short span silently enlarged");
        bool rejected=false;
        try {observe(false,false,true);}catch(const std::runtime_error&){rejected=true;}
        if(!rejected)throw std::runtime_error("foreign member accepted");
        std::cout<<"PASS typed members; swapped roles/short extent preserved; foreign rejected\n";
        BEAM_CUDA_CHECK(cudaGraphDestroy(graph));
        BEAM_CUDA_CHECK(cudaStreamDestroy(stream));
        BEAM_CUDA_CHECK(cudaFree(buffer));
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
