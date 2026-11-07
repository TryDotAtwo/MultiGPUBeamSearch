// Actual CUDA capture, replay and bounded endpoint observation, not inference admission.
#include "../tools/graph_transfer_inventory.hpp"
#include <iostream>
#include <vector>
#include <cstring>

int main(int argc,char** argv) {
    try {
        const int device=argc>1?std::stoi(argv[1]):0;
        BEAM_CUDA_CHECK(cudaSetDevice(device));
        unsigned char *raw=nullptr,*logits=nullptr,*error=nullptr,*foreign=nullptr;
        BEAM_CUDA_CHECK(cudaMalloc(&raw,1536));
        BEAM_CUDA_CHECK(cudaMalloc(&logits,624));
        BEAM_CUDA_CHECK(cudaMalloc(&error,4));
        BEAM_CUDA_CHECK(cudaMalloc(&foreign,624));
        std::vector<unsigned char> input(624);
        for(std::size_t i=0;i<input.size();++i)input[i]=static_cast<unsigned char>(i%251);
        BEAM_CUDA_CHECK(cudaMemcpy(logits,input.data(),624,cudaMemcpyHostToDevice));
        cudaStream_t stream{};cudaGraph_t graph{};cudaGraphExec_t exec{};
        BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        BEAM_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        BEAM_CUDA_CHECK(cudaMemsetAsync(raw,0,1536,stream));
        BEAM_CUDA_CHECK(cudaMemsetAsync(error,0,4,stream));
        BEAM_CUDA_CHECK(cudaMemcpyAsync(raw,logits,624,cudaMemcpyDeviceToDevice,stream));
        BEAM_CUDA_CHECK(cudaMemcpyAsync(raw+624,logits,624,cudaMemcpyDeviceToDevice,stream));
        BEAM_CUDA_CHECK(cudaMemcpyAsync(raw+1248,logits,288,cudaMemcpyDeviceToDevice,stream));
        BEAM_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
        beam::score_mode::GraphTransferInventory observed;
        observed.observe(graph,0,raw,1536,logits,624,error,4);
        const auto captured=observed.payload().at("lanes").at(0);
        for(const auto& operation:captured.at("operations"))
            if(operation.at("kind")=="memcpy")
                for(const auto* endpoint:{"source_pitched","destination_pitched"}) {
                    if(!operation.contains(endpoint) || operation.at(endpoint).size()!=3)
                        throw std::runtime_error("missing captured pitched-pointer geometry");
                    for(const auto* member:{"pitch","xsize","ysize"})
                        if(!operation.at(endpoint).contains(member) ||
                           !operation.at(endpoint).at(member).is_number_unsigned())
                            throw std::runtime_error("missing typed pitched-pointer member");
                }
        if(!captured.contains("dependency_edges") || captured.at("dependency_edges").size()!=4 ||
           captured.value("node_count",0)!=5 || !captured.at("kernel_node_ids").empty())
            throw std::runtime_error("actual graph dependencies not captured");
        BEAM_CUDA_CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        BEAM_CUDA_CHECK(cudaGraphLaunch(exec,stream));
        BEAM_CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<unsigned char> output(1536);
        BEAM_CUDA_CHECK(cudaMemcpy(output.data(),raw,1536,cudaMemcpyDeviceToHost));
        for(std::size_t i=0;i<output.size();++i)
            if(output[i]!=input[i%624])throw std::runtime_error("captured copy replay mismatch");
        // Metadata mutations happen after replay. Never launch a mutated graph.
        // CUDA may reject a changed geometry; that is a runtime observation,
        // not permission to assume fields are unused on an accepted node.
        std::size_t total=0;BEAM_CUDA_CHECK(cudaGraphGetNodes(graph,nullptr,&total));
        std::vector<cudaGraphNode_t> nodes(total);
        BEAM_CUDA_CHECK(cudaGraphGetNodes(graph,nodes.data(),&total));
        const bool request_mutations=argc>2 && std::string(argv[2])=="--mutations";
        bool mutated_copy=false;
        for(const auto node:nodes) {
            if(!request_mutations)break;
            cudaGraphNodeType type{};BEAM_CUDA_CHECK(cudaGraphNodeGetType(node,&type));
            if(type!=cudaGraphNodeTypeMemcpy || mutated_copy)continue;
            cudaMemcpy3DParms original{};BEAM_CUDA_CHECK(cudaGraphMemcpyNodeGetParams(node,&original));
            for(int which=0;which<6;++which) {
                auto changed=original;
                auto& pitched=which<3?changed.srcPtr:changed.dstPtr;
                auto& field=which%3==0?pitched.pitch:(which%3==1?pitched.xsize:pitched.ysize);
                field+=which%3==2?1:64;
                const auto status=cudaGraphMemcpyNodeSetParams(node,&changed);
                if(status!=cudaSuccess) {
                    std::cerr<<"pitched_mutation="<<which<<" cuda_rejected="
                             <<static_cast<int>(status)<<'\n';
                    // Clear the API error before subsequent checked operations.
                    (void)cudaGetLastError();continue;
                }
                cudaMemcpy3DParms roundtrip{};
                BEAM_CUDA_CHECK(cudaGraphMemcpyNodeGetParams(node,&roundtrip));
                beam::score_mode::GraphTransferInventory mutation;
                mutation.observe(graph,0,raw,1536,logits,624,error,4);
                const auto mutation_payload=mutation.payload();
                bool matched=false;
                // Enumeration order need not be stable; match the unique copy offset.
                for(const auto& op:mutation_payload.at("lanes").at(0).at("operations")) {
                    if(op.at("kind")!="memcpy" ||
                       op.at("destination_offset")!=std::uintptr_t(original.dstPtr.ptr)-std::uintptr_t(raw)+original.dstPos.x)
                        continue;
                    matched=true;
                    const auto& actual=which<3?roundtrip.srcPtr:roundtrip.dstPtr;
                    const auto& observed=op.at(which<3?"source_pitched":"destination_pitched");
                    if(observed.at("pitch")!=actual.pitch || observed.at("xsize")!=actual.xsize ||
                       observed.at("ysize")!=actual.ysize)
                        throw std::runtime_error("collector omitted pitched-pointer mutation");
                }
                if(!matched)throw std::runtime_error("mutated copy missing from inventory");
                std::cerr<<"pitched_mutation="<<which<<" observed=1\n";
                BEAM_CUDA_CHECK(cudaGraphMemcpyNodeSetParams(node,&original));
            }
            mutated_copy=true;
        }
        if(request_mutations && !mutated_copy)throw std::runtime_error("no memcpy for typed mutations");
        bool rejected=false;
        try {beam::score_mode::GraphTransferInventory bad;
            bad.observe(graph,0,raw,1536,foreign,624,error,4);
        } catch(const std::runtime_error&) {rejected=true;}
        if(!rejected)throw std::runtime_error("unregistered endpoint accepted");
        rejected=false;
        try {beam::score_mode::GraphTransferInventory bad;
            bad.observe(graph,0,raw,1536,raw,624,error,4);
        } catch(const std::runtime_error&) {rejected=true;}
        if(!rejected)throw std::runtime_error("overlapping span registration accepted");
        rejected=false;
        try {beam::score_mode::GraphTransferInventory bad;
            bad.observe(graph,0,raw,1535,logits,624,error,4);
        } catch(const std::runtime_error&) {rejected=true;}
        if(!rejected)throw std::runtime_error("out-of-bounds reset accepted");
        std::cout<<observed.payload().dump()<<'\n';
        BEAM_CUDA_CHECK(cudaGraphExecDestroy(exec));
        BEAM_CUDA_CHECK(cudaGraphDestroy(graph));
        BEAM_CUDA_CHECK(cudaStreamDestroy(stream));
        for(auto pointer:{raw,logits,error,foreign})BEAM_CUDA_CHECK(cudaFree(pointer));
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
