// Synthetic typed extraction/replay, not production kernel correctness.
#include "../tools/stream1_graph_inventory.hpp"
#include "../tools/graph_pointer_inventory.hpp"
#include <algorithm>
#include <array>
#include <iostream>
namespace beam {
__global__ void stream1_transformer_zero_padded_rows_kernel(half* out,Stream1TransformerDims dims,
    std::uint32_t rows,std::uint32_t tail) {if(threadIdx.x==0)out[0]=__float2half_rn(dims.seq_len+dims.d_model);}
__global__ void stream1_transformer_zero_padded_rows_legacy_kernel(half* out,Stream1TransformerDims dims,
    std::uint32_t rows) {if(threadIdx.x==0)out[0]=__float2half_rn(dims.seq_len+dims.d_model);}
__global__ void stream1_transformer_cls_bias_layernorm_kernel(const half* in,half* out,
    const half* bias,const half* gamma,const half* beta,Stream1TransformerDims dims,std::uint32_t rows) {
    if(threadIdx.x==0)out[0]=__float2half_rn(dims.seq_len+dims.d_model);}
__global__ void stream1_transformer_build_input_kernel_graph_job(const State128* states,
    const std::uint64_t* base,const std::uint32_t* count,const std::uint32_t* job,
    Stream1TransformerNetworkView view,half* out,std::uint32_t rows,std::uint32_t offset) {
    if(threadIdx.x==0)out[0]=__float2half_rn(view.dims.seq_len+view.dims.d_model);}
}
int main(int argc,char** argv) {
    try {
        using namespace beam;
        static_assert(sizeof(Stream1TransformerDims)==15*sizeof(std::uint32_t));
        BEAM_CUDA_CHECK(cudaSetDevice(argc>1?std::stoi(argv[1]):0));
        half* output=nullptr;BEAM_CUDA_CHECK(cudaMalloc(&output,4*sizeof(half)));
        void* input=nullptr;BEAM_CUDA_CHECK(cudaMalloc(&input,16));
        Stream1TransformerDims dims{96,6,56,3,57,64,16,256,8,32,4,1024,24,0,1};
        Stream1TransformerNetworkView view{};view.dims=dims;
        cudaStream_t stream{};cudaGraph_t graph{};cudaGraphExec_t exec{};
        BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        BEAM_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        stream1_transformer_zero_padded_rows_kernel<<<1,32,0,stream>>>(output,dims,13,7);
        stream1_transformer_zero_padded_rows_legacy_kernel<<<1,32,0,stream>>>(output+1,dims,13);
        stream1_transformer_cls_bias_layernorm_kernel<<<1,32,0,stream>>>(static_cast<half*>(input),output+2,static_cast<half*>(input),static_cast<half*>(input),static_cast<half*>(input),dims,13);
        stream1_transformer_build_input_kernel_graph_job<<<1,32,0,stream>>>(static_cast<State128*>(input),static_cast<std::uint64_t*>(input),static_cast<std::uint32_t*>(input),static_cast<std::uint32_t*>(input),view,output+3,13,26);
        BEAM_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
        auto capture=[&](){score_mode::GraphKernelInventory inv;inv.observe(graph,0);return inv.dims_values_payload();};
        const char* fields[]={"state_len","num_classes","num_pieces","max_piece_size","seq_len",
            "padded_seq_len","sequence_alignment","d_model","nhead","head_dim","transformer_layers","ff_dim","output_dim","dtype","activation"};
        std::array<std::uint32_t,15> expected{};std::memcpy(expected.data(),&dims,sizeof(dims));
        auto baseline=capture();auto baseline_nodes=baseline.at("lanes").at(0).at("kernels");
        score_mode::GraphPointerRegistry registry;registry.add("lane_tokens",output,4*sizeof(half));
        registry.add("fixture_inputs",input,16);
        score_mode::GraphPointerInventory pointer_inventory;pointer_inventory.observe(graph,0,registry);
        auto pointer_payload=pointer_inventory.payload();
        std::vector<std::size_t> pointer_counts,output_offsets;
        unsigned nulls=0;
        for(const auto& node:pointer_payload.at("lanes").at(0).at("kernels")) {
            if(node.at("pointer_signature_known")!=true)throw std::runtime_error("missing typed pointer signature");
            pointer_counts.push_back(node.at("pointers").size());
            for(const auto& pointer:node.at("pointers")) {
                if(pointer.at("role")=="fixture_inputs") {
                    if(pointer.at("offset")!=0 || pointer.at("available_bytes")!=16)throw std::runtime_error("invalid input pointer extraction");
                    ++nulls;
                } else {
                    const auto offset=pointer.at("offset").get<std::size_t>();
                    if(pointer.at("role")!="lane_tokens" || pointer.at("available_bytes")!=8-offset)
                        throw std::runtime_error("invalid registered pointer extraction");
                    output_offsets.push_back(offset);
                }
            }
        }
        std::sort(pointer_counts.begin(),pointer_counts.end());std::sort(output_offsets.begin(),output_offsets.end());
        if(pointer_counts!=std::vector<std::size_t>{1,1,5,5} || output_offsets!=std::vector<std::size_t>{0,2,4,6} || nulls!=8)
            throw std::runtime_error("incomplete native pointer coverage");
        std::vector<int> indices;
        for(const auto& node:baseline_nodes) {
            if(node.at("dims_signature_known")!=true)throw std::runtime_error("missing typed dims");
            indices.push_back(node.at("dims_index").get<int>());
            for(unsigned f=0;f<15;++f)if(node.at("dims").at(fields[f])!=expected[f])throw std::runtime_error("baseline field mismatch");
        }
        std::sort(indices.begin(),indices.end());
        if(indices!=std::vector<int>{1,1,4,5})throw std::runtime_error("wrong typed argument indices");
        // Execute only the intact baseline. No mutated graph is instantiated/launched.
        BEAM_CUDA_CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        BEAM_CUDA_CHECK(cudaGraphLaunch(exec,stream));BEAM_CUDA_CHECK(cudaStreamSynchronize(stream));
        half host[4]{};BEAM_CUDA_CHECK(cudaMemcpy(host,output,sizeof(host),cudaMemcpyDeviceToHost));
        for(auto value:host)if(__half2float(value)!=313)throw std::runtime_error("baseline replay mismatch");
        std::size_t count=0;BEAM_CUDA_CHECK(cudaGraphGetNodes(graph,nullptr,&count));
        std::vector<cudaGraphNode_t> nodes(count);BEAM_CUDA_CHECK(cudaGraphGetNodes(graph,nodes.data(),&count));
        unsigned mutations=0;
        for(auto handle:nodes) {
            cudaGraphNodeType type{};BEAM_CUDA_CHECK(cudaGraphNodeGetType(handle,&type));
            if(type!=cudaGraphNodeTypeKernel)continue;
            cudaKernelNodeParams original{};BEAM_CUDA_CHECK(cudaGraphKernelNodeGetParams(handle,&original));
            const char* raw=nullptr;BEAM_CUDA_CHECK(cudaFuncGetName(&raw,original.func));std::string symbol(raw);
            const int index=symbol.find("build_input_kernel_graph_job")!=std::string::npos?4:
                symbol.find("cls_bias_layernorm")!=std::string::npos?5:1;
            const bool network=index==4;
            const auto nargs=score_mode::graph_diagnostic_parameter_count(symbol);
            std::vector<std::vector<unsigned char>> storage(nargs);
            std::vector<void*> arguments(nargs);
            for(std::size_t arg=0;arg<nargs;++arg) {
                std::size_t offset=0,size=0;BEAM_CUDA_CHECK(cudaFuncGetParamInfo(original.func,arg,&offset,&size));
                storage[arg].resize(size);std::memcpy(storage[arg].data(),original.kernelParams[arg],size);arguments[arg]=storage[arg].data();
            }
            const auto saved=storage[index];
            for(unsigned field=0;field<15;++field) {
                auto values=expected;values[field]++;
                if(network) {auto changed=view;std::memcpy(&changed.dims,values.data(),sizeof(dims));std::memcpy(storage[index].data(),&changed,sizeof(changed));}
                else std::memcpy(storage[index].data(),values.data(),sizeof(dims));
                auto changed=original;changed.kernelParams=arguments.data();
                BEAM_CUDA_CHECK(cudaGraphKernelNodeSetParams(handle,&changed));
                auto observed=capture();unsigned altered=0;
                for(const auto& node:observed.at("lanes").at(0).at("kernels")) {
                    for(unsigned f=0;f<15;++f) {
                        const auto value=node.at("dims").at(fields[f]).get<std::uint32_t>();
                        if(value!=expected[f]) {if(f!=field || value!=values[f])throw std::runtime_error("unexpected native mutation");++altered;}
                    }
                }
                if(altered!=1)throw std::runtime_error("mutation not observed exactly once");
                std::memcpy(storage[index].data(),saved.data(),saved.size());
                BEAM_CUDA_CHECK(cudaGraphKernelNodeSetParams(handle,&changed));++mutations;
            }
        }
        if(capture()!=baseline || mutations!=60)throw std::runtime_error("restore/mutation coverage mismatch");
        std::cout<<"{\"scope\":\"synthetic_typed_dims_and_pointer_extraction_not_production_kernel_admission\",\"mutations\":"<<mutations
            <<",\"baseline\":"<<baseline.dump()<<",\"pointer_baseline\":"<<pointer_payload.dump()<<"}\n";
        BEAM_CUDA_CHECK(cudaGraphExecDestroy(exec));BEAM_CUDA_CHECK(cudaGraphDestroy(graph));
        BEAM_CUDA_CHECK(cudaStreamDestroy(stream));BEAM_CUDA_CHECK(cudaFree(output));
        BEAM_CUDA_CHECK(cudaFree(input));
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
