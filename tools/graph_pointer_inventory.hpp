#pragma once
#include <cuda_runtime.h>
#include <cstring>
#include "graph_pointer_registry.hpp"
#include "graph_parameter_layout.hpp"
#include "../third_party/nlohmann/json.hpp"
#include "../cuda/cuda_check.hpp"

namespace beam::score_mode {
// Known top-level pointer parameters only. Struct members and opaque Params
// are deliberately excluded and cannot be promoted to complete admission.
class GraphPointerInventory {
    nlohmann::json lanes_=nlohmann::json::array();
    static std::vector<std::size_t> indices(const std::string& symbol) {
        if(symbol.rfind("_ZN4beam",0)!=0)return {};
        if(symbol.find("stream1_transformer_build_input_layernorm256_generic_kernel")!=std::string::npos)
            return {0,1,2,3,5,8,9,10};
        if(symbol.find("stream1_transformer_bias_layernorm256_copy_kernel")!=std::string::npos)
            return {0,1,2,3,4};
        if(symbol.find("stream1_transformer_layernorm256_copy_kernel")!=std::string::npos)
            return {0,1,2,3};
        if(symbol.find("stream1_transformer_gather_cls256_kernel")!=std::string::npos)
            return {0,1};
        if(symbol.find("stream1_transformer_score_quantize_graph_job_kernel")!=std::string::npos)
            return {0,1,2,3,4,9};
        if(symbol.find("stream1_transformer_build_input_kernel_graph_job")!=std::string::npos)
            return {0,1,2,3,5};
        if(symbol.find("stream1_transformer_zero_padded_rows_kernel")!=std::string::npos ||
           symbol.find("stream1_transformer_zero_padded_rows_legacy_kernel")!=std::string::npos)
            return {0};
        if(symbol.find("stream1_transformer_cls_bias_layernorm_kernel")!=std::string::npos)
            return {0,1,2,3,4};
        return {};
    }
public:
    void observe(cudaGraph_t graph,std::uint32_t lane,const GraphPointerRegistry& registry) {
#if CUDART_VERSION >= 12080
        if(lane>=64 || lanes_.size()>=64)throw std::runtime_error("pointer lane bound exceeded");
        for(const auto& seen:lanes_)if(seen.at("lane")==lane)
            throw std::runtime_error("duplicate pointer lane");
        std::size_t count=0;BEAM_CUDA_CHECK(cudaGraphGetNodes(graph,nullptr,&count));
        if(!count || count>65536)throw std::runtime_error("pointer graph node bound exceeded");
        std::vector<cudaGraphNode_t> nodes(count);
        BEAM_CUDA_CHECK(cudaGraphGetNodes(graph,nodes.data(),&count));
        if(count!=nodes.size())throw std::runtime_error("pointer graph enumeration changed");
        auto kernels=nlohmann::json::array();
        for(const auto node:nodes) {
            cudaGraphNodeType type{};BEAM_CUDA_CHECK(cudaGraphNodeGetType(node,&type));
            if(type==cudaGraphNodeTypeMemcpy || type==cudaGraphNodeTypeMemset)continue;
            if(type!=cudaGraphNodeTypeKernel)throw std::runtime_error("unsupported pointer graph node");
            cudaKernelNodeParams params{};BEAM_CUDA_CHECK(cudaGraphKernelNodeGetParams(node,&params));
            const char* name=nullptr;BEAM_CUDA_CHECK(cudaFuncGetName(&name,params.func));
            if(!name || !*name || std::strlen(name)>65536)
                throw std::runtime_error("pointer kernel symbol unavailable");
            const auto selected=indices(name);
            auto pointers=nlohmann::json::array();
            for(const auto index:selected) {
                std::size_t offset=0,size=0;
                BEAM_CUDA_CHECK(cudaFuncGetParamInfo(params.func,index,&offset,&size));
                if(size!=sizeof(void*) || !params.kernelParams || params.extra || !params.kernelParams[index])
                    throw std::runtime_error("unsupported captured pointer storage");
                const void* address=nullptr;std::memcpy(&address,params.kernelParams[index],sizeof(address));
                if(!address && index>=8 &&
                    std::string(name).find("stream1_transformer_build_input_layernorm256_generic_kernelILb0E")!=std::string::npos) {
                    pointers.push_back({{"index",index},{"role","null"},{"offset",0},{"available_bytes",0}});
                    continue;
                }
                const auto endpoint=registry.resolve(address,1);
                pointers.push_back({{"index",index},{"role",endpoint.role},{"offset",endpoint.offset},
                    {"available_bytes",endpoint.available_bytes}});
            }
            kernels.push_back({{"kernel_index",kernels.size()},{"mangled_name",name},
                {"pointer_signature_known",!selected.empty()},{"pointers",std::move(pointers)}});
        }
        lanes_.push_back({{"lane",lane},{"kernels",std::move(kernels)}});
#else
        (void)graph;(void)lane;(void)registry;
        throw std::runtime_error("pointer inventory requires CUDA12.8+");
#endif
    }
    nlohmann::json payload() const {
        return {{"schema_version",1},{"scope","captured_top_level_pointer_roles_not_extents_or_structs_or_admission"},
            {"production_admitted",false},{"parameter_values_checked",false},{"lanes",lanes_}};
    }
};
}
