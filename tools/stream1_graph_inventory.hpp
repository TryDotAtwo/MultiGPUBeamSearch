#pragma once
#include <cuda_runtime.h>
#include "../third_party/nlohmann/json.hpp"
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include "../cuda/cuda_check.hpp"
#include "graph_parameter_layout.hpp"
#include "graph_pointer_registry.hpp"
#include "graph_attention_members.hpp"
#include "graph_gemm_members.hpp"
#include "../cuda/stream1.hpp"
#include <type_traits>

namespace beam::score_mode {
// Diagnostic-only graph-native inventory. Never called by normal inference;
// capture membership/function attributes do not by themselves attest replay.
class GraphKernelInventory {
    std::map<const void*,std::uint32_t> functions_by_address_;
    nlohmann::json functions_=nlohmann::json::array();
    nlohmann::json lanes_=nlohmann::json::array();
    nlohmann::json parameter_layouts_=nlohmann::json::array();
    nlohmann::json u32_lanes_=nlohmann::json::array();
    nlohmann::json dims_lanes_=nlohmann::json::array();
    nlohmann::json network_lanes_=nlohmann::json::array();
    nlohmann::json attention_lanes_=nlohmann::json::array();
    nlohmann::json gemm_lanes_=nlohmann::json::array();
public:
    void observe(cudaGraph_t graph,std::uint32_t lane,const GraphPointerRegistry* registry=nullptr) {
#if CUDART_VERSION >= 12080
        std::size_t count=0;
        BEAM_CUDA_CHECK(cudaGraphGetNodes(graph,nullptr,&count));
        if(!count || count>65536)throw std::runtime_error("diagnostic graph node bound exceeded");
        std::vector<cudaGraphNode_t> nodes(count);
        BEAM_CUDA_CHECK(cudaGraphGetNodes(graph,nodes.data(),&count));
        if(count!=nodes.size())throw std::runtime_error("diagnostic graph changed during enumeration");
        nlohmann::json kernels=nlohmann::json::array();
        nlohmann::json u32_kernels=nlohmann::json::array();
        nlohmann::json dims_kernels=nlohmann::json::array();
        nlohmann::json network_kernels=nlohmann::json::array();
        nlohmann::json attention_kernels=nlohmann::json::array();
        nlohmann::json gemm_kernels=nlohmann::json::array();
        nlohmann::json types={{"kernel",0},{"memcpy",0},{"memset",0}};
        for(const auto node:nodes) {
            cudaGraphNodeType type{};
            BEAM_CUDA_CHECK(cudaGraphNodeGetType(node,&type));
            if(type==cudaGraphNodeTypeMemcpy) {types["memcpy"]=types["memcpy"].get<unsigned>()+1;continue;}
            if(type==cudaGraphNodeTypeMemset) {types["memset"]=types["memset"].get<unsigned>()+1;continue;}
            if(type!=cudaGraphNodeTypeKernel)
                throw std::runtime_error("unsupported diagnostic graph node: incomplete inventory forbidden");
            cudaKernelNodeParams params{};
            BEAM_CUDA_CHECK(cudaGraphKernelNodeGetParams(node,&params));
            if(!params.func)throw std::runtime_error("captured kernel has no function");
            auto found=functions_by_address_.find(params.func);
            if(found==functions_by_address_.end()) {
                const char* name=nullptr;
                BEAM_CUDA_CHECK(cudaFuncGetName(&name,params.func));
                if(!name || !*name)throw std::runtime_error("captured kernel name unavailable");
                const std::string symbol(name);
                if(symbol.size()>65536)throw std::runtime_error("captured kernel name exceeds bound");
                cudaFuncAttributes attrs{};
                BEAM_CUDA_CHECK(cudaFuncGetAttributes(&attrs,params.func));
                const auto id=static_cast<std::uint32_t>(functions_.size());
                found=functions_by_address_.emplace(params.func,id).first;
                functions_.push_back({{"mangled_name",symbol},{"attributes",{
                    {"binary_version",attrs.binaryVersion},{"ptx_version",attrs.ptxVersion},
                    {"max_threads_per_block",attrs.maxThreadsPerBlock},{"num_regs",attrs.numRegs},
                    {"static_shared_bytes",attrs.sharedSizeBytes},{"local_bytes",attrs.localSizeBytes},
                    {"constant_bytes",attrs.constSizeBytes},
                    {"max_dynamic_shared_bytes",attrs.maxDynamicSharedSizeBytes}}}});
                const auto parameter_count=graph_diagnostic_parameter_count(symbol);
                nlohmann::json layout=nlohmann::json::array();
                std::size_t end=0;
                for(std::size_t index=0;index<parameter_count;++index) {
                    std::size_t offset=0,size=0;
                    BEAM_CUDA_CHECK(cudaFuncGetParamInfo(params.func,index,&offset,&size));
                    if(!size || size>32768 || offset<end || offset>32768-size)
                        throw std::runtime_error("captured kernel parameter layout exceeds bounds");
                    layout.push_back({{"index",index},{"offset",offset},{"size",size}});
                    end=offset+size;
                }
                parameter_layouts_.push_back({{"function_id",id},
                    {"signature_known",parameter_count!=0},{"parameters",std::move(layout)}});
            }
            types["kernel"]=types["kernel"].get<unsigned>()+1;
            const auto indices=graph_diagnostic_u32_indices(
                functions_.at(found->second).at("mangled_name").get<std::string>());
            auto scalars=nlohmann::json::array();
            for(const auto index:indices) {
                const auto& layout=parameter_layouts_.at(found->second).at("parameters").at(index);
                if(layout.at("size")!=sizeof(std::uint32_t) || !params.kernelParams ||
                   params.extra || !params.kernelParams[index])
                    throw std::runtime_error("captured scalar argument representation unsupported");
                std::uint32_t value=0;
                std::memcpy(&value,params.kernelParams[index],sizeof(value));
                scalars.push_back({{"index",index},{"value",value}});
            }
            u32_kernels.push_back({{"kernel_index",kernels.size()},
                {"function_id",found->second},{"scalar_signature_known",!indices.empty()},
                {"u32_values",std::move(scalars)}});
            const auto symbol=functions_.at(found->second).at("mangled_name").get<std::string>();
            attention_kernels.push_back(captured_attention_members(params,symbol,registry,kernels.size(),found->second));
            gemm_kernels.push_back(captured_gemm_members(params,symbol,registry,kernels.size(),found->second));
            int dims_index=-1;bool network_argument=false;
            if(symbol.rfind("_ZN4beam",0)==0) {
                if(symbol.find("stream1_transformer_build_input_layernorm256_generic_kernel")!=std::string::npos ||
                   symbol.find("stream1_transformer_build_input_kernel_graph_job")!=std::string::npos) {
                    dims_index=4;network_argument=true;
                } else if(symbol.find("stream1_transformer_gather_cls256_kernel")!=std::string::npos) dims_index=2;
                else if(symbol.find("stream1_transformer_score_quantize_graph_job_kernel")!=std::string::npos) dims_index=8;
                else if(symbol.find("stream1_transformer_zero_padded_rows_legacy_kernel")!=std::string::npos ||
                        symbol.find("stream1_transformer_zero_padded_rows_kernel")!=std::string::npos) dims_index=1;
                else if(symbol.find("stream1_transformer_cls_bias_layernorm_kernel")!=std::string::npos) dims_index=5;
            }
            nlohmann::json dimensions=nullptr;
            auto members=nlohmann::json::array();
            const bool input_network=symbol.rfind("_ZN4beam",0)==0 &&
                (network_argument || symbol.find("stream1_transformer_build_input_kernel_graph_job")!=std::string::npos);
            if(input_network && registry) {
                const auto& layout=parameter_layouts_.at(found->second).at("parameters").at(4);
                if(layout.at("size")!=sizeof(Stream1TransformerNetworkView) ||
                   !params.kernelParams || params.extra || !params.kernelParams[4])
                    throw std::runtime_error("captured NetworkView representation unsupported");
                Stream1TransformerNetworkView network{};
                std::memcpy(&network,params.kernelParams[4],sizeof(network));
                auto add=[&](const char* name,const void* pointer) {
                    const auto endpoint=registry->resolve(pointer,1);
                    members.push_back({{"member",name},{"role",endpoint.role},
                        {"offset",endpoint.offset},{"available_bytes",endpoint.available_bytes}});
                };
                add("fast_slot_projected",network.fast_slot_projected);
                add("fast_piece_static",network.fast_piece_static);
                add("cls_token",network.cls_token);
                add("piece_positions",network.piece_positions);
                add("piece_mask",network.piece_mask);
                if(symbol.find("stream1_transformer_build_input_layernorm256_generic_kernel")!=std::string::npos) {
                    add("input_ln_gamma",network.input_ln_gamma);
                    add("input_ln_beta",network.input_ln_beta);
                }
            }
            network_kernels.push_back({{"kernel_index",kernels.size()},{"function_id",found->second},
                {"network_signature_known",input_network && registry!=nullptr},
                {"network_index",input_network && registry?nlohmann::json(4):nlohmann::json(nullptr)},
                {"members",std::move(members)}});
            if(dims_index>=0) {
                static_assert(std::is_trivially_copyable_v<Stream1TransformerNetworkView>);
                static_assert(std::is_trivially_copyable_v<Stream1TransformerDims>);
                const auto required=network_argument?sizeof(Stream1TransformerNetworkView):sizeof(Stream1TransformerDims);
                const auto& layout=parameter_layouts_.at(found->second).at("parameters").at(dims_index);
                if(layout.at("size")!=required || !params.kernelParams || params.extra || !params.kernelParams[dims_index])
                    throw std::runtime_error("captured dimensions argument representation unsupported");
                Stream1TransformerDims dims{};
                if(network_argument) {
                    Stream1TransformerNetworkView network{};
                    std::memcpy(&network,params.kernelParams[dims_index],sizeof(network));dims=network.dims;
                } else std::memcpy(&dims,params.kernelParams[dims_index],sizeof(dims));
                dimensions={{"state_len",dims.state_len},{"num_classes",dims.num_classes},
                    {"num_pieces",dims.num_pieces},{"max_piece_size",dims.max_piece_size},
                    {"seq_len",dims.seq_len},{"padded_seq_len",dims.padded_seq_len},
                    {"sequence_alignment",dims.sequence_alignment},{"d_model",dims.d_model},
                    {"nhead",dims.nhead},{"head_dim",dims.head_dim},
                    {"transformer_layers",dims.transformer_layers},{"ff_dim",dims.ff_dim},
                    {"output_dim",dims.output_dim},{"dtype",dims.dtype},{"activation",dims.activation}};
            }
            dims_kernels.push_back({{"kernel_index",kernels.size()},{"function_id",found->second},
                {"dims_signature_known",dims_index>=0},
                {"dims_index",dims_index>=0?nlohmann::json(dims_index):nlohmann::json(nullptr)},
                {"dims",std::move(dimensions)}});
            kernels.push_back({{"function_id",found->second},
                {"grid",{params.gridDim.x,params.gridDim.y,params.gridDim.z}},
                {"block",{params.blockDim.x,params.blockDim.y,params.blockDim.z}},
                {"dynamic_shared_bytes",params.sharedMemBytes}});
        }
        lanes_.push_back({{"lane",lane},{"node_count",count},
            {"node_types",std::move(types)},{"kernels",std::move(kernels)}});
        u32_lanes_.push_back({{"lane",lane},{"kernels",std::move(u32_kernels)}});
        dims_lanes_.push_back({{"lane",lane},{"kernels",std::move(dims_kernels)}});
        network_lanes_.push_back({{"lane",lane},{"kernels",std::move(network_kernels)}});
        attention_lanes_.push_back({{"lane",lane},{"kernels",std::move(attention_kernels)}});
        gemm_lanes_.push_back({{"lane",lane},{"kernels",std::move(gemm_kernels)}});
#else
        (void)graph;(void)lane;
        throw std::runtime_error("graph kernel inventory requires CUDA runtime12.8+");
#endif
    }
    nlohmann::json payload() const {
        return {{"schema_version",1},{"scope","captured_graph_kernel_inventory_not_admission"},
            {"production_admitted",false},{"kernel_coverage_complete",false},
            {"functions",functions_},{"lanes",lanes_}};
    }
    nlohmann::json parameter_layout_payload() const {
        return {{"schema_version",1},{"scope","captured_parameter_layout_not_values_or_admission"},
            {"production_admitted",false},{"parameter_values_checked",false},
            {"functions",parameter_layouts_}};
    }
    nlohmann::json u32_values_payload() const {
        return {{"schema_version",1},{"scope","captured_kernel_u32_not_pointers_or_structs_or_admission"},
            {"production_admitted",false},{"parameter_values_checked",false},{"lanes",u32_lanes_}};
    }
    nlohmann::json dims_values_payload() const {
        return {{"schema_version",1},{"scope","captured_embedded_dims_not_pointer_members_or_admission"},
            {"production_admitted",false},{"parameter_values_checked",false},{"lanes",dims_lanes_}};
    }
    nlohmann::json network_members_payload() const {
        return {{"schema_version",1},{"scope","captured_consumed_network_members_not_opaque_params_or_admission"},
            {"production_admitted",false},{"parameter_values_checked",false},{"lanes",network_lanes_}};
    }
    nlohmann::json attention_members_payload() const {
        return {{"schema_version",1},{"scope","captured_attention_params_not_gemm_or_admission"},
            {"production_admitted",false},{"parameter_values_checked",false},{"lanes",attention_lanes_}};
    }
    nlohmann::json gemm_members_payload() const {
        return {{"schema_version",1},{"scope","captured_known_gemm_members_not_complete_values_or_admission"},
            {"production_admitted",false},{"parameter_values_checked",false},{"lanes",gemm_lanes_}};
    }
};
}
