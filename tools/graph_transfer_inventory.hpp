#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <vector>
#include <array>
#include <map>
#include <stdexcept>
#include "../third_party/nlohmann/json.hpp"
#include "../cuda/cuda_check.hpp"

namespace beam::score_mode {
// Diagnostic-only. Device addresses are resolved locally and never serialized.
class GraphTransferInventory {
    nlohmann::json lanes_=nlohmann::json::array();
    struct Span {const char* role;std::uintptr_t base;std::size_t bytes;};
    static Span span(const char* role,const void* pointer,std::size_t bytes) {
        const auto base=reinterpret_cast<std::uintptr_t>(pointer);
        if(!base || !bytes || bytes>std::numeric_limits<std::uintptr_t>::max()-base)
            throw std::runtime_error("invalid diagnostic transfer span");
        return {role,base,bytes};
    }
    static nlohmann::json endpoint(const std::array<Span,3>& spans,const void* pointer,
                                   std::size_t position,std::size_t bytes) {
        const auto base=reinterpret_cast<std::uintptr_t>(pointer);
        if(!base || !bytes || position>std::numeric_limits<std::uintptr_t>::max()-base)
            throw std::runtime_error("invalid graph transfer endpoint");
        const auto address=base+position;
        for(const auto& s:spans)
            if(address>=s.base && address-s.base<=s.bytes && bytes<=s.bytes-(address-s.base))
                return {{"role",s.role},{"offset",address-s.base}};
        throw std::runtime_error("graph transfer endpoint outside registered spans");
    }
public:
    void observe(cudaGraph_t graph,std::uint32_t lane,
                 const void* raw,std::size_t raw_bytes,const void* logits,std::size_t logits_bytes,
                 const void* error,std::size_t error_bytes) {
        if(lane>=64 || lanes_.size()>=64)throw std::runtime_error("transfer lane bound exceeded");
        for(const auto& seen:lanes_)if(seen.at("lane")==lane)
            throw std::runtime_error("duplicate transfer lane");
        const std::array<Span,3> spans{{span("raw_scores",raw,raw_bytes),
            span("lane_logits",logits,logits_bytes),span("numeric_error",error,error_bytes)}};
        for(std::size_t i=0;i<spans.size();++i)for(std::size_t j=i+1;j<spans.size();++j)
            if(spans[i].base<spans[j].base+spans[j].bytes &&
               spans[j].base<spans[i].base+spans[i].bytes)
                throw std::runtime_error("overlapping diagnostic transfer spans");
        std::size_t count=0;
        BEAM_CUDA_CHECK(cudaGraphGetNodes(graph,nullptr,&count));
        if(!count || count>65536)throw std::runtime_error("transfer graph node bound exceeded");
        std::vector<cudaGraphNode_t> nodes(count);
        BEAM_CUDA_CHECK(cudaGraphGetNodes(graph,nodes.data(),&count));
        if(count!=nodes.size())throw std::runtime_error("transfer graph changed during enumeration");
        auto operations=nlohmann::json::array();
        auto kernel_ids=nlohmann::json::array();
        std::map<cudaGraphNode_t,std::size_t> node_ids;
        for(std::size_t i=0;i<nodes.size();++i)
            if(!node_ids.emplace(nodes[i],i).second)
                throw std::runtime_error("duplicate graph node during dependency enumeration");
        for(const auto node:nodes) {
            cudaGraphNodeType type{};BEAM_CUDA_CHECK(cudaGraphNodeGetType(node,&type));
            if(type==cudaGraphNodeTypeKernel) {kernel_ids.push_back(node_ids.at(node));continue;}
            if(type==cudaGraphNodeTypeMemset) {
                cudaMemsetParams p{};BEAM_CUDA_CHECK(cudaGraphMemsetNodeGetParams(node,&p));
                if(p.height!=1 || (p.elementSize!=1 && p.elementSize!=2 && p.elementSize!=4) ||
                   !p.width || p.width>std::numeric_limits<std::size_t>::max()/p.elementSize)
                    throw std::runtime_error("unsupported graph memset geometry");
                const auto bytes=p.width*p.elementSize;
                const auto dst=endpoint(spans,p.dst,0,bytes);
                operations.push_back({{"kind","memset"},{"destination",dst.at("role")},
                    {"destination_offset",dst.at("offset")},{"bytes",bytes},
                    {"value",p.value},{"element_size",p.elementSize},
                    {"memset_geometry",{{"pitch",p.pitch},{"width",p.width},{"height",p.height}}}});
            } else if(type==cudaGraphNodeTypeMemcpy) {
                cudaMemcpy3DParms p{};BEAM_CUDA_CHECK(cudaGraphMemcpyNodeGetParams(node,&p));
                if(p.srcArray || p.dstArray || p.kind!=cudaMemcpyDeviceToDevice ||
                   p.extent.height!=1 || p.extent.depth!=1 ||
                   p.srcPos.y || p.srcPos.z || p.dstPos.y || p.dstPos.z)
                    throw std::runtime_error("unsupported graph memcpy geometry or direction");
                const auto src=endpoint(spans,p.srcPtr.ptr,p.srcPos.x,p.extent.width);
                const auto dst=endpoint(spans,p.dstPtr.ptr,p.dstPos.x,p.extent.width);
                operations.push_back({{"kind","memcpy"},{"source",src.at("role")},
                    {"source_offset",src.at("offset")},{"destination",dst.at("role")},
                    {"destination_offset",dst.at("offset")},{"bytes",p.extent.width},
                    {"direction","device_to_device"},
                    {"copy_geometry",{{"source_array_null",p.srcArray==nullptr},
                        {"destination_array_null",p.dstArray==nullptr},
                        {"source_position",{{"x",p.srcPos.x},{"y",p.srcPos.y},{"z",p.srcPos.z}}},
                        {"destination_position",{{"x",p.dstPos.x},{"y",p.dstPos.y},{"z",p.dstPos.z}}},
                        {"extent",{{"width",p.extent.width},{"height",p.extent.height},{"depth",p.extent.depth}}}}},
                    {"source_pitched",{{"pitch",p.srcPtr.pitch},{"xsize",p.srcPtr.xsize},{"ysize",p.srcPtr.ysize}}},
                    {"destination_pitched",{{"pitch",p.dstPtr.pitch},{"xsize",p.dstPtr.xsize},{"ysize",p.dstPtr.ysize}}}});
            } else throw std::runtime_error("unsupported transfer graph node");
            operations.back()["node_id"]=node_ids.at(node);
        }
        std::size_t edge_count=0;
        // NULL edgeData requires default zeroed dependency semantics; a lossy
        // query (programmatic/non-default edges) is rejected by CUDA itself.
        BEAM_CUDA_CHECK(cudaGraphGetEdges_v2(graph,nullptr,nullptr,nullptr,&edge_count));
        if(edge_count>262144)throw std::runtime_error("graph dependency edge bound exceeded");
        std::vector<cudaGraphNode_t> from(edge_count),to(edge_count);
        if(edge_count)BEAM_CUDA_CHECK(cudaGraphGetEdges_v2(graph,from.data(),to.data(),nullptr,&edge_count));
        if(edge_count!=from.size())throw std::runtime_error("graph dependencies changed during enumeration");
        auto edges=nlohmann::json::array();
        for(std::size_t i=0;i<edge_count;++i) {
            if(!node_ids.count(from[i]) || !node_ids.count(to[i]))
                throw std::runtime_error("unknown node in graph dependency edges");
            edges.push_back({node_ids.at(from[i]),node_ids.at(to[i])});
        }
        lanes_.push_back({{"lane",lane},{"operations",std::move(operations)},
            {"node_count",count},{"kernel_node_ids",std::move(kernel_ids)},
            {"dependency_edges",std::move(edges)}});
    }
    nlohmann::json payload() const {
        return {{"schema_version",1},{"scope","captured_graph_transfers_not_admission"},
                {"production_admitted",false},{"lanes",lanes_}};
    }
};
}
