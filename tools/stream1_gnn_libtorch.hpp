#pragma once
#include "../src/config.hpp"
#include <torch/script.h>
#include "cube444_blend_libtorch.hpp"

namespace beam::stream1_libtorch {
struct PancakeGnnNative {
    torch::jit::script::Module backbone;
    torch::Tensor output_weight,output_bias;
    std::int64_t max_state_batch=0;
    explicit PancakeGnnNative(const fs::path& directory,const torch::Device& device,bool cutlass) {
        const auto metadata=nlohmann::json::parse(read_text_exact(directory/"gnn.json"));
        if(metadata.at("schema_version")!=1||metadata.at("family")!="pancake_gnn"||
           metadata.at("n")!=STATE_LEN||MOVE_COUNT!=STATE_LEN-1)
            throw std::runtime_error("GNN graph shape mismatch");
        max_state_batch=metadata.at("max_state_batch").get<std::int64_t>();
        if(max_state_batch<1) throw std::runtime_error("invalid GNN state batch limit");
        const auto bytes=read_text_exact(directory/"backbone.pt");
        if(picosha2::hash256_hex_string(bytes)!=metadata.at("files").at("backbone.pt").get<std::string>())
            throw std::runtime_error("GNN backbone checksum mismatch");
        backbone=torch::jit::load((directory/"backbone.pt").string(),device);
        backbone.eval();
        for(auto module:backbone.modules())
            if(module.hasattr("use_cutlass")) module.setattr("use_cutlass",cutlass);
        for(const auto& parameter:backbone.named_parameters()) {
            if(parameter.name=="value_head.3.weight") output_weight=parameter.value;
            if(parameter.name=="value_head.3.bias") output_bias=parameter.value;
            if(parameter.value.is_floating_point() && parameter.value.scalar_type()!=torch::kFloat16)
                throw std::runtime_error("GNN checkpoint must use FP16 parameters");
        }
        if(!output_weight.defined()||!output_bias.defined()||output_weight.dim()!=2||
           output_weight.size(0)!=1||output_weight.size(1)!=metadata.at("d_model").get<int>())
            throw std::runtime_error("GNN output head mismatch");
    }
    torch::Tensor features(const torch::Tensor& states) const {
        // Root-independent deterministic expansion makes this bound exact;
        // never activate the reference's stochastic root-balanced cap.
        std::vector<torch::Tensor> chunks;
        for(std::int64_t start=0;start<states.size(0);start+=max_state_batch) {
            const auto count=std::min(max_state_batch,states.size(0)-start);
            chunks.push_back(backbone.get_method("features")({states.narrow(0,start,count)}).toTensor());
        }
        if(chunks.empty()) throw std::runtime_error("empty GNN feature batch");
        return chunks.size()==1?chunks.front():torch::cat(chunks,0);
    }
};
}
