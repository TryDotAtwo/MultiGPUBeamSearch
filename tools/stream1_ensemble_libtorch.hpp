#pragma once
#include "../src/config.hpp"
#include "cube444_blend_libtorch.hpp"
#include "stream1_mlp_libtorch_backend.hpp"
#include <memory>
#include <cmath>

namespace beam::stream1_libtorch {
// Serial head schedule keeps activation residency independent of model count.
// Each model's final GEMM accumulates directly through the CUTLASS epilogue.
struct NativeEnsemble {
    struct Head {
        std::string family;
        float coefficient;
        std::shared_ptr<Cube444Blend> cube;
        std::unique_ptr<MlpLibTorch> mlp;
        std::unique_ptr<PieceTransformerLibTorch> transformer;
        torch::Tensor weight,bias;
        torch::Tensor features(const torch::Tensor& states) const {
            if(mlp) return mlp->features(states);
            if(transformer) return transformer->features(states);
            return family=="cube444_mlp"?cube->mlp_features(states):cube->transformer_cls(states);
        }
    };
    std::vector<Head> heads;
    torch::Device device;
    std::uint32_t state_len,output_dim,padded_output;
    NativeEnsemble(const fs::path& directory,const torch::Device& target)
        :device(target),state_len(STATE_LEN),output_dim(MOVE_COUNT),padded_output((MOVE_COUNT+7)/8*8) {
        auto manifest=nlohmann::json::parse(read_text_exact(directory/"ensemble.json"));
        if(manifest.at("schema_version")!=1||manifest.at("state_len")!=state_len||
           manifest.at("output_dim")!=output_dim||manifest.at("score_semantics")!="distance_q"||
           !manifest.at("models").is_array()||manifest.at("models").empty())
            throw std::runtime_error("invalid ensemble graph/output/score contract");
        std::map<std::string,std::shared_ptr<Cube444Blend>> shared_cube;
        for(const auto& entry:manifest.at("models")) {
            Head head;head.family=entry.at("family").get<std::string>();
            head.coefficient=entry.at("coefficient").get<float>();
            if(!std::isfinite(head.coefficient)) throw std::runtime_error("nonfinite ensemble coefficient");
            auto path=fs::path(entry.at("weights_dir").get<std::string>());
            if(path.is_relative()) path=directory/path;
            if(head.family=="mlp") {
                head.mlp=std::make_unique<MlpLibTorch>(path,target);
                if(head.mlp->state_len!=state_len||head.mlp->output_dim!=output_dim||head.mlp->dtype!=torch::kFloat16)
                    throw std::runtime_error("ensemble MLP requires matching Q head and FP16");
                head.weight=head.mlp->output.weight.transpose(0,1).contiguous();
                head.bias=head.mlp->output.bias.to(torch::kFloat32).contiguous();
            } else if(head.family=="piece_transformer") {
                head.transformer=std::make_unique<PieceTransformerLibTorch>(path,target);
                if(head.transformer->state_len!=state_len||head.transformer->output_dim!=output_dim||head.transformer->dtype!=torch::kFloat16)
                    throw std::runtime_error("ensemble transformer requires matching Q head and FP16");
                head.weight=head.transformer->output_weight_kxh.contiguous();
                head.bias=head.transformer->output_bias.to(torch::kFloat32).contiguous();
            } else if(head.family=="cube444_mlp"||head.family=="cube444_transformer") {
                if(state_len!=96||output_dim!=24) throw std::runtime_error("cube adapter requires state96/moves24");
                auto canonical=fs::canonical(path).string();
                if(!shared_cube.count(canonical)) shared_cube[canonical]=std::make_shared<Cube444Blend>(path,target);
                head.cube=shared_cube.at(canonical);
                auto prefix=head.family=="cube444_mlp"?"mlp/out":"s3/output_layer";
                head.weight=head.cube->at(std::string(prefix)+"_w").contiguous();
                head.bias=head.cube->at(std::string(prefix)+"_b").to(torch::kFloat32).contiguous();
            } else throw std::runtime_error("unsupported native ensemble model family: "+head.family);
            if(head.weight.size(0)%8) throw std::runtime_error("ensemble hidden size must be aligned to 8");
            if(padded_output!=output_dim) {
                auto w=torch::zeros({head.weight.size(0),padded_output},head.weight.options());
                w.narrow(1,0,output_dim).copy_(head.weight);head.weight=w;
                auto b=torch::zeros({padded_output},head.bias.options());
                b.narrow(0,0,output_dim).copy_(head.bias);head.bias=b;
            }
            heads.push_back(std::move(head));
        }
        if(cudaDeviceSynchronize()!=cudaSuccess) throw std::runtime_error("ensemble startup synchronization failed");
    }
    void score(const torch::Tensor& states,std::uint32_t* destination,std::uint32_t* error,cudaStream_t stream) const {
        auto rows=states.size(0);
        auto options=states.options().dtype(torch::kFloat32);
        auto sum=torch::empty({rows,padded_output},options);
        auto padded_keys=torch::empty({rows,padded_output},options.dtype(torch::kInt32));
        auto key_ptr=padded_output==output_dim?destination:reinterpret_cast<std::uint32_t*>(padded_keys.data_ptr<int>());
        for(std::size_t i=0;i<heads.size();++i) {
            const auto& head=heads[i];auto features=head.features(states);
            stream1_ensemble_head_fp16_cuda(reinterpret_cast<const __half*>(features.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(head.weight.data_ptr<at::Half>()),head.bias.data_ptr<float>(),
                sum.data_ptr<float>(),key_ptr,rows,head.weight.size(0),padded_output,
                head.coefficient,i==0,i+1==heads.size(),error,stream);
        }
        if(padded_output!=output_dim) {
            auto output=torch::from_blob(destination,{rows,output_dim},options.dtype(torch::kInt32));
            output.copy_(padded_keys.narrow(1,0,output_dim),true);
        }
    }
};
}
