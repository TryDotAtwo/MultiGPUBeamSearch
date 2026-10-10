#pragma once
#include "../src/config.hpp"
#include "cube444_blend_libtorch.hpp"
#include "stream1_mlp_libtorch_backend.hpp"
#include "stream1_gnn_libtorch.hpp"
#include <memory>
#include <cmath>

namespace beam::stream1_libtorch {
// Serial head schedule keeps activation residency independent of model count.
// Each model's final GEMM accumulates directly through the CUTLASS epilogue.
struct NativeEnsemble {
    struct Head {
        std::string family;
        float coefficient;
        std::uint32_t output_dim=0;
        std::shared_ptr<Cube444Blend> cube;
        std::unique_ptr<MlpLibTorch> mlp;
        std::unique_ptr<PieceTransformerLibTorch> transformer;
        std::unique_ptr<PancakeGnnNative> gnn;
        torch::Tensor weight,bias;
        torch::Tensor features(const torch::Tensor& states) const {
            if(mlp) return mlp->features(states);
            if(transformer) return transformer->features(states);
            if(gnn) return gnn->features(states);
            return family=="cube444_mlp"?cube->mlp_features(states):cube->transformer_cls(states);
        }
    };
    std::vector<Head> heads;
    struct Workspace {torch::Tensor sum,keys;};
    std::vector<Workspace> workspaces;
    bool use_cutlass=true;
    std::uint32_t inference_micro=0,max_parents=0;
    torch::Device device;
    std::uint32_t state_len,output_dim,padded_output;
    NativeEnsemble(const fs::path& directory,const torch::Device& target,
                   std::uint32_t max_batch=256,std::uint32_t lanes=1)
        :device(target),state_len(STATE_LEN),output_dim(MOVE_COUNT),padded_output(MOVE_COUNT) {
        auto manifest=nlohmann::json::parse(read_text_exact(directory/"ensemble.json"));
        cudaDeviceProp properties{};
        if(cudaGetDeviceProperties(&properties,target.index())!=cudaSuccess)
            throw std::runtime_error("cannot inspect ensemble GPU");
        use_cutlass=properties.major>=8;
        if(const char* backend=std::getenv("BEAM_ENSEMBLE_BACKEND")) {
            const std::string selected(backend);
            if(selected=="libtorch") use_cutlass=false;
            else if(selected=="cutlass") {
                if(properties.major<8) throw std::runtime_error("ensemble CUTLASS requires SM80 or newer");
                use_cutlass=true;
            } else throw std::runtime_error("invalid ensemble inference backend");
        }
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
                if(head.mlp->state_len!=state_len||(head.mlp->output_dim!=output_dim&&head.mlp->output_dim!=1)||head.mlp->dtype!=torch::kFloat16)
                    throw std::runtime_error("ensemble MLP requires matching Q head and FP16");
                head.weight=head.mlp->output.weight.transpose(0,1).contiguous();
                head.bias=head.mlp->output.bias.to(torch::kFloat32).contiguous();
            } else if(head.family=="piece_transformer") {
                head.transformer=std::make_unique<PieceTransformerLibTorch>(path,target);
                if(head.transformer->state_len!=state_len||(head.transformer->output_dim!=output_dim&&head.transformer->output_dim!=1)||head.transformer->dtype!=torch::kFloat16)
                    throw std::runtime_error("ensemble transformer requires matching Q head and FP16");
                head.weight=head.transformer->output_weight_kxh.contiguous();
                head.bias=head.transformer->output_bias.to(torch::kFloat32).contiguous();
            } else if(head.family=="pancake_gnn") {
                head.gnn=std::make_unique<PancakeGnnNative>(path,target,use_cutlass);
                head.weight=head.gnn->output_weight.transpose(0,1).contiguous();
                head.bias=head.gnn->output_bias.to(torch::kFloat32).contiguous();
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
            head.output_dim=head.weight.size(1);
            if(padded_output!=output_dim) {
                auto w=torch::zeros({head.weight.size(0),padded_output},head.weight.options());
                w.narrow(1,0,output_dim).copy_(head.weight);head.weight=w;
                auto b=torch::zeros({padded_output},head.bias.options());
                b.narrow(0,0,output_dim).copy_(head.bias);head.bias=b;
            }
            heads.push_back(std::move(head));
        }
        if(!max_batch||!lanes) throw std::runtime_error("ensemble batch and lanes must be positive");
        max_parents=max_batch;
        inference_micro=max_batch;
        if(const char* value=std::getenv("BEAM_ENSEMBLE_INFERENCE_MICRO")) {
            auto parsed=std::stoul(value);
            if(!parsed||parsed>UINT32_MAX) throw std::runtime_error("invalid ensemble inference microbatch");
            inference_micro=std::min<std::uint32_t>(parsed,max_batch);
        }
        auto options=torch::TensorOptions().device(device).dtype(torch::kFloat32);
        for(std::uint32_t i=0;i<lanes;++i) workspaces.push_back({
            torch::empty({inference_micro,padded_output},options),
            torch::empty({inference_micro,padded_output},options.dtype(torch::kInt32))});
        if(cudaDeviceSynchronize()!=cudaSuccess) throw std::runtime_error("ensemble startup synchronization failed");
    }
    void score(const torch::Tensor& states,std::uint32_t* destination,std::uint32_t* error,
               cudaStream_t stream,std::uint32_t lane=0,const std::uint8_t* generators=nullptr) const {
        if(states.size(0)>max_parents) throw std::runtime_error("ensemble outer slot exceeds admitted parent capacity");
        for(std::int64_t offset=0;offset<states.size(0);offset+=inference_micro) {
            auto count=std::min<std::int64_t>(inference_micro,states.size(0)-offset);
            score_chunk(states.narrow(0,offset,count),destination+offset*output_dim,error,stream,lane,generators);
        }
    }
    void score_chunk(const torch::Tensor& states,std::uint32_t* destination,std::uint32_t* error,
               cudaStream_t stream,std::uint32_t lane,const std::uint8_t* generators) const {
        auto rows=states.size(0);
        if(lane>=workspaces.size()||rows>workspaces[lane].sum.size(0))
            throw std::runtime_error("ensemble score exceeds preallocated lane workspace");
        auto options=states.options().dtype(torch::kFloat32);
        auto sum=workspaces[lane].sum.narrow(0,0,rows);
        auto padded_keys=workspaces[lane].keys.narrow(0,0,rows);
        auto key_ptr=padded_output==output_dim?destination:reinterpret_cast<std::uint32_t*>(padded_keys.data_ptr<int>());
        torch::Tensor children;
        if(std::any_of(heads.begin(),heads.end(),[](const auto& h){return h.output_dim==1;})) {
            if(!generators) throw std::runtime_error("scalar ensemble head requires ordered graph generators");
            children=scalar_children(states,generators,MOVE_COUNT,STATE_LEN,STATE_STORAGE_LEN);
        }
        for(std::size_t i=0;i<heads.size();++i) {
            const auto& head=heads[i];auto features=head.features(head.output_dim==1?children:states);
            if(!use_cutlass) {
                auto value=torch::matmul(features.to(torch::kFloat32),head.weight.to(torch::kFloat32))+head.bias;
                auto view=sum.reshape({features.size(0),head.output_dim});
                if(i==0) view.copy_(value*head.coefficient);else view.add_(value,head.coefficient);
                if(i+1==heads.size()) {
                    auto finite=torch::isfinite(sum).all();
                    auto flag=torch::from_blob(error,{1},options.dtype(torch::kInt32));
                    flag.bitwise_or_(torch::logical_not(finite).to(torch::kInt32));
                    auto output=torch::from_blob(destination,{rows,output_dim},options.dtype(torch::kInt32));
                    output.copy_(torch::round(torch::clamp(sum,0.0,kScoreMaxQ)*kScoreScale).to(torch::kInt32),true);
                }
                continue;
            }
            stream1_ensemble_head_fp16_cuda(reinterpret_cast<const __half*>(features.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(head.weight.data_ptr<at::Half>()),head.bias.data_ptr<float>(),
                sum.data_ptr<float>(),key_ptr,features.size(0),head.weight.size(0),head.output_dim,
                head.coefficient,i==0,i+1==heads.size(),error,stream);
        }
        if(padded_output!=output_dim) {
            auto output=torch::from_blob(destination,{rows,output_dim},options.dtype(torch::kInt32));
            output.copy_(padded_keys.narrow(1,0,output_dim),true);
        }
    }
};
}
