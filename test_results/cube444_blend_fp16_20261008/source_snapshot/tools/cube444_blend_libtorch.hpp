#pragma once
#include "stream1_transformer_libtorch_backend.hpp"
#include "../cuda/stream1_blend_readout.hpp"
#include "../third_party/picosha2/picosha2.h"
#include "../third_party/nlohmann/json.hpp"
#include <map>
#include <sstream>

namespace beam::stream1_libtorch {
// Immutable FP32/int32 source tensors; checked FP16 conversion on load by
// default. Explicit FP32 is retained for the numerical oracle. No BN re-folding
// or model substitution. Reductions and final readout accumulation use FP32.
struct Cube444Blend {
    std::map<std::string,torch::Tensor> tensors;
    torch::Device device;
    torch::ScalarType dtype;
    torch::Tensor readout_bias;
    std::vector<torch::Tensor> folded_slots;
    explicit Cube444Blend(const fs::path& dir,const torch::Device& target,
                          torch::ScalarType precision=torch::kFloat16) : device(target),dtype(precision) {
        if(dtype!=torch::kFloat16 && dtype!=torch::kFloat32)
            throw std::runtime_error("blend execution precision must be fp16 or fp32");
        at::globalContext().setAllowTF32CuBLAS(false);
        at::globalContext().setAllowTF32CuDNN(false);
        auto identity=nlohmann::json::parse(read_text_exact(dir/"blend.json"));
        if(identity.at("backend")!="cube444_q_blend" || identity.at("dtype")!="fp32" ||
           identity.at("transformer_weight")!=0.6 || identity.at("mlp_weight")!=0.4)
            throw std::runtime_error("invalid blend identity or coefficients");
        for(auto it=identity.at("files").begin();it!=identity.at("files").end();++it) {
            const auto name=it.key();
            if(fs::path(name).filename()!=fs::path(name)) throw std::runtime_error("unsafe blend binding");
            auto bytes=read_text_exact(dir/name);
            if(picosha2::hash256_hex_string(bytes)!=it.value().get<std::string>())
                throw std::runtime_error("blend tensor checksum mismatch: "+name);
        }
        std::ifstream table(dir / "tensors.tsv");
        if (!table) throw std::runtime_error("blend tensors.tsv missing");
        std::string line;
        while (std::getline(table,line)) {
            std::istringstream row(line);
            std::string key,file,format,shape;
            if (!std::getline(row,key,'\t') || !std::getline(row,file,'\t') ||
                !std::getline(row,format,'\t') || !std::getline(row,shape))
                throw std::runtime_error("invalid blend tensor row");
            if (file.size()!=68 || file.substr(64)!=".bin" ||
                file.substr(0,64).find_first_not_of("0123456789abcdef")!=std::string::npos)
                throw std::runtime_error("unsafe blend tensor filename");
            if (format!="float32" && format!="int32") throw std::runtime_error("unsupported blend dtype");
            if(!identity.at("files").contains(file)) throw std::runtime_error("unbound blend tensor");
            auto dims=parse_csv_i64(shape);
            std::uint64_t size=1;
            for (auto d:dims) {
                if (d<=0 || d>1000000 || size>100000000/static_cast<std::uint64_t>(d))
                    throw std::runtime_error("invalid blend tensor shape");
                size*=d;
            }
            auto tensor=load_tensor(dir/file,dims,format=="float32"?torch::kFloat32:torch::kInt32,device);
            if (!torch::isfinite(tensor).all().item<bool>()) throw std::runtime_error("nonfinite blend weights");
            if(format=="float32") {
                tensor=tensor.to(dtype);
                if(!torch::isfinite(tensor).all().item<bool>()) throw std::runtime_error("blend weights overflow execution precision");
            }
            if (!tensors.emplace(key,tensor).second) throw std::runtime_error("duplicate blend tensor");
        }
        require("s3/output_layer_w",{256,24}); require("s3/output_layer_b",{24});
        require("s3/piece_positions",{168}); require("s3/piece_mask",{1,56,3,1});
        require("mlp/input_w",{576,1536}); require("mlp/hidden_w",{1536,2304});
        require("mlp/out_w",{2304,24}); require("mlp/out_b",{24});
        require("mlp/position_offsets",{96});
        readout_bias=at("s3/output_layer_b").to(torch::kFloat32).contiguous();
        auto offsets=torch::arange(96,at("mlp/position_offsets").options())*6;
        if(!torch::equal(offsets,at("mlp/position_offsets"))) throw std::runtime_error("invalid position encoding");
        for(int i=0;i<4;++i) {
            require("mlp/blocks/"+std::to_string(i)+"/l1_w",{2304,2304});
            require("mlp/blocks/"+std::to_string(i)+"/l2_w",{2304,2304});
        }
        for(int j=0;j<3;++j)
            folded_slots.push_back(torch::matmul(at("s3/local_value_embedding").narrow(0,j*6,6),
                                                at("s3/piece_projection_w").narrow(0,j*256,256)));
        // Publish startup tables before nonblocking inference streams consume them.
        if(cudaDeviceSynchronize()!=cudaSuccess) throw std::runtime_error("blend initialization synchronization failed");
    }
    const torch::Tensor& at(const std::string& key) const { return tensors.at(key); }
    void require(const std::string& key,std::vector<std::int64_t> shape) const {
        if (at(key).sizes().vec()!=shape) throw std::runtime_error("blend shape mismatch: "+key);
    }
    static torch::Tensor ln(const torch::Tensor& x,const torch::Tensor& w,const torch::Tensor& b) {
        auto fp=x.to(torch::kFloat32);
        auto mean=fp.mean(-1,true);
        auto centered=fp-mean;
        auto variance=(centered*centered).mean(-1,true);
        return (centered*torch::rsqrt(variance+1e-5)*w.to(torch::kFloat32)+b.to(torch::kFloat32)).to(x.scalar_type());
    }
    torch::Tensor affine(const torch::Tensor& x,const std::string& p) const {
        return torch::matmul(x,at(p+"_w"))+at(p+"_b");
    }
    torch::Tensor bn(const torch::Tensor& x,const std::string& p) const {
        return x*at(p+"/0")+at(p+"/1");
    }
    torch::Tensor mlp(const torch::Tensor& states) const {
        auto labels=states.narrow(1,0,96).to(torch::kLong);
        // Bounded 576-column encoding, never [B,96,1536] gathered embeddings.
        auto encoded=torch::one_hot(labels,6).flatten(1).to(dtype);
        auto h=torch::matmul(encoded,at("mlp/input_w"));
        h=at::relu(bn(h+at("mlp/input_b"),"mlp/input_bn"));
        h=at::relu(bn(affine(h,"mlp/hidden"),"mlp/hidden_bn"));
        for(int i=0;i<4;++i) {
            auto p="mlp/blocks/"+std::to_string(i);
            auto z=at::relu(bn(affine(h,p+"/l1"),p+"/bn1"));
            h=at::relu(h+bn(affine(z,p+"/l2"),p+"/bn2"));
        }
        return affine(h,"mlp/out").to(torch::kFloat32);
    }
    torch::Tensor transformer_cls(const torch::Tensor& states) const {
        auto b=states.size(0);
        auto labels=states.narrow(1,0,96).to(torch::kLong);
        auto positions=at("s3/piece_positions").to(torch::kLong);
        auto values=labels.index_select(1,positions).reshape({b,56,3});
        auto h=at("s3/piece_projection_b").reshape({1,1,256}).expand({b,56,256}).clone();
        for(int j=0;j<3;++j) {
            auto gathered=folded_slots[j].index_select(0,values.select(2,j).reshape({-1})).reshape({b,56,256});
            h=h+gathered*at("s3/piece_mask").reshape({1,56,3}).narrow(2,j,1);
        }
        h=h+at("s3/piece_position_embedding").reshape({1,56,256});
        h=h+at("s3/piece_type_embedding").index_select(0,at("s3/piece_types").to(torch::kLong)).unsqueeze(0);
        h=torch::cat({at("s3/cls_token").reshape({1,1,256}).expand({b,1,256}),h},1);
        h=ln(h,at("s3/input_norm_w"),at("s3/input_norm_b"));
        for(int i=0;i<4;++i) {
            auto p="s3/blocks/"+std::to_string(i);
            auto x=ln(h,at(p+"/norm1_w"),at(p+"/norm1_b"));
            auto project=[&](const std::string& axis) {
                return (torch::matmul(x,at(p+"/w"+axis))+at(p+"/b"+axis))
                    .reshape({b,x.size(1),8,32}).permute({0,2,1,3});
            };
            auto q=project("q"),k=project("k"),v=project("v");
            if(i==3) q=q.narrow(2,0,1);
            auto prob=torch::softmax(torch::matmul(q,k.transpose(-2,-1)).to(torch::kFloat32)/std::sqrt(32.f),-1).to(dtype);
            auto context=torch::matmul(prob,v).permute({0,2,1,3}).reshape({b,q.size(2),256});
            h=(i==3?h.narrow(1,0,1):h)+torch::matmul(context,at(p+"/out_w"))+at(p+"/out_b");
            auto ff=ln(h,at(p+"/norm2_w"),at(p+"/norm2_b"));
            ff=at::relu(affine(ff,p+"/ff1"));
            h=h+affine(ff,p+"/ff2");
        }
        return ln(h.select(1,0),at("s3/output_norm_w"),at("s3/output_norm_b")).contiguous();
    }
    torch::Tensor transformer(const torch::Tensor& states) const {
        return affine(transformer_cls(states),"s3/output_layer");
    }
    void readout(const torch::Tensor& cls,const torch::Tensor& mlp_q,std::uint32_t* keys,
                 std::uint32_t rows,std::uint32_t* error,cudaStream_t stream,bool only=false) const {
        const auto* q=only?nullptr:mlp_q.data_ptr<float>();
        if(dtype==torch::kFloat16)
            stream1_blend_readout_fp16_cuda(reinterpret_cast<const __half*>(cls.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(at("s3/output_layer_w").data_ptr<at::Half>()),
                readout_bias.data_ptr<float>(),q,keys,rows,error,stream,only);
        else stream1_blend_readout_cuda(cls.data_ptr<float>(),at("s3/output_layer_w").data_ptr<float>(),
                readout_bias.data_ptr<float>(),q,keys,rows,error,stream,only);
    }
};
}
