#include "stream1_ensemble_libtorch.hpp"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/core/InferenceMode.h>
#include <iostream>

int main(int argc,char** argv) {
    if(argc!=5 && argc!=6) throw std::runtime_error("usage: ensemble-benchmark ensemble-dir batch parents device [--session]");
    const bool session=argc==6 && std::string(argv[5])=="--session";
    if(argc==6 && !session) throw std::runtime_error("unknown benchmark option");
    unsigned batch=std::stoul(argv[2]),parents=std::stoul(argv[3]);
    const unsigned capacity=batch;
    const int device=std::stoi(argv[4]);
    if(!batch||!parents||batch>parents) throw std::runtime_error("invalid benchmark workload");
    c10::cuda::CUDAGuard guard(device);c10::InferenceMode inference;
    torch::NoGradGuard no_grad;
    beam::stream1_libtorch::NativeEnsemble model(argv[1],torch::Device(torch::kCUDA,device),batch);
    auto stream=c10::cuda::getCurrentCUDAStream(device).stream();
    auto states=torch::zeros({batch,beam::STATE_LEN},torch::TensorOptions().device(torch::kCUDA,device).dtype(torch::kUInt8));
    auto keys=torch::empty({batch,beam::MOVE_COUNT},states.options().dtype(torch::kInt32));
    auto flag=torch::zeros({1},keys.options());
    auto identity=nlohmann::json::parse(beam::stream1_libtorch::read_text_exact(
        std::filesystem::path(argv[1])/"ensemble.json"));
    if(identity.contains("calibration_states")) {
        const auto& samples=identity.at("calibration_states");
        if(!samples.is_array()||samples.empty()) throw std::runtime_error("missing calibration states");
        std::vector<std::uint8_t> host(batch*beam::STATE_LEN);
        for(unsigned row=0;row<batch;++row) {
            const auto& sample=samples[row%samples.size()];
            if(sample.size()!=beam::STATE_LEN) throw std::runtime_error("calibration state shape mismatch");
            for(unsigned j=0;j<beam::STATE_LEN;++j) host[row*beam::STATE_LEN+j]=sample[j].get<std::uint8_t>();
        }
        states.copy_(torch::from_blob(host.data(),{batch,beam::STATE_LEN},states.options().device(torch::kCPU)));
    }
    torch::Tensor generators;
    if(identity.contains("generators")) {
        const auto& moves=identity.at("generators");
        if(moves.size()!=beam::MOVE_COUNT) throw std::runtime_error("calibration move count mismatch");
        std::vector<std::uint8_t> host(beam::MOVE_COUNT*beam::STATE_STORAGE_LEN,0);
        for(unsigned i=0;i<beam::MOVE_COUNT;++i) {
            if(moves[i].size()!=beam::STATE_LEN) throw std::runtime_error("calibration generator shape mismatch");
            for(unsigned j=0;j<beam::STATE_LEN;++j) host[i*beam::STATE_STORAGE_LEN+j]=moves[i][j].get<std::uint8_t>();
        }
        generators=torch::from_blob(host.data(),{beam::MOVE_COUNT,beam::STATE_STORAGE_LEN},states.options().device(torch::kCPU)).to(states.device());
    }
    const auto* generator_ptr=generators.defined()?generators.data_ptr<std::uint8_t>():nullptr;
    const bool full_frontier=identity.contains("calibration_frontier_file");
    if(full_frontier) {
        std::ifstream file(identity.at("calibration_frontier_file").get<std::string>(),std::ios::binary|std::ios::ate);
        const auto bytes=static_cast<std::uint64_t>(parents)*beam::STATE_STORAGE_LEN;
        if(!file || file.tellg()!=static_cast<std::streamoff>(bytes)) throw std::runtime_error("calibration frontier size mismatch");
        std::vector<std::uint8_t> host(bytes);file.seekg(0);file.read(reinterpret_cast<char*>(host.data()),bytes);
        if(!file) throw std::runtime_error("calibration frontier read failed");
        for(std::uint64_t row=0;row<parents;++row) for(unsigned j=beam::STATE_LEN;j<beam::STATE_STORAGE_LEN;++j)
            if(host[row*beam::STATE_STORAGE_LEN+j]) throw std::runtime_error("calibration frontier padding is not zero");
        states=torch::from_blob(host.data(),{parents,beam::STATE_STORAGE_LEN},states.options().device(torch::kCPU)).to(states.device()).narrow(1,0,beam::STATE_LEN);
    }
    auto run=[&](unsigned count,unsigned offset=0) {
        model.score(states.narrow(0,offset,count),reinterpret_cast<std::uint32_t*>(keys.data_ptr<int>()),
                    reinterpret_cast<std::uint32_t*>(flag.data_ptr<int>()),stream,0,generator_ptr);
    };
    if(session) std::cout<<nlohmann::json({{"ready",true},{"device",device},{"capacity",capacity}}).dump()<<std::endl;
    do {
    if(session) {
        std::string command;
        if(!std::getline(std::cin,command)) break;
        auto request=nlohmann::json::parse(command);
        if(request.value("stop",false)) break;
        batch=request.at("batch").get<unsigned>();parents=request.at("parents").get<unsigned>();
        if(!batch || batch>capacity || batch>parents) throw std::runtime_error("invalid session workload");
        if(full_frontier && parents!=states.size(0)) throw std::runtime_error("session must preserve its exact verified frontier");
        model.inference_micro=batch;
        flag.zero_();
        c10::cuda::CUDACachingAllocator::resetPeakStats(device);
    }
    run(batch);cudaDeviceSynchronize();
    auto reference=torch::zeros({batch,beam::MOVE_COUNT},states.options().dtype(torch::kFloat32));
    torch::Tensor children;
    for(const auto& head:model.heads) {
        if(head.output_dim==1 && !children.defined())
            children=beam::stream1_libtorch::scalar_children(states.narrow(0,0,batch),generator_ptr,beam::MOVE_COUNT,beam::STATE_LEN,beam::STATE_STORAGE_LEN);
        auto features=head.features(head.output_dim==1?children:states.narrow(0,0,batch)).to(torch::kFloat32);
        auto score=(torch::matmul(features,head.weight.to(torch::kFloat32))+head.bias).reshape({batch,beam::MOVE_COUNT});
        reference.add_(score,head.coefficient);
    }
    if(!torch::isfinite(reference).all().item<bool>()) throw std::runtime_error("nonfinite FP32 readout oracle");
    auto expected=torch::round(torch::clamp(reference,0.0,beam::SCORE_MAX_Q)*beam::SCORE_SCALE).to(torch::kInt32);
    auto max_key_error=(keys.narrow(0,0,batch).to(torch::kInt64)-expected.to(torch::kInt64)).abs().max().item<std::int64_t>();
    if(max_key_error>2) throw std::runtime_error("native ensemble disagrees with FP32 readout oracle");
    for(int i=0;i<3;++i) run(batch);
    cudaDeviceSynchronize();
    cudaEvent_t start,end;cudaEventCreate(&start);cudaEventCreate(&end);
    nlohmann::json samples=nlohmann::json::array();
    for(int repeat=0;repeat<7;++repeat) {
        cudaEventRecord(start,stream);
        for(unsigned offset=0;offset<parents;offset+=batch) run(std::min(batch,parents-offset),full_frontier?offset:0);
        cudaEventRecord(end,stream);cudaEventSynchronize(end);
        float ms=0;cudaEventElapsedTime(&ms,start,end);samples.push_back(ms/1000.0);
    }
    auto error=flag.item<int>();
    auto stats=c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    std::size_t free,total;cudaMemGetInfo(&free,&total);
    nlohmann::json output={{"batch",batch},{"parents",parents},{"device",device},
        {"seconds",samples},{"numeric_error",error},{"model_count",model.heads.size()},
        {"readout_oracle_max_key_error",max_key_error},{"correctness_passed",true},
        {"torch_reserved_peak_bytes",stats.reserved_bytes[0].peak},
        {"free_bytes_after",free},{"score_input",full_frontier?"full_frontier_file":identity.contains("calibration_states")?"graph_states":"synthetic_zero_labels"}};
    std::cout<<output.dump()<<std::endl;
    cudaEventDestroy(start);cudaEventDestroy(end);
    if(error) return 2;
    } while(session);
    return 0;
}
