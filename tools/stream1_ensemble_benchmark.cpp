#include "stream1_ensemble_libtorch.hpp"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/core/InferenceMode.h>
#include <iostream>

int main(int argc,char** argv) {
    if(argc!=5) throw std::runtime_error("usage: ensemble-benchmark ensemble-dir batch parents device");
    const unsigned batch=std::stoul(argv[2]),parents=std::stoul(argv[3]);
    const int device=std::stoi(argv[4]);
    if(!batch||!parents||batch>parents) throw std::runtime_error("invalid benchmark workload");
    c10::cuda::CUDAGuard guard(device);c10::InferenceMode inference;
    torch::NoGradGuard no_grad;
    beam::stream1_libtorch::NativeEnsemble model(argv[1],torch::Device(torch::kCUDA,device));
    auto stream=c10::cuda::getCurrentCUDAStream(device).stream();
    auto states=torch::zeros({batch,beam::STATE_LEN},torch::TensorOptions().device(torch::kCUDA,device).dtype(torch::kUInt8));
    auto keys=torch::empty({batch,beam::MOVE_COUNT},states.options().dtype(torch::kInt32));
    auto flag=torch::zeros({1},keys.options());
    auto run=[&](unsigned count) {
        model.score(states.narrow(0,0,count),reinterpret_cast<std::uint32_t*>(keys.data_ptr<int>()),
                    reinterpret_cast<std::uint32_t*>(flag.data_ptr<int>()),stream);
    };
    for(int i=0;i<3;++i) run(batch);
    cudaDeviceSynchronize();
    cudaEvent_t start,end;cudaEventCreate(&start);cudaEventCreate(&end);
    nlohmann::json samples=nlohmann::json::array();
    for(int repeat=0;repeat<7;++repeat) {
        cudaEventRecord(start,stream);
        for(unsigned offset=0;offset<parents;offset+=batch) run(std::min(batch,parents-offset));
        cudaEventRecord(end,stream);cudaEventSynchronize(end);
        float ms=0;cudaEventElapsedTime(&ms,start,end);samples.push_back(ms/1000.0);
    }
    auto error=flag.item<int>();
    auto stats=c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    std::size_t free,total;cudaMemGetInfo(&free,&total);
    nlohmann::json output={{"batch",batch},{"parents",parents},{"device",device},
        {"seconds",samples},{"numeric_error",error},{"model_count",model.heads.size()},
        {"torch_reserved_peak_bytes",stats.reserved_bytes[0].peak},
        {"free_bytes_after",free},{"score_input","synthetic_zero_labels"}};
    std::cout<<output.dump()<<std::endl;
    cudaEventDestroy(start);cudaEventDestroy(end);
    return error?2:0;
}
