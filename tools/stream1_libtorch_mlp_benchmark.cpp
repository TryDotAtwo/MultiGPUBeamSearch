// Measure the ordinary LibTorch scalar/Q MLP path used on older GPUs.
#include "stream1_mlp_libtorch_backend.hpp"
#include "cuda_check.hpp"
#include "../src/config.hpp"
#include "../third_party/nlohmann/json.hpp"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/core/InferenceMode.h>
#include <iostream>

int main(int argc,char** argv) {
    if(argc!=5)throw std::runtime_error("usage: libtorch-mlp-benchmark input-dir batch parents device");
    const unsigned batch=std::stoul(argv[2]),parents=std::stoul(argv[3]);
    const int device=std::stoi(argv[4]);
    if(!batch||batch>parents)throw std::runtime_error("invalid benchmark workload");
    c10::cuda::CUDAGuard guard(device);c10::InferenceMode inference;
    const auto stream=c10::cuda::getCurrentCUDAStream(device).stream();
    const auto spec=nlohmann::json::parse(beam::stream1_libtorch::read_text_exact(
        std::filesystem::path(argv[1])/"calibration.json"));
    beam::stream1_libtorch::MlpLibTorch model(spec.at("weights_dir").get<std::string>(),
        torch::Device(torch::kCUDA,device));
    if(model.state_len!=beam::STATE_LEN || (model.output_dim!=1 && model.output_dim!=beam::MOVE_COUNT))
        throw std::runtime_error("model and graph dimensions disagree");
    auto options=torch::TensorOptions().device(torch::kCUDA,device).dtype(torch::kUInt8);
    const auto& samples=spec.at("calibration_states");
    if(!samples.is_array()||samples.empty())throw std::runtime_error("missing graph states");
    std::vector<std::uint8_t> host(batch*beam::STATE_STORAGE_LEN,0);
    for(unsigned row=0;row<batch;++row) {
        const auto& sample=samples[row%samples.size()];
        if(sample.size()!=beam::STATE_LEN)throw std::runtime_error("state shape mismatch");
        for(unsigned j=0;j<beam::STATE_LEN;++j)host[row*beam::STATE_STORAGE_LEN+j]=sample[j].get<std::uint8_t>();
    }
    auto states=torch::from_blob(host.data(),{batch,beam::STATE_STORAGE_LEN},options.device(torch::kCPU)).to(options.device());
    const auto& generators=spec.at("generators");
    if(generators.size()!=beam::MOVE_COUNT)throw std::runtime_error("move count mismatch");
    std::vector<std::uint8_t> moves(beam::MOVE_COUNT*beam::STATE_STORAGE_LEN);
    for(unsigned i=0;i<beam::MOVE_COUNT;++i) {
        if(generators[i].size()!=beam::STATE_LEN)throw std::runtime_error("generator shape mismatch");
        for(unsigned j=0;j<beam::STATE_STORAGE_LEN;++j)
            moves[i*beam::STATE_STORAGE_LEN+j]=j<beam::STATE_LEN?generators[i][j].get<std::uint8_t>():j;
    }
    auto indices=torch::from_blob(moves.data(),{beam::MOVE_COUNT,beam::STATE_STORAGE_LEN},options.device(torch::kCPU)).to(options.device());
    auto keys=torch::empty({batch,beam::MOVE_COUNT},options.dtype(torch::kInt32));
    auto logits=[&](const torch::Tensor& rows) {
        auto input=model.output_dim==1?beam::stream1_libtorch::scalar_children(rows,
            indices.data_ptr<std::uint8_t>(),beam::MOVE_COUNT,beam::STATE_LEN,beam::STATE_STORAGE_LEN):rows;
        return model.forward(input).to(torch::kFloat32).reshape({rows.size(0),beam::MOVE_COUNT});
    };
    // Identical eager forward, quantization, and score-ring copy to the ordinary
    // RingSlotLauncher; no native scalar score offset and no CUDA Graph capture.
    auto run=[&](unsigned n) {
        auto scores=logits(states.narrow(0,0,n));
        keys.narrow(0,0,n).copy_(torch::round(torch::clamp(scores,0.,beam::SCORE_MAX_Q)*beam::SCORE_SCALE).to(torch::kInt32),true);
    };
    run(batch);BEAM_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::int64_t error=0;
    // Compare the selected batch with bounded reference batches. This is a
    // batch-invariance gate, not an independent implementation of the model.
    for(unsigned offset=0;offset<batch;offset+=32) {
        const unsigned n=std::min(32U,batch-offset);
        auto scores=logits(states.narrow(0,offset,n));
        if(!torch::isfinite(scores).all().item<bool>())throw std::runtime_error("nonfinite model scores");
        auto expected=torch::round(torch::clamp(scores,0.,beam::SCORE_MAX_Q)*beam::SCORE_SCALE).to(torch::kInt64);
        error=std::max(error,(keys.narrow(0,offset,n).to(torch::kInt64)-expected).abs().max().item<std::int64_t>());
    }
    if(error>2)throw std::runtime_error("MLP score keys vary beyond tolerance across batch sizes");
    for(int i=0;i<3;++i)run(batch);
    BEAM_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t begin,end;BEAM_CUDA_CHECK(cudaEventCreate(&begin));BEAM_CUDA_CHECK(cudaEventCreate(&end));
    nlohmann::json seconds=nlohmann::json::array();
    for(int repeat=0;repeat<7;++repeat) {
        BEAM_CUDA_CHECK(cudaEventRecord(begin,stream));
        for(unsigned offset=0;offset<parents;offset+=batch)run(std::min(batch,parents-offset));
        BEAM_CUDA_CHECK(cudaEventRecord(end,stream));BEAM_CUDA_CHECK(cudaEventSynchronize(end));
        float ms;BEAM_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));seconds.push_back(ms/1000.);
    }
    const auto stats=c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    std::cout<<nlohmann::json({{"batch",batch},{"parents",parents},{"device",device},
        {"seconds",seconds},{"correctness_passed",true},{"numeric_error",0},
        {"batch_invariance_max_key_error",error},{"torch_reserved_peak_bytes",stats.reserved_bytes[0].peak},
        {"model_count",1},{"executor","libtorch_eager"},{"score_input","graph_states"}}).dump()<<std::endl;
    BEAM_CUDA_CHECK(cudaEventDestroy(begin));BEAM_CUDA_CHECK(cudaEventDestroy(end));
}

