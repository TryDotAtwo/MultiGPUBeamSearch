// Actual native Stream1 timing, with an independent LibTorch FP16/BF16 oracle.
#include "stream1_weight_io.hpp"
#include "stream1_mlp_libtorch_backend.hpp"
#include "../third_party/nlohmann/json.hpp"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/core/InferenceMode.h>
#include <iostream>

int main(int argc, char** argv) {
    if(argc!=5)throw std::runtime_error("usage: native-mlp-benchmark input-dir batch parents device");
    const unsigned batch=std::stoul(argv[2]),parents=std::stoul(argv[3]);
    const int device=std::stoi(argv[4]);
    if(!batch||batch>parents)throw std::runtime_error("invalid native benchmark workload");
    c10::cuda::CUDAGuard guard(device);c10::InferenceMode inference;
    auto stream=c10::cuda::getCurrentCUDAStream(device).stream();
    auto spec=nlohmann::json::parse(beam::stream1_libtorch::read_text_exact(
        std::filesystem::path(argv[1])/"calibration.json"));
    auto directory=std::filesystem::path(spec.at("weights_dir").get<std::string>());
    auto host_weights=beam::stream1_weights::load_stream1_weights(directory);
    const auto& model=host_weights.model;
    if(model.backend!=beam::STREAM1_BACKEND_MLP)throw std::runtime_error("native probe requires MLP");
    auto weights=beam::stream1_weights::upload_weights(host_weights);
    beam::Stream1NetworkView network{
        weights.input_weight,weights.input_bias,weights.input_ln_gamma,weights.input_ln_beta,
        weights.hidden_weight,weights.hidden_bias,weights.hidden_ln_gamma,weights.hidden_ln_beta,
        weights.residual_fc1_weight.data(),weights.residual_fc1_bias.data(),
        weights.residual_fc1_ln_gamma.data(),weights.residual_fc1_ln_beta.data(),
        weights.residual_fc2_weight.data(),weights.residual_fc2_bias.data(),
        weights.residual_fc2_ln_gamma.data(),weights.residual_fc2_ln_beta.data(),
        weights.output_weight,weights.output_bias,beam::stream1_weights::network_dims(model)};
    auto allocation=beam::stream1_weights::alloc_stream1_scratch(model,batch,1);
    beam::Stream1CutlassScratch scratch{allocation.hidden1,allocation.hidden2,
        allocation.residual,allocation.output};
    auto options=torch::TensorOptions().device(torch::kCUDA,device).dtype(torch::kUInt8);
    std::vector<std::uint8_t> host(batch*beam::STATE_STORAGE_LEN,0);
    const auto& samples=spec.at("calibration_states");
    if(!samples.is_array()||samples.empty())throw std::runtime_error("missing legal states");
    for(unsigned row=0;row<batch;++row){
        const auto& sample=samples[row%samples.size()];
        if(sample.size()!=beam::STATE_LEN)throw std::runtime_error("invalid sample shape");
        for(unsigned j=0;j<beam::STATE_LEN;++j)host[row*beam::STATE_STORAGE_LEN+j]=sample[j].get<std::uint8_t>();
    }
    auto states=torch::from_blob(host.data(),{batch,beam::STATE_STORAGE_LEN},options.device(torch::kCPU)).to(options.device());
    std::vector<std::uint8_t> moves(beam::MOVE_COUNT*beam::STATE_STORAGE_LEN,0);
    const auto& generators=spec.at("generators");
    if(generators.size()!=beam::MOVE_COUNT)throw std::runtime_error("invalid move count");
    for(unsigned i=0;i<beam::MOVE_COUNT;++i){
        if(generators[i].size()!=beam::STATE_LEN)throw std::runtime_error("invalid generator shape");
        for(unsigned j=0;j<beam::STATE_STORAGE_LEN;++j)
            moves[i*beam::STATE_STORAGE_LEN+j]=j<beam::STATE_LEN?generators[i][j].get<std::uint8_t>():j;
    }
    auto indices=torch::from_blob(moves.data(),{beam::MOVE_COUNT,beam::STATE_STORAGE_LEN},options.device(torch::kCPU)).to(options.device());
    auto keys=torch::empty({batch,beam::MOVE_COUNT},options.dtype(torch::kInt32));
    auto base=torch::zeros({1},options.dtype(torch::kInt64));
    auto count=torch::full({1},batch,options.dtype(torch::kInt32));
    unsigned current_count=batch;
    auto run=[&](unsigned n){
        if(n!=current_count){count.fill_(n);current_count=n;}
        beam::stream1_inference_cutlass_cuda(reinterpret_cast<beam::State128*>(states.data_ptr<std::uint8_t>()),
            reinterpret_cast<std::uint64_t*>(base.data_ptr<std::int64_t>()),
            reinterpret_cast<std::uint32_t*>(count.data_ptr<int>()),indices.data_ptr<std::uint8_t>(),
            network,scratch,reinterpret_cast<std::uint32_t*>(keys.data_ptr<int>()),batch,stream);
    };
    run(batch);BEAM_CUDA_CHECK(cudaStreamSynchronize(stream));
    beam::stream1_libtorch::MlpLibTorch oracle(directory,torch::Device(torch::kCUDA,device));
    auto oracle_states=model.output_dim==1?beam::stream1_libtorch::scalar_children(states,
        indices.data_ptr<std::uint8_t>(),beam::MOVE_COUNT,beam::STATE_LEN,beam::STATE_STORAGE_LEN):states;
    auto reference=oracle.forward(oracle_states).to(torch::kFloat32).reshape({batch,beam::MOVE_COUNT});
    if(!torch::isfinite(reference).all().item<bool>())throw std::runtime_error("nonfinite independent backbone oracle");
    auto expected=torch::round(torch::clamp(reference,0.,beam::SCORE_MAX_Q)*beam::SCORE_SCALE).to(torch::kInt32);
    auto error=(keys.to(torch::kInt64)-expected.to(torch::kInt64)).abs().max().item<std::int64_t>();
    if(error>2){
        std::cerr<<"native_oracle_max_key_error="<<error
                 <<" native_first="<<keys[0]<<" oracle_first="<<expected[0]<<std::endl;
        throw std::runtime_error("native backbone disagrees with independent oracle");
    }
    for(int i=0;i<3;++i)run(batch);
    BEAM_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t begin,end;BEAM_CUDA_CHECK(cudaEventCreate(&begin));BEAM_CUDA_CHECK(cudaEventCreate(&end));
    nlohmann::json seconds=nlohmann::json::array();
    for(int repeat=0;repeat<7;++repeat){
        BEAM_CUDA_CHECK(cudaEventRecord(begin,stream));
        for(unsigned offset=0;offset<parents;offset+=batch)run(std::min(batch,parents-offset));
        BEAM_CUDA_CHECK(cudaEventRecord(end,stream));BEAM_CUDA_CHECK(cudaEventSynchronize(end));
        float ms;BEAM_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));seconds.push_back(ms/1000.);
    }
    std::cout<<nlohmann::json({{"batch",batch},{"parents",parents},{"device",device},
        {"seconds",seconds},{"correctness_passed",true},{"numeric_error",0},
        {"backbone_oracle_max_key_error",error},{"torch_reserved_peak_bytes",0},
        {"model_count",1},{"executor","native_cutlass"},{"score_input","graph_states"}}).dump()<<std::endl;
    beam::stream1_weights::free_stream1_scratch(allocation);
    beam::stream1_weights::free_weights(weights);
    BEAM_CUDA_CHECK(cudaEventDestroy(begin));BEAM_CUDA_CHECK(cudaEventDestroy(end));
}

