// Diagnostic: production scripted backend, slicing, quantizer and score-ring copy.
#include "stream1_transformer_libtorch_backend.hpp"
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc, char** argv) {
 try {
  if (argc != 6 && argc != 7) throw std::runtime_error("args: export corpus micro gpu repeats [concurrency]");
  const int micro=std::stoi(argv[3]), gpu=std::stoi(argv[4]), repeats=std::stoi(argv[5]);
  const int concurrency=argc==7 ? std::stoi(argv[6]) : 1;
  constexpr int outer=8192, width=160;
  if(micro<1 || micro>outer || repeats<3 || (concurrency!=1 && concurrency!=2 && concurrency!=4)) throw std::runtime_error("invalid bounds");
  c10::InferenceMode guard(true);
  torch::Device device(torch::kCUDA,gpu);
  c10::cuda::CUDAGuard device_guard(device);
  beam::stream1_libtorch::PieceTransformerLibTorch model(argv[1],device);
  std::vector<uint8_t> bytes(outer*width);
  std::ifstream corpus(argv[2],std::ios::binary);
  corpus.read(reinterpret_cast<char*>(bytes.data()),bytes.size());
  if(corpus.gcount()!=static_cast<std::streamsize>(bytes.size())) throw std::runtime_error("corpus length");
  auto states=torch::from_blob(bytes.data(),{outer,width},torch::kUInt8).clone().to(device);
  std::vector<torch::Tensor> scores;
  std::vector<c10::cuda::CUDAStream> streams;
  for(int lane=0;lane<concurrency;++lane) {
   scores.push_back(torch::empty({outer,30},torch::TensorOptions().dtype(torch::kInt32).device(device)));
   streams.push_back(c10::cuda::getStreamFromPool(false,gpu));
  }
  auto reference=model.forward(states.narrow(0,0,128));
  torch::cuda::synchronize(gpu);
  auto run=[&]() {
   for(int lane=0;lane<concurrency;++lane) {
   c10::cuda::CUDAStreamGuard stream_guard(streams[lane]);
   for(int begin=0;begin<outer;begin+=micro) {
    int count=std::min(micro,outer-begin);
    auto logits=model.forward(states.narrow(0,begin,count));
    auto keys=torch::round(torch::clamp(logits.to(torch::kFloat32),0.0,beam::stream1_libtorch::kScoreMaxQ)*beam::stream1_libtorch::kScoreScale).to(torch::kInt32).contiguous();
    scores[lane].narrow(0,begin,count).copy_(keys,true);
   }
   }
  };
  for(int i=0;i<2;++i) run();
  torch::cuda::synchronize(gpu);
  std::vector<double> times;
  for(int i=0;i<repeats;++i) {
   auto start=std::chrono::steady_clock::now(); run(); torch::cuda::synchronize(gpu);
   times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
  }
  std::vector<torch::Tensor> parts;
  for(int begin=0;begin<128;begin+=micro) parts.push_back(model.forward(states.narrow(0,begin,std::min(micro,outer-begin))).narrow(0,0,std::min(micro,128-begin)));
  auto actual=torch::cat(parts,0);
  double error=(actual-reference).abs().max().item<double>();
  bool finite=torch::isfinite(actual).all().item<bool>();
  double agreement=(actual.argmin(1)==reference.argmin(1)).to(torch::kFloat32).mean().item<double>();
  if(!finite || error>0.3) throw std::runtime_error("microbatch parity failed");
  auto checksum=scores[0].to(torch::kInt64).sum().item<int64_t>();
  for(int lane=1;lane<concurrency;++lane) if(!torch::equal(scores[0],scores[lane])) throw std::runtime_error("concurrent score buffer mismatch");
  std::cout<<"{\"gpu\":"<<gpu<<",\"micro\":"<<micro<<",\"concurrency\":"<<concurrency<<",\"outer\":8192,\"finite\":true,\"max_abs_vs_128\":"<<error<<",\"top1_agreement\":"<<agreement<<",\"score_checksum\":"<<checksum<<",\"seconds\":[";
  for(size_t i=0;i<times.size();++i) std::cout<<(i?",":"")<<times[i];
  std::cout<<"]}"<<std::endl;
  return 0;
 } catch(const std::exception& e) { std::cerr<<e.what()<<std::endl;return 1; }
}
