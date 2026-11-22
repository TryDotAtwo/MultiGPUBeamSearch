
#include "stream1_gnn_libtorch.hpp"
#include <torch/library.h>
#include <memory>
namespace {
std::unique_ptr<beam::stream1_libtorch::PancakeGnnNative> models[2];
void setup(const std::string& path,int64_t device) {
 for(int i=0;i<2;++i) models[i]=std::make_unique<beam::stream1_libtorch::PancakeGnnNative>(path,torch::Device(torch::kCUDA,device),i==1);
}
torch::Tensor run(const torch::Tensor& input,bool cutlass) {
 auto& model=*models[cutlass?1:0];
 return torch::linear(model.features(input).to(torch::kFloat32),model.output_weight.to(torch::kFloat32),model.output_bias.to(torch::kFloat32));
}
}
TORCH_LIBRARY_FRAGMENT(multigpubeamsearch_gnn,m) {
 m.def("large_setup(str directory, int device) -> ()");m.impl("large_setup",TORCH_FN(setup));
 m.def("large_run(Tensor input, bool cutlass) -> Tensor");m.impl("large_run",TORCH_FN(run));
}
