#include "../tools/stream1_gnn_libtorch.hpp"
#include <torch/library.h>
namespace {
torch::Tensor features(const torch::Tensor& states,const std::string& directory,bool cutlass) {
    beam::stream1_libtorch::PancakeGnnNative model(directory,states.device(),cutlass);
    return model.features(states);
}
}
TORCH_LIBRARY_FRAGMENT(multigpubeamsearch_gnn,m) {
    m.def("probe_features(Tensor states, str directory, bool cutlass) -> Tensor");
    m.impl("probe_features",TORCH_FN(features));
}
