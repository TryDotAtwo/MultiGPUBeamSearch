#include <torch/torch.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "../cuda/stream1_gnn_projection.hpp"
#include <climits>

namespace {
torch::Tensor projection(const torch::Tensor& input,const torch::Tensor& weight,
                         const std::optional<torch::Tensor>& bias,bool cutlass) {
    TORCH_CHECK(input.dim()>=2 && weight.dim()==2 && input.size(-1)==weight.size(1),
                "invalid GNN projection shape");
    // Tiny node-coordinate projections (K=2) remain ATen in either backend.
    if(!cutlass || weight.size(1)%8 || weight.size(0)%8)
        return at::linear(input,weight,bias);
    TORCH_CHECK(input.is_cuda() && input.scalar_type()==torch::kFloat16 &&
                weight.device()==input.device() && weight.scalar_type()==torch::kFloat16,
                "GNN CUTLASS projection requires same-device CUDA FP16 tensors");
    c10::cuda::CUDAGuard guard(input.device());
    cudaDeviceProp properties{};
    TORCH_CHECK(cudaGetDeviceProperties(&properties,input.get_device())==cudaSuccess && properties.major>=8,
                "GNN CUTLASS projection requires SM80 or newer");
    auto a=input.contiguous();auto w=weight.contiguous();
    auto shape=input.sizes().vec();shape.back()=weight.size(0);
    auto output=torch::empty(shape,input.options());
    const auto rows=input.numel()/input.size(-1);
    TORCH_CHECK(rows<=INT_MAX && weight.size(0)<=INT_MAX && weight.size(1)<=INT_MAX,
                "GNN projection exceeds integer GEMM dimensions");
    torch::Tensor b;
    if(bias) {
        TORCH_CHECK(bias->device()==input.device() && bias->scalar_type()==torch::kFloat16 &&
                    bias->dim()==1 && bias->size(0)==weight.size(0),"invalid GNN projection bias");
        b=bias->contiguous();
    }
    beam::stream1_gnn_projection(reinterpret_cast<const __half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(w.data_ptr<at::Half>()),
        b.defined()?reinterpret_cast<const __half*>(b.data_ptr<at::Half>()):nullptr,
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),int(rows),int(weight.size(1)),int(weight.size(0)),
        c10::cuda::getCurrentCUDAStream(input.get_device()).stream());
    return output;
}
}
TORCH_LIBRARY(multigpubeamsearch_gnn,m) {
    m.def("linear(Tensor input, Tensor weight, Tensor? bias, bool cutlass) -> Tensor");
    m.impl("linear",TORCH_FN(projection));
}
