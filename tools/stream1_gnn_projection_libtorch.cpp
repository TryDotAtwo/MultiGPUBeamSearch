#include <torch/torch.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "../cuda/stream1_gnn_projection.hpp"
#include <climits>
#include <cmath>

namespace {
torch::Tensor residual_norm_gelu(const torch::Tensor& input,const torch::Tensor& residual,
    const torch::Tensor& weight,const torch::Tensor& bias,double epsilon) {
    TORCH_CHECK(input.dim()==2 && residual.sizes()==input.sizes(),"invalid GNN normalization shape");
    const auto rows=input.size(0),channels=input.size(1);
    TORCH_CHECK(rows>0 && rows<=INT_MAX && channels>0 && channels<=1024 && std::isfinite(epsilon) && epsilon>0,
        "invalid GNN fused normalization capacity or epsilon");
    TORCH_CHECK(weight.dim()==1 && bias.dim()==1 && weight.numel()==channels && bias.numel()==channels,
        "invalid GNN normalization affine shape");
    for(const auto& tensor:{input,residual,weight,bias})
        TORCH_CHECK(tensor.is_cuda() && tensor.device()==input.device() && tensor.scalar_type()==torch::kFloat16,
            "GNN fused normalization requires same-device CUDA FP16 tensors");
    c10::cuda::CUDAGuard guard(input.device());
    auto a=input.contiguous(),r=residual.contiguous(),w=weight.contiguous(),b=bias.contiguous();
    auto output=torch::empty_like(a);
    const auto half=[](const torch::Tensor& t){return reinterpret_cast<const __half*>(t.data_ptr<at::Half>());};
    beam::stream1_gnn_residual_norm_gelu(half(a),half(r),half(w),half(b),reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        int(rows),int(channels),float(epsilon),c10::cuda::getCurrentCUDAStream(input.get_device()).stream());
    return output;
}
torch::Tensor aggregate(const torch::Tensor& left,const torch::Tensor& right,
    const torch::Tensor& edge,const torch::Tensor& attention,const torch::Tensor& bias,
    const torch::Tensor& indices,const torch::Tensor& types) {
    TORCH_CHECK(left.dim()==3 && left.size(1)==4 && right.sizes()==left.sizes(),"invalid GAT node projections");
    const auto nodes=left.size(0),channels=left.size(2),count=types.numel();
    TORCH_CHECK(nodes>0 && nodes<=INT_MAX && count>0 && count<=INT_MAX && channels>0 && channels<=1024,
        "fused GAT capacity requires positive nodes/edges and channels<=1024");
    TORCH_CHECK(edge.dim()==3 && edge.size(0)==3 && edge.size(1)==4 && edge.size(2)==channels && attention.numel()==4*channels && bias.numel()==channels,
        "invalid compact GAT edge/attention/bias shape");
    TORCH_CHECK(indices.dim()==2 && indices.size(0)==2 && indices.size(1)==count && types.dim()==1,
        "invalid GAT edge indices/types");
    for(const auto& tensor:{left,right,edge,attention,bias})
        TORCH_CHECK(tensor.is_cuda() && tensor.device()==left.device() && tensor.scalar_type()==torch::kFloat16,
            "fused GAT requires same-device CUDA FP16 tensors");
    for(const auto& tensor:{indices,types})
        TORCH_CHECK(tensor.device()==left.device() && tensor.scalar_type()==torch::kInt64,
            "fused GAT requires same-device int64 graph indices");
    c10::cuda::CUDAGuard guard(left.device());
    auto l=left.contiguous(),r=right.contiguous(),e=edge.contiguous(),a=attention.contiguous(),b=bias.contiguous();
    auto ix=indices.contiguous(),ty=types.contiguous();
    auto incoming=torch::full({nodes,3},-1,left.options().dtype(torch::kInt32));
    auto output=torch::empty({nodes,channels},left.options());
    const auto half=[](const torch::Tensor& t){return reinterpret_cast<const __half*>(t.data_ptr<at::Half>());};
    beam::stream1_gnn_gat(half(l),half(r),half(e),half(a),half(b),ix.data_ptr<int64_t>(),ty.data_ptr<int64_t>(),
        incoming.data_ptr<int>(),reinterpret_cast<__half*>(output.data_ptr<at::Half>()),int(nodes),int(count),int(channels),
        c10::cuda::getCurrentCUDAStream(left.get_device()).stream());
    return output;
}
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
    m.def("residual_norm_gelu(Tensor input, Tensor residual, Tensor weight, Tensor bias, float epsilon) -> Tensor");
    m.impl("residual_norm_gelu",TORCH_FN(residual_norm_gelu));
    m.def("linear(Tensor input, Tensor weight, Tensor? bias, bool cutlass) -> Tensor");
    m.impl("linear",TORCH_FN(projection));
    m.def("gat(Tensor left, Tensor right, Tensor edge, Tensor attention, Tensor bias, Tensor indices, Tensor types) -> Tensor");
    m.impl("gat",TORCH_FN(aggregate));
}
