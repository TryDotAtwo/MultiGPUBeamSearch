#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "tools/cube444_blend_libtorch.hpp"
using beam::stream1_libtorch::Cube444Blend;
std::vector<torch::Tensor> readout(Cube444Blend& model,torch::Tensor cls,torch::Tensor mlp,bool only=false) {
    TORCH_CHECK(cls.is_cuda() && cls.is_contiguous() && cls.dim()==2 && cls.size(1)==256,"invalid CLS");
    TORCH_CHECK(cls.scalar_type()==model.dtype && cls.device()==model.device,"precision/device mismatch");
    TORCH_CHECK(only || (mlp.is_cuda() && mlp.scalar_type()==torch::kFloat32 && mlp.is_contiguous() && mlp.device()==cls.device() && mlp.sizes()==torch::IntArrayRef({cls.size(0),24})),"invalid MLP outputs");
    c10::cuda::CUDAGuard guard(cls.device());
    auto keys=torch::empty({cls.size(0),24},cls.options().dtype(torch::kInt32));
    auto error=torch::zeros({1},cls.options().dtype(torch::kInt32));
    model.readout(cls,mlp,reinterpret_cast<std::uint32_t*>(keys.data_ptr<int>()),cls.size(0),reinterpret_cast<std::uint32_t*>(error.data_ptr<int>()),c10::cuda::getCurrentCUDAStream(cls.get_device()).stream(),only);
    return {keys,error};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    pybind11::class_<Cube444Blend>(m,"Model")
        .def(pybind11::init([](const std::string& path,int gpu,bool fp32){return std::make_unique<Cube444Blend>(path,torch::Device(torch::kCUDA,gpu),fp32?torch::kFloat32:torch::kFloat16);}),pybind11::arg("path"),pybind11::arg("gpu"),pybind11::arg("fp32")=false)
        .def("cls",&Cube444Blend::transformer_cls)
        .def("transformer",&Cube444Blend::transformer)
        .def("mlp",&Cube444Blend::mlp)
        .def("projection_reference",[](Cube444Blend& model,torch::Tensor cls,torch::Tensor mlp){
            return .6*(torch::matmul(cls.to(torch::kFloat32),model.at("s3/output_layer_w").to(torch::kFloat32))+model.readout_bias)+.4*mlp;
        })
        .def("readout",&readout,pybind11::arg("cls"),pybind11::arg("mlp"),pybind11::arg("only")=false);
}
