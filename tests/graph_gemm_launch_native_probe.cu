// Captured typed Params metadata only; synthetic GEMM is never launched.
#include "../tools/graph_gemm_members.hpp"
#include <iostream>

template<class G,bool Broadcast>void check(int device) {
    BEAM_CUDA_CHECK(cudaSetDevice(device));
    using K=typename G::GemmKernel;using P=typename K::Params;
    P p{};std::memset(&p,0,sizeof(p));
    p.problem_size=cutlass::gemm::GemmCoord(6,24,256);
    p.grid_tiled_shape=cutlass::gemm::GemmCoord(2,3,1);p.gemm_k_size=256;
    if constexpr(Broadcast) {p.batch_count=1;p.batch_stride_D=144;}
    void* args[]={&p};cudaKernelNodeParams launch{};
    if constexpr(Broadcast)launch.func=reinterpret_cast<void*>(cutlass::Kernel2<K>);
    else launch.func=reinterpret_cast<void*>(cutlass::Kernel<K>);
    launch.gridDim=dim3(2,3,1);launch.blockDim=dim3(K::kThreadCount);
    launch.sharedMemBytes=sizeof(typename K::SharedStorage);launch.kernelParams=args;
    cudaGraph_t graph{};cudaGraphNode_t node{};
    BEAM_CUDA_CHECK(cudaGraphCreate(&graph,0));
    BEAM_CUDA_CHECK(cudaGraphAddKernelNode(&node,graph,nullptr,0,&launch));
    const char* symbol=nullptr;BEAM_CUDA_CHECK(cudaFuncGetName(&symbol,launch.func));
    beam::score_mode::GraphPointerRegistry registry;
    auto extract=[&]() {
        cudaKernelNodeParams captured{};BEAM_CUDA_CHECK(cudaGraphKernelNodeGetParams(node,&captured));
        return beam::score_mode::captured_gemm_members(captured,symbol,&registry,0,0);
    };
    nlohmann::json expected={{"grid_tiled_shape",{{"m",2},{"n",3},{"k",1}}},
        {"swizzle_log_tile",0},{"gemm_k_size",256},{"semaphore_null",true}};
    if constexpr(Broadcast) {expected["mode"]=0;expected["batch_count"]=1;expected["batch_stride_D"]=144;}
    if(extract().value("launch_scalars",nlohmann::json::object())!=expected)
        throw std::runtime_error("captured launch Params values missing or misdecoded");
    if constexpr(!Broadcast) {
        const auto overrides=extract().at("epilogue_pointers");
        for(const auto* key:{"alpha_array","beta_array"})
            if(overrides.at(key)!=nlohmann::json{{"role","null"},{"offset",0},{"available_bytes",0}})
                throw std::runtime_error("missing regular array override default");
        const float* arrays[2]{};registry.add("fixture_coefficient_arrays",arrays,sizeof(arrays));
        p.output_op.alpha_ptr_array=arrays;p.output_op.beta_ptr_array=arrays+1;
        BEAM_CUDA_CHECK(cudaGraphKernelNodeSetParams(node,&launch));
        const auto mutated=extract().at("epilogue_pointers");
        if(mutated.at("alpha_array")!=nlohmann::json{{"role","fixture_coefficient_arrays"},{"offset",0},{"available_bytes",16}} ||
           mutated.at("beta_array")!=nlohmann::json{{"role","fixture_coefficient_arrays"},{"offset",8},{"available_bytes",8}})
            throw std::runtime_error("regular coefficient array pointer mutation hidden");
        p.output_op.alpha_ptr_array=nullptr;p.output_op.beta_ptr_array=nullptr;
        BEAM_CUDA_CHECK(cudaGraphKernelNodeSetParams(node,&launch));
    }
    if constexpr(Broadcast) {
        nlohmann::json zero={{"stride",0},{"increment_row",0},{"increment_group",0},
            {"increment_cluster",0},{"advance_row",0},{"advance_group",0},
            {"advance_cluster",0},{"advance_tile",0}};
        if(extract().value("tensor_iterator",nlohmann::json::object())!=zero)
            throw std::runtime_error("captured Tensor Params missing");
        p.params_Tensor.stride=2;p.params_Tensor.increment_row=4;
        p.params_Tensor.increment_group=6;p.params_Tensor.increment_cluster=8;
        p.params_Tensor.advance_row=10;p.params_Tensor.advance_group=12;
        p.params_Tensor.advance_cluster=14;p.params_Tensor.advance_tile=16;
        BEAM_CUDA_CHECK(cudaGraphKernelNodeSetParams(node,&launch));
        nlohmann::json mutated={{"stride",2},{"increment_row",4},{"increment_group",6},
            {"increment_cluster",8},{"advance_row",10},{"advance_group",12},
            {"advance_cluster",14},{"advance_tile",16}};
        if(extract().value("tensor_iterator",nlohmann::json::object())!=mutated)
            throw std::runtime_error("captured Tensor Params mutation hidden");
    } else if(extract().value("tensor_iterator",nlohmann::json::object())!=nlohmann::json::object())
        throw std::runtime_error("regular Tensor Params promoted");
    p.grid_tiled_shape=cutlass::gemm::GemmCoord(17,19,7);
    p.swizzle_log_tile=5;p.gemm_k_size=1024;
    expected["grid_tiled_shape"]={{"m",17},{"n",19},{"k",7}};
    expected["swizzle_log_tile"]=5;expected["gemm_k_size"]=1024;
    if constexpr(Broadcast) {
        p.mode=cutlass::gemm::GemmUniversalMode::kBatched;p.batch_count=9;p.batch_stride_D=4096;
        expected["mode"]=2;expected["batch_count"]=9;expected["batch_stride_D"]=4096;
    }
    BEAM_CUDA_CHECK(cudaGraphKernelNodeSetParams(node,&launch));
    if(extract().value("launch_scalars",nlohmann::json::object())!=expected)
        throw std::runtime_error("captured launch Params mutation hidden");
    BEAM_CUDA_CHECK(cudaGraphDestroy(graph));
}
int main(int argc,char** argv) {
    try {
        int device=argc>1?std::stoi(argv[1]):0;
        using E=cutlass::half_t;using L=cutlass::layout::RowMajor;
        using S=cutlass::gemm::GemmShape<128,64,32>;using W=cutlass::gemm::GemmShape<64,32,32>;
        using I=cutlass::gemm::GemmShape<16,8,16>;
        using G=cutlass::gemm::device::Gemm<E,L,E,L,E,L,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,S,W,I>;
        using B=beam::score_mode::DiagnosticBroadcast<cutlass::arch::Sm80,I,3,S,W,cutlass::epilogue::thread::Identity<float>>;
        using A75=cutlass::arch::Sm75;using I75=cutlass::gemm::GemmShape<16,8,8>;
        using G75=cutlass::gemm::device::Gemm<E,L,E,L,E,L,float,cutlass::arch::OpClassTensorOp,A75,S,W,I75>;
        using B75=beam::score_mode::DiagnosticBroadcast<A75,I75,2,S,W,cutlass::epilogue::thread::Identity<float>>;
        using R80=beam::score_mode::DiagnosticBroadcast<cutlass::arch::Sm80,I,3,S,W,cutlass::epilogue::thread::ReLu<float>>;
        using R75=beam::score_mode::DiagnosticBroadcast<A75,I75,2,S,W,cutlass::epilogue::thread::ReLu<float>>;
        check<G,false>(device);check<B,true>(device);check<R80,true>(device);
        check<G75,false>(device);check<B75,true>(device);check<R75,true>(device);
        std::cout<<"PASS six SM75/SM80 captured launch Params families and mutations\n";
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
