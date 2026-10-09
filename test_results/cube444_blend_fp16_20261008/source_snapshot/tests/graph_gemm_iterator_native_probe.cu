// Prepared native metadata mutation probe. Never launch the synthetic GEMM.
// Not trained inference, iterator-value admission, or dependency verification.
#include "../tools/graph_gemm_members.hpp"
#include <iostream>

int main(int argc,char** argv) {
    try {
        BEAM_CUDA_CHECK(cudaSetDevice(argc>1?std::stoi(argv[1]):0));
        using E=cutlass::half_t;using L=cutlass::layout::RowMajor;
        using G=cutlass::gemm::device::Gemm<E,L,E,L,E,L,float,
            cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<128,64,32>,cutlass::gemm::GemmShape<64,32,32>,
            cutlass::gemm::GemmShape<16,8,16>>;
        using K=typename G::GemmKernel;using P=typename K::Params;
        P p{};
        // CUTLASS default constructors do not initialize every pointer/member.
        // This is synthetic metadata only: its null-pointer kernel is never run.
        std::memset(&p,0,sizeof(p));
        p.problem_size=cutlass::gemm::GemmCoord(6,24,256);
        p.params_C=decltype(p.params_C)(L(24));
        p.params_D=decltype(p.params_D)(L(24));
        void* args[]={&p};cudaKernelNodeParams launch{};
        launch.func=reinterpret_cast<void*>(cutlass::Kernel<K>);
        launch.gridDim=dim3(1);launch.blockDim=dim3(K::kThreadCount);
        launch.sharedMemBytes=sizeof(typename K::SharedStorage);launch.kernelParams=args;
        cudaGraph_t graph{};cudaGraphNode_t node{};
        BEAM_CUDA_CHECK(cudaGraphCreate(&graph,0));
        BEAM_CUDA_CHECK(cudaGraphAddKernelNode(&node,graph,nullptr,0,&launch));
        const char* symbol=nullptr;BEAM_CUDA_CHECK(cudaFuncGetName(&symbol,launch.func));
        beam::score_mode::GraphPointerRegistry registry;
        auto extract=[&]() {
            cudaKernelNodeParams captured{};
            BEAM_CUDA_CHECK(cudaGraphKernelNodeGetParams(node,&captured));
            return beam::score_mode::captured_gemm_members(captured,symbol,&registry,0,0);
        };
        auto baseline=extract();
        if(baseline["gemm_signature_known"]!=true)throw std::runtime_error("unknown native GEMM");
        using C=decltype(p.params_C);using I=decltype(p.params_C.stride);
        const char* names[]={"stride","increment_row","increment_group","increment_cluster",
            "advance_row","advance_group","advance_cluster","advance_tile"};
        I C::* fields[]={&C::stride,&C::increment_row,&C::increment_group,&C::increment_cluster,
            &C::advance_row,&C::advance_group,&C::advance_cluster,&C::advance_tile};
        for(unsigned output=0;output<2;++output)for(unsigned f=0;f<8;++f) {
            auto& iterator=output?p.params_D:p.params_C;
            const auto old=iterator.*fields[f];iterator.*fields[f]=old+17;
            BEAM_CUDA_CHECK(cudaGraphKernelNodeSetParams(node,&launch));
            const auto changed=extract();const char* key=output?"D":"C";
            if(changed["epilogue_iterators"][key][names[f]]!=old+17)
                throw std::runtime_error("native iterator mutation hidden");
            if(changed["scalars"]!=baseline["scalars"] || changed["problem"]!=baseline["problem"])
                throw std::runtime_error("iterator mutation changed unrelated report");
            iterator.*fields[f]=old;
        }
        BEAM_CUDA_CHECK(cudaGraphDestroy(graph));
        std::cout<<"PASS native captured C/D sixteen public-field mutations observed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
