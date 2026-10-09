// Observe typed metadata mutations only. Synthetic GEMM is never launched.
#include "../tools/graph_gemm_members.hpp"
#include <iostream>

template<class G,bool Broadcast>void check(int device) {
        BEAM_CUDA_CHECK(cudaSetDevice(device));
        using K=typename G::GemmKernel;using P=typename K::Params;
        using Base=cutlass::transform::threadblock::PredicatedTileAccessIteratorParams;
        P p{};std::memset(&p,0,sizeof(p));
        p.problem_size=cutlass::gemm::GemmCoord(6,24,256);
        Base base{};base.stride_=256;base.inc_strided_=2048;
        base.inc_next_=3072;base.inc_advance_=4096;
        p.params_A=decltype(p.params_A)(base);p.params_B=decltype(p.params_B)(base);
        void* args[]={&p};cudaKernelNodeParams launch{};
        if constexpr(Broadcast)launch.func=reinterpret_cast<void*>(cutlass::Kernel2<K>);
        else launch.func=reinterpret_cast<void*>(cutlass::Kernel<K>);
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
        const auto baseline=extract();
        if(!baseline.contains("mainloop_iterators"))
            throw std::runtime_error("captured A/B iterator values missing");
        const char* names[]={"stride","increment_strided","increment_next","advance"};
        Base::LongIndex Base::* fields[]={&Base::stride_,&Base::inc_strided_,&Base::inc_next_,&Base::inc_advance_};
        for(unsigned operand=0;operand<2;++operand)for(unsigned f=0;f<4;++f) {
            Base changed=base;changed.*fields[f]+=17;
            if(operand)p.params_B=decltype(p.params_B)(changed);
            else p.params_A=decltype(p.params_A)(changed);
            BEAM_CUDA_CHECK(cudaGraphKernelNodeSetParams(node,&launch));
            const auto observed=extract();const char* key=operand?"B":"A";
            for(unsigned j=0;j<4;++j)
                if(observed["mainloop_iterators"][key][names[j]]!=changed.*fields[j])
                    throw std::runtime_error("native A/B iterator mutation hidden or misdecoded");
            if(observed["scalars"]!=baseline["scalars"] || observed["problem"]!=baseline["problem"] ||
                observed["pointers"]!=baseline["pointers"] || observed["epilogue_iterators"]!=baseline["epilogue_iterators"])
                throw std::runtime_error("A/B mutation changed unrelated report");
            p.params_A=decltype(p.params_A)(base);p.params_B=decltype(p.params_B)(base);
        }
        BEAM_CUDA_CHECK(cudaGraphDestroy(graph));
}
using E=cutlass::half_t;using L=cutlass::layout::RowMajor;
using S=cutlass::gemm::GemmShape<128,64,32>;using W=cutlass::gemm::GemmShape<64,32,32>;
using I=cutlass::gemm::GemmShape<16,8,16>;using I75=cutlass::gemm::GemmShape<16,8,8>;
template<class Arch,class Inst,int Stages,class Activation>
using ProbeBroadcast=cutlass::gemm::device::GemmUniversalWithBroadcast<E,L,E,L,E,L,float,
    cutlass::arch::OpClassTensorOp,Arch,S,W,Inst,
    cutlass::epilogue::thread::LinearCombinationBiasElementwise<E,float,float,E,E,8,
        Activation,cutlass::plus<float>,false,E>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,Stages>;
int main(int argc,char** argv) {
    try {
        const int device=argc>1?std::stoi(argv[1]):0;
        using R=cutlass::gemm::device::Gemm<E,L,E,L,E,L,float,
            cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,S,W,I>;
        using R75=cutlass::gemm::device::Gemm<E,L,E,L,E,L,float,
            cutlass::arch::OpClassTensorOp,cutlass::arch::Sm75,S,W,I75>;
        check<R,false>(device);check<R75,false>(device);
        check<ProbeBroadcast<cutlass::arch::Sm80,I,3,cutlass::epilogue::thread::Identity<float>>,true>(device);
        check<ProbeBroadcast<cutlass::arch::Sm80,I,3,cutlass::epilogue::thread::ReLu<float>>,true>(device);
        check<ProbeBroadcast<cutlass::arch::Sm75,I75,2,cutlass::epilogue::thread::Identity<float>>,true>(device);
        check<ProbeBroadcast<cutlass::arch::Sm75,I75,2,cutlass::epilogue::thread::ReLu<float>>,true>(device);
        std::cout<<"PASS native captured A/B 48 base-field mutations across six families observed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
