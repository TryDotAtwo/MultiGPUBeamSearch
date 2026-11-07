// Independently compiled classic SM75 baseline Params storage; no CUDA calls.
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <iostream>
#include <cstring>
using E=cutlass::half_t;
using R=cutlass::layout::RowMajor;
using Shape=cutlass::gemm::GemmShape<128,64,32>;
using Warp=cutlass::gemm::GemmShape<64,32,32>;
using Inst=cutlass::gemm::GemmShape<16,8,8>;
using Regular=cutlass::gemm::device::Gemm<E,R,E,R,E,R,float,
    cutlass::arch::OpClassTensorOp,cutlass::arch::Sm75,Shape,Warp,Inst>;
template<class Activation>using Epilogue=cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    E,float,float,E,E,8,Activation,cutlass::plus<float>,false,E>;
template<class Activation>using Broadcast=cutlass::gemm::device::GemmUniversalWithBroadcast<
    E,R,E,R,E,R,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm75,Shape,Warp,Inst,
    Epilogue<Activation>,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,2>;
template<class Gemm>void emit(const char* family,bool comma) {
    using P=typename Gemm::GemmKernel::Params;
    if(comma)std::cout<<',';
    std::cout<<"{\"family\":\""<<family<<"\",\"tile\":[128,64,32],\"warp\":[64,32,32],"
        <<"\"instruction\":[16,8,8],\"stages\":2,\"swizzle\":1,\"parameter_size\":"<<sizeof(P)
        <<",\"parameter_alignment\":"<<alignof(P)<<'}';
}
template<class Activation,class S,class W>using Broadcast80=cutlass::gemm::device::GemmUniversalWithBroadcast<
    E,R,E,R,E,R,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,S,W,
    cutlass::gemm::GemmShape<16,8,16>,Epilogue<Activation>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,3>;
template<class Gemm>void emit80(const char* family,int tile_n,int warp_n,int stages,bool comma) {
    using P=typename Gemm::GemmKernel::Params;
    if(comma)std::cout<<',';
    std::cout<<"{\"family\":\""<<family<<"\",\"tile\":[128,"<<tile_n
        <<",32],\"warp\":[64,"<<warp_n<<",32],\"instruction\":[16,8,16],\"stages\":"
        <<stages<<",\"swizzle\":1,\"parameter_size\":"<<sizeof(P)
        <<",\"parameter_alignment\":"<<alignof(P)<<'}';
}
int main(int argc,char** argv) {
    if(argc==2 && std::strcmp(argv[1],"--sm80")==0) {
        using Inst80=cutlass::gemm::GemmShape<16,8,16>;
        using Regular80=cutlass::gemm::device::Gemm<E,R,E,R,E,R,float,
            cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,Shape,Warp,Inst80>;
        using Identity=cutlass::epilogue::thread::Identity<float>;
        using Relu=cutlass::epilogue::thread::ReLu<float>;
        using SplitShape=cutlass::gemm::GemmShape<128,128,32>;
        using SplitWarp=cutlass::gemm::GemmShape<64,64,32>;
        std::cout<<"{\"schema_version\":1,\"scope\":\"compiled_SM80_classic_gemm_parameters_not_admission\",\"entries\":[";
        emit80<Regular80>("gemm_residual",64,32,3,false);
        emit80<Broadcast80<Identity,Shape,Warp>>("gemm_bias",64,32,3,true);
        emit80<Broadcast80<Relu,Shape,Warp>>("gemm_relu",64,32,3,true);
        emit80<Broadcast80<Identity,SplitShape,SplitWarp>>("gemm_bias",128,64,3,true);
        emit<Regular>("gemm_pipelined",true);
        std::cout<<"]}\n";
        return 0;
    }
    if(argc!=1) {std::cerr<<"usage: gemm_parameter_probe [--sm80]\n";return 2;}
    std::cout<<"{\"schema_version\":1,\"scope\":\"compiled_SM75_baseline_gemm_parameters_not_admission\",\"entries\":[";
    emit<Regular>("gemm_pipelined",false);
    emit<Broadcast<cutlass::epilogue::thread::Identity<float>>>("gemm_bias",true);
    emit<Broadcast<cutlass::epilogue::thread::ReLu<float>>>("gemm_relu",true);
    std::cout<<"]}\n";
}
