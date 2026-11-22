#pragma once
#include "stream1_hopper_native_fp8.cuh"

namespace beam {
// FP8 FF2 GEMM + existing FP16 residual, with a single final FP16 rounding.
// Original FF2 bias remains deferred to the following bias/LN boundary.
inline void hopper_fp8_residual(const cutlass::float_e4m3_t* input,
    const cutlass::float_e4m3_t* weight,half* residual,
    unsigned rows,unsigned k,unsigned n,float scale_product,cudaStream_t stream) {
    using namespace cute;
    using F8=cutlass::float_e4m3_t;using H=cutlass::half_t;
    using Tile=Shape<_128,_128,_128>;using Cluster=Shape<_1,_2,_1>;
    using LA=cutlass::layout::RowMajor;using LB=cutlass::layout::ColumnMajor;
    using Fusion=cutlass::epilogue::fusion::LinearCombination<H,float,H,float>;
    using Epi=typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,
        Shape<_128,_64>,float,float,H,LA,8,H,LA,8,
        cutlass::epilogue::TmaWarpSpecializedCooperative,Fusion>::CollectiveOp;
    using Main=typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,F8,LA,16,F8,LB,16,
        float,Tile,Cluster,cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
    using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Main,Epi>;
    using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    if(!rows)return;
    if(!k||!n||k%16||n%8||!std::isfinite(scale_product)||scale_product<=0)
        throw std::invalid_argument("Invalid Hopper FP8 residual shape/scale");
    auto sa=cutlass::make_cute_packed_stride(typename Kernel::StrideA{},make_shape(int(rows),int(k),1));
    auto sb=cutlass::make_cute_packed_stride(typename Kernel::StrideB{},make_shape(int(n),int(k),1));
    auto sd=cutlass::make_cute_packed_stride(typename Kernel::StrideD{},make_shape(int(rows),int(n),1));
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
        make_shape(int(rows),int(n),int(k),1),{input,sa,weight,sb},
        {{},reinterpret_cast<const H*>(residual),sd,reinterpret_cast<H*>(residual),sd}};
    args.epilogue.thread.alpha=scale_product;args.epilogue.thread.beta=1;
    if(Gemm::get_workspace_size(args))throw std::runtime_error("Unexpected FP8 residual workspace");
    Gemm gemm;
    if(gemm(args,nullptr,stream)!=cutlass::Status::kSuccess)throw std::runtime_error("Native Hopper FP8 residual GEMM failed");
}
}
