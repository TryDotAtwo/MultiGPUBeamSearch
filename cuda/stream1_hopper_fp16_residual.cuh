#pragma once
#include "stream1_transformer_hopper.cuh"
#include <stdexcept>

namespace beam {
#if BEAM_HAS_CUTLASS && defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
// A: row-major MxK; weights: physical column-major KxN (N contiguous
// columns each holding K elements). C and D alias the existing residual.
// Deliberately omit bias: the next bias/LayerNorm owns that boundary.
inline void hopper_fp16_residual(const half* input, const half* weight,
    half* residual, unsigned rows, unsigned k, unsigned n, cudaStream_t stream) {
    using namespace cute;
    using H = cutlass::half_t;
    using LA = cutlass::layout::RowMajor;
    using LB = cutlass::layout::ColumnMajor;
    using Tile = Shape<_128,_128,_64>;
    using Cluster = Shape<_1,_2,_1>;
    using Fusion = cutlass::epilogue::fusion::LinearCombination<H,float,H,float>;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,
        Shape<_128,_64>,float,float,H,LA,8,H,LA,8,
        cutlass::epilogue::TmaWarpSpecializedCooperative,Fusion>::CollectiveOp;
    using Main = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,H,LA,8,H,LB,8,
        float,Tile,Cluster,cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Epi::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Main,Epi>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    if (!rows) return;
    if (!input || !weight || !residual || !k || !n || k%8 || n%8)
        throw std::invalid_argument("Invalid Hopper FP16 residual operands");
    auto sa = cutlass::make_cute_packed_stride(typename Kernel::StrideA{},make_shape(int(rows),int(k),1));
    auto sb = cutlass::make_cute_packed_stride(typename Kernel::StrideB{},make_shape(int(n),int(k),1));
    auto sd = cutlass::make_cute_packed_stride(typename Kernel::StrideD{},make_shape(int(rows),int(n),1));
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
        make_shape(int(rows),int(n),int(k),1),
        {reinterpret_cast<const H*>(input),sa,reinterpret_cast<const H*>(weight),sb},
        {{},reinterpret_cast<const H*>(residual),sd,reinterpret_cast<H*>(residual),sd}};
    args.epilogue.thread.alpha = 1.f;
    args.epilogue.thread.beta = 1.f;
    if (Gemm::get_workspace_size(args))
        throw std::runtime_error("Unexpected Hopper FP16 residual workspace");
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess ||
        gemm(args,nullptr,stream) != cutlass::Status::kSuccess)
        throw std::runtime_error("Hopper FP16 residual GEMM failed");
}
#endif
} // namespace beam
