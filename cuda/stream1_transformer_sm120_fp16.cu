#include "stream1_transformer_sm120_fp16.hpp"
#include "cuda_check.hpp"

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <stdexcept>

namespace {
using Element = cutlass::half_t;
using Accumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int Alignment = 128 / cutlass::sizeof_bits<Element>::value;
using Tile = cute::Shape<cute::_128,cute::_128,cute::_64>;
using Cluster = cute::Shape<cute::_1,cute::_1,cute::_1>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    Accumulator, float, void, LayoutD, Alignment,
    Element, LayoutD, Alignment,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    Element, LayoutA, Alignment,
    Element, LayoutB, Alignment,
    Accumulator, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int,int>,Mainloop,Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
using StrideA=typename Kernel::StrideA;using StrideB=typename Kernel::StrideB;using StrideC=typename Kernel::StrideC;using StrideD=typename Kernel::StrideD;

Gemm::Arguments args(const Element* a,const Element* b,Element* d,std::uint32_t m,std::uint32_t k,std::uint32_t n) {
    auto shape=cute::make_shape(static_cast<int>(m),static_cast<int>(n),static_cast<int>(k),1);
    auto sa=cutlass::make_cute_packed_stride(StrideA{},cute::make_shape(static_cast<int>(m),static_cast<int>(k),1));
    auto sb=cutlass::make_cute_packed_stride(StrideB{},cute::make_shape(static_cast<int>(n),static_cast<int>(k),1));
    auto sc=cutlass::make_cute_packed_stride(StrideC{},cute::make_shape(static_cast<int>(m),static_cast<int>(n),1));
    auto sd=cutlass::make_cute_packed_stride(StrideD{},cute::make_shape(static_cast<int>(m),static_cast<int>(n),1));
    Gemm::Arguments out{cutlass::gemm::GemmUniversalMode::kGemm,shape,{a,sa,b,sb},{{},nullptr,sc,d,sd}};
    out.epilogue.thread.alpha=1.0f;out.epilogue.thread.beta=0.0f;return out;
}
void validate(std::uint32_t m,std::uint32_t k,std::uint32_t n) {
    if (!m||!k||!n||k%8U||n%8U) throw std::invalid_argument("SM120 FP16 GEMM requires non-zero aligned dimensions");
}
}

bool stream1_transformer_sm120_fp16_supported(){int dev=0;cudaDeviceProp p{};BEAM_CUDA_CHECK(cudaGetDevice(&dev));BEAM_CUDA_CHECK(cudaGetDeviceProperties(&p,dev));return p.major==12&&p.minor==0;}
std::size_t stream1_transformer_sm120_fp16_workspace_bytes(std::uint32_t m,std::uint32_t k,std::uint32_t n){validate(m,k,n);return Gemm::get_workspace_size(args(nullptr,nullptr,nullptr,m,k,n));}
void stream1_transformer_sm120_fp16_linear_cuda(const half* a,const half* b,half* d,std::uint32_t m,std::uint32_t k,std::uint32_t n,void* workspace,std::size_t bytes,cudaStream_t stream){
    validate(m,k,n);if(!a||!b||!d)throw std::invalid_argument("SM120 FP16 GEMM received null pointer");auto ar=args(reinterpret_cast<const Element*>(a),reinterpret_cast<const Element*>(b),reinterpret_cast<Element*>(d),m,k,n);const auto need=Gemm::get_workspace_size(ar);if(bytes<need||(need&&!workspace))throw std::invalid_argument("SM120 FP16 workspace too small");Gemm g;if(g.can_implement(ar)!=cutlass::Status::kSuccess)throw std::runtime_error("SM120 FP16 GEMM cannot implement shape");if(g(ar,workspace,stream)!=cutlass::Status::kSuccess)throw std::runtime_error("SM120 FP16 GEMM launch failed");
}
