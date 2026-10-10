#include "stream1_gnn_projection.hpp"
#include <stdexcept>
#include <cstdint>
#if BEAM_HAS_CUTLASS
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
namespace beam {
namespace {
__global__ void add_bias(__half* output,const __half* bias,std::uint64_t count,int width) {
    const auto i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count) output[i]=__float2half_rn(__half2float(output[i])+__half2float(bias[i%width]));
}
}
void stream1_gnn_projection(const __half* input,const __half* weight,const __half* bias,
                           __half* output,int rows,int inputs,int outputs,cudaStream_t stream) {
    if(rows<1||inputs<1||outputs<1||inputs%8||outputs%8)
        throw std::invalid_argument("GNN CUTLASS projection requires aligned positive dimensions");
    using Gemm=cutlass::gemm::device::Gemm<
        cutlass::half_t,cutlass::layout::RowMajor,
        cutlass::half_t,cutlass::layout::ColumnMajor,
        cutlass::half_t,cutlass::layout::RowMajor,float,
        cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128,64,32>,cutlass::gemm::GemmShape<64,32,32>,
        cutlass::gemm::GemmShape<16,8,16>,
        cutlass::epilogue::thread::LinearCombination<cutlass::half_t,8,float,float>>;
    Gemm gemm;
    typename Gemm::Arguments args({rows,outputs,inputs},
        {reinterpret_cast<const cutlass::half_t*>(input),inputs},
        {reinterpret_cast<const cutlass::half_t*>(weight),inputs},
        {reinterpret_cast<const cutlass::half_t*>(output),outputs},
        {reinterpret_cast<cutlass::half_t*>(output),outputs},{1.f,0.f});
    if(gemm.can_implement(args)!=cutlass::Status::kSuccess||
       gemm(args,nullptr,stream)!=cutlass::Status::kSuccess)
        throw std::runtime_error("GNN CUTLASS projection launch failed");
    if(bias) {
        const auto count=std::uint64_t(rows)*outputs;
        add_bias<<<unsigned((count+255)/256),256,0,stream>>>(output,bias,count,outputs);
        if(cudaGetLastError()!=cudaSuccess) throw std::runtime_error("GNN projection bias launch failed");
    }
}
}
#else
namespace beam {
void stream1_gnn_projection(const __half*,const __half*,const __half*,__half*,int,int,int,cudaStream_t) {
    throw std::runtime_error("GNN CUTLASS backend was not built");
}
}
#endif
