#include "stream1_gnn_projection.hpp"
#include <stdexcept>
#include <cstdint>
#include <cassert>
#if BEAM_HAS_CUTLASS
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
namespace beam {
namespace {
// Broadcast C is the bias vector (row stride zero). Preserve the old
// GEMM -> FP16 store -> FP16 bias-add rounding without that store/reload.
struct RoundedBias : cutlass::epilogue::thread::LinearCombination<cutlass::half_t,8,float,float> {
    using Base=cutlass::epilogue::thread::LinearCombination<cutlass::half_t,8,float,float>;
    bool bias_enabled;
    CUTLASS_HOST_DEVICE explicit RoundedBias(Params const& p):Base(p),bias_enabled(p.beta!=0.f) {}
    CUTLASS_HOST_DEVICE void set_k_partition(int,int) {} // split_k=1 only
    CUTLASS_HOST_DEVICE FragmentOutput operator()(FragmentAccumulator const& ab) const {
        return cutlass::NumericArrayConverter<cutlass::half_t,float,8>()(ab);
    }
    CUTLASS_HOST_DEVICE FragmentOutput operator()(FragmentAccumulator const& ab,FragmentSource const& bias) const {
        auto result=operator()(ab);
        if(bias_enabled) {
            CUTLASS_PRAGMA_UNROLL
            for(int i=0;i<8;++i) result[i]=cutlass::half_t(float(result[i])+float(bias[i]));
        }
        return result;
    }
};
// Pancake graphs have at most three incoming edges, including the self edge.
__global__ void gat_incoming(const std::int64_t* edges,int* incoming,int count,int nodes) {
    const auto i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count) return;
    const auto target=edges[count+i];
    assert(target>=0 && target<nodes && edges[i]>=0 && edges[i]<nodes);
    for(int slot=0;slot<3;++slot)
        if(atomicCAS(incoming+target*3+slot,-1,int(i))==-1) return;
    assert(false && "GAT incoming degree exceeds three");
}
__device__ float warp_sum(float value) {
    for(int offset=16;offset;offset/=2) value+=__shfl_down_sync(0xffffffff,value,offset);
    return __shfl_sync(0xffffffff,value,0);
}
__global__ void gat_reduce(const __half* left,const __half* right,const __half* edge,
    const __half* attention,const __half* bias,const std::int64_t* indices,
    const std::int64_t* types,const int* incoming,__half* output,int nodes,int channels) {
    const int node=blockIdx.x,head=threadIdx.x/32,lane=threadIdx.x%32;
    int ids[3]={incoming[std::int64_t(node)*3],incoming[std::int64_t(node)*3+1],incoming[std::int64_t(node)*3+2]};
    // Restore declared edge order; atomic insertion order is not semantic order.
    for(int i=0;i<3;++i) for(int j=i+1;j<3;++j)
        if(ids[j]>=0 && (ids[i]<0 || ids[j]<ids[i])) {int tmp=ids[i];ids[i]=ids[j];ids[j]=tmp;}
    const int degree=(ids[0]>=0)+(ids[1]>=0)+(ids[2]>=0);
    assert(degree>0);
    float logits[3];
    for(int e=0;e<degree;++e) {
        const auto source=indices[ids[e]],type=types[ids[e]];
        assert(type>=0 && type<3);
        float sum=0;
        for(int c=lane;c<channels;c+=32) {
            const auto column=head*channels+c;
            auto value=__hadd(__hadd(left[source*(4LL*channels)+column],right[std::int64_t(node)*4*channels+column]),edge[type*(4LL*channels)+column]);
            if(__half2float(value)<0) value=__float2half_rn(__half2float(value)*.2f);
            sum+=__half2float(__hmul(value,attention[column]));
        }
        // The source ATen sum returns FP16, then converts its logits to FP32.
        logits[e]=__half2float(__float2half_rn(warp_sum(sum)));
    }
    float maximum=logits[0];for(int e=1;e<degree;++e) maximum=fmaxf(maximum,logits[e]);
    float weights[3],denominator=0;
    for(int e=0;e<degree;++e) {weights[e]=expf(logits[e]-maximum);denominator+=weights[e];}
    for(int e=0;e<degree;++e) weights[e]=__half2float(__float2half_rn(weights[e]/denominator));
    extern __shared__ __half heads[];
    for(int c=lane;c<channels;c+=32) {
        __half total=__float2half_rn(0.f);
        for(int e=0;e<degree;++e) {
            const auto source=indices[ids[e]];
            auto message=__hmul(left[source*(4LL*channels)+head*channels+c],__float2half_rn(weights[e]));
            total=__hadd(total,message);
        }
        heads[head*channels+c]=total;
    }
    __syncthreads();
    for(int c=threadIdx.x;c<channels;c+=blockDim.x) {
        float mean=0;for(int h=0;h<4;++h) mean+=__half2float(heads[h*channels+c]);
        output[std::int64_t(node)*channels+c]=__hadd(__float2half_rn(mean*.25f),bias[c]);
    }
}
} // anonymous namespace
void stream1_gnn_gat(const __half* left,const __half* right,const __half* edge,
    const __half* attention,const __half* bias,const std::int64_t* indices,
    const std::int64_t* types,int* incoming,__half* output,
    int nodes,int edges,int channels,cudaStream_t stream) {
    gat_incoming<<<unsigned((std::uint64_t(edges)+255)/256),256,0,stream>>>(indices,incoming,edges,nodes);
    gat_reduce<<<nodes,128,4*channels*sizeof(__half),stream>>>(left,right,edge,attention,bias,indices,types,incoming,output,nodes,channels);
    if(cudaGetLastError()!=cudaSuccess) throw std::runtime_error("GNN fused GAT launch failed");
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
        RoundedBias>;
    Gemm gemm;
    typename Gemm::Arguments args({rows,outputs,inputs},
        {reinterpret_cast<const cutlass::half_t*>(input),inputs},
        {reinterpret_cast<const cutlass::half_t*>(weight),inputs},
        {reinterpret_cast<const cutlass::half_t*>(bias?bias:output),bias?0:outputs},
        {reinterpret_cast<cutlass::half_t*>(output),outputs},{1.f,bias?1.f:0.f});
    if(gemm.can_implement(args)!=cutlass::Status::kSuccess||
       gemm(args,nullptr,stream)!=cutlass::Status::kSuccess)
        throw std::runtime_error("GNN CUTLASS projection launch failed");
}
}
#else
namespace beam {
void stream1_gnn_gat(const __half*,const __half*,const __half*,const __half*,const __half*,
    const std::int64_t*,const std::int64_t*,int*,__half*,int,int,int,cudaStream_t) {
    throw std::runtime_error("GNN fused GAT backend was not built");
}
void stream1_gnn_projection(const __half*,const __half*,const __half*,__half*,int,int,int,cudaStream_t) {
    throw std::runtime_error("GNN CUTLASS backend was not built");
}
}
#endif
