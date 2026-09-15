#include "stream1_transformer_hopper.cuh"
#include "stream1_transformer_ln_qkv_fusion.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace beam {
void stream1_transformer_layernorm_copy_launch(const half*, half*, const half*, const half*,
    std::uint32_t, std::uint32_t, std::uint32_t, cudaStream_t);
}
static void check(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
    half* p=nullptr;
    explicit Buffer(std::size_t n) { check(cudaMalloc(&p,n*sizeof(half))); }
    ~Buffer() { cudaFree(p); }
    void put(const std::vector<half>& x) { check(cudaMemcpy(p,x.data(),x.size()*sizeof(half),cudaMemcpyHostToDevice)); }
    std::vector<half> get(std::size_t n) { std::vector<half> x(n); check(cudaMemcpy(x.data(),p,n*sizeof(half),cudaMemcpyDeviceToHost)); return x; }
};

// Catches missing tail predication, wrong packed-B orientation, wrong gamma/beta,
// dropped bias, and normalization of a different row. Reference calls the
// existing production normalization and existing Hopper GEMM, not fusion helpers.
static void run(int rows, bool timing) {
    constexpr int K=256,N=768;
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    std::vector<half> x(rows*K),g(K),b(K),w(N*K),bias(N);
    for(int r=0;r<rows;++r) for(int k=0;k<K;++k)
        x[r*K+k]=__float2half(r%7==0 ? 0.25f : float((r*71+k*19)%193-96)/31.0f);
    for(int k=0;k<K;++k) { g[k]=__float2half(0.5f+(k%17)/16.0f); b[k]=__float2half((k%13-6)/32.0f); }
    for(int n=0;n<N;++n) { bias[n]=__float2half((n%11-5)/64.0f);
        for(int k=0;k<K;++k) w[n*K+k]=__float2half(((n*13+k*23)%67-33)/256.0f); }
    Buffer dx(x.size()),dg(K),db(K),dw(w.size()),dbias(N),norm(x.size()),ref(rows*N),out(rows*N);
    dx.put(x);dg.put(g);db.put(b);dw.put(w);dbias.put(bias);
    auto baseline=[&] {
        beam::stream1_transformer_layernorm_copy_launch(dx.p,norm.p,dg.p,db.p,rows,K,0,stream);
        beam::stream1_transformer_hopper_fp16_bias_activation<cutlass::epilogue::thread::Identity>(norm.p,dw.p,dbias.p,ref.p,rows,K,N,stream);
    };
    auto fused=[&] {beam::stream1_transformer_ln_qkv_fused(dx.p,dg.p,db.p,dw.p,dbias.p,out.p,rows,stream);};
    baseline(); fused(); check(cudaDeviceSynchronize());
    auto a=ref.get(rows*N),v=out.get(rows*N);
    float maxerr=0;std::size_t unequal=0;
    for(std::size_t i=0;i<a.size();++i) {
        float err=std::abs(__half2float(a[i])-__half2float(v[i]));
        if(!std::isfinite(err)) throw std::runtime_error("non-finite fused output");
        maxerr=std::max(maxerr,err);unequal+=err!=0;
    }
    if(dx.get(x.size())!=x) throw std::runtime_error("fusion mutated persistent input");
    std::cout<<"ln_qkv rows="<<rows<<" max_error="<<maxerr<<" unequal="<<unequal<<" count="<<a.size()<<std::endl;
    if(maxerr>0.008f) throw std::runtime_error("fused LN-QKV differs from production reference");
    if(timing) {
        auto time=[&](auto fn) {
            for(int i=0;i<10;++i)fn();check(cudaDeviceSynchronize());
            cudaGraph_t graph;cudaGraphExec_t exec;
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));fn();
            check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            cudaEvent_t s,e;check(cudaEventCreate(&s));check(cudaEventCreate(&e));
            check(cudaEventRecord(s,stream));for(int i=0;i<200;++i)check(cudaGraphLaunch(exec,stream));check(cudaEventRecord(e,stream));check(cudaEventSynchronize(e));
            float ms;check(cudaEventElapsedTime(&ms,s,e));cudaEventDestroy(s);cudaEventDestroy(e);cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
            return ms*1000/200;
        };
        for(int pass=0;pass<3;++pass){auto a_us=time(baseline),b_us=time(fused);std::cout<<"timing rows="<<rows<<" pass="<<pass<<" baseline_us="<<a_us<<" fused_us="<<b_us<<std::endl;}
    }
    check(cudaStreamDestroy(stream));
}
int main(int argc,char**) {
    try {for(int rows:{1,63,64,65,131})run(rows,false);if(argc==1)run(384*57,true);}
    catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}
}
