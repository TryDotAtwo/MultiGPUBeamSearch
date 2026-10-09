#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/gemm/device/gemm.h>
#include <iostream>
#include <vector>
#include <cstring>
#include "stream1_hopper_fp16_residual.cuh"
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer {
    half* p{};
    explicit Buffer(size_t n){check(cudaMalloc(&p,n*2));}
    ~Buffer(){cudaFree(p);}
    void put(const std::vector<half>& v){check(cudaMemcpy(p,v.data(),v.size()*2,cudaMemcpyHostToDevice));}
};
// Exact current production SM80 FF2 specialization, with its row-major weights.
static void control(half* a,half* b,half* c,int m,cudaStream_t s){
    using H=cutlass::half_t;using R=cutlass::layout::RowMajor;
    using G=cutlass::gemm::device::Gemm<H,R,H,R,H,R,float,
        cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,
        cutlass::gemm::GemmShape<16,8,16>>;
    G g;typename G::Arguments args{{m,256,1024},{reinterpret_cast<H*>(a),1024},
        {reinterpret_cast<H*>(b),256},{reinterpret_cast<H*>(c),256},
        {reinterpret_cast<H*>(c),256},{1.f,1.f}};
    if(g(args,nullptr,s)!=cutlass::Status::kSuccess)throw std::runtime_error("control failed");
}
int main(){try{
    constexpr int m=21888,k=1024,n=256;
    std::vector<half>a(m*k),br(k*n),bc(k*n),r(m*n),v[2];
    for(size_t i=0;i<a.size();++i)a[i]=__float2half((int(i*7%17)-8)/128.f);
    for(int t=0;t<k;++t)for(int j=0;j<n;++j)
        br[t*n+j]=bc[j*k+t]=__float2half((int((j*11+t*3)%19)-9)/256.f);
    for(size_t i=0;i<r.size();++i)r[i]=__float2half((int(i%23)-11)/16.f);
    Buffer da(a.size()),db0(br.size()),db1(bc.size()),dr(r.size());
    da.put(a);db0.put(br);db1.put(bc);
    cudaStream_t s;check(cudaStreamCreate(&s));
    auto launch=[&](int mode){if(mode)beam::hopper_fp16_residual(da.p,db1.p,dr.p,m,k,n,s);else control(da.p,db0.p,dr.p,m,s);};
    for(int mode=0;mode<2;++mode){dr.put(r);launch(mode);check(cudaStreamSynchronize(s));v[mode].resize(r.size());check(cudaMemcpy(v[mode].data(),dr.p,r.size()*2,cudaMemcpyDeviceToHost));}
    if(std::memcmp(v[0].data(),v[1].data(),r.size()*2))throw std::runtime_error("paired output mismatch");
    std::cout<<"paired outputs exact rows="<<m<<std::endl;
    cudaGraph_t graph[2];cudaGraphExec_t exec[2];
    for(int mode=0;mode<2;++mode){dr.put(r);for(int i=0;i<5;++i)launch(mode);check(cudaStreamSynchronize(s));check(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));launch(mode);check(cudaStreamEndCapture(s,&graph[mode]));check(cudaGraphInstantiate(&exec[mode],graph[mode],nullptr,nullptr,0));}
    cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));
    for(int rep=0;rep<7;++rep)for(int step=0;step<2;++step){int mode=(rep+step)%2;dr.put(r);check(cudaEventRecord(start,s));for(int i=0;i<300;++i)check(cudaGraphLaunch(exec[mode],s));check(cudaEventRecord(end,s));check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,start,end));std::cout<<"rep="<<rep<<" mode="<<(mode?"sm90":"sm80")<<" us="<<ms*1000/300<<std::endl;}
    for(int mode=0;mode<2;++mode){check(cudaGraphExecDestroy(exec[mode]));check(cudaGraphDestroy(graph[mode]));}
    check(cudaEventDestroy(start));check(cudaEventDestroy(end));check(cudaStreamDestroy(s));
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
