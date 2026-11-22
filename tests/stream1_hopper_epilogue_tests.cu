#include "stream1_transformer_hopper.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

static void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
    half* p{};
    explicit Buffer(std::size_t n){check(cudaMalloc(&p,n*sizeof(half)));}
    ~Buffer(){cudaFree(p);}
    void put(const std::vector<half>& x){check(cudaMemcpy(p,x.data(),x.size()*2,cudaMemcpyHostToDevice));}
    std::vector<half> get(std::size_t n){std::vector<half>x(n);check(cudaMemcpy(x.data(),p,n*2,cudaMemcpyDeviceToHost));return x;}
};

// A missing epilogue, transposed bias, dropped tails, or changed activation must
// fail the full-output comparison against the incumbent's unchanged auto tile.
static void run(int rows,bool timing) {
    constexpr int K=256,N=1024;
    std::vector<half>x(rows*K),w(N*K),b(N);
    for(int i=0;i<rows*K;++i)x[i]=__float2half(((i*19)%127-63)/31.f);
    for(int i=0;i<N*K;++i)w[i]=__float2half(((i*23)%71-35)/128.f);
    for(int i=0;i<N;++i)b[i]=__float2half((i%17-8)/16.f);
    Buffer dx(x.size()),dw(w.size()),db(b.size()),out(rows*N),ref(rows*N);
    dx.put(x);dw.put(w);db.put(b);
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto launch=[&](int mode,half* dst) {
        if(mode==0)beam::stream1_transformer_hopper_fp16_bias_activation<cutlass::epilogue::thread::ReLu>(dx.p,dw.p,db.p,dst,rows,K,N,stream);
        else if(mode==1)beam::stream1_transformer_hopper_fp16_bias_activation<cutlass::epilogue::thread::ReLu,cute::Shape<cute::_128,cute::_64>>(dx.p,dw.p,db.p,dst,rows,K,N,stream);
        else beam::stream1_transformer_hopper_fp16_bias_activation<cutlass::epilogue::thread::ReLu,cute::Shape<cute::_128,cute::_128>>(dx.p,dw.p,db.p,dst,rows,K,N,stream);
    };
    launch(0,ref.p);check(cudaStreamSynchronize(stream));auto expected=ref.get(rows*N);
    for(int mode:{1,2}) {
        launch(mode,out.p);check(cudaStreamSynchronize(stream));auto actual=out.get(rows*N);
        if(std::memcmp(expected.data(),actual.data(),actual.size()*2))throw std::runtime_error("epilogue output differs from incumbent");
        std::cout<<"exact rows="<<rows<<" mode="<<mode<<std::endl;
    }
    if(timing) {
        cudaGraph_t graphs[3];cudaGraphExec_t execs[3];
        for(int mode=0;mode<3;++mode){for(int i=0;i<5;++i)launch(mode,out.p);check(cudaStreamSynchronize(stream));check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch(mode,out.p);check(cudaStreamEndCapture(stream,&graphs[mode]));check(cudaGraphInstantiate(&execs[mode],graphs[mode],nullptr,nullptr,0));}
        cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));
        for(int rep=0;rep<7;++rep)for(int step=0;step<3;++step){int mode=rep%2?2-step:step;check(cudaEventRecord(start,stream));for(int i=0;i<300;++i)check(cudaGraphLaunch(execs[mode],stream));check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,start,end));std::cout<<"timing rows="<<rows<<" rep="<<rep<<" mode="<<mode<<" us="<<ms*1000/300<<std::endl;}
        for(int i=0;i<3;++i){check(cudaGraphExecDestroy(execs[i]));check(cudaGraphDestroy(graphs[i]));}check(cudaEventDestroy(start));check(cudaEventDestroy(end));
    }
    check(cudaStreamDestroy(stream));
}
int main(int argc,char**){try{for(int n:{1,63,128,131})run(n,false);if(argc>1)run(21888,true);}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
