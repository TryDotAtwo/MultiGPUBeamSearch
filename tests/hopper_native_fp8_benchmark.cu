// Component-only paired benchmark. Both paths include the same warp LN
// algorithm; this is NOT the incumbent full-model LayerNorm implementation.
#include "stream1_hopper_native_fp8.cuh"
#include "stream1_transformer_hopper.cuh"
#include <algorithm>
#include <iostream>
#include <vector>

void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Buf {
    T* p; size_t n;
    explicit Buf(size_t count):n(count){ck(cudaMalloc(&p,n*sizeof(T)));}
    ~Buf(){cudaFree(p);}
    void put(const std::vector<T>& x){ck(cudaMemcpy(p,x.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T>x(n);ck(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return x;}
};
void run(int rows){
    using F8=cutlass::float_e4m3_t;
    constexpr int K=256,N=768;
    constexpr float inv=32.f,ws=1.f/128;
    std::vector<half>x(size_t(rows)*K),g(K),b(K),bias(N),w(K*N);
    std::vector<F8>qw(K*N);
    for(size_t i=0;i<x.size();++i)x[i]=__float2half((int((i*17+i/K*13)%193)-96)/32.f);
    for(int i=0;i<K;++i){g[i]=__float2half(.5f+(i%7)/8.f);b[i]=__float2half((i%9-4)/32.f);}
    for(int i=0;i<N;++i)bias[i]=__float2half((i%13-6)/16.f);
    // Exact same representable weights in both paths, encoded before capture.
    for(int i=0;i<K*N;++i){qw[i]=F8(((i*23)%67-33)/16.f);w[i]=__float2half(float(qw[i])*ws);}
    Buf<half>dx(x.size()),dg(K),db(K),dbias(N),dw(K*N),ln(x.size()),out(size_t(rows)*N);
    Buf<F8>dq(x.size()),dqw(K*N);
    dx.put(x);dg.put(g);db.put(b);dbias.put(bias);dw.put(w);dqw.put(qw);
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto launch=[&](int mode){
        if(mode==0){
            beam::hopper_fp8_detail::ln256<<<(rows+3)/4,128,0,stream>>>(dx.p,dg.p,db.p,ln.p,rows,1.f);
            beam::stream1_transformer_hopper_fp16_bias_activation<cutlass::epilogue::thread::Identity>(ln.p,dw.p,dbias.p,out.p,rows,K,N,stream);
        }else{
            beam::hopper_ln256_fp8(dx.p,dg.p,db.p,dq.p,rows,inv,stream);
            beam::hopper_fp8_linear(dq.p,dqw.p,dbias.p,out.p,rows,K,N,ws/inv,stream);
        }
    };
    launch(0);ck(cudaStreamSynchronize(stream));auto reference=out.get();
    launch(1);ck(cudaStreamSynchronize(stream));auto actual=out.get();
    double sum=0,maxerr=0,checksum=0;
    for(size_t i=0;i<actual.size();++i){double y=__half2float(actual[i]);if(!std::isfinite(y))throw std::runtime_error("nonfinite output");double d=y-__half2float(reference[i]);sum+=d*d;maxerr=std::max(maxerr,std::abs(d));checksum+=y;}
    std::cout<<"component rows="<<rows<<" maxabs="<<maxerr<<" rmse="<<std::sqrt(sum/actual.size())<<" checksum="<<checksum<<std::endl;
    cudaGraph_t graph[2];cudaGraphExec_t exec[2];
    for(int m=0;m<2;++m){for(int i=0;i<10;++i)launch(m);ck(cudaStreamSynchronize(stream));ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch(m);ck(cudaStreamEndCapture(stream,&graph[m]));ck(cudaGraphInstantiate(&exec[m],graph[m],nullptr,nullptr,0));}
    cudaEvent_t start,end;ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
    for(int rep=0;rep<7;++rep)for(int step=0;step<2;++step){int m=rep%2?1-step:step;ck(cudaEventRecord(start,stream));for(int i=0;i<500;++i)ck(cudaGraphLaunch(exec[m],stream));ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,start,end));std::cout<<"timing rows="<<rows<<" rep="<<rep<<" mode="<<m<<" us="<<ms*2<<std::endl;}
    for(int i=0;i<2;++i){ck(cudaGraphExecDestroy(exec[i]));ck(cudaGraphDestroy(graph[i]));}ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));ck(cudaStreamDestroy(stream));
}
int main(){try{for(int rows:{21888,43776,87552,175104})run(rows);}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
