// Missing normalization, wrong FP8 scale, transposed weights or dropped tails
// must fail the independent CPU oracle below. No teacher-quality claim.
#include "stream1_hopper_native_fp8.cuh"
#include <vector>
#include <iostream>
#include <cmath>
#include <stdexcept>

void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Buf {
    T* p; size_t n;
    explicit Buf(size_t count):n(count){ck(cudaMalloc(&p,n*sizeof(T)));}
    ~Buf(){cudaFree(p);}
    void put(const std::vector<T>& x){ck(cudaMemcpy(p,x.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T>x(n);ck(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return x;}
};
int main(){try{
    using F8=cutlass::float_e4m3_t;
    constexpr int K=256,N=768; constexpr float inv=32.f,ws=.00390625f;
    for(int M:{1,31,128,131}) {
        std::vector<half>x(M*K),g(K),b(K),bias(N);std::vector<F8>w(N*K);
        for(int i=0;i<M*K;++i)x[i]=__float2half(((i*17+i/K*13)%193-96)/32.f);
        for(int i=0;i<K;++i){g[i]=__float2half(.5f+(i%7)/8.f);b[i]=__float2half((i%9-4)/32.f);}
        for(int i=0;i<N;++i)bias[i]=__float2half((i%13-6)/16.f);
        // Physical KxN column-major == contiguous K for each output column.
        for(int i=0;i<N*K;++i)w[i]=F8(((i*23)%67-33)/16.f);
        Buf<half>dx(x.size()),dg(K),db(K),dbias(N),dy(M*N);Buf<F8>dq(M*K),dw(N*K);
        dx.put(x);dg.put(g);db.put(b);dbias.put(bias);dw.put(w);
        beam::hopper_ln256_fp8(dx.p,dg.p,db.p,dq.p,M,inv,nullptr);
        ck(cudaDeviceSynchronize());auto q=dq.get();
        for(int r=0;r<M;++r){double mean=0,var=0;for(int k=0;k<K;++k)mean+=__half2float(x[r*K+k]);mean/=K;
            for(int k=0;k<K;++k){double d=__half2float(x[r*K+k])-mean;var+=d*d;}var/=K;
            for(int k=0;k<K;++k){float ref=((__half2float(x[r*K+k])-mean)/std::sqrt(var+1e-5))*__half2float(g[k])+__half2float(b[k]);
                float got=float(q[r*K+k])/inv;
                if(std::abs(got-ref)>.065f*std::abs(ref)+.002f)throw std::runtime_error("LN FP8 independent oracle mismatch");}}
        beam::hopper_fp8_linear(dq.p,dw.p,dbias.p,dy.p,M,K,N,ws/inv,nullptr);
        ck(cudaDeviceSynchronize());auto y=dy.get();
        for(int r=0;r<M;++r)for(int n=0;n<N;++n){double acc=0;for(int k=0;k<K;++k)acc+=float(q[r*K+k])*float(w[n*K+k]);
            float ref=acc*ws/inv+__half2float(bias[n]);float got=__half2float(y[r*N+n]);
            if(std::abs(got-ref)>.002f+.002f*std::abs(ref))throw std::runtime_error("FP8 GEMM independent oracle mismatch");}
        std::cout<<"PASS independent LN/FP8 GEMM M="<<M<<std::endl;
        // Bias must update the FP16 residual, but LN must see the unrounded
        // FP32 sum. Reuse a nontrivial small bias vector distinct from beta.
        dx.put(x);
        beam::hopper_ln256_fp8(dx.p,dg.p,db.p,dq.p,M,inv,nullptr,db.p,dx.p);
        ck(cudaDeviceSynchronize());q=dq.get();auto residual=dx.get();
        for(int r=0;r<M;++r){double mean=0,var=0;
            for(int k=0;k<K;++k)mean+=double(__half2float(x[r*K+k]))+__half2float(b[k]);mean/=K;
            for(int k=0;k<K;++k){double v=double(__half2float(x[r*K+k]))+__half2float(b[k]);var+=(v-mean)*(v-mean);
                if(__half2float(residual[r*K+k])!=__half2float(__float2half(float(v))))throw std::runtime_error("bias residual rounding mismatch");}
            var/=K;
            for(int k=0;k<K;++k){double v=double(__half2float(x[r*K+k]))+__half2float(b[k]);
                double ref=(v-mean)/std::sqrt(var+1e-5)*__half2float(g[k])+__half2float(b[k]);
                if(std::abs(float(q[r*K+k])/inv-ref)>.065*std::abs(ref)+.002)throw std::runtime_error("bias LN FP8 oracle mismatch");}}
        std::cout<<"PASS independent bias/residual/LN M="<<M<<std::endl;
    }
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
