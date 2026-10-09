// Independent implementation oracle for direct ReLU->FP8 and FP8 FF2+residual.
#include "stream1_hopper_native_fp8.cuh"
#include "stream1_hopper_native_fp8_ffn.cuh"
#include <vector>
#include <iostream>
#include <cmath>
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct B{
 T*p;size_t n;explicit B(size_t n):n(n){ck(cudaMalloc(&p,n*sizeof(T)));}~B(){cudaFree(p);}
 void put(const std::vector<T>&x){ck(cudaMemcpy(p,x.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
 std::vector<T>get(){std::vector<T>x(n);ck(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return x;}
};
int main(){try{using F=cutlass::float_e4m3_t;constexpr int K=256,H=1024;
 for(int M:{1,31,131}){
  std::vector<F>x(M*K),w1(K*H),w2(H*K);std::vector<half>b(H),r(M*K);
  for(size_t i=0;i<x.size();++i)x[i]=F((int(i*13%31)-15)/16.f);
  for(size_t i=0;i<w1.size();++i){w1[i]=F((int(i*17%37)-18)/32.f);w2[i]=F((int(i*19%41)-20)/32.f);}
  for(int i=0;i<H;++i)b[i]=__float2half((i%17-8)/16.f);
  for(size_t i=0;i<r.size();++i)r[i]=__float2half((int(i*7%23)-11)/8.f);
  B<F>dx(x.size()),dw1(w1.size()),dw2(w2.size()),hidden(M*H);B<half>db(H),res(M*K);
  dx.put(x);dw1.put(w1);dw2.put(w2);db.put(b);res.put(r);
  // Inputs/bias are already in explicit scaled units. No weight conversion.
  beam::hopper_fp8_linear<F,cutlass::epilogue::thread::ReLu>(dx.p,dw1.p,db.p,hidden.p,M,K,H,.25f,nullptr);
  ck(cudaDeviceSynchronize());auto h=hidden.get();
  for(int m=0;m<M;++m)for(int n=0;n<H;++n){double v=0;for(int k=0;k<K;++k)v+=float(x[m*K+k])*float(w1[n*K+k]);
   float expected=std::max(float(v*.25+__half2float(b[n])),0.f);float got=float(h[m*H+n]);
   if(std::abs(got-expected)>.065f*std::abs(expected)+.002f)throw std::runtime_error("FF1 ReLU FP8 mismatch");}
  beam::hopper_fp8_residual(hidden.p,dw2.p,res.p,M,H,K,.03125f,nullptr);
  ck(cudaDeviceSynchronize());auto y=res.get();
  for(int m=0;m<M;++m)for(int n=0;n<K;++n){double v=0;for(int k=0;k<H;++k)v+=float(h[m*H+k])*float(w2[n*H+k]);
   float expected=float(v*.03125+__half2float(r[m*K+n])),got=__half2float(y[m*K+n]);
   if(std::abs(got-expected)>.002f+.002f*std::abs(expected))throw std::runtime_error("FF2 FP8 residual mismatch");}
  std::cout<<"PASS FF1 ReLU FP8 + FF2 residual M="<<M<<std::endl;
 }
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
