#include "../cuda/stream1_blend_readout.hpp"
#include "../src/config.hpp"
#include <cuda_fp16.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <stdexcept>

void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
template<class T> T* alloc(std::size_t n) {T* p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char** argv) {
    int device=argc>1?std::stoi(argv[1]):0;check(cudaSetDevice(device));
    constexpr unsigned rows=257,hidden=32,outputs=24;
    auto a=alloc<__half>(rows*hidden),w=alloc<__half>(hidden*outputs);
    auto bias=alloc<float>(outputs),sum=alloc<float>(rows*outputs);
    auto keys=alloc<std::uint32_t>(rows*outputs),error=alloc<std::uint32_t>(1);
    std::vector<__half> ha(rows*hidden,__float2half(0.5f)),hw(hidden*outputs,__float2half(0.125f));
    std::vector<float> hb(outputs,1.f);
    check(cudaMemcpy(a,ha.data(),ha.size()*sizeof(__half),cudaMemcpyHostToDevice));
    check(cudaMemcpy(w,hw.data(),hw.size()*sizeof(__half),cudaMemcpyHostToDevice));
    check(cudaMemcpy(bias,hb.data(),hb.size()*sizeof(float),cudaMemcpyHostToDevice));
    for(unsigned heads:{1U,2U,3U,8U,17U}) {
        check(cudaMemset(error,0,4));
        for(unsigned i=0;i<heads;++i) beam::stream1_ensemble_head_fp16_cuda(
            a,w,bias,sum,keys,rows,hidden,outputs,1.f/heads,i==0,i+1==heads,error,nullptr);
        check(cudaDeviceSynchronize());
        std::uint32_t flag;check(cudaMemcpy(&flag,error,4,cudaMemcpyDeviceToHost));
        std::vector<std::uint32_t> result(rows*outputs);
        check(cudaMemcpy(result.data(),keys,result.size()*4,cudaMemcpyDeviceToHost));
        if(flag) throw std::runtime_error("numeric flag on finite ensemble");
        for(auto k:result) if(k!=3072U) throw std::runtime_error("ensemble reference mismatch");
        std::cout<<"heads="<<heads<<" rows="<<rows<<" reference=PASS device="<<device<<"\n";
    }
    // Intermediate negative sums must survive until final quantization.
    check(cudaMemset(error,0,4));
    beam::stream1_ensemble_head_fp16_cuda(a,w,bias,sum,keys,rows,hidden,outputs,-1.f,true,false,error,nullptr);
    beam::stream1_ensemble_head_fp16_cuda(a,w,bias,sum,keys,rows,hidden,outputs,2.f,false,true,error,nullptr);
    check(cudaDeviceSynchronize());
    std::vector<std::uint32_t> result(rows*outputs);check(cudaMemcpy(result.data(),keys,result.size()*4,cudaMemcpyDeviceToHost));
    for(auto k:result) if(k!=3072U) throw std::runtime_error("premature clamp/quantization");
    std::cout<<"signed_accumulation=PASS\n";
    check(cudaFree(a));check(cudaFree(w));check(cudaFree(bias));check(cudaFree(sum));check(cudaFree(keys));check(cudaFree(error));
}
