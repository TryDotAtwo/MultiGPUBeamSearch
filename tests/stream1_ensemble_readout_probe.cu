#include "../cuda/stream1_blend_readout.hpp"
#include "../src/config.hpp"
#include <cuda_fp16.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>

void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
template<class T> T* alloc(std::size_t n) {T* p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char** argv) {
    int device=argc>1?std::stoi(argv[1]):0;check(cudaSetDevice(device));
    constexpr unsigned rows=257,hidden=32;
    for(unsigned outputs:{1U,3U,24U}) {
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
        std::cout<<"heads="<<heads<<" outputs="<<outputs<<" rows="<<rows<<" reference=PASS device="<<device<<"\n";
    }
    // Intermediate negative sums must survive until final quantization.
    check(cudaMemset(error,0,4));
    beam::stream1_ensemble_head_fp16_cuda(a,w,bias,sum,keys,rows,hidden,outputs,-1.f,true,false,error,nullptr);
    beam::stream1_ensemble_head_fp16_cuda(a,w,bias,sum,keys,rows,hidden,outputs,2.f,false,true,error,nullptr);
    check(cudaDeviceSynchronize());
    std::vector<std::uint32_t> result(rows*outputs);check(cudaMemcpy(result.data(),keys,result.size()*4,cudaMemcpyDeviceToHost));
    for(auto k:result) if(k!=3072U) throw std::runtime_error("premature clamp/quantization");
    std::cout<<"signed_accumulation=PASS\n";
    // A nonfinite intermediate must remain rejected even under a zero weight.
    hb[0]=std::nanf("");
    check(cudaMemcpy(bias,hb.data(),hb.size()*sizeof(float),cudaMemcpyHostToDevice));
    check(cudaMemset(error,0,4));
    beam::stream1_ensemble_head_fp16_cuda(a,w,bias,sum,keys,rows,hidden,outputs,0.f,true,false,error,nullptr);
    hb[0]=1.f;
    check(cudaMemcpy(bias,hb.data(),hb.size()*sizeof(float),cudaMemcpyHostToDevice));
    beam::stream1_ensemble_head_fp16_cuda(a,w,bias,sum,keys,rows,hidden,outputs,1.f,false,true,error,nullptr);
    check(cudaDeviceSynchronize());
    std::uint32_t rejected=0;check(cudaMemcpy(&rejected,error,4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(result.data(),keys,result.size()*4,cudaMemcpyDeviceToHost));
    if(!rejected||result[0]!=0xffffffffU) throw std::runtime_error("nonfinite ensemble was accepted");
    std::cout<<"nonfinite_guard=PASS outputs="<<outputs<<"\n";
    // Independent CPU oracle with distinct signed operands for every head.
    // Dyadic values keep FP16 inputs and FP32 reductions exact, including ties.
    std::mt19937 rng(20261009+outputs);
    std::uniform_int_distribution<int> values(-8,8);
    for(unsigned heads:{1U,3U,17U}) {
        std::vector<float> reference(rows*outputs,0.f);
        check(cudaMemset(error,0,4));
        for(unsigned head=0;head<heads;++head) {
            for(auto& x:ha)x=__float2half(float(values(rng))/8.f);
            for(auto& x:hw)x=__float2half(float(values(rng))/16.f);
            for(auto& x:hb)x=float(values(rng))/8.f;
            float coefficient=float(int(head%5)-2)/4.f;
            check(cudaMemcpy(a,ha.data(),ha.size()*sizeof(__half),cudaMemcpyHostToDevice));
            check(cudaMemcpy(w,hw.data(),hw.size()*sizeof(__half),cudaMemcpyHostToDevice));
            check(cudaMemcpy(bias,hb.data(),hb.size()*sizeof(float),cudaMemcpyHostToDevice));
            for(unsigned row=0;row<rows;++row) for(unsigned col=0;col<outputs;++col) {
                float dot=0.f;
                for(unsigned k=0;k<hidden;++k)
                    dot+=__half2float(ha[row*hidden+k])*__half2float(hw[k*outputs+col]);
                reference[row*outputs+col]+=coefficient*(dot+hb[col]);
            }
            beam::stream1_ensemble_head_fp16_cuda(a,w,bias,sum,keys,rows,hidden,
                outputs,coefficient,head==0,head+1==heads,error,nullptr);
        }
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(&rejected,error,4,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(result.data(),keys,result.size()*4,cudaMemcpyDeviceToHost));
        if(rejected)throw std::runtime_error("random finite ensemble numeric flag");
        for(unsigned i=0;i<result.size();++i) {
            float score=std::fmin(std::fmax(reference[i],0.f),beam::SCORE_MAX_Q);
            auto expected=static_cast<std::uint32_t>(std::nearbyint(score*float(beam::SCORE_SCALE)));
            if(result[i]!=expected)throw std::runtime_error("random independent CPU oracle mismatch");
        }
        std::cout<<"random_cpu_oracle=PASS heads="<<heads<<" outputs="<<outputs<<" device="<<device<<"\n";
    }
    check(cudaFree(a));check(cudaFree(w));check(cudaFree(bias));check(cudaFree(sum));check(cudaFree(keys));check(cudaFree(error));
    }
}

