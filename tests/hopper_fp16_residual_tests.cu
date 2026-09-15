#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "stream1_hopper_fp16_residual.cuh"

static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffer {
    half* p{};
    explicit Buffer(size_t n) { check(cudaMalloc(&p, n * sizeof(half))); }
    ~Buffer() { cudaFree(p); }
    void put(const std::vector<half>& v) {
        check(cudaMemcpy(p, v.data(), v.size()*sizeof(half), cudaMemcpyHostToDevice));
    }
};

// Independent exact dyadic oracle catches transposed weights, missing residual,
// and dropped rows. All intermediate sums are exactly representable in FP32.
static void run(unsigned m) {
    constexpr unsigned k=1024, n=256;
    std::vector<half> a(m*k), b(k*n), residual(m*n), actual(m*n);
    for (unsigned i=0;i<a.size();++i) a[i]=__float2half((int(i*7%17)-8)/32.f);
    for (unsigned col=0;col<n;++col) for(unsigned t=0;t<k;++t)
        b[col*k+t]=__float2half((int((col*11+t*3)%19)-9)/64.f);
    for(unsigned i=0;i<residual.size();++i) residual[i]=__float2half((int(i%23)-11)/16.f);
    Buffer da(a.size()), db(b.size()), dr(residual.size());
    da.put(a); db.put(b); dr.put(residual);
    beam::hopper_fp16_residual(da.p,db.p,dr.p,m,k,n,nullptr);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(actual.data(),dr.p,actual.size()*sizeof(half),cudaMemcpyDeviceToHost));
    for(unsigned row=0;row<m;++row) for(unsigned col=0;col<n;++col) {
        float sum=0;
        for(unsigned t=0;t<k;++t) sum += __half2float(a[row*k+t])*__half2float(b[col*k+t]);
        const float expected=__half2float(__float2half(sum+__half2float(residual[row*n+col])));
        if(__half2float(actual[row*n+col])!=expected)
            throw std::runtime_error("FP16 residual differs from independent oracle");
    }
    std::cout << "PASS rows=" << m << '\n';
}
int main() {
    try { for(unsigned m: {1U,31U,128U,131U}) run(m); }
    catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
