// Isolate the H1024 seed-17 FP4 midpoint: reciprocal-multiply and division
// are different FP32 operations, even when their real-number formula matches.
#include <cutlass/numeric_types.h>
#include <cuda_runtime.h>
#include "sm120_fused_numeric_reference.hpp"
#include <cstdio>

__global__ void probe(float value, float scale, float* out) {
  const float inverse = 1.0F / scale;
  out[0] = inverse;
  out[1] = value / scale;
  out[2] = value * inverse;
  out[3] = float(cutlass::float_e2m1_t(out[1])) * scale;
  out[4] = float(cutlass::float_e2m1_t(out[2])) * scale;
}

int main() {
  static_assert(sm120_fused_reference::kHidden == 1024);
  auto f = sm120_fused_reference::make_fixture(17);
  sm120_fused_reference::compute(f);
  constexpr int index = 73302;
  float maximum = 0;
  for (int i = index / 16 * 16; i < (index / 16 + 1) * 16; ++i)
    maximum = std::max(maximum, f.ff1[i] + f.bias[i % 1024]);
  const float value = std::max(0.0F, f.ff1[index] + f.bias[index % 1024]);
  const float scale = float(cutlass::float_ue4m3_t(maximum / 6));
  const float inverse = 1.0F / scale;
  float* device = nullptr;
  if (cudaMalloc(&device, 5*sizeof(float)) != cudaSuccess) return 2;
  probe<<<1,1>>>(value, scale, device);
  auto status = cudaDeviceSynchronize();
  float host[5]{};
  if (status == cudaSuccess) status = cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) return 2;
  std::printf("index=%d raw=%.9g bias=%.9g value=%.9g max=%.9g scale=%.9g\n",
              index, f.ff1[index], f.bias[index%1024], value, maximum, scale);
  std::printf("cpu_inverse=%.9g cpu_div=%.9g cpu_mul=%.9g cpu_div_hidden=%.9g cpu_mul_hidden=%.9g\n",
      inverse, value/scale, value*inverse,
      float(cutlass::float_e2m1_t(value/scale))*scale,
      float(cutlass::float_e2m1_t(value*inverse))*scale);
  std::printf("gpu_inverse=%.9g gpu_div=%.9g gpu_mul=%.9g gpu_div_hidden=%.9g gpu_mul_hidden=%.9g\n",
      host[0], host[1], host[2], host[3], host[4]);
  const bool pass = host[0] == inverse && host[1] == value/scale &&
      host[2] == value*inverse && host[3] == 60.0F && host[4] == 90.0F;
  std::printf("midpoint_characterization=%s\n", pass?"pass":"FAIL");
  cudaFree(device);
  return pass?0:1;
}
