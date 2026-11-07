// Regression: byte-swizzled FP4 storage must be readable by CUTLASS's actual
// smem tensor view, not by a host decoder that repeats the writer's indexing.
#include "stream1_transformer_sm120_nvfp4_cutlass_persistent_dsm.hpp"
#include <cuda_runtime.h>
#include <cstdio>

using namespace stream1_sm120_nvfp4_cutlass_dsm;

__host__ __device__ unsigned pattern(unsigned row, unsigned col) {
  return ((row * 7U + col * 3U) ^ ((row >> 2U) + col / 16U)) & 15U;
}

template<bool LanePairs, bool UseHelper>
__global__ void roundtrip(unsigned* errors) {
  __shared__ DsmASlot slot;
  for (unsigned i = threadIdx.x; i < sizeof(slot); i += blockDim.x)
    reinterpret_cast<unsigned char*>(&slot)[i] = 0xa5;
  __syncthreads();
  if constexpr (LanePairs) {
    unsigned lane = threadIdx.x & 3U;
    for (unsigned work = threadIdx.x / 4U; work < 128U*8U;
         work += blockDim.x / 4U) {
      unsigned row = work / 8U, group = work % 8U;
      unsigned col = group * 16U + lane * 2U;
      auto lo = pattern(row,col) | pattern(row,col+1) << 4U;
      auto hi = pattern(row,col+8) | pattern(row,col+9) << 4U;
      auto sf = ElementScale(float(1U + (row + group) % 7U) / 4.0F).raw();
      Ff1HiddenRingWriter::store_group_from_lane_pairs(slot,row,group,lo,hi,sf);
    }
  } else {
    for (unsigned work = threadIdx.x; work < 128U*8U; work += blockDim.x) {
      unsigned row = work / 8U, group = work % 8U;
      std::uint8_t raw[16];
      for (unsigned i=0;i<16;++i) raw[i] = pattern(row,group*16U+i);
      auto sf = ElementScale(float(1U + (row + group) % 7U) / 4.0F).raw();
      Ff1HiddenRingWriter::store_group(slot,row,group,raw,sf);
    }
  }
  __syncthreads();
  // These are the same pointer/layout construction primitives as the FF2
  // consumer. No raw-address calculation is shared with the writer.
  auto a = [&]() {
    if constexpr (UseHelper) {
      return make_dsm_a_stage_view(slot.values);
    } else {
      // Independent canonical CUTLASS construction also tests that our helper
      // uses a subbyte iterator, rather than an ordinary byte-stepped FP4*.
      return make_tensor(
          make_smem_ptr<typename Ff2Mainloop::SmemAllocTypeA>(
              static_cast<void*>(slot.values)), Ff2StageLayoutA{});
    }
  }();
  auto sfa = make_dsm_sfa_stage_view(slot.scales);
  for (unsigned i=threadIdx.x;i<128U*128U;i+=blockDim.x) {
    unsigned row=i/128U, col=i%128U;
    typename Ff2Mainloop::SmemAllocTypeA value = a(row,col);
    ElementScale sf = sfa(row,col);
    if (static_cast<unsigned>(value) != pattern(row,col)) atomicAdd(errors,1U);
    auto expected_sf = ElementScale(float(1U+(row+col/16U)%7U)/4.0F).raw();
    if (sf.raw() != expected_sf) atomicAdd(errors+1,1U);
  }
}

int main() {
  unsigned* errors=nullptr;
  if (cudaMalloc(&errors,2*sizeof(unsigned)) != cudaSuccess) return 2;
  bool pass=true;
  for (int helper=0;helper<2;++helper) {
    for (int mode=0;mode<2;++mode) {
      auto status=cudaMemset(errors,0,2*sizeof(unsigned));
      if (status != cudaSuccess) return 2;
      if (helper) {
        if (mode) roundtrip<true,true><<<1,128>>>(errors);
        else roundtrip<false,true><<<1,128>>>(errors);
      } else {
        if (mode) roundtrip<true,false><<<1,128>>>(errors);
        else roundtrip<false,false><<<1,128>>>(errors);
      }
      status=cudaGetLastError();
      if (status==cudaSuccess) status=cudaDeviceSynchronize();
      unsigned counts[2]{~0U,~0U};
      if (status==cudaSuccess)
        status=cudaMemcpy(counts,errors,sizeof(counts),cudaMemcpyDeviceToHost);
      bool ok=status==cudaSuccess && counts[0]==0 && counts[1]==0;
      std::printf("reader=%s writer=%s values_errors=%u scales_errors=%u cuda=%s verdict=%s\n",
                  helper?"helper":"canonical",mode?"lane_pairs":"whole_group",
                  counts[0],counts[1],cudaGetErrorString(status),ok?"pass":"FAIL");
      if (status!=cudaSuccess) return 1;
      pass=pass&&ok;
    }
  }
  cudaFree(errors);
  return pass?0:1;
}
