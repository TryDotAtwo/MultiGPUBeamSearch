#include "../cuda/stream3.hpp"
#include "../cuda/stream4.hpp"
#include "../cuda/score_sort_bits.cuh"
#include "../cuda/static_memory.hpp"
#include "config.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace beam;
static_assert(detail::score_key_bits(0U) == 1U);
static_assert(detail::score_key_bits(1U) == 1U);
static_assert(detail::score_key_bits(2U) == 2U);
static_assert(detail::score_key_bits(3U) == 2U);
static_assert(detail::score_key_bits(4U) == 3U);
static_assert(detail::score_key_bits(UINT32_MAX) == 32U);
static_assert(detail::SCORE_HISTOGRAM_KEY_BITS == 19U);
#if BEAM_EXPERIMENTAL_SCORE_BITS
static_assert(detail::SCORE_HISTOGRAM_SORT_END_BIT == 19);
#else
static_assert(detail::SCORE_HISTOGRAM_SORT_END_BIT == 32);
#endif

namespace {
void cu(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
template<class T> struct Buffer {
 T* p = nullptr; explicit Buffer(std::size_t n) { cu(cudaMalloc(&p, n * sizeof(T))); }
 ~Buffer() { if (p) cudaFree(p); }
 Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
 void upload(const T* v, std::size_t n) { if (n) cu(cudaMemcpy(p, v, n*sizeof(T), cudaMemcpyHostToDevice)); }
 void download(T* v, std::size_t n) { if (n) cu(cudaMemcpy(v, p, n*sizeof(T), cudaMemcpyDeviceToHost)); }
};
std::uint32_t sample(unsigned i) {
 switch (i % 8U) {
  case 0: return 0; case 1: return 1; case 2: return 1U << 18;
  case 3: return SCORE_MAX_KEY; case 4: return SCORE_MAX_KEY-1;
  case 5: return SCORE_BIN_COUNT; case 6: return UINT32_MAX; default: return 1;
 }
}
}
int main() {
 try {
  cu(cudaSetDevice(0));
  constexpr std::uint32_t cap=513, micro=2, per=micro*MOVE_COUNT, slots=(cap+per-1)/per;
  constexpr std::size_t scratch_bytes=8ULL*1024*1024;
  cudaStream_t stream=nullptr; cu(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  Buffer<std::uint32_t> scores(cap), parents(slots), key_a(cap), key_b(cap), unique(1);
  Buffer<std::uint64_t> val_a(cap), val_b(cap), hist3(SCORE_BIN_COUNT);
  Buffer<std::byte> scratch(scratch_bytes);
  // Exercise either independently owned test scratch or the exact production
  // arena extents. The latter must need no test-side initialization.
  struct Arena {
   StaticDeviceMemory memory;
   ~Arena() { free_static_device_memory(memory); }
  } arena;
  void* scratch3 = scratch.p; std::size_t scratch3_bytes = scratch_bytes;
  void* scratch4 = scratch.p; std::size_t scratch4_bytes = scratch_bytes;
  if (std::getenv("BEAM_TEST_USE_ARENA_SCRATCH")) {
   RuntimeConfig config;
   config.b_micro=micro; config.stream3_batch_candidates=slots*per;
   config.stream4_batch_candidates=64; config.stream4_active_sort_slots=2;
   config.ring_count=2; config.shard_count=4; config.shard_capacity_candidates=640;
   config.global_spill_capacity=640; config.stream4_batch_alignment=1;
   config.user_global_beam_width=256; config.solved_result_capacity=16;
   const auto plan=make_static_memory_plan(config);
   allocate_static_device_memory(plan,arena.memory);
   scratch3=arena.memory.streams.stream3_cub_temp;
   scratch3_bytes=arena.memory.streams.stream3_cub_temp_bytes;
   scratch4=arena.memory.streams.stream4_cub_temp;
   scratch4_bytes=arena.memory.streams.stream4_cub_temp_bytes;
   std::cout << "using production arena CUB scratch stream3=" << scratch3_bytes
             << " stream4=" << scratch4_bytes << '\n';
  }
  if (std::getenv("BEAM_TEST_ZERO_SCRATCH")) {
   cu(cudaMemset(scratch.p, 0, scratch_bytes));
   cu(cudaDeviceSynchronize());
  }
  auto s3=[&] {
   cu(cudaMemsetAsync(hist3.p,0,SCORE_BIN_COUNT*sizeof(std::uint64_t),stream));
   stream3_accumulate_score_hist_cuda(scores.p,parents.p,key_a.p,key_b.p,val_a.p,val_b.p,
       unique.p,hist3.p,scratch3,scratch3_bytes,micro,cap,stream);
  };
  cudaGraph_t graph3=nullptr; cudaGraphExec_t exec3=nullptr;
  cu(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)); s3();
  cu(cudaStreamEndCapture(stream,&graph3)); cu(cudaGraphInstantiate(&exec3,graph3,nullptr,nullptr,0));
  for (unsigned pattern=0; pattern<5; ++pattern) {
   std::vector<std::uint32_t> host_scores(cap), host_parents(slots);
   std::vector<std::uint64_t> expected(SCORE_BIN_COUNT), actual(SCORE_BIN_COUNT);
   for (unsigned j=0; j<slots; ++j) host_parents[j]=pattern==0?0:pattern==1?micro:(j+pattern)%3;
   for (unsigned i=0; i<cap; ++i) {
    host_scores[i]=sample(i+pattern);
    if ((i%per)/MOVE_COUNT<host_parents[i/per] && host_scores[i]<SCORE_BIN_COUNT) ++expected[host_scores[i]];
   }
   scores.upload(host_scores.data(),cap); parents.upload(host_parents.data(),slots);
   for (unsigned mode=0; mode<3; ++mode) {
    if (mode==0) s3(); else cu(cudaGraphLaunch(exec3,stream));
    cu(cudaStreamSynchronize(stream)); hist3.download(actual.data(),actual.size());
    require(actual==expected,"stream3 score histogram differs from CPU oracle");
   }
  }
  cu(cudaGraphExecDestroy(exec3)); cu(cudaGraphDestroy(graph3));

  Buffer<CandidateMeta> survivors(cap), sort_val(cap), reduce_val(cap);
  Buffer<Hash128> sort_key(cap), reduce_key(cap);
  Buffer<std::uint32_t> keep(cap), blocks((cap+255)/256), offsets((cap+255)/256);
  Buffer<std::uint32_t> clean(1), dirty(1), processing(1), hist_a(SCORE_BIN_COUNT), hist_b(SCORE_BIN_COUNT), active(1);
  if (std::getenv("BEAM_TEST_PRINT_ALLOCATIONS")) {
   auto range = [](const char* name, const void* p, std::size_t bytes) {
    auto first = reinterpret_cast<std::uintptr_t>(p);
    std::cerr << "ALLOCATION " << name << " 0x" << std::hex << first << " 0x" << first + bytes << std::dec << '\n';
   };
   range("scratch",scratch.p,scratch_bytes);
   range("survivors",survivors.p,cap*sizeof(CandidateMeta));
   range("sort_val",sort_val.p,cap*sizeof(CandidateMeta));
   range("reduce_val",reduce_val.p,cap*sizeof(CandidateMeta));
   range("sort_key",sort_key.p,cap*sizeof(Hash128));
   range("reduce_key",reduce_key.p,cap*sizeof(Hash128));
  }
  auto s4=[&] {
   stream4_shard_job_cuda(survivors.p,clean.p,dirty.p,processing.p,SCORE_MAX_KEY,cap,
    sort_key.p,reduce_key.p,sort_val.p,reduce_val.p,key_a.p,key_b.p,val_a.p,val_b.p,
    keep.p,blocks.p,offsets.p,unique.p,hist_a.p,hist_b.p,active.p,scratch4,scratch4_bytes,stream);
  };
  cudaGraph_t graph4=nullptr; cudaGraphExec_t exec4=nullptr;
  cu(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)); s4();
  cu(cudaStreamEndCapture(stream,&graph4)); cu(cudaGraphInstantiate(&exec4,graph4,nullptr,nullptr,0));
  for (unsigned n : {0U,1U,2U,255U,256U,257U,513U}) {
   std::vector<CandidateMeta> input(cap);
   std::vector<std::uint32_t> expected(SCORE_BIN_COUNT),actual(SCORE_BIN_COUNT);
   unsigned valid=0;
   for (unsigned i=0; i<n; ++i) {
    auto score=sample(i); input[i]=CandidateMeta{Hash128{i+1U,0xabc000ULL+i},i,score,pack_route(0,0,i%MOVE_COUNT)};
    if (score<=SCORE_MAX_KEY) { ++expected[score]; ++valid; }
   }
   for (unsigned mode=0; mode<3; ++mode) {
    const unsigned zero=0, one=1;
    survivors.upload(input.data(),cap); clean.upload(&zero,1); dirty.upload(&n,1);
    processing.upload(&one,1); active.upload(&zero,1);
    cu(cudaMemset(hist_a.p,0,SCORE_BIN_COUNT*sizeof(std::uint32_t)));
    cu(cudaMemset(hist_b.p,0,SCORE_BIN_COUNT*sizeof(std::uint32_t)));
    cu(cudaDeviceSynchronize());
    if (mode==0) s4(); else cu(cudaGraphLaunch(exec4,stream));
    cu(cudaStreamSynchronize(stream)); unsigned observed=0,selected=0;
    clean.download(&observed,1); active.download(&selected,1);
    (selected ? hist_b : hist_a).download(actual.data(),actual.size());
    require(observed==valid,"stream4 retained count mismatch");
    require(actual==expected,"stream4 score histogram differs from CPU oracle");
   }
  }
  cu(cudaGraphExecDestroy(exec4)); cu(cudaGraphDestroy(graph4)); cu(cudaStreamDestroy(stream));
  std::cout << "score_histograms stream3_stream4_cpu_oracle_and_graph=PASS\n";
  return 0;
 } catch (const std::exception& e) { std::cerr << "score_histograms FAIL " << e.what() << '\n'; return 1; }
}
