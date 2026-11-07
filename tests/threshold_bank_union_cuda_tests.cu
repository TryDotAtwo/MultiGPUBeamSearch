#include "threshold.hpp"
#include "config.hpp"
#include <algorithm>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>
using namespace beam;
void check(cudaError_t x) {if(x!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(x));}
int main() {
 for(unsigned capacity:{1U,37U,1024U,65536U}) {
  std::vector<CandidateMeta> rows(4ULL*capacity);
  std::vector<std::uint32_t> counts={capacity,capacity,capacity/2,capacity};
  std::vector<std::uint64_t> expected(SCORE_BIN_COUNT,0),actual(SCORE_BIN_COUNT);
  for(unsigned shard=0;shard<2;++shard) {
   std::map<std::uint64_t,std::uint32_t> oracle;
   for(unsigned bank=0;bank<2;++bank) {
    auto* begin=rows.data()+(shard*2ULL+bank)*capacity;
    for(unsigned i=0;i<counts[shard*2+bank];++i) {
     const auto hash=i+bank*(capacity/3)+1ULL;
     const auto key=(i*13+bank*7+shard*3)%31;
     begin[i]={Hash128{hash,hash>>4},i,key,0};
     auto it=oracle.find(hash);if(it==oracle.end())oracle[hash]=key;else it->second=std::min(it->second,key);
    }
    std::sort(begin,begin+counts[shard*2+bank],[](auto a,auto b){return a.hash.hi<b.hash.hi || (a.hash.hi==b.hash.hi && a.hash.lo<b.hash.lo);});
   }
   for(auto item:oracle)++expected[item.second];
  }
  CandidateMeta* drows;std::uint32_t* dcounts;std::uint64_t* hist;
  check(cudaMalloc(&drows,rows.size()*sizeof(CandidateMeta)));check(cudaMalloc(&dcounts,16));check(cudaMalloc(&hist,actual.size()*8));
  check(cudaMemcpy(drows,rows.data(),rows.size()*sizeof(CandidateMeta),cudaMemcpyHostToDevice));
  check(cudaMemcpy(dcounts,counts.data(),16,cudaMemcpyHostToDevice));
  threshold_build_exact_bank_union_histogram_cuda(drows,dcounts,hist,2,capacity,nullptr);
  check(cudaMemcpy(actual.data(),hist,actual.size()*8,cudaMemcpyDeviceToHost));
  if(actual!=expected)throw std::runtime_error("exact bank union histogram differs from independent hash/min-score oracle");
  check(cudaFree(drows));check(cudaFree(dcounts));check(cudaFree(hist));
 }
 std::cout<<"exact_bank_union_histogram=PASS duplicates,min-score,empty-small-bank,partial-warp,65536-capacity\n";
}
