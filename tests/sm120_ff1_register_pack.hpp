#pragma once

#include "stream1_transformer_sm120_nvfp4_cutlass_persistent_dsm.hpp"
#include <array>

namespace sm120_ff1_register_pack {
using namespace stream1_sm120_nvfp4_cutlass_dsm;

// Fail closed on a different accumulator layout. One quad owns 16 columns:
// each lane has adjacent pairs at columns 2*lane and 8+2*lane. The pairs are
// 16 register positions apart in the selected SM120 M128/N128 fragment.
inline bool layout_contract() {
  Ff1Mainloop::TiledMma mma;
  const auto identity = make_identity_tensor(make_shape(Int<128>{}, Int<128>{}));
  std::array<int, 128*128> values{};
  std::array<int, 128*8> scales{};
  for (int t=0;t<128;++t) {
    auto coords = mma.get_slice(t).partition_C(identity);
    auto owner = mma.get_slice(t & ~3).partition_C(identity);
    if (size(coords)!=128) return false;
    for (int n=0;n<4;++n) for (int r=0;r<8;++r) {
      const int i=n*32+r*2;
      const int row=get<0>(owner(i)), first=get<1>(owner(i));
      if (row<0 || row>=128 || first<0 || first>=128 || first%16) return false;
      if ((t&3)==0) ++scales[row*8+first/16];
      for (int j=0;j<4;++j) {
        const int index=i+(j/2)*16+j%2;
        const int column=first+(t&3)*2+(j/2)*8+j%2;
        if (get<0>(coords(index))!=row || get<1>(coords(index))!=column) return false;
        ++values[row*128+column];
        if ((j&1)==0) {
          const int lo=Ff2PhysicalStageLayoutA{}(row,column);
          const int hi=Ff2PhysicalStageLayoutA{}(row,column+1);
          if ((lo&1) || hi!=lo+1) return false;  // unique whole-byte writer
        }
      }
    }
  }
  for (int count:values) if (count!=1) return false;
  for (int count:scales) if (count!=1) return false;
  return true;
}

template <int Hidden, class Fragment>
CUTLASS_DEVICE void store(
    Fragment const& accumulators, DsmASlot& destination,
    ElementBias const* bias, int logical_slice, float* raw_ff1) {
  Ff1Mainloop::TiledMma mma;
  const auto identity = make_identity_tensor(make_shape(Int<128>{}, Int<128>{}));
  auto coordinates = mma.get_slice(static_cast<int>(threadIdx.x%128)).partition_C(identity);
  static_assert(decltype(size(accumulators))::value==128);
  CUTLASS_PRAGMA_UNROLL
  for (int n=0;n<4;++n) {
    CUTLASS_PRAGMA_UNROLL
    for (int r=0;r<8;++r) {
      constexpr unsigned mask=0xffffffffU;
      const int i=n*32+r*2;
      const int row=get<0>(coordinates(i));
      const int column=get<1>(coordinates(i));
      float v[4];
      CUTLASS_PRAGMA_UNROLL
      for (int j=0;j<4;++j) {
        const int index=i+(j/2)*16+j%2;
        const int col=column+(j/2)*8+j%2;
        const float raw=static_cast<float>(accumulators(index));
        if (raw_ff1) raw_ff1[row*Hidden+logical_slice*128+col]=raw;
        const float biased=raw+static_cast<float>(bias[logical_slice*128+col]);
        v[j]=biased>0.0F ? biased : 0.0F;
      }
      float maximum=fmaxf(fmaxf(v[0],v[1]),fmaxf(v[2],v[3]));
      maximum=fmaxf(maximum,__shfl_xor_sync(mask,maximum,1,4));
      maximum=fmaxf(maximum,__shfl_xor_sync(mask,maximum,2,4));
      const ElementScale scale(maximum>0.0F ? maximum/6.0F : 1.0F);
      const float reciprocal=1.0F/static_cast<float>(scale);
      const unsigned a=ElementHidden(v[0]*reciprocal).raw()&15;
      const unsigned b=ElementHidden(v[1]*reciprocal).raw()&15;
      const unsigned c=ElementHidden(v[2]*reciprocal).raw()&15;
      const unsigned d=ElementHidden(v[3]*reciprocal).raw()&15;
      Ff1HiddenRingWriter::store_group_from_lane_pairs(
          destination,row,column/16,static_cast<std::uint8_t>(a|(b<<4)),
          static_cast<std::uint8_t>(c|(d<<4)),scale.raw());
    }
  }
}
}  // namespace sm120_ff1_register_pack
