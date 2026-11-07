#include "stream1_transformer_sm120_nvfp4_cutlass_ff1_shared_handoff.hpp"
#include <array>
#include <iostream>

using namespace stream1_sm120_nvfp4_cutlass_dsm;

int main() {
  using Mma = Ff1Mainloop::TiledMma;
  Mma mma;
  const auto identity = cute::make_identity_tensor(cute::make_shape(cute::Int<128>{}, cute::Int<128>{}));
  std::array<int, 128*128> coverage{};
  for (int thread=0; thread<128; ++thread) {
    auto coordinates = mma.get_slice(thread).partition_C(identity);
    for (int i=0; i<cute::size(coordinates); ++i) {
      const int row = cute::get<0>(coordinates(i));
      const int col = cute::get<1>(coordinates(i));
      if (row<0 || row>=128 || col<0 || col>=128) return 2;
      ++coverage[row*128+col];
      if (thread<5 && i<32)
        std::cout << "coordinate thread="<<thread<<" register="<<i<<" row="<<row<<" col="<<col<<'\n';
    }
    if (thread==0) {
      std::cout << "fragment_size=" << cute::size(coordinates) << " layout=";
      cute::print(coordinates.layout());
      std::cout << '\n';
    }
  }
  for (int count:coverage) if (count!=1) return 3;
  std::cout << "fragment_coverage=pass elements=16384 threads=128\n";
}
