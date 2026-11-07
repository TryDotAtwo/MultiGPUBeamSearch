#include "stream1_transformer_sm120_nvfp4_cutlass_persistent_dsm.hpp"

#include <iostream>
#include <type_traits>
#include <vector>

using namespace stream1_sm120_nvfp4_cutlass_dsm;

template <class Policy>
struct EpiloguePolicyTraits;

template <int StagesC, int StagesD, int FragmentSize, bool Reuse, bool Delay>
struct EpiloguePolicyTraits<
    cutlass::epilogue::Sm90TmaWarpSpecialized<
        StagesC, StagesD, FragmentSize, Reuse, Delay>> {
    static constexpr int kStagesC = StagesC;
    static constexpr int kStagesD = StagesD;
    static constexpr bool kReuse = Reuse;
};

using Ff1Policy = EpiloguePolicyTraits<typename Ff1Epilogue::DispatchPolicy>;
using Ff1SmemLayoutD = decltype(cute::tile_to_shape(
    typename Ff1Epilogue::SmemLayoutAtomD{},
    cute::make_shape(
        cute::size<0>(typename Ff1Epilogue::EpilogueTile{}),
        cute::size<1>(typename Ff1Epilogue::EpilogueTile{}),
        cute::Int<Ff1Policy::kReuse ? Ff1Policy::kStagesC : Ff1Policy::kStagesD>{}),
    cute::conditional_t<
        cutlass::epilogue::collective::detail::is_m_major<
            typename Ff1Epilogue::StrideD>(),
        cute::Step<cute::_2, cute::_1, cute::_3>,
        cute::Step<cute::_1, cute::_2, cute::_3>>{}));

using Ff1Stage0Layout = decltype(Ff1SmemLayoutD{}(_, _, cute::Int<0>{}));

int main() {
  int linear_quarter_mismatches = 0;
  int first_m = -1;
  int first_n = -1;
  int first_q = -1;
  int first_ff1 = -1;
  int first_ff2 = -1;
  for (int q = 0; q < 4; ++q) {
    for (int m = 0; m < 128; ++m) {
      for (int n = 0; n < 32; ++n) {
        const int ff1 = Ff1SmemLayoutD{}(m, n, 0) + q * 4096;
        const int ff2 = Ff2StageLayoutA{}(m, q * 32 + n);
        if (ff1 != ff2) {
          ++linear_quarter_mismatches;
          if (first_m < 0) {
            first_m = m;
            first_n = n;
            first_q = q;
            first_ff1 = ff1;
            first_ff2 = ff2;
          }
        }
      }
    }
  }
  std::vector<int> source_to_dest(16384, -1);
  for (int q = 0; q < 4; ++q) {
    for (int m = 0; m < 128; ++m) {
      for (int n = 0; n < 32; ++n) {
        const int source = Ff1SmemLayoutD{}(m, n, 0) + q * 4096;
        const int dest = Ff2StageLayoutA{}(m, q * 32 + n);
        source_to_dest[source] = dest;
      }
    }
  }
  int missing_nibbles = 0;
  int byte_pair_mismatches = 0;
  int byte_pair_swaps = 0;
  int contiguous_dest_byte_runs = 0;
  int previous_dest_byte = -2;
  int run_histogram[33]{};
  int current_run = 0;
  for (int source = 0; source < 16384; ++source) {
    missing_nibbles += source_to_dest[source] < 0;
  }
  for (int byte = 0; byte < 8192; ++byte) {
    const int lo = source_to_dest[byte * 2];
    const int hi = source_to_dest[byte * 2 + 1];
    if ((lo >> 1) != (hi >> 1)) {
      ++byte_pair_mismatches;
    } else {
      byte_pair_swaps += ((lo & 1) != 0 || (hi & 1) != 1);
      const int dest_byte = lo >> 1;
      if (dest_byte != previous_dest_byte + 1) {
        ++contiguous_dest_byte_runs;
        if (current_run > 0) {
          ++run_histogram[current_run < 32 ? current_run : 32];
        }
        current_run = 1;
      } else {
        ++current_run;
      }
      previous_dest_byte = dest_byte;
    }
  }
  if (current_run > 0) {
    ++run_histogram[current_run < 32 ? current_run : 32];
  }
  std::cout << "ff1_size=" << cute::size(Ff1SmemLayoutD{})
            << " ff1_cosize=" << cute::cosize(Ff1SmemLayoutD{})
            << " ff2_size=" << cute::size(Ff2StageLayoutA{})
            << " ff2_cosize=" << cute::cosize(Ff2StageLayoutA{})
            << " exact_type_match="
            << std::is_same_v<Ff1Stage0Layout, Ff2StageLayoutA>
            << " linear_quarter_mismatches=" << linear_quarter_mismatches
            << " first=(q=" << first_q << ",m=" << first_m << ",n=" << first_n
            << ",ff1=" << first_ff1 << ",ff2=" << first_ff2 << ")"
            << " missing_nibbles=" << missing_nibbles
            << " byte_pair_mismatches=" << byte_pair_mismatches
            << " byte_pair_swaps=" << byte_pair_swaps
            << " dest_byte_runs=" << contiguous_dest_byte_runs
            << '\n';
  std::cout << "run_histogram";
  for (int i = 1; i <= 32; ++i) {
    if (run_histogram[i] != 0) {
      std::cout << ' ' << i << ':' << run_histogram[i];
    }
  }
  std::cout << '\n';
}
