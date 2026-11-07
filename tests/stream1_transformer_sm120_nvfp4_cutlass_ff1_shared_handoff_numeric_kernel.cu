#include "stream1_transformer_sm120_nvfp4_cutlass_ff1_shared_handoff.hpp"

#include <type_traits>

using namespace stream1_sm120_nvfp4_cutlass_dsm;

#ifndef STREAM1_HANDOFF_CUSTOM
#define STREAM1_HANDOFF_CUSTOM 0
#endif

static_assert(STREAM1_HANDOFF_CUSTOM >= 0 && STREAM1_HANDOFF_CUSTOM <= 2);
using SelectedKernel = std::conditional_t<
    STREAM1_HANDOFF_CUSTOM == 2,
    Ff1SharedOnlyKernel,
    std::conditional_t<
        STREAM1_HANDOFF_CUSTOM == 1, Ff1SharedHandoffKernel, Ff1Kernel>>;

__global__ __launch_bounds__(
    SelectedKernel::MaxThreadsPerBlock,
    SelectedKernel::MinBlocksPerMultiprocessor)
void run_selected_kernel(
    CUTLASS_GRID_CONSTANT typename SelectedKernel::Params const params) {
  extern __shared__ char shared_storage[];
  SelectedKernel{}(params, shared_storage);
}
