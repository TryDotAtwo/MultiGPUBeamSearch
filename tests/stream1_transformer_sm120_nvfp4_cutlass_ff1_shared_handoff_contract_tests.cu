#include "stream1_transformer_sm120_nvfp4_cutlass_ff1_shared_handoff.hpp"

#ifndef STREAM1_HANDOFF_NO_MAIN
#include <iostream>
#include <type_traits>
#endif

using namespace stream1_sm120_nvfp4_cutlass_dsm;

#ifndef STREAM1_HANDOFF_NO_MAIN
static_assert(std::is_same_v<
    typename Ff1SharedHandoffKernel::CollectiveEpilogue,
    Ff1SharedHandoffEpilogue>);
static_assert(Ff1SharedHandoffKernel::MaxThreadsPerBlock == 384);
static_assert(std::is_same_v<
    typename Ff1SharedOnlyKernel::CollectiveEpilogue,
    Ff1SharedOnlyEpilogue>);
static_assert(Ff1SharedOnlyKernel::MaxThreadsPerBlock == 384);
static_assert(std::is_void_v<typename Ff1SharedOnlyEpilogue::ElementD>);
static_assert(sizeof(typename Ff1SharedScaleStore::SharedStorage) >=
              sizeof(DsmASlot) * Contract::kRingSlots);
#endif

__global__ __launch_bounds__(
    Ff1SharedOnlyKernel::MaxThreadsPerBlock,
    Ff1SharedOnlyKernel::MinBlocksPerMultiprocessor)
void instantiate_ff1_shared_only_operator(
    CUTLASS_GRID_CONSTANT Ff1SharedOnlyKernel::Params const params) {
#if defined(STREAM1_HANDOFF_DEVICE_TRACE)
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("ff1_trace kernel_enter\\n");
  }
#endif
  extern __shared__ char storage[];
  Ff1SharedOnlyKernel{}(params, storage);
#if defined(STREAM1_HANDOFF_DEVICE_TRACE)
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("ff1_trace kernel_exit\\n");
  }
#endif
}

#ifndef STREAM1_HANDOFF_NO_MAIN
int main() {
  auto* shared_only_kernel = instantiate_ff1_shared_only_operator;
  std::cout << "ff1_shared_handoff_type_gate=pass"
            << " epilogue_storage="
            << sizeof(typename Ff1SharedHandoffEpilogue::SharedStorage)
            << " kernel_storage="
            << sizeof(typename Ff1SharedHandoffKernel::SharedStorage)
            << " shared_only="
            << (shared_only_kernel != nullptr ? "pass" : "fail")
            << " shared_only_epilogue_storage="
            << sizeof(typename Ff1SharedOnlyEpilogue::SharedStorage)
            << " shared_only_kernel_storage="
            << sizeof(typename Ff1SharedOnlyKernel::SharedStorage)
            << '\n';
  return shared_only_kernel == nullptr;
}
#endif
