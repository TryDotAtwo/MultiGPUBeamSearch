#include "stream1_transformer_sm120_nvfp4_cutlass_ff1_shared_handoff.hpp"

#include <cutlass/util/packed_stride.hpp>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

using namespace stream1_sm120_nvfp4_cutlass_dsm;

namespace ff1_ff2_fused_test {

constexpr int kM = 128;
constexpr int kHidden = 256;
constexpr int kModel = 256;
constexpr int kSlices = kHidden / static_cast<int>(Contract::kSliceColumns);

using Ff1Storage = typename Ff1SharedOnlyKernel::SharedStorage;
using Ff2Storage = BulkDsmFf2Collective::SharedStorage;

CUTLASS_DEVICE void invoke_ff1_for_diagnostic(
    typename Ff1SharedOnlyKernel::Params const& params,
    char* shared_storage) {
#if defined(STREAM1_CUTLASS_DIAG_NOOP_FF1_CALL)
  (void)params;
  (void)shared_storage;
  return;
#elif defined(STREAM1_CUTLASS_DIAG_DIRECT_FF1_COLLECTIVES)
  auto& storage = *reinterpret_cast<Ff1Storage*>(shared_storage);
  using NestedMainloop =
      typename Ff1SharedOnlyKernel::CollectiveMainloop;
  using NestedEpilogue =
      typename Ff1SharedOnlyKernel::CollectiveEpilogue;
  // SM120 block-scaled TMA collectives are stateless.  Their runtime state is
  // carried explicitly by Params and the pipeline objects owned by the kernel;
  // unlike the SM100 TMA specialization, this collective is default-created.
  NestedMainloop mainloop;
  NestedEpilogue epilogue(
      params.epilogue, storage.tensors.epilogue);
  (void)mainloop;
  (void)epilogue;
#if defined(STREAM1_CUTLASS_DIAG_DIRECT_FF1_PREFETCH)
  const int warp_idx = static_cast<int>(threadIdx.x) / 32;
  const int lane_predicate = cute::elect_one_sync();
  if (warp_idx == 0 && lane_predicate) {
    NestedMainloop::prefetch_tma_descriptors(params.mainloop);
    NestedEpilogue::prefetch_tma_descriptors(params.epilogue);
  }
#endif
#if defined(STREAM1_CUTLASS_DIAG_DIRECT_FF1_PIPELINE_CONSTRUCT)
  using MainloopPipeline = typename NestedMainloop::MainloopPipeline;
  typename MainloopPipeline::Params pipeline_params;
  const int warp_group = static_cast<int>(threadIdx.x) / 128;
  const int warp_in_group =
      (static_cast<int>(threadIdx.x) / 32) % 4;
  if (warp_group == 0 &&
      (warp_in_group == 0 || warp_in_group == 3)) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
  }
  if (warp_group == 1 || warp_group == 2) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
  }
  pipeline_params.is_leader =
      (static_cast<int>(threadIdx.x) % 128) == 0;
  pipeline_params.num_consumers = 256;
  pipeline_params.num_producers = NestedMainloop::NumProducerThreadEvents;
  pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;
  MainloopPipeline pipeline(
      storage.pipelines.mainloop, pipeline_params,
      typename Ff1SharedOnlyKernel::ClusterShape{});
  (void)pipeline;
#if defined(STREAM1_CUTLASS_DIAG_DIRECT_FF1_LOAD_INIT)
  // The logical FF1 collective is a singleton even though the surrounding
  // handoff kernel launches a physical two-CTA cluster.  Match CUTLASS' own
  // singleton visibility fence before materializing the tiled TMA tensors.
  __syncthreads();
  auto problem_shape_mnkl =
      cute::append<4>(params.problem_shape, cute::Int<1>{});
  auto load_inputs = mainloop.load_init(problem_shape_mnkl, params.mainloop);
  (void)load_inputs;
#endif
#endif
  return;
#else
  Ff1SharedOnlyKernel{}(params, shared_storage);
#endif
}

__device__ void move_bytes(
    std::uint8_t* destination, const std::uint8_t* source,
    std::size_t bytes) {
  if (destination < source) {
    for (std::size_t i = 0; i < bytes; ++i) destination[i] = source[i];
  } else if (destination > source) {
    for (std::size_t i = bytes; i-- > 0;) destination[i] = source[i];
  }
}

__global__ __launch_bounds__(384, 1) __cluster_dims__(2, 1, 1)
void ff1_ff2_fused_two_slice_kernel(
    CUTLASS_GRID_CONSTANT typename Ff1SharedOnlyKernel::Params const ff1_params,
    CUTLASS_GRID_CONSTANT typename Ff2Mainloop::Params const ff2_params,
    float* rank_checksums,
    std::uint32_t* address_info,
    int stop_stage) {
  namespace cg = cooperative_groups;
  extern __shared__ __align__(1024) unsigned char shared_bytes[];
  auto& ff1_storage = *reinterpret_cast<Ff1Storage*>(shared_bytes);
  auto cluster = cg::this_cluster();
  const std::uint32_t rank = cluster.block_rank();

  // Bounded Molab diagnostics. 10 proves that the physical cluster itself is
  // valid; 11/12 isolate one logical-singleton FF1 CTA at a time. These modes
  // return before the DSM handoff and never participate in production runs.
  if (stop_stage == 10) {
    cluster.sync();
    return;
  }

  if (stop_stage == -3 && rank == 0U) {
    invoke_ff1_for_diagnostic(
        ff1_params, reinterpret_cast<char*>(shared_bytes));
  }
  if (stop_stage == -3) cluster.sync();
  if ((stop_stage != -2 && stop_stage != -3 &&
       stop_stage != 11 && stop_stage != 12) ||
      (stop_stage == -2 && rank == 1U) ||
      (stop_stage == -3 && rank == 1U) ||
      (stop_stage == 11 && rank == 0U) ||
      (stop_stage == 12 && rank == 1U)) {
    invoke_ff1_for_diagnostic(
        ff1_params, reinterpret_cast<char*>(shared_bytes));
  }
#if defined(STREAM1_CUTLASS_DIAG_OUTER_RETURN_AFTER_FF1)
  // Pair with an early-return gate inside the nested CUTLASS kernel.  Returning
  // the complete physical-cluster kernel here prevents the surrounding DSM
  // handoff synchronization from obscuring whether the CUTLASS gate itself
  // was reached.  The define is uniform for both CTAs and all threads.
  return;
#endif
  __syncthreads();
  cluster.sync();

  if (threadIdx.x == 0U) address_info[rank * 2U] = 1U;
  if (stop_stage == 1 || stop_stage == 11 || stop_stage == 12) return;
  auto* produced_ring = ff1_shared_only_hidden_ring(ff1_storage);
  auto& ff2_storage = *reinterpret_cast<Ff2Storage*>(shared_bytes);

  if (threadIdx.x == 0U) {
    auto* destination = reinterpret_cast<std::uint8_t*>(
        &ff2_storage.hidden[rank]);
    auto* source = reinterpret_cast<const std::uint8_t*>(
        &produced_ring[rank]);
    address_info[rank * 2U] = static_cast<std::uint32_t>(
        source - reinterpret_cast<std::uint8_t*>(shared_bytes));
    address_info[rank * 2U + 1U] = static_cast<std::uint32_t>(
        destination - reinterpret_cast<std::uint8_t*>(shared_bytes));
    move_bytes(destination, source, sizeof(DsmASlot));
  }
  __syncthreads();
  cluster.sync();
  if (threadIdx.x == 0U) address_info[rank * 2U] = 2U;
  if (stop_stage == 2) return;

  const std::uint32_t warp_group = threadIdx.x / 128U;
  const std::uint32_t wg_thread = threadIdx.x % 128U;
  if (threadIdx.x == 0U) {
    cutlass::arch::ClusterBarrier::init(&ff2_storage.handoff_armed, 2U);
  }

  using Pipeline = BulkDsmFf2Collective::MainloopPipeline;
  using PipelineState = BulkDsmFf2Collective::PipelineState;
  typename Pipeline::Params pipeline_params;
  pipeline_params.transaction_bytes =
      BulkDsmFf2Collective::kTotalTransactionBytes;
  pipeline_params.num_consumers = 128U;
  pipeline_params.is_leader = wg_thread == 0U;
  pipeline_params.role = warp_group == 0U
      ? Pipeline::ThreadCategory::Producer
      : Pipeline::ThreadCategory::Consumer;
  Pipeline pipeline(
      ff2_storage.pipeline, pipeline_params, LaunchClusterShape{});
  __syncthreads();
  cutlass::arch::fence_barrier_init();
  cute::cluster_arrive_relaxed();
  cute::cluster_wait();

  constexpr auto problem_shape = make_shape(kM, kModel, kHidden, 1);
  Ff2Mainloop mainloop;
  const auto load_inputs = mainloop.load_init(problem_shape, ff2_params);
  if (warp_group == 0U && wg_thread == 0U) {
    PipelineState write_state = cutlass::make_producer_start_state<Pipeline>();
    for (std::uint32_t slice = 0; slice < kSlices; ++slice) {
      pipeline.producer_acquire(write_state);
      issue_ff2_b_sfb_tma(
          ff2_params, pipeline, write_state, load_inputs,
          rank, slice, ff2_storage);
      const std::uint32_t owner = slice & 1U;
      arrive_handoff_armed(&ff2_storage.handoff_armed, owner);
      if (rank == owner) {
        cutlass::arch::ClusterBarrier::wait(
            &ff2_storage.handoff_armed, 0U);
        issue_ff2_a_sfa_bulk_dsm(
            pipeline, write_state, ff2_storage,
            slice % Contract::kRingSlots);
      }
      ++write_state;
    }
    pipeline.producer_tail(write_state);
  } else if (warp_group == 1U) {
    typename Ff2Mainloop::TiledMma tiled_mma;
    auto accum = partition_fragment_C(
        tiled_mma, take<0, 2>(TileShape{}));
    PipelineState read_state;
    mainloop.mma(
        pipeline, read_state, accum, kSlices,
        static_cast<int>(wg_thread), ff2_storage.tensors,
        ff2_params, make_coord(0, rank, 0, 0));
    float local = 0.0F;
    for (int i = 0; i < size(accum); ++i) {
      local += accum(i) < 0.0F ? -accum(i) : accum(i);
    }
    atomicAdd(rank_checksums + rank, local);
  }
}

void require_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
    std::exit(2);
  }
}

template <class T>
T* allocate_bytes(std::size_t bytes) {
  T* pointer = nullptr;
  require_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc");
  return pointer;
}

}  // namespace ff1_ff2_fused_test

using namespace ff1_ff2_fused_test;

int main(int argc, char** argv) {
  const int stop_stage = argc > 1 ? std::atoi(argv[1]) : 0;
  constexpr auto ff1_shape = make_shape(kM, kHidden, kModel, 1);
  constexpr auto ff2_shape = make_shape(kM, kModel, kHidden, 1);
  using Ff1ScaleConfig = typename Ff1Mainloop::Sm1xxBlkScaledConfig;
  using Ff2ScaleConfig = typename Ff2Mainloop::Sm1xxBlkScaledConfig;
  using OutputScaleConfig =
      cutlass::detail::Sm1xxBlockScaledOutputConfig<Contract::kScaleVector>;

  const auto ff1_stride_a = cutlass::make_cute_packed_stride(
      typename Ff1SharedOnlyKernel::StrideA{}, {kM, kModel, 1});
  const auto ff1_stride_b = cutlass::make_cute_packed_stride(
      typename Ff1SharedOnlyKernel::StrideB{}, {kHidden, kModel, 1});
  const auto ff1_stride_c = cutlass::make_cute_packed_stride(
      typename Ff1SharedOnlyKernel::StrideC{}, {kM, kHidden, 1});
  const auto ff1_stride_d = cutlass::make_cute_packed_stride(
      typename Ff1SharedOnlyKernel::StrideD{}, {kM, kHidden, 1});
  const auto ff1_sfa_layout =
      Ff1ScaleConfig::tile_atom_to_shape_SFA(ff1_shape);
  const auto ff1_sfb_layout =
      Ff1ScaleConfig::tile_atom_to_shape_SFB(ff1_shape);
  const auto ff1_sfd_layout =
      OutputScaleConfig::tile_atom_to_shape_SFD(ff1_shape);

  auto* ff1_a = allocate_bytes<ElementHidden>(kM * kModel / 2U);
  auto* ff1_b = allocate_bytes<ElementHidden>(kHidden * kModel / 2U);
  auto* ff1_c = allocate_bytes<ElementOutput>(kM * kHidden * sizeof(ElementOutput));
  auto* ff1_d = allocate_bytes<ElementHidden>(kM * kHidden / 2U);
  auto* ff1_sfa = allocate_bytes<ElementScale>(size(filter_zeros(ff1_sfa_layout)));
  auto* ff1_sfb = allocate_bytes<ElementScale>(size(filter_zeros(ff1_sfb_layout)));
  auto* ff1_sfd = allocate_bytes<ElementScale>(size(filter_zeros(ff1_sfd_layout)));
  auto* ff1_bias = allocate_bytes<ElementBias>(kHidden * sizeof(ElementBias));
  auto* norm = allocate_bytes<ElementCompute>(sizeof(ElementCompute));
  require_cuda(cudaMemset(ff1_a, 0x11, kM * kModel / 2U), "memset ff1 A");
  require_cuda(cudaMemset(ff1_b, 0x21, kHidden * kModel / 2U), "memset ff1 B");
  require_cuda(cudaMemset(ff1_c, 0, kM * kHidden * sizeof(ElementOutput)), "memset ff1 C");
  require_cuda(cudaMemset(ff1_d, 0, kM * kHidden / 2U), "memset ff1 D");
  require_cuda(cudaMemset(ff1_bias, 0, kHidden * sizeof(ElementBias)), "memset bias");
  const auto scale_one = ElementScale(1.0F).raw();
  require_cuda(cudaMemset(ff1_sfa, scale_one, size(filter_zeros(ff1_sfa_layout))), "memset ff1 SFA");
  require_cuda(cudaMemset(ff1_sfb, scale_one, size(filter_zeros(ff1_sfb_layout))), "memset ff1 SFB");
  const ElementCompute norm_value = 6.0F;
  require_cuda(cudaMemcpy(norm, &norm_value, sizeof(norm_value), cudaMemcpyHostToDevice), "copy norm");

  typename Ff1SharedOnlyKernel::Arguments ff1_arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      ff1_shape,
      {ff1_a, ff1_stride_a, ff1_b, ff1_stride_b,
       ff1_sfa, ff1_sfa_layout, ff1_sfb, ff1_sfb_layout},
      {{1.0F, 0.0F}, ff1_c, ff1_stride_c, ff1_d, ff1_stride_d}};
  ff1_arguments.epilogue.thread.block_scale_factor_ptr = ff1_sfd;
  ff1_arguments.epilogue.thread.norm_constant_ptr = norm;
  ff1_arguments.epilogue.thread.bias_ptr = ff1_bias;
  if (!Ff1SharedOnlyKernel::can_implement(ff1_arguments)) return 3;
  auto ff1_params = Ff1SharedOnlyKernel::to_underlying_arguments(
      ff1_arguments, nullptr);

  const typename Ff2Mainloop::StrideA ff2_stride_a =
      make_stride(std::int64_t{kHidden}, _1{}, std::int64_t{kM * kHidden});
  const typename Ff2Mainloop::StrideB ff2_stride_b =
      make_stride(std::int64_t{kHidden}, _1{}, std::int64_t{kModel * kHidden});
  const auto ff2_sfa_layout =
      Ff2ScaleConfig::tile_atom_to_shape_SFA(ff2_shape);
  const auto ff2_sfb_layout =
      Ff2ScaleConfig::tile_atom_to_shape_SFB(ff2_shape);
  auto* ff2_unused_a = allocate_bytes<ElementHidden>(kM * kHidden / 2U);
  auto* ff2_b = allocate_bytes<ElementHidden>(kModel * kHidden / 2U);
  auto* ff2_unused_sfa = allocate_bytes<ElementScale>(size(filter_zeros(ff2_sfa_layout)));
  auto* ff2_sfb = allocate_bytes<ElementScale>(size(filter_zeros(ff2_sfb_layout)));
  require_cuda(cudaMemset(ff2_unused_a, 0, kM * kHidden / 2U), "memset ff2 A");
  require_cuda(cudaMemset(ff2_b, 0x31, kModel * kHidden / 2U), "memset ff2 B");
  require_cuda(cudaMemset(ff2_unused_sfa, 0, size(filter_zeros(ff2_sfa_layout))), "memset ff2 SFA");
  require_cuda(cudaMemset(ff2_sfb, scale_one, size(filter_zeros(ff2_sfb_layout))), "memset ff2 SFB");
  typename Ff2Mainloop::Arguments ff2_arguments{
      ff2_unused_a, ff2_stride_a, ff2_b, ff2_stride_b,
      ff2_unused_sfa, ff2_sfa_layout, ff2_sfb, ff2_sfb_layout};
  if (!Ff2Mainloop::can_implement(ff2_shape, ff2_arguments)) return 4;
  auto ff2_params = Ff2Mainloop::to_underlying_arguments(
      ff2_shape, ff2_arguments, nullptr);

  float* checksums = allocate_bytes<float>(2U * sizeof(float));
  auto* address_info = allocate_bytes<std::uint32_t>(4U * sizeof(std::uint32_t));
  constexpr std::size_t shared_bytes =
      sizeof(Ff1Storage) > sizeof(Ff2Storage)
          ? sizeof(Ff1Storage) : sizeof(Ff2Storage);
  require_cuda(cudaFuncSetAttribute(
      ff1_ff2_fused_two_slice_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared_bytes)), "set fused shared memory");
  require_cuda(cudaMemset(checksums, 0, 2U * sizeof(float)), "clear checksums");
  ff1_ff2_fused_two_slice_kernel<<<2, 384, shared_bytes>>>(
      ff1_params, ff2_params, checksums, address_info, stop_stage);
  require_cuda(cudaGetLastError(), "fused launch");
  require_cuda(cudaDeviceSynchronize(), "fused sync");
  float host_checksums[2]{};
  std::uint32_t host_addresses[4]{};
  require_cuda(cudaMemcpy(host_checksums, checksums, sizeof(host_checksums), cudaMemcpyDeviceToHost), "copy checksums");
  require_cuda(cudaMemcpy(host_addresses, address_info, sizeof(host_addresses), cudaMemcpyDeviceToHost), "copy addresses");
  std::cout << "ff1_ff2_fused_two_slice=pass"
            << " rank0=" << host_checksums[0]
            << " rank1=" << host_checksums[1]
            << " ff1_hidden_offset=" << host_addresses[0]
            << " ff2_hidden_offset=" << host_addresses[1]
            << " shared_bytes=" << shared_bytes << '\n';
  return !(host_checksums[0] > 0.0F && host_checksums[1] > 0.0F);
}
