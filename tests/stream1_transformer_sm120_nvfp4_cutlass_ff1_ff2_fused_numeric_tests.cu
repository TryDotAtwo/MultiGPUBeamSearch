#include "stream1_transformer_sm120_nvfp4_cutlass_ff1_shared_handoff.hpp"
#include "stream1_transformer_sm120_nvfp4_fused_epilogue.cuh"

#include <cutlass/util/packed_stride.hpp>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include "sm120_fused_numeric_reference.hpp"
#include "sm120_ff1_register_pack.hpp"

using namespace stream1_sm120_nvfp4_cutlass_dsm;

namespace ff1_ff2_fused_test {

constexpr int kM = 128;
constexpr int kHidden = sm120_fused_reference::kHidden;
constexpr int kModel = 256;
constexpr int kSlices = kHidden / static_cast<int>(Contract::kSliceColumns);
// Ping-pong uses two independent 128-thread math groups on different tiles.
// This two-slice diagnostic invokes one tile per CTA, so only one math group
// may consume it. Cooperative 256-thread ownership is a different MMA type.
constexpr int kFf1ConsumerThreads = 128;
#ifndef STREAM1_FUSED_NATIVE_EPILOGUE
#define STREAM1_FUSED_NATIVE_EPILOGUE 0
#endif
#ifndef STREAM1_FUSED_TEST_THREADS
#define STREAM1_FUSED_TEST_THREADS 384
#endif
constexpr int kKernelThreads = STREAM1_FUSED_TEST_THREADS;
static_assert(kKernelThreads == 256 || kKernelThreads == 384);
#ifndef STREAM1_FUSED_DIRECT_PACK
#define STREAM1_FUSED_DIRECT_PACK 0
#endif
constexpr bool kDirectRegisterPack = STREAM1_FUSED_DIRECT_PACK != 0;
#ifndef STREAM1_FUSED_NO_RELOCATION
#define STREAM1_FUSED_NO_RELOCATION 0
#endif
#ifndef STREAM1_FUSED_ACCUMULATE_DIRECT
#define STREAM1_FUSED_ACCUMULATE_DIRECT 0
#endif
constexpr bool kNoRelocation = STREAM1_FUSED_NO_RELOCATION != 0;
constexpr bool kAccumulateDirect = STREAM1_FUSED_ACCUMULATE_DIRECT != 0;
static_assert(!kNoRelocation || kDirectRegisterPack);
#ifndef STREAM1_FUSED_SHARED_CARRY
#define STREAM1_FUSED_SHARED_CARRY 0
#endif
#ifndef STREAM1_FUSED_PROXY_FENCE
#define STREAM1_FUSED_PROXY_FENCE 1
#endif
#ifndef STREAM1_FUSED_HANDOFF_TRACE
#define STREAM1_FUSED_HANDOFF_TRACE 0
#endif
#ifndef STREAM1_FUSED_INVALIDATE_BARRIERS
#define STREAM1_FUSED_INVALIDATE_BARRIERS 1
#endif
#ifndef STREAM1_ROLE_SEPARATED_4CTA
#define STREAM1_ROLE_SEPARATED_4CTA 0
#endif
#ifndef STREAM1_ROLE_PRECOMPUTE_FF1
#define STREAM1_ROLE_PRECOMPUTE_FF1 0
#endif
#ifndef STREAM1_ROLE_DUAL_FF1_MATH_WG
#define STREAM1_ROLE_DUAL_FF1_MATH_WG 0
#endif
#ifndef STREAM1_FUSED_WAIT_DEPENDENT_GRIDS
#define STREAM1_FUSED_WAIT_DEPENDENT_GRIDS 1
#endif
constexpr bool kSharedCarry = STREAM1_FUSED_SHARED_CARRY != 0;
constexpr bool kRoleSeparated4Cta = STREAM1_ROLE_SEPARATED_4CTA != 0;
constexpr bool kDualRoleFf1MathWg = STREAM1_ROLE_DUAL_FF1_MATH_WG != 0;
static_assert(!kSharedCarry || (kDirectRegisterPack && kAccumulateDirect && !kNoRelocation));
static_assert(!kRoleSeparated4Cta || kDirectRegisterPack || STREAM1_FUSED_NATIVE_EPILOGUE);
static_assert(!kDualRoleFf1MathWg ||
              (kRoleSeparated4Cta && kKernelThreads == 384 &&
               STREAM1_FUSED_NATIVE_EPILOGUE));
using NextActivationScaleConfig = typename Ff2Mainloop::Sm1xxBlkScaledConfig;
using NextActivationScaleLayout = decltype(
    NextActivationScaleConfig::tile_atom_to_shape_SFA(ProblemShape{}));

// Keep ElementC in the collective ABI even though beta is identically zero.
// CUTLASS' SM120 block-scale visitor requires the concrete C element while its
// runtime source-load predicate removes the unused read.  D remains void, so
// the quantized FF1 result is materialized only in the shared handoff ring.
#if STREAM1_FUSED_SHARED_CARRY
using Ff1TestMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,cutlass::arch::OpClassBlockScaledTensorOp,
    ElementAB,LayoutA,32,ElementAB,LayoutB,32,ElementAccumulator,
    TileShape,CollectiveClusterShape,cutlass::gemm::collective::StageCount<2>,
    KernelSchedule>::CollectiveOp;
using Ff1TestKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,Ff1TestMainloop,Ff1SharedOnlyEpilogue,void>;
static_assert(std::is_same_v<typename Ff1TestMainloop::TiledMma,typename Ff1Mainloop::TiledMma>);
#else
using Ff1TestMainloop = Ff1Mainloop;
using Ff1TestKernel = Ff1SharedOnlyKernel;
#endif
#if STREAM1_FUSED_SHARED_CARRY
// No CUTLASS epilogue executes in the direct-register path. Do not reserve its
// shared storage or scheduler gaps while an FF2 partial sum remains live.
struct alignas(1024) Ff1Storage {
  struct alignas(1024) Tensors {
    typename Ff1TestMainloop::TensorStorage mainloop;
  } tensors;
  struct Pipelines {
    typename Ff1TestMainloop::MainloopPipeline::SharedStorage mainloop;
  } pipelines;
};
#else
using Ff1Storage = typename Ff1TestKernel::SharedStorage;
#endif
using Ff2Storage = BulkDsmFf2Collective::SharedStorage;
constexpr std::size_t kProducedRingOffset = kNoRelocation
    ? offsetof(Ff2Storage,hidden) : kM * Contract::kSliceColumns * sizeof(float);
constexpr int kCarryRegisterElements = 64;
constexpr int kCarrySharedElements = 128-kCarryRegisterElements;
constexpr std::size_t kCarryBytes = 128U*kCarrySharedElements*sizeof(float);
constexpr std::size_t kFf1BaseOffset = kSharedCarry ? kCarryBytes : 0U;
constexpr std::size_t kFf1TensorEnd = offsetof(Ff1Storage,tensors.mainloop)
    + sizeof(((Ff1Storage*)nullptr)->tensors.mainloop);
constexpr std::size_t kFf1PipelineEnd = offsetof(Ff1Storage,pipelines.mainloop)
    + sizeof(((Ff1Storage*)nullptr)->pipelines.mainloop);
constexpr std::size_t kFf1ActiveBytes = std::max(kFf1TensorEnd,kFf1PipelineEnd);
constexpr std::size_t kSharedCarryArenaBytes = std::max(
    std::max(kFf1BaseOffset+kFf1ActiveBytes,kProducedRingOffset+Contract::kRingBytes),
    sizeof(Ff2Storage));
static_assert(!kSharedCarry || kSharedCarryArenaBytes <= 99U*1024U);

constexpr std::size_t kRoleSeparatedArenaBytes = std::max(
    std::max(kFf1BaseOffset + kFf1ActiveBytes,
             kProducedRingOffset + Contract::kRingBytes),
    sizeof(Ff2Storage));
struct alignas(1024) RoleSeparatedNumericStorage {
  alignas(1024) unsigned char arena[kRoleSeparatedArenaBytes];
  alignas(8) std::uint64_t armed;
  alignas(8) std::uint64_t free;
  alignas(8) std::uint64_t receive[Ff2Mainloop::DispatchPolicy::Stages];
};
static_assert(sizeof(RoleSeparatedNumericStorage) <= 99U * 1024U);

template <class Kernel = Ff1TestKernel, bool DualMath = false>
#if STREAM1_FUSED_NOINLINE_FF1
__device__ __noinline__
#else
CUTLASS_DEVICE
#endif
void invoke_ff1_for_diagnostic(
    typename Ff1TestKernel::Params const& params,
    char* shared_storage,
    ElementBias const* bias,
    float* raw_ff1,
    int first_slice = 0,
    int logical_ff1_rank = -1,
    int physical_cluster_ctas = 2) {
#if defined(STREAM1_CUTLASS_DIAG_NOOP_FF1_CALL)
  (void)params;
  (void)shared_storage;
  return;
#elif defined(STREAM1_CUTLASS_DIAG_DIRECT_FF1_COLLECTIVES)
  auto& storage = *reinterpret_cast<Ff1Storage*>(shared_storage);
  using NestedMainloop =
      typename Ff1TestKernel::CollectiveMainloop;
  using NestedEpilogue =
      typename Kernel::CollectiveEpilogue;
  // SM120 block-scaled TMA collectives are stateless.  Their runtime state is
  // carried explicitly by Params and the pipeline objects owned by the kernel;
  // unlike the SM100 TMA specialization, this collective is default-created.
  NestedMainloop mainloop;
#if !STREAM1_FUSED_SHARED_CARRY
  NestedEpilogue epilogue(
      params.epilogue, storage.tensors.epilogue);
  (void)epilogue;
#endif
  (void)mainloop;
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
  if (warp_group == 1 || (DualMath && warp_group == 2)) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
  }
  pipeline_params.is_leader =
      (static_cast<int>(threadIdx.x) % 128) == 0;
  pipeline_params.num_consumers = kFf1ConsumerThreads;
  pipeline_params.num_producers = NestedMainloop::NumProducerThreadEvents;
  pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;
  MainloopPipeline pipeline(
      storage.pipelines.mainloop, pipeline_params,
      typename Ff1TestKernel::ClusterShape{});
  (void)pipeline;
#if defined(STREAM1_CUTLASS_DIAG_DIRECT_FF1_EPILOGUE)
  using EpiLoadPipeline = typename NestedEpilogue::LoadPipeline;
  typename EpiLoadPipeline::Params epi_load_params;
  if (warp_group == 0 && warp_in_group == 2) {
    epi_load_params.role = EpiLoadPipeline::ThreadCategory::Producer;
  }
  if (warp_group == 1 || (DualMath && warp_group == 2)) {
    epi_load_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
  }
  epi_load_params.dst_blockid = cute::block_rank_in_cluster();
  epi_load_params.producer_arv_count = 32;
  epi_load_params.consumer_arv_count = kFf1ConsumerThreads;
  if constexpr (NestedEpilogue::RequiresTransactionBytes) {
    epi_load_params.transaction_bytes = params.epilogue.tma_transaction_bytes;
  }
  EpiLoadPipeline epi_load_pipeline(
      storage.pipelines.epi_load, epi_load_params);

  using EpiStorePipeline = typename NestedEpilogue::StorePipeline;
  typename EpiStorePipeline::Params epi_store_params;
  epi_store_params.always_wait = true;
  EpiStorePipeline epi_store_pipeline(epi_store_params);

#endif
#if STREAM1_ROLE_DUAL_FF1_MATH_WG
  using MathWarpGroupOrderBarrier =
      typename Kernel::MathWarpGroupOrderBarrier;
  typename MathWarpGroupOrderBarrier::Params math_order_params;
  math_order_params.group_id = warp_group - 1;
  math_order_params.group_size = kFf1ConsumerThreads;
  MathWarpGroupOrderBarrier math_order_barrier(
      storage.pipelines.math_wg_order, math_order_params);
#endif
  // Match GemmUniversal's post-construction visibility point.  The manual
  // collective path is embedded in a larger physical cluster, but the nested
  // FF1 pipelines and ordered barrier are CTA-owned.
  __syncthreads();
#if defined(STREAM1_CUTLASS_DIAG_DIRECT_FF1_LOAD_INIT)
  // The logical FF1 collective is a singleton even though the surrounding
  // handoff kernel launches a physical two-CTA cluster.  Match CUTLASS' own
  // singleton visibility fence before materializing the tiled TMA tensors.
  __syncthreads();
  auto problem_shape_mnkl =
      cute::append<4>(params.problem_shape, cute::Int<1>{});
  auto load_inputs = mainloop.load_init(problem_shape_mnkl, params.mainloop);
  (void)load_inputs;
#if defined(STREAM1_CUTLASS_DIAG_DIRECT_FF1_MMA)
  auto gA_mkl = cute::get<0>(load_inputs);
  auto gB_nkl = cute::get<1>(load_inputs);
  const int ff1_rank = logical_ff1_rank >= 0
      ? logical_ff1_rank
      : static_cast<int>(cute::block_rank_in_cluster());
  auto m_coord = cute::idx2crd(
      static_cast<int>(blockIdx.x) / physical_cluster_ctas,
      cute::shape<2>(gA_mkl));
  // The surrounding physical cluster owns the two adjacent 128-column FF1
  // slices.  Their logical rank is independent of the physical rank so the
  // exact same collective can run in producer ranks 0/1 of a four-CTA
  // producer/consumer cluster.
  // The nested CUTLASS collective remains a logical singleton per CTA.
  const int ff1_n_tile = DualMath
      ? first_slice + ff1_rank * 2 +
            (warp_group >= 1 ? warp_group - 1 : 0)
      : first_slice + ff1_rank;
  auto n_coord = cute::idx2crd(
      ff1_n_tile,
      cute::shape<2>(gB_nkl));
  auto l_coord = cute::idx2crd(0, cute::shape<4>(gB_nkl));
  auto block_coord =
      cute::make_coord(m_coord, n_coord, cute::_, l_coord);
  const int k_tile_count = static_cast<int>(cute::size<3>(gA_mkl));
  auto k_tile_iter = cute::make_coord_iterator(
      cute::idx2crd(0, cute::shape<3>(gA_mkl)),
      cute::shape<3>(gA_mkl));
  typename NestedMainloop::PipelineState producer_state =
      cutlass::make_producer_start_state<MainloopPipeline>();
  typename NestedMainloop::PipelineState consumer_state;

#if defined(STREAM1_CUTLASS_DIAG_DIRECT_FF1_EPILOGUE)
  typename NestedEpilogue::LoadPipelineState epi_load_consumer_state;
  typename NestedEpilogue::LoadPipelineState epi_load_producer_state =
      cutlass::make_producer_start_state<EpiLoadPipeline>();
  typename NestedEpilogue::StorePipelineState epi_store_producer_state =
      cutlass::make_producer_start_state<EpiStorePipeline>();
  const bool epilogue_load_needed = epilogue.is_producer_load_needed();
  // CUTLASS ping-pong assigns the second math warp-group the next epilogue
  // pipeline tile.  Sharing the initial state makes both groups wait/store on
  // the same stage and deadlocks even though their N tiles are disjoint.
  if constexpr (DualMath) {
    if (warp_group == 2) {
      epi_load_consumer_state.advance(
          NestedEpilogue::get_load_pipe_increment(
              typename Ff1TestKernel::TileShape{}));
      epi_store_producer_state.advance(
          NestedEpilogue::get_store_pipe_increment(
              typename Ff1TestKernel::TileShape{}));
    }
  }
#endif

  using TiledMma = typename NestedMainloop::TiledMma;
  TiledMma tiled_mma;
  auto accumulators = cute::partition_fragment_C(
      tiled_mma,
      cute::take<0, 2>(typename Ff1TestKernel::TileShape{}));
  if (warp_group == 0 && warp_in_group == 0) {
#if STREAM1_FUSED_WAIT_DEPENDENT_GRIDS
    cutlass::arch::wait_on_dependent_grids();
#endif
    mainloop.load(
        params.mainloop, pipeline, producer_state, load_inputs,
        block_coord, k_tile_iter, k_tile_count,
        static_cast<int>(threadIdx.x) % 32, 0U,
        storage.tensors.mainloop);
    producer_state.advance(k_tile_count);
    if constexpr (DualMath) {
      const int second_ff1_n_tile = first_slice + ff1_rank * 2 + 1;
      auto second_n_coord = cute::idx2crd(
          second_ff1_n_tile, cute::shape<2>(gB_nkl));
      auto second_block_coord = cute::make_coord(
          m_coord, second_n_coord, cute::_, l_coord);
      auto second_k_tile_iter = cute::make_coord_iterator(
          cute::idx2crd(0, cute::shape<3>(gA_mkl)),
          cute::shape<3>(gA_mkl));
      mainloop.load(
          params.mainloop, pipeline, producer_state, load_inputs,
          second_block_coord, second_k_tile_iter, k_tile_count,
          static_cast<int>(threadIdx.x) % 32, 0U,
          storage.tensors.mainloop);
      producer_state.advance(k_tile_count);
    }
    mainloop.load_tail(pipeline, producer_state);
#if defined(STREAM1_CUTLASS_DIAG_DIRECT_FF1_EPILOGUE)
  } else if (warp_group == 0 && warp_in_group == 2 &&
             epilogue_load_needed) {
    epi_load_producer_state = epilogue.load(
        epi_load_pipeline, epi_load_producer_state,
        problem_shape_mnkl,
        typename Ff1TestKernel::TileShape{}, block_coord,
        tiled_mma, static_cast<int>(threadIdx.x) % 32,
        storage.tensors.epilogue, 0);
    epilogue.load_tail(epi_load_pipeline, epi_load_producer_state);
#endif
  } else if (warp_group == 1 || (DualMath && warp_group == 2)) {
#if STREAM1_ROLE_DUAL_FF1_MATH_WG
    if constexpr (DualMath) math_order_barrier.wait();
#endif
    if constexpr (DualMath) {
      if (warp_group == 2) {
        consumer_state.advance(k_tile_count);
      }
    }
    mainloop.mma(
        pipeline, consumer_state, accumulators, k_tile_count,
        static_cast<int>(threadIdx.x) % kFf1ConsumerThreads,
        storage.tensors.mainloop, params.mainloop, block_coord);
#if STREAM1_ROLE_DUAL_FF1_MATH_WG
    // Match CUTLASS' ping-pong kernel exactly: hand the ordered MMA turn to
    // the next math warp-group before draining this group's pipeline stages.
    // mma_tail() may wait for buffers that only the next group can release;
    // arriving after the tail therefore deadlocks the two consumers.
    if constexpr (DualMath) math_order_barrier.arrive();
#endif
    mainloop.mma_tail(pipeline, consumer_state, k_tile_count);
  }
#if defined(STREAM1_CUTLASS_DIAG_DIRECT_FF1_MANUAL_MATERIALIZE)
  // The scalar tile aliases the mainloop pipeline barriers as well as its
  // operands. mma_tail alone cannot end the producer's load_tail lifetime.
  // Keep the register fragments live until every role has drained the old
  // arena, then let the consumer overwrite it for quantization.
  __syncthreads();
#if STREAM1_FUSED_INVALIDATE_BARRIERS
  // Completion ends users, not the mbarrier object's lifetime. Invalidate all
  // drained barriers before their bytes become ordinary packed data/carry.
  if (threadIdx.x==0) {
    for (int stage=0;stage<NestedMainloop::DispatchPolicy::Stages;++stage) {
      cutlass::arch::ClusterBarrier::invalidate(reinterpret_cast<std::uint64_t const*>(
          &storage.pipelines.mainloop.full_barrier_[stage]));
      cutlass::arch::ClusterBarrier::invalidate(reinterpret_cast<std::uint64_t const*>(
          &storage.pipelines.mainloop.empty_barrier_[stage]));
    }
  }
  __syncthreads();
#endif
#endif
  if (warp_group == 1 || (DualMath && warp_group == 2)) {
#if defined(STREAM1_CUTLASS_DIAG_DIRECT_FF1_MANUAL_MATERIALIZE)
#if STREAM1_FUSED_DIRECT_PACK
    // Retain the same arena placement and all rendezvous points for this A/B.
    // The 64 KiB region is no longer written/read as an FP32 intermediate.
    auto* direct_ring = reinterpret_cast<DsmASlot*>(
        shared_storage + kProducedRingOffset-kFf1BaseOffset);
    const int physical_rank = DualMath ? (ff1_n_tile & 1) : ff1_rank;
    sm120_ff1_register_pack::store<kHidden>(
        accumulators, direct_ring[physical_rank], bias, ff1_n_tile, raw_ff1);
#else
    // SM120 GeForce accumulators are register fragments (not SM100 TMEM).
    // Partition an identity tile with the identical MMA thread slice to map
    // each register directly to its logical (row, column) destination.
    auto thread_mma = tiled_mma.get_slice(
        static_cast<int>(threadIdx.x) % kFf1ConsumerThreads);
    auto coordinates = cute::make_identity_tensor(
        cute::make_shape(cute::Int<kM>{}, cute::Int<Contract::kSliceColumns>{}));
    auto thread_coordinates = thread_mma.partition_C(coordinates);
    float* tile = reinterpret_cast<float*>(shared_storage);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < cute::size(accumulators); ++i) {
      auto mn = thread_coordinates(i);
      const int row = static_cast<int>(cute::get<0>(mn));
      const int column = static_cast<int>(cute::get<1>(mn));
      tile[row * Contract::kSliceColumns + column] =
          static_cast<float>(accumulators(i));
      if (raw_ff1) {
        raw_ff1[row * kHidden + (first_slice + ff1_rank) *
                    Contract::kSliceColumns + column] =
            static_cast<float>(accumulators(i));
      }
    }
#endif
#endif
#if defined(STREAM1_CUTLASS_DIAG_DIRECT_FF1_EPILOGUE)
#if STREAM1_ROLE_DUAL_FF1_MATH_WG
    if constexpr (DualMath) math_order_barrier.wait();
#endif
    // Diagnostic raw accumulator snapshot is excluded from timed launches.
    if (raw_ff1) {
      auto coordinates = cute::make_identity_tensor(
          cute::make_shape(cute::Int<kM>{}, cute::Int<Contract::kSliceColumns>{}));
      auto owned = tiled_mma.get_slice(static_cast<int>(threadIdx.x) %
          kFf1ConsumerThreads).partition_C(coordinates);
      for (int i=0; i<cute::size(accumulators); ++i) {
        auto mn = owned(i);
        raw_ff1[int(cute::get<0>(mn))*kHidden +
            ff1_n_tile*Contract::kSliceColumns + int(cute::get<1>(mn))] = accumulators(i);
      }
    }
    auto next_epilogue_states = epilogue.store(
        epi_load_pipeline, epi_load_consumer_state,
        epi_store_pipeline, epi_store_producer_state,
        problem_shape_mnkl,
        typename Ff1TestKernel::TileShape{}, block_coord,
        accumulators, tiled_mma,
        static_cast<int>(threadIdx.x) % kFf1ConsumerThreads,
        // -1 means every epilogue subtile. Zero computes only the first
        // 64x32 subtile, leaving 7/8 of the 128x128 handoff tile unwritten.
        storage.tensors.epilogue, -1);
    epi_load_consumer_state = cute::get<0>(next_epilogue_states);
    epi_store_producer_state = cute::get<1>(next_epilogue_states);
    epilogue.store_tail(
        epi_load_pipeline, epi_load_consumer_state,
        epi_store_pipeline, epi_store_producer_state);
#if STREAM1_ROLE_DUAL_FF1_MATH_WG
    if constexpr (DualMath) math_order_barrier.arrive();
#endif
#endif
  }
#if defined(STREAM1_CUTLASS_DIAG_DIRECT_FF1_MANUAL_MATERIALIZE)
  __syncthreads();
#if !STREAM1_FUSED_DIRECT_PACK
  constexpr std::size_t kTileBytes =
      std::size_t{kM} * Contract::kSliceColumns * sizeof(float);
  auto* manual_ring = reinterpret_cast<DsmASlot*>(shared_storage + kTileBytes);
  if (threadIdx.x < kM) {
    const std::uint32_t row = threadIdx.x;
    const std::uint32_t physical_rank =
        static_cast<std::uint32_t>(ff1_rank);
    float* tile = reinterpret_cast<float*>(shared_storage);
    CUTLASS_PRAGMA_UNROLL
    for (std::uint32_t group = 0;
         group < Contract::kSliceColumns / Contract::kScaleVector; ++group) {
      float values[Contract::kScaleVector];
      float max_abs = 0.0F;
      CUTLASS_PRAGMA_UNROLL
      for (std::uint32_t lane = 0; lane < Contract::kScaleVector; ++lane) {
        const std::uint32_t column = group * Contract::kScaleVector + lane;
        float value = tile[row * Contract::kSliceColumns + column] +
            static_cast<float>(bias[(first_slice + physical_rank) * Contract::kSliceColumns + column]);
        value = value > 0.0F ? value : 0.0F;
        values[lane] = value;
        max_abs = value > max_abs ? value : max_abs;
      }
      const float requested_scale = max_abs > 0.0F ? max_abs / 6.0F : 1.0F;
      const cutlass::float_ue4m3_t encoded_scale(requested_scale);
      const float reciprocal_scale = 1.0F / static_cast<float>(encoded_scale);
      std::uint8_t raw[Contract::kScaleVector];
      CUTLASS_PRAGMA_UNROLL
      for (std::uint32_t lane = 0; lane < Contract::kScaleVector; ++lane) {
        raw[lane] = static_cast<std::uint8_t>(
            cutlass::float_e2m1_t(values[lane] * reciprocal_scale).raw() & 0x0FU);
      }
      Ff1HiddenRingWriter::store_group(
          manual_ring[physical_rank], row, group, raw,
          static_cast<std::uint8_t>(encoded_scale.raw()));
    }
  }
#endif
  __syncthreads();
#endif
#endif
#endif
#endif
  return;
#else
  Ff1TestKernel{}(params, shared_storage);
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

#if STREAM1_FUSED_HANDOFF_TRACE
__device__ void trace_bytes(char const* stage, unsigned rank, unsigned slice,
                           unsigned char const* actual, unsigned char const* expected,
                           unsigned bytes) {
  if (threadIdx.x != 0) return;
  unsigned mismatches=0,first=bytes;
  for (unsigned i=0;i<bytes;++i) if (actual[i]!=expected[i]) {
    if (!mismatches) first=i;
    ++mismatches;
  }
  printf("handoff_trace stage=%s rank=%u slice=%u bad=%u first=%u actual=%u expected=%u\n",
         stage,rank,slice,mismatches,first,first<bytes?actual[first]:0,first<bytes?expected[first]:0);
}
#endif

// Production-shape FFN candidate: two intact 384-thread CUTLASS FF1 CTAs
// produce adjacent H128 slices while two intact FF2 CTAs consume separate
// D128 output halves.  The packed hidden activation exists only in producer
// shared memory and the consumer CUTLASS stages; no global hidden tensor is
// addressable.  B/SFB remains ordinary local TMA.  A/SFA is a bulk-DSM
// transaction into the same full barrier that the stock FF2 MMA waits on.
template <bool Production>
__global__ __launch_bounds__(kKernelThreads, 1) __cluster_dims__(4, 1, 1)
void ff1_ff2_role_separated_four_cta_kernel(
    CUTLASS_GRID_CONSTANT typename Ff1TestKernel::Params const ff1_params,
    CUTLASS_GRID_CONSTANT typename Ff2Mainloop::Params const ff2_params,
    ElementBias const* ff1_bias,
    std::uint8_t* ring_snapshots,
    float* raw_ff1,
    float* output_ff2,
    float* rank_checksums,
    int stop_stage) {
  namespace cg = cooperative_groups;
  extern __shared__ __align__(1024) unsigned char shared_bytes[];
  auto& role_storage = *reinterpret_cast<RoleSeparatedNumericStorage*>(shared_bytes);
  auto cluster = cg::this_cluster();
  const std::uint32_t rank = cluster.block_rank();
  const bool producer = RoleSeparatedCtaPlan::is_producer(rank);
  const std::uint32_t m_tile = blockIdx.x /
      RoleSeparatedCtaPlan::kClusterCtas;
  const std::uint32_t warp_group = threadIdx.x / 128U;
  const std::uint32_t wg_thread = threadIdx.x % 128U;
  if (raw_ff1) raw_ff1 += std::size_t(m_tile) * kM * kHidden;
  if (output_ff2) output_ff2 += std::size_t(m_tile) * kM * kModel;
  if (ring_snapshots) {
    ring_snapshots += std::size_t(m_tile) * kSlices * sizeof(DsmASlot);
  }
  if (rank_checksums) rank_checksums += std::size_t(m_tile) * 2U;

  using Pipeline = BulkDsmFf2Collective::MainloopPipeline;
  using PipelineState = BulkDsmFf2Collective::PipelineState;

  if (producer) {
    if (threadIdx.x == 0U) {
      cutlass::arch::ClusterBarrier::init(
          &role_storage.armed, RoleSeparatedCtaPlan::kConsumerCtas);
      cutlass::arch::ClusterBarrier::init(
          &role_storage.free, RoleSeparatedCtaPlan::kConsumerCtas);
    }
  } else if (threadIdx.x == 0U) {
    for (int stage = 0; stage < Ff2Mainloop::DispatchPolicy::Stages;
         ++stage) {
      cutlass::arch::ClusterTransactionBarrier::init(
          &role_storage.receive[stage], 1U);
    }
  }
  __syncthreads();
  cutlass::arch::fence_barrier_init();
  cluster.sync();

  if constexpr (!Production) {
    if (stop_stage == 10) return;
  }
  if constexpr (!Production) if (stop_stage == 11) {
    if (producer) {
      invoke_ff1_for_diagnostic<Ff1TestKernel, kDualRoleFf1MathWg>(
          ff1_params, reinterpret_cast<char*>(role_storage.arena),
          ff1_bias, raw_ff1, 0, static_cast<int>(rank),
          RoleSeparatedCtaPlan::kClusterCtas);
    }
    cluster.sync();
    return;
  }

  const bool one_pair_gate = !Production &&
      (stop_stage == 14 || stop_stage == 15);
  constexpr int kSlicesPerFf1Wave = kDualRoleFf1MathWg ? 4 : 2;
  const int active_slices = one_pair_gate ? kSlicesPerFf1Wave : kSlices;
  const int active_waves = active_slices / kSlicesPerFf1Wave;

  if (producer) {
    for (int wave = 0; wave < active_waves; ++wave) {
#if !STREAM1_ROLE_PRECOMPUTE_FF1
      if (!kDualRoleFf1MathWg && threadIdx.x == 0U) {
        cutlass::arch::ClusterBarrier::wait(
            &role_storage.armed, wave & 1U);
      }
      __syncthreads();
#endif
      const int first_slice = wave * kSlicesPerFf1Wave;
      invoke_ff1_for_diagnostic<Ff1TestKernel, kDualRoleFf1MathWg>(
          ff1_params, reinterpret_cast<char*>(role_storage.arena),
          ff1_bias, Production ? nullptr : raw_ff1, first_slice,
          static_cast<int>(rank), RoleSeparatedCtaPlan::kClusterCtas);
      __syncthreads();
#if STREAM1_FUSED_NATIVE_EPILOGUE
      auto& producer_storage = *reinterpret_cast<Ff1Storage*>(role_storage.arena);
      auto* produced_ring = ff1_shared_only_hidden_ring(producer_storage);
      // Complete native pipeline lifetimes before another FF1 invocation.
      // The payload remains disjoint and alive until both remote readers ack.
      if (threadIdx.x == 0U) {
        for (int s=0;s<Ff1TestMainloop::DispatchPolicy::Stages;++s) {
          cutlass::arch::ClusterBarrier::invalidate(reinterpret_cast<std::uint64_t const*>(&producer_storage.pipelines.mainloop.full_barrier_[s]));
          cutlass::arch::ClusterBarrier::invalidate(reinterpret_cast<std::uint64_t const*>(&producer_storage.pipelines.mainloop.empty_barrier_[s]));
        }
        for (int s=0;s<Ff1SharedOnlyEpilogue::DispatchPolicy::StagesC;++s) {
          cutlass::arch::ClusterBarrier::invalidate(reinterpret_cast<std::uint64_t const*>(&producer_storage.pipelines.epi_load.full_barrier_[s]));
          cutlass::arch::ClusterBarrier::invalidate(reinterpret_cast<std::uint64_t const*>(&producer_storage.pipelines.epi_load.empty_barrier_[s]));
        }
      }
      __syncthreads();
#else
      auto* produced_ring = reinterpret_cast<DsmASlot*>(
          role_storage.arena + kProducedRingOffset);
#endif
      if constexpr (!Production) if (ring_snapshots) {
        constexpr std::uint32_t kLocalSlots =
            kDualRoleFf1MathWg ? 2U : 1U;
        for (std::uint32_t local_slot = 0; local_slot < kLocalSlots;
             ++local_slot) {
          auto* source = reinterpret_cast<std::uint8_t const*>(
              &produced_ring[kDualRoleFf1MathWg ? local_slot : rank]);
          const std::uint32_t slice = first_slice +
              (kDualRoleFf1MathWg ? rank * 2U + local_slot : rank);
          auto* destination = ring_snapshots +
              slice * sizeof(DsmASlot);
          for (std::uint32_t offset = threadIdx.x;
               offset < sizeof(DsmASlot); offset += blockDim.x) {
            destination[offset] = source[offset];
          }
        }
      }
      __syncthreads();
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      if (threadIdx.x == 0U) {
#if STREAM1_ROLE_PRECOMPUTE_FF1
        // FF1 writes only its producer-owned arena. The previous pair's
        // free acknowledgement below protects source reuse; armed protects
        // the remote destination, so it is needed only before DSM submission.
        if (!kDualRoleFf1MathWg) {
          cutlass::arch::ClusterBarrier::wait(
              &role_storage.armed, wave & 1U);
        }
#endif
        constexpr std::uint32_t values_bytes =
            Contract::kValuesPerSlot;
        constexpr std::uint32_t scales_bytes =
            Contract::kScalesPerSlot;
        constexpr std::uint32_t kLocalSlots =
            kDualRoleFf1MathWg ? 2U : 1U;
        for (std::uint32_t local_slot = 0; local_slot < kLocalSlots;
             ++local_slot) {
          const std::uint32_t handoff_phase =
              wave * kLocalSlots + local_slot;
          if (kDualRoleFf1MathWg) {
            cutlass::arch::ClusterBarrier::wait(
                &role_storage.armed, handoff_phase & 1U);
          }
          const std::uint32_t slice = first_slice +
              (kDualRoleFf1MathWg ? rank * 2U + local_slot : rank);
          const std::uint32_t handoff_ordinal =
              wave * kSlicesPerFf1Wave + local_slot * 2U + rank;
          const std::uint32_t stage = handoff_ordinal %
              Ff2Mainloop::DispatchPolicy::Stages;
          DsmASlot& produced =
              produced_ring[kDualRoleFf1MathWg ? local_slot : rank];
          for (std::uint32_t consumer_rank =
                   RoleSeparatedCtaPlan::kProducerCtas;
               consumer_rank < RoleSeparatedCtaPlan::kClusterCtas;
               ++consumer_rank) {
            auto& local_view = *reinterpret_cast<Ff2Storage*>(
                role_storage.arena);
            auto* local_a = reinterpret_cast<std::uint8_t*>(
                &local_view.tensors.smem_A) +
                std::size_t(stage) * Contract::kValuesPerSlot;
            auto* local_sfa = reinterpret_cast<std::uint8_t*>(
                &local_view.tensors.smem_SFA) +
                std::size_t(stage) * Contract::kScalesPerSlot;
            auto* target_a = cluster.map_shared_rank(local_a, consumer_rank);
            auto* target_sfa = cluster.map_shared_rank(local_sfa, consumer_rank);
            auto* target_receive = cluster.map_shared_rank(
                &role_storage.receive[stage], consumer_rank);
            cuda::ptx::cp_async_bulk(
                cuda::ptx::space_cluster, cuda::ptx::space_shared,
                target_a, produced.values, values_bytes,
                target_receive);
            cuda::ptx::cp_async_bulk(
                cuda::ptx::space_cluster, cuda::ptx::space_shared,
                target_sfa, produced.scales, scales_bytes,
                target_receive);
          }
          // The consumer releases this producer only after both physical
          // FF2 CTAs consumed the corresponding local slot.  In the dual-WG
          // path this per-slot phase prevents a fourth in-flight slice from
          // cycling onto the three-stage FF2 pipeline before stage 0 drains.
          cutlass::arch::ClusterBarrier::wait(
              &role_storage.free, handoff_phase & 1U);
        }
        // Never wait on a remote mbarrier.  Each consumer waits on its own
        // full barrier (B/SFB TMA + A/SFA DSM) and acknowledges source-read
        // completion through this producer-owned phase barrier.
        if (!kDualRoleFf1MathWg) {
          cutlass::arch::ClusterBarrier::wait(
              &role_storage.free, wave & 1U);
        }
      }
      __syncthreads();
    }
  } else {
    // Producer CTAs deliberately never construct the CUTLASS FF2 pipeline.
    // Its constructor initializes/queries pipeline role state and is only
    // valid for the two physical consumer CTAs which own this storage.
    Ff2Mainloop ff2_mainloop;
    PipelineState write_state =
        cutlass::make_producer_start_state<Pipeline>();
    PipelineState read_state;
    auto& ff2_storage = *reinterpret_cast<Ff2Storage*>(role_storage.arena);
    typename Pipeline::Params pipeline_params;
    pipeline_params.transaction_bytes =
        BulkDsmFf2Collective::kTotalTransactionBytes;
    pipeline_params.num_consumers = 128U;
    pipeline_params.is_leader = wg_thread == 0U;
    pipeline_params.role = warp_group == 0U
        ? Pipeline::ThreadCategory::Producer
        : Pipeline::ThreadCategory::Consumer;
    Pipeline pipeline(
        ff2_storage.pipeline, pipeline_params, CollectiveClusterShape{});
    const std::uint32_t n_tile =
        RoleSeparatedCtaPlan::consumer_n_tile(rank);
    const auto problem_shape = make_shape(
        int(get<0>(ff1_params.problem_shape)), int(kModel), int(kHidden), 1);
    const auto load_inputs = ff2_mainloop.load_init(
        problem_shape, ff2_params);
    if (warp_group == 0U && wg_thread == 0U) {
      // Arm both H128 producers as one logical H256 pair.  The old
      // slice-at-a-time loop waited for producer 0's DSM transaction before
      // producer 1 was even released, accidentally serializing the two FF1
      // CTAs.  Prepare both local receive barriers and B/SFB transactions
      // first, release both producers, then retire their independent DSM
      // arrivals in slice order for the stock FF2 consumer pipeline.
      for (std::uint32_t wave = 0;
           wave < static_cast<std::uint32_t>(active_waves); ++wave) {
        constexpr std::uint32_t kLocalSlots =
            kDualRoleFf1MathWg ? 2U : 1U;
        for (std::uint32_t local_slot = 0; local_slot < kLocalSlots;
             ++local_slot) {
          PipelineState subwave_states[2] = {write_state, write_state};
          for (std::uint32_t producer_rank = 0; producer_rank < 2U;
               ++producer_rank) {
            pipeline.producer_acquire(write_state);
            subwave_states[producer_rank] = write_state;
            const std::uint32_t stage = write_state.index();
            const std::uint32_t slice = wave * kSlicesPerFf1Wave +
                (kDualRoleFf1MathWg
                     ? producer_rank * 2U + local_slot
                     : producer_rank);
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                &role_storage.receive[stage],
                BulkDsmFf2Collective::kDsmTransactionBytes);
            issue_ff2_b_sfb_tma(
                ff2_params, pipeline, write_state, load_inputs,
                n_tile, slice, ff2_storage);
            ++write_state;
          }
          for (std::uint32_t producer_rank = 0; producer_rank < 2U;
               ++producer_rank) {
            cutlass::arch::ClusterBarrier::arrive(
                &role_storage.armed, producer_rank, 1U);
          }
          for (std::uint32_t producer_rank = 0; producer_rank < 2U;
               ++producer_rank) {
            const std::uint32_t slice = wave * kSlicesPerFf1Wave +
                (kDualRoleFf1MathWg
                     ? producer_rank * 2U + local_slot
                     : producer_rank);
            const std::uint32_t handoff_ordinal =
                wave * kSlicesPerFf1Wave + local_slot * 2U + producer_rank;
            const PipelineState state = subwave_states[producer_rank];
            const std::uint32_t stage = state.index();
            cutlass::arch::ClusterTransactionBarrier::wait(
                &role_storage.receive[stage],
                (handoff_ordinal / Ff2Mainloop::DispatchPolicy::Stages) & 1U);
            auto* local_full = pipeline.producer_get_barrier(state);
            cutlass::arch::ClusterTransactionBarrier::complete_transaction(
                local_full, rank,
                BulkDsmFf2Collective::kDsmTransactionBytes, 1U);
            cutlass::arch::ClusterBarrier::arrive(
                &role_storage.free, producer_rank, 1U);
          }
        }
      }
      if (Production || stop_stage != 14) pipeline.producer_tail(write_state);
    } else if (warp_group == 1U && (Production || stop_stage != 14)) {
      typename Ff2Mainloop::TiledMma tiled_mma;
      auto accum = partition_fragment_C(
          tiled_mma, take<0, 2>(TileShape{}));
      ff2_mainloop.mma(
          pipeline, read_state, accum, active_slices,
          static_cast<int>(wg_thread), ff2_storage.tensors,
          ff2_params, make_coord(m_tile, n_tile, 0, 0));
      ff2_mainloop.mma_tail(pipeline, read_state, active_slices);
      auto coordinates = make_identity_tensor(
          make_shape(Int<kM>{}, Int<128>{}));
      auto thread_coordinates = tiled_mma.get_slice(
          static_cast<int>(wg_thread)).partition_C(coordinates);
      float checksum = 0.0F;
      for (int i = 0; i < size(accum); ++i) {
        const auto mn = thread_coordinates(i);
        const int row = static_cast<int>(get<0>(mn));
        const int column = static_cast<int>(get<1>(mn));
        if (output_ff2) {
          output_ff2[row * kModel + n_tile * 128U + column] = accum(i);
        }
        checksum += fabsf(accum(i));
      }
      if constexpr (!Production) {
        if (rank_checksums) atomicAdd(rank_checksums + n_tile, checksum);
      }
    }
  }
  __syncthreads();
  cluster.sync();
}

#if !STREAM1_ROLE_SEPARATED_4CTA
__global__ __launch_bounds__(kKernelThreads, 1) __cluster_dims__(2, 1, 1)
void ff1_ff2_fused_serial_pairs_kernel(
    CUTLASS_GRID_CONSTANT typename Ff1TestKernel::Params const ff1_params,
    CUTLASS_GRID_CONSTANT typename Ff2Mainloop::Params const ff2_params,
    ElementBias const* ff1_bias,
    ElementBias const* ff2_bias,
    ElementBias const* residual_input,
    ElementBias const* layernorm_gamma,
    ElementBias const* layernorm_beta,
    std::uint8_t* ring_snapshots,
    float* raw_ff1,
    float* output_ff2,
    ElementBias* residual_output,
    std::uint8_t* next_values,
    ElementScale* next_scales,
    NextActivationScaleLayout next_scale_layout,
    float* rank_checksums,
    std::uint32_t* address_info,
    int stop_stage,
    bool parallel_move,
    float layernorm_epsilon) {
  namespace cg = cooperative_groups;
  extern __shared__ __align__(1024) unsigned char shared_bytes[];
  auto& ff1_storage = *reinterpret_cast<Ff1Storage*>(shared_bytes);
  auto cluster = cg::this_cluster();
  const std::uint32_t rank = cluster.block_rank();
  // Each physical cluster owns a distinct M128 tile. Validation buffers have
  // the same disjoint ownership; no checksum/address write races across tiles.
  const std::uint32_t m_tile = blockIdx.x / 2U;
  if (raw_ff1) raw_ff1 += std::size_t(m_tile) * kM * kHidden;
  if (output_ff2) output_ff2 += std::size_t(m_tile) * kM * kModel;
  if (ring_snapshots) ring_snapshots += std::size_t(m_tile) * kSlices * sizeof(DsmASlot);
  if (rank_checksums) rank_checksums += m_tile * 2U;
  if (address_info) address_info += m_tile * 4U;
  typename Ff2Mainloop::TiledMma total_mma;
#if !STREAM1_FUSED_SHARED_CARRY
  auto total = partition_fragment_C(total_mma, take<0, 2>(TileShape{}));
  clear(total);
#else
  // Keep only half of the FF2 sum alive in registers through FF1. The other
  // half has a disjoint 32-KiB lifetime beside FF1's compact two-stage arena.
  auto carry_registers=make_tensor<float>(make_layout(Int<kCarryRegisterElements>{}));
  clear(carry_registers);
#endif

  // Correctness reference: two packed slots are reused four times for H1024.
  // FF1 and FF2 alias the arena, so every pair is fully drained before reuse.
  // The later overlapped pipeline must be checked against this reference.
  for (int first_slice = 0; first_slice < kSlices; first_slice += 2) {

  // Bounded Molab diagnostics. 10 proves that the physical cluster itself is
  // valid; 11/12 isolate one logical-singleton FF1 CTA at a time. These modes
  // return before the DSM handoff and never participate in production runs.
  if (stop_stage == 10) {
    cluster.sync();
    return;
  }

  if (stop_stage == -3 && rank == 0U) {
    invoke_ff1_for_diagnostic(
        ff1_params, reinterpret_cast<char*>(shared_bytes+kFf1BaseOffset), ff1_bias, raw_ff1, first_slice);
  }
  if (stop_stage == -3) cluster.sync();
  if ((stop_stage != -2 && stop_stage != -3 &&
       stop_stage != 11 && stop_stage != 12) ||
      (stop_stage == -2 && rank == 1U) ||
      (stop_stage == -3 && rank == 1U) ||
      (stop_stage == 11 && rank == 0U) ||
      (stop_stage == 12 && rank == 1U)) {
    invoke_ff1_for_diagnostic(
        ff1_params, reinterpret_cast<char*>(shared_bytes+kFf1BaseOffset), ff1_bias, raw_ff1, first_slice);
  }
#if defined(STREAM1_CUTLASS_DIAG_OUTER_RETURN_AFTER_FF1)
  // Pair with an early-return gate inside the nested CUTLASS kernel.  Returning
  // the complete physical-cluster kernel here prevents the surrounding DSM
  // handoff synchronization from obscuring whether the CUTLASS gate itself
  // was reached.  The define is uniform for both CTAs and all threads.
  return;
#else
  __syncthreads();
  cluster.sync();

  if (stop_stage == 11 || stop_stage == 12) return;
#if defined(STREAM1_CUTLASS_DIAG_DIRECT_FF1_MANUAL_MATERIALIZE)
  constexpr std::size_t kManualTileBytes =
      std::size_t{kM} * Contract::kSliceColumns * sizeof(float);
  auto* produced_ring = reinterpret_cast<DsmASlot*>(
      shared_bytes + kProducedRingOffset);
  auto* produced_bytes = reinterpret_cast<std::uint8_t*>(
      &produced_ring[rank]);
  if (ring_snapshots) {
    for (std::uint32_t offset = threadIdx.x;
         offset < sizeof(DsmASlot); offset += blockDim.x) {
      ring_snapshots[(first_slice + rank) * sizeof(DsmASlot) + offset] = produced_bytes[offset];
    }
  }
  __syncthreads();
#else
  auto* produced_ring = ff1_shared_only_hidden_ring(ff1_storage);
  if (ring_snapshots) {
    auto* bytes = reinterpret_cast<std::uint8_t*>(&produced_ring[rank]);
    for (unsigned offset=threadIdx.x; offset<sizeof(DsmASlot); offset+=blockDim.x)
      ring_snapshots[(first_slice+rank)*sizeof(DsmASlot)+offset]=bytes[offset];
  }
  __syncthreads();
#if STREAM1_FUSED_NATIVE_EPILOGUE
  // All producers and consumers have drained at the outer rendezvous.
  // End barrier lifetimes before FF2 reuses this shared arena.
  if (threadIdx.x==0) {
    for (int s=0;s<Ff1TestMainloop::DispatchPolicy::Stages;++s) {
      cutlass::arch::ClusterBarrier::invalidate(reinterpret_cast<std::uint64_t const*>(&ff1_storage.pipelines.mainloop.full_barrier_[s]));
      cutlass::arch::ClusterBarrier::invalidate(reinterpret_cast<std::uint64_t const*>(&ff1_storage.pipelines.mainloop.empty_barrier_[s]));
    }
    for (int s=0;s<Ff1SharedOnlyEpilogue::DispatchPolicy::StagesC;++s) {
      cutlass::arch::ClusterBarrier::invalidate(reinterpret_cast<std::uint64_t const*>(&ff1_storage.pipelines.epi_load.full_barrier_[s]));
      cutlass::arch::ClusterBarrier::invalidate(reinterpret_cast<std::uint64_t const*>(&ff1_storage.pipelines.epi_load.empty_barrier_[s]));
    }
  }
  __syncthreads();
#endif
#endif
  if (stop_stage == 1) {
    cluster.sync();
    continue;
  }
#if STREAM1_FUSED_SHARED_CARRY
  // The full fragment begins after FF1 is dead. Reload its second half before
  // FF2's TMA arena overwrites shared [0,32KiB); the first half stayed in regs.
  auto total = partition_fragment_C(total_mma,take<0,2>(TileShape{}));
  static_assert(size(total)==kCarryRegisterElements+kCarrySharedElements);
  if (threadIdx.x/128U==1U && first_slice>0) {
    const float* saved=reinterpret_cast<const float*>(shared_bytes);
    CUTLASS_PRAGMA_UNROLL
    for (int i=0;i<kCarryRegisterElements;++i) total(i)=carry_registers(i);
    CUTLASS_PRAGMA_UNROLL
    for (int i=0;i<kCarrySharedElements;++i)
      total(i+kCarryRegisterElements)=saved[(threadIdx.x%128U)*kCarrySharedElements+i];
  }
  __syncthreads();
#endif
  auto& ff2_storage = *reinterpret_cast<Ff2Storage*>(shared_bytes);

  auto* destination = reinterpret_cast<std::uint8_t*>(
        &ff2_storage.hidden[rank]);
  auto* source = reinterpret_cast<const std::uint8_t*>(
        &produced_ring[rank]);
  if (threadIdx.x == 0U && address_info) {
    address_info[rank * 2U] = static_cast<std::uint32_t>(
        source - reinterpret_cast<std::uint8_t*>(shared_bytes));
    address_info[rank * 2U + 1U] = static_cast<std::uint32_t>(
        destination - reinterpret_cast<std::uint8_t*>(shared_bytes));
  }
  if constexpr (!kNoRelocation) {
  if (parallel_move) {
    // In this reference the destination ends exactly before the source.
    // Prove non-overlap before replacing memmove with a collective copy.
    constexpr std::size_t source_offset = kM * Contract::kSliceColumns * sizeof(float);
    static_assert(source_offset >= offsetof(Ff2Storage, hidden) + sizeof(DsmASlot));
    static_assert(sizeof(DsmASlot) % sizeof(uint4) == 0);
    for (unsigned i = threadIdx.x; i < sizeof(DsmASlot)/sizeof(uint4); i += blockDim.x)
      reinterpret_cast<uint4*>(destination)[i] = reinterpret_cast<uint4 const*>(source)[i];
  } else if (threadIdx.x == 0U) {
    move_bytes(destination, source, sizeof(DsmASlot));
  }
  }
  __syncthreads();
#if STREAM1_FUSED_PROXY_FENCE
  // Generic shared writes feed cp.async.bulk's async proxy. CTA/cluster
  // rendezvous is not a substitute for the cross-proxy publication fence.
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
  cluster.sync();
  if (stop_stage == 2) return;
#if STREAM1_FUSED_HANDOFF_TRACE
  if (ring_snapshots) trace_bytes("post_copy",rank,first_slice+rank,
      destination,ring_snapshots+(first_slice+rank)*sizeof(DsmASlot),sizeof(DsmASlot));
  __syncthreads();
#endif

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
#if STREAM1_FUSED_HANDOFF_TRACE
  if (ring_snapshots) trace_bytes("post_pipeline_init",rank,first_slice+rank,
      destination,ring_snapshots+(first_slice+rank)*sizeof(DsmASlot),sizeof(DsmASlot));
  __syncthreads();
#endif

  const auto problem_shape = make_shape(int(get<0>(ff1_params.problem_shape)), int(kModel), int(kHidden), 1);
  Ff2Mainloop mainloop;
  const auto load_inputs = mainloop.load_init(problem_shape, ff2_params);
  if (warp_group == 0U && wg_thread == 0U) {
    PipelineState write_state = cutlass::make_producer_start_state<Pipeline>();
    for (std::uint32_t slice = 0; slice < 2U; ++slice) {
      pipeline.producer_acquire(write_state);
      issue_ff2_b_sfb_tma(
          ff2_params, pipeline, write_state, load_inputs,
          rank, first_slice + slice, ff2_storage);
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
    PipelineState read_state;
#if STREAM1_FUSED_ACCUMULATE_DIRECT
      // Optional audited CUTLASS seam: clear at the first pair only. All old
      // eight-argument callers keep the upstream clear-on-call behavior.
      mainloop.mma(
          pipeline,read_state,total,2,static_cast<int>(wg_thread),
          ff2_storage.tensors,ff2_params,make_coord(m_tile,rank,0,0),first_slice==0);
      mainloop.mma_tail(pipeline,read_state,2);
#else
    typename Ff2Mainloop::TiledMma tiled_mma;
    auto accum = partition_fragment_C(
        tiled_mma, take<0, 2>(TileShape{}));
    mainloop.mma(
        pipeline, read_state, accum, 2,
        static_cast<int>(wg_thread), ff2_storage.tensors,
        ff2_params, make_coord(m_tile, rank, 0, 0));
    mainloop.mma_tail(pipeline, read_state, 2);
    for (int i = 0; i < size(accum); ++i) {
      total(i) += accum(i);
    }
#endif
  }
  // Includes producer_tail, outstanding DSM reads of both packed slots, and
  // consumer completion before the next FF1 lifetime overwrites this arena.
  __syncthreads();
  cluster.sync();
#if STREAM1_FUSED_HANDOFF_TRACE
  if (ring_snapshots) for (unsigned slice=0;slice<2;++slice) {
    auto* expected=ring_snapshots+(first_slice+slice)*sizeof(DsmASlot);
    trace_bytes("mma_A",rank,first_slice+slice,
        reinterpret_cast<unsigned char const*>(&ff2_storage.tensors.smem_A)+slice*Contract::kValuesPerSlot,
        expected,Contract::kValuesPerSlot);
    trace_bytes("mma_SFA",rank,first_slice+slice,
        reinterpret_cast<unsigned char const*>(&ff2_storage.tensors.smem_SFA)+slice*Contract::kScalesPerSlot,
        expected+Contract::kValuesPerSlot,Contract::kScalesPerSlot);
  }
  __syncthreads();
#endif
#if STREAM1_FUSED_SHARED_CARRY
  if (threadIdx.x/128U==1U) {
    auto coordinates=make_identity_tensor(make_shape(Int<kM>{},Int<128>{}));
    auto tc=total_mma.get_slice(static_cast<int>(threadIdx.x%128U)).partition_C(coordinates);
    float* saved=reinterpret_cast<float*>(shared_bytes);
    float checksum=0;
    CUTLASS_PRAGMA_UNROLL
    for (int i=0;i<size(total);++i) {
      const int row=get<0>(tc(i)), col=get<1>(tc(i));
      if (first_slice+2<kSlices) {
        if (i<kCarryRegisterElements) carry_registers(i)=total(i);
        else saved[(threadIdx.x%128U)*kCarrySharedElements+i-kCarryRegisterElements]=total(i);
      }
      else {
        const std::uint32_t global_column = rank * 128U + col;
        const std::size_t global_index =
            (static_cast<std::size_t>(m_tile) * kM + row) * kModel +
            global_column;
        if (output_ff2) output_ff2[row*kModel+global_column]=total(i);
        if (ff2_bias && residual_input) {
          auto* local_unrounded = reinterpret_cast<float*>(shared_bytes);
          local_unrounded[row * 128U + col] =
              total(i) + static_cast<float>(ff2_bias[global_column]) +
              static_cast<float>(residual_input[global_index]);
        }
        checksum+=fabsf(total(i));
      }
    }
    if (first_slice+2==kSlices && rank_checksums) atomicAdd(rank_checksums+rank,checksum);
  }
  __syncthreads();
  cluster.sync();
  if (first_slice + 2 == kSlices && stop_stage == 0 && ff2_bias &&
      residual_input && layernorm_gamma && layernorm_beta &&
      residual_output && next_values && next_scales) {
    stream1_sm120_nvfp4_finish_ff2_tile(
        cluster, rank, m_tile, reinterpret_cast<float*>(shared_bytes),
        ff2_bias, residual_input, layernorm_gamma, layernorm_beta,
        residual_output, next_values, next_scales, next_scale_layout,
        layernorm_epsilon);
  }
#endif
#if STREAM1_FUSED_INVALIDATE_BARRIERS
  if (threadIdx.x==0) {
    for (int stage=0;stage<Ff2Mainloop::DispatchPolicy::Stages;++stage) {
      cutlass::arch::ClusterBarrier::invalidate(reinterpret_cast<std::uint64_t const*>(
          &ff2_storage.pipeline.full_barrier_[stage]));
      cutlass::arch::ClusterBarrier::invalidate(reinterpret_cast<std::uint64_t const*>(
          &ff2_storage.pipeline.empty_barrier_[stage]));
    }
    cutlass::arch::ClusterBarrier::invalidate(&ff2_storage.handoff_armed);
  }
  __syncthreads();
  cluster.sync();
#endif
#endif
  }
#if !STREAM1_FUSED_SHARED_CARRY
  if (stop_stage == 0 && threadIdx.x / 128U == 1U) {
    auto thread_mma = total_mma.get_slice(static_cast<int>(threadIdx.x % 128U));
    auto coordinates = make_identity_tensor(make_shape(Int<kM>{}, Int<128>{}));
    auto thread_coordinates = thread_mma.partition_C(coordinates);
    float local = 0.0F;
    for (int i = 0; i < size(total); ++i) {
      const auto mn = thread_coordinates(i);
      if (output_ff2) output_ff2[static_cast<int>(get<0>(mn)) * kModel +
          rank * 128U + static_cast<int>(get<1>(mn))] = total(i);
      if (ff2_bias && residual_input) {
        const std::uint32_t row = static_cast<std::uint32_t>(get<0>(mn));
        const std::uint32_t column =
            rank * 128U + static_cast<std::uint32_t>(get<1>(mn));
        const std::size_t global_index =
            (static_cast<std::size_t>(m_tile) * kM + row) * kModel + column;
        auto* local_unrounded = reinterpret_cast<float*>(shared_bytes);
        local_unrounded[row * 128U + static_cast<std::uint32_t>(get<1>(mn))] =
            total(i) + static_cast<float>(ff2_bias[column]) +
            static_cast<float>(residual_input[global_index]);
      }
      local += total(i) < 0.0F ? -total(i) : total(i);
    }
    if (rank_checksums) atomicAdd(rank_checksums + rank, local);
  }
  __syncthreads();
  cluster.sync();
  if (stop_stage == 0 && ff2_bias && residual_input && layernorm_gamma &&
      layernorm_beta && residual_output && next_values && next_scales) {
    stream1_sm120_nvfp4_finish_ff2_tile(
        cluster, rank, m_tile, reinterpret_cast<float*>(shared_bytes),
        ff2_bias, residual_input, layernorm_gamma, layernorm_beta,
        residual_output, next_values, next_scales, next_scale_layout,
        layernorm_epsilon);
  }
#endif
}
#endif  // !STREAM1_ROLE_SEPARATED_4CTA

void require_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
    std::exit(2);
  }
}

std::uint64_t fnv1a(std::vector<std::uint8_t> const& bytes) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (auto byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
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
  const std::uint32_t seed = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 1U;
  const int repetitions = argc > 3 ? std::atoi(argv[3]) : 1;
  const int required_hidden = argc > 4 ? std::atoi(argv[4]) : kHidden;
  const bool parallel_move = argc > 5 ? std::atoi(argv[5]) != 0 : false;
  const int benchmark_iterations = argc > 6 ? std::atoi(argv[6]) : 0;
  const bool require_register_pack = argc > 7 && std::atoi(argv[7]) != 0;
  const int m_tiles = argc > 8 ? std::atoi(argv[8]) : 1;
  const int timing_mode = argc > 9 ? std::atoi(argv[9]) : 0;
  const int oracle_tiles = argc > 10 ? std::atoi(argv[10]) : m_tiles;
  if (argc > 11 || m_tiles < 1 || m_tiles > 512 || timing_mode < 0 || timing_mode > 1 ||
      oracle_tiles < 1 || oracle_tiles > m_tiles) {
    std::cerr << "measurement_contract=FAIL tiles must be 1..512, mode 0=eager/1=graph, "
                 "and oracle_tiles must be 1..tiles\n";
    return 8;
  }
  const int rows = kM * m_tiles;
  if (require_register_pack && !kDirectRegisterPack && !STREAM1_FUSED_NATIVE_EPILOGUE) {
    std::cerr << "handoff_contract=FAIL direct_register_pack_required; "
                 "this binary materializes an FP32 shared tile\n";
    return 8;
  }
  if (kDirectRegisterPack && !sm120_ff1_register_pack::layout_contract()) {
    std::cerr << "register_layout_contract=FAIL\n";
    return 6;
  }
  if (kDirectRegisterPack) std::cout << "register_layout_contract=pass; fp32_shared_intermediate_bytes=0\n";
  if (benchmark_iterations < 0 || benchmark_iterations > 1000 ||
      (benchmark_iterations && stop_stage != 0 && stop_stage != 11 &&
       stop_stage != 14 && stop_stage != 15)) return 8;
  if (required_hidden != kHidden) {
    std::cerr << "workload_contract=FAIL required_hidden=" << required_hidden
              << " supported_hidden=" << kHidden << '\n';
    return 8;
  }
  if (kSlices > 2 && stop_stage != 0 && stop_stage != 1 &&
      !(kRoleSeparated4Cta && (stop_stage == 10 || stop_stage == 11 ||
                              stop_stage == 14 || stop_stage == 15))) {
    std::cerr << "multi-pair reference supports full or FF1-only validation\n";
    return 8;
  }
  if (repetitions < 1 || repetitions > 100) {
    std::cerr << "repetitions must be in [1, 100]\n";
    return 5;
  }
  constexpr int ff1_mma_threads = cute::thr_size(typename Ff1TestMainloop::TiledMma{});
  constexpr int ff2_mma_threads = cute::thr_size(typename Ff2Mainloop::TiledMma{});
  std::cout << "ff1_mma_threads=" << ff1_mma_threads
            << " ff2_mma_threads=" << ff2_mma_threads << '\n';
  if (ff1_mma_threads != kFf1ConsumerThreads || (stop_stage == 0 && ff2_mma_threads != 128)) {
    std::cerr << "launch_contract=FAIL: harness consumer ownership does not match the selected CUTLASS MMA\n";
    return 6;
  }
#if !defined(STREAM1_CUTLASS_DIAG_DIRECT_FF1_MANUAL_MATERIALIZE) && !STREAM1_FUSED_NATIVE_EPILOGUE
  if (stop_stage == 0 || stop_stage == 1 || stop_stage == 2) {
    std::cerr << "numeric oracle requires the instrumented direct FF1 path\n";
    return 5;
  }
#endif
  const auto oracle_rows = sm120_fused_reference::select_oracle_rows(rows, oracle_tiles);
  std::cout << "reference_start rows=" << rows << " input_pattern="
            << (seed ? "deterministic_independent" : "uniform_hand_fixture") << '\n' << std::flush;
  auto fixture = sm120_fused_reference::make_fixture(seed, rows);
#if STREAM1_FUSED_NATIVE_EPILOGUE
  // First integration gate preserves the measured FF1's zero per-row bias
  // contract. Nonzero per-output-column model bias is a separate open gate.
  if (parallel_move || kSharedCarry) {
    std::cerr << "native gate forbids unproved parallel relocation or shared carry\n";
    return 8;
  }
  std::fill(fixture.bias.begin(), fixture.bias.end(), 0.0F);
  std::cout << "native_ff1_epilogue=1 bias_contract=zero_only\n";
#endif
  sm120_fused_reference::compute_rows(fixture, oracle_rows);
  std::cout << "oracle_scope=" << (oracle_tiles == m_tiles ? "full" : "sampled")
            << " oracle_tiles=" << oracle_tiles << " total_tiles=" << m_tiles
            << " oracle_rows=" << oracle_rows.size() << '\n' << std::flush;
  if (seed == 0) {
    const std::vector<float> expected_ff1(rows * kHidden, 96.0F);
    const std::vector<float> expected_hidden(rows * kHidden, 96.0F);
    const std::vector<float> expected_ff2(
        rows * kModel, 96.0F * kHidden * 0.25F);
    const bool hand_fixture_ok =
        sm120_fused_reference::compare_rows("hand_ff1", fixture.ff1, expected_ff1, kHidden, oracle_rows) &&
        sm120_fused_reference::compare_rows("hand_hidden", fixture.hidden, expected_hidden, kHidden, oracle_rows) &&
        sm120_fused_reference::compare_rows("hand_ff2", fixture.ff2, expected_ff2, kModel, oracle_rows);
    if (!hand_fixture_ok) return 7;
  }
  const auto ff1_shape = make_shape(rows, kHidden, kModel, 1);
  const auto ff2_shape = make_shape(rows, kModel, kHidden, 1);
  using Ff1ScaleConfig = typename Ff1TestMainloop::Sm1xxBlkScaledConfig;
  using Ff2ScaleConfig = typename Ff2Mainloop::Sm1xxBlkScaledConfig;
  using OutputScaleConfig =
      cutlass::detail::Sm1xxBlockScaledOutputConfig<Contract::kScaleVector>;

  const auto ff1_stride_a = cutlass::make_cute_packed_stride(
      typename Ff1TestKernel::StrideA{}, {rows, kModel, 1});
  const auto ff1_stride_b = cutlass::make_cute_packed_stride(
      typename Ff1TestKernel::StrideB{}, {kHidden, kModel, 1});
  const auto ff1_stride_c = cutlass::make_cute_packed_stride(
      typename Ff1TestKernel::StrideC{}, {rows, kHidden, 1});
  const auto ff1_stride_d = cutlass::make_cute_packed_stride(
      typename Ff1TestKernel::StrideD{}, {rows, kHidden, 1});
  const auto ff1_sfa_layout =
      Ff1ScaleConfig::tile_atom_to_shape_SFA(ff1_shape);
  const auto ff1_sfb_layout =
      Ff1ScaleConfig::tile_atom_to_shape_SFB(ff1_shape);
  const auto ff1_sfd_layout =
      OutputScaleConfig::tile_atom_to_shape_SFD(ff1_shape);

  auto* ff1_a = allocate_bytes<ElementHidden>(rows * kModel / 2U);
  auto* ff1_b = allocate_bytes<ElementHidden>(kHidden * kModel / 2U);
  auto* ff1_c = allocate_bytes<ElementOutput>(rows * kHidden * sizeof(ElementOutput));
  auto* ff1_d = allocate_bytes<ElementHidden>(rows * kHidden / 2U);
  auto* ff1_sfa = allocate_bytes<ElementScale>(size(filter_zeros(ff1_sfa_layout)));
  auto* ff1_sfb = allocate_bytes<ElementScale>(size(filter_zeros(ff1_sfb_layout)));
  auto* ff1_sfd = allocate_bytes<ElementScale>(size(filter_zeros(ff1_sfd_layout)));
  auto* ff1_bias = allocate_bytes<ElementBias>(std::max(rows,kHidden) * sizeof(ElementBias));
  require_cuda(cudaMemset(ff1_bias,0,std::max(rows,kHidden)*sizeof(ElementBias)), "clear bias");
  auto* norm = allocate_bytes<ElementCompute>(sizeof(ElementCompute));
  require_cuda(cudaMemcpy(ff1_a, fixture.a.packed.data(), fixture.a.packed.size(), cudaMemcpyHostToDevice), "copy ff1 A");
  require_cuda(cudaMemcpy(ff1_b, fixture.b1.packed.data(), fixture.b1.packed.size(), cudaMemcpyHostToDevice), "copy ff1 B");
  require_cuda(cudaMemset(ff1_c, 0, rows * kHidden * sizeof(ElementOutput)), "memset ff1 C");
  require_cuda(cudaMemset(ff1_d, 0, rows * kHidden / 2U), "memset ff1 D");
  std::vector<ElementBias> host_bias;
  for (float value : fixture.bias) host_bias.emplace_back(value);
  require_cuda(cudaMemcpy(ff1_bias, host_bias.data(), kHidden * sizeof(ElementBias), cudaMemcpyHostToDevice), "copy bias");
  auto host_sfa = sm120_fused_reference::encode_scales(fixture.a, ff1_sfa_layout, size(filter_zeros(ff1_sfa_layout)));
  auto host_sfb1 = sm120_fused_reference::encode_scales(fixture.b1, ff1_sfb_layout, size(filter_zeros(ff1_sfb_layout)));
  require_cuda(cudaMemcpy(ff1_sfa, host_sfa.data(), host_sfa.size(), cudaMemcpyHostToDevice), "copy ff1 SFA");
  require_cuda(cudaMemcpy(ff1_sfb, host_sfb1.data(), host_sfb1.size(), cudaMemcpyHostToDevice), "copy ff1 SFB");
  // CUTLASS internally divides by max(E2M1)=6. Unity gives amax/6,
  // matching the scalar oracle. The old manual writer ignored this pointer.
  const ElementCompute norm_value = STREAM1_FUSED_NATIVE_EPILOGUE ? 1.0F : 6.0F;
  require_cuda(cudaMemcpy(norm, &norm_value, sizeof(norm_value), cudaMemcpyHostToDevice), "copy norm");

  typename Ff1TestKernel::Arguments ff1_arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      ff1_shape,
      {ff1_a, ff1_stride_a, ff1_b, ff1_stride_b,
       ff1_sfa, ff1_sfa_layout, ff1_sfb, ff1_sfb_layout},
      {{1.0F, 0.0F}, nullptr, ff1_stride_c, nullptr, ff1_stride_d}};
  ff1_arguments.epilogue.thread.block_scale_factor_ptr = ff1_sfd;
  ff1_arguments.epilogue.thread.norm_constant_ptr = norm;
  ff1_arguments.epilogue.thread.bias_ptr = ff1_bias;
  if (!Ff1TestKernel::can_implement(ff1_arguments)) return 3;
  auto ff1_params = Ff1TestKernel::to_underlying_arguments(
      ff1_arguments, nullptr);

  const typename Ff2Mainloop::StrideA ff2_stride_a =
      make_stride(std::int64_t{kHidden}, _1{}, std::int64_t{rows * kHidden});
  const typename Ff2Mainloop::StrideB ff2_stride_b =
      make_stride(std::int64_t{kHidden}, _1{}, std::int64_t{kModel * kHidden});
  const auto ff2_sfa_layout =
      Ff2ScaleConfig::tile_atom_to_shape_SFA(ff2_shape);
  const auto ff2_sfb_layout =
      Ff2ScaleConfig::tile_atom_to_shape_SFB(ff2_shape);
  auto* ff2_unused_a = allocate_bytes<ElementHidden>(rows * kHidden / 2U);
  auto* ff2_b = allocate_bytes<ElementHidden>(kModel * kHidden / 2U);
  auto* ff2_unused_sfa = allocate_bytes<ElementScale>(size(filter_zeros(ff2_sfa_layout)));
  auto* ff2_sfb = allocate_bytes<ElementScale>(size(filter_zeros(ff2_sfb_layout)));
  require_cuda(cudaMemset(ff2_unused_a, 0, rows * kHidden / 2U), "memset ff2 A");
  require_cuda(cudaMemcpy(ff2_b, fixture.b2.packed.data(), fixture.b2.packed.size(), cudaMemcpyHostToDevice), "copy ff2 B");
  require_cuda(cudaMemset(ff2_unused_sfa, 0, size(filter_zeros(ff2_sfa_layout))), "memset ff2 SFA");
  auto host_sfb2 = sm120_fused_reference::encode_scales(fixture.b2, ff2_sfb_layout, size(filter_zeros(ff2_sfb_layout)));
  require_cuda(cudaMemcpy(ff2_sfb, host_sfb2.data(), host_sfb2.size(), cudaMemcpyHostToDevice), "copy ff2 SFB");
  typename Ff2Mainloop::Arguments ff2_arguments{
      ff2_unused_a, ff2_stride_a, ff2_b, ff2_stride_b,
      ff2_unused_sfa, ff2_sfa_layout, ff2_sfb, ff2_sfb_layout};
  if (!Ff2Mainloop::can_implement(ff2_shape, ff2_arguments)) return 4;
  auto ff2_params = Ff2Mainloop::to_underlying_arguments(
      ff2_shape, ff2_arguments, nullptr);

  const auto next_scale_layout = NextActivationScaleConfig::tile_atom_to_shape_SFA(
      make_shape(rows, kModel, kModel, 1));
  auto to_half = [](std::vector<float> const& values) {
    std::vector<ElementBias> result;
    result.reserve(values.size());
    for (float value : values) result.emplace_back(value);
    return result;
  };
  const auto host_ff2_bias = to_half(fixture.ff2_bias);
  const auto host_residual_input = to_half(fixture.residual);
  const auto host_layernorm_gamma = to_half(fixture.layernorm_gamma);
  const auto host_layernorm_beta = to_half(fixture.layernorm_beta);
  auto* ff2_bias = allocate_bytes<ElementBias>(kModel * sizeof(ElementBias));
  auto* residual_input = allocate_bytes<ElementBias>(rows * kModel * sizeof(ElementBias));
  auto* layernorm_gamma = allocate_bytes<ElementBias>(kModel * sizeof(ElementBias));
  auto* layernorm_beta = allocate_bytes<ElementBias>(kModel * sizeof(ElementBias));
  auto* residual_output = allocate_bytes<ElementBias>(rows * kModel * sizeof(ElementBias));
  auto* next_values = allocate_bytes<std::uint8_t>(rows * kModel / 2U);
  auto* next_scales = allocate_bytes<ElementScale>(size(filter_zeros(next_scale_layout)));
  require_cuda(cudaMemcpy(ff2_bias, host_ff2_bias.data(),
      kModel * sizeof(ElementBias), cudaMemcpyHostToDevice), "copy ff2 bias");
  require_cuda(cudaMemcpy(residual_input, host_residual_input.data(),
      rows * kModel * sizeof(ElementBias), cudaMemcpyHostToDevice), "copy residual input");
  require_cuda(cudaMemcpy(layernorm_gamma, host_layernorm_gamma.data(),
      kModel * sizeof(ElementBias), cudaMemcpyHostToDevice), "copy layernorm gamma");
  require_cuda(cudaMemcpy(layernorm_beta, host_layernorm_beta.data(),
      kModel * sizeof(ElementBias), cudaMemcpyHostToDevice), "copy layernorm beta");

  float* checksums = allocate_bytes<float>(m_tiles * 2U * sizeof(float));
  float* raw_ff1 = allocate_bytes<float>(rows * kHidden * sizeof(float));
  float* output_ff2 = allocate_bytes<float>(rows * kModel * sizeof(float));
  auto* ring_snapshots = allocate_bytes<std::uint8_t>(
      m_tiles * kSlices * sizeof(DsmASlot));
  auto* address_info = allocate_bytes<std::uint32_t>(m_tiles * 4U * sizeof(std::uint32_t));
  constexpr std::size_t serial_base_shared_bytes =
      kSharedCarry ? kSharedCarryArenaBytes :
      sizeof(Ff1Storage) > sizeof(Ff2Storage)
          ? sizeof(Ff1Storage) : sizeof(Ff2Storage);
  constexpr std::size_t serial_shared_bytes = serial_base_shared_bytes >
      Stream1Sm120Nvfp4FusedEpilogueContract::kSharedBytes
          ? serial_base_shared_bytes
          : Stream1Sm120Nvfp4FusedEpilogueContract::kSharedBytes;
  constexpr std::size_t shared_bytes = kRoleSeparated4Cta
      ? sizeof(RoleSeparatedNumericStorage) : serial_shared_bytes;
#if STREAM1_ROLE_SEPARATED_4CTA
  require_cuda(cudaFuncSetAttribute(
      ff1_ff2_role_separated_four_cta_kernel<false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared_bytes)), "set role-separated shared memory");
  require_cuda(cudaFuncSetAttribute(
      ff1_ff2_role_separated_four_cta_kernel<true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared_bytes)), "set production role-separated shared memory");
#else
  require_cuda(cudaFuncSetAttribute(
      ff1_ff2_fused_serial_pairs_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared_bytes)), "set fused shared memory");
#endif
  static_assert(kM == sm120_fused_reference::kRows &&
                kHidden == sm120_fused_reference::kHidden &&
                kModel == sm120_fused_reference::kModel);
  std::cout << "validation_only=1 seed=" << seed
            << " shape=" << rows << 'x' << kHidden << 'x' << kModel
            << " m_tiles=" << m_tiles << " grid_ctas="
            << m_tiles * (kRoleSeparated4Cta ? 4 : 2)
            << " oracle_tiles=" << oracle_tiles
            << " slices=" << kSlices << " physical_slots=2 pair_overlap=0 repetitions=" << repetitions
            << " threads=" << kKernelThreads << " parallel_move=" << parallel_move
            << " direct_register_pack=" << kDirectRegisterPack
            << " no_relocation=" << kNoRelocation << " direct_accumulation=" << kAccumulateDirect
            << " shared_carry=" << kSharedCarry << " ff1_active_bytes=" << kFf1ActiveBytes
            << " carry_smem_bytes=" << (kSharedCarry?kCarryBytes:0)
            << " carry_reg_elements=" << (kSharedCarry?kCarryRegisterElements:128)
            << " proxy_fence=" << STREAM1_FUSED_PROXY_FENCE
            << " invalidate_barriers=" << STREAM1_FUSED_INVALIDATE_BARRIERS
            << " role_separated_4cta=" << kRoleSeparated4Cta
            << " stop_stage=" << stop_stage << '\n';
  bool all_pass = true;
  for (int repetition = 0; repetition < repetitions; ++repetition) {
    require_cuda(cudaMemset(checksums, 0, m_tiles * 2U * sizeof(float)), "clear checksums");
    require_cuda(cudaMemset(raw_ff1, 0xff, rows * kHidden * sizeof(float)), "poison raw FF1");
    require_cuda(cudaMemset(output_ff2, 0xff, rows * kModel * sizeof(float)), "poison output FF2");
    require_cuda(cudaMemset(residual_output, 0xff,
        rows * kModel * sizeof(ElementBias)), "poison residual output");
    require_cuda(cudaMemset(next_values, 0xa5,
        rows * kModel / 2U), "poison next values");
    require_cuda(cudaMemset(next_scales, 0xa5,
        size(filter_zeros(next_scale_layout))), "poison next scales");
    require_cuda(cudaMemset(address_info, 0xff, m_tiles * 4U * sizeof(std::uint32_t)), "poison offsets");
    require_cuda(cudaMemset(ring_snapshots, 0xa5, m_tiles * kSlices * sizeof(DsmASlot)), "poison snapshots");
#if STREAM1_ROLE_SEPARATED_4CTA
    ff1_ff2_role_separated_four_cta_kernel<false><<<
        m_tiles * RoleSeparatedCtaPlan::kClusterCtas,
        kKernelThreads, shared_bytes>>>(
        ff1_params, ff2_params, ff1_bias, ring_snapshots,
        raw_ff1, output_ff2, checksums, stop_stage);
#else
    ff1_ff2_fused_serial_pairs_kernel<<<m_tiles*2, kKernelThreads, shared_bytes>>>(
        ff1_params, ff2_params, ff1_bias, ff2_bias, residual_input,
        layernorm_gamma, layernorm_beta, ring_snapshots,
        raw_ff1, output_ff2, residual_output, next_values, next_scales,
        next_scale_layout, checksums, address_info, stop_stage, parallel_move,
        1.0e-5F);
#endif
    require_cuda(cudaGetLastError(), "fused launch");
    require_cuda(cudaDeviceSynchronize(), "fused sync");
    if (stop_stage == 11) {
      std::vector<float> host_ff1(rows * kHidden);
      require_cuda(cudaMemcpy(host_ff1.data(), raw_ff1,
          host_ff1.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "copy diagnostic raw FF1");
      constexpr int kDiagnosticSlices = kDualRoleFf1MathWg ? 4 : 2;
      constexpr int kDiagnosticColumns =
          kDiagnosticSlices * Contract::kSliceColumns;
      std::size_t errors = 0;
      double max_abs = 0.0;
      for (int row : oracle_rows) {
        for (int column = 0; column < kDiagnosticColumns; ++column) {
          const auto index = std::size_t(row) * kHidden + column;
          const double delta = std::abs(
              double(host_ff1.at(index)) - fixture.ff1.at(index));
          const double tolerance =
              1e-4 + 1e-5 * std::abs(double(fixture.ff1.at(index)));
          if (!std::isfinite(host_ff1.at(index)) || delta > tolerance) {
            if (errors < 4) {
              std::cerr << "diagnostic_raw_ff1 mismatch index=" << index
                        << " actual=" << host_ff1.at(index)
                        << " expected=" << fixture.ff1.at(index) << '\n';
            }
            ++errors;
          }
          if (std::isfinite(delta)) max_abs = std::max(max_abs, delta);
        }
      }
      const bool pass = errors == 0;
      std::cout << "oracle=diagnostic_raw_ff1 mismatches=" << errors
                << " elements="
                << std::size_t(kDiagnosticColumns) * oracle_rows.size()
                << " sampled_rows=" << oracle_rows.size()
                << " active_columns=" << kDiagnosticColumns
                << " max_abs=" << max_abs << '\n'
                << "ff1_role_dual_diagnostic="
                << (pass ? "pass" : "FAIL")
                << " repetition=" << repetition << '\n';
      all_pass = all_pass && pass;
      if (!pass) break;
      continue;
    }
    if (stop_stage != 0 && stop_stage != 1 && stop_stage != 2) {
      std::cout << "structural_launch=pass numeric_validation=not_run\n";
      continue;
    }
    std::vector<float> host_checksums(m_tiles*2);
    std::vector<std::uint8_t> host_ring(m_tiles*kSlices * sizeof(DsmASlot));
    std::vector<float> host_ff1(rows * kHidden), host_ff2(rows * kModel);
    std::vector<float> host_hidden(rows * kHidden);
    std::vector<float> host_next(rows * kModel);
    std::vector<ElementBias> host_residual_output(rows * kModel);
    std::vector<std::uint8_t> host_next_values(rows * kModel / 2U);
    std::vector<ElementScale> host_next_scales(size(filter_zeros(next_scale_layout)));
    std::vector<std::uint32_t> host_addresses(m_tiles*4);
    require_cuda(cudaMemcpy(host_checksums.data(), checksums, host_checksums.size()*sizeof(float), cudaMemcpyDeviceToHost), "copy checksums");
    require_cuda(cudaMemcpy(host_ring.data(), ring_snapshots, host_ring.size(), cudaMemcpyDeviceToHost), "copy ring snapshots");
    require_cuda(cudaMemcpy(host_ff1.data(), raw_ff1, host_ff1.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy raw FF1");
    require_cuda(cudaMemcpy(host_ff2.data(), output_ff2, host_ff2.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy output FF2");
    require_cuda(cudaMemcpy(host_residual_output.data(), residual_output,
        host_residual_output.size() * sizeof(ElementBias), cudaMemcpyDeviceToHost), "copy residual output");
    require_cuda(cudaMemcpy(host_next_values.data(), next_values,
        host_next_values.size(), cudaMemcpyDeviceToHost), "copy next values");
    require_cuda(cudaMemcpy(host_next_scales.data(), next_scales,
        host_next_scales.size() * sizeof(ElementScale), cudaMemcpyDeviceToHost), "copy next scales");
    require_cuda(cudaMemcpy(host_addresses.data(), address_info, host_addresses.size()*sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "copy addresses");
    // Decode the public native layout, then compare logical numbers against an
    // independent dense reference. No checksum participates in acceptance.
    for (int row : oracle_rows) {
      for (int col = 0; col < kHidden; ++col) {
        const auto base = std::size_t((row/kM)*kSlices + col/128) * sizeof(DsmASlot);
        const auto nibble = static_cast<std::size_t>(Ff2PhysicalStageLayoutA{}(row%kM, col % 128));
        const auto sf = static_cast<std::size_t>(Ff2StageLayoutSFA{}(row%kM, col % 128));
        const auto raw = (host_ring.at(base + nibble / 2) >> (4 * (nibble & 1))) & 15;
        ElementScale scale;
        static_assert(sizeof(ElementScale) == 1);
        const auto scale_raw = host_ring.at(base + offsetof(DsmASlot, scales) + sf);
        std::memcpy(&scale, &scale_raw, sizeof(scale));
        host_hidden[row * kHidden + col] =
            sm120_fused_reference::decode_e2m1(raw) * float(scale);
      }
    }
#if STREAM1_FUSED_NATIVE_EPILOGUE
    // Distinguish scale-address corruption from FP4 payload-address corruption.
    int bad_scales=0, bad_payload=0;
    for (int row : oracle_rows) for (int col=0; col<kHidden; ++col) {
      const auto base=std::size_t((row/kM)*kSlices+col/128)*sizeof(DsmASlot);
      const auto nibble=std::size_t(Ff2PhysicalStageLayoutA{}(row%kM,col%128));
      const auto sf=std::size_t(Ff2StageLayoutSFA{}(row%kM,col%128));
      float maximum=0;
      for(int j=(col/16)*16;j<(col/16)*16+16;++j)
        maximum=std::max(maximum,std::max(0.0F,fixture.ff1[row*kHidden+j]));
      ElementScale expected_scale(maximum>0?maximum/6.0F:1.0F);
      const float inverse=float(expected_scale)>0?1.0F/float(expected_scale):0.0F;
      cutlass::float_e2m1_t expected_value(std::max(0.0F,fixture.ff1[row*kHidden+col])*inverse);
      const int actual_value=(host_ring.at(base+nibble/2)>>(4*(nibble&1)))&15;
      const int actual_scale=host_ring.at(base+offsetof(DsmASlot,scales)+sf);
      bad_scales+=(actual_scale!=int(expected_scale.raw()));
      bad_payload+=(actual_value!=int(expected_value.raw()));
      if(row==0 && col<64 && (actual_value!=int(expected_value.raw()) || actual_scale!=int(expected_scale.raw())))
        std::cout<<"native_map col="<<col<<" value="<<actual_value<<" expected_value="<<int(expected_value.raw())
          <<" scale="<<actual_scale<<" expected_scale="<<int(expected_scale.raw())<<'\n';
    }
    std::cout<<"native_components bad_scale_elements="<<bad_scales<<" bad_payload_elements="<<bad_payload<<'\n';
#endif
    bool pass = sm120_fused_reference::compare_rows(
        "raw_ff1", host_ff1, fixture.ff1, kHidden, oracle_rows);
    pass = sm120_fused_reference::compare_rows(
        "quantized_hidden", host_hidden, fixture.hidden, kHidden, oracle_rows) && pass;
    if (stop_stage == 0) pass = sm120_fused_reference::compare_rows(
        "ff2", host_ff2, fixture.ff2, kModel, oracle_rows) && pass;
    if (stop_stage == 0 && !kRoleSeparated4Cta) {
      std::vector<float> host_residual_float(host_residual_output.size());
      std::transform(host_residual_output.begin(), host_residual_output.end(),
          host_residual_float.begin(), [](ElementBias value) {
            return static_cast<float>(value);
          });
      auto scale_tensor = cute::make_tensor(host_next_scales.data(), next_scale_layout);
      for (int row : oracle_rows) {
        for (int column = 0; column < kModel; ++column) {
          const std::size_t logical = static_cast<std::size_t>(row) * kModel + column;
          const std::uint8_t raw = static_cast<std::uint8_t>(
              (host_next_values[logical / 2U] >> ((logical & 1U) * 4U)) & 0x0fU);
          host_next[logical] = sm120_fused_reference::decode_e2m1(raw) *
              static_cast<float>(scale_tensor(row, column, 0));
        }
      }
      pass = sm120_fused_reference::compare_rows(
          "residual_fp16", host_residual_float, fixture.residual_fp16,
          kModel, oracle_rows) && pass;
      pass = sm120_fused_reference::compare_rows(
          "next_nvfp4", host_next, fixture.next_nvfp4,
          kModel, oracle_rows) && pass;
    }
    std::cout << (kRoleSeparated4Cta
                      ? "ff1_ff2_role_separated_four_cta="
                      : "ff1_ff2_fused_serial_pairs=")
              << (pass ? "pass" : "FAIL")
              << " repetition=" << repetition << " rank0=" << host_checksums[0]
              << " rank1=" << host_checksums[1] << " ring_hash=" << fnv1a(host_ring)
              << " shared_bytes=" << shared_bytes;
    if (stop_stage != 1) std::cout << " ff1_hidden_offset=" << host_addresses[0]
                                 << " ff2_hidden_offset=" << host_addresses[1];
    std::cout << '\n';
    all_pass = all_pass && pass;
    if (!pass) break;
  }
  if (all_pass && benchmark_iterations) {
    // Standalone FFN component, NOT complete Stream1. Every two-CTA cluster
    // owns distinct rows. Omit snapshots and checksums in the timed path.
    cudaStream_t stream;
    require_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking), "benchmark stream");
    auto launch_timed = [&]() {
#if STREAM1_ROLE_SEPARATED_4CTA
      if (stop_stage == 0) {
        ff1_ff2_role_separated_four_cta_kernel<true><<<
            m_tiles * RoleSeparatedCtaPlan::kClusterCtas,
            kKernelThreads, shared_bytes, stream>>>(
            ff1_params, ff2_params, ff1_bias, nullptr,
            nullptr, output_ff2, nullptr, 0);
      } else {
        ff1_ff2_role_separated_four_cta_kernel<false><<<
            m_tiles * RoleSeparatedCtaPlan::kClusterCtas,
            kKernelThreads, shared_bytes, stream>>>(
            ff1_params, ff2_params, ff1_bias, nullptr,
            nullptr, output_ff2, nullptr, stop_stage);
      }
#else
      ff1_ff2_fused_serial_pairs_kernel<<<m_tiles*2, kKernelThreads, shared_bytes, stream>>>(
          ff1_params, ff2_params, ff1_bias, ff2_bias, residual_input,
          layernorm_gamma, layernorm_beta, nullptr,
          nullptr, nullptr, residual_output, next_values, next_scales,
          next_scale_layout, nullptr, nullptr, 0, parallel_move, 1.0e-5F);
#endif
    };
    require_cuda(cudaMemsetAsync(residual_output, 0xff,
        rows*kModel*sizeof(ElementBias),stream), "poison warmup residual");
    for (int i=0;i<5;++i) launch_timed();
    require_cuda(cudaDeviceSynchronize(), "benchmark warmup");
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (timing_mode == 1) {
      require_cuda(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal), "capture begin");
      for (int i=0;i<benchmark_iterations;++i) launch_timed();
      require_cuda(cudaStreamEndCapture(stream,&graph), "capture end");
      require_cuda(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0), "instantiate graph");
      require_cuda(cudaGraphLaunch(executable,stream), "graph warmup");
      require_cuda(cudaStreamSynchronize(stream), "graph warmup sync");
    }
    cudaEvent_t begin, end;
    require_cuda(cudaEventCreate(&begin), "event begin");
    require_cuda(cudaEventCreate(&end), "event end");
    std::vector<float> samples;
    std::vector<double> enqueue_samples;
    for (int trial=0;trial<7;++trial) {
      // Skipped writes must not inherit valid results from the numerical gate.
      // This memset is queued before, and excluded from, the timed interval.
      require_cuda(cudaMemsetAsync(residual_output, 0xff,
          rows*kModel*sizeof(ElementBias),stream), "poison timed residual");
      require_cuda(cudaEventRecord(begin,stream), "record begin");
      const auto host_begin = std::chrono::steady_clock::now();
      if (timing_mode == 1) require_cuda(cudaGraphLaunch(executable,stream), "graph launch");
      else for (int i=0;i<benchmark_iterations;++i) launch_timed();
      const auto host_end = std::chrono::steady_clock::now();
      enqueue_samples.push_back(std::chrono::duration<double,std::micro>(host_end-host_begin).count()/benchmark_iterations);
      require_cuda(cudaGetLastError(), "timed launch");
      require_cuda(cudaEventRecord(end,stream), "record end");
      require_cuda(cudaEventSynchronize(end), "sync end");
      float ms=0;
      require_cuda(cudaEventElapsedTime(&ms,begin,end), "elapsed time");
      samples.push_back(ms*1000.0F/benchmark_iterations);
    }
    if (kRoleSeparated4Cta && stop_stage == 0) {
      std::vector<float> final_ff2(rows * kModel);
      require_cuda(cudaMemcpy(final_ff2.data(), output_ff2,
          final_ff2.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "timed FF2 output");
      all_pass = sm120_fused_reference::compare_rows(
          "timed_ff2", final_ff2, fixture.ff2, kModel, oracle_rows);
    } else if (!kRoleSeparated4Cta) {
      std::vector<ElementBias> final_residual_half(rows*kModel);
      require_cuda(cudaMemcpy(final_residual_half.data(), residual_output,
          final_residual_half.size()*sizeof(ElementBias), cudaMemcpyDeviceToHost), "timed residual output");
      std::vector<float> final_residual(final_residual_half.size());
      std::transform(final_residual_half.begin(), final_residual_half.end(),
          final_residual.begin(), [](ElementBias value) { return static_cast<float>(value); });
      all_pass = sm120_fused_reference::compare_rows(
          "timed_residual_fp16", final_residual, fixture.residual_fp16,
          kModel, oracle_rows);
    }
    cudaFuncAttributes resources{};
#if STREAM1_ROLE_SEPARATED_4CTA
    require_cuda(cudaFuncGetAttributes(&resources,
        ff1_ff2_role_separated_four_cta_kernel<true>), "kernel resources");
#else
    require_cuda(cudaFuncGetAttributes(&resources,ff1_ff2_fused_serial_pairs_kernel), "kernel resources");
#endif
    cudaDeviceProp device{};
    require_cuda(cudaGetDeviceProperties(&device,0), "device properties");
    cudaLaunchConfig_t config{};
    config.gridDim=dim3(m_tiles*(kRoleSeparated4Cta ? 4 : 2),1,1);
    config.blockDim=dim3(kKernelThreads,1,1);
    config.dynamicSmemBytes=shared_bytes;
    int active_clusters=0;
#if STREAM1_ROLE_SEPARATED_4CTA
    require_cuda(cudaOccupancyMaxActiveClusters(&active_clusters,
        ff1_ff2_role_separated_four_cta_kernel<true>,&config), "cluster occupancy");
#else
    require_cuda(cudaOccupancyMaxActiveClusters(&active_clusters,
        ff1_ff2_fused_serial_pairs_kernel,&config), "cluster occupancy");
#endif
    std::cout << "timing_samples_us=";
    for (auto sample:samples) std::cout << sample << ',';
    std::cout << '\n';
    std::sort(samples.begin(),samples.end());
    std::sort(enqueue_samples.begin(),enqueue_samples.end());
    // Dense useful GEMM work: FF1 2*M*D*H plus FF2 2*M*H*D.
    double useful_flops=4.0*rows*kHidden*kModel;
    if (stop_stage == 11) {
      useful_flops /= kDualRoleFf1MathWg ? 4.0 : 8.0;
    } else if (stop_stage == 14) {
      useful_flops /= 8.0;
    }
    if (stop_stage == 15) useful_flops /= 4.0;
    std::cout << "timing_scope=standalone_multicluster_serial_pair_ffn hidden=" << kHidden
              << " rows=" << rows << " m_tiles=" << m_tiles
              << " oracle_tiles=" << oracle_tiles
              << " stop_stage=" << stop_stage
              << " timing_mode=" << (timing_mode ? "graph" : "eager")
              << " useful_flops=" << std::uint64_t(useful_flops)
              << " useful_tflops=" << useful_flops/(samples[3]*1e6)
              << " median_enqueue_us_per_kernel=" << enqueue_samples[3]
              << " gpu_sms=" << device.multiProcessorCount << " active_clusters=" << active_clusters
              << " threads=" << kKernelThreads << " parallel_move=" << parallel_move
              << " direct_register_pack=" << kDirectRegisterPack
              << " no_relocation=" << kNoRelocation << " direct_accumulation=" << kAccumulateDirect
              << " shared_carry=" << kSharedCarry
              << " role_separated_4cta=" << kRoleSeparated4Cta
              << " median_us=" << samples[3] << " min_us=" << samples.front()
              << " max_us=" << samples.back() << " trials=7 iterations=" << benchmark_iterations
              << " registers=" << resources.numRegs << " local_bytes=" << resources.localSizeBytes
              << " shared_bytes=" << shared_bytes << " snapshot_writes=0\n";
    require_cuda(cudaEventDestroy(begin), "destroy begin");
    require_cuda(cudaEventDestroy(end), "destroy end");
    if (executable) require_cuda(cudaGraphExecDestroy(executable), "destroy graph executable");
    if (graph) require_cuda(cudaGraphDestroy(graph), "destroy graph");
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
  }
  return all_pass ? 0 : 1;
}
