#pragma once

#include "stream1_transformer_sm120_nvfp4_cutlass_persistent_dsm.hpp"
#if defined(STREAM1_HANDOFF_FIXED_AFFINE)
#include <stdexcept>
#endif

// Use the native thread-partitioned tensor, never the relative coordinates
// tC_cSFD (which omit the thread displacement). The independent address audit
// verifies all 1024 scale bytes have exactly one writer and match CUTLASS SFD.
#ifndef STREAM1_HANDOFF_SCALE_NATIVE_PARTITION
#define STREAM1_HANDOFF_SCALE_NATIVE_PARTITION 1
#endif
#if defined(STREAM1_HANDOFF_SCALE_DISTRIBUTED_QUAD_STORE)
#error "Rejected experiment: distributed ownership does not match row-scale visitor"
#endif

#if defined(STREAM1_HANDOFF_DEVICE_TRACE) || \
    defined(STREAM1_HANDOFF_SCALE_MAP_TRACE) || \
    defined(STREAM1_HANDOFF_VALUE_READBACK)
#include <cstdio>
#endif

namespace stream1_sm120_nvfp4_cutlass_dsm {

template <
    int SFVecSize,
    class EpilogueTile,
    class CtaTileShapeMNK,
    int FragmentSize,
    class ElementOutput_,
    class ElementCompute_,
    class ElementBlockScaleFactor,
    cutlass::FloatRoundStyle RoundStyle =
        cutlass::FloatRoundStyle::round_to_nearest>
struct Sm120BlockScaleFactorRowSharedHandoff
    : cutlass::epilogue::fusion::Sm120BlockScaleFactorRowStore<
          SFVecSize, EpilogueTile, CtaTileShapeMNK, FragmentSize,
          ElementOutput_, ElementCompute_, ElementBlockScaleFactor, RoundStyle> {
  using Base = cutlass::epilogue::fusion::Sm120BlockScaleFactorRowStore<
      SFVecSize, EpilogueTile, CtaTileShapeMNK, FragmentSize,
      ElementOutput_, ElementCompute_, ElementBlockScaleFactor, RoundStyle>;
  using Params = typename Base::Params;

  struct SharedStorage : Base::SharedStorage {
    alignas(1024) DsmASlot hidden[Contract::kRingSlots];
  };

  DsmASlot* hidden = nullptr;

  CUTLASS_HOST_DEVICE
  Sm120BlockScaleFactorRowSharedHandoff() = default;

  CUTLASS_HOST_DEVICE
  Sm120BlockScaleFactorRowSharedHandoff(
      Params const& params, SharedStorage const& shared_storage)
      : Base(params, shared_storage),
        hidden(const_cast<DsmASlot*>(shared_storage.hidden)) {}

  template <class BaseCallbacks, class TiledCopy, class SharedScaleTensor>
  struct ConsumerStoreCallbacks : BaseCallbacks {
    DsmASlot* hidden;
    TiledCopy tiled_copy;
    int thread_idx;
    int n_tile;
    SharedScaleTensor shared_scale_partition;

    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
        BaseCallbacks&& callbacks,
        DsmASlot* hidden_,
        TiledCopy const& tiled_copy_,
        int thread_idx_,
        int n_tile_,
        SharedScaleTensor const& shared_scale_partition_)
        : BaseCallbacks(cute::move(callbacks)),
          hidden(hidden_),
          tiled_copy(tiled_copy_),
          thread_idx(thread_idx_),
          n_tile(n_tile_),
          shared_scale_partition(shared_scale_partition_) {}

    template <class SmemTensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void reduce(
        SmemTensor&& smem_buffer,
        SyncFn const& sync_fn,
        int epi_m,
        int epi_n,
        bool is_last_iteration,
        VTensor visit_results) {
#if defined(STREAM1_HANDOFF_CHECKSUM_CONTROL)
      // Diagnostic only: observable consumption of EVERY accumulator fragment.
      // Each CTA owns 128 floats in the preallocated diagnostic SFD allocation.
      // Volatile stores retain every epilogue iteration, not only the last one.
      float checksum = 0.0F;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < cute::size(visit_results); ++i) {
        auto fragment = visit_results(i);
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < fragment.size(); ++j) {
          checksum += static_cast<float>(fragment[j]);
        }
      }
      const auto cta = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      reinterpret_cast<volatile float*>(this->params_ptr->ptr_scale_factor)[
          cta * 128 + thread_idx] = checksum;
      return;
#endif
#if defined(STREAM1_HANDOFF_DEVICE_TRACE)
      if (thread_idx == 0 && n_tile == 0) {
        printf("ff1_trace reduce_begin epi=%d,%d hidden=%p\\n", epi_m, epi_n,
               static_cast<void*>(hidden));
      }
#endif
      BaseCallbacks::reduce(
          cute::forward<SmemTensor>(smem_buffer), sync_fn,
          epi_m, epi_n, is_last_iteration, visit_results);
#if defined(STREAM1_HANDOFF_DEVICE_TRACE)
      if (thread_idx == 0 && n_tile == 0) {
        printf("ff1_trace base_reduce_done epi=%d,%d\\n", epi_m, epi_n);
      }
#endif

      // The stock visitor has already computed the exact row-wise SFD values
      // in tC_rSFD. Mirror those registers into FF2's native SFA stage. The
      // numerical-oracle build may retain CUTLASS's global SFD store; the
      // producer-only performance build defines
      // STREAM1_CUTLASS_SKIP_GLOBAL_SFD and consumes only this shared copy.
      // For SFVecSize=16 each quad owns one complete scale reduction. CUTLASS
      // leaves the result replicated across the quad, so lane 0 of every quad
      // is the unique writer. Preserve that native ownership and address FF2's shared SFA tensor
      // directly. Reconstructing a global SFD tensor and subtracting its base
      // pointer here generated expensive wide integer address arithmetic in
      // every epilogue callback and nearly doubled measured FF1 latency.
      static_assert(SFVecSize == Contract::kScaleVector);
#if !defined(STREAM1_HANDOFF_SKIP_SHARED_SCALES)
#if defined(STREAM1_HANDOFF_SCALE_DISTRIBUTED_QUAD_STORE)
      if (
#else
      if ((thread_idx & 3) == 0 &&
#endif
          elem_less(
              this->tC_cSFD(
                  _0{}, _0{}, _0{}, epi_m, epi_n),
              this->residue_tC_cSFD)) {
        auto register_scales = cute::filter_zeros(this->tC_rSFD);
#if defined(STREAM1_HANDOFF_SCALE_NATIVE_PARTITION)
        cute::copy_aligned(
            this->tC_rSFD,
            shared_scale_partition(_, _, _, epi_m, epi_n));
#if defined(STREAM1_HANDOFF_SCALE_MAP_TRACE)
        if (n_tile == 0) {
          auto destination = cute::filter_zeros(
              shared_scale_partition(_, _, _, epi_m, epi_n));
          auto global = cute::filter_zeros(this->tC_gSFD(
              _, _, _, _0{}, _0{},
              cute::get<0>(this->tile_coord_mn) + epi_m,
              cute::get<1>(this->tile_coord_mn) + epi_n));
          const auto shared_base =
              reinterpret_cast<std::uint8_t*>(cute::raw_pointer_cast(destination.data())) -
              hidden[n_tile % Contract::kRingSlots].scales;
          const auto global_base =
              cute::raw_pointer_cast(global.data()) -
              reinterpret_cast<typename Base::UnderlyingElementBlockScaleFactor*>(
                  this->params_ptr->ptr_scale_factor);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < cute::size(register_scales); ++i) {
            printf(
                "ff1_native_scale_map thread=%d epi_m=%d epi_n=%d i=%d "
                "native=%d global=%d expected=%u observed=%u\n",
                thread_idx, epi_m, epi_n, i,
                static_cast<int>(shared_base + destination.layout()(i)),
                static_cast<int>((global_base + global.layout()(i)) &
                                 (Contract::kScalesPerSlot - 1)),
                static_cast<unsigned>(register_scales(i).raw()),
                static_cast<unsigned>(reinterpret_cast<volatile std::uint8_t*>(
                    cute::raw_pointer_cast(destination.data()))[
                        destination.layout()(i)]));
          }
        }
#endif
#elif defined(STREAM1_HANDOFF_SCALE_VECTOR_COPY_MASKED_GLOBAL) || \
    defined(STREAM1_HANDOFF_SCALE_ADDRESSING_MASKED_GLOBAL)
        static_assert(
            (Contract::kScalesPerSlot & (Contract::kScalesPerSlot - 1)) == 0,
            "masked CTA-local SFA addressing requires a power-of-two slot");
        auto global_scales = cute::filter_zeros(
            this->tC_gSFD(
                _, _, _, _0{}, _0{},
                cute::get<0>(this->tile_coord_mn) + epi_m,
                cute::get<1>(this->tile_coord_mn) + epi_n));
        auto* global_base = cute::raw_pointer_cast(global_scales.data());
        auto* scale_root = reinterpret_cast<
            typename Base::UnderlyingElementBlockScaleFactor*>(
                this->params_ptr->ptr_scale_factor);
        const auto cta_local_base = static_cast<std::uintptr_t>(
            global_base - scale_root) & (Contract::kScalesPerSlot - 1);
#if defined(STREAM1_HANDOFF_SCALE_VECTOR_COPY_MASKED_GLOBAL)
        auto shared_register_scales = cute::make_tensor(
            cute::make_smem_ptr(
                reinterpret_cast<typename Base::UnderlyingElementBlockScaleFactor*>(
                    hidden[n_tile % Contract::kRingSlots].scales) +
                cta_local_base),
            global_scales.layout());
        cute::copy_aligned(register_scales, shared_register_scales);
#else
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < cute::size(register_scales); ++i) {
          const auto physical_offset =
              (cta_local_base + static_cast<std::uintptr_t>(
                  global_scales.layout()(i))) &
              (Contract::kScalesPerSlot - 1);
#if defined(STREAM1_HANDOFF_SCALE_MAP_TRACE)
          if (n_tile == 0) {
            printf(
                "ff1_scale_global_map thread=%d epi_m=%d epi_n=%d i=%d "
                "base=%llu layout=%d physical=%llu\n",
                thread_idx, epi_m, epi_n, i,
                static_cast<unsigned long long>(cta_local_base),
                static_cast<int>(global_scales.layout()(i)),
                static_cast<unsigned long long>(physical_offset));
          }
#endif
          hidden[n_tile % Contract::kRingSlots]
              .scales[physical_offset] = register_scales(i).raw();
        }
#endif
#else
#if defined(STREAM1_HANDOFF_SCALE_ANALYTIC_OFFSET)
        const int warp = thread_idx >> 5;
        const int quad = (thread_idx & 31) >> 2;
        const int physical_base =
            warp * 256 + quad * 16 + epi_m * 8 + (epi_n >> 1) * 512 +
            (epi_n & 1) * 2;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < cute::size(register_scales); ++i) {
          const int physical_offset =
              physical_base + (i >> 1) * 4 + (i & 1) * 128;
          hidden[n_tile % Contract::kRingSlots]
              .scales[physical_offset] = register_scales(i).raw();
        }
#else
        auto scale_coordinates = cute::filter_zeros(
            this->tC_cSFD(_, _, _, epi_m, epi_n));
        auto shared_scales = make_dsm_sfa_stage_view(
            hidden[n_tile % Contract::kRingSlots].scales);
#if defined(STREAM1_HANDOFF_SCALE_DISTRIBUTED_QUAD_STORE)
        const int scale_index = thread_idx & 3;
        auto coordinate = scale_coordinates(scale_index);
        const int row = static_cast<int>(cute::get<0>(coordinate)) &
                        (Contract::kRows - 1);
        const int column = static_cast<int>(cute::get<1>(coordinate)) &
                           (Contract::kSliceColumns - 1);
        shared_scales(row, column) = register_scales(scale_index);
#else
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < cute::size(register_scales); ++i) {
          auto coordinate = scale_coordinates(i);
          const int row = static_cast<int>(cute::get<0>(coordinate)) &
                          (Contract::kRows - 1);
          const int column = static_cast<int>(cute::get<1>(coordinate)) &
                             (Contract::kSliceColumns - 1);
#if defined(STREAM1_HANDOFF_SCALE_MAP_TRACE)
          if (n_tile == 0) {
            printf(
                "ff1_scale_map thread=%d epi_m=%d epi_n=%d i=%d row=%d "
                "column=%d physical=%d\n",
                thread_idx, epi_m, epi_n, i, row, column,
                static_cast<int>(Ff2StageLayoutSFA{}(row, column)));
          }
#endif
          shared_scales(row, column) = register_scales(i);
        }
#endif
#endif
#endif
      }
#endif

    }

#if defined(STREAM1_HANDOFF_DIRECT_PACK_STORE)
    template <class RegisterTensor, class NativeTiledCopy>
    CUTLASS_DEVICE void register_compute_ready(
        RegisterTensor const& registers, NativeTiledCopy const& native_tiled_copy,
        int epi_m, int epi_n, int) {
      auto hidden_a = cute::recast<ElementHidden>(make_dsm_a_stage_view(
          hidden[n_tile % Contract::kRingSlots].values));
      auto hidden_epi = cute::flat_divide(hidden_a, EpilogueTile{});
      auto thread_hidden = native_tiled_copy.get_slice(thread_idx).partition_D(hidden_epi);
      auto destination = thread_hidden(_, _, _, epi_m, epi_n);
      auto target_bytes = cute::recast<std::uint8_t>(destination);
      static_assert(CUTE_STATIC_V(cute::size(typename RegisterTensor::layout_type{})) ==
                    2 * CUTE_STATIC_V(cute::size(target_bytes)));
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < cute::size(target_bytes); ++i) {
        float a = registers(2 * i), b = registers(2 * i + 1);
        unsigned addr = cute::cast_smem_ptr_to_uint(&target_bytes(i));
        asm volatile("{ .reg .b8 packed;\n"
                     "cvt.rn.satfinite.e2m1x2.f32 packed, %2, %1;\n"
                     "st.shared.b8 [%0], packed; }" ::
                     "r"(addr), "f"(a), "f"(b) : "memory");
      }
    }
#endif
    template <class RegisterTensor, class NativeTiledCopy>
    CUTLASS_DEVICE void register_d_ready(
        RegisterTensor const& registers,
        NativeTiledCopy const& native_tiled_copy,
        int epi_m,
        int epi_n,
        int) {
#if !defined(STREAM1_HANDOFF_SKIP_SHARED_VALUES)
      // The MMA storage view is uint4 raw bits; the producer registers are
      // float_e2m1. Match the element type without changing the physical
      // layout, so a scalar-copy fallback cannot numerically cast FP4 to int4.
      auto hidden_a = cute::recast<ElementHidden>(make_dsm_a_stage_view(
          hidden[n_tile % Contract::kRingSlots].values));
      auto hidden_epi = cute::flat_divide(hidden_a, EpilogueTile{});
      auto thread_copy = native_tiled_copy.get_slice(thread_idx);
      // Scalar stores perform no cross-lane copy-atom exchange. Test source
      // ownership against the final logical tensor before adopting this map.
#if defined(STREAM1_HANDOFF_SOURCE_OWNED_STORE)
      auto thread_hidden = thread_copy.partition_S(hidden_epi);
#else
      auto thread_hidden = thread_copy.partition_D(hidden_epi);
#endif
      auto destination = thread_hidden(_, _, _, epi_m, epi_n);
      // The isolated producer has no FF2 consumer yet. Make actual shared
      // writes observable to the compiler, not merely the diagnostic build.
      // This native fragment owns contiguous PAIRS, not eight nibbles.
      // Byte stores preserve both nibbles without racing subbyte RMWs.
      static_assert(CUTE_STATIC_V(cute::max_common_vector(registers, destination)) >= 2,
                    "FF1 shared handoff requires two contiguous owned nibbles");
      auto packed_source = cute::recast<std::uint8_t>(registers);
      auto packed_target = cute::recast<std::uint8_t>(destination);
#if !defined(STREAM1_HANDOFF_DIRECT_PACK_STORE)
      CUTLASS_PRAGMA_UNROLL
      for (int word = 0; word < cute::size(packed_source)
#if defined(STREAM1_HANDOFF_STORE_QUAD_PAIR)
          / 2
#endif
          ; ++word) {
        auto address = cute::cast_smem_ptr_to_uint(&packed_target(word));
        std::uint32_t value = packed_source(word);
#if defined(STREAM1_HANDOFF_VALUE_READBACK)
        if (n_tile == 0 && epi_m == 0 && epi_n == 0) {
          printf("ff1_byte_map thread=%d word=%d offset=%u\n", thread_idx, word,
                 address - cute::cast_smem_ptr_to_uint(hidden[0].values));
        }
#endif
#if defined(STREAM1_HANDOFF_STORE_QUAD)
        // Measured native map: lanes 4q..4q+3 own bytes 4a..4a+3
        // of the same word, at equal fragment indices. Keep raw FP4 bits.
        const unsigned lane_byte = thread_idx & 3;
#if defined(STREAM1_HANDOFF_VALUE_READBACK)
        unsigned base = __shfl_sync(0xffffffffU, address, thread_idx & 28);
        if (address != base + lane_byte || (base & 3)) {
          printf("ff1_quad_address_error thread=%d word=%d\n", thread_idx, word);
          asm volatile("trap;");
        }
#endif
#if defined(STREAM1_HANDOFF_STORE_QUAD_PAIR)
        static_assert(cute::size(packed_source) == 8);
#if defined(STREAM1_HANDOFF_VALUE_READBACK)
        if (cute::cast_smem_ptr_to_uint(&packed_target(word + 4)) != address + 4 ||
            (base & 7)) {
          printf("ff1_pair_address_error thread=%d word=%d\n", thread_idx, word);
          asm volatile("trap;");
        }
#endif
        value |= std::uint32_t(packed_source(word + 4)) << 16;
        value |= __shfl_xor_sync(0xffffffffU, value, 1) << 8;
        unsigned partner = __shfl_xor_sync(0xffffffffU, value, 2);
        unsigned lo = __byte_perm(value, partner, 0x5410);
        unsigned hi = __byte_perm(value, partner, 0x7632);
        if (lane_byte == 0) {
          asm volatile("st.shared.v2.u32 [%0], {%1, %2};" ::
                       "r"(address), "r"(lo), "r"(hi) : "memory");
        }
#else
#if defined(STREAM1_HANDOFF_STORE_QUAD_XOR)
        // Only lane 4q publishes. First gather [b0,b1] and [b2,b3],
        // then concatenate the pairs. No lane-dependent variable shift.
        value |= __shfl_xor_sync(0xffffffffU, value, 1) << 8;
        value |= __shfl_xor_sync(0xffffffffU, value, 2) << 16;
#else
        value <<= 8 * lane_byte;
        value |= __shfl_down_sync(0xffffffffU, value, 1, 4);
        value |= __shfl_down_sync(0xffffffffU, value, 2, 4);
#endif
        if (lane_byte == 0) {
          asm volatile("st.shared.u32 [%0], %1;" :: "r"(address), "r"(value) : "memory");
        }
#endif
#elif defined(STREAM1_HANDOFF_STORE_BATCH_CLOBBER)
        asm volatile("st.shared.u8 [%0], %1;" :: "r"(address), "r"(value));
#else
        asm volatile("st.shared.u8 [%0], %1;" :: "r"(address), "r"(value) : "memory");
#endif
      }
#if defined(STREAM1_HANDOFF_STORE_QUAD) || defined(STREAM1_HANDOFF_STORE_WARP_SYNC)
      __syncwarp();
#endif
#if defined(STREAM1_HANDOFF_STORE_BATCH_CLOBBER)
      // Compiler boundary only; not a CUDA synchronization primitive.
      asm volatile("" ::: "memory");
#endif
#endif
#if defined(STREAM1_HANDOFF_VALUE_READBACK)
      // Diagnostic only. Independently map logical copy coordinates to the
      // FF2 physical nibble address, then force a real shared-byte readback.
      auto coordinates = cute::make_identity_tensor(cute::shape(hidden_a));
      auto coordinate_epi = cute::flat_divide(coordinates, EpilogueTile{});
      // Registers are the SOURCE fragment. STSM permutes ownership across
      // lanes, so destination-partition index i is not source register i.
      auto thread_coordinates = thread_copy.partition_S(coordinate_epi);
      auto positions = thread_coordinates(_, _, _, epi_m, epi_n);
      unsigned long long expected_hash = 1469598103934665603ULL;
      unsigned long long observed_hash = 1469598103934665603ULL;
      unsigned mismatches = 0;
      asm volatile("" ::: "memory");
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < cute::size(registers); ++i) {
        auto coordinate = positions(i);
        const int nibble = static_cast<int>(Ff2PhysicalStageLayoutA{}(
            cute::get<0>(coordinate), cute::get<1>(coordinate)));
        const unsigned observed =
            (reinterpret_cast<volatile std::uint8_t*>(
                hidden[n_tile % Contract::kRingSlots].values)[nibble >> 1] >>
             ((nibble & 1) * 4)) & 15U;
        const unsigned expected = ElementHidden(registers(i)).raw() & 15U;
        if (n_tile == 0 && thread_idx == 0 && epi_m == 0 && epi_n == 0) {
          printf("ff1_value_detail i=%d nibble=%d expected=%u observed=%u source_bits=%d target_bits=%d\n",
                 i, nibble, expected, observed,
                 int(cutlass::sizeof_bits<typename RegisterTensor::value_type>::value),
                 int(cutlass::sizeof_bits<typename decltype(hidden_a)::value_type>::value));
        }
        mismatches += expected != observed;
        expected_hash = (expected_hash ^ expected) * 1099511628211ULL;
        observed_hash = (observed_hash ^ observed) * 1099511628211ULL;
      }
      printf("ff1_value_readback n=%d thread=%d epi_m=%d epi_n=%d count=%d bad=%u expected=%llu observed=%llu\n",
             n_tile, thread_idx, epi_m, epi_n, int(cute::size(registers)),
             mismatches, expected_hash, observed_hash);
#endif
#endif
#if defined(STREAM1_HANDOFF_DEVICE_TRACE)
      if (thread_idx == 0 && n_tile == 0) {
        printf("ff1_trace native_shared_copy_done epi=%d,%d\\n", epi_m, epi_n);
      }
#endif
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      cutlass::epilogue::fusion::ConsumerStoreArgs<Args...> const& args) {
    auto callbacks =
        Base::template get_consumer_store_callbacks<ReferenceSrc>(args);
    const int n_tile = static_cast<int>(cute::get<1>(args.tile_coord_mnkl));
    auto shared_scale_tensor = make_dsm_sfa_stage_view(
        hidden[n_tile % Contract::kRingSlots].scales);
    auto shared_scale_partition =
        cutlass::epilogue::fusion::detail::sm90_partition_for_epilogue<ReferenceSrc>(
            shared_scale_tensor, args.epi_tile, args.tiled_copy, args.thread_idx);
    return ConsumerStoreCallbacks<
        decltype(callbacks), decltype(args.tiled_copy),
        decltype(shared_scale_partition)>(
        cute::move(callbacks), hidden, args.tiled_copy, args.thread_idx,
        n_tile, shared_scale_partition);
  }
};

using Ff1SharedScaleStore = Sm120BlockScaleFactorRowSharedHandoff<
    Contract::kScaleVector,
    typename Ff1Epilogue::EpilogueTile,
    typename Ff1Epilogue::CtaTileMNK,
    Ff1Epilogue::DispatchPolicy::FragmentSize,
    ElementHidden,
    ElementCompute,
    ElementScale>;

#if defined(STREAM1_HANDOFF_FIXED_AFFINE)
// FF1 contract: ReLU(acc + bias). No source C or device scalar fetch.
using Ff1SharedCompute = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::epilogue::thread::ReLU, ElementCompute, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>,
    cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::Sm90Compute<
            cutlass::plus, ElementCompute, ElementCompute,
            cutlass::FloatRoundStyle::round_to_nearest>,
        cutlass::epilogue::fusion::Sm90AccFetch,
        cutlass::epilogue::fusion::Sm90ColBroadcast<
            0, typename Ff1Epilogue::CtaTileMNK, ElementBias, ElementCompute,
            Stride<_1, _0, std::int64_t>,
            128 / cutlass::sizeof_bits<ElementBias>::value>>>;
#else
using Ff1SharedCompute =
    cutlass::epilogue::fusion::Sm90LinCombPerRowBiasEltAct<
        typename Ff1Epilogue::CtaTileMNK,
        cutlass::epilogue::thread::ReLU,
        ElementCompute,
        ElementCompute,
        ElementBias,
        ElementOutput,
        ElementCompute,
        128 / cutlass::sizeof_bits<ElementBias>::value,
        cutlass::FloatRoundStyle::round_to_nearest>;
#endif

struct Ff1SharedFusionCallbacks
    : cutlass::epilogue::fusion::Sm90EVT<
          Ff1SharedScaleStore, Ff1SharedCompute> {
  using Impl = cutlass::epilogue::fusion::Sm90EVT<
      Ff1SharedScaleStore, Ff1SharedCompute>;
  using ActivationArguments = typename cutlass::epilogue::fusion::Sm90Compute<
      cutlass::epilogue::thread::ReLU,
      ElementHidden,
      ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>::Arguments;
  // Required by the SM90 TMA collective when ElementD is compile-time void:
  // it still needs a concrete internal element for the shared epilogue tile,
  // even though no destination descriptor or global store is emitted.
  using ElementAux = ElementHidden;

  struct Arguments {
    ElementCompute alpha = ElementCompute(1);
    ElementCompute beta = ElementCompute(0);
    ElementCompute const* alpha_ptr = nullptr;
    ElementCompute const* beta_ptr = nullptr;
    ElementScale* block_scale_factor_ptr = nullptr;
    ElementCompute const* norm_constant_ptr = nullptr;
    Stride<_0, _0, std::int64_t> dNormConst = {_0{}, _0{}, 0};
    Stride<_0, _0, std::int64_t> dAlpha = {_0{}, _0{}, 0};
    Stride<_0, _0, std::int64_t> dBeta = {_0{}, _0{}, 0};
    ElementBias const* bias_ptr = nullptr;
    Stride<_1, _0, std::int64_t> dBias = {};
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
#if defined(STREAM1_HANDOFF_FIXED_AFFINE)
      if (alpha != ElementCompute(1) || beta != ElementCompute(0) ||
          alpha_ptr != nullptr || beta_ptr != nullptr) {
        throw std::invalid_argument("fixed FF1 requires alpha=1, beta=0, no scalar pointers");
      }
      return {{{{}, {bias_ptr, ElementBias(0), dBias}, {}}, activation},
              {block_scale_factor_ptr, norm_constant_ptr, dNormConst}};
#else
      return {
          {
              {
                  {{beta}, {beta_ptr}, {dBeta}},
                  {},
                  {
                      {{alpha}, {alpha_ptr}, {dAlpha}},
                      {},
                      {bias_ptr, ElementBias(0), dBias},
                      {}}},
              activation},
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst}};
#endif
    }
  };

  using Impl::Impl;

  template <class BaseCallbacks>
  struct ConsumerStoreCallbacks : BaseCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(BaseCallbacks&& callbacks)
        : BaseCallbacks(cute::move(callbacks)) {}

#if defined(STREAM1_HANDOFF_DIRECT_PACK_STORE)
    template <class RegisterTensor, class NativeTiledCopy>
    CUTLASS_DEVICE void register_compute_ready(
        RegisterTensor const& registers, NativeTiledCopy const& copy,
        int epi_m, int epi_n, int iteration) {
      cute::get<1>(this->callbacks_tuple).register_compute_ready(
          registers, copy, epi_m, epi_n, iteration);
    }
#endif
#if !defined(STREAM1_HANDOFF_DIRECT_PACK_STORE) || defined(STREAM1_HANDOFF_VALUE_READBACK)
    template <class RegisterTensor, class NativeTiledCopy>
    CUTLASS_DEVICE void register_d_ready(
        RegisterTensor const& registers,
        NativeTiledCopy const& native_tiled_copy,
        int epi_m, int epi_n, int store_iteration) {
      // EVT forwards only its standard hooks. This custom seam must be
      // explicitly forwarded to child 1 (Store), otherwise the collective's
      // requires-expression silently removes the hidden-payload write.
      cute::get<1>(this->callbacks_tuple).register_d_ready(
          registers, native_tiled_copy, epi_m, epi_n, store_iteration);
    }
#endif
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      cutlass::epilogue::fusion::ConsumerStoreArgs<Args...> const& args) {
    auto callbacks =
        Impl::template get_consumer_store_callbacks<ReferenceSrc>(args);
    return ConsumerStoreCallbacks<decltype(callbacks)>(cute::move(callbacks));
  }
};

using Ff1SharedHandoffEpilogue =
    cutlass::epilogue::collective::CollectiveEpilogue<
        typename Ff1Epilogue::DispatchPolicy,
        typename Ff1Epilogue::CtaTileMNK,
        typename Ff1Epilogue::EpilogueTile,
        typename Ff1Epilogue::ElementC,
        typename Ff1Epilogue::StrideC,
        typename Ff1Epilogue::ElementD,
        typename Ff1Epilogue::StrideD,
        Ff1SharedFusionCallbacks,
        typename Ff1Epilogue::CopyOpG2S,
        typename Ff1Epilogue::SmemLayoutAtomC,
        typename Ff1Epilogue::CopyOpS2R,
        typename Ff1Epilogue::CopyOpS2G,
        typename Ff1Epilogue::SmemLayoutAtomD,
        typename Ff1Epilogue::CopyOpR2S,
        typename Ff1Epilogue::CopyAtomC,
        typename Ff1Epilogue::CopyOpR2R>;

using Ff1SharedHandoffKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, Ff1Mainloop, Ff1SharedHandoffEpilogue, void>;

// Production FF1 handoff: preserve the exact SM90-family TMA epilogue ABI
// selected by the SM120 NVFP4 mainloop, but make D compile-time void.  CUTLASS
// still runs the fusion visitor (ReLU, row amax, UE4M3 scale and E2M1 pack),
// while its register-to-smem D copy and smem-to-global TMA store are removed by
// is_destination_supported=false.  The visitor above is therefore the sole
// owner of the quantized hidden values and scales.
using Ff1SharedOnlyEpilogue =
    cutlass::epilogue::collective::CollectiveEpilogue<
        typename Ff1Epilogue::DispatchPolicy,
        typename Ff1Epilogue::CtaTileMNK,
        typename Ff1Epilogue::EpilogueTile,
        typename Ff1Epilogue::ElementC,
        typename Ff1Epilogue::StrideC,
        void,
        typename Ff1Epilogue::StrideD,
        Ff1SharedFusionCallbacks,
        typename Ff1Epilogue::CopyOpG2S,
        typename Ff1Epilogue::SmemLayoutAtomC,
        typename Ff1Epilogue::CopyOpS2R,
        typename Ff1Epilogue::CopyOpS2G,
        typename Ff1Epilogue::SmemLayoutAtomD,
        typename Ff1Epilogue::CopyOpR2S,
        typename Ff1Epilogue::CopyAtomC,
        typename Ff1Epilogue::CopyOpR2R>;

using Ff1SharedOnlyKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, Ff1Mainloop, Ff1SharedOnlyEpilogue, void>;

CUTLASS_DEVICE DsmASlot* ff1_shared_only_hidden_ring(
    typename Ff1SharedOnlyKernel::SharedStorage& storage) {
  // Sm90EVT<Store, Compute> stores child Compute at tuple index 0 and the
  // node Store at index 1. The latter is our SharedHandoff leaf.
  auto& fusion_storage = storage.tensors.epilogue.thread;
  return cute::get<1>(fusion_storage).hidden;
}

static_assert(cute::is_void_v<typename Ff1SharedOnlyEpilogue::ElementD>);
static_assert(cute::is_same_v<
              typename Ff1SharedOnlyEpilogue::DispatchPolicy,
              typename Ff1SharedHandoffEpilogue::DispatchPolicy>);

}  // namespace stream1_sm120_nvfp4_cutlass_dsm
