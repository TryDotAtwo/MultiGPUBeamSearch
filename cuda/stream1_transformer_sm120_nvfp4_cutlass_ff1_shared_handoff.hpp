#pragma once

#include "stream1_transformer_sm120_nvfp4_cutlass_persistent_dsm.hpp"

#if defined(STREAM1_HANDOFF_DEVICE_TRACE)
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

  template <class BaseCallbacks, class TiledCopy>
  struct ConsumerStoreCallbacks : BaseCallbacks {
    DsmASlot* hidden;
    TiledCopy tiled_copy;
    int thread_idx;
    int n_tile;

    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
        BaseCallbacks&& callbacks,
        DsmASlot* hidden_,
        TiledCopy const& tiled_copy_,
        int thread_idx_,
        int n_tile_)
        : BaseCallbacks(cute::move(callbacks)),
          hidden(hidden_),
          tiled_copy(tiled_copy_),
          thread_idx(thread_idx_),
          n_tile(n_tile_) {}

    template <class SmemTensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void reduce(
        SmemTensor&& smem_buffer,
        SyncFn const& sync_fn,
        int epi_m,
        int epi_n,
        bool is_last_iteration,
        VTensor visit_results) {
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
      // in tC_rSFD.  Mirror those registers into FF2's native SFA stage while
      // retaining the global SFD store as a temporary numerical oracle.  For
      // SFVecSize=16 one quad owns a complete scale vector, hence lane 0 is
      // the unique writer and no additional inter-warp synchronization is
      // required.
      static_assert(SFVecSize == Contract::kScaleVector);
      if ((thread_idx & 3) == 0 &&
          elem_less(
              this->tC_cSFD(
                  _0{}, _0{}, _0{}, epi_m, epi_n),
              this->residue_tC_cSFD)) {
        auto register_scales = cute::filter_zeros(this->tC_rSFD);
        auto global_scales = cute::filter_zeros(
            this->tC_gSFD(
                _, _, _, _0{}, _0{},
                cute::get<0>(this->tile_coord_mn) + epi_m,
                cute::get<1>(this->tile_coord_mn) + epi_n));
        auto* global_base = cute::raw_pointer_cast(global_scales.data());
        auto* scale_root = reinterpret_cast<
            typename Base::UnderlyingElementBlockScaleFactor*>(
                this->params_ptr->ptr_scale_factor);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < cute::size(register_scales); ++i) {
          const auto physical_offset = static_cast<std::ptrdiff_t>(
              global_base - scale_root + global_scales.layout()(i));
          if (physical_offset >= 0 &&
              physical_offset < Contract::kScalesPerSlot) {
            hidden[n_tile % Contract::kRingSlots]
                .scales[physical_offset] = register_scales(i).raw();
          }
        }
      }

    }

    template <class RegisterTensor, class NativeTiledCopy>
    CUTLASS_DEVICE void register_d_ready(
        RegisterTensor const& registers,
        NativeTiledCopy const& native_tiled_copy,
        int epi_m,
        int epi_n,
        int) {
      auto hidden_a = make_dsm_a_stage_view(
          hidden[n_tile % Contract::kRingSlots].values);
      auto hidden_epi = cute::flat_divide(hidden_a, EpilogueTile{});
      auto thread_copy = native_tiled_copy.get_slice(thread_idx);
      auto thread_hidden = thread_copy.partition_D(hidden_epi);
      cute::copy(
          native_tiled_copy, registers,
          thread_hidden(_, _, _, epi_m, epi_n));
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
    return ConsumerStoreCallbacks<decltype(callbacks), decltype(args.tiled_copy)>(
        cute::move(callbacks), hidden, args.tiled_copy, args.thread_idx,
        static_cast<int>(cute::get<1>(args.tile_coord_mnkl)));
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
    }
  };

  using Impl::Impl;
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
