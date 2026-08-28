import os
from pathlib import Path

path = Path(
    os.environ.get(
        "CUTLASS_ROOT",
        "/tmp/cayleypy_sm120_native/cutlass_7107b055",
    )
) / (
    "include/cutlass/epilogue/collective/"
    "sm90_epilogue_tma_warpspecialized.hpp"
)
text = path.read_text()

replacements = {
    """  using SmemDStorage = cute::conditional_t<is_destination_supported,
                         SmemArrayTypeD,
                         EmptyType>;""": """  // A void global destination can still require the epilogue shared tile
  // as fusion-callback reduction workspace (Stream1 FF1 -> FF2 handoff).
  using SmemDStorage = SmemArrayTypeD;""",
    """    Tensor mD_mn = params.tma_store_d.get_tma_tensor(make_shape(M,N,L));                               //       (M,N,L)
    Tensor mD = coalesce(mD_mn, take<0,2>(CtaTileMNK{}));
    Tensor gD = local_tile(mD, take<0,2>(CtaTileMNK{}), coord_shape);                                  // (CTA_M,CTA_N)
""": """    auto mD_mn = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      if constexpr (is_destination_supported) {
        return params.tma_store_d.get_tma_tensor(make_shape(M,N,L));
      }
      else {
        return make_identity_tensor(make_shape(M,N,L));
      }
    }();
    Tensor mD = coalesce(mD_mn, take<0,2>(CtaTileMNK{}));
    Tensor gD = local_tile(mD, take<0,2>(CtaTileMNK{}), coord_shape);                                  // (CTA_M,CTA_N)
""",
    """    // thread(b)lock-partition for (s)mem to (g)mem copy (bSG_)
    ThrCopy thrblk_s2g = params.tma_store_d.get_slice(Int<0>{});
    Tensor bSG_sD = thrblk_s2g.partition_S(sD_epi);                                    // (S2G,S2G_M,S2G_N,PIPE_D)
    Tensor bSG_gD = thrblk_s2g.partition_D(gD_epi);                                    // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)

""": "",
    """    bool issue_tma_store = (thread_idx / NumThreadsPerWarp) == 0;""": """    bool issue_tma_store = is_destination_supported &&
                           (thread_idx / NumThreadsPerWarp) == 0;""",
    """      if constexpr (is_destination_supported) {
        if (issue_tma_store) {
          copy(params.tma_store_d, bSG_sD(_,_,_,store_pipe_producer_state.index()), bSG_gD(_,_,_,epi_m,epi_n));
        }
      }""": """      if constexpr (is_destination_supported) {
        ThrCopy thrblk_s2g = params.tma_store_d.get_slice(Int<0>{});
        Tensor bSG_sD = thrblk_s2g.partition_S(sD_epi);
        Tensor bSG_gD = thrblk_s2g.partition_D(gD_epi);
        if (issue_tma_store) {
          copy(params.tma_store_d, bSG_sD(_,_,_,store_pipe_producer_state.index()), bSG_gD(_,_,_,epi_m,epi_n));
        }
      }""",
    """    // wait for all TMA stores to complete
    store_pipeline.producer_tail(store_pipe_producer_state);""": """    // wait for all real TMA stores to complete
    if constexpr (is_destination_supported) {
      store_pipeline.producer_tail(store_pipe_producer_state);
    }""",
    """        // Copy tile from register to smem
        if constexpr (is_destination_supported) {""": """        // Optional fused-consumer seam: expose CUTLASS's already converted D
        // registers before the ordinary shared/global destination path.
        if constexpr (requires {
          cst_callbacks.register_d_ready(
              tRS_rD, tiled_r2s, epi_m, epi_n,
              store_pipe_producer_state.index());
        }) {
          cst_callbacks.register_d_ready(
              tRS_rD, tiled_r2s, epi_m, epi_n,
              store_pipe_producer_state.index());
        }
        // Copy tile from register to smem
        if constexpr (is_destination_supported) {""",
}

for old, new in replacements.items():
    if old not in text:
        if new not in text:
            raise RuntimeError(f"CUTLASS patch anchor missing: {old[:100]!r}")
        continue
    text = text.replace(old, new, 1)

path.write_text(text)
print("cutlass_void_d_patch=applied", path, flush=True)

kernel_path = Path(
    os.environ.get(
        "CUTLASS_ROOT",
        "/tmp/cayleypy_sm120_native/cutlass_7107b055",
    )
) / "include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp"
kernel_text = kernel_path.read_text()
kernel_replacements = {
    """    int cluster_size = size(cluster_shape);
    uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();""": """    int cluster_size = size(cluster_shape);
    uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();
#if defined(STREAM1_PHYSICAL_CLUSTER_LOGICAL_SINGLETON)
    // Two independent static-1x1 block-scaled GEMMs share a physical cluster
    // only so the following FF2 stage can consume their DSM-resident output.
    cluster_size = 1;
    cta_rank_in_cluster = 0;
#endif""",
    """    dim3 block_id_in_cluster = cute::block_id_in_cluster();""": """    dim3 block_id_in_cluster = cute::block_id_in_cluster();
#if defined(STREAM1_PHYSICAL_CLUSTER_LOGICAL_SINGLETON)
    block_id_in_cluster = dim3(0, 0, 0);
#endif""",
}
for old, new in kernel_replacements.items():
    if old not in kernel_text:
        if new not in kernel_text:
            raise RuntimeError(f"CUTLASS kernel patch anchor missing: {old!r}")
        continue
    kernel_text = kernel_text.replace(old, new, 1)
kernel_path.write_text(kernel_text)
print("cutlass_logical_singleton_patch=applied", kernel_path, flush=True)

# The block-scaled mainloop reconstructs its TMA partitions in load_init() and
# used the physical cluster rank again.  A physical rank of one is outside the
# static 1x1 collective layout and leaves that CTA waiting on an impossible
# transaction.  Keep the physical rank only for the outer DSM handoff; inside
# the singleton FF1 collective both CTAs must partition as logical rank zero.
mainloop_path = Path(
    os.environ.get(
        "CUTLASS_ROOT",
        "/tmp/cayleypy_sm120_native/cutlass_7107b055",
    )
) / "include/cutlass/gemm/collective/sm100_blockscaled_mma_mixed_tma_cpasync_warpspecialized.hpp"
mainloop_text = mainloop_path.read_text()
mainloop_old = """    uint32_t cta_rank_in_cluster = static_cast<uint32_t>(cute::block_rank_in_cluster());
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster);"""
mainloop_new = """    uint32_t cta_rank_in_cluster = static_cast<uint32_t>(cute::block_rank_in_cluster());
#if defined(STREAM1_PHYSICAL_CLUSTER_LOGICAL_SINGLETON)
    cta_rank_in_cluster = 0;
#endif
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster);"""
if mainloop_old not in mainloop_text:
    if mainloop_new not in mainloop_text:
        raise RuntimeError("CUTLASS block-scaled load_init patch anchor missing")
else:
    mainloop_text = mainloop_text.replace(mainloop_old, mainloop_new, 1)
mainloop_path.write_text(mainloop_text)
print("cutlass_blockscaled_logical_singleton_patch=applied", mainloop_path, flush=True)

# The SM120 block-scaled array mainloop also chooses its MMA and SFB thread
# slices directly from physical blockIdx.x.  That is independent of the CTA
# rank used for TMA partitioning, so physical rank 1 still enters AtomThrID 1
# even after the logical-rank override above.  In the embedded singleton FF1
# subproblem each physical CTA must instantiate AtomThrID 0; the outer wrapper
# remains responsible for selecting the distinct N=128 parameters.
array_mainloop_path = Path(
    os.environ.get(
        "CUTLASS_ROOT",
        "/tmp/cayleypy_sm120_native/cutlass_7107b055",
    )
) / "include/cutlass/gemm/collective/sm100_blockscaled_mma_array_warpspecialized.hpp"
array_mainloop_text = array_mainloop_path.read_text()
array_mainloop_replacements = (
    (
        """    ThrMMA cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));""",
        """#if defined(STREAM1_PHYSICAL_CLUSTER_LOGICAL_SINGLETON)
    ThrMMA cta_mma = TiledMma{}.get_slice(0);
#else
    ThrMMA cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));
#endif""",
    ),
    (
        """    ThrMMA cta_mma_sfb = TiledMMA_SF{}.get_slice(blockIdx.x % size(typename TiledMMA_SF::AtomThrID{}));""",
        """#if defined(STREAM1_PHYSICAL_CLUSTER_LOGICAL_SINGLETON)
    ThrMMA cta_mma_sfb = TiledMMA_SF{}.get_slice(0);
#else
    ThrMMA cta_mma_sfb = TiledMMA_SF{}.get_slice(blockIdx.x % size(typename TiledMMA_SF::AtomThrID{}));
#endif""",
    ),
)
for old, new in array_mainloop_replacements:
    if old not in array_mainloop_text:
        if new not in array_mainloop_text:
            raise RuntimeError(f"CUTLASS block-scaled AtomThrID patch anchor missing: {old!r}")
        continue
    array_mainloop_text = array_mainloop_text.replace(old, new, 1)
array_mainloop_path.write_text(array_mainloop_text)
print(
    "cutlass_blockscaled_atom_thread_logical_singleton_patch=applied",
    array_mainloop_path,
    flush=True,
)

scheduler_path = Path(
    os.environ.get(
        "CUTLASS_ROOT",
        "/tmp/cayleypy_sm120_native/cutlass_7107b055",
    )
) / "include/cutlass/gemm/kernel/static_tile_scheduler.hpp"
scheduler_text = scheduler_path.read_text()
scheduler_old = """    total_grid_size_ = uint64_t(gridDim.x) * uint64_t(gridDim.y) * uint64_t(gridDim.z);"""
scheduler_new = """#if defined(STREAM1_PHYSICAL_CLUSTER_LOGICAL_SINGLETON)
    // Diagnostic/embedded-subproblem mode: each physical CTA owns an
    // independent singleton GEMM and therefore starts at logical worker zero.
    current_work_linear_idx_ = 0;
    total_grid_size_ = 1;
#else
    total_grid_size_ = uint64_t(gridDim.x) * uint64_t(gridDim.y) * uint64_t(gridDim.z);
#endif"""
if scheduler_old not in scheduler_text:
    if scheduler_new not in scheduler_text:
        raise RuntimeError("CUTLASS static scheduler logical-worker patch anchor missing")
else:
    scheduler_text = scheduler_text.replace(scheduler_old, scheduler_new, 1)
scheduler_path.write_text(scheduler_text)
print("cutlass_static_scheduler_logical_singleton_patch=applied", scheduler_path, flush=True)
