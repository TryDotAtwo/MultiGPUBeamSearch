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

# Older revisions of this patcher matched the unguarded copy anchor inside the
# already inserted seam and could append the seam again on every invocation.
# Repair such vendor trees deterministically and make repetition a hard error.
register_d_ready_seam = """        // Optional fused-consumer seam: expose CUTLASS's already converted D
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
"""
while register_d_ready_seam + register_d_ready_seam in text:
    text = text.replace(
        register_d_ready_seam + register_d_ready_seam,
        register_d_ready_seam,
    )
if text.count(register_d_ready_seam) != 1:
    raise RuntimeError(
        "CUTLASS register_d_ready seam must occur exactly once, found "
        f"{text.count(register_d_ready_seam)}"
    )

path.write_text(text)
print("cutlass_void_d_patch=applied", path, flush=True)

# Stream1's production FF1 epilogue consumes the block-scale registers through
# its shared-memory handoff.  CUTLASS's stock visitor also stores the same SFD
# tile to global memory.  Keep that store available for standalone numerical
# oracles, but allow the no-global-hidden performance build to suppress it
# without changing the scale computation or register layout.
visitor_path = Path(
    os.environ.get(
        "CUTLASS_ROOT",
        "/tmp/cayleypy_sm120_native/cutlass_7107b055",
    )
) / (
    "include/cutlass/epilogue/fusion/"
    "sm120_visitor_store_tma_warpspecialized.hpp"
)
visitor_text = visitor_path.read_text()
global_sfd_store = """      if (write_sf && elem_less(tC_cSFD(_0{}, _0{}, _0{}, epi_m, epi_n), residue_tC_cSFD)) {
        copy_aligned(tC_rSFD, tC_gSFD(_, _, _, _0{}, _0{}, get<0>(tile_coord_mn) + epi_m, get<1>(tile_coord_mn) + epi_n));
      }"""
guarded_global_sfd_store = """#if !defined(STREAM1_CUTLASS_SKIP_GLOBAL_SFD)
      if (write_sf && elem_less(tC_cSFD(_0{}, _0{}, _0{}, epi_m, epi_n), residue_tC_cSFD)) {
        copy_aligned(tC_rSFD, tC_gSFD(_, _, _, _0{}, _0{}, get<0>(tile_coord_mn) + epi_m, get<1>(tile_coord_mn) + epi_n));
      }
#endif"""
if guarded_global_sfd_store not in visitor_text:
    store_count = visitor_text.count(global_sfd_store)
    if store_count != 2:
        raise RuntimeError(
            f"CUTLASS global SFD patch expected 2 anchors, found {store_count}"
        )
    visitor_text = visitor_text.replace(
        global_sfd_store, guarded_global_sfd_store
    )
visitor_path.write_text(visitor_text)
print("cutlass_optional_global_sfd_patch=applied", visitor_path, flush=True)

kernel_path = Path(
    os.environ.get(
        "CUTLASS_ROOT",
        "/tmp/cayleypy_sm120_native/cutlass_7107b055",
    )
) / "include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp"
kernel_text = kernel_path.read_text()
kernel_replacements = {
    """  operator() (Params const& params, char* smem_buf) {

    using namespace cute;""": """  operator() (Params const& params, char* smem_buf) {

#if defined(STREAM1_CUTLASS_DIAG_STOP_AT_OPERATOR_ENTRY)
    return;
#else

    using namespace cute;""",
    """    uint32_t lane_predicate = cute::elect_one_sync();
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{});""": """    uint32_t lane_predicate = cute::elect_one_sync();
#if defined(STREAM1_CUTLASS_DIAG_STOP_AFTER_WARP_SETUP)
    return;
#endif
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{});""",
    """    [[maybe_unused]] uint32_t mma_peer_cta_rank = has_mma_peer_cta ? cta_rank_in_cluster ^ 1 : cta_rank_in_cluster;

    // Kernel level shared memory storage""": """    [[maybe_unused]] uint32_t mma_peer_cta_rank = has_mma_peer_cta ? cta_rank_in_cluster ^ 1 : cta_rank_in_cluster;

#if defined(STREAM1_CUTLASS_DIAG_STOP_AFTER_CLUSTER_RANK_SETUP)
    return;
#endif

    // Kernel level shared memory storage""",
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
    """    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // In a warp specialized kernel, collectives expose data movement and compute operations separately""": """    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

#if defined(STREAM1_CUTLASS_DIAG_STOP_BEFORE_COLLECTIVE_CONSTRUCTION)
    return;
#endif

    // In a warp specialized kernel, collectives expose data movement and compute operations separately""",
    """    CollectiveMainloop collective_mainloop(params.mainloop, cluster_shape, cta_rank_in_cluster);
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

    // Issue Tma Descriptor Prefetch from a single thread""": """    CollectiveMainloop collective_mainloop(params.mainloop, cluster_shape, cta_rank_in_cluster);
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

#if defined(STREAM1_CUTLASS_DIAG_STOP_AFTER_COLLECTIVE_CONSTRUCTION)
    return;
#endif

    // Issue Tma Descriptor Prefetch from a single thread""",
    """    if ((warp_category == WarpCategory::EpilogueLoad) && lane_predicate) {
      collective_epilogue.prefetch_tma_descriptors(params.epilogue);
    }

    // Do we load source tensor C or other aux inputs""": """    if ((warp_category == WarpCategory::EpilogueLoad) && lane_predicate) {
      collective_epilogue.prefetch_tma_descriptors(params.epilogue);
    }

#if defined(STREAM1_CUTLASS_DIAG_STOP_AFTER_DESCRIPTOR_PREFETCH)
    return;
#endif

    // Do we load source tensor C or other aux inputs""",
    """    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop,
                                       mainloop_pipeline_params,
                                       cluster_shape,
                                       cute::true_type{},   // Perform barrier init
                                       cute::false_type{}); // Delay mask calculation

    // Epilogue Load pipeline""": """    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop,
                                       mainloop_pipeline_params,
                                       cluster_shape,
                                       cute::true_type{},   // Perform barrier init
                                       cute::false_type{}); // Delay mask calculation

#if defined(STREAM1_CUTLASS_DIAG_STOP_AFTER_MAINLOOP_PIPELINE_CONSTRUCTION)
    return;
#endif

    // Epilogue Load pipeline""",
    """    // We need this to guarantee that the Pipeline init is visible
    // To all producers and consumer threadblocks in the cluster
    pipeline_init_arrive_relaxed(cluster_size);""": """#if defined(STREAM1_CUTLASS_DIAG_STOP_BEFORE_PIPELINE_INIT_ARRIVE)
    return;
#endif

    // We need this to guarantee that the Pipeline init is visible
    // To all producers and consumer threadblocks in the cluster
    pipeline_init_arrive_relaxed(cluster_size);""",
    """    else {
    }
  }
};

///////////////////////////////////////////////////////////////////////////////""": """    else {
    }
#endif
  }
};

///////////////////////////////////////////////////////////////////////////////""",
}
for old, new in kernel_replacements.items():
    if new in kernel_text:
        continue
    if old not in kernel_text:
        raise RuntimeError(f"CUTLASS kernel patch anchor missing: {old!r}")
    kernel_text = kernel_text.replace(old, new, 1)

pipeline_stop_old = """    pipeline_init_wait(cluster_size);

    if (is_participant.main_load) {"""
pipeline_stop_new = """#if defined(STREAM1_CUTLASS_DIAG_STOP_BEFORE_PIPELINE_INIT_WAIT)
    // Diagnostic-only gate: pipeline objects and their shared-memory state
    // have been constructed, but the collective initialization barrier has
    // not executed yet.  The define is uniform for the whole CTA.
    return;
#endif

    pipeline_init_wait(cluster_size);

#if defined(STREAM1_CUTLASS_DIAG_STOP_AFTER_PIPELINE_INIT)
    // Diagnostic-only gate: every warp has completed CTA-local pipeline
    // construction and initialization, but no scheduler, TMEM, TMA or MMA
    // operation has started yet.
    return;
#endif

    if (is_participant.main_load) {"""
if pipeline_stop_new not in kernel_text:
    if pipeline_stop_old not in kernel_text:
        raise RuntimeError("CUTLASS pipeline-init diagnostic anchor missing")
    kernel_text = kernel_text.replace(pipeline_stop_old, pipeline_stop_new, 1)
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
    if new in array_mainloop_text:
        continue
    if old not in array_mainloop_text:
        raise RuntimeError(f"CUTLASS block-scaled AtomThrID patch anchor missing: {old!r}")
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

# PipelineTmaAsync normally maps the singleton cluster's consumer-release
# destination to logical block zero.  When a logical singleton is embedded in
# a larger physical cluster, CTA rank 1 would consequently release CTA 0's
# EMPTY barrier and deadlock its own producer.  Keep the ordinary singleton
# behavior unchanged, but target the physical self CTA in the guarded
# embedded-subproblem mode.
pipeline_path = Path(
    os.environ.get(
        "CUTLASS_ROOT",
        "/tmp/cayleypy_sm120_native/cutlass_7107b055",
    )
) / "include/cutlass/pipeline/sm90_pipeline.hpp"
pipeline_text = pipeline_path.read_text()
pipeline_old = """      if (cluster_size == 1) {
        is_signaling_thread_ = true;
        dst_blockid_ = 0;
      }"""
pipeline_new = """      if (cluster_size == 1) {
        is_signaling_thread_ = true;
#if defined(STREAM1_PHYSICAL_CLUSTER_LOGICAL_SINGLETON)
        // The logical pipeline is local, but DSM barrier addressing still uses
        // the enclosing physical cluster rank.
        dst_blockid_ = cute::block_rank_in_cluster();
#else
        dst_blockid_ = 0;
#endif
      }"""
if pipeline_old not in pipeline_text:
    if pipeline_new not in pipeline_text:
        raise RuntimeError("CUTLASS singleton pipeline self-release anchor missing")
else:
    pipeline_text = pipeline_text.replace(pipeline_old, pipeline_new, 1)
pipeline_path.write_text(pipeline_text)
print("cutlass_pipeline_logical_singleton_self_release_patch=applied", pipeline_path, flush=True)
