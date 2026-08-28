# Molab SM120 FF1 -> FF2 shared handoff

Target: NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120), CUDA 13,
CUTLASS `7107b05535f8977f5ecb9d01ee203205b1fd9bc4`.

## Verified

- Exact FF1 epilogue -> FF2 A-stage byte permutation: PASS.
- `768` vector runs per 128x128 tile (`512 x 8 B`, `256 x 16 B`).
- Re-run on the current Molab session:
  - compile: `43.782 s`;
  - errors: `0`;
  - CUDA error: none;
  - measured handoff overhead: `767.343 cycles/tile`, `0.315779 us/tile`;
  - effective shared-memory copy bandwidth: `25.9422 GB/s`.
- Full custom CUTLASS `GemmUniversal::operator()` compile gate after adding
  direct shared SFA mirroring: PASS.
  - compile: `58.886 s`;
  - epilogue shared storage: `30,720 B`;
  - kernel shared storage: `86,016 B`.
- Exact numerical FF1 A/B on the real SM120: PASS.
  - stock and shared-handoff `D` hash:
    `9872799873217201027`;
  - stock and shared-handoff `SFD` hash:
    `5428121410781643651`;
  - `D` nonzero bytes: `65,536` for both variants;
  - `SFD` nonzero bytes: `8,192` for both variants.
- Optimized timing snapshot on the same workload (`M=128, N=1024, K=256`):
  - stock O3: `8.23648 us`;
  - shared-handoff O2 with the temporary global oracle still enabled:
    `41.1139 us`.
  - This is not the target fast path: the custom variant currently performs
    both the stock global materialization and the new shared materialization.

## Implementation state

`Sm120BlockScaleFactorRowSharedHandoff` now:

1. keeps the stock FP32 accumulation, ReLU, row-wise amax, UE4M3 scale and
   NVFP4 conversion as the correctness oracle;
2. writes quantized FF1 values directly into FF2's native shared A layout;
3. mirrors the already computed scale registers directly into FF2's native
   shared SFA ring slot;
4. temporarily retains stock global D/SFD stores as a numerical oracle.

The global materialization is not removed yet. Exact end-to-end D/SFD A/B now
passes; the next implementation step is a shared-only root epilogue/store that
retains the verified scale reduction and NVFP4 conversion while suppressing
the stock TMA/global D store. The existing CUTLASS visitor has no public
`no-global-store` switch, so this must be done as an explicit custom epilogue
contract rather than by passing a null/dummy pointer.

## Molab execution constraint

The pair endpoint interrupts a single foreground SSE execution when a heavy
CUTLASS translation unit runs longer than roughly one minute. A notebook cell
started through a short launcher is also interrupted once the controlling SSE
disconnects, so it is not a valid workaround. Detached/background jobs were
not used. The split-TU direct `Kernel::Params` launcher did make complete
stock/custom numerical builds possible, and those binaries produced the exact
A/B and timing results above. An additional debug readback wrapper crossed the
same compile limit and was dropped rather than leaving an unverified test path.

## Shared-only CUTLASS gate

- Added a `Ff1SharedOnlyKernel` whose global `ElementD` is compile-time void,
  while the existing fusion visitor remains the sole owner of the packed FF1
  hidden values and UE4M3 scales.
- Full SM120 operator compilation passed in `59.13-59.72 s`; the separate
  host TU compiled in `45.57 s`; link passed.  The type/ABI executable passed:
  `shared_only=pass`, epilogue storage `30,720 B`, kernel storage `86,016 B`
  before the CUTLASS compatibility experiment.
- The first runtime launch failed before entering the custom fusion `reduce`
  callback. Device tracing showed `kernel_enter` on all eight output CTAs and
  no `reduce_begin`, excluding the verified shared NVFP4 value/scale mirror as
  the source of the fault.
- Source inspection found an upstream CUTLASS void-D gap in
  `sm90_epilogue_tma_warpspecialized.hpp`: despite
  `is_destination_supported=false`, `store()` still constructs output tensors
  from the default/empty TMA-D descriptor, omits the shared D tile required as
  fusion-reduction workspace, and advances the empty store pipeline.
- A minimal Molab-only compatibility patch was prepared and compiled: retain
  the internal shared D workspace, derive output iteration shape without a
  global descriptor, and compile-time suppress D descriptor/store-pipeline
  operations. The final traced rebuild was interrupted when the Molab sandbox
  expired, so no runtime or timing claim is made for the patched path yet.

Next exact gate: recreate the Molab session, apply the idempotent compatibility
patch, compile the device and host TUs in foreground, then require a clean
single launch with global D unchanged. Only after that result is the
shared-only visitor eligible for the persistent FF1 -> FF2 kernel.
## Docker compile / Molab execute workflow

The expensive `sm_120a` compile is now performed locally in the official
`nvidia/cuda:13.3.0-devel-ubuntu24.04` image against CUTLASS commit
`7107b05535f8977f5ecb9d01ee203205b1fd9bc4`. Molab remains the only execution
and benchmark environment.

- Local nvcc: CUDA 13.3 (`V13.3.33`).
- Molab nvcc/linker: CUDA 13.3 (`V13.3.73`).
- First diagnostic build: kernel 1,418,744 bytes, host 2,444,336 bytes.
- Upload succeeded; Molab link completed in 0.315 s.
- The resulting binary reproduced the same illegal-memory-access failure,
  proving that the split Docker artifact path itself is viable.
- CUTLASS trace reached kernel entry and shared-storage binding, but not
  pipeline construction. A finer trace reached descriptor-prefetch begin and
  then faulted before the mainloop-prefetch completion marker.
- Suppressing only the void-D epilogue descriptor prefetch did not change the
  failure. Both descriptor prefetch operations are optional hints; a build
  with both suppressed for void-D is prepared but local Docker Desktop stopped
  before that final object could be produced.

No end-to-end speed claim is made from these diagnostic `-Xptxas=-O0` builds.
## Root cause and corrected O3 result

The illegal access was caused by a split-translation-unit ABI mismatch, not by
TMA, NVFP4 MMA, or the void-D shared workspace:

- kernel definition used mutable by-value `Params`;
- host declaration used `CUTLASS_GRID_CONSTANT Params const`.

A monolithic compile rejected this mismatch. After making the definition match
the declaration, the Molab kernel completed mainloop MMA, epilogue reduction,
and shared handoff successfully.

Optimized validation (`-O3`, normal prefetch restored, 5 warmups, 200 measured):

- status: PASS;
- latency: **41.0318 us**;
- global hidden D: zero writes observed (`d_nonzero=0`);
- SFD hash: `5428121410781643651` (oracle match);
- workspace: 0 bytes.

This is essentially equal to the earlier dual global+shared custom callback
(`41.1139 us`) and slower than stock (`8.23648 us`). Resource usage is 168
registers for both stock and shared-only, but stack rises from 32 to 64 bytes
per thread. Therefore the next optimization is to consume CUTLASS's native sD
tile directly in FF2 and eliminate the second custom conversion/copy path.
## Native CUTLASS register-D seam

The 41 us callback duplicated work already performed by CUTLASS: it allocated
another register tensor and repeated the FP32-to-NVFP4 conversion. A narrow
`register_d_ready()` hook now receives CUTLASS's native `tRS_rD` immediately
after stock conversion and copies it into the DSM ring.

- shared-only resource usage before: 168 registers, 64-byte stack/thread;
- after native seam: 168 registers, 32-byte stack/thread;
- stock: 168 registers, 32-byte stack/thread.

Molab O3 result:

- latency: **8.23056 us**;
- stock reference: **8.23648 us**;
- global hidden D writes: none (`d_nonzero=0`);
- SFD oracle hash: `5428121410781643651`;
- status: PASS.

This establishes a cost-neutral FF1 producer seam. The next required benchmark
is fused FF1-to-FF2 end-to-end with the existing DSM consumer collective.

## First fused FF1 -> FF2 gate

A 2-CTA, 384-thread fused correctness kernel was added for the reduced
`M=128, hidden=256, model=256` workload. It runs FF1, retains the two native
NVFP4 hidden slices in shared storage, cluster-synchronizes, and reuses the
existing FF2 B/SFB TMA plus A/SFA bulk-DSM consumer path.

Observed on Molab before the sandbox expired:

- both CTAs entering the stock static-1x1 FF1 universal wrapper inside a
  physical 2-CTA cluster caused an unspecified launch failure/hang;
- rank 0 alone completed both the FF1-only and handoff-only stage gates;
- the failure therefore precedes FF2 and is not caused by the DSM copy;
- attempting a runtime-cluster SM120 block-scaled collective is rejected by
  CUTLASS at compile time with `Cluster has to be static` and
  `no programmatic multicast on this arch`.

The prepared correction keeps a physical 2-CTA cluster for the following DSM
handoff while giving each FF1 CTA logical singleton scheduler/pipeline state.
The global block index still assigns distinct N tiles. This is guarded by
`STREAM1_PHYSICAL_CLUSTER_LOGICAL_SINGLETON`; normal CUTLASS kernels are
unchanged. Docker CUDA 13.3 compiled the corrected binary successfully:

- binary: `ff1_ff2_fused_numeric`;
- SHA256: `7022a2d9b11d0bcc9f379d112e85abd446d25ec83855f7a32146a4ba9faee9ec`.

The Molab endpoint returned HTTP 410 after the intentionally failing control
hung the GPU context. Consequently the corrected binary has **not** yet been
executed, and no correctness or speed claim is made for it. The next gate is a
bounded FF1-only run in a fresh Molab sandbox, followed by the complete fused
numeric run only if that stage succeeds.

## Rank-1 isolation and scheduler oracle

Subsequent bounded Molab isolation sharpened the failure boundary:

- physical 2-CTA cluster creation only: PASS;
- rank-0 FF1 only: PASS;
- rank-1 FF1 only: 15-second timeout, leaving the device at 100% utilization;
- forcing logical singleton CTA rank in both the CUTLASS kernel wrapper and
  block-scaled mainloop TMA partition did not change the rank-1 timeout.

This makes FF2, DSM handoff, and physical cluster creation unlikely causes.
The next candidate is `StaticPersistentTileScheduler`, whose worker seed still
comes from physical `blockIdx.x`. A diagnostic-only build forces
`current_work_linear_idx_=0` and `total_grid_size_=1` under the existing
logical-singleton macro. It compiled successfully in Docker CUDA 13.3:

- binary: `build-docker-sm120/ff1-ff2-worker0/ff1_ff2_fused_numeric`;
- SHA256: `2da68d1a08b297ce1846c7c883d01c0bbb259af68c6c4263588deb1a3bd819e3`;
- planned Molab gate: mode 12 only, foreground, hard timeout 15 seconds.

The current Molab GPU is poisoned by the prior rank-1 hang: `nvidia-smi`
reports 100% utilization with 0 MiB and no processes. Restarting the marimo
process does not reset that device state, so no scheduler-oracle run is valid
until Molab provisions a new sandbox. No runtime correctness or performance
claim is made for the oracle build.

The oracle was subsequently run once in fresh sandbox
`sb-ed6183c58e81b6a3` through the foreground marimo SSE path. Its artifact hash
matched, but mode 12 again timed out at 15.12 seconds with no program output.
Therefore the static scheduler worker seed is **not** the root cause.

The next physical-rank leak was found in
`sm100_blockscaled_mma_array_warpspecialized.hpp`: both the main MMA slice and
the SFB MMA slice are selected directly from physical `blockIdx.x`, independent
of the already corrected CTA rank and scheduler state. A guarded diagnostic
override now selects AtomThrID 0 for both slices under
`STREAM1_PHYSICAL_CLUSTER_LOGICAL_SINGLETON`. Compilation is pending because
Docker Desktop's Linux engine stopped during the local compile-only build; no
Molab execution claim is made for this correction yet.

## Actual SM120 dispatch and direct-collective gates

Fresh foreground tests in Molab sandbox `sb-2d24b73047510832` disproved the
remaining SM100-oriented diagnostic assumption.  The FF1 builder selects:

- dispatch policy `MainloopSm120TmaWarpSpecializedBlockScaled<3,3,...>`;
- collective implementation `sm120_blockscaled_mma_tma.hpp`;
- kernel wrapper `sm90_gemm_tma_warpspecialized_cooperative.hpp`.

It does **not** execute the instrumented
`sm100_gemm_tma_warpspecialized.hpp` path.  The selected SM120 mainloop is a
stateless collective and must be default-constructed; all runtime state is
passed explicitly through `Params`, pipeline objects, and shared storage.

This also explains the earlier contradictory constructor diagnostics.  The
full nested `GemmUniversal::operator()` remained an invalid composition inside
the physical two-CTA handoff kernel, even when its body was reduced to an early
return.  A no-op wrapper using the same parameter reference and shared pointer
passed, isolating the kernel-level universal boundary rather than the argument
ABI or DSM storage.

The replacement direct-collective path was advanced through four bounded
Molab gates, all using two CTAs, 384 threads per CTA, a physical cluster of two,
and the logical singleton FF1 collective:

| Gate | SHA256 | Result |
|---|---|---|
| construct stateless SM120 mainloop and shared-only epilogue | `13018da3b993e592069c9f56ae23950bcc5e00083438597c0de602c8e81d4aff` | PASS |
| add FF1/epilogue TMA descriptor prefetch | `b9e671b3704e1df839c4f0a228b638e90d37efb56f89972060bd4565bed98158` | PASS |
| add CUTLASS mainloop pipeline construction | `fe98ceda4516bdfbc8afbe4295172a8e4f352520e2a668d10259c17fa6d91f8a` | PASS |
| add singleton visibility fence and block-scaled `load_init` | `fc697165bc0e0e60e34f41431993965d0350b941496a889ffc3c7edda25e0e53` | PASS |

Each binary completed in roughly 0.3 seconds including process/CUDA startup.
The test program returns code 1 because these diagnostic gates intentionally
produce zero checksums; its structural verdict is
`ff1_ff2_fused_two_slice=pass`.  There was no timeout, launch failure, or
poisoned GPU context.

Therefore the next valid step is a scheduler-free, fixed-tile producer/consumer
gate using the selected SM120 collective's `load`, `mma`, `mma_tail`, and
shared-only epilogue APIs.  Reintroducing the universal scheduler would undo
the isolation and is not justified by these results.  No end-to-end numerical
or throughput claim is made yet.
