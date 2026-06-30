# Prompt History

## 2026-06-30
- User requested running the split Stream1 piece-transformer benchmark on Kaggle 2xT4 and checking speed. Use a separate benchmark package, not the smoke/solve notebooks, so speed logs are clean and compare per-GPU T4 throughput.
- User clarified during Task 1 quality review that MLP and transformer benchmark code should be split into separate files to avoid confusion; keep transformer work isolated from MLP files/paths where practical.
- User assigned Task 1 in `D:\100XH100\.worktrees\stream1-piece-transformer`: add a transformer-only Stream1 microbenchmark path to `tools/stream_benchmark.cu`, explicitly branch on `STREAM1_BACKEND_PIECE_TRANSFORMER`, preserve the existing MLP benchmark path, sweep `B_MICRO={512,1024,2048,4096,8192}` and `STREAM1_CONCURRENCY={1,2,4}`, skip memory-exceeding configs with clear report lines, avoid fallback behavior, update project memory/test results, and keep writes limited to the allowed files. Controller later clarified that no distillation direction or fallback path is allowed and the transformer benchmark must remain an explicit backend branch only.
- User requested solving the same puzzle as the transformer inference example notebook and verifying that the Stream1 `piece_transformer` beam-search run obtains the same solution length. The comparison target is puzzle `991` from the downloaded example output, where the notebook reports length `82` with `B=65536`.
## 2026-06-29
- User assigned Task 7 for the Stream1 piece-transformer worktree: create a private Kaggle 2xT4 small-beam smoke package that clones GitHub branch `codex/stream1-piece-transformer`, exports the Kaggle `megaminx-qtransformer-1782210824` `.pth` as fp16 `piece-transformer` weights, builds `production_runner` and `stream1_transformer_cuda_tests` for SM75 with CUTLASS under `/tmp`, runs a tiny two-rank torchrun smoke with `BEAM_WEIGHT_DIR` pointed at the transformer export, fails clearly instead of falling back when model discovery is uncertain, validates notebook JSON/Python locally, records package-only verification, and commits without pushing.
- User assigned Task 5 quality-fix for the Stream1 piece-transformer worktree: fix only `tools/stream_benchmark.cu` MLP `Stream1NetworkView` construction after LN pointer fields were added, do not change transformer forward semantics, do not wire production transformer dispatch, run a `stream_benchmark` compile check and `stream1_transformer_cuda_tests` if practical, update verification notes, amend the Task 5 commit, and return the final commit SHA.
- User assigned Task 5 for the Stream1 piece-transformer worktree: implement only the standalone CUDA transformer forward path, keep production dispatcher/runner fail-closed guards, add separate transformer view/scratch API, match exported p900 fast input construction and 4-block pre-norm transformer architecture, use no fallback backend, prepare a fused/tiled attention path for seq_len=51 without adding a new FA2 dependency, verify against the real exported reference where available, and commit the work.
- User added a later-task requirement for Stream1 piece-transformer forward: use FlashAttention 2 or similar attention speedups where applicable, preserve the exported weight format, and keep the forward path broadly compatible with transformer-style architectures. Task 4 quality fix must only record this requirement, not implement transformer forward.
- User assigned Task 4 for the Stream1 piece-transformer worktree: implement transformer manifest parsing, exact-size host weight loading, device upload/free containers, and one-lane reusable scratch arena for `backend=piece_transformer`, with no fallback backend, no transformer CUDA forward kernel, no production runner/dispatcher wiring, and no Stream3/Stream4/State128/MLP contract changes.
- User requested an explicit separate Stream1 `piece_transformer` backend for the Kaggle Megaminx Q-transformer, with no fallback behavior and without breaking the current MLP backend. Manifests must be generated from checkpoint/config metadata, correctness tests come before integration, and target validation is a small 2xT4 Kaggle smoke after correctness tests pass.

## 2026-06-20
- User cancelled the segment/frequent-prefix repair direction and requested a
  normal beam-search-to-center mode for puzzles with known solution length 23.
  Requirements: do not stop immediately on first solution; finish the current
  depth and the next depth, collect all solutions from all ranks, store found
  solutions in a static device bucket and host RAM, report counts by solution
  length, and name the mode `solve-bucket`.
- User clarified that the solve-bucket launcher must run both original and
  reflected variants, like the previous original-plus-reflected workflow.

## 2026-06-13
- User requested bf16 support for the IHES one-output Stream1 model on the
  existing CUTLASS path, with no CUDA fallback behavior.
- User requested stopping the current IHES depth-200 run and adding a `+3`
  runtime score offset for the single-output model because its output range can
  start at `-3`.
- User provided an IHES submission CSV and requested tracing a known solution
  through the beam pipeline to find why the solver does not detect it.
- User required the debug tracker to follow a tracked solution across ranks,
  without changing the beam-search architecture.

## 2026-06-12
- User requested making the beam search compile natively for each puzzle's
  state size instead of hard-coding `State128`, with state logical length and
  alignment derived from puzzle config at compile time.
- User asked to try the `cayleypy-ihes-cube` Kaggle competition locally,
  requiring compile-time move-count specialization and direct `puzzle_info`
  generator loading.
- User requested pushing all cluster-start files for IHES cube tests to GitHub.
- User requested avoiding `scancel`, using SLURM dependencies instead, and
  setting cluster script time limits to 24 hours.

## 2026-06-11
- User requested a GitHub-ready portable folder and script so another person can
  clone the repository and run the current original-plus-reflected beam search
  on 8xA100 80GB with `BEAM_WIDTH=1400000000` and `SHARD_COUNT=64`.
- User explicitly requested removing fallback behavior from the common-script
  sourcing path.

## 2026-06-05
- User asked to inspect current uncommitted local work, analyze GitHub and the
  local project, then requested pushing all of the current uncommitted work to
  GitHub.

## 2026-06-04
- User requested replacing the full parameter-combination tuning approach with
  separate staged parameter searches, finding the best value for one parameter
  family at a time.
- User requested changing final materialization exchange capacity from
  `final_chunk * WORLD_SIZE` to a Stream5-style `final_chunk * scale` formula,
  with that scale exposed as a runtime config/environment knob.
- User requested making `FINAL_MATERIALIZE_CHUNK_SWEEP` test exactly
  `65536 131072 262144 524288`, and asked for a phase-by-phase memory estimate
  after the updated Stream5 and final materialization formulas.
- User requested decoupling final materialization chunk sizing from Stream3 batch sizing so multi-GPU `world_size` does not inflate finalization VRAM, and asked whether the required chunk size can be measured with a script/sweep.
- User requested using the MEPhI HPC Work Rules plugin to launch the
  multi-GPU beam search on `8xA100 40GB`.
- User clarified that the cluster job should download the repository from
  GitHub and use depth `12` for the initial run.
- User requested safe auto-cleanup in the cluster script so the run does not
  fill shared disk, without risking damage to cluster files.

## 2026-06-03
- User requested detailed per-depth model prediction statistics for Stream3
  only, without touching Stream1 or Stream2: minimum, mean, percentiles,
  maximum, frontier best/mean, threshold, with later notebook graphing.
- User requested launching the Molab run with `K1=5`, depth 80, and hybrid history mode.
- User reported the 250M run was working well and asked to check memory
  availability and whether depth 80 with beam 250M fits in hybrid mode.
- User asked Codex to read the in-app browser output after the budget cell
  produced an error.
- User switched molab to the RTX Pro 6000-class GPU runtime and requested a
  100M-size beam-like calculation on the 24-output model.
- User requested creating a separate molab folder in the project, checking what
  compute/GPU resources are available in molab, determining whether 2 GPUs or
  one stronger GPU can be used, and running a task there.
- User requested trying the current code on molab at `https://molab.marimo.io/`.

## 2026-05-27
- User required a no-prune CPU history storage redesign: keep `HISTORY_SLOT_COUNT=2/3` pinned `CandidateMeta` copy slots, count slot memory in startup RAM budgeting, compress `CandidateMeta[32B]` to `HistoryEntry[16B]` asynchronously on CPU, store `HistoryEntry` densely in disk/RAM arenas, write disk first, fall back to RAM instead of crashing when disk budget/write fails, and preserve existing `route_packed.source_rank + parent_idx` distributed reconstruction semantics. User requested Kaggle config `HISTORY_SLOT_COUNT=2`, RAM budget `28GB`, disk budget `49GB`, then build/push and test on Kaggle T4x2.
- User approved adding Stream2 solved-neighborhood lookup as the first implementation step. Required naming: not `K1`, use a clear solved-neighborhood radius. Runtime scheme: CPU precomputes the solved neighborhood hashes and suffixes before the run; GPU stores only hash/fingerprint lookup data in readonly VRAM; Stream2 checks `Hash128(parent+move)` against this table; CPU later recovers suffix from the host map.
- User postponed `K2` implementation but required saving the design: Stream2 may later expand descendants from each generated candidate by a separate candidate-suffix radius, and suffix generators for that expansion should be precomputed before the run rather than generated dynamically inside kernels.

## 2026-05-20
- User requested creation of project in `D:\100XH100` from the supplied CUDA multi-stream beam-search architecture.
- User supplied architecture details for config constants, `State128`, `Hash128`, `CandidateMeta`, final exchange layout, memory layout, Stream 1-5 semantics, solved path, final materialization, and approved Stream 4 custom fixed-capacity GPU dedup clarification.
- User clarified that previous CPU scaffold is insufficient and requested separate implementation and testing of every stream, then stitched integration, with all tests running in Docker on local RTX 3070.
- User requested benchmark of first 100 puzzles with depth=100 and beam=2**22, including average time per depth step, solution-length histogram, average solution length, minimum solution length, and maximum solution length.

## 2026-05-22
- User required NVIDIA Nsight profiling for bottleneck analysis, depth=20, state id=0 only, beam=2**22, with GPU memory usage and remaining GPU memory printed before start.
- User explicitly prohibited fallback paths: "????????????????????????????? ?????????????????????????????????"; fallback code is considered error masking.
- User requested Docker cleanup: delete excess Docker data, keep only one CayleyBeam100H100-related image if available, then continue architecture-aligned code work.
- User reiterated continuation requirement: continue project work and do not deviate from architecture.
- User clarified profiler requirement: use NVIDIA Nsight, understand correct usage, build a new clearly named shared GPU image for GPU projects, and audit remaining atomic operations.
- User clarified atomic policy: atomic operations are never allowed anywhere except the found-solution path; Stream 3 and Stream 4 must not use atomics.
- User requested: implement Stream 1 as CUTLASS/custom inference, fold BatchNorm into Linear, remove PyTorch `EmbeddingBag` as runtime kernel, add NVTX ranges in C++/CUDA stream jobs, implement production dispatcher with real CUDA streams, verify in Nsight that Stream 2/3/4/5 overlap with Stream 1, keep Stream 3/4 atomics-free via scan/sort/partition/owned regions, and write Stream 1 fully on CUTLASS.
- User confirmed next work: complete Stream 1 full CUTLASS graph, stitch production dispatcher, profile full Stream 1/2/3/4/5 pipeline in Nsight, and use CUTLASS/NVIDIA/C++/CUDA as much as possible. User explicitly reminded to implement CUTLASS score epilogue.
- User requested: implement CUTLASS epilogue for Stream 1, finish Stream 3/4, then run full production test for puzzle 0 with depth=100 and beam=2**22.
- User demanded removal of PyTorch baseline/deviation, strict architecture-only implementation, and fixed allocation of all arrays before program start.
- User confirmed continuation: implement code strictly according to architecture.
- User requested removal of smoke/fallback habit and reiterated strict architecture: CPU stores history and reconstructs solution only; all other data-plane work runs on GPU through CUDA Graph.
- User requested continuation after the strict CUDA Graph/static memory plan.
- User requested continuation after depth CUDA Graph dispatch work: "Nais, delay dalshe".
- User requested full next-stage implementation: Stream3 remote recv collector, full Stream5/NCCL exchange, Stream3/4 parallel production kernels, threshold histograms plus AllReduce plus final threshold, final load balancing plus materialization plus CPU history reconstruction, and full production runner for puzzle0 depth=100 beam=2**22 with Nsight profile.
- User requested continued implementation of parallel Stream3 sort/dedup/collector, parallel Stream4 insert/dump, multi-rank NCCL orchestration, and AllReduce/AllGather wiring.
- User clarified scalability target: same architecture/code path must work for `WORLD_SIZE=1`, `WORLD_SIZE=2`, and `WORLD_SIZE=100`; no separate intermediate Stream 3/4 variants; implementation must follow the supplied architecture directly.
- User rejected explicit `WORLD_SIZE=1/2/100` config/static-memory checks as misleading because `WORLD_SIZE` is a normal config value; user required Stream 3 sort to move directly to the architecture-preferred CUB/fixed-temp path without intermediate alternatives.
- User decided Stream 4 hash-table path should be removed. User reason: Stream 4 should quickly do threshold + dedup + compact, shard sorting must be batched through a small number of sort scratch slots to avoid excessive VRAM use, and compact-before-sort is acceptable because sorting does not reduce `N_sort` by itself.
- User requested production scheduler work: implement real free-slot management for Stream 4 sort slots, parallelize Stream 3 restore/owner/split, and parallelize Stream 3 collectors.
- User approved global spill ping-pong implementation: add `global_spill_buffer_a + global_spill_buffer_b` so Stream 3 drain reads one spill buffer and writes still-blocked candidates to the other buffer.
- User clarified current phase is single-GPU algorithm polishing; multi-rank is not required now. User requested periodic threshold update, final global threshold, final load balancing, and materialize wiring in the production depth loop, followed by `production_runner 0 20 4194304`.
- User requested connection of real data and neural network assets, with explicit reminder to preserve alignment: "????????????????????????????????? ???????????????? ????????????????? ???? ?????????????????????????. ????????????????????????????????? ???? ????????????????? ???? ????????????????? ?????????????????. ???????? ????????????????????????????? ???? ??????????????????????????????????????".
- User reported Docker container logs only showed NVIDIA header and `ninja: no work to do`, requiring visible runtime progress logging for local Docker runs.
- User requested `B_MICRO` around `8192` for RTX 3070 tuning, per-stream isolated speed benchmarks before production runs, Stream 1 TensorOp conversion, and consideration of future T4 testing.
- User clarified Stream 1 must also be benchmarked with 1/2/3/4 concurrent inference jobs across different batch sizes, with a fixed table recorded in the benchmark document.
- User requested a profiler run to capture how the production pipeline works and asked whether Stream 1 throughput around `22M candidates/sec` should make `beam=2**22` processing effectively instant.
- User accepted the Stream1 metric correction and requested implementation work on parallel no-atomic threshold histogram through fixed scratch plus CUB sort/reduce or deterministic block reduction, and parallel no-atomic final filter/load-balance through mark/count/scan/scatter. User also requested baseline timing before the change for Stream1/2/3/4/5 before finalization, including seconds, throughput, and launch counts.
- User requested a production run for puzzle `0` with depth `100`; current context implies beam `2**22=4194304`.
- User asked to diagnose why execution time grows with depth, whether CPU/history slows the run, and why `stream4_jobs=1088` is high.

## 2026-05-23
- User reported another apparent hang and requested more production work plus production logs.
- User requested current-code bottleneck diagnosis and explicit check whether Stream 2-5 machinery is hidden behind TensorCore Stream 1 inference.
- User asked what `static_allocation_bytes=4054851072`, `layout_streams_bytes=1470860288`, and `layout_final_bytes=2981072128` mean, and whether `layout_streams` and `layout_final` should reuse one static scratch buffer through remapping.
- User accepted the event-driven production scheduler direction and requested the decision be fixed in project memory. User also requested explanation of the Nsight finding where `stream1_folded_input_kernel` took `9.68s` while CUTLASS GEMM took `1.97s` in the depth8 profile.
- User requested rewriting the slow Stream1 folded input section and asked to use NVIDIA CUDA libraries such as CUTLASS or cuBLAS where appropriate.
- User requested implementation of the event-driven scheduler without wave barriers after the Stream1 folded-input optimization.
- User suggested using `RING_COUNT=4` so Stream1 always has write targets.
- User asked to discuss why enabling Stream1 before Stream4 broke spill convergence and clarified that Stream3 should announce shard readiness while host should manage less stream/shard state.
- User approved implementing the Stream3-owned shard-ready handoff: "???????? ????????, ?????????????????????".
- User clarified score/hash ring ownership architecture: Stream1 is the throughput limiter, Stream2 uses the same `B_MICRO` parent batch and then sleeps, Stream3 consumes a full score/hash ring and immediately frees the ring without cleanup, Stream4 and Stream5 must not block score/hash ring reuse, and Stream5 needs its own ring buffers later for send/recv independence.
- User approved implementing the ring pipeline policy with `RING_COUNT=4` and Stream4-independent ring reuse.
- User clarified Stream3 collector architecture: Stream3 should partition input by `shard_id` once, sort by shard id, and write batched transactions into each target shard instead of rescanning every shard over the full input.
- User requested scheduler polling cleanup: ring readiness should not scan all events; Stream3 should launch immediately after a full ring is ready; Stream4 slots should use oldest busy slot waits instead of polling all slots; threshold updates should be tied to accumulated Stream4 completion work without unnecessary synchronization; final flush remains a blocking drain before `layout_final`.
- User clarified Stream5 threshold architecture: histogram state should be fixed VRAM state; Stream5 is communication plus global threshold computation; Stream5 sends/receives `CandidateMeta`, computes global threshold from ready buffers, and may compute/read local card histograms per available shard.
- User requested focused real-data work: optimize Stream 1 and Stream 3 independently for maximum speed, then test combined Stream 1 plus Stream 3 behavior in the production loop.
- User requested running production solutions for puzzles 1 through 20 with depth=200. Current context implies beam=4194304 and Docker `beam-tests` execution.
- User corrected solved CSV semantics: solved/submission CSV must store the generator sequence path, not the final `State128` state text.
- User requested Kaggle test launch flow: push current project code to GitHub `main`, remove the other branch, preserve current local notebook/code, create a Kaggle notebook that clones GitHub main and runs calculation.
- User required Kaggle notebook first cell to contain main config: `BEAM_WIDTH`, start puzzle id from `test.csv`, and puzzle count; current first test is beam `2**22`, first 20 puzzles excluding puzzle 0.
- User required Kaggle notebook last cell to create a PNG histogram of solved solution lengths excluding unsolved puzzles, plus solved count, minimum solution length, maximum solution length, average solution length, and modal solution length.
- User required production logs to be compile-time controlled: per-depth logs only when the corresponding flag is compiled/enabled; debug logs only when the corresponding flag is compiled/enabled; fast base build logs only puzzle solved status, elapsed seconds, solution length, and solution path.
- User required configurable log frequency by depth and by puzzle.
- User required Kaggle interaction through Kaggle CLI with local proxy bypass.
- User requested retrying the puzzle 7 production run with beam `2**23=8388608` after diagnosing that beam `2**22` likely pruned the known solution prefix.
- User requested running puzzle 0 with beam `2**23=8388608`; current production-run context implies depth=200.
- User requested enabling production-run logging after every completed depth.
- User requested trying puzzle 0 with beam `2**24=16777216`, depth=200, and per-depth logs enabled.
- User identified that current config sizing was not consistently derived from `USER_GLOBAL_BEAM_WIDTH` and requested internal static buffers to be determined through `USER_GLOBAL_BEAM_WIDTH`, `WORLD_SIZE`, VRAM budget, and `B_MICRO`.
- User approved removing `reserved_width = GLOBAL_BEAM_WIDTH_EFFECTIVE - spill_reserve`; all periodic and final threshold computations must count strictly by `GLOBAL_BEAM_WIDTH_EFFECTIVE`.
- User requested diagnosing the final spill convergence failure by running with per-shard information, so the exact Stream3-to-Stream4 handoff failure location can be identified.
- User requested using Kaggle T4 testing while a local shared CUDA/Nsight/CUTLASS Docker image rebuild runs in parallel after the local Docker image cache was found empty.
- User confirmed that shard distribution skew should be fixed by replacing the distribution function with a more uniform deterministic mixer.
- User corrected the diagnosis: Stream3 batch was too large relative to Stream4 batch, so Stream3/Stream4 code should not change; configuration derivation must make Stream4 batch and Stream3 batch consistent from `USER_GLOBAL_BEAM_WIDTH`, `WORLD_SIZE`, and `B_MICRO`.
- User specified config derivation order: compute aligned `N_global`, compute `N_local`, determine mandatory final static layout size, use that as the expected stream-layout budget, derive `STREAM4_BATCH_CANDIDATES`, derive other static arrays from Stream4 batch, derive `SHARD_COUNT = N_local / STREAM4_BATCH_CANDIDATES`, set `STREAM3_BATCH_CANDIDATES = SHARD_COUNT * STREAM4_BATCH_CANDIDATES`, set `RING_SLOT_COUNT = STREAM3_BATCH_CANDIDATES / (B_MICRO * MOVE_COUNT)`, use `RING_COUNT=2`, and derive Stream5 batch arrays from `STREAM3_BATCH_CANDIDATES`.
- User clarified final tail policy: pre-final streams may produce equal-score tails, but the final stage must cut the tail and leave exactly the aligned beam target per load-balanced frontier, effectively `GLOBAL_BEAM_WIDTH_EFFECTIVE / WORLD_SIZE` states per card when enough candidates exist.
- User clarified that uniform distribution across GPU cards and across local shards is foundational for the whole solver and approved making `owner` and `shard` routing more uniform with deterministic mixing before modulo.
- User clarified final phase architecture: after Stream1/2 finish on all cards, the pipeline switches from throughput mode to final-local mode; each card locally dedups all pending candidates/spill first, then Stream5 coordinates candidate counts/global histogram, then global threshold and final load balance happen.
- User requested Kaggle diagnostics for the slow run: enable detailed logs and per-depth logs, keep config changes on Kaggle/notebook only without GitHub code push, and set beam width to `2**24`.
- User requested a Kaggle batch run for puzzles 1 through 20 with beam width `2**24`.
- User asked why the Kaggle run stopped around depth 48; diagnosis found no internal timeout, Kaggle host memory OOM, and `return_code=-9` because RAM history reached about `29.96GB`.
- User clarified dead-branch history must be cleaned: if a branch dies, its candidate history record is no longer needed.
- User requested three follow-up changes: discuss and raise neural-network output rounding precision, make CPU history cleanup independent from the main GPU loop, and add a configurable worker count.
- User requested setting `SCORE_SCALE=1024` despite alignment/memory concerns, then checking whether known solution paths for puzzles 1 through 20 survive beam pruning.
- User confirmed the final stage must be fixed after discovering that `final_threshold=UINT32_MAX` plus exact beam count meant final capping was preserving layout-first candidates rather than score-best candidates. User also asked why `final_threshold=UINT32_MAX` persisted at depths where `final_candidate_count=16820224`.
- User requested full tracked solution path information in logs, including score, location, route metadata, and enough detail to determine where and why the known path disappears.
- User requested continuing the search after diagnostics showed puzzle 9 `prefix_len=6` survived final filtering but `prefix_len=7` was missing before final filtering; next required diagnostic is generated-candidate logging immediately after Stream1/2 and before Stream3 thresholding.
- User approved adding Stream4-level tracking after `track_solution_generated` showed puzzle 9 depth 6 child was generated with `score_key=251278` but disappeared before prefinal with `final_threshold=113806`.
- User identified the main risk as output-index mapping: neural-network outputs `[0..23]` may be interpreted as moves `[0..23]` while the model order may differ from `p900.json`; user requested comparing CUDA and reference-forward on the problematic tracked parent for all 24 moves, including `q`, rank, and `move_name`.
- User requested detailed candidate path tracing on puzzle 0 using path `DL.BL.U.L.U.F.U.R.FR.U.BR.U.R.-BR.FL.FR.DR.-U.FR.F.R.-F.-BL.U.-DL.BR.B.-L.U.BR.U.L.BL.DL.BL.U.L.-BL.-L.-FR.U.F.U.U.D.-R.-U.-F.-FR.-DR.BR.DR.FR.R.-BR.-R.-FR.-BR`, with stopping two depths after the candidate first becomes missing.
- User requested fixing the Stream1 CUTLASS residual in-place GEMM issue without adding extra runtime checks, then verifying the fix.
- User required any Stream3/global-spill overflow or hidden data drop to produce an explicit error and stop execution; hidden loss of candidates is forbidden.
- User required removing the local uncommitted Stream3 spill backpressure patch and discussing the architecture-level solution before implementing any fix.
- User accepted the puzzle 0 trace diagnosis after Stream1 fix and stated the remaining issue: `Stream1=????????????????????????????????????`, path loss is `depth6/prefix7`, `score_key=22288 <= final_threshold=27866`, loss location is after Stream1/2 and before Stream4, likely Stream3 collector/global spill; requested checking generated-candidate writes through Stream3 output/local pending/global spill and final spill-drain at depth 6.
- User corrected the Stream3/Stream4 sizing model: `GLOBAL_BEAM_WIDTH_MAX_SAFE` must be removed, `GLOBAL_BEAM_WIDTH` is only aligned from the user beam, `STREAM3_BATCH_CANDIDATES = RING_SLOT_COUNT * B_MICRO * MOVE_COUNT`, `RING_COUNT = ceil(LOGICAL_SHARD_SIZE / (B_MICRO * MOVE_COUNT))`, and `SHARD_COUNT/STREAM4_BATCH_CANDIDATES` must be selected by a memory-budget config search.
- User clarified that Stream4 shard resident capacity must not mean logical shard size. Logical shards are larger than `STREAM4_BATCH_CANDIDATES`; Stream4 processes each logical shard in batches, with stream arrays sized by `SHARD_COUNT` and `STREAM4_BATCH_CANDIDATES`.
- User rejected an extra per-Stream3-launch `global_spill_free` runtime check because spill capacity should be guaranteed by architecture/config; the existing fatal out-of-bounds guard remains acceptable.
- User clarified Stream4 batch and shard count selection requirement: `STREAM4_BATCH_CANDIDATES` must be neither too large, causing memory/spill latency, nor too small, causing excessive initialization/job overhead; shard decomposition has the same balance requirement.
- User corrected the shard residency model again: every shard buffer is resident in VRAM and `physical_shard_capacity` must be derived from `LOGICAL_SHARD_SIZE`, not `2*STREAM4_BATCH_CANDIDATES`; add configurable shard-capacity multiplier for tail reserve, keep `STREAM4_BATCH_CANDIDATES` as the Stream4 launch/processing threshold, and derive spill capacity from `STREAM4_ACTIVE_SORT_SLOTS * STREAM3_BATCH_CANDIDATES` multiplied by a configurable spill multiplier.
- User clarified the spill formula must also multiply by Stream4 worker count if missing; using current config fields, Stream4 worker count is represented by `STREAM4_ACTIVE_SORT_SLOTS`.
- User requested per-stream speed measurements to derive a stable spill size and make the pipeline run confidently.
- User required a two-level debug model: master debug flag first, then independent speed/inference/path-trace debug flags; when master debug is off, subflags must not affect runtime and debug instrumentation must not be compiled into the production binary.
- User approved adding the required final request validation after a Kaggle path-trace run crashed in `cudaStreamSynchronize final materialize` with illegal memory access, then requested launching Kaggle with the validation enabled: "?????????????????, ???????????????????????? ???????????????????????? ????????????????????????????????????? ???? ????????????????????????????????? ????????????????".
- User diagnosed the post-validation Kaggle run as too slow without logs: full-beam depths used `stream3_jobs=2049`, `stream4_jobs~4647`, and depth time about `130s`; user asked why current sizing chose such bad `stream4_jobs`.
- User proposed deriving config from `stream4_jobs` and `stream3_jobs`, clarified that Stream3 receives `N_LOCAL * 24` generated candidates, and required batch execution time to be part of config reasoning, not just job count.
- User corrected implementation scope: runtime config and auto-detection must live in a separate config file; `production_runner` must not own the config-search implementation.
- User requested switching debugging from Kaggle to local Docker: use an existing image, create a container, test locally, and use available NVIDIA Nsight tooling inside the container.
- User requested supplementing the Docker image with CUTLASS and related tooling.
- User required Docker debug output to appear in Docker Desktop container logs, not only in Codex terminal output or redirected files.
- User requested disabling automatic config selection for now, setting the whole config manually, testing locally, and adding a short throughput meter for Stream3 batch time, Stream4 batch time, and spill growth before further design discussion.
- User requested replacing the shared global spill with per-shard resident double buffers: `survivor_shard_A[shard_capacity]` and `survivor_shard_B[shard_capacity]` for each logical shard, so Stream3 always has a shard-local write target and Stream4 alternates between A/B buffers.
- User specified the Stream4 scheduling split: choose `STREAM4_BATCH_CANDIDATES` by Stream4 job time, then use a separate `STREAM4_TRIGGER_CANDIDATES` so `stream4_job_time * stream4_job_count / active_sort_slots` stays well below Stream1/2 time.
- User requested a `beam=2**24`, `depth_limit=60` local Docker run without Nsight and without extra debug instrumentation, keeping only per-depth logs.
- User requested publishing the working solver to Kaggle so other people can use the notebook.

## 2026-05-26
- User required reverting all same-day experimental changes and restoring the repository to the state after commit `6d0ab88` (`Record Kaggle public release`).
- User confirmed the repository was restored and then requested a proper multi-GPU implementation with no new branch and with `WORLD_SIZE=1` continuing to use the current working single-GPU code path.
- User required `torchrun` or a Python launcher to only start multiple C++ processes. C++ must read `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`, call `cudaSetDevice(LOCAL_RANK)`, and create the NCCL communicator itself.
- User required Stream 3 after dedup to split candidates by `owner = hash % WORLD_SIZE`: local candidates go to local shard buffers, remote candidates go to Stream 5 send buffers.
- User required Stream 5 to exchange `CandidateMeta` between ranks, then collect received remote candidates into local shard buffers.
- User required finalization to perform local dedup, NCCL AllReduce histogram, global threshold selection, rank load balancing, `FinalRequest`/`FinalResponse` exchange, and construction of the next local frontier.
- User rejected forced disk history and shared `BEAM_HISTORY_RUN_ID` behavior. Multi-rank history must not invent mandatory disk persistence without approval.
- User required Stream 5 buffer sizes to be explicit in config, Stream 5 buffers to be segmented per card/ring to avoid collisions, and the receive buffer to have multiple parts so a writer always has a free target.
- User required large Stream 5 transactions because with many ranks most `CandidateMeta` records are remote and small network transactions would become the bottleneck.
- User approved the implementation plan and requested testing on the current Kaggle 2xT4 notebook, where only one GPU had previously been used.
- User approved pushing the multi-GPU code to GitHub and testing on Kaggle 2xT4.
- User observed Kaggle version 48 looked hung after build completion because notebook output stayed empty while C++ rank logs were redirected to files. User requested immediate log streaming from selected ranks through a config parameter, while still writing both rank logs to disk.
- User reported Kaggle version 50 still failed at depth 1 with `rank=1 what(): final request return rank exceeds WORLD_SIZE` and stopped the notebooks because the run was wrong.
- User rejected local fake `WORLD_SIZE=2` reproduction on one GPU with both ranks using `LOCAL_RANK=0`; future multi-rank validation must use real multi-GPU hardware such as Kaggle 2xT4 unless explicitly approved otherwise.
- User approved adding detailed final-exchange logging like other stream diagnostics and requested a compile/config flag so production runs do not compile or pay for the diagnostic code.
- User required no unapproved algorithmic changes while diagnosing the final exchange failure; current approved scope is detailed logging, full-debug Kaggle run, and investigation from produced logs.
- User approved the final exchange fix after reviewing the offset corruption: final exchange phases must have independent buffer layout plans, `FinalHistoryRecord` must be `align=32` and size multiple of `32/64`, and alignment must be respected for service buffers and any `ExchangePlan`-style helper.
- User required the final architecture to use three logical layouts over one scratch pool: layout 1 for Stream 1/2/3/4/5 inference, layout 2 for final dedup/threshold/selection, and layout 3 for next-frontier materialization.
- User required layout 3 to be a static chunked materialization layout with a 3-slot ring, events/streams conceptually separating materialization, transfers, and final writes, and alignment respected for every request/response/history buffer.
- User clarified CPU responsibilities in finalization: CPU receives only `CandidateMeta` for RAM history; CPU must not participate in GPU-GPU `FinalRequest`/`FinalResponse` communication.
- User required final materialization to reuse selected `CandidateMeta` chunks after request/history extraction, with GPU requests sent to source ranks, source ranks materializing responses, self-source/self-return paths staying local, and response slots pre-partitioned by `return_rank` to avoid atomic writes.
- User required no dynamic allocation in the final materialization path; static arrays, pointers, service buffers, and precomputed offsets are the required implementation style.
- User required implementing the layout-3 final materialization scheme while preserving `WORLD_SIZE=1` behavior exactly and reducing static allocation by relying on freed `CandidateMeta` chunks where possible.
- User rejected Kaggle source dataset transport for this validation and required pushing the current source to GitHub `main`, then launching the Kaggle T4x2 kernel from GitHub.
- User pasted Kaggle T4x2 debug run output where `rank=1` failed at depth 6 with `cuda stream fatal error: phase=stream3_remote_recv_collect flag=3002 code=3002 shard=7 group=3 existing=4135892 available=0 raw_count=24057 write_count=0 spill_count=24057 spill_capacity=0 clean_count=3988985 dirty_count=146907 processing_flag=1 shard_capacity_candidates=4194304 stream4_batch_candidates=196608 append_to_active_spill=1`; user requested reading project docs/recent work first, then discussing the failure cause.
- User stopped the backpressure implementation discussion and clarified that Stream3/remote backpressure is likely a crutch; user requested more diagnostics instead, especially to determine whether both A/B shard buffers are full or one is only processing, and requested enabling all existing logging flags in the code/notebook. User also clarified that synchronized threshold barriers are not wanted; each GPU should asynchronously update threshold using currently available peer data.
- User requested pushing the all-diagnostics build to Kaggle and watching logs to identify whether the current failure is due to insufficient Stream4 speed/capacity; if Stream4 is the bottleneck, user wants to speed Stream4 potentially at the expense of Stream3/Stream5, while preserving Stream1 throughput.
- User corrected the Stream3/4/5 architecture: Stream5 only owns inter-card staging buffers and never writes local Stream4 shard buffers; Stream3 consumes local Stream1/2 output plus Stream5-received candidates and writes Stream4 resident shard buffers. For each logical Stream4 shard, one physical A/B buffer may be processed by Stream4 while the other physical buffer remains writable for Stream3; Stream4 must never process both A/B buffers of the same logical shard at once.
- User requested Kaggle 2xT4 validation after the Stream4 A/B logical-shard mutex fix. User clarified that the local `production_runner 0 8 1048576` run validates the single-GPU branch only, that the A/B mutex must also apply in `WORLD_SIZE=1`, and that Stream3 writable-buffer backpressure may stay only in single-GPU mode but is strictly forbidden in multi-GPU mode because it masks capacity bugs during 100-GPU debugging.
- User requested another Kaggle T4x2 diagnostic run with all logs enabled and a short stream-time/buffer-size analysis before further architectural changes. User suggested that threshold may need to be updated continuously and requested avoiding threshold spam/noise for a future 100-GPU run.
- User finalized the threshold architecture: no new atomic operations are allowed; only Stream2 may use atomics. Threshold publication must be async-safe via double-buffered `current_threshold` plus active-index commit. Histogram snapshot must remain safe against parallel Stream4 through the existing per-shard histogram A/B scheme. Stream4/Stream3/Stream1/Stream2 remain local; Stream5 owns inter-rank communication. Stream4 exposes a local threshold request signal after enough processed work, Stream5 sees the signal, all ranks participate in the collective threshold update, the triggering and participating ranks reset their local counters, and `GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS` must not drive multi-rank threshold scheduling.
- User observed that Kaggle version 58 looked hung because logs stopped around 122 seconds while the run stayed active past 300 seconds, then requested detailed non-spam diagnostics for the 2xT4 run to identify the stall point.
- User stopped Kaggle version 59 after another stall/failure around 137 seconds and requested diagnosis from downloaded logs. The required focus is collective threshold ordering and pipeline state, not adding multi-rank Stream3 backpressure.
- User set a strict diagnostic operating rule: if a Kaggle run emits no logs for 120 seconds, stop the run and inspect downloaded logs. User forbids architecture/code changes without explicit approval; only config changes are allowed without approval when useful.
- User approved adding a separate debug flag to determine why final threshold histogram counts diverge from actual `survivor_shard + clean_count` final-selection counts.
- User approved adding a separate debug flag around Stream4 histogram A/B state after `results.zip` showed active histogram totals diverging from `clean_count` while inactive buffers often matched `clean_count`.
- User requested launching the Kaggle diagnostic run with auto-stop after 200 seconds.
- User approved fixing the Stream4 A/B scheduler invariant after v71 diagnostics confirmed duplicate physical-shard launches. User clarified that Stream4 must use the existing physical-shard `processing_flag`: if one physical A/B buffer of a logical shard is busy, Stream4 must not start the other physical buffer concurrently.
- User requested the next Kaggle T4x2 validation run to depth 16 after Kaggle version 72 completed depth 8 successfully.
- User identified that `beam=2**26` should fit on Kaggle T4x2 and requested fixing the manual Kaggle config because shard capacity was incorrectly derived from global beam width instead of per-rank local beam width. User then requested building/pushing the corrected config.
- User clarified that final phases 1, 2, and 3 must have separate layouts over one static scratch pool, not one monolithic final layout; requested implementing `scratch_pool_bytes = max(phase1, phase2, phase3)`.
- User asked why `global_spill_capacity` exists. Clarified project intent: `global_spill_capacity` is legacy overflow storage for single-buffer Stream4 mode; target A/B mode uses `SHARD_BUFFER_COUNT=2`, keeps one physical shard writable, and should use `GLOBAL_SPILL_CAPACITY=0` for multi-GPU.
- User clarified that resident Stream4 shard storage must always be A/B; `SHARD_BUFFER_COUNT=1` must not remain a valid runtime configuration. User requested launching Kaggle T4x2 validation for `beam=2**26`.
- User approved adding the missing diagnostics needed to determine the final histogram mismatch cause, with architecture changes explicitly forbidden. Required next action: add debug only, then test on Kaggle T4x2.
- User clarified the correct scratch-pool phase lifetime model after v76 diagnostics: phase 1, phase 2, and phase 3 share one static scratch pool, but phase 2 must receive pointers to the still-live phase-1 outputs and may reuse only dead phase-1 memory. Phase 2 must not overwrite live `survivor_shard`, `clean_count`, histograms, or threshold state. Phase 3 may reuse phase-1 live-input memory only after phase 2 has consumed those inputs, while preserving phase-2 selected-candidate outputs needed for materialization.
- User pasted depth 9-16 T4x2 runtime logs for `beam=2**26` with alternating full and small frontiers, then asked why some depths take about `45s` while neighboring depths take about `17s`.
- User supplied a compact runtime-diagnosis summary: slow depths are full-input-frontier depths, fast depths are small-input-frontier depths, CPU history and Stream4 are not primary causes for the alternating pattern, and the primary runtime driver is `current_frontier_size -> ring_slot_jobs -> stream3_jobs -> Stream1/2/3 work`.
- User clarified that depth-only logs should not materially slow the code because heavy debug trace flags are compiled off, then requested config-only speed tuning for Kaggle T4x2 through notebook parameters rather than architecture changes.
- User rejected the assumption that the full/small frontier alternation is normal, stated that the pattern is likely a bug, and requested adding debug only to determine the cause. Architecture changes are forbidden; code additions are allowed only for diagnostics.
- User stopped Kaggle version 79 manually and requested setting `RUN_TIMEOUT_SEC=300` in the Kaggle config.
- User approved fixing the Kaggle v80 diagnosis by adding the missing multi-rank final-reset cleanup for stale Stream4 histograms: "????????????????, ???????????? ???????????? ?????????. ????????????????, ????????????????????? ????????????????????? ????????????????????????????????????? ??????????????????????????????, ?????????????????????????????".
- User requested Kaggle T4x2 validation after the reset cleanup fix: "?????????????????, ??????????????????????????????????? ????????????????????????? ???????? 2?????????4 ????????????????????".
- User requested a Stream2 solved-neighborhood feature first, with correct naming instead of informal `K_1`: `BEAM_SOLVED_NEIGHBORHOOD_RADIUS`. Required behavior: CPU precomputes the central-state neighborhood by inverse moves, GPU stores only read-only hashes/fingerprints, Stream2 detects candidates that are within the configured radius from the solution, and CPU appends the matching suffix after history reconstruction.
- User requested preserving the current K2 discussion without implementing K2 yet. Future K2 means Stream2 descendant/suffix expansion from each generated candidate; K2 suffix generators should be precomputed and stored, not generated on the fly inside CUDA kernels.
- User requested testing the new Stream2 solved-neighborhood behavior on Kaggle T4x2 for puzzle IDs 1 through 10.
- User requested disabling unnecessary `depth_flow_trace` diagnostics because the trace pollutes Kaggle logs and slows runtime, then requested discussing the rational value for `BEAM_SOLVED_NEIGHBORHOOD_RADIUS`.
- User decided that `BEAM_SOLVED_NEIGHBORHOOD_RADIUS=4` should be reasonable and requested testing radius 4 on the same Kaggle T4x2 puzzle IDs `1..10`, then testing radius 5 on the same puzzle IDs.
- User approved implementing Stream2 generated-candidate suffix expansion after K1: keep `BEAM_SOLVED_NEIGHBORHOOD_RADIUS` unchanged, add `BEAM_STREAM2_SUFFIX_RADIUS`, add backend switch `BEAM_STREAM2_SUFFIX_BACKEND={base_generators,composed_permutations}`, store suffix chains precomputed on CPU, let Stream2 check `parent + move + suffix` against the K1 hash table or exact central state, store `solved_suffix_list` per solved hit, and reconstruct by appending K2 suffix first and K1 suffix second. User requested backend comparison for K2 after implementation.
- User changed the Kaggle K2 benchmark scope from easier puzzle ID `10`/short batch checks to the harder single puzzle ID `20`; K2 benchmark matrix should use `START_PUZZLE_ID=20` and `PUZZLE_COUNT=1`.
- User requested the missing `K2=2 composed_permutations` Kaggle T4x2 run and a comparison table for `K=0,1,2` across `base_generators` and `composed_permutations`, including whether `composed_permutations` slows depth runtime and by how much.
- User requested changing static history budget calculation to `beam_width * (depth - depth_reach_target_beam_width - K1 - K2) + states_before_reaching_target_beam_width`, because K1/K2 solved-neighborhood/suffix detection reduce the deepest history prefix that must be retained.
- User requested no source-code changes and asked for a config-only speed/memory sweep like the K2 table, especially varying `SHARD_COUNT`, `STREAM4_BATCH_CANDIDATES`, `STREAM3_RING_SLOTS`/Stream3 batch, and related pipeline parameters.
- User requested a larger Kaggle T4x2 config sweep with `BEAM_WIDTH=2**26+15_506_660`, `DEPTH_LIMIT=72`, shard-count testing through `1024` to find the shard-count plateau, and a separate check for `B_MICRO` plus parallel Stream1 model launches.
- User requested a follow-up Kaggle T4x2 sweep with `BEAM_WIDTH=2**26`, `B_MICRO=2048`, target Stream1 concurrency `4` from the isolated benchmark, and a search over `SHARD_COUNT` plus `STREAM4_ACTIVE_SORT_SLOTS`; user accepted testing `ring_slots=1` (`batch=49152`) and selected `ring_slots=2` (`batch=98304`) variants after clarification that `batch=98304` does not follow directly from `B_MICRO=2048`.
- User approved making Stream1 production concurrency configurable. Required implementation: expose `BEAM_STREAM1_CONCURRENCY`, enforce `STREAM1_CONCURRENCY <= STREAM3_RING_SLOTS`, and allocate separate Stream1 scratch buffers per concurrent lane so parallel inference launches cannot overwrite `hidden1/hidden2/residual/output`.
- User requested Kaggle T4x2 tuning for `BEAM_WIDTH=2**26` without the previous `+15_506_660` beam addition. Required fixed parameters: `B_MICRO=2048`, `STREAM1_CONCURRENCY=4`, `SOLVED_NEIGHBORHOOD_RADIUS=4`, K2 disabled. User suggested `STREAM4_TRIGGER_CANDIDATES=196608` and `STREAM3/STREAM4_BATCH_CANDIDATES=98304`; implementation must preserve the approved guard `STREAM1_CONCURRENCY <= STREAM3_RING_SLOTS`, so production `STREAM3_RING_SLOTS` must be at least `4` and Stream3 batch is at least `196608` for this sweep. Sweep target: find optimal `SHARD_COUNT` and Stream4 active sort slots.
- User requested launching Kaggle T4x2 for puzzle IDs `991..999` with `BEAM_WIDTH=2**26 + 15_506_660`, `DEPTH_LIMIT=72`, `RUN_TIMEOUT_SEC=0`, `HISTORY_MODE=static_hybrid`, `HISTORY_RAM_BYTES=28*1024**3`, `HISTORY_DISK_BYTES=59*1024**3`, `SOLVED_NEIGHBORHOOD_RADIUS=4`, K2 disabled, and the selected production concurrency config: `SHARD_COUNT=64`, `STREAM4_ACTIVE_SORT_SLOTS=4`, `B_MICRO=2048`, `STREAM1_CONCURRENCY=4`, `STREAM3_RING_SLOTS=4`, `STREAM4_BATCH_CANDIDATES=196608`, `STREAM4_TRIGGER_CANDIDATES=393216`.
- User requested Kaggle notebook-only fix after v122 disk quota failure: clean static history arena directories between puzzles, and add a new post-config/pre-compilation cell that computes whether the configured run can fit before compiling project code. User explicitly forbade project C++ code changes for this fix.
- User provided new model file `%USERPROFILE%\Downloads\Telegram Desktop\p900-t000-q_1779830329_best.pth` and requested comparison against the current project model: parameter count, architecture differences, runtime implications, and compatibility with the existing Stream1 CUDA runner.
- User requested implementing a universal MLP model interface where Stream1 model dimensions and residual block count are inferred from/exported with weights, required blocks are duplicated automatically, and the existing optimized CUTLASS inference code remains the execution path.
- User requested running the new model on Kaggle for puzzle ID `0` with the current large-beam parameters, approximately `2**26 + 16M` and specifically the notebook's current `BEAM_WIDTH=2**26+15_506_660`.
- User observed new-model puzzle-0 depth `7..8` runtimes around `108..111s` with `stream4_jobs????????959..1021`, compared those metrics against previous runs, and requested a Kaggle run with timing debug enabled to identify the bottleneck.
- User interrupted the attempted dispatcher timing-code investigation and explicitly required no dispatcher/source-code changes; requested only reducing the Kaggle diagnostic beam by `1,000,000`.
- User requested measuring Stream1 inference speed and Stream3 speed under the current T4x2 parameters, because Stream3 should not dominate the pipeline. Required Stream1 benchmark table: `B_MICRO=[2048,4096,8192,16384]` and concurrent inference count `[1,2,4,8]`.
- User accepted that current production bottleneck is Stream1 inference and requested launching puzzle ID `0` on Kaggle T4x2 with the previously working good config, without further optimization work, letting the run continue without timeout.
- User approved implementing support for Stream1 models with `output_dim=1`. Required behavior: infer mode from model `output_dim`, keep `B_MICRO` as CUTLASS row budget, compute `parents_per_job=floor(B_MICRO/24)`, generate child features inside the Stream1 input kernel without writing child `State128` records to VRAM, preserve candidate order `parent0_move0..23`, keep Stream2/3/4 downstream contracts unchanged, and use the existing CUTLASS path with `N=1`.
- User clarified the implementation constraint for `output_dim=1`: do not touch Stream2/3/4/5 logic. Adapt only Stream1 and Stream1-related runtime sizing so downstream streams continue to receive the unchanged `parents * 24` candidate layout.
- User requested a Kaggle push where the first config line is a variable for a model-parameter link. Requirement: user can paste a `.pth` URL/path into the notebook, and the notebook must export weights for either `output_dim=1` or `output_dim=24` automatically before running on T4x2.
- User rejected the guard requiring `STREAM4_BATCH_CANDIDATES >= STREAM3_BATCH_CANDIDATES`. Correct invariant: `STREAM3_BATCH_CANDIDATES` must fit into one physical shard buffer, so `STREAM3_BATCH_CANDIDATES <= SHARD_CAPACITY_CANDIDATES`; Stream4 batch may be smaller than Stream3 batch.
- User provided local `weights_megaminx2048_512_8_e4000.pth` and requested local Docker profiling for the same one-output model/config because Kaggle `beam=2**22` runtime looked much slower than expected. Goal: determine whether the slowdown is real Stream1 inference cost or a config/runtime issue.
- User provided Kaggle `output_dim=24` depth logs for `BEAM_WIDTH=2**22` showing full-depth runtimes around `24..27s` and requested comparing this against the one-output profiling because the 24-output model also appears much slower than expected.
- User redirected the speed investigation away from Docker and required Kaggle-only testing with `WORLD_SIZE=1`, multiple Stream1 models, and `BEAM_WIDTH=2**22`; historical target runtime is roughly `3..9s` per full depth, while current logs show much slower runtimes.
- User asked what prevents uploading and running a precompiled binary on Molab after the 100M/24-output GPU benchmark.
- User requested a native `sm_120` Molab build and, conditionally, a Stream1 improvement using Transformer Engine if Transformer Engine is available in the Molab environment.
- User requested running Molab beam search for puzzle `992` with about `100M` beam, the current `output_dim=24` model, and only per-depth debug logs.
- User requested diagnosing the Molab `puzzle=992`, `beam=100000000`, `output_dim=24` crash after the run reached depth 7 and failed during final materialization.
- User requested finding the exact source of the Molab survivor/final materialization corruption and fixing it.
- User requested extending the Molab Stream1 benchmark sweep with `STREAM1_CONCURRENCY=16,32` and `B_MICRO=32768,65536` as powers-of-two continuation, then using the best Stream1 point before shard tuning.
- User requested tuning Stream4 batch size, shard size, and shard count for the algorithm after selecting the best Stream1 `B_MICRO` and inference concurrency.
- User requested a Molab full run for puzzle `992` with `beam=260_000_000`, `depth=80`, `BEAM_SOLVED_NEIGHBORHOOD_RADIUS=5`, static-hybrid history, Stream3 prediction statistics enabled, and a separate notebook cell that plots the collected per-depth prediction statistics after the run.
- User requested a personal Codex plugin that should be used whenever an agent wants to work with Molab, and explicitly asked to use Plugin Eval for validation.
- User requested a Kaggle notebook for an H100/multi-GPU run that launches the existing working multi-GPU `production_runner` via `torchrun`, using global `beam=260_000_000`, `depth=80`, puzzle `992`, K1 radius `5`, static-hybrid history, and the provided Stream1/3/4/shard environment config.
- User clarified that torchrun GPU/node topology must be configured explicitly through `TORCHRUN_NPROC_PER_NODE` and `TORCHRUN_NNODES`; the notebook should derive world size from those torchrun settings, not auto-detect GPU count as the source of truth.
- User explicitly approved pushing the torchrun Kaggle notebook publicly, and required using only the multi-node/rendezvous torchrun launcher path while still working correctly for a single-node 2xT4 run.
- User requested not changing the H100 torchrun notebook further, and instead creating a second Kaggle notebook for validating torchrun machinery on 2xT4 using the provided `2**24` beam, puzzle `992`, depth `80`, `repo_output1` model path, static-hybrid history, and T4-friendly runtime parameters.
- User clarified that the T4 torchrun notebook should mirror the existing `cayley-beam-gpu-runner` style, but launch through torchrun and use only the repository 24-output model; the 1-output model/input path should not be used for this machinery check.
- User requested exactly two torchrun notebooks, one for H100 and one for 2xT4, with the same settings except CUDA architecture, using puzzle `0` and depth `10`.
- User requested cleaning only artifacts that are definitely not needed for work, explicitly excluding test scripts and working scripts.
- User noted that `test_results/` is about 25 GB and asked whether it can be cleaned.
- User requested reading project memory and creating a personal Codex plugin for Kaggle work, covering Kaggle CLI/notebook/kernel/dataset workflows, GPU validation runs, log monitoring, and project-specific safety rules.
- User requested implementing a personal Codex plugin `highload-gpu` / `HighloadGpu` with multiple skills for high-load GPU and multi-GPU system design, Python/Rust/C++/CUDA stack boundaries, NVIDIA documentation lookup, C++/CUDA/CUTLASS/CUB/NCCL hot paths, Nsight Systems, Nsight Compute, Compute Sanitizer, benchmarking sweeps, and remote Kaggle/Molab/H100/T4/Blackwell profiling. User required the full source guidance map with links to be embedded.
- User requested creating a personal Codex plugin for Docker usage, including Docker workflows for HighloadGpu/GPU development environments.
- User requested adding a dedicated Docker plugin skill that teaches Codex to avoid creating unnecessary duplicate images, reuse existing images/tags/cache, and make containers write useful logs to stdout/stderr for `docker logs` / Compose logs.
- User requested checking MEPhI FAQ before installing Ninja, then approved installing Ninja safely via a venv under `/mnt/pool/3/vokirova` and updating GitHub so the cluster `start.sh` can be refreshed from the repository before launch.
- User installed PyTorch and `nvidia-nccl-cu12` into the existing MEPhI `ninja-venv` and requested updating the GitHub launch script to use those NCCL include/library paths and venv Python for torchrun.
- User asked to inspect the published Kaggle notebook `trydotatwo/cayley-beam-gpu-runner` and requested making the MEPhI `start.sh` follow that notebook's runtime derivation, adding torchrun topology on top.
- User approved updating the MEPhI torchrun launcher with the Kaggle runner's `BEAM_NCCL_ID_FILE` rendezvous-file behavior after `ncclCommInitRank` failed.
- User requested that MEPhI cluster monitoring use the actual current job id automatically and that launcher logs only show rank 0 in the main output while other ranks write their own logs automatically.
- User requested replacing the single MEPhI 8xA100 launcher with a tuning workflow: script 1 benchmarks `B_MICRO` and Stream1 inference concurrency, script 2 sweeps shard count plus Stream3/Stream4 batch sizing through depth 12 for speed/crash behavior, and script 3 launches the best selected parameters.
- User confirmed the MEPhI A100 Stream1 sweep result should be used:
  `B_MICRO=8192`, `STREAM1_CONCURRENCY=8`, and `STREAM3_RING_SLOTS=8`; the
  ring-slot value should not be left at `4` because it invalidates the selected
  Stream1 concurrency.
- User requested a repeatable MEPhI script that first solves a puzzle from
  `data/test.csv`, then applies the found solution to the central solved state
  to create a reflected synthetic puzzle, solves that synthetic puzzle, and
  prints both solutions plus the inverted reflected solution as a candidate for
  the original puzzle.
- User requested adding script support for comparing pure solution lengths and
  showing deltas between an existing submission and newly found solutions.
- User requested making the Artgor `m_az_v4_v_only.pt` ResMLPDistance model
  runnable in the project's Stream1 runtime by adding LayerNorm-aware export
  and runtime support, using bf16 on modern GPUs and fp16 on T4, with no
  fallback CUDA path.
- User requested downloading the Artgor value-only model, exporting it for
  Stream1, pushing the exported weights to GitHub, and using that model for a
  MEPhI cluster run on puzzle `991`.
- User clarified that for one-output models `BEAM_B_MICRO` should be treated as
  a Stream1 row budget and divided by `24` to get the effective parent batch;
  otherwise the 8xA100 run overloads GPU buffers by treating `8192` as parent
  rows and materializing `8192 * 24` model rows.
- User requested a reflected-only MEPhI job with a 24-hour SLURM limit, using
  the already found puzzle `991` original solution instead of spending time
  solving the original again.
- User provided local IHES model file
  `C:/Users/???????????????? ??????????????????????????/Downloads/p888-t000_1778521793_e32692.pth` and requested
  launching the IHES cube solver with that model on MEPhI. The model is a
  BatchNorm-folded QMLP-style checkpoint with `input_dim=5184=72*72`,
  `hd1=2556`, `hd2=218`, `nrd=16`, and `output_dim=1`; IHES move count is 18,
  so one-output row budgeting must divide `BEAM_B_MICRO` by 18, not by the
  megaminx-specific 24.
- User forbade changing solver architecture without discussion, but allowed
  diagnostics. User asked to verify whether padding is zeroed everywhere and
  why the tracked IHES solution path reaches the frontier but is not detected
  as solved.
- User identified the IHES tracked-path mismatch as a script/data-path issue
  and requested pushing the corrected GitHub version plus cluster commands.
- User requested stopping the direct IHES length-24 jobs and rerunning those
  puzzles with both original and reflected solving, because the current
  `ihes24_*` jobs only solve the original state.
- User requested a clear project README explaining how to run the solver on a
  cluster, Kaggle, and similar environments; then clarified it should avoid
  specific MEPhI cluster rules and instead document generic 1-GPU and
  multi-GPU `torchrun` usage.
- User requested implementing segment repair for IHES solutions: provide custom
  start and target states to the runner, build K1 BFS from the target state,
  keep history/reconstruction unchanged so the runner returns `state_i ->
  state_j`, and splice `original_prefix + repaired_segment + original_suffix`.
  The first requested sweep is suffix/segment repair with size `K1 + max_depth`,
  where `max_depth` is the largest depth whose full frontier layer fits into the
  configured beam width.
- User clarified the desired cluster workflow: process the whole
  `solv_uniq.csv`, with one SLURM job per original solution. Inside each job,
  repair segments sequentially using block step `K1 + 7` for `beam=900M`, stitch
  the repaired segments, write all new paths and metadata, and optionally add
  diversity by testing windows `K1+1` through `K1+7`.
- User requested adding `solv_uniq.csv` to GitHub so the cluster can receive it
  through the normal repository update path instead of manual upload.
- User observed MEPhI rejecting repair arrays above index `999` and requested
  continuing the full `solv_uniq.csv` submission; the repair launcher should
  support chunk offsets so each submitted SLURM array can stay within
  `0-999`.
- User requested precompiled IHES repair jobs: compile the runner once and then
  vary only runtime start/target states so K1 is recomputed per segment without
  paying CMake/NVCC cost for every solution-repair task.
- User clarified that fallback behavior is not acceptable, but explicit modes
  are: keep the old repair behavior as a named mode and add the new plan mode
  as another named mode.
- User requested removing the repair diversity windows for now: generate only
  one segment grid with `window = K1 + 7` and run that.
- User requested a small purpose-built launcher for plan mode instead of
  `torchrun`, while keeping legacy mode on `torchrun`.
- User clarified the immediate request: do not broadly replace legacy behavior;
  add a runner without `torchrun` specifically for the current IHES repair task,
  and inspect current logs first.
- User requested a separate special repair mode where the beam-search runner
  itself reads `solv_uniq.csv`, creates the segment plan, processes every
  segment quickly, cleans per-segment state/history internally, saves all needed
  metadata to disk, and uses a stable device arena for solved-neighborhood so
  target K1 tables are overwritten without recreating CUDA graph templates.
- User clarified resident repair prefetch requirements: do not change K1 depth
  to 10; instead keep data for the next 10 queued segment tasks prepared in CPU
  RAM, whether they are windows from the current puzzle/solution or upcoming
  ones, so GPU execution can advance to the next task without waiting for K1
  CPU preparation. The buffer must be host RAM, not VRAM.
- User requested debugging why `K1 + 7` repair windows are reported as missed
  even though the full BFS frontier should fit in the configured beam width,
  and requested an explicit repair mode without model scoring where Stream1 is
  effectively disabled rather than using model scores or fallback behavior.
- User requested changing resident repair processing order so the run starts
  from high puzzle ids (`1000`, `999`, ...), instead of beginning at puzzle
  `0`, then `1`, and so on.

- User requested locally verifying why the reported IHES solve-bucket reflected improvement for puzzle 27 did not actually solve the puzzle, using the GitHub code/data. The required fix is to ensure reported solve-bucket improvements are real solutions for their corresponding original puzzle ids.

- User requested replacing the IHES solve-bucket workflow with a fresh original-plus-reflected search: choose puzzle ids whose solv_uniq.csv baseline length is 23 or 24, solve original from scratch while saving all bucket solutions, skip reflected if original improves, otherwise reflect from found original solutions sequentially until a shorter solution is found, and persist all found solutions for future analysis.

- User requested adding reflected deduplication for different paths that lead to the same state, and restart-safe progress so the current puzzle does not lose completed original/reflected work after a job restart.

- User clarified that PUZZLE_LIMIT=1 should be used for one-puzzle-per-array-task scheduling, but the overall run should cover many puzzles. User requested automatic publishing after each puzzle to https://github.com/TryDotAtwo/cayleypy-beam-results, with each puzzle's solutions and metadata saved in its own folder so other people can inspect and reuse the results.

- User clarified that all one-puzzle array jobs should use one shared precompiled beam-search binary instead of rebuilding production_runner in every job.

- User showed a completed IHES puzzle 33 run where compute succeeded but GitHub publishing failed because compute nodes could not resolve github.com and index generation crashed on metadata fields variants/source_files; requested continuing to collect and publish all cluster IHES results robustly.

- User requested Task 5 spec-fix in `D:\100XH100\.worktrees\stream1-piece-transformer`: make registered `stream1_transformer_cuda_tests` safe in clean checkout by skipping with a clear report line when ignored reference fixture files are absent, preserve the full p900 fixture test when present, document clean-checkout and fixture-present verification, run both paths if feasible, and amend the Task 5 commit without production dispatcher wiring or fallback backend.

- User requested Task 6 in `D:\100XH100\.worktrees\stream1-piece-transformer`: wire Stream1 piece-transformer into dispatcher and production runner with explicit MLP vs piece-transformer backend selection, no fallback backend, uniform mode preserved, transformer scratch-aware runtime sizing, MLP-only benchmark fail-closed unless explicitly wired, dispatcher-level transformer coverage if practical, clear production logs, verification note, and commit.
- User requested fixing P3 quality-review finding in `D:\100XH100\.worktrees\stream1-piece-transformer`: make `cuda/runtime_config.cpp` estimate piece-transformer `fast_slot_projected` as `max_piece_size * num_classes * d_model`, update `tests/dispatcher_cuda_tests.cu`, keep changes minimal with no fallbacks or architecture changes, run relevant checks including Docker GPU dispatcher tests if available, update project memory/test results, commit, and report SHA/tests.
- User requested a small portability fix in `D:\100XH100\.worktrees\stream1-piece-transformer`: replace POSIX-only `setenv`/`unsetenv` usage in `tests/dispatcher_cuda_tests.cu` runtime estimate test with a portable scoped environment helper, preserve restore/clear behavior on exceptions, avoid production changes, run a quick dispatcher CUDA test if available, and commit.
- User noted that the transformer Stream1 backend is far too slow compared with the MLP path: transformer should be roughly comparable and at most about 5x slower, not orders of magnitude slower. User asked to inspect how fast MLP runs and explain/fix the transformer backend without fallback or distillation.
- User clarified that the ~5x transformer-vs-MLP expectation is empirical from the PyTorch fast path, so the native Stream1 transformer backend should be treated as under-optimized rather than dismissed as intrinsically much heavier.
