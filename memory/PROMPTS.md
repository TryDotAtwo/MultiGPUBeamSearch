# Prompt History

## 2026-09-26 — Finish audited paper corrections and packaging

User requested actual synchronized EN/RU edits using EDITORIAL_AUDIT, READY_CORRECTIONS and MAIN_AGENT_PLAN: Cube4 57 tokens/input LN/H200, DeepCubeA weight, empty top-B, owner notation, scalar terminology/timing, separate run cohorts; improve layout without shrinking fonts or dropping evidence. Compile twice, inspect every page, run structural and publication tests, refresh QA/changelog/Prism ZIP/publication checksums. No subagents, Computer Use, local GPU runs or unsupported stronger claims; do not claim online Prism sync without verification.

## 2026-08-18 — One-click reviewer notebook

- User requested that Codex create a complete Kaggle notebook itself so a collaborator only needs to select GPU and press Run, rather than manually mounting a dataset and entering launcher code.

## 2026-08-18 — Dataset-mounted reviewer launch

- After collaborator feedback that another account's `Save Version` does not consume the owner's latest Quick Save draft, user requested storing the current notebooks in a Kaggle dataset so the latest dataset version is mounted during committed runs.

## 2026-08-18
- User requested a short Pilgrim scalar boundary run with one extrapolated threshold point and one higher point, to obtain a measured complete/OOM pair without fine-grained search.
- User requested two fresh reviewer-launchable Kaggle notebook versions through Quick Save so a CayleyPy collaborator can run the pending Pilgrim and MultiGPUBeamSearch measurements without consuming the owner's exhausted GPU quota.

## 2026-08-13
- User required two reviewer-facing private Kaggle notebooks, one for 1xT4 and one for 1xP100, each sweeping beam sizes internally for both Pilgrim and MultiGPUBeamSearch with genuinely native `output_dim=1` and native `output_dim=24` heads.
- User explicitly authorized uploading the compact private Kaggle bundle containing the Pilgrim runtime, worker scripts, generators, fixed benchmark state, and model metadata; checkpoints remain attached as separate Kaggle Models.
- After P100 runtime incompatibilities were verified, user narrowed continued remote testing to Kaggle 1xT4 only.
- User required the T4 work to be split into two independent notebooks: one notebook tests Pilgrim and the second tests MultiGPUBeamSearch.

- User explicitly authorized commits, pushes, deployment/resource operations,
  public notebook/results work, and other non-beam implementation decisions.
  The only retained approval boundary is changing the beam-search/CUDA architecture itself, which still requires prior user agreement.

## 2026-08-07
- User requested fixing the stuck promote-ingest flow by ensuring the `cayleypy-results/exact-candidate` check run creation in `.github/workflows/merge-ingest.yml` uses the actual `$candidate` SHA before PR merge.


## 2026-07-22
- User requested using the old GitHub-release IHES cluster workflow, not
  cluster-side Kaggle CLI, to run fresh solve-bucket original plus reflected
  searches for puzzles `1001` and `1002` with the Kaggle model
  `arabidopsisthalian/model-ihes-1780290207-e40960`.
- User requested pushing everything needed locally to GitHub first so the
  cluster can launch through the repository/release path as before.

## 2026-06-29
- User asked to inspect the Kaggle transformer inference notebook/model and add
  a separate Stream1 backend path for transformer weights. Requirements: do not
  touch or break the MLP path; automatically select the backend from the model
  weights provided by the user; generate the transformer manifest from the
  checkpoint/metadata instead of hard-coding dimensions.

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
- User explicitly prohibited fallback paths: "РЅРёРєР°РєРёС… С„РѕР»Р±РµРєРѕРІ"; fallback code is considered error masking.
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
- User requested connection of real data and neural network assets, with explicit reminder to preserve alignment: "РїРѕРґСЂСѓР±Р°Р№ СЂРёР°Р» РґР°С‚Р° Рё РїСЂРѕС‡РµРµ. РќРµР№СЂРѕРЅРєР° Рё РґР°С‚Р° Сѓ С‚РµР±СЏ РµСЃС‚СЊ. РќРµ Р·Р°Р±РёРІР°Р№ Рѕ Р°Р»Р°Р№РјРµРЅС‚Рµ".
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
- User approved implementing the Stream3-owned shard-ready handoff: "РќСѓ РґР°, РґРµР»Р°Р№".
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
- User accepted the puzzle 0 trace diagnosis after Stream1 fix and stated the remaining issue: `Stream1=РёСЃРїСЂР°РІР»РµРЅ`, path loss is `depth6/prefix7`, `score_key=22288 <= final_threshold=27866`, loss location is after Stream1/2 and before Stream4, likely Stream3 collector/global spill; requested checking generated-candidate writes through Stream3 output/local pending/global spill and final spill-drain at depth 6.
- User corrected the Stream3/Stream4 sizing model: `GLOBAL_BEAM_WIDTH_MAX_SAFE` must be removed, `GLOBAL_BEAM_WIDTH` is only aligned from the user beam, `STREAM3_BATCH_CANDIDATES = RING_SLOT_COUNT * B_MICRO * MOVE_COUNT`, `RING_COUNT = ceil(LOGICAL_SHARD_SIZE / (B_MICRO * MOVE_COUNT))`, and `SHARD_COUNT/STREAM4_BATCH_CANDIDATES` must be selected by a memory-budget config search.
- User clarified that Stream4 shard resident capacity must not mean logical shard size. Logical shards are larger than `STREAM4_BATCH_CANDIDATES`; Stream4 processes each logical shard in batches, with stream arrays sized by `SHARD_COUNT` and `STREAM4_BATCH_CANDIDATES`.
- User rejected an extra per-Stream3-launch `global_spill_free` runtime check because spill capacity should be guaranteed by architecture/config; the existing fatal out-of-bounds guard remains acceptable.
- User clarified Stream4 batch and shard count selection requirement: `STREAM4_BATCH_CANDIDATES` must be neither too large, causing memory/spill latency, nor too small, causing excessive initialization/job overhead; shard decomposition has the same balance requirement.
- User corrected the shard residency model again: every shard buffer is resident in VRAM and `physical_shard_capacity` must be derived from `LOGICAL_SHARD_SIZE`, not `2*STREAM4_BATCH_CANDIDATES`; add configurable shard-capacity multiplier for tail reserve, keep `STREAM4_BATCH_CANDIDATES` as the Stream4 launch/processing threshold, and derive spill capacity from `STREAM4_ACTIVE_SORT_SLOTS * STREAM3_BATCH_CANDIDATES` multiplied by a configurable spill multiplier.
- User clarified the spill formula must also multiply by Stream4 worker count if missing; using current config fields, Stream4 worker count is represented by `STREAM4_ACTIVE_SORT_SLOTS`.
- User requested per-stream speed measurements to derive a stable spill size and make the pipeline run confidently.
- User required a two-level debug model: master debug flag first, then independent speed/inference/path-trace debug flags; when master debug is off, subflags must not affect runtime and debug instrumentation must not be compiled into the production binary.
- User approved adding the required final request validation after a Kaggle path-trace run crashed in `cudaStreamSynchronize final materialize` with illegal memory access, then requested launching Kaggle with the validation enabled: "РћРєРµР№, РґРѕР±Р°РІСЊ РЅСѓР¶РЅСѓСЋ РІР°Р»РёРґР°С†РёСЋ Рё Р·Р°РїСѓСЃРєР°Р№ РєР°РіР»".
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
- User approved fixing the Kaggle v80 diagnosis by adding the missing multi-rank final-reset cleanup for stale Stream4 histograms: "РђР°Р°Р°, РІРѕРЅ РѕРЅРѕ С‡Рµ. РўР°РєСЃ, С‚РѕРіРґР° РґР°РІР°Р№ РґРѕР±Р°РІР»СЏС‚СЊ РѕС‡РёСЃС‚РєСѓ, РєРѕРЅРµС‡РЅРѕ".
- User requested Kaggle T4x2 validation after the reset cleanup fix: "РћРєРµР№, С‚РµСЃС‚РёСЂСѓР№ С‚РµРїРµСЂСЊ РЅР° 2С…Рў4 РєР°РіР»Р°".
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
- User observed new-model puzzle-0 depth `7..8` runtimes around `108..111s` with `stream4_jobsв‰€959..1021`, compared those metrics against previous runs, and requested a Kaggle run with timing debug enabled to identify the bottleneck.
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
  `C:/Users/РРІР°РЅ Р›РёС‚РІР°Рє/Downloads/p888-t000_1778521793_e32692.pth` and requested
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

- User requested adding an explicit separate Stream1 backend for the Kaggle
  Megaminx Q-transformer model, with no fallback behavior and without breaking
  the current MLP backend. The model must be analyzed carefully first; manifest
  fields should be generated automatically from checkpoint tensor shapes plus
  model/generator metadata. The target validation path is a small-beam 2xT4
  Kaggle run using the existing beam search after local correctness tests.
- User required final materialization exchange scale default to be derived as `WORLD_SIZE * 1000000` instead of hard-coded `8000000`, while preserving explicit env overrides.
- User requested allocating a couple of GiB of explicit scratch for 30M Megaminx solve-bucket gather instead of reducing `BEAM_SOLVED_RESULT_CAPACITY`, because the run should keep a large solution bucket.

- User requested persistent Kaggle iteration until real 2xT4 MultiGPUBeamSearch is tuned separately for output dimensions 1 and 24 and integrated into the cayleypy-rw-models2-tetraminx notebook.
- User requested reducing the roughly nine-hour 2xT4 MLP run time, tuning real
  2xT4 profiles for beam anchors `2**16` through `2**25`, and delivering one
  reusable Kaggle notebook. The notebook must support exactly the currently
  exportable MLP checkpoint families and both `output_dim=1` and
  `output_dim=move_count` modes, state those limits prominently, accept the
  user's exact beam width, select the nearest anchor with rounded `log2`, and
  preserve the requested beam apart from the solver's documented alignment.

- User required public CayleyPy result output to use exact long Kaggle competition slugs and legacy-style TSV plus metadata. Each competition index.tsv must contain every solution for every puzzle; each puzzle exposes all solutions and a separate best solution, with metadata stored separately.

- User requested fixing the failing GitHub publication of beam-search solutions and intermittent missing human-readable views, while preserving the existing Kaggle notebooks, ingest API, and zero-reconfiguration behavior for external submitters.

- User requested an iterative, maximally concise LaTeX paper and compiled PDF
  about the multi-GPU beam search. The literature review must use downloaded
  primary papers and expose claim-by-claim source evidence. The central system
  claim is that the search data plane runs exclusively on GPUs; CPU use is
  restricted to compressed history storage and lightweight orchestration. The
  headline experiment is a 700M-state Megaminx beam with very high throughput,
  and CayleyPy must be reviewed as the closest predecessor.
- User requires simultaneous Russian and English paper versions that match
  one-to-one.
## 2026-08-11 — Paper Section 3 threshold proof

- Clearly distinguish classical global top-k from the GPU-resident threshold
  scheme, including threshold updates.
- State and prove mathematically why the monotonically tightening
  distance-to-centre boundary returns the same global top-k.
- Keep the Russian and English LaTeX versions synchronized one-to-one.
## 2026-08-11 — Paper performance and result sections

- Add the next paper sections with tables for Kaggle, 8xA100, and local RTX
  3070 logs.
- Compare against CayleyPy published tables and explain the achieved scale.
- Include the 51-move Megaminx superflip result, with evidence for every number.
- Keep Russian and English versions synchronized.

## 2026-08-12 — Strict baseline matrix

- Do not discuss local optimization history. Report dry per-depth speed,
  maximum stable beam, peak memory, and failure boundary.
- Compare our beam search with CayleyPy, AlphaCube, and DeepCubeA on RTX 3070,
  one Kaggle T4, and one Kaggle P100.
- Do not mix different puzzles, models, search algorithms, microbenchmarks, or
  incomplete runs into one speedup claim.
## 2026-08-12 — exact benchmark parameters

> Да, делай. И везде указывай точные параметры, типа какая нейронка, ширина и прочее

- Report exact network architecture/checkpoint, search width or algorithm-
  appropriate batch parameters, dtype, move set, timing, node count/rate, and
  peak VRAM for every benchmark cell.
- Run Kaggle baselines on exactly one visible accelerator: 1×T4 first, then
  1×P100; test the local RTX 3070 last.

## 2026-08-12 — AlphaCube failure classification and next baselines

- State explicitly in the comparison table that AlphaCube's measured boundary
  is a host-memory SIGKILL, not a CUDA out-of-memory failure.
- Proceed with measured CayleyPy and native MultiGPUBeamSearch tests; preserve
  semantic and hardware incompatibilities instead of inventing paired cells.

## 2026-08-12 — memory/time benchmark objective

- Solution success and solution length are not primary endpoints for this
  comparison; evaluate algorithm memory and time on fixed work instead.
- Do not modify algorithms or substitute a common scorer.  Benchmark each
  released implementation as-is and form concise tables over its native scale
  parameter, time, throughput, host RAM, VRAM, and failure boundary.

## 2026-08-12 — paired Cube4 comparison

- After completing the AlphaCube/DeepCubeA block, compare CayleyPy and
  MultiGPUBeamSearch on 4x4x4 with the exact same neural checkpoint and states.
- Do not alter either search algorithm.  Representation-only checkpoint loaders
  or exporters are allowed only when they preserve the same scalar function and
  are numerically cross-validated before timing.
- Run the paired Cube4 comparison on exactly 1xT4 and exactly 1xP100, not 2xT4,
  so both implementations see identical hardware.
- Run every point through exactly depth 10 and report the time, throughput,
  frontier/candidate counts, host RAM, and VRAM observed specifically on the
  tenth layer; total time through depth 10 is secondary.
- One fixed deterministic Cube4 state is sufficient; do not run 50 states for
  this computational memory/speed comparison.
- Use a fixed Cube4 state with a known solution of approximately 20 moves, so
  the depth-10 timing run does not terminate early.
- 2026-08-13: Replace repeated one-point Kaggle kernels with exactly two private reviewer-facing notebooks: one fail-closed `1xT4` notebook and one fail-closed `1xP100` notebook. Each notebook must run both CayleyPy/Pilgrim and MultiGPUBeamSearch on the same hardware, test real `output_dim=1` and `output_dim=24` checkpoint contracts, sweep beam widths internally in isolated child processes, retain logs/JSON/CSV, reproduce the first failure, and report maximum completed separately from the first reproducible failure. Keep the local RTX 3070 paused.
- 2026-08-13: The output-1 and output-24 benchmark cells must use native, independently trained checkpoint heads. Do not average, repeat, truncate, expand, or otherwise convert one head into the other.
- 2026-08-13: User clarified that the paper benchmark must distinguish Pilgrim from the original native CayleyPy implementation, and requested checking native CayleyPy itself on 1xT4 rather than attributing Pilgrim's capacity boundary to CayleyPy.
- 2026-08-13: User required the native CayleyPy notebook to discover its memory limit adaptively: when a tested beam CUDA-OOMs, reduce the beam and search for the maximum width that still completes.
- 2026-08-13: User identified `alexandervc/cayleypy-rw-modelbaselines-megaminx` as the native CayleyPy Megaminx baseline and required Pilgrim Searcher, native CayleyPy, and MultiGPUBeamSearch to use the identical neural checkpoint so the experiment isolates search-algorithm efficiency.
- 2026-08-13: User fixed the final comparison roster at five algorithms: Pilgrim, native CayleyPy, MultiGPUBeamSearch, AlphaCube, and DeepCubeA. Native CayleyPy output-24 must be reported as unsupported/incompatible when the unchanged library cannot consume it.
- 2026-08-13: Upstream author supplied native CayleyPy latest working commit `5b2b53f`; use `pip install git+https://github.com/cayleypy/cayleypy.git@5b2b53f --no-deps` as the native-library baseline pin.
- 2026-08-13: User removed artificial per-point benchmark timeouts and set capacity depth to 8 because `24^8` exceeds the tested beam widths; tests must run honestly until completion, explicit failure/OOM, or the Kaggle platform limit.
- 2026-08-14: User requested adding the final native CayleyPy `history_depth=0` capacity evidence clearly to both one-to-one paper versions and delivering both compiled PDFs.
- 2026-08-14: User required native CayleyPy capacity testing with `history_depth=0`, matching MultiGPUBeamSearch; the earlier CayleyPy `history_depth=10` capacity boundary is not a comparable table result and must be replaced.
- 2026-08-14: User required the paper PDFs themselves, not only the benchmark matrix, to include explicit AlphaCube and DeepCubeA result tables.
- 2026-08-14: User required one clear 1xGPU comparison block with exactly four approaches—AlphaCube, DeepCubeA, native CayleyPy, and MultiGPUBeamSearch—showing each maximum measured frontier/native scale, speed at that point, all measured frontier-speed points, and a textual interpretation.
## 2026-08-14 -- Explicit seconds per search depth

- In the paper tables, report how many seconds one search depth takes, including
  the platform summary and matched Cube4 table.
- Make DeepCubeA speed explicit; because it is weighted A* rather than beam
  search, report seconds per native search iteration instead of inventing a beam
  depth.
## 2026-08-15 -- Quota-efficient Kaggle capacity confirmation

- Do not spend many hours finely bisecting the capacity boundary or stepping by
  1024 near a large frontier.
- Sweep every power of two from `2**10` through the first failure, recording
  search time and peak memory at every point. Then estimate the boundary from
  the observed memory growth and run exactly one confirmation point below it.
- Apply this protocol separately to native CayleyPy, Pilgrim, and
  MultiGPUBeamSearch on 1xT4. Test native output dimensions 1 and 24 for
  Pilgrim and MultiGPUBeamSearch; native CayleyPy supports only scalar output,
  so do not run a fabricated output-24 cell.
- After the broad Pilgrim sweep, use its measured VRAM trend to run a small
  dedicated 1xT4 boundary notebook near the predicted limit for both native
  heads. Add the maximum experimentally completed near-limit width and its
  time/memory telemetry to the comparison table.

## 2026-08-15 -- Compare only identical model contracts

- Compare performance only when checkpoint, architecture, output head, puzzle,
  state, move set, depth, beam width, dtype, and hardware are identical.
- Keep scalar `output_dim=1` comparisons (Pilgrim Searcher, native CayleyPy,
  MultiGPUBeamSearch) separate from native `output_dim=24` comparisons
  (Pilgrim QSearcher and MultiGPUBeamSearch).
- Do not attribute scalar-versus-Q timing differences to the search algorithm.
  Native CayleyPy output-24 remains incompatible, not transformed or estimated.

## 2026-08-15 -- Keep the Megaminx output-24 MLP

- Keep the already benchmarked native Megaminx output-24 QMLP2RB checkpoint
  (`p900-t000-q-sym_1777988767_best.pth`) for Pilgrim and
  MultiGPUBeamSearch. Do not replace it with the available Q Transformer; the
  Transformer is a separate model contract and is not part of this capacity
  sweep.

## 2026-08-15 -- Primary model is the eight-block scalar MLP

- Clarification: the model intended for the primary algorithm comparison is
  `weights_megaminx2048_512_8_e4000.pth`, with `hd1=2048`, `hd2=512`,
  `nrd=8`, and native `output_dim=1`.
- Use this byte-identical checkpoint for Pilgrim Searcher, native CayleyPy, and
  MultiGPUBeamSearch. The two-block output-24 QMLP is a separate secondary
  model family and must not replace or be mixed into the primary comparison.
- No native output-24 checkpoint with the same eight-block architecture is
  present in the inspected artifacts; do not synthesize one from scalar weights.

## 2026-08-17 -- Exact architecture disclosure and matched relaunch

- For every model used in the benchmark tables, report the exact trainable
  parameter count and the ordered sequence of neural blocks.
- Keep the existing eight-residual-block scalar Megaminx MLP as the primary
  model. For native output-24 use the existing QMLP2RB checkpoint, but only
  compare Pilgrim and MultiGPUBeamSearch when they load the byte-identical
  checkpoint.
- Relaunch only the corrected Kaggle notebook that still lacks a completed
  result; do not rerun already completed Pilgrim or native CayleyPy sweeps.

## 2026-08-17 -- Approximate narrow capacity brackets

- Replace powers-of-two-only capacity statements with approximate narrow
  experimental brackets like the native CayleyPy 20.97M/21.04M pair wherever
  the algorithm has a beam-width capacity parameter.
- Estimate from measured memory growth and use only a few broad confirmation
  probes near the predicted limit; do not perform fine binary search.

## 2026-08-18 -- Finish MultiGPU boundary after cancellation

- Diagnose why the corrected MultiGPU one-T4 run was cancelled and continue it.
- Narrow the remaining Pilgrim and MultiGPU capacity brackets with a few broad
  evidence-driven probes, not a fine search.

## 2026-08-18 -- Reviewer-launchable notebooks without execution

- Prepare the remaining Pilgrim and MultiGPU one-T4 boundary notebooks and
  upload them to Kaggle without starting a kernel, so other people can launch
  the runs manually.
- Deliver them as actual private Kaggle Notebook pages, not merely as `.ipynb`
  files inside a dataset, so a reviewer can open a link and press Run All.
- Ensure the notebook collaborators can read every private attached dataset and
  model; a visible input in the owner's editor is insufficient if the runner's
  account cannot mount it.
- Keep the `CayleyPy` private group at **Can edit** on shared benchmark assets.
  Publish the compact benchmark bundle so notebook runners do not depend on
  private-dataset mount permissions.

## 2026-08-18 -- Save-Version-only reviewer handoff

- Reviewers always launch benchmarks through `Save Version -> Save & Run All`,
  not by executing an individual cell or relying on an owner's Quick Save.
- Put every required runtime file, fixed input, manifest, and checkpoint in a
  Kaggle dataset attached to the committed notebook page.
- The reviewer should only need to select a T4 accelerator and save a version;
  no manual input attachment or code editing is acceptable.

## 2026-08-18 -- Run the remaining one-T4 boundaries

- Measure the pending Pilgrim scalar-output boundary and both native
  MultiGPUBeamSearch boundaries (scalar output and output-24) on one visible
  Kaggle T4 using the prepared self-contained notebooks.
- Keep the local RTX 3070 paused and classify completed, CUDA OOM, process exit,
  and platform cancellation separately.
## 2026-08-18 -- Isolated access to AlexanderVC GPU host

- Connect to the provided `AlexanderVC.conf` network without changing or
  disconnecting the user's AmneziaVPN setup.
- Keep the VPN route isolated from Windows; use it only to reach
  `192.168.0.97` and inspect/start the requested GPU workload.
- Do not repair, update, restart, or reconfigure the remote host while
  investigating it; first inventory the existing system and workload assets.
- 2026-08-18: For the AlexanderVC Tesla P4 Megaminx run, use the existing native beam-search implementation and take only model checkpoint/metadata from `zcloud_beam_bundle.zip`. Run original first and reflected afterward every time; collect as many solutions as available, continue two depths after the first hit, and choose a reasonable touch-BFS radius.
## 2026-08-19 -- Common per-step timing methodology

- Keep the primary cross-implementation metric consistent: divide clean total
  search wall time by the number of completed native algorithm steps.
- For the fixed Megaminx depth-8 sweeps this is total search wall divided by
  eight. Exact timing of the eighth/final depth may be retained only as
  auxiliary telemetry and must not replace the common average in comparison
  tables.

## 2026-08-24 -- Finish the paper benchmark evidence

- Continue the article work, provide the newest Russian and English PDFs, and
  identify the experiments that are still genuinely required.
- Complete the pending one-T4 scalar MultiGPUBeamSearch upper-boundary check
  without repeating already measured sweeps.

## 2026-08-25 -- Central paper claim and systematic prior-art audit

- Make the paper concise and centre it on one genuinely distributed search:
  one logical beam is physically sharded across GPUs, rather than represented
  by independent beams, batched queries, model replicas, or local surrogate
  top-k stages.
- State the headline scale precisely: the near-billion retained-frontier
  request is 770,883,178 states; with 24 Megaminx generators, one saturated
  depth has at least 18,501,196,272 nominal logical child candidates.
- Emphasize the conjunction of extreme frontier capacity, high measured
  throughput, a GPU-resident modular data plane, and mathematically exact
  global score top-B semantics. Keep capacity-only evidence and timed speed
  evidence separate.
- Search the known primary literature carefully and cite the strongest
  counterexamples. Use a world-first claim only for the complete, verified
  conjunction; do not claim that parallel, GPU, distributed, multi-GPU,
  expand--deduplicate--top-k, or billion-scale graph search is new by itself.
- Preserve the Megaminx-superflip 51-move replay as a separate application
  result, without an unqualified world-record or optimality claim.

## 2026-08-25 -- Exact headline object

- Make explicit that `770,883,178` is the requested number of retained raw
  states in one logically global beam, not a cumulative counter, dataset size,
  or batch of independent searches.
- For a saturated Megaminx depth, report the distinct scale quantity
  `770,883,178 * 24 = 18,501,196,272` as nominal logical parent--action child
  candidates before global de-duplication and top-B selection; do not call them
  distinct states or distinct neural forward calls.
- Centre the priority claim on the complete conjunction: one physically
  sharded global path-finding beam, GPU-resident hot-data path, global key
  reduction, and exact monolithic reduced-key score top-B under the same
  clipped integer score quantization and fixed-layout tie order.  Do not describe
  independent beams, model-parallel decoding, exhaustive traversal, or local
  surrogate selection as equivalent systems.
- Keep the downstream search architecture scorer-agnostic: Streams 2--5 accept
  deterministic quantized score keys from a learned or hand-engineered Stream
  1 implementation.  The shipped production Stream 1 backends and the reported
  experiments are neural; do not imply that arbitrary heuristics are currently
  runtime-pluggable without implementing that interface.

## 2026-08-25 -- Direct system-level priority and public artifacts

- Frame the contribution directly as the first published high-throughput
  multi-GPU beam-search system with the full claimed conjunction, rather than
  limiting the priority object to implicit combinatorial path finding.
- Keep the claim indivisible: one physically sharded global frontier, a
  GPU-resident hot-data path, global 128-bit-key reduction, no shard-local
  semantic caps, and exact monolithic reduced-key top-B under the stated score
  quantization, tie order, and identity model.
- Remove dated-review, ``to our knowledge'', and ``archived'' prose while
  retaining the exact evidence limits: eight-A100 capacity has no wall time or
  exact effective width, whereas measured throughput is supported separately
  through two T4 GPUs.
- Add public paper/source, implementation, release, Kaggle dataset, and Kaggle
  notebook URLs to matching English and Russian artifact-availability sections.

## 2026-08-25 -- Separate public paper repository

- Publish the final paper as a separate public GitHub repository rather than
  mixing it into the implementation repository.
- Include the exact English/Russian sources and PDFs, compact raw evidence,
  model/checkpoint manifests, public Kaggle notebooks, deterministic
  reproduction code, third-party notices, and verifiable release assets.
- Make all paper, implementation, Kaggle, and release links directly usable by
  a reviewer; do not depend on private saved Kaggle kernel pages.

## 2026-08-28 -- Personal plugin responsibility split

- Read the full project history and capture reusable GPU engineering lessons in
  personal Codex plugins.
- Keep `HighloadGpu` general. It may contain general SM120 / RTX PRO 6000
  Blackwell hardware guidance, profiling, CUDA, and throughput accounting, but
  not beam-search pipeline internals, neural-inference internals, or Molab
  operational details.
- Put beam/frontier/search pipeline details into a separate plugin, put MLP and
  Transformer inference/quantization details into another separate plugin, and
  keep all Molab pairing/session/execution/persistence rules in the Molab
  plugin.

## 2026-08-28 -- Exhaustive conversation and experiment systematization

- Re-read the current Codex task from its first message rather than relying on
  the recent summary or a quick keyword pass.
- Systematize the complete chronology: approaches tried, measured experiments,
  accepted results, failures, incorrect early conclusions, corrected models,
  dead ends, prepared-but-unverified branches, and remaining validation gates.
- Preserve the distinctions between compile proof, primitive microbenchmark,
  component benchmark, end-to-end run, task-quality validation, and production
  acceptance. Do not upgrade an idea or compiled binary into a runtime result.
- Record exact numbers where the conversation contains them, including the
  2xT4 `2**26` tuning lineage, exact-FP16 Transformer progression, SM120
  MXFP8/NVFP4 rooflines, accepted/rejected DSM and persistent-kernel seams, and
  Molab lifecycle failures.
- Write one detailed project report and route reusable knowledge into the four
  responsible personal plugins: general GPU engineering, GPU beam search, GPU
  neural inference, and Molab operations.
- Keep private Molab endpoints and credentials out of generated evidence and
  plugin documentation.

## 2026-08-28 -- Reassess Molab SM120 maximum-throughput direction

- Re-evaluate the Molab RTX PRO 6000 / SM120 Transformer task using the newly
  systematized project history and plugin runbooks.
- Consult relevant external GPU specialists, but validate generic advice
  against the project's measured Cube4 shapes and rejected experiments.
- Determine how to extract the maximum practical Tensor Core throughput from
  the card without confusing raw MMA peak, sparse marketing peak, component
  throughput, end-to-end Stream1 throughput, or solve-quality acceptance.
- Produce a concrete next-session plan with Amdahl/roofline accounting,
  profiling counters, safe Molab execution order, correctness and quality
  gates, and explicit stop conditions. Do not change CUDA/runtime code during
  this planning pass.
- Keep all target-hardware validation on Molab; local/Docker work may establish
  compilation only.

## 2026-08-31 -- CayleyPy upstream research for native beam integration

- User request: "Собери максимум информации о библиотеке CayleyPy. Клонируй репо сюда, посмотри че-каво. Будем изучать вопрос интеграции нашего бимсерча в библиотеку".
- Clone current canonical upstream in this workspace without replacing old benchmark copies or changing existing native worktrees.
- Investigate graph/state/move/model/search/BFS/multi-GPU contracts, current upstream PRs/releases/docs/research, and concrete compatibility with our native search.
- Preserve source-linked findings and reproducible verification artifacts. This turn researches integration; it does not implement or publish a backend.

## 2026-08-31 -- Clarify CayleyPy integration scope

- User clarified: the goal is convenient invocation of our native beam search from CayleyPy's graph/search API to investigate supported library graphs. Reproducing the upstream beam algorithm or supporting every arbitrary task/model is not required.
- Preserve our algorithm and expose its existing smaller-state/move-count specialization automatically. Determine support by representation/scorer capabilities, not a hardcoded puzzle-name whitelist.
- User prompt: "Смотри, произвольная задача пока недоступна. Суть в том, чтобы можно было наш бимсерч удобно из келипай библы запускать, исследуя те графы, которые библа позволяет исследовать. По сути настроить мой бимсерч на апи келипая или чета такое. Наш бимсерч сильно лучше того, что в келипае - вот в чем суть. И наш бимсерч умеет подстраиваться под меньший размер стейта нативно, насколько я знаю".

## 2026-08-31 -- Implement the optional CayleyPy native wrapper

- User confirmed the direction: "Ну вот, да, по сути мы пишем обвязку, чтобы пользователь юзал также удобно келипай как раньше, но уже был мой ахуенный бимсерч если возможно, а не текущий келипаевский".
- Implement convenient opt-in dispatch for the existing CayleyPy beam_search call, selecting our native algorithm when graph/model/runtime capabilities permit it.
- Keep the native beam algorithm and smaller-state specialization; unsupported calls may use the original CayleyPy search with an explicit reason. A failure after native selection is not permission to silently restart with another algorithm.
- Preserve original upstream and dirty native checkouts. Do not imply arbitrary model support, CUDA execution, throughput superiority measurements, external publication or upstream merge from CPU/API verification.

## 2026-08-31 -- Authorize real two-T4 CayleyPy adapter validation

- User: "Окей, ты можешь на 2хТ4 проверять чтобы всё работало. Задача позволить келипаю удобно использовать наш бимсерч без лишних изменений библиотеки. С минимумом вмешательства, но чтобы исследователи получили удобство келипая и скорость нашего бимсерча".
- Run actual two-T4 end-to-end validation, preserving the optional adapter and zero upstream source changes. Check both ranks, real native output and replay, then measure setup/orchestration overhead separately from solver timings.
- Use private Kaggle diagnostic runs through the ordinary configured CLI. Include only the task's adapter/test payload; fetch native and CayleyPy from explicit public commits instead of exporting the dirty native checkout. No public kernel or dataset publication is requested.

## 2026-08-31 - Public CayleyPy native backend PRs
- User requested a PR in cayleypy/cayleypy and TryDotAtwo/MultiGPUBeamSearch so other users can install and run the integration without local paths.
- Preserve CayleyPy convenience and native search behavior; use a minimal optional upstream backend hook and a separately installable adapter.
- User authorized MIT licensing and requested continuation without subagents.
## 2026-09-26 - Continue the Prism transfer
- User: "Призм на компе открыт" following "Перенеси статью в призм пж" and "В присм перенос продолжи". Complete the transfer in the open Prism project, retain both English and Russian canonical manuscripts with shared bibliography, and verify the resulting PDFs.
## 2026-09-26 - Paper author and eight-H200 runs
- User: "Ivan Litvak, independent resercher, vanya.litvak@gmail.com. По бимсерчу были запуски на 8хH200 - их найди и поставь". Use the corrected English affiliation "Independent researcher", add contact details to both paper versions, find primary eight-H200 run logs, and publish only measured depth/frontier/time/rate facts with explicit incomplete-solve and unmatched-scaling caveats.
## 2026-09-26 - Paired visual explanation of beam search
- Via the authorized paper-coordination conversation, the user requested a much clearer RU/EN paper with many useful diagrams, then explicitly approved the first three paired figures: one global beam across GPUs, conceptual overlap of five streams, and a numerical global de-duplication/top-3 example including boundary ties. Both languages must carry identical content, data, formulas, figure geometry/order, and caveats. Prism collaboration was requested; a 5.6 Sol / very-high setting may be used only if verified available in its UI.

## 2026-09-26 - Recover the interrupted paper side conversation
- User explicitly requested recovering the interrupted side chat, saving its work to files, composing a detailed continuation plan, and sending that plan to the main paper agent. Scope of this recovery: preserve available evidence and hand off; no concurrent paper edits or Prism operations. The recovered packet is `test_results/paper_side_recovery_2026-09-26/`; seven delivered messages are preserved, while a complete independent side transcript was not accessible.

## 2026-09-26 - Continue recovered paper editorial work
- User followed the recovery request with "Окей, продолжай работу". The side conversation continued paired EN/RU source review and full-page PDF inspection, keeping manuscript edits with the main paper thread and storing actionable corrections under `test_results/paper_side_recovery_2026-09-26/`. No Computer Use, subagents, local GPU work, or unrelated workspace changes.

## 2026-09-26 - Prism interaction boundary
- User explicitly said "Не используй компьютер юс пз. Иначе делай" after the Windows Computer Use helper stopped before accessing the already-open Prism project. Continue available local paper, verification, and packaging work, but do not operate Prism with Computer Use or claim online synchronization without a working permitted channel.

## 2026-09-26 - Deep audit of the current beam-search project

- User requested a deep correctness/performance/blocker audit and detailed implementation plan for the current MultiGPUBeamSearch project, aligned with the intended near-continuous native FP16 Stream1 pipeline, large global unique beam and bounded RAM/disk history.
- Latest clarification: "Просто отчёты сделай в конре и всё". Leave reports in the repository root; do not send them to another task. Work without subagents. Audit and local diagnostics do not authorize production fixes, new rentals or cloud spending.
- Deliverables: `AUDIT_2026-09-26.md`, `IMPLEMENTATION_PLAN_2026-09-26.md`; evidence and reproducible local checks under `test_results/deep_audit_2026-09-26/`. Distinguish CPU/source findings from historical H200 logs and unperformed target-hardware validation.

## 2026-09-28 - Cube555 notebook request
- User requested finding their Kaggle notebooks for other cubes and creating the same kind of notebook using Artgor's Cube555 model and Cube555 data. Source: https://www.kaggle.com/code/artgor/cayleypy-cube555-tpu-beam-q . User explicitly selected the existing MultiGPUBeamSearch 2xT4 route.
- Live notebook/checkpoint inspection found a public 120-element state contract, incompatible Cube555 transformer keys and no public blend configuration. Preparation and concrete integration scope: test_results/cube555_20260928/COMPATIBILITY.md. A runnable notebook has not yet been produced; no remote push or GPU run occurred.

## 2026-10-04 - Current-version 1/2/4/8 strong scaling
- User requests article-ready scaling on one node of identical GPUs using current working sources; fixed GLOBAL beam fitting one GPU, identical model/precision/task/initial states/seeds and fixed full steps without solution-triggered exits. Admit global selection/dedup/work distribution before timing. Warmup + at least three repeats per p; synchronized full-search time including history, separate initialization; work counts, median/dispersion/speedup/efficiency/peak memory, raw logs/commands/JSON and time/speedup plots. No algorithm changes or other experimental series. New paid rental requires approval. User confirmed MEPhI is currently unavailable.

## 2026-10-04 - Vast RTX 3060 scaling scope
- User: «Цель поставь себе. Локально ниче не пробуй. На vast на дешевых гпу тестить давай. 3060 1х, 2х, 4х, 8х и всё». Explicit goal created; no local builds/tests/search/benchmarks, only remote admission and the requested RTX 3060 scaling series. Use one 8-GPU node with nested subsets to preserve hardware comparison. New rental spending ceiling pending.

## 2026-10-04 - Authorized scaling rental
- User: «Даю 10 баксов, делай». Authorizes a total ceiling of USD 10 for the Vast RTX 3060 1/2/4/8 series. No local trials; no unrelated series.

2026-10-04 Ivan: Попробуй новую версию потестить. Continue remote-only RTX3060 admission using current Hopper worktree union, preserving original p900 inputs; no unrelated experiments.

2026-10-04 Ivan: Да, разрешаю. 20 баксов тебе. Explicit new Vast rental authority for same8xRTX3060 machine/new-version admission and scaling, USD20 ceiling.
2026-10-04 Ivan approved four scaling series: fixed global beam/common profile; fixed perGPU beam/common profile; fixed global beam/selected perconfiguration profiles; fixed perGPU beam/selected profiles. B=4194304, p=1/2/4/8, samep900 model/precision/seed/full12steps, warmup+3 repeats eachpoint. All raw artifacts directVasttoHF in new PUBLIC repository. No subagents/local computational trials. User: Супер, делай; Репо открытый делай. CurrentUSD20 authorization continues.
2026-10-04 Ivan: «Создай токен себе сам». Authorizes creating Hugging Face credentials for the already approved public artifact repository. Created a fine-grained token restricted to writing only this task's repository; transferred through SSH stdin to the authorized remote node, without logging its value.

## 2026-10-04 — Joint article integration with Astra
User explicitly requested contacting dot Astra and jointly integrating the new scaling data into the current article, including a clear conclusion that GPU execution load depends on program profile settings. Preserve independent Prism edits; identify and freeze current sources before integration. No new experiments or paid rental authorized by this editorial request.

2026-10-05: User supplied final EN PDF/source ZIP for beam-search arXiv submission, chose CC BY 4.0 and explicitly confirmed accepting arXiv Submission Agreement. Existing submission 8138889 used.

## 2026-10-10 - Cube555 GitHub-only notebook inputs
- User: «Сделай чтобы чисто через гитхаб работало, без кагл датасетов» for https://www.kaggle.com/code/trydotatwo/cube555-native-2xt4-blend.
- Preserve the current solver/configuration; replace attached model dataset and competition inputs with the existing public GitHub release bundle.

## 2026-10-10 results repository cleanup
User requested one branch in TryDotAtwo/cayleypy-beam-results and convenient browsing of all solutions by puzzle family, preserving the previous user experience and existing paths.

## 2026-10-27 - Kaggle 2.5M frontier timing
- User: «Замеряй 2.5 миллиона фронтир пж»; clarified «Именно на кагле». Run a bounded one-full-step measurement on Kaggle T4×2, no long search, and report measured seconds/step plus a clearly labeled 140-step linear projection. Evidence: beam-api-integration-20261025/test_results/kaggle_frontier_2p5m_20261027/REPORT.md.

2026-10-28 Ivan: «слушай давай запусти индекс 1000 для начала на луче 256 млн ...» (initial long run request superseded by clarification below); «Замеряй 2.5 миллиона фронтир пж», «Именно на кагле»; then requested separate Cube555 short (length about 10–12) and long (about 100) solutions using the reference 2×T4 blend notebook. Keep the 2.5M solves distinct from the separate 3.5M one-full-frontier capacity measurement. Evidence: test_results/kaggle_two_puzzles_20261028/REPORT.md.
2026-10-28 Ivan: verify whether the four long-run solution records were auto-published from Kaggle through Cloudflare, distinguishing notebook auto-publish from a local/manual/CLI submission and GitHub promotion.


2026-10-30 Ivan: «Запусти 1 ноут на фронтире 2.5 миллиона для пазла с длиной решения 10-12 где-то и один ноут с длиной решения 100 или около того. На каггле. Параметры как в ноуте [kamilla1158/cube555-native-2xt4-blend]»; уточнил: «А хрома расширение и прочее», «Не компьютер юз там». Использовать обычный Chrome/расширение для Kaggle UI, не встроенный браузер; не подменять один полный шаг запуском поиска на 140 шагов. Позднее установлен scope холодного acceptance: 2×T4, естественно вырастить до 2.5M, ровно один полный фронтир и остановка. Результаты/ограничения: test_results/kaggle_cold_current_20261030/REPORT.md.


2026-10-30 Ivan clarified the active goal: recall and finish the two Kaggle experiments he requested—2×T4 Cube555, requested beam 2.5M, one puzzle with a ~10–12 move solution and one ~100 moves, parameters based on kamilla1158/cube555-native-2xt4-blend. Use Chrome extension for Kaggle UI; no 140-step run requested. Audit results: test_results/kaggle_two_puzzles_20261027/TWO_RUNS_AUDIT_20261030.md.
