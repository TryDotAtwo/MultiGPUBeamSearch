# Project Memory

## 2026-05-15 bottleneck_short_latex_report

- prompt_summary: User requested short Russian code bottleneck analysis with proofs and LaTeX report.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- analysis_result: primary bottleneck is Stream2/Stream3 candidate pipeline: candidate materialization, hash/dedup/probing/atomics, fixed bucket NCCL exchange, histogram threshold, prune, hash clear/rebuild, and compaction; not the CUTLASS Stream1 GEMM scorer.
- proof_code_refs: `beam_engine.cpp:383-408` static CUTLASS scorer; `beam_engine.cpp:1107-1138` score-slot processing after each microbatch; `beam_kernels.cu:351-488` hash insert/update with probing/atomics; `beam_engine.cpp:972-986` NCCL all-to-all counts/payload; `beam_engine.cpp:989-1032` threshold/prune/clear/rebuild; `beam_engine.cpp:766-790` status waits all streams.
- proof_memory_refs: existing memory records note Stream2/3 can dominate, `DEPTH_TUNING` measures wall time/counters/tile counts, and prior conclusion says dominant prepass cost is hash clear, `k_work` scans, NCCL, histogram path, not neural GEMM.
- artifact: `docs/bottleneck_report.tex`.
- code_change_status: documentation/report only; no algorithm/runtime logic modified.

## 2026-05-14 depth_tuning_log_kaggle_kernel_metadata

- prompt_summary: User requested logs to pick optimal tuning parameters; noted Stream2/3 can dominate time and smaller B_MICRO with higher INFERENCE_PARALLELISM is often more efficient; asked GitHub push and Kaggle kernel test for cayleybeam-user-friendly-cpu-history.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`.
- source_patch_solver: `scripts/solve_testcsv_2gpu.py` adds `DEPTH_TUNING_LOG=1` path printing `DEPTH_TUNING` JSON per depth after step: `wall_ms_local`, `wall_ms_max_rank` (all_reduce MAX), `num_micro_batches`, `expand_tiles_upper_bound`, named counters, `tuning_params` snapshot; optional `cuda.synchronize` around step when flag on.
- source_patch_docs: `docs/DEBUG.md` documents `DEPTH_TUNING_LOG`.
- source_patch_notebooks: `notebooks/kaggle_user_friendly_cpu_history.ipynb` and `kaggle_user_friendly_kernel_stage/kaggle_user_friendly_cpu_history.ipynb` add `DEPTH_TUNING_LOG` and pass into `base_env`.
- source_patch_kaggle_stage: `kaggle_user_friendly_kernel_stage/kernel-metadata.json` added for `kaggle kernels push` targeting `trydotatwo/cayleybeam-user-friendly-cpu-history`.
- caveat: `wall_ms` merges Stream1+2+3 completion per rank; separate stream timings need Nsight; tuning log adds sync overhead — disable for final runs.
- remote_verification_status: `git push origin master` succeeded (commits through `23bcaac`); `kaggle kernels push -p kaggle_user_friendly_kernel_stage --accelerator NvidiaTeslaT4` succeeded as kernel version 8; progress URL `https://www.kaggle.com/trydotatwo/cayleybeam-user-friendly-cpu-history`.

## 2026-05-14 prepass_fill_phase_torchscript_opt_in

- prompt_summary: User requested two-phase search: first cheap beam widening and buffer fill while recording each depth for history compatibility, then full solver path; remove accidental TorchScript fallback so Kaggle uses CUTLASS static scorer by default.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`.
- source_patch_cpp: `beam_engine.cpp` adds `prepass_light_solved_scan_`, `set_prepass_light_solved_scan(bool)`, and caps `launch_check_current_solved` grid to `min(n_local, STATUS_CURRENT_SIZE)` when enabled (compacted frontier dense at indices `[0, current_size)`).
- source_patch_python_solver: `scripts/solve_testcsv_2gpu.py` enables light solved-scan during uniform prepass, uses `PREPASS_HISTOGRAM_PERIOD_MICRO` (default `1048576`) to reduce mid-depth histogram/NCCL/hash churn during uniform fill, optional `PREPASS_STOP_AT_WIDTH` / `PREPASS_STOP_WIDTH_FRAC` early exit from uniform when global `current_size` sum reaches target, emits `phase` and `PREPASS_WIDTH_REACHED` in logs, `export_scorer` requires `ALLOW_TORCHSCRIPT_SCORER=1`.
- source_patch_python_engine: `beam_engine.py` default `INFERENCE_BACKEND=fullbeamnice_static`; `configure_engine` raises on `torchscript_ensemble` unless `ALLOW_TORCHSCRIPT_SCORER=1`.
- source_patch_scripts: `scripts/run_local_2h100.sh` default backend `fullbeamnice_static`; `scripts/fullbeamnice_current_solver_2gpu.py` uses static scorer (drops export subprocess); `scripts/kaggle_correctness_check.py` sets `ALLOW_TORCHSCRIPT_SCORER=1` for the explicit TorchScript regression case; `kaggle_cpu_history_matrix_kernel_stage/cpu_history_shared_scorer_matrix.py` uses `fullbeamnice_static`; `docker-compose.2h100.yml` and status-submit notebooks use `fullbeamnice_static`.
- remaining_gap: per-depth cost still includes full `hash_capacity` clear at `clear_step_state_async` start and full `k_work` prune/compact grids; further prepass speed needs CUDA changes beyond solved-scan cap and histogram throttling.
- local_verification: `python -m py_compile` on edited Python files passed; C++ compile not run on Windows host (MSVC absent).
- code_change_status: implementation complete for scoped request.

## 2026-05-14 kaggle_silence_depth0_vs_prepass_cost

- prompt_summary: User asked why the run stays silent for tens of seconds after `DEPTH_RESULT` at `depth=0` with `prepass_depth=6`, GPU at 100%, and expected the first six prepass steps to be fast.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`.
- answer_fact: `DEPTH_RESULT` at `depth=0` is emitted before any `engine.step_current` / `engine.step` call in `scripts/solve_testcsv_2gpu.py` loop; `depth=0` iteration only runs `reset_search` plus `status`/`allreduce`; no beam expansion yet.
- answer_fact: `step_current` limits microbatches via `active_limit_override_` from `current_size`, so uniform-score forward work scales with the compacted frontier, not `n_local`.
- answer_fact: `enqueue_one_depth` still calls `launch_check_current_solved` with `cfg_.n_local` grid (`beam_kernels.cu` + `beam_engine.cpp`); each depth pays a full `n_local` scan.
- answer_fact: prune/final compact path uses kernels sized by `k_work` (`launch_prune_by_threshold`, `launch_rebuild_hash_from_active`, `launch_clear_step_state`); each depth pays full `k_work` passes.
- answer_fact: `launch_clear_hash_table` clears `hash_capacity` entries every threshold cycle and again after the microbatch loop; cost scales with static hash table size, not active frontier count.
- answer_fact: Python `engine.status()` synchronizes `stream_infer_`, all inference lane streams, `stream_ingest_`, `stream_net_` before each `DEPTH_RESULT`; host prints only after those syncs complete.
- answer_fact: if `BEAM_DEBUG=1` but `DEPTH_LOG_EVERY` defaults to `0`, `DEPTH_RESULT` is suppressed after `depth=0`; long `silent_for_sec` then means no periodic stdout, not necessarily a single stuck depth (user log may still use `DEPTH_LOG_EVERY=1`).
- relation: expected_fast_prepass_assumption → false; dominant_cost_per_depth → hash_clear + k_work_scan + n_local_check + NCCL + histogram path, not neural GEMM during uniform prepass.
- code_change_status: project memory update only; no algorithm/runtime logic modified.

## 2026-05-14 static_fullbeamnice_cutlass_prepass

- prompt_summary: User requested continuing implementation with a no-inference prepass before neural beam search, then a C++/CUDA FullBeamNice inference backend using static input/output buffers, FP16, Tensor Cores, CUDA Graph compatibility, and no PyTorch allocator inside the hot path; user additionally requested NVIDIA CUTLASS usage.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- model_inspection: current FullBeamNice model is fixed action24 `QMLP2RB` with `state_size=120`, `num_classes=120`, `hd1=1536`, `hd2=512`, `nrd=2`, and `23,978,008` parameters.
- model_layers: input is `LegacyCompatibleEmbeddingBagLinear` equivalent to summing 120 position-token embeddings from a folded `(14400,1536)` table; then folded BN+Linear `1536->512`; then two residual MLP blocks `512->512->512`; then output `512->24`.
- optimization_decision: BatchNorm is folded into weights/biases; embedding sum uses a custom CUDA kernel; GEMMs use CUTLASS FP16 Tensor Core GEMM; bias/ReLU/residual/quantization use fixed CUDA kernels; future improvement can fuse vector bias+ReLU into custom CUTLASS epilogues after correctness is stable.
- source_patch_static_loader: `scripts/static_fullbeamnice_inference.py` loads FullBeamNice weights, folds BN, emits FP16/FP32 static tensors, and validates static scores against the reference scorer.
- source_patch_engine: `beam_engine.py` allocates static FullBeamNice activation buffers only for `INFERENCE_BACKEND=fullbeamnice_static`, vendors CUTLASS include paths, and loads static FP16 weights into the C++ extension.
- source_patch_cpp_cuda: `beam_engine.cpp` adds `FullBeamNiceStaticBackend`, fixed tensor validation, no PyTorch forward in hot path, and uniform-score phase switching; `beam_kernels.cu` adds embedding, uniform score, bias/ReLU, residual/ReLU, CUTLASS GEMM, and quantize-to-score-ring launchers.
- source_patch_prepass: `scripts/solve_testcsv_2gpu.py` defaults to `INFERENCE_BACKEND=fullbeamnice_static`, skips TorchScript export unless explicitly requested, estimates `PREPASS_DEPTH` from `GLOBAL_BEAM_WIDTH` using 24 generators and `PREPASS_DEDUP_FACTOR=0.95`, runs prepass with uniform score before neural scoring, archives every prepass depth in CPU-history mode, and restores CUDA Graphs for the main scorer after prepass.
- source_patch_sizing: `scripts/t4_sizing.py` and `scripts/h100_sizing.py` include static FullBeamNice FP16 weights and activation buffers in memory estimates.
- dependency_change: NVIDIA CUTLASS `v3.5.1` vendored under `third_party/cutlass` as a source snapshot; nested `.git` metadata removed.
- local_verification: `python -m py_compile beam_engine.py scripts\solve_testcsv_2gpu.py scripts\t4_sizing.py scripts\h100_sizing.py scripts\static_fullbeamnice_inference.py` passed.
- local_verification: static FullBeamNice CPU FP32 compare passed with `max_abs_score_diff=1`; CUDA FP16 compare passed with `max_abs_score_diff=36`.
- local_compile_status: local PyTorch extension compile remains blocked by missing Windows MSVC `cl.exe`; `ninja` was installed successfully, but CUDA extension compilation still requires MSVC on Windows.
- sizing_result_2xt4_requested: with `GLOBAL_BEAM_WIDTH=81,000,000`, `WORLD_SIZE=2`, `B_MICRO=8192`, `K_EXPAND_TILE=16384`, `BETA=1.01`, `HASH_LOAD_FACTOR=0.45`, `HISTORY_BACKEND=cpu`, `INFERENCE_BACKEND=fullbeamnice_static`, total modeled static GPU buffers are `14.121 GiB` per rank with `0.879 GiB` modeled T4 headroom before CUDA/NCCL/runtime overhead.
- git_push_status: commits `4b894ae Add CUTLASS FullBeamNice static scorer` and `2cfb767 Set Kaggle debug config for static scorer` pushed to `origin/master`.
- kaggle_debug_config: user-friendly Kaggle notebook config set to `SAMPLE_START=1`, `SAMPLE_COUNT=1`, `GLOBAL_BEAM_WIDTH=81_000_000`, `B_MICRO=8192`, `BETA=1.01`, `MAX_DEPTH=60`, `HASH_LOAD_FACTOR=0.45`, `PROBE_LIMIT=128`, `HISTORY_BACKEND=cpu`, `CPU_HISTORY_CHECKPOINT=1`, `RESUME_BEAMSEARCH=1`, `RESUME_SUBMISSION=0`, `CPU_HISTORY_WORKERS=2`, `BEAM_DEBUG=1`, `DEPTH_LOG_EVERY=1`.
- remote_verification_status: Kaggle 2xT4 compile/runtime verification blocked because local Kaggle CLI now returns `401 Unauthorized` for `kaggle kernels list` and JSON parse failure for `kaggle kernels push`; no remote Kaggle run was stopped or killed.

## 2026-05-14 lane_safe_cutlass_fusion

- prompt_summary: User requested `[INFERENCE_PARALLELISM, B_MICRO, hidden]` activation buffers and fusion of `CUTLASS GEMM + bias + ReLU` plus final `GEMM + bias + action_perm + quantize`.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- source_patch_buffers: `beam_engine.py` now allocates `fb_act1`, `fb_act2`, `fb_act3`, and `fb_out` with leading lane dimension `INFERENCE_PARALLELISM`; `FullBeamNiceStaticBackend.forward(...)` receives `infer_lane` and uses lane-specific activation slices.
- source_patch_shared_weights: FullBeamNice FP16 weights remain one shared set per rank; multiple inference lanes share read-only weights and write only lane-private activation buffers.
- source_patch_cutlass_fusion: CUTLASS GEMM launcher now supports `LinearCombinationRelu` epilogue; hidden and residual linear layers prefill output buffers with bias or residual+bias, then CUTLASS performs `GEMM + C + ReLU` in the epilogue.
- source_patch_final_layer: final output buffer is prefilled with `out_bias`, then CUTLASS performs final `GEMM + bias`; `action_perm + quantize + score_ring layout write` remains a dedicated CUDA scatter kernel because score ring layout is action-major int16 and not a simple row-major FP16 CUTLASS output.
- source_patch_sizing: T4/H100 sizing scripts multiply static FullBeamNice activation buffers by `INFERENCE_PARALLELISM`.
- local_verification: `python -m py_compile beam_engine.py scripts\solve_testcsv_2gpu.py scripts\t4_sizing.py scripts\h100_sizing.py scripts\static_fullbeamnice_inference.py` passed.
- sizing_result_2xt4_requested: with `GLOBAL_BEAM_WIDTH=81,000,000`, `WORLD_SIZE=2`, `B_MICRO=8192`, `INFERENCE_PARALLELISM=2`, `BETA=1.01`, `HASH_LOAD_FACTOR=0.45`, `HISTORY_BACKEND=cpu`, `INFERENCE_BACKEND=fullbeamnice_static`, total modeled static GPU buffers are `14.161 GiB` per rank with `0.839 GiB` modeled T4 headroom before CUDA/NCCL/runtime overhead.
- local_compile_status: local PyTorch extension compile remains blocked by missing Windows MSVC `cl.exe`.

## 2026-05-14 kaggle_2xt4_static_cutlass_test_blocked

- prompt_summary: User requested Kaggle 2xT4 validation for static CUTLASS FullBeamNice backend with `GLOBAL_BEAM_WIDTH=81_000_000`, `SAMPLE_START=1`, `SAMPLE_COUNT=1`, `BETA=1.01`, `MAX_DEPTH=60`, `HASH_LOAD_FACTOR=0.45`, `PROBE_LIMIT=128`, `HISTORY_BACKEND=cpu`, `CPU_HISTORY_CHECKPOINT=1`, `RESUME_BEAMSEARCH=1`, `RESUME_SUBMISSION=0`, `CPU_HISTORY_WORKERS=2`, debug run `BEAM_DEBUG=1`, `DEPTH_LOG_EVERY=1`, then final run `BEAM_DEBUG=0`, `DEPTH_LOG_EVERY=0`.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- notebook_config_verified: `kaggle_user_friendly_kernel_stage/kaggle_user_friendly_cpu_history.ipynb` currently contains requested debug config and `INFERENCE_BACKEND='fullbeamnice_static'`, `INFERENCE_PARALLELISM=2`.
- local_kaggle_cli_check: `kaggle --version` returned `Kaggle CLI 2.0.2`.
- local_kaggle_auth_failure: `kaggle kernels list -m` returned `401 Client Error: Unauthorized`.
- local_kaggle_push_failure: `kaggle kernels push -p kaggle_user_friendly_kernel_stage --accelerator NvidiaTeslaT4` returned `Expecting value: line 1 column 1 (char 0)`, consistent with unauthorized/non-JSON API response.
- remote_verification_status: Kaggle debug and release runs not started because Kaggle API credentials are invalid/expired in the local environment.
- safety_note: no Kaggle kernel was stopped, canceled, interrupted, or killed.
- retry_2026_05_14: user requested another attempt; `kaggle kernels list -m` still returned `401 Client Error: Unauthorized`; `kaggle kernels push -p kaggle_user_friendly_kernel_stage --accelerator NvidiaTeslaT4` still returned `Expecting value: line 1 column 1`; no Kaggle run was started.

## 2026-05-14 prepass_runtime_gap_after_kaggle_v7

- prompt_summary: User pointed out that Kaggle v7 solved sample 1 at depth 1 but still took `1105s`, proving the no-inference prepass was not implemented as the intended cheap pre-main expansion.
- root_cause: prepass was logically using `UniformScoreBackend`, but `BeamEngine.step(...)` still iterated over `cfg.n_local=40,500,000`, cleared full hash table, and scanned full `k_work`, so the no-inference phase still paid huge full-beam static-buffer cost.
- source_patch_partial: `BeamEngine.step_current(...)` now limits microbatch/inference traversal to current compacted frontier size; `scripts/solve_testcsv_2gpu.py` uses `step_current(...)` during uniform prepass.
- remaining_gap: current patch reduces `n_local` inference traversal but does not yet remove full hash-table clear and full `k_work` prune/compact scans from prepass; a true fast prepass still needs a separate native no-hash/no-full-clear prepass path or touched-slot static bookkeeping.
- notebook_patch: Kaggle competition submit cell is commented out for debug runs.

## 2026-05-15 logical_capacity_prepass_fix

- prompt_summary: User clarified required design: separate physical allocated static capacity from logical active/next capacity; prepass kernels must launch over current logical frontier and logical next pool, not `n_local` / `k_work` / full `hash_capacity`.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- source_patch_cpp_api: `BeamEngine` now exposes `set_active_limit(uint64_t)`, `set_next_limit(uint64_t)`, and `clear_logical_limits()` through pybind; setters clamp limits to static allocations and invalidate CUDA Graph when capture shape changes.
- source_patch_cpp_hotpath: `enqueue_one_depth(...)` now computes `active_limit`, `next_limit`, `hash_limit`, and `current_output_cap`; prepass path uses these logical values for microbatch count, solved scan bound, hash clear, step-state clear, hash insert capacity, recv ingest capacity, prune, hash rebuild, current flag clear, and compaction output cap.
- source_patch_cuda: `launch_compact_next_to_current(...)` now launches `k_work` threads only; previous `max(k_work,n_local)` launch caused unnecessary `n_local` work when logical `k_work` was small.
- source_patch_reset: initial `reset_search(...)` solved scan now checks slot `0` only, because reset creates at most one active initial state per rank.
- source_patch_solver: `PREPASS_EXPECTED_CAPS` default added as `1,24,469,7779,104720,1334491`; uniform prepass sets `active_limit=last_local_frontier` and `next_limit=expected_cap_for_output_depth`; full solver clears logical limits before CUDA Graph/full static step.
- source_patch_notebook: user-friendly notebook exports `PREPASS_EXPECTED_CAPS`; debug Kaggle config uses `BEAM_DEBUG=1`, `DEPTH_LOG_EVERY=1`; submit command remains commented out.
- verification_local: `python -m py_compile scripts/solve_testcsv_2gpu.py beam_engine.py scripts/t4_sizing.py scripts/h100_sizing.py scripts/static_fullbeamnice_inference.py` passed; notebook JSON validation passed; `git diff --check` reported only CRLF normalization warning for Kaggle stage notebook.
- expected_effect: depth1 sample with one active state should no longer clear/scan full `40M`/`81M` buffers before finding one-move solution; allocated static buffers remain unchanged, logical work is reduced during prepass.
- git_status: local commit `1c3167a Add logical capacity limits for prepass` created after rebase onto remote commit `824295d`; unrelated user notebook edits were stashed during rebase and restored.
- remote_publication_status: `git push origin master` failed repeatedly with `Could not resolve host: github.com`; Kaggle validation was not started because Kaggle notebook clones GitHub `master` and would not include local commit `1c3167a` until publication succeeds.

## 2026-05-15 kaggle_iterative_validation_instrumentation

- prompt_summary: User requested implementation of the Kaggle iterative validation plan: publish local commits, run 2xT4 tests, validate `81M`, add TorchScript vs static CUTLASS Stream1 benchmark, add config tuning diagnostics, verify CPU-history checkpoint/resume, and review whether CUTLASS helps Stream2/Stream3.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- network_diagnosis: environment variables include bad proxy values `HTTP_PROXY=http://127.0.0.1:9`, `HTTPS_PROXY=http://127.0.0.1:9`, `ALL_PROXY=http://127.0.0.1:9`, `GIT_HTTP_PROXY=http://127.0.0.1:9`, `GIT_HTTPS_PROXY=http://127.0.0.1:9`; Kaggle CLI failed with proxy connection refused through `127.0.0.1:9`.
- source_patch_cpp: `BeamEngine.benchmark_inference(micro_size,repeats,warmup)` added; method uses existing static buffers, concurrent inference lanes, CUDA events, and does not alter beam-search hot path.
- source_patch_benchmark: `scripts/benchmark_inference_backends_2gpu.py` added; script compares `torchscript_ensemble` and `fullbeamnice_static` on 2xT4 with same `B_MICRO`, `INFERENCE_PARALLELISM`, warmup/iteration counts, and prints `STREAM1_BENCHMARK` JSON with speedup.
- source_patch_solver: `scripts/solve_testcsv_2gpu.py` prints `RESUME_BEAMSEARCH_RESTORED` when CPU-history checkpoint resume succeeds.
- source_patch_notebook: user-friendly Kaggle notebook adds `RUN_STREAM1_BENCHMARK`, `BENCH_ITERS`, `BENCH_WARMUP`, and optional `RUN_RESUME_CHECK`; benchmark runs before sizing/solver; resume smoke test can run before main solver; competition submit remains commented.
- source_patch_review: `docs/CUTLASS_STREAM2_STREAM3_REVIEW.md` records that Stream2/Stream3 are irregular hash/scatter/NCCL workloads where CUTLASS is not applicable; recommended optimization remains logical limits, memory coalescing, histogram cadence, bucket sizing, and hash/probe tuning.
- local_verification: `python -m py_compile scripts/solve_testcsv_2gpu.py scripts/benchmark_inference_backends_2gpu.py beam_engine.py scripts/t4_sizing.py scripts/h100_sizing.py scripts/static_fullbeamnice_inference.py` passed; both user-friendly notebooks JSON-parse.
- git_publication_status: `git -c http.curloptResolve=github.com:443:140.82.121.4 push origin master` succeeded for commits through `42f0dd4`.
- kaggle_v9_result: Kaggle kernel version 9 failed before benchmark because `/kaggle/input` was empty; root cause was `kaggle_user_friendly_kernel_stage/kernel-metadata.json` having `dataset_sources=[]` and `competition_sources=[]`.
- source_patch_v10_candidate: kernel metadata restores `competition_sources=["cayley-py-megaminx"]` and `dataset_sources=["trydotatwo/cayleybeam-fullbeamnice-project"]`; benchmark script now forces `GLOBAL_BEAM_WIDTH=BENCH_GLOBAL_BEAM_WIDTH` instead of inheriting the main `81M` beam.
- local_verification_v10_candidate: py_compile passed for benchmark/solver/engine/sizing/static-loader scripts; kernel metadata and both user-friendly notebooks JSON-parse.
- kaggle_v10_result: version 10 completed; benchmark printed `speedup_static_vs_torchscript=1.5013`; `81M` debug run solved sample `id=1` at depth `1` with `path=BR`; prepass depth1 step took `wall_ms_max_rank=93.021` and sample elapsed was `0.409s`; submit command remained commented.
- kaggle_v11_resume_result: version 11 restored checkpoint successfully (`RESUME_BEAMSEARCH_RESTORED depth=1`) and produced valid resumed path output, but subprocess timed out after `SUBMISSION_WRITTEN`; root cause is post-success distributed/process cleanup rather than checkpoint restore failure.
- source_patch_cleanup: `scripts/solve_testcsv_2gpu.py` now destroys the distributed process group and calls `os._exit(0)` after successful flush to avoid Kaggle torchrun hangs after resume/checkpoint runs.
- notebook_default_release: user-friendly notebooks default to release execution: `RUN_STREAM1_BENCHMARK=0`, `RUN_RESUME_CHECK=0`, `BEAM_DEBUG=0`, `DEPTH_LOG_EVERY=0`, `DEPTH_TUNING_LOG=0`; benchmark/resume cells remain available as opt-in diagnostics.

## 2026-05-14 user_friendly_kaggle_notebook

- prompt_summary: User requested a user-friendly Kaggle notebook with first-cell primary config, second-cell advanced config with comments/examples, metrics histogram cell, submit cell, competition input files from Kaggle competition input, code cloned from GitHub, separate custom scorer documentation cell, and Yandex Cloud TODO cell.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- implementation_scope: create `notebooks/kaggle_user_friendly_cpu_history.ipynb` and `kaggle_user_friendly_kernel_stage` for Kaggle testing; preserve solver hot-path code.
- notebook_config_primary: first cell contains `SAMPLE_START`, `SAMPLE_COUNT`, `GLOBAL_BEAM_WIDTH`, `B_MICRO`, `BETA` with Russian comments and examples.
- notebook_config_advanced: second cell contains `MAX_DEPTH`, `INFERENCE_PARALLELISM`, `K_EXPAND_TILE`, `SCORE_RING_DEPTH`, `NET_RING_DEPTH`, `BUCKET_CAP_PER_PEER`, `HASH_LOAD_FACTOR`, `PROBE_LIMIT`, `HISTORY_BACKEND`, checkpoint/resume flags, logging, timeout, GitHub clone config, model dataset hint, scorer initializer path, and submission path.
- notebook_data_source_change: notebook copies `puzzle_info.json`, `sample_submission.csv`, and `test.csv` from `/kaggle/input/cayley-py-megaminx` into cloned project `data/`; attached dataset is used only for `FullBeamNice` model files.
- notebook_code_source_change: notebook clones `https://github.com/TryDotAtwo/MultiGPUBeamSearch.git` branch `master` into `/kaggle/working/CayleyBeam100H100`.
- notebook_docs_cells: custom scorer cell documents `SCORER_INIT_PY`, `action24`, `action12`, `value1_after_move`, `heuristic24`, canonical `[B,24]` contract, and TODO for arbitrary generator count; Yandex Cloud cell is comments/TODO only and executes no cloud sync code.
- notebook_metrics_submit: metrics cell prints `total_count`, `unsolved_count`, `solved_percent`, `total_len`, all/solved length statistics, `solved_lengths`, and ASCII histogram; submit cell builds requested `submit_message` and runs `kaggle competitions submit`.
- verification_status: notebook JSON and stage metadata validate locally; GitHub push and Kaggle 2xT4 test pending.


## 2026-05-14 solution_found_check_cost_answer

- prompt_summary: User asked briefly whether the solution-found check takes much time.
- answer_fact: `kernel_check_current_solved` is a linear scan over the current frontier comparing active states to `central_state`.
- answer_fact: cost is memory-read bound and small relative to neural inference, candidate expansion, hash/dedup, NCCL exchange, thresholding, prune, and compaction for large depths.
- caveat: at very large frontier sizes the scan is not free, but expected dominant runtime remains inference plus Stream2/Stream3 candidate processing.
- code_change_status: project memory update only; no algorithm/runtime logic modified.

## 2026-05-14 solution_found_check_stage_answer

- prompt_summary: User asked briefly at what stage the solver checks whether a solution is found.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- inspected_code: `docs/ARCHITECTURE.md`, `beam_engine.cpp`, `beam_kernels.cu`.
- answer_fact: `reset_search(...)` checks the initial/current frontier with `launch_check_current_solved(...)` before the first expansion.
- answer_fact: each depth step checks the current frontier for `central_state` at the start of `enqueue_one_depth(...)`, then after prune/rebuild it compacts survivors via `launch_compact_next_to_current(...)`; only compacted survivors publish final found status for the next frontier.
- answer_fact: multi-GPU global found status is synchronized in `enqueue_found_allreduce_and_finish()` via NCCL `ncclAllReduce(..., ncclMax, ...)`.
- answer_fact: host loop `BeamEngine.search(...)` reads `status().found` before expanding the next depth and returns success at that depth.
- code_change_status: project memory update only; no algorithm/runtime logic modified.

## 2026-05-14 cpu_history_archive_shared_scorer_implementation

- prompt_summary: User approved implementing CPU-history archive, optional disk checkpoint/resume, shared single TorchScript scorer with multiple inference lanes, custom Python scorer initializer API, incremental `submission.csv`, and Kaggle verification matrix.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- implementation_constraints: preserve current 3-stream architecture; keep static arrays; default mode must preserve current GPU-history behavior; disabled debug/checkpoint/history code should be compiled out through extension variants where practical.
- design_lock: `HISTORY_BACKEND=gpu` keeps full-depth GPU transition history; `HISTORY_BACKEND=cpu` keeps one GPU transition layer and copies compacted transitions into preallocated CPU archive storage after each depth.
- design_lock: CPU archive reconstructs CPU frontier from transitions and rank-sharded CPU frontier files/checkpoints; exact resume requires a CPU-derived frontier checkpoint, not transition-only replay at restart time.
- design_lock: scorer export should produce one TorchScript file per rank; C++ should load one module and dispatch multiple inference lanes to the shared module when `INFERENCE_PARALLELISM>1`.
- verification_plan: Python syntax checks, sizing checks for GPU/CPU history, compile smoke where local CUDA toolchain permits, Kaggle CLI verification if available without stopping remote kernels.
- source_patch_python_control: `beam_engine.py` adds `HISTORY_BACKEND`, `CPU_HISTORY_CHECKPOINT`, compile-time extension variants `beam_engine_ext_h{gpu|cpu}_c{0|1}_d{0|1}`, and static history buffer sizing through `ext.derive_sizes(cfg)`.
- source_patch_cpp_cuda: `beam_engine.cpp` and `beam_kernels.cu` add `BEAM_HISTORY_CPU`, `BEAM_DEBUG_ON`, one-layer GPU transition history for CPU-history mode, and shared single-module TorchScript inference lanes when only one scorer path is loaded.
- source_patch_cpu_archive: `cpu_history_archive.py` adds preallocated CPU RAM transition arrays, pinned fixed-capacity host transfer slabs, rank-sharded frontier/transition checkpoint files, atomic manifest with `config_hash`, and resume upload of local frontier to `beam_current`.
- source_patch_solver: `scripts/solve_testcsv_2gpu.py` adds CPU-history reconstruction, checkpoint resume flow, `RESUME_SUBMISSION=1` skip of already written sample IDs, and continued per-sample `submission.csv` append/flush behavior.
- source_patch_scorer: `scripts/export_fullbeamnice_scorer.py` changes `--copies` semantics to inference lanes, adds `--physical-copies`, supports `SCORER_INIT_PY`, and emits canonical `[B,24]` higher-is-better TorchScript scores for `action24`, `action12`, `value1_after_move`, and `heuristic24`.
- source_patch_sizing: `scripts/t4_sizing.py` and `scripts/h100_sizing.py` add `HISTORY_BACKEND`; CPU-history mode sizes GPU transition history as one depth instead of `MAX_DEPTH` depths.
- local_verification: `python -m py_compile beam_engine.py cpu_history_archive.py scripts\solve_testcsv_2gpu.py scripts\export_fullbeamnice_scorer.py scripts\t4_sizing.py scripts\h100_sizing.py` passed.
- local_verification: CPU archive fake transition/checkpoint test passed; T4 sizing reports `HISTORY_BACKEND=cpu` total static buffers `0.696 GiB` vs `HISTORY_BACKEND=gpu` `0.764 GiB` for `GLOBAL_BEAM_WIDTH=262144`, `WORLD_SIZE=2`, `MAX_DEPTH=80`.
- local_cuda_compile_status: local extension compile is blocked by missing `ninja` in the local Python environment after fixing local temp/extension directory permissions; Kaggle compile remains required.
- kaggle_packaging_status: dataset stage and matrix kernel stage prepared for `trydotatwo/cayleybeam-fullbeamnice-project` and `trydotatwo/cayleybeam-cpu-history-shared-scorer-test`; remote verification still pending.
- kaggle_dataset_update: dataset `trydotatwo/cayleybeam-fullbeamnice-project` uploaded with updated CPU-history/shared-scorer sources; dataset file listing confirmed updated `cpu_history_archive.py`, `beam_engine.cpp`, `beam_engine.py`, `beam_kernels.cu`, solver/exporter scripts, and docs.
- kaggle_kernel_v1: first push without effective accelerator assignment ran on single P100 and failed with `invalid device ordinal`; no remote stop/cancel/kill command was used.
- kaggle_kernel_v2_v3: explicit `kaggle kernels push ... --accelerator NvidiaTeslaT4` assigned 2xTesla_T4; v2 passed checkpoint/resume smoke and failed first matrix with static hash overflow at depth 5; v3 passed checkpoint/resume smoke and `2**18,count20`, then failed `2**16,count50` with static hash overflow at depth 5.
- kaggle_capacity_test_adjustment: matrix kernel now uses `BETA=1.20` for `2**18,count20`, `BETA=32.0` for smaller-beam matrix cases, `HASH_LOAD_FACTOR=0.35`, and `PROBE_LIMIT=512`; static sizing remains below `0.343 GiB` for the largest adjusted smaller-beam case and below T4 memory limits.
- kaggle_kernel_v4_result: kernel `trydotatwo/cayleybeam-cpu-history-shared-scorer-test` version 4 completed on `2xTesla_T4`; no remote stop/cancel/kill command was used.
- kaggle_kernel_v4_passed: log contains `CPU_HISTORY_RESUME_SMOKE_OK` and `CPU_HISTORY_SHARED_SCORER_MATRIX_OK`; row-count checks passed for `submission_beam_2_18_count20.csv=20`, `submission_beam_2_16_count50.csv=50`, `submission_beam_2_14_count100.csv=100`, `submission_beam_2_12_count1001.csv=1001`, `resume_part1.csv=1`, `resume_part2.csv=1`.
- kaggle_kernel_v4_artifacts: outputs downloaded to `kaggle_outputs/cpu_history_shared_scorer_v4`; checkpoint manifest shows `sample_id=-1`, `depth=2`, `rank_count=2`, `n_local=8192`, `global_beam_width=16384`, `complete=true`.
- code_change_status: implementation complete locally; Kaggle 2xT4 matrix verification passed.

## 2026-05-13 kaggle_status_submit_append_fix

- prompt_summary: User asked to persist root-cause classification that Kaggle notebook is only an orchestration wrapper, persist CPU-history-archive architecture notes, test everything in a new Kaggle 2xT4 notebook, and make `submission.csv` update after every solved sample without hurting speed.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- root_cause_classification: `kaggle_notebook_fault_scope` is not notebook-only; notebook launches `scripts/solve_testcsv_2gpu.py`; likely runtime inconsistency is in `beam_engine.cpp`/`beam_kernels.cu`; notebook correctly stops on path-validation failure.
- architecture_note_persisted: `cpu_history_archive_design` is reasonable; GPU can keep only current compacted transition layer, copy compacted survivor transitions to CPU RAM after depth compaction, and CPU can maintain the current history graph without storing states.
- source_patch_cpp: `beam_engine.cpp` now clears `STATUS_LOCAL_FOUND` immediately before compaction together with `STATUS_FOUND`, `STATUS_FOUND_LOCAL_INDEX`, and `STATUS_FOUND_ACTION`, so only `kernel_compact_next_to_current` should publish local found for the compacted survivor layer.
- source_patch_solver: `scripts/solve_testcsv_2gpu.py` now all-reduces `local_found`, asserts global found has a local owner, rejects invalid owner index, optionally emits `DEPTH_RESULT`, initializes `submission.csv` at start, and appends/flushed one row after each solved sample when `SUBMISSION_APPEND_EACH=1`.
- notebook_added: `notebooks/kaggle_2xt4_status_submit_append_debug.ipynb` runs a 2xT4 known scramble test, checks status owner reporting, checks incremental append output, and asserts `STATUS_APPEND_TEST_OK`.
- local_verification: `python -m py_compile scripts\solve_testcsv_2gpu.py scripts\fullbeamnice_current_solver_2gpu.py beam_engine.py` passed; notebook JSON validation passed; T4 sizing for debug config reported `total_static_buffers_GiB=0.095`.
- kaggle_dataset_update: first dataset upload without `--dir-mode` skipped folders; corrected by uploading `trydotatwo/cayleybeam-fullbeamnice-project` again with `--dir-mode zip`, including `scripts`, `FullBeamNice`, `docs`, and `notebooks`.
- kaggle_kernel_run: pushed new private kernel `trydotatwo/cayleybeam-status-submit-2t4-debug` with `--accelerator NvidiaTeslaT4`; status reached `KernelWorkerStatus.COMPLETE`; no stop/cancel operation was used.
- kaggle_result: downloaded output to `kaggle_outputs/status_submit_2t4_v1`; log contains `DEPTH_RESULT` depth 2 with `found_sum=2`, `local_found_sum=1`, `bucket_overflow=0`, `hash_overflow=0`, `cuda_graph_captured_sum=2`; log contains `SAMPLE_RESULT` with `found=true`, `depth=2`, `path=-R.-U`; log contains `SUBMISSION_WRITTEN` with `append_each=true`; log contains `STATUS_APPEND_TEST_OK`.
- submission_result: `kaggle_outputs/status_submit_2t4_v1/submission.csv` contains header plus one row `-1,-R.-U`.
- code_change_status: runtime status fix, solver append behavior, and new Kaggle debug notebook added; CPU-history archive is documented as design direction, not implemented yet.

## 2026-05-13 path_validation_failed_local_found_stale_risk

- prompt_summary: User pasted Kaggle failure where samples 100 and 101 solved, then `scripts/solve_testcsv_2gpu.py` failed with `FATAL_EXIT type=AssertionError; message=path validation failed`, `variant=direct`, `distance=50`.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- inspected_code: `scripts/solve_testcsv_2gpu.py`, `scripts/fullbeamnice_current_solver_2gpu.py`, `beam_engine.cpp`, `beam_kernels.cu`, `beam_engine_common.hpp`, `beam_engine.py`.
- failure_interpretation: this is not a `BETA`/capacity failure; shown log contains path-validation failure after a reported found path, not bucket/hash overflow.
- likely_root_cause: `STATUS_LOCAL_FOUND` can remain stale because `clear_step_state_async` clears counters/hist/active/compacted/max-score but not `STATUS_LOCAL_FOUND`, and the pre-compaction `cudaMemsetAsync(beam_status + STATUS_FOUND, 0, 3*sizeof(int32_t))` clears only indices `[STATUS_FOUND, STATUS_FOUND_LOCAL_INDEX, STATUS_FOUND_ACTION]`, not `STATUS_LOCAL_FOUND` at index 7.
- consequence: `solve_testcsv_2gpu.py` chooses `found_rank` from gathered `local_found`; stale `local_found=1` can make a rank report `found_local_index` that does not point to the actual compacted central state, so `reconstruct_path(...)` walks a valid-looking but wrong history chain and CPU validation fails with nonzero distance.
- supporting_prior_signal: previous debug output pattern showed a rank with global found status and `found_local_index=0` where `state_is_central=false`, while another rank had the actual central state.
- fix_direction_requires_approval: clear `STATUS_LOCAL_FOUND` together with found fields at each step/reset boundary before compaction and ensure only `kernel_compact_next_to_current` publishes local found for a compacted central survivor.
- code_change_status: documentation memory update only; no algorithm/runtime logic modified.

## 2026-05-13 kaggle_kernel_stop_forbidden

- prompt_summary: User explicitly stated that Kaggle kernels must never be stopped under any circumstances.
- operational_rule: never stop, cancel, interrupt, kill, or otherwise terminate Kaggle kernels or Kaggle notebook runs.
- scope: all Kaggle kernels, Kaggle notebook runs, Kaggle CLI operations, and remote Kaggle execution monitoring.
- allowed_actions: read logs, read status, download outputs when permitted, analyze local artifacts, report diagnostics.
- forbidden_actions: `kaggle kernels stop`, UI stop/cancel, process termination of remote Kaggle runs, any equivalent stop/kill/cancel operation.
- priority: critical; this rule overrides diagnostic convenience and timeout cleanup preferences for remote Kaggle kernels.
- code_change_status: project memory update only; no algorithm/runtime logic modified.

## 2026-05-13 torchscript_copy_weight_check

- prompt_summary: User asked whether `--copies 2` creates real duplicate model weights in VRAM and requested a short answer based on code inspection.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- inspected_code: `scripts/export_fullbeamnice_scorer.py`, `scripts/solve_testcsv_2gpu.py`, `beam_engine.py`, `beam_engine.cpp`.
- answer_fact: `scripts/export_fullbeamnice_scorer.py` loops over `args.copies`; each iteration calls `build_model(...)`, loads the checkpoint with `torch.load(...)`, calls `model.load_state_dict(...)`, moves that model to CUDA, traces/freezes it, and saves a separate `.ts` file.
- answer_fact: `beam_engine.py` passes all scorer paths to `engine.load_torchscript_ensemble(paths)`.
- answer_fact: `beam_engine.cpp` stores `std::vector<torch::jit::Module> modules` and calls `torch::jit::load(path, device)` once per path; inference selects `modules[slot % modules.size()]`.
- conclusion: current code does not implement shared TorchScript weight storage across copies; `--copies 2` creates and loads two independent TorchScript modules, so two real weight copies in VRAM are expected per rank.
- code_change_status: documentation memory update only; no algorithm/runtime logic modified.

## 2026-05-13 fullbeamnice_model_size_estimate

- prompt_summary: User asked how much VRAM two current FullBeamNice TorchScript weight copies consume.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- inspected_files: `FullBeamNice/weights/p900-t000-q-sym_1777988767_best.pth`, `FullBeamNice/logs/model_p900-t000-q-sym_1777988767.json`, `runtime/**/*.ts`.
- measured_checkpoint_size: `p900-t000-q-sym_1777988767_best.pth` is `95,961,231` bytes, about `91.5 MiB`.
- model_parameter_count: `23,978,008`.
- vram_weight_estimate: fp16 weights are about `45.7 MiB` per copy and `91.5 MiB` for two copies; fp32 weights are about `91.5 MiB` per copy and `183 MiB` for two copies.
- caveat: TorchScript/CUDA runtime, allocator fragmentation, cuDNN/cuBLAS workspace, activations, graph capture, and duplicated module metadata add extra VRAM beyond raw weight tensors.
- code_change_status: documentation memory update only; no algorithm/runtime logic modified.

## 2026-05-13 inference_parallelism_memory_scope_check

- prompt_summary: User asked to verify in code whether two scorers only affect Stream1 and should occupy little memory.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- inspected_code: `beam_engine.cpp`, `beam_engine.py`, `scripts/t4_sizing.py`.
- answer_fact: `INFERENCE_PARALLELISM` is clamped to `score_ring_depth`, creates only `stream_infer_lanes_` CUDA streams/events at engine construction, and selects `infer_lane = mb % inference_parallelism` in `enqueue_one_depth`.
- answer_fact: static solver buffers are not multiplied by `INFERENCE_PARALLELISM`; `score_ring` size is `score_ring_depth * B_MICRO * fanout * 2`, independent of lane count.
- answer_fact: TorchScript backend selects `modules[slot % modules.size()]` and writes each result into the shared `score_ring`.
- caveat: `TorchScriptEnsembleBackend` stores `outputs_by_slot[slot] = y`, so up to `score_ring_depth` TorchScript output tensors can remain referenced; for `B_MICRO=32768`, `fanout=24`, `int16`, one output is about `1.5 MiB`, so this is not a multi-GiB item.
- conclusion: code supports the user's intuition that `INFERENCE_PARALLELISM=2` primarily affects Stream1 scheduling and scorer/module duplication; direct static memory delta is small, so large OOM is more likely from TorchScript/CUDA runtime behavior during export/load/capture rather than beam static arrays.
- code_change_status: documentation memory update only; no algorithm/runtime logic modified.

## 2026-05-13 beam_1e9_beta_1_01_memory_estimate

- prompt_summary: User asked to calculate static buffer memory for `GLOBAL_BEAM_WIDTH=1,000,000,000`, `BETA=1.01`, `WORLD_SIZE=2`.
- sizing_command: `GLOBAL_BEAM_WIDTH=1000000000 WORLD_SIZE=2 BETA=1.01 MAX_DEPTH=100 B_MICRO=32768 K_EXPAND_TILE=32768 SCORE_RING_DEPTH=8 NET_RING_DEPTH=2 BUCKET_CAP_PER_PEER=524288 HASH_LOAD_FACTOR=0.45 python scripts/t4_sizing.py`.
- derived: `N_LOCAL=500,000,000`, `K_KEEP=525,000,000`, `K_WORK=530,250,000`, `HASH_CAPACITY=1,178,333,333`.
- per_rank_static_buffers: `532,140,112,604` bytes, `495.594 GiB`, about `0.484 TiB`.
- two_rank_static_buffers_total: about `991.188 GiB`, about `0.968 TiB`.
- largest_per_rank_buffers: history total about `325.963 GiB`; state pools plus hash/meta about `166.059 GiB`.
- caveat: estimate excludes TorchScript weights/runtime, CUDA context, NCCL internals, inference outputs, allocator fragmentation, and workspace.
- code_change_status: documentation memory update only; no algorithm/runtime logic modified.

## 2026-05-13 max_beam_capacity_depth80

- prompt_summary: User asked how much beam fits on `100xH100` and `2xA100` when history depth is `80`.
- assumptions: `BETA=1.01`, `GAMMA=1.05`, `HASH_LOAD_FACTOR=0.45`, `MAX_DEPTH=80`, `B_MICRO=32768`, `K_EXPAND_TILE=32768`, `SCORE_RING_DEPTH=8`, `NET_RING_DEPTH=2`, `BUCKET_CAP_PER_PEER=524288`; estimates are static buffers only.
- result_100xH100_80GiB: maximum global beam about `5,670,292,900`; per-rank local beam `56,702,929`; per-rank static buffers exactly about `80.000 GiB`.
- result_2xA100_80GiB: maximum global beam about `184,665,850`; per-rank local beam `92,332,925`; per-rank static buffers exactly about `80.000 GiB`.
- result_2xA100_40GiB: maximum global beam about `91,591,580`; per-rank local beam `45,795,790`; per-rank static buffers exactly about `40.000 GiB`.
- caveat: practical beam should be lower because estimates exclude model/runtime, CUDA context, NCCL internals, allocator fragmentation, graph capture, and inference workspaces.
- code_change_status: documentation memory update only; no algorithm/runtime logic modified.

## 2026-05-11 algorithm_optimality_expand_cost_assessment

- prompt_summary: User asked whether current algorithm phases are logically optimal and whether neighbor expansion is expensive.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- inspected_code: `beam_kernels.cu`, `beam_engine.cpp`.
- assessment: current phase ordering is logical for correctness and static-array GPU residency: inference first, score-ring expansion, threshold filter, state materialization, hash/routing, distributed exchange, dedup, histogram threshold, prune/reclaim, hash rebuild, compact.
- cost_assessment: neighbor expansion is not the dominant neural cost; per candidate it performs 120-byte permutation copy, two 120-byte hash/fingerprint passes, possible 120-byte candidate storage or network bucket copy, and hash-table probing/atomics.
- likely_bottleneck_order: neural inference and hash/dedup memory traffic/atomics dominate more often than pure action permutation; remote candidate exchange can dominate at high multi-GPU scale.
- optimization_note: possible improvements include inverse-move masking, pre-hash/permutation fusion, stronger early thresholding, score-top candidate filtering before full state materialization, and better distributed load balancing; all are algorithmic/runtime changes requiring explicit approval.
- code_change_status: read-only inspection plus project memory update; no algorithm files modified.

## 2026-05-11 nn_scoring_dedup_explanation

- prompt_summary: User asked whether neural inference scores each neighbor separately or one inference returns scores for all 24 neighbors, and how score+state dedup/top-k works.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- inspected_code: `beam_engine.cpp`, `beam_kernels.cu`, `beam_engine.py`, `scripts/export_fullbeamnice_scorer.py`, `scripts/solve_testcsv_2gpu.py`.
- answer_fact: TorchScript scorer receives current states as `[micro_size,120]` and returns `[micro_size,fanout]`, where `fanout=24`; no separate neural forward per neighbor is used on the TorchScript FullBeamNice path.
- answer_fact: CUDA `kernel_process_score_slot` iterates candidate lanes after inference; each lane maps to `(parent_state, action)`, reads the precomputed score from `score_ring`, applies the action to materialize candidate state, hashes candidate state, routes by owner, deduplicates through the hash table, and updates/replaces metadata by score.
- answer_fact: top-k is threshold/histogram based; global score histogram computes `threshold_cell`, candidates with `score <= threshold` are filtered/pruned, and static free-list reclaim plus hash rebuild keep resident candidates bounded.
- code_change_status: read-only source inspection plus project memory update; no algorithm files modified.

## 2026-05-09 yandex_2xa100_container_push

- prompt_summary: User prepared Yandex Cloud 2xA100 VM workflow, requested Yandex Container Registry push, SSH preparation, InfiniBand/NCCL testing preparation, and exact README instructions for running the full Kaggle-equivalent beam instead of a smoke run.
- docker_base_change: Dockerfile base image changed from private/auth-required `nvcr.io/nvidia/pytorch:25.04-py3` to public `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel` because NGC returned `401 Unauthorized`.
- transformer_engine_check_change: hard build-time requirement for `transformer_engine` was removed; current FullBeamNice TorchScript solver path does not require Transformer Engine for the 2xA100 run.
- yandex_image_pushed: image `cr.yandex/crp7o66ucs8c14sjctp5/multigpu-beam-search:a100-kaggle-2t4-baseline` pushed successfully with digest `sha256:b43c12874691f886b5f54aadfa0cf269fca74e72deae1cc8417fb5a4f617184a`.
- ssh_key_preparation: `ssh-key-1778315981146.zip` was extracted locally into ignored `.ssh/`; private key content was not printed; ACL was restricted to current Windows user.
- deployment_docs_added: added `docs/YANDEX_2XA100_RUNBOOK.md` and `scripts/push_yandex_container.ps1`; README now includes exact Yandex 2xA100 pull/run commands and IB/RDMA diagnostics.
- full_beam_default: container default command remains `bash scripts/run_local_2h100.sh`; default runner uses `torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/solve_testcsv_2gpu.py` with Kaggle 2xT4-equivalent `GLOBAL_BEAM_WIDTH=65536`, `MAX_DEPTH=100`, CUDA Graphs, and FullBeamNice TorchScript scorer.
- no_algorithm_files_modified: true.

## 2026-05-08 docker_github_packaging

- prompt_summary: User requested Docker packaging for easy SSH/VM launch of the current beam-search solver, inspection of Kaggle launch behavior first, assessment of 2xA100 changes, and GitHub publication for cluster use.
- kaggle_launch_contract_observed: successful Kaggle `test.csv` path uses `torchrun --standalone --nproc_per_node=2 scripts/solve_testcsv_2gpu.py`, `USE_CUDA_GRAPHS=1`, `INFERENCE_BACKEND=torchscript_ensemble`, `GLOBAL_BEAM_WIDTH=65536`, `MAX_DEPTH=100`, `INFERENCE_PARALLELISM=1`, `B_MICRO=32768`, `K_EXPAND_TILE=32768`, `SCORE_RING_DEPTH=8`, `BETA=1.20`, `HASH_LOAD_FACTOR=0.45`, and `PROBE_LIMIT=256`.
- packaging_changes: Docker default command now runs `scripts/run_local_2h100.sh`; `scripts/run_local_2h100.sh` now defaults to the real `scripts/solve_testcsv_2gpu.py` two-GPU solver instead of `beam_engine.py`; `docker-compose.2h100.yml` mirrors the Kaggle-tested small production defaults.
- a100_assessment: no algorithm changes are required for 2xA100; deployment needs CUDA arch `sm80` support, so Dockerfile now uses `TORCH_CUDA_ARCH_LIST="8.0;9.0"`; NCCL settings may need host-specific adjustment (`NCCL_IB_DISABLE=1` for no IB, `NCCL_IB_DISABLE=0` for healthy IB/RDMA, optionally omit fixed `NCCL_P2P_LEVEL` if NCCL auto-detection is better on A100 PCIe).
- public_repo_safety: `.dockerignore` and `.gitignore` now exclude `kaggle.json`, Kaggle staging/output folders, runtime generated files, build outputs, and Python caches.
- documentation_changes: added root `README.md`; expanded `docs/DEPLOY_DOCKER_SLURM.md` with SSH Docker quickstart, smoke commands, default solver settings, and 2xA100 notes.
- local_verification: `scripts/h100_sizing.py` for the default `2xGPU`, `GLOBAL_BEAM_WIDTH=65536`, `MAX_DEPTH=100` config reports `total_static_buffers_GiB=0.163` per rank, excluding model/runtime overhead.
- docker_build_status: attempted `docker build -t cayley-beam-h100:latest .`; build failed while pulling `nvcr.io/nvidia/pytorch:25.04-py3` with `short read ... unexpected EOF`; failure occurred before project source compile and appears to be external registry/network transfer failure.
- github_status: local git repository initialized; GitHub CLI account `TryDotAtwo` has invalid keyring token and API access also failed through proxy `127.0.0.1:9`, so remote GitHub publication is blocked until authentication/network are fixed.
- no_algorithm_files_modified: true.

## 2026-05-07 testcsv_2_16_restart_no_timeout

- prompt_summary: User requested restarting the `test.csv` beam `2**16` run with no wrapper timeout and progress logs every 25 samples including solved paths.
- correctness_status_before_restart: Kaggle version 17 confirmed known depth-2 scramble `U,R` with `found=true`, `depth=2`, `path_len=2`, and `cuda_graph_captured_sum=2`; static history and CUDA graph worked together.
- benchmark_result: one depth-2 benchmark selected `INFERENCE_PARALLELISM=1`, `B_MICRO=32768`, `K_EXPAND_TILE=32768`, `SCORE_RING_DEPTH=8`; best measured row was `p1_b32768` with `elapsed_sec=17.256`.
- source_patch_logging: `scripts/solve_testcsv_2gpu.py` now includes `path` in each `SAMPLE_RESULT` progress log.
- notebook_patch_runtime: Kaggle notebook wrapper treats `TIMEOUT_SEC=0` as no timeout and defaults `LOG_EVERY=25` for the production `smoke_2_16` mode.
- kaggle_dataset_version: uploaded to `trydotatwo/cayleybeam-fullbeamnice-project` with message `progress logging every 25 samples and no timeout`.
- kaggle_kernel_restart: pushed `trydotatwo/cayleybeam-fullbeamnice-2t4-test` version 20 with `RUN_MODE=smoke_2_16`, `GLOBAL_BEAM_WIDTH=65536`, `MAX_DEPTH=100`, all `test.csv` rows, CUDA graph enabled, and path validation enabled.
- remote_status: completed successfully; output downloaded to `kaggle_outputs/fullbeamnice_2_16_v20/submission.csv`.
- submission_result: `rows=1001`, `nonempty_paths=385`, `empty_paths=616`, `max_path_len=100`, Kaggle log marker `SUBMISSION_WRITTEN` reported `elapsed_sec=18204.819`.

## 2026-05-06 static_path_history_fullbeamnice_runner

- prompt_summary: User requested static path history so the current distributed solver can reconstruct the found path when using the FullBeamNice model.
- approved_change_scope: add history storage and path reconstruction; preserve Stream1/Stream2/Stream3 search logic; keep history only after pruning/compaction; keep GPU arrays statically preallocated before search.
- source_patch_cpp_cuda: `kernel_compact_next_to_current` now writes survivor-only history records into static arrays `history_parent_idx/history_parent_rank/history_action/history_valid` at `history_depth*n_local+compacted_index`; found central state records `found_local_index` after compaction.
- source_patch_python_buffers: `beam_engine.allocate_buffers` preallocates static history arrays sized `max_depth*n_local`; `BeamEngine.history_entry(depth,local_index)` exposes one survivor record for distributed reconstruction.
- source_patch_sizing: `scripts/t4_sizing.py` and `scripts/h100_sizing.py` now include `--max-depth` and history memory buffers; practical history record layout is 7 bytes per kept state per depth before allocator alignment.
- source_patch_runner: added `scripts/fullbeamnice_current_solver_2gpu.py`; runner uses FullBeamNice scorer adapter, current 3-stream solver, beam width `2**23`, static history, distributed parent-chain reconstruction, and CPU validation that reconstructed moves restore the central state.
- cuda_graph_decision: FullBeamNice path-history runner sets `USE_CUDA_GRAPHS=0` because captured graph arguments would otherwise freeze `history_depth` and overwrite one history layer; this is a correctness requirement for path reconstruction, not a replacement of Stream1/Stream2/Stream3 logic.
- kaggle_notebook_patch: separate notebook `kaggle_fullbeamnice_kernel_stage/fullbeamnice_current_solver.ipynb` now calls `scripts/fullbeamnice_current_solver_2gpu.py` and prints `FULLBEAMNICE_CURRENT_SOLVER_RESULT` with path fields.
- local_verification: `python -m py_compile beam_engine.py scripts\t4_sizing.py scripts\h100_sizing.py scripts\export_fullbeamnice_scorer.py scripts\fullbeamnice_current_solver_2gpu.py` passed.
- remote_verification_status: pending Kaggle dataset version and kernel run after staging updated source files.

## 2026-05-07 static_path_history_restore_fix

- prompt_summary: User pasted Kaggle log: depth 20 found center with no overflows, then path validation failed with `reconstructed path does not restore central state`.
- failure_root_cause: insertion path set `STATUS_FOUND` and `STATUS_FOUND_LOCAL_INDEX` before final threshold prune and compaction; `found_local_index` could reference `next_state_pool` work index instead of compacted `beam_current`/history index.
- source_patch_cpp: `enqueue_one_depth` now clears `STATUS_FOUND`, `STATUS_FOUND_LOCAL_INDEX`, and `STATUS_FOUND_ACTION` immediately before compaction; only `kernel_compact_next_to_current` can publish final found status for a survivor with valid history row.
- stream_logic_change: false; Stream1 inference, Stream2 processing, Stream3 exchange, thresholding, pruning, and static buffers remain unchanged.
- expected_effect: reconstruction starts from a compacted current-state index, so parent-chain traversal reads the correct static history rows.
- local_verification_status: pending syntax check, staging, Kaggle dataset version, and separate kernel rerun.

## 2026-05-07 testcsv_solver_cuda_graph_request

- prompt_summary: User requested inspecting the new Kaggle notebook/log, moving cell fixes into files, enabling CUDA graph without corrupting path history, tuning 2xT4 parameters, disabling debug logging, solving `test.csv` first with beam `2**16`, then with beam `2**24` if memory allows, and making sample selection convenient.
- source_patch_cuda_graph_history: added static GPU `history_depth_cell`; compaction reads depth from device memory and increments after each depth, so CUDA graph can be reused without writing all history layers to depth zero.
- source_patch_reset: `reset_search` no longer invalidates the captured graph; fixed-size graph can be reused across `test.csv` samples because buffer addresses, dimensions, and kernels remain unchanged.
- source_patch_runner: added `scripts/solve_testcsv_2gpu.py`; runner solves selected rows from `test.csv`, reconstructs and validates paths, writes `submission.csv`, and supports `TEST_START`, `TEST_COUNT`, and `TEST_IDS`.
- source_patch_notebook: separate Kaggle notebook now runs `scripts/solve_testcsv_2gpu.py` with configurable `RUN_MODE`, `GLOBAL_BEAM_WIDTH`, `MAX_DEPTH`, `INFERENCE_PARALLELISM`, `B_MICRO`, and `K_EXPAND_TILE`.
- sizing_result_t4_depth100: beam `2**24` with static history uses `8.750 GiB` per T4 before model/runtime overhead and is the large candidate; beam `2**25` uses `17.409 GiB` per T4 before overhead and is not viable on 15 GiB T4 with current 7-byte history layout.
- current_perf_defaults: `MAX_DEPTH=100`, `USE_CUDA_GRAPHS=1`, `INFERENCE_PARALLELISM=1`, `B_MICRO=32768`, `K_EXPAND_TILE=32768`, `SCORE_RING_DEPTH=8`, `BETA=1.20`, `HASH_LOAD_FACTOR=0.45`, `PROBE_LIMIT=256`.
- local_verification: Python syntax passed for `beam_engine.py`, `scripts/fullbeamnice_current_solver_2gpu.py`, `scripts/solve_testcsv_2gpu.py`, sizing scripts, and FullBeamNice exporter.
- remote_verification_status: pending dataset upload, Kaggle smoke run with beam `2**16`, output download, then large run with beam `2**24`.

## 2026-05-07 local_found_path_fix

- prompt_summary: User required correctness first, then CUDA graph plus history validation, then one speed benchmark, then fixed production notebook and beam `2**16` run.
- failure_root_cause_update: `STATUS_FOUND` is all-reduced across cards, so every card reports `found=1` after global reduction even when only one card has the real central state and valid `found_local_index`; path reconstruction could start from the wrong card/index.
- source_patch: added `STATUS_LOCAL_FOUND=7`; kernels set `STATUS_LOCAL_FOUND` only on the card that locally finds central state; Python runners choose `found_rank` using `local_found`, not globally reduced `found`.
- expected_effect: reconstructed parent chain starts from the real card-local compacted index, not a bogus index on another card.
- notebook_patch: default run mode changed to `check_depth2`; it uses beam `2**16`, depth `2`, 16 rows, CUDA graph enabled, and path validation enabled.
- verification_status: pending Kaggle depth-2 correctness run.

## 2026-05-06 fullbeamnice_model_inspection

- prompt_summary: User asked to inspect `FullBeamNice`, identify the neural network and search behavior, then state concretely what must be done before integrating that model into the current distributed solver.
- inspection_scope: read-only; current Kaggle notebook must not be touched; no algorithm code changes performed.
- inspected_files: `FullBeamNice/notebooks/q_model_inference_standalone.ipynb`, `FullBeamNice/generators/p900.json`, `FullBeamNice/logs/model_p900-t000-q-sym_1777988767.json`, `FullBeamNice/targets/p900-t000.pt`, `FullBeamNice/weights/p900-t000-q-sym_1777988767_best.pth`.
- model_identity: `model_name=p900-t000-q-sym`, `model_id=1777988767`, `model_mode=QMLP2RB`, `num_parameters=23978008`, `n_gens=24`, `state_size=120`.
- model_architecture: `Pilgrim` with `LegacyCompatibleEmbeddingBagLinear(state_size=120,num_classes=120,out=1536)`, `BatchNorm1d(1536)`, `Linear(1536,512)`, `BatchNorm1d(512)`, `2xResidualBlock(512)`, `Linear(512,24)`.
- checkpoint_shapes: `input_layer.weight=(1536,14400)`, `hidden_layer.weight=(512,1536)`, residual weights `(512,512)`, `output_layer.weight=(24,512)`, all checkpoint tensors loaded as `float32`.
- inference_dtype_rule: FullBeamNice notebook converts model to `float16` on CUDA and keeps `float32` on CPU; input states are integer token ids.
- target_contract: `targets/p900-t000.pt` is tensor shape `(120,)`, dtype `int8`, values `0..119`; target equals solved/central state for search.
- generator_contract: `generators/p900.json` contains 24 permutation actions, names `[U,D,F,B,L,DR,BL,FR,BR,FL,R,DL,U',D',F',B',L',DR',BL',FR',BR',FL',R',DL']`.
- current_solver_action_mapping: current `data_loader.ACTION_NAMES` order is `[-B,-BL,-BR,-D,-DL,-DR,-F,-FL,-FR,-L,-R,-U,B,BL,BR,D,DL,DR,F,FL,FR,L,R,U]`; mapping `-X -> X'`, `X -> X` produced exact permutation equality for all 24 actions.
- required_score_adapter: FullBeamNice output order differs from current solver action order, so model outputs must be permuted from FullBeamNice order into current solver order before Stream1 writes `score_ring`.
- score_semantics: FullBeamNice QSearcher uses `torch.topk(q_flat, largest=False)`; lower Q value means better next move.
- required_score_sign_adapter: current solver threshold rule keeps larger scores (`candidate.score <= threshold -> discard`), so FullBeamNice Q values must be converted to higher-is-better scores, likely `score = -Q` after scaling/clamping to current `int16` score ring.
- original_search_behavior: FullBeamNice performs per-step Q inference, masks only immediate inverse moves, picks lowest-Q candidates, computes hashes after moves, removes visited hashes, keeps first unique candidates, and reconstructs path through fixed `tree_move/tree_idx` arrays.
- integration_constraint: take only the model/scorer contract from FullBeamNice; do not copy FullBeamNice beam-search logic into the distributed Stream1/Stream2/Stream3 solver unless explicitly requested.
- next_required_work: create an adapter/export path for TorchScript scorer accepting `[B,120]` integer states and returning current-solver ordered higher-is-better `int16/float` scores; add a separate Kaggle notebook/kernel for FullBeamNice model tests; compare produced path/result against FullBeamNice notebook on the same target/scramble.

## 2026-05-06 fullbeamnice_reproduction_gap_audit

- prompt_summary: User requested a complete comparison of FullBeamNice behavior against the current solver and asked for a concrete Russian-only plan to reproduce FullBeamNice with the current code while reusing the trained model.
- language_constraint: user-facing answers should be Russian-only and should avoid English technical insertions such as `frontier` and `pool`.
- target_alignment_check: current `data/puzzle_info.json` central state equals `FullBeamNice/targets/p900-t000.pt`; both are length 120 and values `0..119`.
- action_alignment_check: all current action permutations equal FullBeamNice permutations after name mapping `-X -> X'`, `X -> X`; only output action order differs.
- fullbeamnice_step_logic: computes `Q(state, action)` for 24 actions, masks immediate inverse of the previous move, checks one-move solved predecessors, picks lowest-Q candidates, removes already visited hashes, keeps first unique states, then applies selected moves.
- fullbeamnice_visited_scope: keeps a cumulative visited hash set across search depths and across failed attempts.
- fullbeamnice_path_output: stores fixed `tree_move[num_steps,B]` and `tree_idx[num_steps,B]`, then reconstructs and normalizes the final move sequence.
- current_solver_step_logic: computes 24 scores per active state, applies threshold, applies move plus hash in CUDA, routes by owner, inserts or updates by hash, exchanges remote candidates, prunes by global threshold, compacts active next states into the next current state array.
- current_solver_missing_inverse_mask: no immediate inverse move mask is present in `kernel_process_score_slot`; all 24 actions are considered for every state.
- current_solver_visited_scope: hash table deduplicates candidates inside the current depth/step work set, but compaction discards the prior hash table; no persistent cumulative visited set across depths was found.
- current_solver_path_gap: `BeamMeta` stores only one-step parent metadata for the current next-state array; after compaction, parent chains are not preserved across depths; current `search()` returns found/depth/status but not the full move sequence.
- required_for_exact_reproduction: use FullBeamNice scorer adapter, reproduce FullBeamNice random-walk start state, add immediate inverse masking, add persistent visited hash storage, add multi-depth parent/move storage and path reconstruction, and keep all new runtime arrays statically preallocated.
- logic_change_gate: inverse masking, persistent visited storage, and path reconstruction are algorithm/runtime changes; implementation requires explicit approval before code edits.

## 2026-05-05 task_cpp_cuda_kaggle_cluster

- prompt_summary: User defined work plan for C++/CUDA distributed beam-search development targeting 100xH100; stage 1 uses Kaggle 2xT4 for code debugging; stage 2 uses SSH cluster with 2xH100 for final debugging; stage 3 targets 100xH100 scale.
- kaggle_notebook_url: `https://www.kaggle.com/code/trydotatwo/notebookaafc902d8e/edit`
- required_access_method_stage_1: `kaggle` CLI.
- local_kaggle_cli_version_observed: `Kaggle API 1.7.4.5`
- hard_constraint_logic_change: no C++/CUDA/Python/notebook algorithmic logic changes without explicit user approval.
- required_before_logic_change: explain problem, reason, proposed change, expected effect, risk, and verification plan before editing logic.
- communication_protocol: AML-HIP; explicit entities; key=value dense lines; low ambiguity.
- project_rules_update: root `AGENTS.md` created to persist startup rules, memory rules, logic-change gate, and staged execution plan.
- repository_state_observed: current workspace is not a git repository; `git status --short` returned `fatal: not a git repository`.
- docs_read_for_startup: `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`, `docs/README_RU.md`, `docs/INDEX.md`.
- no_logic_files_modified: true.

## 2026-05-05 kaggle_auth_probe

- credential_file_location: repository root `kaggle.json`.
- credential_secret_handling: file content not printed; token not copied into chat.
- kaggle_cli_auth_probe: `KAGGLE_CONFIG_DIR=<repo_root> kaggle kernels list --mine --page-size 1`.
- kaggle_cli_auth_result: success; CLI listed private/mine notebook `trydotatwo/notebook020dea0288`.
- target_notebook_status_probe: `KAGGLE_CONFIG_DIR=<repo_root> kaggle kernels status trydotatwo/notebookaafc902d8e`.
- target_notebook_status_result: `404 Client Error`; likely causes include wrong slug, deleted notebook, or notebook inaccessible under current Kaggle account.
- target_notebook_search_probe: `KAGGLE_CONFIG_DIR=<repo_root> kaggle kernels list --mine --search notebookaafc902d8e --page-size 10`.
- target_notebook_search_result: found `trydotatwo/notebookaafc902d8e`; title `notebookaafc902d8e`; author `Иван Литвак`; lastRunTime `2026-05-04 15:38:34.917000`.
- interpretation: Kaggle auth works and target notebook exists; `kernels status` endpoint returned 404 despite list visibility.
- secret_exposure_mitigation: `.gitignore` created with `kaggle.json`.
- no_logic_files_modified: true.

## 2026-05-05 kaggle_cli_status_answer

- prompt_summary: User asked whether Kaggle CLI connection is ready for running notebooks and inspecting results.
- current_answer: Kaggle CLI auth works through repository-root `kaggle.json` when `KAGGLE_CONFIG_DIR=<repo_root>` is set.
- verified_capability_list: `kaggle kernels list --mine --search notebookaafc902d8e --page-size 10` found target notebook `trydotatwo/notebookaafc902d8e`.
- verified_capability_run_method: `kaggle kernels push --path <folder>` uploads notebook/kernel sources and starts a Kaggle run; local folder must contain `kernel-metadata.json`.
- verified_capability_output_method: `kaggle kernels output <owner>/<kernel-name> --path <folder>` downloads run output files.
- caveat_status_method: `kaggle kernels status trydotatwo/notebookaafc902d8e` returned `404 Client Error`; use `list`, `push`, and `output` as primary control path unless status endpoint starts working after a new pushed run.
- no_notebook_run_started_yet: true.
- no_logic_files_modified: true.

## 2026-05-05 kaggle_notebook_intake_and_run

- prompt_summary: User reported that notebook was placed in the project folder and repeated constraints for C++/CUDA 100xH100 work with Kaggle 2xT4 first-stage debugging.
- local_notebook_detected: `notebooks/notebookaafc902d8e (6).ipynb`; size `60002` bytes; last write `2026-05-05 21:28:43`; cells `13`.
- local_notebook_outputs_observed: cells `8` and `12` contain `KeyboardInterrupt` outputs from previous Kaggle execution; source logic was not modified.
- remote_metadata_pull: `kaggle kernels pull trydotatwo/notebookaafc902d8e -p kaggle_remote_pull -m`; result success; metadata includes GPU, internet, and dataset source `trydotatwo/kekaff`.
- original_ref_push_attempt_1: `kaggle kernels push -p kaggle_push_stage`; result `Kernel push error: Notebook not found`.
- original_ref_push_attempt_2: same notebook with staging metadata field `id_no` removed; result `Kernel push error: Notebook not found`.
- workaround_run: created new private debug kernel `trydotatwo/cayleybeam-t4-debug-20260505` using same notebook source and same dataset source; no notebook logic changed.
- workaround_run_push_result: `Kernel version 1 successfully pushed`; URL `https://www.kaggle.com/code/trydotatwo/cayleybeam-t4-debug-20260505`.
- workaround_run_status_probe: `kaggle kernels status trydotatwo/cayleybeam-t4-debug-20260505`; result `KernelWorkerStatus.RUNNING`.
- workaround_run_final_status: `KernelWorkerStatus.ERROR`; output downloaded to `kaggle_outputs/cayleybeam-t4-debug-20260505`.
- workaround_run_error_root_cause: Kaggle assigned `Tesla P100-PCIE-16GB` with CUDA capability `sm_60`; current Kaggle PyTorch supports `sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`; failure at `beam_engine.allocate_buffers(...); tensor.zero_()` with `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`.
- cli_upgrade_for_accelerator: upgraded local `kaggle` package from `1.7.4.5` to `2.1.0`, found import failure, then installed `kaggle==2.0.2`; `kaggle kernels push --help` now includes `--accelerator ACC`.
- dependency_warning: `pip install --force-reinstall kaggle==2.0.2` emitted many dependency-conflict warnings in the user Python environment.
- t4_rerun: `kaggle kernels push -p kaggle_push_stage_new --accelerator NvidiaTeslaT4`; result `Kernel version 2 successfully pushed`.
- t4_rerun_status_probe: `kaggle kernels status trydotatwo/cayleybeam-t4-debug-20260505`; result `KernelWorkerStatus.RUNNING`.
- t4_rerun_final_status: `KernelWorkerStatus.ERROR`; outputs downloaded to `kaggle_outputs/cayleybeam-t4-debug-20260505-v2`.
- t4_rerun_gpu_observed: two GPUs visible: `gpu=0; name=Tesla T4`, `gpu=1; name=Tesla T4`.
- t4_rerun_passed_checks_before_failure: data loader passed, CUDA extension build passed, 1-GPU correctness passed, 2-GPU run emitted rank reports with `cuda_graph_captured=true` and `remote_packed=10` on both ranks.
- t4_rerun_failure: notebook cell 9 timed out after `900.3s` while command `torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 scripts/kaggle_correctness_check.py` was still alive.
- t4_rerun_hang_signal: 2-GPU rank JSON reports printed around log time `73s`, then wrapper heartbeat continued silently until timeout at `969s`; likely hang after report print and before process exit.
- suspected_hang_location: `scripts/kaggle_correctness_check.py` lines after report print, especially `dist.destroy_process_group()` at end of `main()`.
- proposed_next_change_requires_user_approval: add diagnostic/cleanup instrumentation around distributed teardown, e.g. print before/after `dist.destroy_process_group()`, delete `engine` before process-group teardown, synchronize CUDA streams, optionally skip destroy under Kaggle debug env if destroy is confirmed as hang point.
- generated_artifact_guard: `.gitignore` updated for Kaggle token, root pulled metadata/notebook, staging folders, and downloaded outputs.
- no_logic_files_modified: true.

## 2026-05-06 kaggle_t4_torchrun_exit_fix_v3

- prompt_summary: User approved continuing work and fixing torchrun completion: "Делай дальше, можешь фиксить чтоб торчран завершался".
- approved_change_scope: process-control and distributed-teardown fixes that make torchrun terminate after success or failure.
- source_patch_1: `scripts/kaggle_correctness_check.py` gained Kaggle distributed fast-exit after JSON report; goal was to avoid NCCL/process-group teardown hang after successful 2-GPU correctness report.
- source_patch_1_verification: `python -m py_compile scripts\kaggle_correctness_check.py` passed.
- dataset_update: Kaggle dataset `trydotatwo/kekaff` versioned from `kaggle_dataset_stage`; dataset status became `ready`; dataset contains patched `scripts/kaggle_correctness_check.py`.
- kaggle_run_v3: kernel `trydotatwo/cayleybeam-t4-debug-20260505` version 3 was pushed with `--accelerator NvidiaTeslaT4`.
- kaggle_run_v3_gpu_observed: two GPUs visible: `gpu=0; name=Tesla T4`, `gpu=1; name=Tesla T4`.
- kaggle_run_v3_cell9_result: fixed; log contains `teardown_fast_exit=true` for ranks 0 and 1; notebook wrapper reported `process_exit | returncode=0` for 2-GPU `scripts/kaggle_correctness_check.py`.
- kaggle_run_v3_optional_mlp_result: passed; optional 1-GPU TorchScript MLP ensemble quick check returned `process_exit | returncode=0`.
- kaggle_run_v3_new_failure_cell: cell 12, one-cell shared-scorer 2-GPU 20M TorchScript test.
- kaggle_run_v3_cell12_algorithm_failure: `STEP_SUMMARY` at depth 5 reported `global_hash_overflow` around `56k`; rank processes raised `AssertionError: hash_overflow at depth=5`.
- kaggle_run_v3_cell12_process_failure: after both rank tracebacks, parent `torch.distributed.run` remained alive until notebook wrapper timeout `900 sec`; cell reported `timeout_kill`.
- diagnosis: cell 9 torchrun teardown hang is solved; remaining torchrun completion issue is cell 12 failure-path process termination after assertion.
- unapproved_change_scope: changing `GLOBAL_BEAM_WIDTH`, `MAX_DEPTH`, `HASH_LOAD_FACTOR`, `PROBE_LIMIT`, hash-table sizing, or overflow assertion semantics is an algorithm/test-logic change and still requires explicit user approval.
- source_patch_2: `notebooks/notebookaafc902d8e (6).ipynb` and `kaggle_push_stage_new/notebookaafc902d8e.ipynb` cell 12 process-control code patched only; added `signal`, `start_new_session=True`, timeout process-group kill, and generated runner `BaseException` handler with `os._exit(1)`.
- source_patch_2_expected_effect: if cell 12 asserts on `hash_overflow`, rank processes exit with code 1, torchrun returns nonzero promptly, notebook records real failure instead of waiting 900 seconds.
- source_patch_2_logic_change: false; search configuration and correctness assertions unchanged.

## 2026-05-06 kaggle_t4_torchrun_exit_fix_v4

- kaggle_run_v4: kernel `trydotatwo/cayleybeam-t4-debug-20260505` version 4 was pushed with `--accelerator NvidiaTeslaT4`.
- local_verification_before_push: cell 12 source compiled successfully for `notebooks/notebookaafc902d8e (6).ipynb` and `kaggle_push_stage_new/notebookaafc902d8e.ipynb`; markers `start_new_session`, `timeout_kill_process_group`, and `FATAL_EXIT` present.
- kaggle_run_v4_status: `KernelWorkerStatus.ERROR`.
- kaggle_run_v4_outputs: downloaded to `kaggle_outputs/cayleybeam-t4-debug-20260505-v4`; first download hit `ConnectionResetError(10054)`, second download completed and saved `cayleybeam-t4-debug-20260505.log`.
- kaggle_run_v4_gpu_observed: two GPUs visible: `gpu=0; name=Tesla T4`, `gpu=1; name=Tesla T4`.
- kaggle_run_v4_cell9_result: fixed; 2-GPU `scripts/kaggle_correctness_check.py` printed `teardown_fast_exit=true` for ranks 0 and 1 and exited with `process_exit | returncode=0 | elapsed_sec=6.6`.
- kaggle_run_v4_optional_mlp_result: passed; optional 1-GPU TorchScript MLP check exited with `process_exit | returncode=0 | elapsed_sec=4.9`.
- kaggle_run_v4_cell12_exit_result: fixed; generated runner printed `FATAL_EXIT type=AssertionError; message=hash_overflow at depth=6: 2715202`; `torch.distributed.run` raised `ChildFailedError`; notebook wrapper exited with `process_exit | returncode=1 | elapsed_sec=41.8`.
- kaggle_run_v4_timeout_result: no cell 12 `timeout_kill`; no 900-second hang after assertion.
- remaining_failure: algorithm/test correctness still fails in cell 12 because `global_hash_overflow` becomes nonzero at depth 6 with full shared-scorer config.
- cell12_config_at_failure: `GLOBAL_BEAM_WIDTH=1048576`, `WORLD_SIZE=2`, `B_MICRO=8192`, `MAX_DEPTH=8`, `HASH_LOAD_FACTOR=0.55`, `PROBE_LIMIT=64` default, `INFERENCE_PARALLELISM=4`.
- derived_cell12_sizes: `N_LOCAL=524288`, `K_KEEP=550502`, `K_WORK=633077`, `HASH_CAPACITY=1151049` per rank.
- failure_interpretation: cell 12 now reaches a real correctness assertion quickly; remaining question is how to size/prune/hash for the 20M TorchScript 2-GPU workload.
- next_change_requires_user_approval: changing depth limit, beam width, hash capacity/load factor, probe limit, threshold schedule, or overflow assertion semantics changes test/algorithm behavior and requires explicit user approval.
- proposed_next_lowest_risk_change: config-only cell 12 capacity run with `BETA=10.0`, `HASH_LOAD_FACTOR=0.55`, `MAX_DEPTH=8`, `GLOBAL_BEAM_WIDTH=1048576`; projected static buffer use is `1.458 GiB` per rank before CUDA context/NCCL/TorchScript/PyTorch allocator overhead.

## 2026-05-06 canonical_stream_pipeline_logic

- prompt_summary: User restated core distributed runtime logic and required locking this logic as non-negotiable without explicit agreement.
- hard_constraint_core_logic: do not deviate from the Stream1/Stream2/Stream3 pipeline semantics without first explaining the issue, proposed change, expected effect, risk, and receiving explicit user approval.
- stream1_role: continuous TE/Q-inference from neural networks; model weights loaded once; N neural networks/inference lanes run concurrently; output is written as `beam_current[m] -> score_ring[slot]`.
- stream1_event: after score production, Stream1 records `score_ready[slot]`.
- stream2_role_local: Stream2 waits for `score_ready[slot]`, reads `score_ring[slot]`, processes candidate lanes in `K_EXPAND_TILE` tiles, applies threshold filtering, applies puzzle move and hash in one kernel, computes `owner = hash % WORLD_SIZE`.
- stream2_local_owner_path: when `owner == local_rank`, Stream2 performs local hash insert/update and updates `local_hist`.
- stream2_remote_owner_path: when `owner != local_rank`, Stream2 packs candidate into `send_bucket[net_slot][owner]`.
- stream2_event_send: after bucket packing, Stream2 records `send_bucket_ready[net_slot]`.
- stream3_role_exchange: Stream3 waits for `send_bucket_ready[net_slot]`, performs counts exchange, performs grouped `ncclSend/ncclRecv` payload exchange, and records `recv_bucket_ready[net_slot]`.
- stream2_role_remote_ingest: Stream2 waits for `recv_bucket_ready[net_slot]` and ingests remote candidates into the local hash table.
- stream3_role_threshold: Stream3 periodically performs `ncclAllReduce(local_hist, global_hist)` and runs `kernel_compute_threshold(global_hist -> threshold_cell_gpu)`.
- stream2_threshold_rule: Stream2 reads `threshold_cell_gpu`; when `threshold_valid == false`, filtering is off; when `threshold_valid == true`, `candidate.score <= threshold` is discarded.
- design_implication: fixes must make Stream2 keep up with Stream1 and must prevent unbounded buffer growth without replacing the canonical roles of Stream1, Stream2, and Stream3.
- current_patch_review_needed: `PIPELINE_MAX_INFLIGHT_MICRO` throttles Stream1 by waiting for prior Stream2 score consumption; this may be useful as a debug guard but may conflict with the final requirement that Stream1 remains continuous; do not treat this knob as final design without user confirmation.

## 2026-05-06 kaggle_stream2_tile_backpressure_fix

- prompt_summary: User approved fixing all current issues and testing on Kaggle; user also required using the observed Kaggle project-code packaging path.
- approved_change_scope: implement a canonical Stream2 tile path so Stream2 drains Stream1 score slots through `K_EXPAND_TILE` candidate-lane tiles and Stream3 performs count/payload exchange per tile.
- preserved_core_logic: Stream1 still performs continuous inference into `score_ring[slot]`; Stream2 still waits `score_ready[slot]`, applies threshold, apply_move, hash, local insert/update, local histogram, and remote pack; Stream3 still owns NCCL count/payload exchange and global threshold computation; Stream2 still ingests remote candidates after `recv_bucket_ready`.
- removed_debug_throttle: `PIPELINE_MAX_INFLIGHT_MICRO` removed from active engine and active notebook source; only score-ring safety remains through `score_consumed[score_slot]` before slot reuse.
- source_patch_engine_cpp: `beam_engine.cpp` now splits each microbatch's `micro_size * fanout` lanes into `K_EXPAND_TILE` tiles, resets net slot counts per tile, records `send_ready`, performs Stream3 all-to-all, waits `recv_ready`, ingests remote candidates, and updates threshold on the configured period.
- source_patch_kernels_cu: `beam_kernels.cu` now lets `kernel_process_score_slot` process `[candidate_lane_offset, candidate_lane_offset + candidate_lanes)`, adds `recv_counts`, and makes `kernel_ingest_recv_slot` ingest only count-bounded received records instead of scanning stale bucket `valid` flags.
- source_patch_python: `beam_engine.py` now carries `k_expand_tile` and `recv_counts` buffers; sizing scripts account for `recv_counts` and print `K_EXPAND_TILE`.
- notebook_patch: `notebooks/notebookaafc902d8e (6).ipynb` and `kaggle_push_stage_new/notebookaafc902d8e.ipynb` active sources set `K_EXPAND_TILE=8192`, remove `PIPELINE_MAX_INFLIGHT_MICRO`, and set cell12 `BUCKET_CAP_PER_PEER=8192`.
- local_verification: `python -m py_compile beam_engine.py scripts\kaggle_correctness_check.py scripts\t4_sizing.py scripts\h100_sizing.py` passed.
- local_cuda_compile_probe: `nvcc -std=c++17 -c beam_kernels.cu -o C:\tmp\beam_kernels_check.obj` failed because local Windows `cl.exe` was not present in `PATH`; remote Kaggle compile remains required.
- kaggle_packaging_rule: modified source files must be copied into `kaggle_dataset_stage` before running `kaggle datasets version`; pushed notebook uses Kaggle dataset `trydotatwo/kekaff` as project source and copies the project from `/kaggle/input/...` to `/kaggle/working/CayleyBeam100H100`.
- remote_verification_status: pending Kaggle dataset version and kernel run.

## 2026-05-06 static_arrays_architecture_constraint

- prompt_summary: User declared a critical architecture rule: work exclusively with static arrays allocated before program/search hot path starts; no dynamic buffers in runtime code.
- hard_constraint_static_arrays: GPU/data-plane candidate, score, hash, histogram, bucket, counter, and status storage must be fixed-capacity and preallocated; runtime growth, dynamic device allocation, unbounded queues, and candidate container resizing are forbidden.
- allowed_error_handling_pattern: fixed-capacity structures may use overflow counters or fail-fast assertions when capacity is insufficient; silent capacity growth is forbidden.
- current_patch_static_array_audit: `K_EXPAND_TILE` implementation reuses preallocated `score_ring`, `send_buckets`, `recv_buckets`, `send_counts`, `recv_counts`, `hash_table`, `next_state_pool`, `local_hist`, `global_hist`, `threshold_cell`, and counters; no new runtime device allocation was added to the Stream1/Stream2/Stream3 data path.
- current_patch_note: host-side notebook packaging may unpack `source.zip` into Kaggle working storage before runtime; this is deployment/package staging, not GPU data-plane dynamic buffering.
- uncertainty_to_confirm_if_needed: host-side C++ control-plane objects such as `std::vector<cudaEvent_t>` and `std::vector<cudaStream_t>` already exist for fixed stream/event handles; if the rule also forbids host-side vector allocation during engine construction, replace these with fixed-size arrays after explicit approval.

## 2026-05-06 kaggle_v5_result_static_capacity_followup

- kaggle_run_v5: kernel `trydotatwo/cayleybeam-t4-debug-20260505` version 5 was pushed with `--accelerator NvidiaTeslaT4`.
- kaggle_dataset_packaging_result: latest dataset `trydotatwo/kekaff` became ready and contains updated minimal project files including `beam_engine.cpp`, `beam_kernels.cu`, `beam_engine.py`, `docs/PROJECT_MEMORY.md`, and notebook copy.
- kaggle_run_v5_source_result: log confirms project was copied from `/kaggle/input/datasets/trydotatwo/kekaff`; downloaded remote source contains `static_arrays_only`, `k_expand_tile`, `candidate_lane_offset`, and `recv_counts`.
- kaggle_run_v5_passed: extension build passed; cell9 2-GPU correctness passed; optional MLP quick check passed; process exit handling still works.
- kaggle_run_v5_cell12_progress: `K_EXPAND_TILE=8192`, `BUCKET_CAP_PER_PEER=8192`, count-bounded receive, and per-tile exchange reduced cell12 failure from previous large overflow to `global_hash_overflow=5966` at depth 5 with `bucket_overflow=0`.
- kaggle_run_v5_failure_root: fixed `next_state_pool`/`K_WORK` capacity is still slightly too small after per-tile pruning; depth 5 rank0 counters show `next_pool_size=635787` while derived `K_WORK=633077`, so overflow is static capacity exhaustion, not dynamic buffer growth.
- approved_static_capacity_fix: notebook cell12 now sets `BETA=1.50`; this increases preallocated `K_WORK` and hash capacity before engine start and preserves the static-array architecture.
- remote_verification_status: pending kernel v6.

## 2026-05-06 kaggle_v6_result_static_capacity_followup

- kaggle_run_v6: kernel `trydotatwo/cayleybeam-t4-debug-20260505` version 6 was pushed with `--accelerator NvidiaTeslaT4`.
- kaggle_run_v6_status: `KernelWorkerStatus.ERROR`; log saved at `kaggle_outputs/cayleybeam-t4-debug-20260505-v6/cayleybeam-t4-debug-20260505.log`.
- kaggle_run_v6_passed: extension build passed; cell9 2-GPU correctness passed; optional MLP quick check passed; cell12 depth 5 passed with `global_hash_overflow=0` and `global_bucket_overflow=0`.
- kaggle_run_v6_failure: cell12 depth 6 failed with `FATAL_EXIT type=AssertionError; message=hash_overflow at depth=6: 9872472`.
- kaggle_run_v6_depth6_summary: `global_current_size=1048201`, `global_next_pool_size=3002738`, `global_local_inserted=1651506`, `global_local_updated=669578`, `global_remote_packed=6321023`, `global_bucket_overflow=0`, `global_hash_overflow=9872472`, `global_pruned=603305`, `threshold_valid_sum=2`, `threshold_q_sum=65612`.
- kaggle_run_v6_interpretation: per-rank static `K_WORK=825753` and `HASH_CAPACITY=1501369` were insufficient for depth 6 after threshold filtering; failure remains fixed-capacity exhaustion, not runtime buffer growth.
- approved_static_capacity_fix: notebook cell12 now sets `BETA=3.50`; expected per-rank static `K_WORK=1926757` and `HASH_CAPACITY=3503195`; this preserves static preallocation and does not change Stream1/Stream2/Stream3 logic.
- notebook_source_verification: active source in `notebooks/notebookaafc902d8e (6).ipynb` and `kaggle_push_stage_new/notebookaafc902d8e.ipynb` contains `K_EXPAND_TILE=8192`, `BUCKET_CAP_PER_PEER=8192`, `BETA=3.50`, and no active `PIPELINE_MAX_INFLIGHT_MICRO`.
- remote_verification_status: pending kernel v7.

## 2026-05-06 kaggle_v7_result_static_capacity_followup

- kaggle_run_v7: kernel `trydotatwo/cayleybeam-t4-debug-20260505` version 7 was pushed with `--accelerator NvidiaTeslaT4`.
- kaggle_run_v7_status: `KernelWorkerStatus.ERROR`; log saved at `kaggle_outputs/cayleybeam-t4-debug-20260505-v7/cayleybeam-t4-debug-20260505.log`.
- kaggle_run_v7_passed: extension build passed; cell9 2-GPU correctness passed; optional MLP quick check passed; cell12 depths 0 through 5 passed with `global_hash_overflow=0` and `global_bucket_overflow=0`.
- kaggle_run_v7_failure: cell12 depth 6 failed with `FATAL_EXIT type=AssertionError; message=hash_overflow at depth=6: 83151`.
- kaggle_run_v7_depth6_summary: `global_current_size=1048066`, `global_next_pool_size=3936663`, `global_local_inserted=3853514`, `global_local_updated=355163`, `global_remote_packed=2356981`, `global_bucket_overflow=0`, `global_hash_overflow=83151`, `global_pruned=2805027`, `threshold_valid_sum=2`, `threshold_q_sum=67066`.
- kaggle_run_v7_interpretation: per-rank static `K_WORK=1926757` was slightly too small; rank0 `local_counters[0]=1968210` exceeded rank0 `local_counters[1]=1926757` by 41453; failure remains fixed-capacity exhaustion, not dynamic buffer growth.
- approved_static_capacity_fix: notebook cell12 now sets `BETA=4.00`; expected per-rank static `K_WORK=2202008` and `HASH_CAPACITY=4003651`; static buffer model reports `total_static_buffers_GiB=0.508` and `memory_ok=True` for T4.
- notebook_source_verification: active source in `notebooks/notebookaafc902d8e (6).ipynb` and `kaggle_push_stage_new/notebookaafc902d8e.ipynb` contains `BETA=4.00`, `K_EXPAND_TILE=8192`, `BUCKET_CAP_PER_PEER=8192`, and no active `PIPELINE_MAX_INFLIGHT_MICRO`.
- remote_verification_status: pending kernel v8.

## 2026-05-06 architecture_question_stream2_threshold_timing

- prompt_summary: User clarified intended architecture: `next_state_pool` should stay approximately `beam_width / gpu_count`; Stream1 continuously infers all scores; Stream2 should drain results and top-k/prune while Stream1 spends time on inference; Stream3 should keep global threshold synchronized; overflow should not happen if cutting is timely.
- code_audit_result: current implementation is partially aligned but not fully equivalent to intended overlap model.
- current_stream_order: `enqueue_one_depth` launches inference per microbatch, then Stream2 waits for that microbatch `score_ready`, processes all `K_EXPAND_TILE` tiles for that microbatch, Stream3 exchanges per tile, and only then the loop proceeds to the next microbatch's Stream2 work.
- current_threshold_update: `enqueue_threshold_update()` waits for Stream3 allreduce and then waits Stream2 on `threshold_ready`; this makes threshold update synchronous at tile boundaries rather than independently periodic/asynchronous.
- current_pool_prune_timing: `kernel_prune_by_threshold` runs only after all microbatches in the depth have been processed; during the depth, discarded candidates are filtered before insert only by the current threshold, but already inserted low-score candidates are not physically freed from `next_state_pool`.
- current_overflow_root: `COUNTER_HASH_OVERFLOW` includes both next-pool capacity exhaustion and hash probe overflow; v7 rank0 depth6 `local_counters[0]=1968210` exceeded `local_counters[1]=1926757`, proving next-pool capacity exhaustion occurred before final prune/compact.
- architecture_gap: current code does not maintain `next_state_pool ≈ beam_width / gpu_count` during the depth; current code lets `next_state_pool` grow to all accepted inserts during the depth, then prunes/compacts at the end.
- next_required_analysis: decide whether to implement in-depth top-k compaction/reclaim using static arrays or change threshold timing; this is a logic change and requires explicit approval before editing.

## 2026-05-06 global_topk_balanced_reservoir_requirement

- prompt_summary: User approved the local fixed top-k reservoir direction but clarified that purely local pruning can delete candidates that should move to another GPU; Stream3 must support inter-card local-top-k exchange and immediate candidate balancing.
- architecture_requirement: Stream2 local pruning must not finalize global discard before Stream3 has had a chance to exchange enough local top-k/frontier candidates across cards.
- stream3_new_role_detail: Stream3 should periodically exchange local top-k summaries/payloads, compute/propagate global cutoff or quotas, and rebalance accepted candidates by owner/load so every rank keeps approximately `beam_width / world_size` candidates.
- correctness_risk: local-only reservoir can bias global top-k and produce unfair top-k when one rank owns many strong candidates and another rank owns weaker candidates.
- design_direction: use fixed-size local reservoir plus fixed-size Stream3 balancing exchange; keep all buffers static and preallocated; avoid unbounded candidate queues.

## 2026-05-06 distributed_bounded_reclaim_patch

- prompt_summary: User approved implementing the bounded-reservoir direction.
- code_change_scope: added static free-list based in-depth reclaim after global threshold updates; no dynamic buffers or runtime allocation growth added.
- source_patch_python: `beam_engine.py` now preallocates fixed `free_indices[k_work]` and `free_count[1]` GPU buffers.
- source_patch_cuda: `beam_kernels.cu` now lets `kernel_prune_by_threshold` mark below-global-threshold entries inactive, tombstone matching hash slots, and push freed pool indices into the static free list.
- source_patch_cuda_insert: `hash_insert_or_update` now appends until `k_work`, then reuses indices from the static free list; if no static slot is available, it increments `COUNTER_HASH_OVERFLOW`.
- source_patch_cpp: `enqueue_threshold_update()` now performs global histogram allreduce, computes `threshold_cell`, waits Stream2 on threshold readiness, and immediately launches prune/reclaim inside the depth.
- source_patch_sizing: T4/H100 sizing scripts now account for `free_indices` and `free_count`.
- test_config_change: cell12 `BETA` changed from `4.00` to `1.20` to test memory reduction with bounded reclaim; T4 sizing for `GLOBAL_BEAM_WIDTH=1048576`, `WORLD_SIZE=2`, `BETA=1.20` reports `K_WORK=660602`, `HASH_CAPACITY=1201095`, `total_static_buffers_GiB=0.208`, `memory_ok=True`.
- local_verification: `python -m py_compile beam_engine.py scripts\t4_sizing.py scripts\h100_sizing.py scripts\kaggle_correctness_check.py` passed.
- local_cuda_verification: pending remote Kaggle compile because local Windows CUDA host compiler `cl.exe` is unavailable.
- kaggle_run_v8_result: remote compile passed; cell9 2-GPU correctness passed; optional MLP passed; cell12 failed at depth 5 with `global_hash_overflow=65`, `global_bucket_overflow=0`, `global_pruned=247188`, and rank0 `local_counters=[647307,647307,244595,532094,0,38,123366,158372]`.
- kaggle_run_v8_interpretation: bounded reclaim reduced the previous depth5/depth6 overflow from millions to 65 while using `BETA=1.20`; remaining failure is hash table probe/load pressure, not `next_state_pool` capacity.
- test_config_change_after_v8: notebook cell12 keeps `BETA=1.20`, adds `HASH_LOAD_FACTOR=0.45`, and adds `PROBE_LIMIT=128`; T4 sizing reports `K_WORK=660602`, `HASH_CAPACITY=1468004`, `total_static_buffers_GiB=0.216`, `memory_ok=True`.
- kaggle_run_v9_result: cell12 still failed at depth 5 with `global_hash_overflow=83`, `global_bucket_overflow=0`, `global_next_pool_size=1294702`, `global_pruned=246863`, and rank0 `local_counters=[647027,647027,243715,532055,0,40,123088,159357]`.
- kaggle_run_v9_interpretation: hash overflow persisted while next-pool attempts stayed below static `K_WORK`; remaining issue is stale tombstone/probe-chain pressure after in-depth prune, not next-state-pool capacity.
- source_patch_after_v9: added static hash-table rebuild after each global prune; implementation clears preallocated `hash_table` and rebuilds entries from active `next_meta`/`active_flags` without allocating dynamic memory.
- kaggle_run_v10_result: kernel `trydotatwo/cayleybeam-t4-debug-20260505` version 10 completed successfully on Kaggle 2xT4.
- kaggle_run_v10_passed: extension build passed; cell9 2-GPU correctness passed; optional MLP check passed; full cell12 `2-GPU + 1x20M TorchScript path/rank + 4 inference lanes + CUDA Graph + Stream1/2/3 + NCCL` passed.
- kaggle_run_v10_final_verdict: log contains `FULL_ALGORITHM_OK`; `bucket_overflow_zero_all_steps=true`; `hash_overflow_zero_all_steps=true`; final process exit `returncode=0`.
- kaggle_run_v10_depths: depth 0 through depth 8 completed; depth 8 summary included `global_bucket_overflow=0`, `global_hash_overflow=0`, `global_pruned=9758617`, `global_current_size=1041982`.
- architecture_status: bounded reclaim plus static hash rebuild now keeps physical active pool bounded by `K_WORK` while logical insert attempts can exceed `K_WORK` through free-list slot reuse; `global_next_pool_size` counter is now cumulative allocation attempts, not active resident pool size.
- remote_verification_status: passed on Kaggle 2xT4 for current debug config.

## 2026-05-06 kaggle_2xt4_max_beam_search

- prompt_summary: User requested benchmarking the maximum beam width that Kaggle 2xT4 can hold with the current bounded static implementation.
- baseline_result: `GLOBAL_BEAM_WIDTH=1048576`, `BETA=1.20`, `HASH_LOAD_FACTOR=0.45`, `PROBE_LIMIT=128`, `MAX_DEPTH=8` passed on Kaggle kernel v10; cell12 elapsed `127.7s`; static model `total_static_buffers_GiB=0.216` per rank.
- first_candidate: `GLOBAL_BEAM_WIDTH=2097152`; static model reports `N_LOCAL=1048576`, `K_WORK=1321206`, `HASH_CAPACITY=2936013`, `total_static_buffers_GiB=0.415`, `memory_ok=True`.
- benchmark_method: increase beam until Kaggle fails or timeout/OOM, then refine pass/fail bound; code logic unchanged, only benchmark env config changes.
- remote_verification_status: pending Kaggle run for `GLOBAL_BEAM_WIDTH=2097152`.

## 2026-05-02 task_notebook_debug

- prompt: "Работаем четко с ноутом notebookaafc902d8e (3).ipynb. Посмотри логи и скажи в чем проблема. Потом поправь ее и сохрани исправленний ноут с другим названием"
- source_notebook: `notebooks/notebookaafc902d8e (3).ipynb`
- observed_logs: 1-rank run completed; 2-rank cell emitted c10d hostname warnings and no `CompletedProcess` confirmation in stored output.
- root_cause: 2-rank notebook cell manually forced `HOSTNAME=127.0.0.1`, `MASTER_ADDR=127.0.0.1`, `MASTER_PORT=29505`, and `NCCL_COMM_ID=127.0.0.1:29505` while also using `torch.distributed.run --standalone`; this mixed two rendezvous/control-plane configurations and produced fragile Kaggle networking behavior.
- fix_plan: save corrected notebook copy; allow `torchrun --standalone` to own rendezvous; keep Kaggle-safe NCCL disables; add timeout and unbuffered output; clean mojibake comments and status prints.

## 2026-05-14 user_friendly_kaggle_notebook

- prompt_summary: User requested a user-friendly Kaggle notebook with primary config cell, advanced config cell, metrics cell, submit cell, competition-file inputs from Kaggle competition mount, model-only dataset input, GitHub project clone, custom scorer documentation, Yandex Cloud TODO-only cell, and 2xT4 validation.
- source_files_changed: `notebooks/kaggle_user_friendly_cpu_history.ipynb`, `kaggle_user_friendly_kernel_stage/kaggle_user_friendly_cpu_history.ipynb`, `kaggle_user_friendly_kernel_stage/kernel-metadata.json`.
- notebook_config_layout: first cell contains only `SAMPLE_START`, `SAMPLE_COUNT`, `GLOBAL_BEAM_WIDTH`, `B_MICRO`, `BETA`; second cell contains advanced parameters including `LOG_EVERY`, `HISTORY_BACKEND`, `TIMEOUT_SEC`, checkpoint/resume flags, GitHub URL/branch, model dataset hint, and scorer initializer path.
- notebook_data_source_rule: project source is cloned from `https://github.com/TryDotAtwo/MultiGPUBeamSearch.git` branch `master`; Kaggle competition files `puzzle_info.json`, `sample_submission.csv`, and `test.csv` are discovered recursively under `/kaggle/input` and copied into project `data/`; model files remain sourced from dataset `trydotatwo/cayleybeam-fullbeamnice-project`.
- notebook_docs_added: custom scorer cell documents `SCORER_INIT_PY`, `output_kind=action24`, `action12`, `value1_after_move`, `heuristic24`, canonical `[B,24]` higher-is-better scores, and TODO for arbitrary generator count beyond 24.
- notebook_cloud_cell: Yandex Cloud integration cell is comment/TODO-only and prints `YANDEX_CLOUD_STATUS: disabled; current_run_location=Kaggle_only`; no cloud code is active.
- notebook_metrics_cell: computes and prints `total_count`, `unsolved_count`, `solved_percent`, `total_len`, `mean_len_all`, `median_len_all`, `max_len_solved`, `min_len_solved`, `mean_len_solved`, `median_len_solved`, `solved_lengths`, and ASCII solved-length histogram.
- notebook_submit_cell: builds requested `submit_message` and runs `kaggle competitions submit -c cayley-py-megaminx -f /kaggle/working/submission.csv -m "$submit_message"`.
- git_commit_main: `0794e30 Add CPU history archive and user-friendly Kaggle notebook` pushed to `origin/master`.
- git_commit_fix: `ab98087 Find Kaggle competition files recursively` pushed to `origin/master`; fix replaced hardcoded competition mount path with recursive file discovery preferring `/kaggle/input/competitions/...`.
- kaggle_kernel: `trydotatwo/cayleybeam-user-friendly-cpu-history`; metadata requests `enable_gpu=true`, `enable_internet=true`, `competition_sources=["cayley-py-megaminx"]`, `dataset_sources=["trydotatwo/cayleybeam-fullbeamnice-project"]`.
- kaggle_v1_result: status `ERROR`; root cause `FileNotFoundError` from hardcoded `/kaggle/input/cayley-py-megaminx`; hardware log confirmed `2x Tesla T4`.
- kaggle_v2_result: status `KernelWorkerStatus.COMPLETE`; run cloned GitHub source, detected competition directory `/kaggle/input/competitions/cayley-py-megaminx`, detected model dataset `/kaggle/input/datasets/trydotatwo/cayleybeam-fullbeamnice-project`, and executed 2-rank `torch.distributed.run` on `2x Tesla T4`.
- kaggle_v2_config: `GLOBAL_BEAM_WIDTH=4096`, `B_MICRO=8192`, `BETA=32.0`, `MAX_DEPTH=100`, `INFERENCE_PARALLELISM=2`, `K_EXPAND_TILE=16384`, `HISTORY_BACKEND=cpu`.
- kaggle_v2_memory: `scripts/t4_sizing.py` reported `total_static_buffers_GiB=0.098`, `memory_ok=True`.
- kaggle_v2_solver_result: log contains `SAMPLE_RESULT {"pos": 0, "id": 0, "found": false, "depth": -1, "path_len": 0, "path": "", "cuda_graph_captured_sum": 2, "elapsed_sec": 2.576}` and `SUBMISSION_WRITTEN {"path": "/kaggle/working/submission.csv", "rows": 1, "append_each": true, "elapsed_sec": 2.576}`.
- kaggle_v2_metrics: `total_count=1`, `unsolved_count=1`, `solved_percent=0.0`, `total_len=0`, `mean_len_all=0`, `median_len_all=0`, `solved_lengths=[]`.
- kaggle_v2_submit: log contains `SUBMIT_MESSAGE: test run: beam=4096 depth=100 samples=0..0 parallelism=2 b_micro=8192 k_tile=16384 beta=32.0` and `Successfully submitted to CayleyPy Megaminx Solve Optimally`.
- kaggle_cli_note: `kaggle kernels output` hit a local Windows cp1251 encoding failure while printing downloaded output filenames, but `kaggle kernels logs` succeeded and confirmed remote completion; downloaded `submission.csv` contains header plus row `0,`.
- safety_note: no Kaggle kernel was stopped or killed during validation.
- followup_2026_05_14_mojibake_fix: User screenshot showed first notebook comments rendered as `???????`; root cause was literal question-mark replacement already present in notebook JSON, not Kaggle rendering; fixed both `notebooks/kaggle_user_friendly_cpu_history.ipynb` and `kaggle_user_friendly_kernel_stage/kaggle_user_friendly_cpu_history.ipynb` by rewriting user-facing comments in ASCII English while preserving all code values and cell order.
- followup_2026_05_14_mojibake_validation: JSON parse passed for both notebooks; combined notebook text has `question_runs=0` for `???` and `non_ascii=0`, so Kaggle UI cannot show mojibake for these comments.
- followup_2026_05_14_timeout_fix: Kaggle v3 after comment fix failed because `TIMEOUT_SEC=0` was documented as disabled but passed to `run_live` as a literal zero, causing immediate `TimeoutError`; patched solver call to pass `None if TIMEOUT_SEC <= 0 else TIMEOUT_SEC`.
- kaggle_v4_result_after_fixes: kernel version 4 completed on 2xT4; logs confirmed ASCII custom scorer/Yandex cells, GitHub clone, competition data discovery, model dataset discovery, memory sizing, 2-rank solver returncode `0`, `SUBMISSION_WRITTEN rows=1`, metrics cell output, and successful Kaggle competition submit.

## 2026-05-14 release_debug_flag_cleanup

- prompt_summary: User requested `BEAM_DEBUG_ON` release behavior so unnecessary C++/CUDA debug code is excluded from release builds, while per-sample progress logging remains configurable.
- existing_engine_support: `beam_engine.py` already builds extension variants with `-DBEAM_DEBUG_ON=0/1`, extension name suffix `_d0/_d1`, and `beam_engine.cpp` already wraps engine debug code in `#if BEAM_DEBUG_ON`.
- source_patch_solver: `scripts/solve_testcsv_2gpu.py` now treats `DEPTH_LOG_EVERY` as active only when `BEAM_DEBUG` or `ENGINE_DEBUG` is enabled; default release mode has `depth_log_every=0` regardless of `DEPTH_LOG_EVERY`.
- source_patch_solver_logging: per-sample `SAMPLE_RESULT` logging is now controlled by `SAMPLE_LOG_EVERY`, with backward-compatible fallback to `LOG_EVERY`; default is `1`, so rank0 can still log after every completed sample.
- notebook_patch: user-friendly Kaggle notebooks now expose `BEAM_DEBUG=0`, `DEPTH_LOG_EVERY=0`, and `SAMPLE_LOG_EVERY=1`; comments state that `BEAM_DEBUG=0` builds release extension with debug C++ code excluded by `#if`.
- local_verification: `python -m py_compile beam_engine.py scripts\solve_testcsv_2gpu.py` passed; both notebooks JSON-parse; notebook text has `question_runs=0` and `non_ascii=0`.

## 2026-05-14 cuda_graph_huge_beam_guard

- prompt_summary: User confirmed CUDA Graph caused huge-beam OOM and requested fixing CUDA Graph usage without breaking static-buffer architecture.
- root_cause: current `capture_cuda_graph()` captured a full depth; at `GLOBAL_BEAM_WIDTH=70M`, `N_LOCAL=35M`, `B_MICRO=8192`, this records about 4273 microbatches per rank, including TorchScript forward calls, and can create a large CUDA/PyTorch graph-private memory pool.
- source_patch_cpp: `TorchScriptEnsembleBackend` no longer retains scorer output tensors in `outputs_by_slot`; scorer output remains transient and is copied/quantized into the preallocated `score_ring` slot.
- reverted_graph_guard: User rejected `CUDA_GRAPH_MAX_MICRO` as a bogus knob; removed `cuda_graph_max_micro` from C++ config, Python config, solver graph expectation, notebooks, and environment export.
- remaining_cuda_graph_state: `USE_CUDA_GRAPHS` is again the single graph toggle; `TorchScriptEnsembleBackend` still does not retain `outputs_by_slot`, preserving the static `score_ring` path improvement.

## 2026-05-14 disable_torchscript_default

- prompt_summary: User requested removing TorchScript for now and returning to the previous lightweight scorer behavior.
- source_patch_solver: `scripts/solve_testcsv_2gpu.py` now defaults `INFERENCE_BACKEND=central_hamming` instead of `torchscript_ensemble`; `export_scorer(cfg)` runs only when `INFERENCE_BACKEND=torchscript_ensemble`.
- notebook_patch: user-friendly notebooks expose `INFERENCE_BACKEND='central_hamming'`; environment export uses the notebook variable; `SCORER_INIT_PY` comments now state that the initializer is used only with `torchscript_ensemble`.
- retained_code: TorchScript support remains in code as an opt-in backend, but the default Kaggle path no longer exports or loads TorchScript.
- local_verification: `python -m py_compile beam_engine.py scripts\solve_testcsv_2gpu.py` passed; both notebooks JSON-parse; notebook text has `question_runs=0` and `non_ascii=0`.

## 2026-05-14 restore_fullbeamnice_action24

- prompt_summary: User clarified that the old working path was the FullBeamNice TorchScript scorer with native 24 outputs; the unwanted change was support for arbitrary neural outputs/adapters, not TorchScript itself.
- source_patch_solver: `scripts/solve_testcsv_2gpu.py` default restored to `INFERENCE_BACKEND=torchscript_ensemble`; FullBeamNice scorer export is unconditional again for the solver path.
- source_patch_exporter: `scripts/export_fullbeamnice_scorer.py` now exports only the fixed FullBeamNice action24 scorer; custom `SCORER_INIT_PY`, `Canonical24ScoreAdapter`, `action12`, `value1_after_move`, and `heuristic24` adapter paths are removed from active exporter code.
- notebook_patch: user-friendly notebooks default `INFERENCE_BACKEND='torchscript_ensemble'`; custom scorer cell now states custom scorer support is disabled and active scorer is `FullBeamNice action24`.
- retained_fix: `beam_engine.cpp` still avoids retaining `outputs_by_slot`; TorchScript output tensor remains transient before copy/quantize into static `score_ring`.

## 2026-05-14 bucket_cap_per_peer_question

- prompt_summary: User asked briefly what `BUCKET_CAP_PER_PEER` does.
- symbol_search: `rg "BUCKET_CAP_PER_PEER"` and related patterns found no exact symbol in current workspace files.
- answer_basis: explained likely distributed static-buffer meaning: fixed maximum candidate/message slots in one preallocated send/receive bucket for one peer GPU/rank.
- source_changes: documentation memory only; no algorithm/code change.
- followup_prompt: User asked for a short explanation of the same previous entity.
- followup_prompt_ru: User asked to explain the same previous entity in Russian.
- followup_stream3: User asked whether `BUCKET_CAP_PER_PEER` is for stream type 3; answer should clarify that this is likely the per-peer fixed bucket capacity used by that stream path when stream 3 sends/receives peer-partitioned candidates/messages, not the stream id itself.

## 2026-05-14 current_sample_status_question

- prompt_summary: User asked shortly whether current Kaggle run is solving sample 0 or sample 1.
- observed_config: `kaggle_user_friendly_kernel_stage/kaggle_user_friendly_cpu_history.ipynb` has `SAMPLE_START=1` and `SAMPLE_COUNT=1`.
- kaggle_status: `kaggle kernels status trydotatwo/cayleybeam-user-friendly-cpu-history` returned `KernelWorkerStatus.RUNNING`.
- kaggle_logs: `kaggle kernels logs trydotatwo/cayleybeam-user-friendly-cpu-history` returned empty output at check time.
- answer_basis: current configured sample is sample 1; no completed `SAMPLE_RESULT` was visible in logs at check time.
