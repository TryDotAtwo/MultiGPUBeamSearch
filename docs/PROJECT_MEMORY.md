# Project Memory

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
