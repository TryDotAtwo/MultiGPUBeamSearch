# Project Memory

## 2026-05-18 architecture_v6_depth_loop_frontier_drain_fix

- entity_id: `architecture_v6_depth_loop_frontier_drain_fix`
- type: `patch_stage`
- state: `host_green_kaggle_pending`
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- prompt_summary: User requested cleanup and verification of current frontier-drain plus Stream5 receive-capacity fix against architecture_v6, without architecture expansion, performance tuning, legacy solver substitution, path reconstruction changes, action convention changes, or logical120/State128 boundary changes.
- constraints_preserved: no fallback backend, no runtime 120-slice, no separate `nn_input_120_buffer`, no legacy/staged solver substitution, no performance tuning, no Stream3/Stream4 semantic change, no path reconstruction change, no solver-quality claim.
- code_change: `production_v6_dispatcher.py::_run_stream5` remains cleaned to derive `remote_capacity` only from `self.cfg["bucket_cap_per_peer"]`, allocate `remote_recv` by capacity, create `recv_count` and `recv_offset` before `torch.cuda.synchronize()`, call `self.engine.v6_stream5_exchange_candidate_meta(...)` before synchronization, assert `remote_recv_count <= remote_capacity`, and return exactly `remote_recv`, `recv_count`, `recv_offset`, `remote_recv_count`, `remote_capacity`.
- code_change: `production_v6_dispatcher.py` now derives dispatcher `bucket_cap_per_peer` with `pow2_ceil(max(131072, b_micro * MOVE_COUNT))`; for `B_MICRO=8192`, `MOVE_COUNT=24`, `K_EXPAND_TILE=196608`, dispatcher capacity becomes `262144`.
- code_change: `tests/frontier_coverage_audit_world2.py` now defaults `FRONTIER_COVERAGE_B_MICRO` to `8192` for the requested Kaggle frontier coverage audit target.
- test_change: `tests/test_architecture_v6_static.py` adds static guards for Stream5 capacity source, absence of `stream3["unique_count"]` receive-capacity use, no duplicate `remote_recv_count` return key, exchange-before-synchronize ordering, dispatcher capacity derivation, `B_MICRO=8192` pow2 result `262144`, frontier drain loop, parent offset increment, depth row counters, and C++ capacity derivation.
- host_verification: `python -m py_compile production_v6_dispatcher.py beam_engine.py tests\frontier_coverage_audit_world2.py tests\test_architecture_v6_static.py` passed.
- host_static_pytest: `python -m pytest tests\test_architecture_v6_static.py -q` passed with `46 passed`.
- static_audit: searched for quick-patch Stream5 capacity artifact and verified no remaining `remote_capacity = max(int(stream3...` match in edited dispatcher.
- kaggle_validation_status: pending; required target is frontier coverage audit on Kaggle 2xT4 with `task_count=10`, `max_depth=12`, `beam_width=65536`, `b_micro=8192`.
- green_claim: false until Kaggle 2xT4 frontier coverage audit passes with runtime markers, output CSV, JSONL, and coverage invariants.
- test_result_file: `test_results/architecture_v6_depth_loop_frontier_drain_fix_2026-05-18.md`

## 2026-05-18 architecture_v6_stream1_state128_input_fix

- entity_id: `architecture_v6_stream1_state128_input_fix`
- type: `patch_before_retry`
- state: `host_green_kaggle_blocked`
- prompt_summary: User required Stream1 contract change from physical 120-wide FullBeamNice input to physical `State128`/128-byte input with effective semantic bytes `0..119`, zero padding bytes `120..127`, and zero first-layer weights for padding columns.
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- previous_failure_class: `architecture_v6_stream1_stream2_ring_batch_world2` failed because smoke mixed `State128` storage with 120-wide model input.
- constraints_preserved: no runtime `State128 -> 120` slice buffer, no separate `nn_input[:,120]` runtime buffer, no `STATE_STORAGE_LEN` change, no Stream2/hash/goal semantic change, no Stream3/4/5 change, no real puzzle solve claim, no performance tuning.
- code_change: `scripts/static_fullbeamnice_inference.py` now expands old FullBeamNice embedding table from logical `120*120` tokens to physical `128*128` tokens; copied region is positions `0..119` and values `0..119`; padding positions `120..127` and extra values remain zero.
- code_change: `scripts/static_fullbeamnice_inference.py` keeps 120-state reference compatibility by padding 120-wide reference tensors to `State128` only inside validation/reference code.
- code_change: `beam_engine.cpp` validates `fullbeamnice_static` with physical `state_size=128` and `num_classes=128`; runtime config accepts `state_size_bytes=128` for architecture v6 Stream1.
- code_change: `beam_engine.py` default config now uses `state_size_bytes=128`.
- test_change: `tests/stream1_cutlass_score_key_smoke.py` now uses synthetic `embed_w_t` shape `128*128 x 1536`, `state_size=128`, `num_classes=128`, and `State128` input storage.
- test_change: `tests/stream1_real_weights_smoke.py` validates zero padding token weights and validates `FullBeamNice120` versus `FullBeamNice128` padded output equivalence.
- test_change: `tests/stream1_stream2_ring_batch_world2_smoke.py` validates physical `State128` weights, zero padding token weights, 120-vs-128 padded output equivalence, Stream1 `score_ring`, Stream2 `hash_ring`, and goal solved metadata.
- host_verification: `python -m py_compile beam_engine.py tests\test_architecture_v6_static.py tests\stream1_cutlass_score_key_smoke.py tests\stream1_real_weights_smoke.py tests\stream1_stream2_ring_batch_world2_smoke.py scripts\static_fullbeamnice_inference.py` passed.
- host_static_pytest: `python -m pytest tests\test_architecture_v6_static.py -q` passed with `33 passed`.
- kaggle_dataset_update: `kaggle datasets version -p kaggle_stream1_stream2_ring_batch_world2_payload_dataset -m "architecture_v6_stream1_state128_input_fix" --dir-mode zip` succeeded.
- kaggle_kernel_push_status: blocked by escalation policy review because `kaggle kernels push` uploads staged project source/test files to external Kaggle service.
- kaggle_green_claim: false.
- next_required_action: user must provide explicit post-risk approval for `kaggle kernels push -p kaggle_stream1_stream2_ring_batch_world2_stage --accelerator NvidiaTeslaT4`, or request manual upload instructions.
- final_state_update: `architecture_v6_stream1_stream2_ring_batch_world2` is green after user post-risk approval and Kaggle kernel retry.
- kaggle_kernel: `trydotatwo/stream1-stream2-ring-batch-world2`
- kaggle_kernel_version: `2`
- kaggle_runtime: log confirmed `cuda_device_count=2`, `cuda_device_0=Tesla T4`, `cuda_device_1=Tesla T4`.
- kaggle_build: `setup.py build_ext --inplace` passed on Kaggle CUDA 12.8 runtime.
- kaggle_result: `torchrun` returned `0`.
- pass_marker_rank0: `STREAM1_STREAM2_RING_BATCH_WORLD2_SMOKE_OK rank=0 world_size=2 score_max_abs_diff=8 score_unique=40 hash_unique=96 solved_count=1 input_dim=128 padding_weight_max=0`
- pass_marker_rank1: `STREAM1_STREAM2_RING_BATCH_WORLD2_SMOKE_OK rank=1 world_size=2 score_max_abs_diff=8 score_unique=48 hash_unique=96 solved_count=1 input_dim=128 padding_weight_max=0`
- pass_marker_completion: `=== STREAM1_STREAM2_RING_BATCH_WORLD2_TEST_COMPLETE ===`
- validated_stream1_state128: Stream1 uses physical `State128` input with `input_dim=128`; FullBeamNice semantic input remains bytes `0..119`; padding token weights are zero.
- validated_stream1_scores: real FullBeamNice Stream1 wrote `score_ring` with `score_max_abs_diff=8` versus FP16 static reference on both ranks.
- validated_stream2_after_stream1: Stream2 consumed the prepared State128 parent batch, wrote `hash_ring`, and published goal metadata with `solved_count=1` on both ranks.
- green_claim: true.

## 2026-05-18 architecture_v6_synthetic_full_depth_with_stream1_batch_world2

- entity_id: `architecture_v6_synthetic_full_depth_with_stream1_batch_world2`
- type: `batch_stage`
- state: `green`
- prompt_summary: User requested first full synthetic depth iteration with real Stream1 score source, connecting real FullBeamNice Stream1 score_ring and Stream2 hash/goal path to already-green synthetic downstream depth loop semantics.
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- constraints_preserved: no performance tuning, no real solver claim, no architecture deviation, no runtime 128-to-120 slice, no separate `nn_input_120_buffer`, no fallback backend, no new Stream3/4/5 logic.
- test_added: `tests/synthetic_full_depth_with_stream1_batch_world2_smoke.py`
- static_guard_added: `tests/test_architecture_v6_static.py::test_synthetic_full_depth_with_stream1_batch_world2_smoke_contract`
- included_smoke_1: `one_depth_unsolved_real_stream1_to_final_materialization_world2`
- included_smoke_2: `one_depth_remote_exchange_real_stream1_scores_world2`
- included_smoke_3: `one_depth_stream4_threshold_from_real_score_ring_world2`
- included_smoke_4: `one_depth_solved_goal_stops_before_final_world2`
- included_smoke_5: `one_depth_drain_order_with_active_stream4_world2`
- included_smoke_6: `one_depth_padding_contract_after_materialization_world2`
- host_verification: `python -m py_compile tests\synthetic_full_depth_with_stream1_batch_world2_smoke.py tests\test_architecture_v6_static.py` passed.
- host_static_pytest: `python -m pytest tests\test_architecture_v6_static.py -q` passed with `34 passed`.
- kaggle_stage_added: `kaggle_synthetic_full_depth_with_stream1_batch_world2_stage`
- kaggle_payload_update_status: `kaggle datasets version -p kaggle_stream1_stream2_ring_batch_world2_payload_dataset -m "architecture_v6_synthetic_full_depth_with_stream1_batch_world2" --dir-mode zip` succeeded after explicit user approval.
- kaggle_kernel: `trydotatwo/synthetic-full-depth-with-stream1-batch-world2`
- kaggle_kernel_version: `1`
- kaggle_runtime: log confirmed `cuda_device_count=2`, `cuda_device_0=Tesla T4`, `cuda_device_1=Tesla T4`, `GPU 0: Tesla T4`, `GPU 1: Tesla T4`.
- kaggle_build: `setup.py build_ext --inplace` passed on Kaggle CUDA runtime.
- kaggle_result: `torchrun` returned `0`.
- pass_marker_rank0: `SYNTHETIC_FULL_DEPTH_WITH_STREAM1_BATCH_WORLD2_SMOKE_OK rank=0 tests=6 score_max_abs_diff=8 score_unique=34 hash_unique=90 remote_send=56 remote_recv=50 clean=83 keep=9`
- pass_marker_rank1: `SYNTHETIC_FULL_DEPTH_WITH_STREAM1_BATCH_WORLD2_SMOKE_OK rank=1 tests=6 score_max_abs_diff=8 score_unique=46 hash_unique=90 remote_send=50 remote_recv=56 clean=97 keep=8`
- pass_marker_completion: `=== SYNTHETIC_FULL_DEPTH_WITH_STREAM1_BATCH_WORLD2_TEST_COMPLETE ===`
- validated_path: `Stream1 real weights -> score_ring -> Stream2 hash/goal -> synthetic Stream3 split -> synthetic Stream5 exchange -> collector -> Stream4 -> threshold/final/materialization`.
- green_claim: true.

## 2026-05-18 architecture_v6_external_upload_policy

- entity_id: `architecture_v6_external_upload_policy`
- type: `standing_user_approval`
- state: `active`
- prompt_summary: User provided global project policy approving external Kaggle uploads for architecture_v6 when staged upload follows architecture_v6 exactly and does not deviate from `ARCHITECTURE_NEED.md` or `PlanRefact.md`.
- allowed_external_actions: `kaggle datasets create/version`, `kaggle kernels push`, staged source/test/notebook payload upload, FullBeamNice validation payload upload, third-party CUTLASS header upload.
- hard_constraints: no architecture deviation, `ARCHITECTURE_NEED.md` source of truth, `PlanRefact.md` implementation plan source of truth, no fallback backend, no TorchScript/dummy/central_hamming fallback, no runtime 120-slice, no separate `nn_input_120_buffer`, no unplanned Stream3/4/5 logic changes, no real solver claim before real validation, no performance tuning before functional green path.
- workflow_rule: use larger batch stages when components share risk class; avoid tiny approval/micro-stage loops when architecture constraints are unchanged.
- before_upload_required: verify staged file list, verify no secret filenames or obvious credentials, verify stage matches current batch scope, verify no architecture deviation.
- before_green_required: actual Kaggle pass, runtime gate, rank markers, completion marker, update `docs/PROJECT_MEMORY.md` only after pass.

## 2026-05-18 architecture_v6_multi_depth_dispatcher_loop_batch_world2

- entity_id: `architecture_v6_multi_depth_dispatcher_loop_batch_world2`
- type: `batch_stage`
- state: `green`
- prompt_summary: User accepted single synthetic full-depth iteration with real Stream1 and requested next risk class: multiple depth-iteration lifecycle between depths.
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- constraints_preserved: no performance tuning, no real solver claim, no architecture deviation, no runtime 128-to-120 slice, no separate `nn_input_120_buffer`, no fallback backend, no new Stream3/4/5 logic.
- test_added: `tests/multi_depth_dispatcher_loop_batch_world2_smoke.py`
- static_guard_added: `tests/test_architecture_v6_static.py::test_multi_depth_dispatcher_loop_batch_world2_smoke_contract`
- included_smoke_1: `multi_depth_two_iterations_real_stream1_world2_smoke`
- included_smoke_2: `layout_streams_layout_final_switching_world2_smoke`
- included_smoke_3: `current_frontier_copy_between_depths_world2_smoke`
- included_smoke_4: `threshold_initialized_persists_across_depths_world2_smoke`
- included_smoke_5: `stop_solved_early_exit_across_depths_world2_smoke`
- included_smoke_6: `multi_depth_padding_contract_world2_smoke`
- host_verification: `python -m py_compile tests\multi_depth_dispatcher_loop_batch_world2_smoke.py tests\test_architecture_v6_static.py` passed.
- host_static_pytest: `python -m pytest tests\test_architecture_v6_static.py -q` passed with `35 passed`.
- kaggle_stage_added: `kaggle_multi_depth_dispatcher_loop_batch_world2_stage`
- kaggle_payload_updated_local: `kaggle_stream1_stream2_ring_batch_world2_payload_dataset/tests/multi_depth_dispatcher_loop_batch_world2_smoke.py`
- kaggle_payload_update_status: `kaggle datasets version -p kaggle_stream1_stream2_ring_batch_world2_payload_dataset -m "architecture_v6_multi_depth_dispatcher_loop_batch_world2" --dir-mode zip` succeeded.
- kaggle_kernel: `trydotatwo/multi-depth-dispatcher-loop-batch-world2`
- kaggle_kernel_version: `1`
- kaggle_runtime: log confirmed `cuda_device_count=2`, `cuda_device_0=Tesla T4`, `cuda_device_1=Tesla T4`, `GPU 0: Tesla T4`, `GPU 1: Tesla T4`.
- kaggle_build: `setup.py build_ext --inplace` passed on Kaggle CUDA runtime.
- kaggle_result: `torchrun` returned `0`.
- pass_marker_rank0: `MULTI_DEPTH_DISPATCHER_LOOP_BATCH_WORLD2_SMOKE_OK rank=0 tests=6 depth_count=2 threshold_initialized=1 threshold=15640 depth0_clean=83 depth1_clean=9 global_stop=1`
- pass_marker_rank1: `MULTI_DEPTH_DISPATCHER_LOOP_BATCH_WORLD2_SMOKE_OK rank=1 tests=6 depth_count=2 threshold_initialized=1 threshold=15640 depth0_clean=97 depth1_clean=8 global_stop=1`
- pass_marker_completion: `=== MULTI_DEPTH_DISPATCHER_LOOP_BATCH_WORLD2_TEST_COMPLETE ===`
- validated_multi_depth_lifecycle: two depth iterations completed with real Stream1 score source, threshold initialization persisted into depth1, next-frontier copy preserved padding zero, and solved/stop flag propagated across depths.
- green_claim: true.

## 2026-05-18 architecture_v6_real_data_functional_validation_world2

- entity_id: `architecture_v6_real_data_functional_validation_world2`
- type: `batch_stage`
- state: `green`
- prompt_summary: User requested small real-data functional validation on Kaggle 2xT4 using real `test.csv` rows, real `puzzle_info.json`, real FullBeamNice Stream1, small task count, small max depth, submission-like output, and detailed per-task logs.
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- constraints_preserved: no performance tuning, no real solver quality claim, no leaderboard claim, no architecture deviation, no runtime 128-to-120 slice, no separate `nn_input_120_buffer`, no fallback backend, no new Stream3/4/5 logic.
- code_change: `tests/synthetic_full_depth_with_stream1_batch_world2_smoke.py` helper `run_stream1_stream2` accepts optional real State128-compatible state/generator/central inputs while preserving existing synthetic defaults.
- test_added: `tests/real_data_functional_validation_world2.py`
- static_guard_added: `tests/test_architecture_v6_static.py::test_real_data_functional_validation_world2_contract`
- kaggle_stage_added: `kaggle_real_data_functional_validation_world2_stage`
- kaggle_payload_updated_local: copied `tests/real_data_functional_validation_world2.py`, updated helper test file, and copied real `data/` directory into `kaggle_stream1_stream2_ring_batch_world2_payload_dataset`.
- host_verification: `python -m py_compile tests\real_data_functional_validation_world2.py tests\synthetic_full_depth_with_stream1_batch_world2_smoke.py tests\test_architecture_v6_static.py` passed.
- host_static_pytest: `python -m pytest tests\test_architecture_v6_static.py -q` passed with `36 passed`.
- kaggle_payload_update_status: `kaggle datasets version -p kaggle_stream1_stream2_ring_batch_world2_payload_dataset -m "architecture_v6_real_data_functional_validation_world2" --dir-mode zip` succeeded.
- kaggle_kernel: `trydotatwo/real-data-functional-validation-world2`
- kaggle_kernel_version: `1`
- kaggle_runtime: log confirmed `cuda_device_count=2`, `cuda_device_0=Tesla T4`, `cuda_device_1=Tesla T4`, `GPU 0: Tesla T4`, `GPU 1: Tesla T4`.
- kaggle_build: `setup.py build_ext --inplace` passed on Kaggle CUDA runtime.
- kaggle_result: `torchrun` returned `0`.
- per_task_status_rank0_task0: `REAL_DATA_TASK_STATUS rank=0 task_id=0 status=functional_checked depths=2 final_keep=8 threshold_initialized=1`
- per_task_status_rank1_task0: `REAL_DATA_TASK_STATUS rank=1 task_id=0 status=functional_checked depths=2 final_keep=9 threshold_initialized=1`
- per_task_status_rank0_task1: `REAL_DATA_TASK_STATUS rank=0 task_id=1 status=functional_checked depths=2 final_keep=12 threshold_initialized=1`
- per_task_status_rank1_task1: `REAL_DATA_TASK_STATUS rank=1 task_id=1 status=functional_checked depths=2 final_keep=7 threshold_initialized=1`
- output_file: `REAL_DATA_OUTPUT_FILE path=/kaggle/working/real_data_functional_validation_world2.csv rows=2`
- pass_marker_rank0: `REAL_DATA_FUNCTIONAL_VALIDATION_WORLD2_SMOKE_OK rank=0 tasks=2 max_depth=2 gathered_task_counts=[2, 2] no_leaderboard_claim=1 no_real_solver_quality_claim=1 no_performance_tuning_claim=1`
- pass_marker_rank1: `REAL_DATA_FUNCTIONAL_VALIDATION_WORLD2_SMOKE_OK rank=1 tasks=2 max_depth=2 gathered_task_counts=[2, 2] no_leaderboard_claim=1 no_real_solver_quality_claim=1 no_performance_tuning_claim=1`
- pass_marker_completion: `=== REAL_DATA_FUNCTIONAL_VALIDATION_WORLD2_TEST_COMPLETE ===`
- validated_real_data_functional_path: real `data/test.csv` rows and real `data/puzzle_info.json` action tables fed real FullBeamNice Stream1 and architecture_v6 functional validation path with small task count and max depth.
- green_claim: true.

## 2026-05-18 architecture_v6_real_data_larger_batch_correctness_world2

- entity_id: `architecture_v6_real_data_larger_batch_correctness_world2`
- type: `batch_stage`
- state: `green`
- prompt_summary: User accepted small real-data functional validation and requested larger real-data functional/correctness stability batch with modest task count/depth and without performance, leaderboard, or solver-quality claims.
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- constraints_preserved: no performance tuning, no real solver quality claim, no leaderboard claim, no architecture deviation, no runtime 128-to-120 slice, no separate `nn_input_120_buffer`, no fallback backend, no new Stream3/4/5 logic, no large beam, no full test.csv.
- test_added: `tests/real_data_larger_batch_correctness_world2.py`
- static_guard_added: `tests/test_architecture_v6_static.py::test_real_data_larger_batch_correctness_world2_contract`
- kaggle_stage_added: `kaggle_real_data_larger_batch_correctness_world2_stage`
- limits: `task_count=20`, `max_depth=5`, `beam=modest`.
- host_verification: `python -m py_compile tests\real_data_larger_batch_correctness_world2.py tests\real_data_functional_validation_world2.py tests\synthetic_full_depth_with_stream1_batch_world2_smoke.py tests\test_architecture_v6_static.py` passed.
- host_static_pytest: `python -m pytest tests\test_architecture_v6_static.py -q` passed with `37 passed`.
- kaggle_payload_update_status: `kaggle datasets version -p kaggle_stream1_stream2_ring_batch_world2_payload_dataset -m "architecture_v6_real_data_larger_batch_correctness_world2" --dir-mode zip` succeeded.
- kaggle_kernel: `trydotatwo/real-data-larger-batch-correctness-world2`
- kaggle_kernel_version: `1`
- kaggle_runtime: log confirmed `cuda_device_count=2`, `cuda_device_0=Tesla T4`, `cuda_device_1=Tesla T4`, `GPU 0: Tesla T4`, `GPU 1: Tesla T4`.
- kaggle_build: `setup.py build_ext --inplace` passed on Kaggle CUDA runtime.
- kaggle_result: `torchrun` returned `0`.
- per_task_status_count: at least `40` `REAL_DATA_LARGER_TASK_STATUS` records logged across rank0/rank1 for `20` tasks.
- output_file: `REAL_DATA_LARGER_OUTPUT_FILE path=/kaggle/working/real_data_larger_batch_correctness_world2.csv rows=20`
- accounting_rank0: `tasks=20`, `max_depth=5`, `solved=0`, `unsolved=20`, `total_depth_rows=100`, `gathered_counts=[[20,0,20,100],[20,0,20,100]]`.
- accounting_rank1: `tasks=20`, `max_depth=5`, `solved=0`, `unsolved=20`, `total_depth_rows=100`, `gathered_counts=[[20,0,20,100],[20,0,20,100]]`.
- pass_marker_rank0: `REAL_DATA_LARGER_BATCH_CORRECTNESS_WORLD2_SMOKE_OK rank=0 tasks=20 max_depth=5 solved=0 unsolved=20 total_depth_rows=100 gathered_counts=[[20, 0, 20, 100], [20, 0, 20, 100]] no_leaderboard_claim=1 no_real_solver_quality_claim=1 no_performance_tuning_claim=1 no_large_beam=1 no_full_test_csv=1`
- pass_marker_rank1: `REAL_DATA_LARGER_BATCH_CORRECTNESS_WORLD2_SMOKE_OK rank=1 tasks=20 max_depth=5 solved=0 unsolved=20 total_depth_rows=100 gathered_counts=[[20, 0, 20, 100], [20, 0, 20, 100]] no_leaderboard_claim=1 no_real_solver_quality_claim=1 no_performance_tuning_claim=1 no_large_beam=1 no_full_test_csv=1`
- pass_marker_completion: `=== REAL_DATA_LARGER_BATCH_CORRECTNESS_WORLD2_TEST_COMPLETE ===`
- validated_larger_real_data_stability: real `data/test.csv` rows `0..19` completed `max_depth=5` on both ranks with output CSV, per-task statuses, and solved/unsolved accounting; no crash/deadlock/OOM observed.
- green_claim: true.

## 2026-05-16 architecture_v6_implementation_start

- prompt_summary: User approved and requested implementation of Target Architecture Rewrite v6 for CayleyBeam100H100.
- architecture_v6: Runtime must move to `State128 + Hash128 + CandidateMeta + scratch_pool + dispatcher_outside_graph`.
- replace_required: Remove production dependence on `BeamMeta`, `HashSlot`, stream-phase `next_state_pool`, full-depth CUDA Graph capture, TorchScript backend, dummy backend, and central_hamming backend.
- memory_contract: `current_frontier_states`, solved result buffers, and stop flags live outside `scratch_pool`; `layout_streams` and `layout_final` overlay one physical `scratch_pool`; `GLOBAL_BEAM_WIDTH_MAX_SAFE` uses `current_frontier_states + max(layout_streams_bytes, layout_final_bytes) + model_weights_fp16 + read_only_tables + CUDA/NCCL/headroom`.
- fixed_types: Required public data types are `State128`, `Hash128`, `CandidateMeta`, `FinalRequest`, and `FinalResponse = State128`; `CandidateMeta` must stay 32 bytes and `Hash128` must stay 16 bytes.
- padding_contract: Persistent frontier states must keep `State128.v[120..127] = 0`; `FinalResponse.v[120..123]` temporarily stores `target_local_idx`; `generators[move][120..127]=120..127`; `central_state[120..127]=0`; `zobrist[120..127][*]=Hash128{0,0}`.
- solved_contract: Goal candidates use `GOAL_SCORE_KEY=0`; solved results are stored in bounded `solved_meta_list` and `solved_depth_list`; publishing solved results uses list writes before `solved_flag`/`stop_flag`, with `__threadfence_system()` for polling safety.
- threshold_rule: Add `threshold_initialized`; before initialization and insufficient survivors, `current_threshold=UINT32_MAX`; after sufficient survivors, updates use `current_threshold=min(current_threshold,new_threshold)` and never relax.
- implementation_order: First add split headers/config/memory layout and tests, then replace buffer allocation, then implement Stream1/2/3/5/4/final dispatcher pieces.
- verification_plan: Run Python compile/static tests locally; CUDA build verification may require Docker/Kaggle because local Windows CUDA host compiler availability is uncertain.
- code_change_status: implementation started; architecture deviations require explicit user approval.
- implementation_update: Added `beam_types.hpp`, `beam_config.hpp/cpp`, `beam_memory.hpp/cpp`, and `tests/test_architecture_v6_static.py`; wired new C++ sources into `setup.py` and runtime JIT build list.
- allocation_update: `beam_engine.py` now exposes v6 config fields and allocates `current_frontier_states`, `scratch_pool`, `solved_flag`, `stop_flag`, `solved_count`, `solved_overflow`, `solved_meta_list`, and `solved_depth_list` alongside legacy buffers during transition.
- derive_sizes_update: C++ `derive_sizes()` now reports v6 type sizes, score constants, ring/shard config, effective beam width, scratch overlay bytes, current frontier bytes, and solved buffer bytes.
- fallback_update: Python production configure path now rejects non-`fullbeamnice_static` inference backends for target v6.
- verification_result: `python -m py_compile beam_engine.py tests\test_architecture_v6_static.py` passed; `python -m pytest tests\test_architecture_v6_static.py -q` passed with 7 tests.
- cuda_build_status: local CUDA extension build not run because `where.exe cl` did not find MSVC `cl.exe`; Docker/Kaggle build remains required for CUDA compilation verification.
- next_phase_2026_05_16: User accepted foundation status and specified next order: Stream2 first, Final second, Stream3, Stream4, Stream5+dispatcher, Stream1 CUTLASS/custom last.
- stream2_phase_scope: Implement Stream2 v6 kernel foundation with `State128` local child state, padded generators/central/zobrist, `hash_ring:Hash128`, and bounded solved list path; add allocation/layout byte-size consistency test before deeper kernels.
- stream2_update: Added `beam_kernels_stream2.cu` with `kernel_v6_stream2_hash_goal` and launcher; kernel applies padded `generators[24][128]`, computes `Hash128` through `zobrist[128][128]`, writes `hash_ring`, and appends bounded goal results with `GOAL_SCORE_KEY=0`.
- solved_visibility_update: Stream2 goal path writes `solved_meta_list` and `solved_depth_list`, calls `__threadfence_system()`, then publishes `solved_flag` and `stop_flag`.
- build_wiring_update: Added `beam_kernels_stream2.cu` to `setup.py` and `beam_engine.py` extension source lists.
- stream2_static_tests: Extended v6 static tests to verify v6 allocation buffer names, Stream2 kernel symbols, solved visibility order, and scratch overlay max rule.
- stream2_verification_result: `python -m py_compile beam_engine.py tests\test_architecture_v6_static.py` passed; `python -m pytest tests\test_architecture_v6_static.py -q` passed with 10 tests.
- docker_stage0_result: Docker RTX 3070 CUDA build with `TORCH_CUDA_ARCH_LIST=8.6` and `python setup.py build_ext --inplace` passed after build-only fix `std::max<int64_t>(...)` in `beam_engine.cpp`.
- docker_stage0_static_tests: Container image initially lacked `pytest`; ephemeral `python -m pip install pytest -q` inside Docker succeeded; `python -m pytest tests/test_architecture_v6_static.py -q` passed with 10 tests.
- stream2_gpu_reference_test: Added `tests/stream2_reference_smoke.py`; Docker GPU run passed and printed `STREAM2_REFERENCE_SMOKE_OK`.
- stream2_reference_checks: Verified GPU `hash_ring` equals CPU Zobrist reference, goal writes solved list, `score_key == GOAL_SCORE_KEY`, route packing is preserved, and zero Zobrist padding makes `State128.v[120..127]` hash-neutral.
- next_required_work: Do not proceed to Stream3/4/5/dispatcher yet; next architecture stage is Final materialization using the same `apply_move` contract.
- final_stage_update: Added `beam_kernels_final.cu` with `kernel_v6_final_materialize` and `kernel_v6_final_scatter_responses`; added pybind exports `v6_final_materialize` and `v6_final_scatter_responses`.
- final_smoke_test: Added `tests/final_materialization_smoke.py`; verifies GPU child state equals CPU apply_move, `FinalResponse.v[120..123]` stores `target_local_idx`, scatter writes `next_frontier_states_tmp[target_local_idx]`, and persistent padding `v[120..127]` is zero.
- final_verification_result: Docker RTX 3070 run passed `python setup.py build_ext --inplace`, `python -m pytest tests/test_architecture_v6_static.py -q` with 11 tests, `python tests/stream2_reference_smoke.py`, and `python tests/final_materialization_smoke.py`.
- next_required_work_after_final: Stream3 isolated test may start next; Stream4/Stream5/dispatcher/Stream1 still must wait for their own stages.
- stream3_stage_start: User approved Stage3 Stream3 isolated implementation only: one ring / one `STREAM3_BATCH_CANDIDATES`, no Stream4, no Stream5, no dispatcher, no Stream1 changes.
- stream3_stage_scope: Implement threshold+compact, CUB sort/reduce by `Hash128`, dedup with `min(stream3_val)`, owner after dedup, local/remote split, remote grouping by owner, and smoke test `tests/stream3_dedup_smoke.py`.
- stream3_implementation_update: Added `beam_kernels_stream3.cu` with `kernel_v6_stream3_pack_threshold_compact`, CUB `DeviceRadixSort::SortPairs` using `Hash128KeyDecomposer`, sorted segment dedup by `Hash128`, and `kernel_v6_stream3_restore_split`.
- stream3_test_update: Added `tests/stream3_dedup_smoke.py`; verifies threshold filtering, original `payload_id` preservation, dedup `min(stream3_val)`, score and payload tie behavior, owner computed after dedup, parent/move restoration, local pending output, remote grouping, and send counts/offsets.
- stream3_build_fix: Initial CUB sort compile failed for custom `Hash128Key` without decomposer; fixed with CUB custom decomposer returning `(hi, lo)` tuple, preserving sort order `hash.hi, hash.lo`.
- stream3_verification_result: Host `py_compile` passed; host static pytest passed with 12 tests; Docker RTX 3070 run passed `python setup.py build_ext --inplace`, static pytest with 12 tests, Stream2 smoke, Final smoke, and Stream3 smoke.
- next_required_work_after_stream3: Stream4 isolated shard merge may start next; Stream5/dispatcher/Stream1 remain blocked until their stages.
- stream4_stage_start: User approved Stage4 Stream4 isolated shard merge only: one shard job, no Stream5, no dispatcher, no Stream1, no global threshold.
- stream4_stage_scope: Implement threshold+compact, CUB sort by `Hash128`, dedup with best `CandidateMeta` by `min(score_key), min(parent_idx), min(route_packed)`, write clean region, set `dirty_count=0`, update `clean_count`, and clear `processing_flag`.
- stream4_implementation_update: Added `beam_kernels_stream4.cu` with `kernel_v6_stream4_threshold_compact`, CUB `DeviceRadixSort::SortPairs` using `Stream4HashKeyDecomposer`, sorted segment dedup, and clean-region writeback.
- stream4_test_update: Added `tests/stream4_shard_smoke.py`; verifies threshold filtering, clean+dirty merge, duplicate hash selection, score tie deterministic parent/route tie-break, dirty reset, clean count update, processing flag reset, and no shard cap behavior.
- stream4_verification_result: Host `py_compile` passed; host static pytest passed with 13 tests; Docker RTX 3070 run passed `python setup.py build_ext --inplace`, static pytest with 13 tests, Stream2 smoke, Final smoke, Stream3 smoke, and Stream4 smoke.
- next_required_work_after_stream4: Stream5 isolated CandidateMeta exchange may start next; dispatcher and Stream1 remain blocked until later stages.
- stream5_stage_start: User approved Stage5 Stream5 isolated CandidateMeta exchange only: no dispatcher, no Stream1, no full depth loop, no Stream3/4 changes.

## 2026-05-16 stage6_dispatcher_skeleton_resume_after_threshold_patch

- prompt_summary: User provided comprehensive Stage6 dispatcher skeleton resumption plan after `pybind int rejects UINT32_MAX` failure.
- last_failure: pybind binding rejected `UINT32_MAX` literal for `current_threshold`; binding expected signed int instead of uint32_t.
- last_patch_applied: Changed `dispatcher_skeleton_smoke` test to use `current_threshold=100` instead of `UINT32_MAX`; preserves test intent (`score_key=[5,10,20]` kept, `[1000]` dropped) while working around binding type mismatch.
- current_action: Rerun host and Docker tests with already-applied patch; verify dispatcher smoke passes without new changes; if failure persists, fix only Stage6 control-plane/smoke/binding issues.
- hard_constraints_for_fixes: [no_Stream1_changes, no_Stream3_semantics, no_Stream4_semantics, no_Stream5_semantics, no_custom_sort, no_fallback_backend, no_full_depth_loop].
- allowed_fixes: [python_dispatcher_skeleton_only, test_only_threshold_types, pybind_wrapper_type_mismatch, tensor_shape_stride_layout_mismatch, isolated_function_reordering].
- host_test_status: `python -m py_compile beam_engine.py tests\test_architecture_v6_static.py tests\dispatcher_skeleton_smoke.py` passed with no output; `python -m pytest tests\test_architecture_v6_static.py -q` passed 15/15 tests.
- docker_build_status: local Windows MSVC not available (`where cl` failed); Docker image `cayley-beam-h100:latest` exists and is ready; pre-compiled `.so` file found at `build/lib.linux-x86_64-cpython-311/beam_engine_ext.cpython-311-x86_64-linux-gnu.so`.
- continuation_plan: Docker build_ext and all smoke tests via Kaggle Notebook environment due to local Docker mount/path encoding issues on Windows PowerShell terminal.
- docker_local_gpu_blocker: Docker Desktop on Windows cannot access local NVIDIA GPU; `nvidia-container-cli: device error: 1: unknown device` when trying `docker-compose run --rm beam-2h100 ...`. Host-only RTX3070 CUDA build unavailable; Kaggle 2xT4 GPU remains required for CUDA smoke test verification.
- stage6_smoke_command_ready: Created `run_stage6_smoke_tests.sh` bash script with full test sequence: build_ext → pytest static → stream2 smoke → final smoke → stream3 smoke → stream4 smoke → stream5 smoke → dispatcher smoke. Ready to paste into Kaggle notebook cell for execution.
- next_after_green: Execute stage6_smoke_command in Kaggle 2xT4 notebook; if all tests pass (expected output DISPATCHER_SKELETON_SMOKE_OK), update PROJECT_MEMORY with `stage6_dispatcher_skeleton = green_world1`; then proceed to micro-stage pybind uint32_t binding fix; then choose Kaggle 2xT4 Stream5 NCCL continuation or Stream1 CUTLASS.
- stream5_stage_scope: Implement/test `CandidateMeta` byte-identical exchange with `send_count/send_offset` and `recv_count/recv_offset`; first support `WORLD_SIZE=1`, then `torchrun --standalone --nproc_per_node=2` multi-rank smoke.
- stream5_implementation_update: Added `BeamEngine.v6_stream5_exchange_candidate_meta(...)`; `WORLD_SIZE=1` path copies `CandidateMeta` records device-to-device; `WORLD_SIZE>1` path exchanges counts and payload bytes through NCCL `ncclSend`/`ncclRecv` using precomputed `send_count/send_offset` and `recv_count/recv_offset`.
- stream5_test_update: Added `tests/stream5_exchange_smoke.py`; test packs `CandidateMeta` as 32-byte records and checks byte-identical preservation of `hash`, `parent_idx`, `score_key`, and `route_packed`; static test now verifies Stream5 binding contract.
- stream5_host_verification_result: `python -m py_compile beam_engine.py tests\test_architecture_v6_static.py tests\stream5_exchange_smoke.py` passed; `python -m pytest tests\test_architecture_v6_static.py -q` passed with 14 tests.
- stream5_docker_single_result: Docker RTX 3070 run passed `python setup.py build_ext --inplace`, static pytest with 14 tests, Stream2 smoke, Final smoke, Stream3 smoke, Stream4 smoke, and `WORLD_SIZE=1` Stream5 smoke; output included `STREAM5_EXCHANGE_SMOKE_OK rank=0 world=1`.
- stream5_multirank_local_limit: `torchrun --standalone --nproc_per_node=2 tests/stream5_exchange_smoke.py` cannot execute NCCL exchange on the local one-visible-GPU RTX 3070 Docker host; raw NCCL failure was `Duplicate GPU detected : rank 0 and rank 1 both on CUDA device 1000`.
- stream5_multirank_guard: `tests/stream5_exchange_smoke.py` now exits successfully with explicit `STREAM5_EXCHANGE_SMOKE_SKIPPED world=2 visible_cuda_devices=1 reason=NCCL_requires_distinct_visible_GPU_per_rank` when visible CUDA devices are fewer than `WORLD_SIZE`; actual two-GPU NCCL byte exchange remains pending for Kaggle 2xT4 or another host with at least two visible GPUs.
- next_required_work_after_stream5: Dispatcher skeleton can start only after user accepts the local single-rank Stream5 result and the explicit two-GPU NCCL verification limitation; Stream1 CUTLASS/custom remains blocked until later stage.
- stage6_dispatcher_skeleton_result: User confirmed Stage6 dispatcher skeleton as `green_world1` from Kaggle CUDA log with `15 passed`, `STREAM2_REFERENCE_SMOKE_OK`, `FINAL_MATERIALIZATION_SMOKE_OK`, `STREAM3_DEDUP_SMOKE_OK`, `STREAM4_SHARD_SMOKE_OK`, `STREAM5_EXCHANGE_SMOKE_OK rank=0 world=1`, `DISPATCHER_SKELETON_SMOKE_OK`, and `returncode=0`.
- threshold_binding_fix_scope: User required micro-stage `architecture_v6_threshold_binding_fix` before Stream5 2GPU NCCL and before Stream1 CUTLASS; scope limited to pybind/Python compatibility for `current_threshold = UINT32_MAX = 0xffffffff`.
- threshold_binding_fix_update: Changed Stream3 and Stream4 pybind threshold parameters from signed `int` to `uint64_t` with explicit `<= 0xffffffff` range checks, restored dispatcher skeleton `current_threshold=0xffffffff`, and added Python `_v6_validate_u32(...)` validation.
- threshold_binding_fix_test_update: Static test now asserts uint32 threshold binding contract; dispatcher skeleton smoke expectation now matches `UINT32_MAX` behavior where Stream3 compact keeps all `B_MICRO * MOVE_COUNT = 48` candidates before dedup.
- threshold_binding_fix_verification_result: Host `python -m py_compile beam_engine.py tests\test_architecture_v6_static.py tests\stream2_reference_smoke.py tests\final_materialization_smoke.py tests\stream3_dedup_smoke.py tests\stream4_shard_smoke.py tests\stream5_exchange_smoke.py tests\dispatcher_skeleton_smoke.py` passed; host `python -m pytest tests\test_architecture_v6_static.py -q` passed with 16 tests.
- threshold_binding_fix_docker_result: Docker RTX 3070 run passed `python setup.py build_ext --inplace`, static pytest with 16 tests, Stream2 smoke, Final smoke, Stream3 smoke, Stream4 smoke, Stream5 WORLD_SIZE=1 smoke, and dispatcher skeleton smoke with `DISPATCHER_SKELETON_SMOKE_OK`.
- next_required_work_after_threshold_fix: Stream5 2GPU NCCL explicit `torchrun` test on Kaggle 2xT4 is the next architecture validation candidate; Stream1 CUTLASS/custom remains blocked until user decision after the 2GPU NCCL step.
- stream5_2gpu_nccl_explicit_stage: `architecture_v6_stream5_2gpu_nccl_explicit_test` is green on Kaggle 2xT4.
- stream5_2gpu_nccl_explicit_runner: Kaggle notebook kernel `trydotatwo/stream5-2gpu-nccl-explicit-notebook` ran `python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 tests/stream5_2gpu_nccl_explicit_smoke.py`.
- stream5_2gpu_nccl_explicit_runtime: log confirmed `cuda_device_count=2`, `GPU 0: Tesla T4`, and `GPU 1: Tesla T4`; notebook runtime gate rejected non-2xT4 hardware.
- stream5_2gpu_nccl_explicit_result: both ranks printed success markers: `STREAM5_2GPU_NCCL_EXPLICIT_SMOKE_OK rank=0 sent=1 received=3` and `STREAM5_2GPU_NCCL_EXPLICIT_SMOKE_OK rank=1 sent=3 received=1`; notebook printed `=== STREAM5_2GPU_NCCL_EXPLICIT_TEST_COMPLETE ===`.
- stream5_2gpu_nccl_explicit_validated: validated `CandidateMeta` byte identity, `hash`, `parent_idx`, `score_key`, `route_packed`, `send_count`, `send_offset`, `recv_count`, and `recv_offset` for asymmetric rank0->rank1 count 1 and rank1->rank0 count 3.
- stream5_2gpu_nccl_explicit_method: successful path used notebook kernel staging with `kernel_type=notebook`, notebook metadata `kaggle.accelerator=nvidiaTeslaT4`, Kaggle CLI push `kaggle kernels push -p kaggle_stream5_2gpu_nccl_notebook_stage --accelerator NvidiaTeslaT4`, and Kaggle-safe NCCL env `NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=1`, `NCCL_SOCKET_IFNAME=lo`, `GLOO_SOCKET_IFNAME=lo`.
- stream5_2gpu_nccl_cli_note: local Kaggle CLI log/status calls may require clearing proxy env variables before `kaggle kernels logs/status` when VPN/proxy interferes.
- stream5_2gpu_nccl_scope_preserved: no Stream1 integration, no dispatcher expansion, no Stream3 collector integration, no Stream4 scheduler integration, and no final materialization expansion were performed during the explicit 2GPU Stream5 validation.
- next_required_work_after_stream5_2gpu_nccl_green: next allowed stage is `architecture_v6_stream5_dispatcher_binding_world2` or separate user-approved Stream1 CUTLASS planning; architecture_v6 production Stream1 remains unstarted.
- dispatcher_stream5_world2_stage: `architecture_v6_stream5_dispatcher_binding_world2` is green on Kaggle 2xT4.
- dispatcher_stream5_world2_code_update: Added dispatcher-level Python binding `v6_dispatcher_skeleton_world2_stream5_smoke(...)` that launches existing `BeamEngine.v6_stream5_exchange_candidate_meta(...)` with `WORLD_SIZE=2`, existing `CandidateMeta` buffers, existing `send_count/send_offset/recv_count/recv_offset`, and no Stream1/model backend work.
- dispatcher_stream5_world2_test_update: Added `tests/dispatcher_stream5_world2_smoke.py`; test requires exactly two visible Tesla T4 devices, runs under `torch.distributed.run --standalone --nproc_per_node=2`, checks dispatcher-launched Stream5 exchange, validates asymmetric rank0->rank1 count 2 and rank1->rank0 count 4, validates byte-identical `CandidateMeta` payloads, and asserts no Stream1/fallback/Stream3 collector/Stream4 scheduler/final expansion flags.
- dispatcher_stream5_world2_static_result: Host `python -m py_compile beam_engine.py tests\test_architecture_v6_static.py tests\dispatcher_stream5_world2_smoke.py` passed; host `python -m pytest tests\test_architecture_v6_static.py -q` passed with 18 tests.
- dispatcher_stream5_world2_kaggle_method: Successful path used notebook kernel stage `kaggle_dispatcher_stream5_world2_notebook_stage`, metadata `kernel_type=notebook`, notebook metadata `kaggle.accelerator=nvidiaTeslaT4`, CLI push `kaggle kernels push -p kaggle_dispatcher_stream5_world2_notebook_stage --accelerator NvidiaTeslaT4`, and proxy env cleared for Kaggle CLI calls.
- dispatcher_stream5_world2_kaggle_result: Kernel `trydotatwo/dispatcher-stream5-world2-smoke` completed; log confirmed `cuda_device_count 2`, `cuda_device_0=Tesla T4`, `cuda_device_1=Tesla T4`, build_ext passed, and torchrun executed `tests/dispatcher_stream5_world2_smoke.py`.
- dispatcher_stream5_world2_pass_markers: log contained `DISPATCHER_STREAM5_WORLD2_SMOKE_OK rank=0 sent=2 received=4`, `DISPATCHER_STREAM5_WORLD2_SMOKE_OK rank=1 sent=4 received=2`, and `=== DISPATCHER_STREAM5_WORLD2_TEST_COMPLETE ===`.
- dispatcher_stream5_world2_scope_preserved: no Stream1 integration, no model backend work, no Stream3 collector complexity expansion, no Stream4 scheduler expansion, no final materialization expansion, and no threshold logic change were performed.
- next_required_work_after_dispatcher_stream5_world2_green: architecture_v6 can proceed only after user selects next micro-stage; likely candidates are Stream1 CUTLASS/custom planning or a separate larger-count Stream5/dispatcher stress smoke.
- dispatcher_stream3_stream5_collector_world2_stage: `architecture_v6_dispatcher_stream3_stream5_collector_world2_smoke` is green on Kaggle 2xT4.
- dispatcher_stream3_stream5_collector_world2_code_update: Added `v6_dispatcher_stream3_stream5_collector_world2_smoke(...)` in `beam_engine.py`; function builds synthetic `CandidateMeta` records, performs deterministic local/remote owner split for `WORLD_SIZE=2`, launches existing `BeamEngine.v6_stream5_exchange_candidate_meta(...)`, and ingests `local_pending_buffer + remote_recv_buffer` into a synthetic survivor dirty region.
- dispatcher_stream3_stream5_collector_world2_test_update: Added `tests/dispatcher_stream3_stream5_collector_world2_smoke.py`; test requires exactly two visible Tesla T4 devices, runs under `torch.distributed.run --standalone --nproc_per_node=2`, checks remote payload byte identity, collector source order, dirty count, and scope flags.
- dispatcher_stream3_stream5_collector_world2_static_result: Host `python -m py_compile beam_engine.py tests\test_architecture_v6_static.py tests\dispatcher_stream3_stream5_collector_world2_smoke.py` passed; host `python -m pytest tests\test_architecture_v6_static.py -q` passed with 19 tests.
- dispatcher_stream3_stream5_collector_world2_kaggle_method: Successful path used notebook kernel stage `kaggle_dispatcher_stream3_stream5_collector_world2_stage`, metadata `kernel_type=notebook`, notebook metadata `kaggle.accelerator=nvidiaTeslaT4`, CLI push `kaggle kernels push -p kaggle_dispatcher_stream3_stream5_collector_world2_stage --accelerator NvidiaTeslaT4`, and proxy env cleared for Kaggle CLI calls.
- dispatcher_stream3_stream5_collector_world2_kaggle_result: Kernel `trydotatwo/dispatcher-stream3-stream5-collector-world2` completed; log confirmed `cuda_device_count 2`, `cuda_device_0=Tesla T4`, `cuda_device_1=Tesla T4`, build_ext passed, and torchrun executed `tests/dispatcher_stream3_stream5_collector_world2_smoke.py`.
- dispatcher_stream3_stream5_collector_world2_pass_markers: log contained `DISPATCHER_STREAM3_STREAM5_COLLECTOR_WORLD2_SMOKE_OK rank=0 local=2 sent=3 dirty=6`, `DISPATCHER_STREAM3_STREAM5_COLLECTOR_WORLD2_SMOKE_OK rank=1 local=3 sent=4 dirty=6`, and `=== DISPATCHER_STREAM3_STREAM5_COLLECTOR_WORLD2_TEST_COMPLETE ===`.
- dispatcher_stream3_stream5_collector_world2_scope_preserved: no Stream1 integration, no model backend work, no full dispatcher loop, no Stream4 scheduler expansion, no shard dirty/clean lifecycle expansion, no threshold logic change, and no final materialization expansion were performed.
- next_required_work_after_dispatcher_stream3_stream5_collector_world2_green: architecture_v6 can proceed only after user selects next micro-stage; likely candidates are minimal Stream4 scheduler binding after collector dirty region, larger-count Stream5/collector stress smoke, or Stream1 CUTLASS/custom planning.
- collector_shard_dirty_spill_world2_stage: `architecture_v6_collector_shard_dirty_spill_world2_smoke` is green on Kaggle 2xT4.
- collector_shard_dirty_spill_world2_code_update: Added `v6_collector_shard_dirty_spill_world2_smoke(...)` in `beam_engine.py`; function builds synthetic `CandidateMeta` inputs, performs deterministic `WORLD_SIZE=2` owner split, launches existing `BeamEngine.v6_stream5_exchange_candidate_meta(...)`, writes shard-free candidates into `survivor_shard` dirty region, and writes candidates whose shard has `processing_flag[shard] == true` into `global_spill_buffer`.
- collector_shard_dirty_spill_world2_test_update: Added `tests/collector_shard_dirty_spill_world2_smoke.py`; test requires exactly two visible Tesla T4 devices, runs under `torch.distributed.run --standalone --nproc_per_node=2`, checks remote payload byte identity, dirty-region write, global-spill write, and scope flags.
- collector_shard_dirty_spill_world2_static_result: Host `python -m py_compile beam_engine.py tests\test_architecture_v6_static.py tests\collector_shard_dirty_spill_world2_smoke.py` passed; host `python -m pytest tests\test_architecture_v6_static.py -q` passed with 20 tests.
- collector_shard_dirty_spill_world2_kaggle_method: Successful path used notebook kernel stage `kaggle_collector_shard_dirty_spill_world2_stage`, metadata `kernel_type=notebook`, notebook metadata `kaggle.accelerator=nvidiaTeslaT4`, CLI push `kaggle kernels push -p kaggle_collector_shard_dirty_spill_world2_stage --accelerator NvidiaTeslaT4`, and proxy env cleared for Kaggle CLI calls.
- collector_shard_dirty_spill_world2_kaggle_result: Kernel `trydotatwo/collector-shard-dirty-spill-world2` completed; log confirmed `cuda_device_count 2`, `cuda_device_0=Tesla T4`, `cuda_device_1=Tesla T4`, build_ext passed, and torchrun executed `tests/collector_shard_dirty_spill_world2_smoke.py`.
- collector_shard_dirty_spill_world2_pass_markers: log contained `COLLECTOR_SHARD_DIRTY_SPILL_WORLD2_SMOKE_OK rank=0 dirty=4 spill=4`, `COLLECTOR_SHARD_DIRTY_SPILL_WORLD2_SMOKE_OK rank=1 dirty=4 spill=4`, and `=== COLLECTOR_SHARD_DIRTY_SPILL_WORLD2_TEST_COMPLETE ===`.
- collector_shard_dirty_spill_world2_scope_preserved: no Stream1 integration, no model backend work, no full dispatcher loop, no Stream4 kernel launch, no Stream4 scheduler expansion, no clean/dirty lifecycle after Stream4, no threshold logic change, and no final materialization expansion were performed.
- next_required_work_after_collector_shard_dirty_spill_world2_green: architecture_v6 can proceed only after user selects next micro-stage; likely candidates are minimal Stream4 scheduler binding after collector dirty region, Stream4 flush integration, or Stream1 CUTLASS/custom planning.

## 2026-05-15 tier2b_architecture_revised

- prompt_summary: User clarified TIER 2B architecture: full states should NOT be stored in next_state_pool; only materialized when they become frontier buffer for next depth.
- key_insight: Current 32GB next_state_pool stores all 2B candidates, but only ~2M become survivors. States should be deferred until finalize_survivors phase.
- architecture_change: Three-buffer model instead of two:
  1. **current_state_pool** (immutable, 18GB): Active beam states for current depth
  2. **compact_metadata** (hot, 4.8GB): K_WORK × 24B (parent_idx, move, score, hash, fingerprint)
  3. **frontier_buffer** (18GB): Filled ONLY at end of depth with full states, becomes next_state_pool
- memory_impact: Remove 32GB next_state_pool (full states), replace with 4.8GB compact metadata + 18GB frontier (materialized once/depth). Total: 56GB → 40GB per GPU (~29% savings).
- pipeline_change: compact metadata flows through dedup/prune/compact → survivors (48MB) → finalize_survivors materializes directly into frontier_buffer. Full states never stored intermediate.
- risk_mitigation: Stack-local state reconstruction validated in BEAM_DEBUG mode; register pressure profiled with NCU; frontier capacity asserted at finalize.
- expected_impact: 4.8x memory write reduction (240GB → 50GB per depth).
- status: TIER 2B proposal revised and ready for approval. Implementation can begin only after user confirms this architecture.

## 2026-05-15 docker_rtx3070_profile_smoke

- prompt_summary: User suggested using Docker for local RTX 3070 testing/profiling instead of installing Windows MSVC toolchain.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- docker_status: Docker Desktop engine reachable with elevated pipe access; `cayley-beam-h100:latest` runs with `--gpus all`; container sees the local RTX 3070 GPU.
- container_toolchain: container provides Python/Torch/CUDA build environment and Nsight Compute CLI `ncu` version `2025.1.0`; `nsys` is not in PATH in the image, but earlier image path discovery found Nsight Systems binaries under the Nsight Compute install tree.
- local_run_config: `WORLD_SIZE=1`, `GLOBAL_BEAM_WIDTH=2_000_000`, `TEST_START=8`, `TEST_COUNT=1`, `MAX_DEPTH=5`, `HISTORY_BACKEND=cpu`, `CPU_HISTORY_CHECKPOINT=0`, `INFERENCE_BACKEND=fullbeamnice_static`, `PREPASS_NO_INFERENCE=1`, `PREPASS_DEPTH=5`, `PREPASS_STOP_AT_WIDTH=0`, `PREPASS_DEDUP_FACTOR=1.0`, `TORCH_CUDA_ARCH_LIST=8.6`.
- local_run_result: Docker smoke run passed; no bucket/hash overflow; prepass used logical capacities and same static arrays; final `FULL_BEAM_START` logged after depth 5.
- local_timing_result: depth5 uniform prepass on RTX 3070 measured `total_cuda_event_ms=61.139`, `micro_pipeline_ms=36.323`, `final_prune_compact_found_ms=24.051`, `current_size_sum=1_336_269`, `logical_next_limit=2_000_000`.
- profiler_limitation: Nsight Compute speed-of-light report file was generated earlier but full hardware counters were blocked by `ERR_NVGPUCTRPERM`; Docker Desktop/WSL Nsight Systems report generation did not expose CUDA kernel data in exported stats.
- code_change_status: documentation memory only; no algorithm/runtime logic modified in this record.

## 2026-05-15 nsight_compute_counters_unlocked

- prompt_summary: User unlocked NVIDIA GPU performance counters and asked how to connect the Nsight Compute GUI for normal analysis.
- validation_result: `ncu --set basic` inside Docker successfully profiled a PyTorch CUDA kernel; hardware counters are available.
- project_profile_result: `ncu --set basic --kernel-name regex:kernel_process_score_slot --launch-count 5` successfully profiled the project Stream2 kernel and exported `runtime/nsight/process_score_slot_basic_unlocked.ncu-rep`.
- key_profile_observation: first profiled launches had small grids (`1`, `3`, `52`, `64`, `64` blocks); Nsight Compute reported low achieved occupancy and launch configuration underutilization for early/small tiles.
- gui_usage: open existing `.ncu-rep` report from Windows Nsight Compute GUI via `File -> Open File`; direct GUI profiling of Docker requires configuring `Start Activity` target as `docker.exe` with full `docker run ... ncu ...` command, but opening CLI-generated reports is simpler and reproducible.
- followup_profile_attempt: larger `GLOBAL_BEAM_WIDTH=18_000_000` ncu run failed with driver resource/performance-counter unavailable message while Nsight Compute GUI was open; likely concurrent counter ownership/tool conflict, not project runtime failure; `docker ps` showed no leftover running container.
- code_change_status: documentation memory only; no algorithm/runtime logic modified in this record.

## 2026-05-15 stream2_ncu_k_expand_tile_ab

- prompt_summary: User closed Nsight Compute GUI and requested running the profile; user also asked to analyze Stream2 behavior and implementation issue.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- sanity_check: `ncu --set basic` on a small PyTorch CUDA workload succeeded after GUI close; performance counters remained unlocked.
- failed_large_case: plain `GLOBAL_BEAM_WIDTH=18_000_000`, `PREPASS_DEPTH=6`, `PREPASS_DEDUP_FACTOR=1.0` ended with `hash_overflow=598092` at depth 6 because prepass `logical_next_limit` was clamped to `GLOBAL_BEAM_WIDTH`, while inserted candidates exceeded 18M; case is invalid for performance comparison without larger next capacity or shallower depth.
- baseline_profile: `GLOBAL_BEAM_WIDTH=12_000_000`, `K_EXPAND_TILE=16384`, depth5 profile exported `runtime/nsight/process_score_slot_12m_depth5_basic.ncu-rep`; profiled `kernel_process_score_slot` launches used mostly `grid=64`, `waves_per_sm=0.27`, `achieved_occupancy≈24-26%`, `memory_throughput≈39-41%`, `compute_throughput≈5-8%`, showing small-grid and memory/irregular-work bottleneck.
- ab_profile: `GLOBAL_BEAM_WIDTH=12_000_000`, `K_EXPAND_TILE=65536`, depth5 profile exported `runtime/nsight/process_score_slot_12m_depth5_k65536_basic.ncu-rep`; profiled launches used mostly `grid=256`, `waves_per_sm=1.07`, `achieved_occupancy≈74-82%`, `memory_throughput≈44-50%`, `compute_throughput≈6-19%`.
- runtime_ab_result: with `K_EXPAND_TILE=16384`, depth5 `micro_pipeline_ms≈29.56` and `expand_tiles_upper_bound=157`; with `K_EXPAND_TILE=65536`, depth5 `micro_pipeline_ms≈17.46` and `expand_tiles_upper_bound=42`; improvement is about `1.69x` for the Stream2 micro-pipeline on local RTX 3070 single-GPU prepass.
- analysis_result: primary observed issue is over-tiling into many small `kernel_process_score_slot` launches; small launches underfill SMs and pay launch/sync/event overhead repeatedly; larger tile improves occupancy and reduces launch count, while remaining kernel behavior is still memory/irregular-hash dominated.
- code_change_status: documentation memory only; no algorithm/runtime logic modified in this record.

## 2026-05-15 full_cuda_graph_profile_sample0_beam14m

- prompt_summary: User requested normal full run on sample/state 0 with profiler only, no debug logging, CUDA Graph enabled, `GLOBAL_BEAM_WIDTH=14_000_000`, `BETA=1.05`, and bottleneck analysis including Tensor Core usage and stream interaction.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- config: `WORLD_SIZE=1`, `TEST_START=0`, `TEST_COUNT=1`, `GLOBAL_BEAM_WIDTH=14_000_000`, `BETA=1.05`, `B_MICRO=8192`, `K_EXPAND_TILE=65536`, `INFERENCE_BACKEND=fullbeamnice_static`, `USE_CUDA_GRAPHS=1`, `BEAM_DEBUG=0`, `DEPTH_LOG_EVERY=0`, `DEPTH_TUNING_LOG=0`, `PREPASS_DEPTH=5`, `PREPASS_STOP_AT_WIDTH=1`.
- nsys_result: bounded Nsight Systems report `runtime/nsight/nsys_full_sample0_beam14m_k65536_cudagraph_120s.nsys-rep` was generated, but exported stats contained CUDA API only and no CUDA GPU trace/kernel data under Docker/WSL.
- cuda_graph_status: full NCU runs printed `cuda_graph_captured_sum=1`, confirming CUDA Graph was active in the release/no-debug path.
- cutlass_profile: NCU report `runtime/nsight/cutlass_kernel_regex_full_sample0_beam14m_k65536_cudagraph_basic.ncu-rep` captured generic CUTLASS `Kernel` launches; representative durations were about `204-940 us`, SM throughput about `33-43%`, memory throughput about `17-38%`, occupancy about `15-17%`, registers about `206-224`.
- tensor_core_metric: NCU report `runtime/nsight/cutlass_tensorcore_metric_sample0_beam14m.ncu-rep` showed Tensor Core HMMA instructions on CUTLASS kernels: `sm__inst_executed_pipe_tensor_op_hmma.sum` values included `6_291_456` and `2_097_152`; Tensor Cores are active.
- stream2_profile: NCU report `runtime/nsight/process_score_slot_full_sample0_beam14m_k65536_cudagraph_basic.ncu-rep` captured `kernel_process_score_slot` in full CUDA Graph run; representative launches used `grid=256`, `waves_per_sm=1.07`, occupancy about `73-78%`, memory/L2 throughput about `37-49%`, compute throughput about `8-18%`, duration about `437-986 us`.
- analysis_result: Stream2 `process_score_slot` remains memory/irregular-hash dominated; K_EXPAND_TILE=65536 fixed the small-grid issue but hash/dedup/candidate materialization still has low compute throughput. CUTLASS scorer uses Tensor Cores, but Nsight Systems GPU trace is unavailable in the current Docker/WSL path, so stream overlap cannot be visually proven from timeline yet.
- code_change_status: documentation memory only; no algorithm/runtime logic modified in this record.

## 2026-05-15 per_stage_ncu_profile_sample0_beam14m

- prompt_summary: User requested per-stage profiling on real run to quantify which actions slow the code and each action contribution.
- config: `WORLD_SIZE=1`, `TEST_START=0`, `GLOBAL_BEAM_WIDTH=14_000_000`, `BETA=1.05`, `K_EXPAND_TILE=65536`, `USE_CUDA_GRAPHS=1`, `BEAM_DEBUG=0`, `DEPTH_LOG_EVERY=0`, `DEPTH_TUNING_LOG=0`, `PREPASS_DEPTH=5`.
- artifact_prepass_tail: `runtime/nsight/all_kernels_sample0_beam14m_k65536_cudagraph_basic.ncu-rep`; first 120 captured launches mostly cover prepass/tail before full neural scorer.
- artifact_full_start: `runtime/nsight/full_solver_kernels_sample0_beam14m_k65536_cudagraph_basic.ncu-rep`; `launch-skip=120`, `launch-count=160`; captured prepass tail plus start of full scorer.
- prepass_tail_profile: over captured kernels, `kernel_process_score_slot` dominated with `24_062 us` of `29_333 us` (`82.0%`), followed by CUTLASS scorer kernels sampled after full start at `3_550 us` (`12.1%`); this mixed capture is not a clean full-depth percentage.
- full_solver_start_profile: captured first 20 full neural scorer kernels only; CUTLASS GEMM kernels consumed `3_549.6 us` of `4_444.1 us` (`79.9%`), fill-bias `357.5 us` (`8.1%`), fill-residual-bias `295.5 us` (`6.7%`), final GEMM `204.1 us` (`4.6%`), quantize `24.9 us` (`0.6%`), embed `12.4 us` (`0.3%`).
- caveat: NCU kernel replay perturbs runtime and launch-skip slicing mixes prepass/full phases; for exact wall-clock per-stage percentages across a whole depth, code-level CUDA event phase timers are still needed because Nsight Systems GPU timeline is unavailable under current Docker/WSL setup.
- immediate_interpretation: after full scorer starts, Stream1 scorer has nontrivial cost; during prepass/Stream2 processing, `kernel_process_score_slot` remains dominant. The next profiling/code step should add release-safe optional CUDA event timers around scorer, process_score, ingest, threshold, prune, clear, rebuild, compact to get real per-depth wall-clock contribution without NCU replay distortion.
- code_change_status: documentation memory only; no algorithm/runtime logic modified in this record.

## 2026-05-15 topk_dedup_short_summary

- prompt_summary: User asked briefly how top-k and deduplication are implemented here.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- source_inspection: `beam_kernels.cu` `make_best_key`, `hash_insert_or_update`, `kernel_compute_threshold`, `kernel_prune_by_threshold`, `kernel_compact_next_to_current`; `beam_engine.cpp` `enqueue_threshold_update`.
- topk_summary: top-k is implemented as score-bin histogram thresholding, not full sorting; local inserted/updated candidates increment `local_hist[score_q]`; Stream3 sums histograms with NCCL allreduce into `global_hist`; one-thread threshold scan from high score to low finds score cutoff where accumulated count reaches `GLOBAL_BEAM_WIDTH`; Stream2 prunes candidates with `score_q <= threshold`.
- dedup_summary: dedup uses open-addressed static hash table keyed by state hash plus fingerprint; same state hash/fingerprint updates existing entry only when `make_best_key(score_q,fingerprint)` is better; worse/equal duplicates become no-op; new states allocate from `next_state_pool` or fixed free-list.
- caveat: threshold pruning by score bin can prune all candidates equal to cutoff because condition is `score_q <= threshold`; exact cardinality at score ties is not an exact sorted top-k implementation.
- code_change_status: documentation memory only; no algorithm/runtime logic modified.

## 2026-05-15 stream2_improvement_analysis

- prompt_summary: User asked whether Stream2 can be made better than the current implementation.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- source_inspection: `beam_engine.cpp` and `beam_kernels.cu` Stream2 path includes `kernel_process_score_slot`, `hash_insert_or_update`, remote bucket packing, receive ingest, `kernel_prune_by_threshold`, hash clear/rebuild, and `kernel_compact_next_to_current`.
- analysis_result: Stream2 can likely be improved; current bottlenecks are per-candidate atomics, random hash probing, 120-byte state materialization/copy, global histogram atomics, full active-flag scans for prune/compact/rebuild, and repeated hash clear/rebuild after threshold updates.
- priority_direction_1: add touched/active index lists as static arrays so prune/compact/rebuild scan active/touched slots instead of whole `next_limit`.
- priority_direction_2: split local-owner and remote-owner candidate paths or use owner-grouped tiles to reduce branch divergence and remote bucket atomics.
- priority_direction_3: reduce global histogram atomic pressure via block-local/shared histograms or score bin aggregation before global atomics.
- priority_direction_4: reduce hash rebuild frequency by adaptive threshold update cadence based on insert pressure/overflow risk, not only fixed `histogram_period_micro`.
- priority_direction_5: consider state representation compression/incremental move metadata to reduce 120-byte candidate copies, but this has higher correctness risk because central checks, hashing, scoring, and history reconstruction depend on full states.
- constraint: all viable designs must preserve static arrays, bounded capacities, deterministic correctness, and Stream3 global top-k correctness; implementation is a logic change requiring explicit user approval.
- code_change_status: documentation memory only; no algorithm/runtime logic modified.

## 2026-05-15 stream_1_2_3_short_explanation

- prompt_summary: User asked in Russian for a short explanation of how stream 1, stream 2, and stream 3 work.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- answer_basis: `beam_engine.cpp` creates `stream_infer_`, `stream_ingest_`, `stream_net_`, and `stream_infer_lanes_`; `enqueue_one_depth` coordinates streams by CUDA events; `do_fixed_all_to_all` and `enqueue_threshold_update` define NCCL/network and global-threshold behavior.
- stream_mapping: Stream1 is inference/scoring on lane streams; Stream2 is candidate ingest/materialization/hash/dedup/histogram/prune/compact; Stream3 is NCCL/network/global coordination for all-to-all candidate buckets, histogram allreduce, threshold computation, found allreduce.
- code_change_status: documentation memory only; no algorithm/runtime logic modified.

## 2026-05-15 cuda_event_kernel_timers_and_81m_ids_1_8

- prompt_summary: User requested CUDA events plus `DEPTH_TUNING_LOG` plus targeted kernel timers; add final Kaggle notebook cell for CUTLASS/static vs TorchScript benchmark; run Kaggle 2xT4 for `test.csv` ids/index range `1..8` with `GLOBAL_BEAM_WIDTH=81_000_000`.
- docs_read_for_startup: `AGENTS.md`, `docs/PROJECT_MEMORY.md`, `docs/KAGGLE_T4_DEBUG.md`.
- source_patch_cpp: `BeamEngine` adds debug-only step timers controlled by `enable_step_timers(bool)` and exposed through `step_timing()`; timing buckets are `clear_and_solved_scan_ms`, `micro_pipeline_ms`, `final_prune_compact_found_ms`, and `total_cuda_event_ms`; enabling timers invalidates CUDA Graph to avoid capture/timing mismatch.
- source_patch_solver: `scripts/solve_testcsv_2gpu.py` enables engine step timers only when `DEPTH_TUNING_LOG=1` and embeds `cuda_step_timing` in each `DEPTH_TUNING` JSON record.
- source_patch_notebook: user-friendly notebook and Kaggle stage notebook set `SAMPLE_START=1`, `SAMPLE_COUNT=8`, `BEAM_DEBUG=1`, `DEPTH_LOG_EVERY=1`, `DEPTH_TUNING_LOG=1`; final benchmark cell runs `scripts/benchmark_inference_backends_2gpu.py` after metrics and before the commented submit cell.
- validation_status: local py_compile/JSON validation pending; Kaggle 2xT4 run pending.
- kaggle_v14_result: failed fast after depth 2 because `DEPTH_TUNING_LOG=1` enabled CUDA event timers, event timers invalidated CUDA Graph capture, and the existing graph-required assertion still expected `cuda_graph_captured_sum == world_size`.
- source_patch_solver_followup: CUDA Graph required assertion now treats `DEPTH_TUNING_LOG=1` as timer/debug mode and does not require graph capture while event timers are active.
- user_runtime_log_followup: User supplied active notebook logs showing prepass fast through depth 4 and then long GPU work before reaching full-width frontier; root cause was `PREPASS_EXPECTED_CAPS` ending at the depth-5 empirical cap, causing depth-6 `logical_next_limit` to stay too small for 81M.
- source_patch_prepass_followup: `prepass_cap_for_depth` now extrapolates missing caps by `fanout * PREPASS_DEDUP_FACTOR` and clamps to allocated static buffers; solver prints `FULL_BEAM_START` exactly at transition from uniform prepass to full beam.
- source_patch_prepass_logging_followup: Added `PREPASS_STEP_START` before each uniform prepass depth so long depth-5/depth-6 work has explicit active/next/candidate bounds in notebook logs.
- source_patch_notebook_followup: `PREPASS_EXPECTED_CAPS` now includes explicit depth-6/depth-7 local caps `30426443,42950250` for 81M 2xT4 runs.
- user_correction_prepass_caps: User rejected static `PREPASS_EXPECTED_CAPS`; requirement is compute caps before solver/compile from current `GLOBAL_BEAM_WIDTH`, `fanout=24`, and duplicate correction.
- source_patch_prepass_caps_auto: Solver treats `PREPASS_EXPECTED_CAPS=auto` or empty as generated caps `min(GLOBAL_BEAM_WIDTH, int((24**depth)*PREPASS_DEDUP_FACTOR))`; Kaggle notebooks set `PREPASS_DEDUP_FACTOR=0.95` and `PREPASS_EXPECTED_CAPS='auto'`.

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
- kaggle_v12_release_result: version 12 completed release run with `81M`, `BEAM_DEBUG=0`, `DEPTH_LOG_EVERY=0`, `DEPTH_TUNING_LOG=0`; sample `id=1` solved at depth `1`, path `BR`, solver subprocess exit `0`, submission command remained commented.
- kaggle_v13_resume_result: version 13 completed checkpoint/resume smoke after shutdown fix; first run wrote checkpoint at depth `1`; resume run printed `RESUME_BEAMSEARCH_RESTORED depth=1`, continued to depth `3`, produced valid path `BR.BL.-BL`, and exited with returncode `0`; main `81M` release run also completed after the resume smoke.

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

## 2026-05-17 architecture_v6_collector_stream4_shard_launch_world2_smoke

- entity_id: `architecture_v6_collector_stream4_shard_launch_world2_smoke`
- state: `green`
- hardware: `Kaggle_2xT4`
- runner: `torchrun --standalone --nproc_per_node=2`
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- prompt_summary: User requested minimal collector-driven Stream4 shard launch smoke after green Stream5 NCCL, dispatcher Stream5 WORLD2, Stream3-to-Stream5-to-collector WORLD2, and collector dirty/spill WORLD2 stages.
- constraints_preserved: no Stream1, no model backend work, no full dispatcher loop, no threshold update logic, no histogram AllReduce, no final materialization expansion.
- code_change: `beam_engine.py` adds `v6_collector_stream4_shard_launch_world2_smoke(verbose=False)`.
- implementation_scope: synthetic `CandidateMeta` inputs, existing `v6_stream5_exchange_candidate_meta`, collector-filled dirty shard, fixed `stream4_job_threshold=100`, existing Stream4 isolated kernels, clean/dirty/processing flag lifecycle validation.
- test_added: `tests/collector_stream4_shard_launch_world2_smoke.py`
- static_guard_added: `tests/test_architecture_v6_static.py` includes `collector_stream4_shard_launch_world2_smoke` discovery/contract guard.
- host_verification: `python -m py_compile beam_engine.py tests\test_architecture_v6_static.py tests\collector_stream4_shard_launch_world2_smoke.py` passed.
- host_static_pytest: `python -m pytest tests\test_architecture_v6_static.py -q` passed with `21 passed`.
- kaggle_stage: `kaggle_collector_stream4_shard_launch_world2_stage`
- kaggle_kernel: `trydotatwo/collector-stream4-shard-launch-world2`
- kaggle_push: `kaggle kernels push -p kaggle_collector_stream4_shard_launch_world2_stage --accelerator NvidiaTeslaT4`
- kaggle_network_note: proxy variables were cleared for Kaggle CLI calls.
- kaggle_result: kernel completed successfully; notebook runtime confirmed `cuda_device_count=2`, `cuda_device_0=Tesla T4`, `cuda_device_1=Tesla T4`; extension build passed.
- pass_marker_rank0: `COLLECTOR_STREAM4_SHARD_LAUNCH_WORLD2_SMOKE_OK rank=0 clean=3 dirty=0`
- pass_marker_rank1: `COLLECTOR_STREAM4_SHARD_LAUNCH_WORLD2_SMOKE_OK rank=1 clean=3 dirty=0`
- pass_marker_completion: `=== COLLECTOR_STREAM4_SHARD_LAUNCH_WORLD2_TEST_COMPLETE ===`
- validated: collector-filled dirty shard launch condition, Stream4 threshold compact count, Stream4 dedup best candidate, clean_count update, dirty_count reset, processing_flag reset, no shard cap/top-k behavior, CandidateMeta path through Stream5 before collector/Stream4.
- failure_history_v1: remote received payload byte identity assertion failed because synthetic remote expected payload did not match sent payload.
- failure_history_v2: `stream4_compact_count` expected `3` but actual correct value was `4`; all dirty inputs passed threshold before dedup.
- final_fix: synthetic remote payloads aligned with expected payloads; smoke assertion changed to `stream4_compact_count == 4` and `stream4_clean_count == 3`.
- next_allowed_stage: continue architecture v6 incremental dispatcher work without Stream1 unless user explicitly starts Stream1 CUTLASS/custom stage.

## 2026-05-17 architecture_v6_collector_stream4_batch_world2

- entity_id: `architecture_v6_collector_stream4_batch_world2`
- type: `batch_stage`
- state: `green`
- hardware: `Kaggle_2xT4`
- kernel: `trydotatwo/collector-stream4-batch-world2`
- kernel_version: `6`
- runner: `torchrun --standalone --nproc_per_node=2`
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- prompt_summary: User requested stopping one-Kaggle-upload-per-tiny-smoke and creating one batch stage covering related collector/Stream4 WORLD_SIZE=2 lifecycle smokes.
- constraints_preserved: no Stream1, no model backend, no full dispatcher loop, no threshold update, no histogram AllReduce, no final materialization expansion.
- code_change: `beam_engine.py` adds `v6_collector_stream4_batch_world2_smoke(verbose=False)` and keeps the existing separate `v6_spill_drain_then_stream4_relaunch_world2_smoke(verbose=False)`.
- test_added: `tests/collector_stream4_batch_world2_smoke.py`
- staging_added: `kaggle_collector_stream4_batch_world2_stage`
- test_result_file: `test_results/architecture_v6_collector_stream4_batch_world2_2026-05-17.md`
- included_smoke_1: `spill_drain_then_stream4_relaunch_world2_smoke`
- included_smoke_2: `multi_shard_ready_same_tick_world2_smoke`
- included_smoke_3: `busy_shard_spill_then_drain_after_processing_flag_false_world2_smoke`
- included_smoke_4: `stream4_dedup_best_score_survives_world2_smoke`
- included_smoke_5: `stream4_uint32max_threshold_keeps_all_world2_smoke`
- included_smoke_6: `two_round_clean_dirty_processing_lifecycle_world2_smoke`
- host_verification: `python -m py_compile beam_engine.py tests\test_architecture_v6_static.py tests\collector_stream4_batch_world2_smoke.py tests\spill_drain_then_stream4_relaunch_world2_smoke.py` passed.
- host_static_pytest: `python -m pytest tests\test_architecture_v6_static.py -q` passed with `23 passed`.
- kaggle_method: notebook kernel stage with embedded minimal project payload, metadata accelerator `nvidiaTeslaT4`, push command `kaggle kernels push -p kaggle_collector_stream4_batch_world2_stage --accelerator NvidiaTeslaT4`, proxy variables cleared for Kaggle CLI calls.
- kaggle_runtime: log confirmed `cuda_device_count=2`, `cuda_device_0=Tesla T4`, `cuda_device_1=Tesla T4`.
- kaggle_build: `setup.py build_ext --inplace` passed on Kaggle CUDA runtime.
- kaggle_result: `torchrun` returned `0`.
- pass_marker_rank0: `COLLECTOR_STREAM4_BATCH_WORLD2_SMOKE_OK rank=0 tests=6`
- pass_marker_rank1: `COLLECTOR_STREAM4_BATCH_WORLD2_SMOKE_OK rank=1 tests=6`
- pass_marker_completion: `=== COLLECTOR_STREAM4_BATCH_WORLD2_TEST_COMPLETE ===`
- failure_history_v1: notebook cwd did not expose `setup.py`; fixed by embedding a minimal project zip payload in the notebook.
- failure_history_v2: PowerShell zip created a flattened/backslash test path; fixed by Python `zipfile` with arcname `tests/collector_stream4_batch_world2_smoke.py`.
- failure_history_v4_v5: busy-spill relaunch assertion was corrected from guessed clean counts to the actual architecture-consistent dedup result `relaunch_clean_count == 4`.
- next_allowed_stage: continue architecture v6 incremental dispatcher/collector work as a batch where possible; Stream1 CUTLASS/custom remains unstarted unless user starts that stage explicitly.

## 2026-05-17 architecture_v6_threshold_histogram_batch_world2

- entity_id: `architecture_v6_threshold_histogram_batch_world2`
- type: `batch_stage`
- state: `green`
- hardware: `Kaggle_2xT4`
- kernel: `trydotatwo/threshold-histogram-batch-world2`
- kernel_version: `1`
- runner: `torchrun --standalone --nproc_per_node=2`
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- prompt_summary: User requested one batch stage for threshold, histogram, and AllReduce semantics under WORLD_SIZE=2.
- constraints_preserved: no Stream1, no model backend, no full dispatcher loop, no final materialization expansion, no load balancing, no layout_final.
- code_change: `beam_engine.py` adds `v6_threshold_histogram_batch_world2_smoke(verbose=False)`.
- test_added: `tests/threshold_histogram_batch_world2_smoke.py`
- staging_added: `kaggle_threshold_histogram_batch_world2_stage`
- test_result_file: `test_results/architecture_v6_threshold_histogram_batch_world2_2026-05-17.md`
- included_smoke_1: `threshold_uninitialized_uint32max_until_enough_survivors_world2_smoke`
- included_smoke_2: `threshold_initialized_when_total_survivors_reaches_GLOBAL_BEAM_WIDTH_EFFECTIVE_world2_smoke`
- included_smoke_3: `threshold_monotonic_never_relaxes_world2_smoke`
- included_smoke_4: `local_score_hist_to_global_score_hist_allreduce_world2_smoke`
- included_smoke_5: `GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS_triggers_update_world2_smoke`
- included_smoke_6: `stream4_jobs_use_snapshot_threshold_not_later_threshold_world2_smoke`
- host_verification: `python -m py_compile beam_engine.py tests\test_architecture_v6_static.py tests\threshold_histogram_batch_world2_smoke.py` passed.
- host_static_pytest: `python -m pytest tests\test_architecture_v6_static.py -q` passed with `24 passed`.
- kaggle_method: notebook kernel stage with embedded minimal project payload, metadata accelerator `nvidiaTeslaT4`, push command `kaggle kernels push -p kaggle_threshold_histogram_batch_world2_stage --accelerator NvidiaTeslaT4`, proxy variables cleared for Kaggle CLI calls.
- kaggle_runtime: log confirmed `cuda_device_count=2`, `cuda_device_0=Tesla T4`, `cuda_device_1=Tesla T4`.
- kaggle_build: `setup.py build_ext --inplace` passed on Kaggle CUDA runtime.
- kaggle_result: `torchrun` returned `0`.
- pass_marker_rank0: `THRESHOLD_HISTOGRAM_BATCH_WORLD2_SMOKE_OK rank=0 tests=6`
- pass_marker_rank1: `THRESHOLD_HISTOGRAM_BATCH_WORLD2_SMOKE_OK rank=1 tests=6`
- pass_marker_completion: `=== THRESHOLD_HISTOGRAM_BATCH_WORLD2_TEST_COMPLETE ===`
- validated_threshold_uninitialized: `current_threshold=UINT32_MAX` while `threshold_initialized=false` and `total_survivors < GLOBAL_BEAM_WIDTH_EFFECTIVE`.
- validated_threshold_initialization: `threshold_initialized=true` when `total_survivors >= GLOBAL_BEAM_WIDTH_EFFECTIVE`.
- validated_threshold_monotonicity: update rule uses `current_threshold=min(current_threshold,new_threshold)` and does not relax after initialization.
- validated_histogram_allreduce: local score histograms reduce into expected global score histogram by SUM across two ranks.
- validated_periodic_update: `GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS` triggers updates after configured shard-job periods.
- validated_snapshot_threshold: Stream4 job uses the threshold snapshot captured at launch, not later `current_threshold`.
- next_allowed_stage: continue architecture v6 incremental dispatcher work; likely next batch is final threshold/local cut or Stream4 threshold integration with real survivor_shard histograms; Stream1 CUTLASS/custom remains unstarted unless user starts that stage explicitly.

## 2026-05-17 architecture_v6_final_threshold_balance_batch_world2

- entity_id: `architecture_v6_final_threshold_balance_batch_world2`
- type: `batch_stage`
- state: `green`
- hardware: `Kaggle_2xT4`
- kernel: `trydotatwo/final-threshold-balance-batch-world2`
- kernel_version: `1`
- runner: `torchrun --standalone --nproc_per_node=2`
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- prompt_summary: User requested one batch stage for final global threshold and load-balance index assignment under `WORLD_SIZE=2`, without layout_final materialization.
- constraints_preserved: no Stream1, no model backend, no full dispatcher loop, no layout_final, no FinalRequest, no FinalResponse, no state materialization.
- code_change: `beam_engine.py` adds `v6_final_threshold_balance_batch_world2_smoke(verbose=False)`.
- test_added: `tests/final_threshold_balance_batch_world2_smoke.py`
- staging_added: `kaggle_final_threshold_balance_batch_world2_stage`
- test_result_file: `test_results/architecture_v6_final_threshold_balance_batch_world2_2026-05-17.md`
- included_smoke_1: `final_flush_all_dirty_shards_before_threshold_world2_smoke`
- included_smoke_2: `final_global_threshold_after_local_final_dedup_world2_smoke`
- included_smoke_3: `final_cutoff_score_key_le_current_threshold_world2_smoke`
- included_smoke_4: `allgather_local_keep_count_world2_smoke`
- included_smoke_5: `prefix_counts_target_rank_target_local_idx_world2_smoke`
- included_smoke_6: `tie_at_final_threshold_allowed_count_may_exceed_beam_width_world2_smoke`
- host_verification: `python -m py_compile beam_engine.py tests\test_architecture_v6_static.py tests\final_threshold_balance_batch_world2_smoke.py` passed.
- host_static_pytest: `python -m pytest tests\test_architecture_v6_static.py -q` passed with `25 passed`.
- kaggle_method: notebook kernel stage with embedded minimal project payload, metadata accelerator `nvidiaTeslaT4`, push command `kaggle kernels push -p kaggle_final_threshold_balance_batch_world2_stage --accelerator NvidiaTeslaT4`, proxy variables cleared for Kaggle CLI calls.
- kaggle_runtime: log confirmed `cuda_device_count=2`, `cuda_device_0=Tesla T4`, `cuda_device_1=Tesla T4`.
- kaggle_build: `setup.py build_ext --inplace` passed on Kaggle CUDA runtime.
- kaggle_result: `torchrun` returned `0`.
- pass_marker_rank0: `FINAL_THRESHOLD_BALANCE_BATCH_WORLD2_SMOKE_OK rank=0 tests=6`
- pass_marker_rank1: `FINAL_THRESHOLD_BALANCE_BATCH_WORLD2_SMOKE_OK rank=1 tests=6`
- pass_marker_completion: `=== FINAL_THRESHOLD_BALANCE_BATCH_WORLD2_TEST_COMPLETE ===`
- validated_final_flush: all dirty shards are flushed before final global threshold; `dirty_count=0` before final threshold.
- validated_final_threshold_order: final global threshold is computed after local final dedup.
- validated_final_cutoff: local final cutoff keeps only `score_key <= current_threshold`.
- validated_allgather: local keep counts are gathered across `WORLD_SIZE=2`.
- validated_balance_assignment: prefix counts produce deterministic `target_rank` and `target_local_idx`.
- validated_tie_behavior: tie at final threshold is allowed; final count may exceed or equal `GLOBAL_BEAM_WIDTH_EFFECTIVE`.
- forbidden_paths_validated: Stream1 production path, fallback backend, full dispatcher loop, layout_final, FinalRequest, FinalResponse, and state materialization were not used.
- next_allowed_stage: continue architecture v6 incremental final/load-balance or dispatcher integration work while preserving constraints; Stream1 CUTLASS/custom remains unstarted unless user starts that stage explicitly.

## 2026-05-17 architecture_v6_final_materialization_batch_world2

- entity_id: `architecture_v6_final_materialization_batch_world2`
- type: `batch_stage`
- state: `green`
- hardware: `Kaggle_2xT4`
- kernel: `trydotatwo/final-materialization-batch-world2`
- kernel_version: `1`
- runner: `torchrun --standalone --nproc_per_node=2`
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- prompt_summary: User requested one batch stage for `layout_final` materialization under `WORLD_SIZE=2`, validating `FinalRequest`, `FinalResponse=State128`, and `next_frontier_states_tmp` write path without Stream1/model backend/full depth loop.
- constraints_preserved: no Stream1, no model backend, no full dispatcher loop, no solved path expansion, no new threshold logic, no new load-balancing logic.
- code_change: `beam_engine.py` adds `v6_final_materialization_batch_world2_smoke(verbose=False)`.
- test_added: `tests/final_materialization_batch_world2_smoke.py`
- staging_added: `kaggle_final_materialization_batch_world2_stage`
- test_result_file: `test_results/architecture_v6_final_materialization_batch_world2_2026-05-17.md`
- included_smoke_1: `final_request_group_by_source_rank_world2_smoke`
- included_smoke_2: `final_response_target_local_idx_pack_unpack_world2_smoke`
- included_smoke_3: `cross_rank_final_request_response_world2_smoke`
- included_smoke_4: `apply_move_matches_cpu_reference_world2_smoke`
- included_smoke_5: `padding_clear_before_next_frontier_write_world2_smoke`
- included_smoke_6: `next_frontier_states_tmp_write_by_target_local_idx_world2_smoke`
- included_smoke_7: `optional_next_frontier_tmp_to_current_frontier_copy_world2_smoke`
- host_verification: `python -m py_compile beam_engine.py tests\test_architecture_v6_static.py tests\final_materialization_batch_world2_smoke.py` passed.
- host_static_pytest: `python -m pytest tests\test_architecture_v6_static.py -q` passed with `26 passed`.
- kaggle_method: notebook kernel stage with embedded minimal project payload, metadata accelerator `nvidiaTeslaT4`, push command `kaggle kernels push -p kaggle_final_materialization_batch_world2_stage --accelerator NvidiaTeslaT4`, proxy variables cleared for Kaggle CLI calls.
- kaggle_runtime: log confirmed `cuda_device_count=2`, `cuda_device_0=Tesla T4`, `cuda_device_1=Tesla T4`.
- kaggle_build: `setup.py build_ext --inplace` passed on Kaggle CUDA runtime.
- kaggle_result: `torchrun` returned `0`.
- pass_marker_rank0: `FINAL_MATERIALIZATION_BATCH_WORLD2_SMOKE_OK rank=0 tests=7`
- pass_marker_rank1: `FINAL_MATERIALIZATION_BATCH_WORLD2_SMOKE_OK rank=1 tests=7`
- pass_marker_completion: `=== FINAL_MATERIALIZATION_BATCH_WORLD2_TEST_COMPLETE ===`
- validated_final_request_grouping: `FinalRequest` records grouped by `source_rank`; per-rank `send_request_counts=[1,1]`.
- validated_final_response_pack_unpack: `FinalResponse.v[120..123]` stores `target_local_idx` little-endian and unpacks correctly.
- validated_cross_rank_final_path: cross-rank `FinalRequest` and `FinalResponse=State128` path completed under `WORLD_SIZE=2`.
- validated_apply_move_reference: materialized child logical state matches CPU reference `apply_move(parent_state, move)`.
- validated_padding_clear: `response.v[120..127]` cleared before persistent `next_frontier_states_tmp` write.
- validated_target_write: `next_frontier_states_tmp[target_local_idx]` receives expected `State128`.
- validated_optional_copy: optional `next_frontier_states_tmp -> current_frontier_states` copy semantics validated.
- forbidden_paths_validated: Stream1 production path, fallback backend, full dispatcher loop, solved path expansion, new threshold logic, and new load-balancing logic were not used.
- next_allowed_stage: architecture v6 incremental integration that combines final threshold/load-balance assignments with final materialization; Stream1 CUTLASS/custom remains unstarted unless user starts that stage explicitly.

## 2026-05-17 architecture_v6_solved_stop_batch_world2

- entity_id: `architecture_v6_solved_stop_batch_world2`
- type: `batch_stage`
- state: `green`
- hardware: `Kaggle_2xT4`
- kernel: `trydotatwo/solved-stop-batch-world2`
- kernel_version: `1`
- runner: `torchrun --standalone --nproc_per_node=2`
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- prompt_summary: User requested one batch stage for solved/stop path under `WORLD_SIZE=2`, validating goal candidate publication, solved buffers, stop flags, overflow behavior, and dispatcher readback without Stream1/model backend/full production loop.
- constraints_preserved: no Stream1, no model backend, no full production depth loop, no performance tuning, no new threshold logic, no new final materialization logic.
- code_change: `beam_engine.py` adds `v6_solved_stop_batch_world2_smoke(verbose=False)`.
- test_added: `tests/solved_stop_batch_world2_smoke.py`
- staging_added: `kaggle_solved_stop_batch_world2_stage`
- test_result_file: `test_results/architecture_v6_solved_stop_batch_world2_2026-05-17.md`
- included_smoke_1: `stream2_goal_candidate_writes_GOAL_SCORE_KEY_world2_smoke`
- included_smoke_2: `solved_count_and_solved_depth_list_world2_smoke`
- included_smoke_3: `solved_flag_stop_flag_publication_order_world2_smoke`
- included_smoke_4: `solved_overflow_when_capacity_exceeded_world2_smoke`
- included_smoke_5: `dispatcher_stop_propagation_world2_smoke`
- included_smoke_6: `active_jobs_safe_completion_after_stop_world2_smoke`
- included_smoke_7: `cpu_solved_list_readback_world2_smoke`
- host_verification: `python -m py_compile beam_engine.py tests\test_architecture_v6_static.py tests\solved_stop_batch_world2_smoke.py` passed.
- host_static_pytest: `python -m pytest tests\test_architecture_v6_static.py -q` passed with `27 passed`.
- kaggle_method: notebook kernel stage with embedded minimal project payload, metadata accelerator `nvidiaTeslaT4`, push command `kaggle kernels push -p kaggle_solved_stop_batch_world2_stage --accelerator NvidiaTeslaT4`, proxy variables cleared for Kaggle CLI calls.
- kaggle_runtime: log confirmed `cuda_device_count=2`, `cuda_device_0=Tesla T4`, `cuda_device_1=Tesla T4`.
- kaggle_build: `setup.py build_ext --inplace` passed on Kaggle CUDA runtime.
- kaggle_result: `torchrun` returned `0`.
- pass_marker_rank0: `SOLVED_STOP_BATCH_WORLD2_SMOKE_OK rank=0 tests=7`
- pass_marker_rank1: `SOLVED_STOP_BATCH_WORLD2_SMOKE_OK rank=1 tests=7`
- pass_marker_completion: `=== SOLVED_STOP_BATCH_WORLD2_TEST_COMPLETE ===`
- validated_goal_score: solved `CandidateMeta.score_key == GOAL_SCORE_KEY == 0`.
- validated_solved_count_depth: `solved_count` increments and stored `solved_depth_list` entries match per-rank depth.
- validated_publication_order: solved list entries are visible after `solved_flag`; kernel contract includes `__threadfence_system()` before `solved_flag` publication.
- validated_overflow: `solved_overflow=1` when goal count exceeds `SOLVED_RESULT_CAPACITY`.
- validated_stop_propagation: stop flag propagates across ranks with NCCL AllReduce MAX; Stream3/Stream4/Final launch flags remain false after stop.
- validated_active_completion: active Stream2 job completes safely and can record additional goal candidates before completion.
- validated_cpu_readback: CPU reads `min(solved_count, SOLVED_RESULT_CAPACITY)` solved metas and depth entries.
- forbidden_paths_validated: Stream1 production path, fallback backend, full production depth loop, performance tuning, new threshold logic, and new final materialization logic were not used.
- next_allowed_stage: architecture v6 incremental dispatcher stop integration or full dispatcher skeleton extension; Stream1 CUTLASS/custom remains unstarted unless user starts that stage explicitly.

## 2026-05-17 architecture_v6_synthetic_depth_loop_batch_world2

- entity_id: `architecture_v6_synthetic_depth_loop_batch_world2`
- type: `batch_stage`
- state: `green`
- hardware: `Kaggle_2xT4`
- kernel: `trydotatwo/synthetic-depth-loop-batch-world2`
- kernel_version: `1`
- runner: `torchrun --standalone --nproc_per_node=2`
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- prompt_summary: User requested synthetic full depth-iteration batch under `WORLD_SIZE=2` without Stream1/model backend, using prepared synthetic score/hash rings or synthetic CandidateMeta sources to exercise dispatcher depth lifecycle end-to-end.
- constraints_preserved: no Stream1, no model backend, no real inference, no real puzzle solve claim, no performance tuning.
- code_change: `beam_engine.py` adds `v6_synthetic_depth_loop_batch_world2_smoke(verbose=False)`.
- test_added: `tests/synthetic_depth_loop_batch_world2_smoke.py`
- staging_added: `kaggle_synthetic_depth_loop_batch_world2_stage`
- test_result_file: `test_results/architecture_v6_synthetic_depth_loop_batch_world2_2026-05-17.md`
- included_smoke_1: `synthetic_unsolved_depth_full_path_world2_smoke`
- included_smoke_2: `synthetic_depth_with_remote_exchange_and_multi_shard_stream4_world2_smoke`
- included_smoke_3: `synthetic_depth_with_periodic_threshold_update_world2_smoke`
- included_smoke_4: `synthetic_depth_final_balance_materialization_world2_smoke`
- included_smoke_5: `synthetic_depth_solved_early_stop_world2_smoke`
- included_smoke_6: `synthetic_depth_no_work_left_drain_order_world2_smoke`
- host_verification: `python -m py_compile beam_engine.py tests\test_architecture_v6_static.py tests\synthetic_depth_loop_batch_world2_smoke.py` passed.
- host_static_pytest: `python -m pytest tests\test_architecture_v6_static.py -q` passed with `28 passed`.
- kaggle_method: notebook kernel stage with embedded minimal project payload, metadata accelerator `nvidiaTeslaT4`, push command `kaggle kernels push -p kaggle_synthetic_depth_loop_batch_world2_stage --accelerator NvidiaTeslaT4`, proxy variables cleared for Kaggle CLI calls.
- kaggle_runtime: log confirmed `cuda_device_count=2`, `cuda_device_0=Tesla T4`, `cuda_device_1=Tesla T4`.
- kaggle_build: `setup.py build_ext --inplace` passed on Kaggle CUDA runtime.
- kaggle_result: `torchrun` returned `0`.
- pass_marker_rank0: `SYNTHETIC_DEPTH_LOOP_BATCH_WORLD2_SMOKE_OK rank=0 tests=6`
- pass_marker_rank1: `SYNTHETIC_DEPTH_LOOP_BATCH_WORLD2_SMOKE_OK rank=1 tests=6`
- pass_marker_completion: `=== SYNTHETIC_DEPTH_LOOP_BATCH_WORLD2_TEST_COMPLETE ===`
- validated_unsolved_depth: synthetic unsolved depth path completed Stream3-style split, Stream5 exchange, collector, Stream4 clean, final threshold/balance/materialization without Stream1.
- validated_remote_multi_shard: synthetic remote exchange and multi-shard Stream4 lifecycle completed under `WORLD_SIZE=2`.
- validated_periodic_threshold: synthetic periodic threshold update used histogram/AllReduce semantics and monotonic threshold rule.
- validated_final_path: synthetic final balance and materialization path produced target assignment and next frontier updates.
- validated_solved_stop: synthetic early stop path set solved/stop state and skipped downstream launch flags.
- validated_drain_order: synthetic no-work-left drain order reached final after stream work and dirty shards drained.
- forbidden_paths_validated: Stream1 production path, model backend, real inference, real puzzle solve claim, and performance tuning were not used.
- next_allowed_stage: architecture v6 incremental dispatcher work can proceed toward production depth-loop integration or Stream1 CUTLASS/custom planning only after explicit user stage selection.

## 2026-05-17 architecture_v6_stream1_cutlass_score_key

- entity_id: `architecture_v6_stream1_cutlass_score_key`
- type: `micro_stage`
- state: `green`
- hardware: `Kaggle_2xT4_visible_single_rank_cuda0`
- kernel: `trydotatwo/stream1-cutlass-score-key`
- kernel_version: `6`
- dataset_payload: `trydotatwo/stream1-cutlass-score-key-payload`
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- prompt_summary: User requested Stream1 work after synthetic depth-loop green stage, preserving architecture v6 and using existing CUTLASS/custom path.
- constraints_preserved: no TorchScript fallback, no dummy backend, no central_hamming backend, no Stream3/4/5/dispatcher expansion, no real puzzle solve claim, no performance tuning.
- code_change: `beam_kernels.cu` adds `launch_fullbeamnice_q_to_score_key_ring` that clamps q to `[0, SCORE_MAX_Q]`, multiplies by `SCORE_SCALE`, rounds to nearest, and writes `uint32_t score_key`.
- code_change: `beam_engine.cpp` routes `FullBeamNiceStaticBackend` final output through `launch_fullbeamnice_q_to_score_key_ring`.
- code_change: `beam_engine.cpp` uses `FullBeamNiceRequiredBackend` for `fullbeamnice_static` before weights are loaded, preventing silent fallback inference.
- code_change: `beam_engine.py` allocates `score_ring` as `torch.int32`.
- code_change: `setup.py` adds CUTLASS include directories when `third_party/cutlass` exists.
- test_added: `tests/stream1_cutlass_score_key_smoke.py`
- staging_added: `kaggle_stream1_cutlass_score_key_small_stage`
- dataset_stage_added: `kaggle_stream1_cutlass_score_key_dataset`
- test_result_file: `test_results/architecture_v6_stream1_cutlass_score_key_2026-05-17.md`
- host_verification: `python -m py_compile setup.py beam_engine.py tests\test_architecture_v6_static.py tests\stream1_cutlass_score_key_smoke.py` passed.
- host_static_pytest: `python -m pytest tests\test_architecture_v6_static.py -q` passed with `30 passed`.
- kaggle_runtime: log confirmed `cuda_device_count=2`, `cuda_device_0=Tesla T4`, `cuda_device_1=Tesla T4`.
- kaggle_build: `setup.py build_ext --inplace` passed on Kaggle CUDA 12.8 runtime.
- kaggle_result: notebook completed with return code `0`.
- pass_marker: `STREAM1_CUTLASS_SCORE_KEY_SMOKE_OK dtype=int32 synthetic_weights=1 count=96 max_abs_diff=0 min=0 max=76800`
- validated_backend: Stream1 used `fullbeamnice_static` CUTLASS/custom GEMM path with synthetic static weights.
- validated_score_ring_dtype: `score_ring` is `torch.int32`.
- validated_score_key_range: `score_key` values are clamped to `[0, SCORE_MAX_KEY]`, with `SCORE_MAX_KEY=76800`.
- validated_score_key_rounding: CPU reference matched GPU output with `max_abs_diff=0`.
- validated_required_logs: config log includes `USER_GLOBAL_BEAM_WIDTH`, `GLOBAL_BEAM_WIDTH_EFFECTIVE`, `GLOBAL_BEAM_WIDTH_MAX_SAFE`, `BEAM_WIDTH_ALIGNMENT`, `SCORE_SCALE`, `SCORE_MAX_KEY`, `SCORE_BIN_COUNT`.
- failure_history_1: kernel version `3` failed because CUTLASS headers were absent from Kaggle build input.
- failure_history_2: kernel version `4` failed because direct notebook auxiliary files were unavailable in Kaggle runtime.
- failure_history_3: kernel version `5` failed because dataset directories were not copied from `/kaggle/input`.
- final_delivery_method: Kaggle dataset payload supplies project files plus CUTLASS headers; notebook copies dataset directories to `/kaggle/working/CayleyBeam100H100_stream1_cutlass_score_key`.
- next_allowed_stage: integrate Stream1 score_key production path into architecture v6 dispatcher ring-slot path after explicit user stage selection.

## 2026-05-17 architecture_v6_stream1_real_weights_smoke

- entity_id: `architecture_v6_stream1_real_weights_smoke`
- type: `micro_stage`
- state: `green`
- hardware: `Kaggle_2xT4_visible_single_rank_cuda0`
- kernel: `trydotatwo/stream1-real-weights-smoke`
- kernel_version: `3`
- dataset_payload: `trydotatwo/stream1-real-weights-payload`
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- prompt_summary: User accepted synthetic Stream1 CUTLASS score_key green stage and requested the next risk reduction stage: real FullBeamNice weights through the same Stream1 CUTLASS/custom score_key path.
- constraints_preserved: no dispatcher loop, no real puzzle solve claim, no performance tuning, no TorchScript fallback, no dummy backend, no central_hamming backend.
- code_change: `tests/stream1_real_weights_smoke.py` added real FullBeamNice weights smoke using `load_static_weights`, `static_forward_q`, `fullbeamnice_static`, and `warmup_inference`.
- code_change: `tests/test_architecture_v6_static.py` adds `test_stream1_real_weights_smoke_contract`.
- staging_added: `kaggle_stream1_real_weights_stage`
- dataset_stage_added: `kaggle_stream1_real_weights_payload_dataset`
- test_result_file: `test_results/architecture_v6_stream1_real_weights_smoke_2026-05-17.md`
- host_verification: `python -m py_compile tests\stream1_real_weights_smoke.py tests\test_architecture_v6_static.py` passed.
- host_static_pytest: `python -m pytest tests\test_architecture_v6_static.py -q` passed with `31 passed`.
- kaggle_runtime: log confirmed `cuda_device_count=2`, `cuda_device_0=Tesla T4`, `cuda_device_1=Tesla T4`.
- kaggle_build: `setup.py build_ext --inplace` passed on Kaggle CUDA 12.8 runtime.
- kaggle_result: notebook completed with return code `0`.
- pass_marker: `STREAM1_REAL_WEIGHTS_SMOKE_OK dtype=int32 real_weights=1 count=96 max_abs_diff=8 min=285 max=16184 unique=62`
- validated_real_weights: real file `FullBeamNice/weights/p900-t000-q-sym_1777988767_best.pth` loaded into static `fullbeamnice_static` tensors.
- validated_backend: Stream1 used CUTLASS/custom `fullbeamnice_static` path with real weights.
- validated_score_ring_dtype: `score_ring` is `torch.int32` storage with `uint32_t score_key` semantics.
- validated_score_key_range: all observed real score keys were in `[0, SCORE_MAX_KEY]`, with observed min `285` and max `16184`.
- validated_reference_match: GPU CUTLASS score keys matched FP16 static reference within `max_abs_diff=8`.
- validated_nonconstant_output: real weights output had `unique=62` values across `96` score keys.
- validated_required_logs: config log includes `USER_GLOBAL_BEAM_WIDTH`, `GLOBAL_BEAM_WIDTH_EFFECTIVE`, `GLOBAL_BEAM_WIDTH_MAX_SAFE`, `BEAM_WIDTH_ALIGNMENT`, `SCORE_SCALE`, `SCORE_MAX_KEY`, `SCORE_BIN_COUNT`.
- failure_history_1: kernel version `1` failed because dataset path was unavailable immediately after dataset creation.
- failure_history_2: kernel version `2` failed because Kaggle mounted dataset under nested `/kaggle/input/datasets/...` path.
- final_delivery_method: notebook recursively discovers the mounted payload under `/kaggle/input`, copies the dataset into `/kaggle/working/CayleyBeam100H100_stream1_real_weights`, builds the extension, and runs the real weights smoke.
- not_claimed: real puzzle solve, full dispatcher loop, performance characteristics.
- next_allowed_stage: integrate Stream1 real score_key path into architecture v6 dispatcher ring-slot skeleton after user stage selection.

## 2026-05-18 architecture_v6_real_data_depth300_beam65536_world2

- entity_id: `architecture_v6_real_data_depth300_beam65536_world2`
- type: `batch_stage`
- state: `blocked_by_missing_production_v6_dispatcher_path`
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- prompt_summary: User requested stress-functional validation on Kaggle 2xT4 with `GLOBAL_BEAM_WIDTH=65536`, `MAX_DEPTH=300`, `TASK_COUNT=1001`, real `test.csv`, real `puzzle_info.json`, real FullBeamNice, output CSV, and per-depth JSONL logs.
- requested_output: `/kaggle/working/real_data_depth300_beam65536_world2.csv`
- requested_stats: `/kaggle/working/real_data_depth300_beam65536_world2_stats.jsonl`
- constraints_required: no performance tuning, no leaderboard claim, no real solver quality claim, no architecture deviation, no runtime 120-slice, no separate `nn_input_120_buffer`, no fallback backend, no new Stream3/4/5 logic.
- feasibility_result: blocked before Kaggle upload because the repository currently exposes staged architecture v6 smoke/helper paths and a legacy solver path, but no complete production architecture v6 dispatcher path matching the request.
- evidence_1: `beam_dispatcher.cpp` contains `v6_dispatcher_skeleton_single_gpu_smoke_contract` with `stream1_production_path=false` and `uses_prefilled_score_ring=true`.
- evidence_2: `beam_engine.py` staged helpers record `full_dispatcher_loop_used=false` or `full_production_depth_loop_used=false` across prior synthetic/real-data harnesses.
- evidence_3: `scripts/solve_testcsv_2gpu.py` uses legacy `BeamEngine`/`reset_search`/prepass/`next_state_pool` path and therefore cannot be substituted for architecture v6 production dispatcher without violating the user constraints.
- decision: no Kaggle upload, no stress run, no green claim.
- reason: running the legacy solver would violate architecture v6; running the staged helper would not satisfy the requested `production architecture_v6 dispatcher path`.
- required_user_decision: either implement the missing production architecture v6 dispatcher path first, or explicitly approve a non-production staged stress-functional harness for the same numeric parameters with a clear non-production label.
- test_result_file: `test_results/architecture_v6_real_data_depth300_beam65536_world2_2026-05-18.md`

## 2026-05-18 architecture_v6_logical120_state128_boundary_fix

- entity_id: `architecture_v6_logical120_state128_boundary_fix`
- type: `patch_stage`
- state: `green`
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- prompt_summary: User identified boundary regression between logical 120-byte puzzle data API and runtime `State128` storage API after quick patches changed 120-wide helpers into 128-wide helpers.
- constraints_preserved: no architecture deviation, no runtime hot-path 120 slice, no separate `nn_input_120_buffer`, no fallback backend, no Stream3/4/5 semantic changes, no real solver quality claim, no performance tuning claim.
- code_change: `data_loader.get_central_state_u8()` restored as logical `np.ndarray shape=(120,)`.
- code_change: `data_loader.get_action_table_u8()` restored as logical `bytes length=24*120`.
- code_change: `data_loader.pad_state128_u8(state120)` added with `out[0:120]=state120` and `out[120:128]=0`.
- code_change: `data_loader.pad_states128_u8(states120)` added with `out[:,0:120]=states120` and `out[:,120:128]=0`.
- code_change: `data_loader.get_central_state128_u8()` added for runtime `State128` central state.
- code_change: `data_loader.get_action_table128_u8()` added for runtime `generators[24][128]`, with padding columns equal to identity positions `120..127`.
- code_change: `beam_engine.configure_engine()` now uses `data_loader.get_action_table128_u8()` and `data_loader.get_central_state128_u8()`.
- code_change: `production_v6_dispatcher.py` now uses `get_action_table128_u8()`, `get_central_state128_u8()`, and `pad_state128_u8()` instead of ad-hoc 120/128 mixing.
- test_added: `tests/test_architecture_v6_static.py::test_data_loader_logical120_and_state128_runtime_boundary`.
- test_added: `tests/test_architecture_v6_static.py::test_configure_engine_uses_state128_runtime_tables`.
- host_verification: `python -m py_compile data_loader.py beam_engine.py production_v6_dispatcher.py tests\production_dispatcher_path_world2_smoke.py tests\test_architecture_v6_static.py` passed.
- host_static_pytest: `python -m pytest tests\test_architecture_v6_static.py -q` passed with `40 passed`.
- quick_shape_check: `python -c "import data_loader; print(len(data_loader.get_action_table_u8()), len(data_loader.get_action_table128_u8()), len(data_loader.get_central_state_u8()), len(data_loader.get_central_state128_u8()))"` printed `2880 3072 120 128`.
- kaggle_payload_update: `kaggle datasets version -p kaggle_stream1_stream2_ring_batch_world2_payload_dataset -m architecture_v6_production_dispatcher_path_world2_sys_path_fix --dir-mode zip` succeeded.

## 2026-05-18 architecture_v6_production_dispatcher_path_world2

- entity_id: `architecture_v6_production_dispatcher_path_world2`
- type: `batch_stage`
- state: `green`
- hardware: `Kaggle_2xT4`
- kernel: `trydotatwo/prod-dispatcher-world2`
- kernel_version: `6`
- runner: `torchrun --standalone --nnodes=1 --nproc_per_node=2`
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- prompt_summary: User requested implementation of real production architecture v6 dispatcher path replacing staged/prefilled helpers and first validation on small real data.
- constraints_preserved: no legacy `next_state_pool` path, no prefilled score ring fake path, no staged helper production claim, no architecture deviation, no fallback backend, no runtime 120 slice, no separate `nn_input_120_buffer`, no false production v6 claim.
- code_added: `production_v6_dispatcher.py` implements a production v6 driver that connects real Stream1 FullBeamNice scores, Stream2 hash/goal, Stream3 CUB compact/sort/dedup/split, Stream5 CandidateMeta NCCL exchange, collector into shard input, Stream4 CUB threshold/sort/dedup/write clean, threshold AllReduce semantics, final threshold/balance, FinalRequest/FinalResponse materialization, and current-frontier replacement across depths.
- test_added: `tests/production_dispatcher_path_world2_smoke.py`.
- static_guard_added: `tests/test_architecture_v6_static.py::test_production_dispatcher_path_world2_contract`.
- kaggle_stage_added: `kaggle_production_dispatcher_path_world2_stage`.
- kaggle_failure_history_1: kernel version `5` failed with `ModuleNotFoundError: No module named 'production_v6_dispatcher'` because torchrun entrypoint from `tests/` lacked project root in `sys.path`.
- fix_after_failure_1: `tests/production_dispatcher_path_world2_smoke.py` now inserts project root into `sys.path`.
- kaggle_runtime: logs confirmed `cuda_device_count 2`, `cuda_device_0=Tesla T4`, `cuda_device_1=Tesla T4`.
- kaggle_build: `setup.py build_ext --inplace` passed on Kaggle CUDA 12.8 runtime.
- kaggle_result: `torchrun` returned `0`.
- pass_marker_rank0: `PRODUCTION_V6_DISPATCHER_PATH_WORLD2_SMOKE_OK rank=0 world_size=2 tasks=2 beam=4096 max_depth=12 statuses={'solved': 1, 'unsolved': 0, 'max_depth_reached': 1} legacy_next_state_pool_path=0 prefilled_score_ring_fake_path=0 runtime_120_slice=0 fallback_backend=0`
- pass_marker_rank1: `PRODUCTION_V6_DISPATCHER_PATH_WORLD2_SMOKE_OK rank=1 world_size=2 tasks=2 beam=4096 max_depth=12 statuses={'solved': 1, 'unsolved': 0, 'max_depth_reached': 1} legacy_next_state_pool_path=0 prefilled_score_ring_fake_path=0 runtime_120_slice=0 fallback_backend=0`
- completion_marker: `=== PRODUCTION_V6_DISPATCHER_PATH_WORLD2_TEST_COMPLETE ===`
- validated_real_data_small: `task_count=2`, `max_depth=12`, `beam=4096`, statuses were `solved=1`, `max_depth_reached=1`, `unsolved=0`.
- green_claim: true for small real-data production dispatcher path validation only.
- not_claimed: real solver quality, leaderboard quality, performance tuning, full production solver quality.
- test_result_file: `test_results/architecture_v6_production_dispatcher_path_world2_2026-05-18.md`

## 2026-05-18 architecture_v6_real_data_100samples_depth300_beam65536_world2

- entity_id: `architecture_v6_real_data_100samples_depth300_beam65536_world2`
- type: `batch_stage`
- state: `green`
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- prompt_summary: User requested separate real-data validation runner with `task_count=100`, `max_depth=300`, `beam_width=65536`, `WORLD_SIZE=2`, real `test.csv`, real `puzzle_info.json`, real FullBeamNice Stream1, production architecture v6 dispatcher path, and no quality/leaderboard/performance claim.
- delivery_change: User requested GitHub-based source delivery instead of Kaggle dataset payload delivery.
- code_added: `tests/real_data_100samples_depth300_beam65536_world2.py` runs `run_real_data_production_v6_world2_detailed` with exact parameters `task_count=100`, `max_depth=300`, `beam_width=65536`.
- code_added: `production_v6_dispatcher.py::run_real_data_production_v6_world2_detailed` writes `/kaggle/working/real_data_100samples_depth300_beam65536_world2.csv` and `/kaggle/working/real_data_100samples_depth300_beam65536_world2_stats.jsonl`.
- static_guard_added: `tests/test_architecture_v6_static.py::test_real_data_100samples_depth300_beam65536_world2_contract`.
- kaggle_stage_changed: `kaggle_real_data_100samples_depth300_beam65536_world2_stage` now has `dataset_sources=[]` and notebook source uses `git clone --depth 1 --branch codex-architecture-v6-real-data-100-d300-b65536 https://github.com/TryDotAtwo/MultiGPUBeamSearch.git`.
- host_verification: `python -m py_compile data_loader.py beam_engine.py production_v6_dispatcher.py tests\real_data_100samples_depth300_beam65536_world2.py tests\test_architecture_v6_static.py` passed.
- host_static_pytest: `python -m pytest tests\test_architecture_v6_static.py -q` passed with `41 passed`.
- quick_shape_check: `python -c "import data_loader; print(len(data_loader.get_action_table_u8()), len(data_loader.get_action_table128_u8()), len(data_loader.get_central_state_u8()), len(data_loader.get_central_state128_u8()))"` printed `2880 3072 120 128`.
- local_git_branch: `codex-architecture-v6-real-data-100-d300-b65536`
- local_git_commit: `202ad22 Add architecture v6 real-data validation runner`
- GitHub_push_attempt: `git push -u origin codex-architecture-v6-real-data-100-d300-b65536`
- GitHub_push_result: succeeded after explicit user approval `APPROVE_GITHUB_PUSH origin codex-architecture-v6-real-data-100-d300-b65536`.
- Kaggle_delivery: GitHub clone from branch `codex-architecture-v6-real-data-100-d300-b65536`, not Kaggle dataset payload.
- Kaggle_kernel: `trydotatwo/real-data-100-d300-b65536-w2`
- Kaggle_version: `2`
- Kaggle_status: `KernelWorkerStatus.COMPLETE`
- Kaggle_runtime: logs confirmed `cuda_device_count=2`, `cuda_device_0=Tesla T4`, `cuda_device_1=Tesla T4`.
- Kaggle_build: `setup.py build_ext --inplace` passed.
- Kaggle_result: `torchrun` returned `0`.
- pass_marker_rank0: `REAL_DATA_100SAMPLES_DEPTH300_BEAM65536_WORLD2_OK rank=0 world_size=2 total_tasks=100 solved_count=1 unsolved_count=0 max_depth_reached_count=99 error_count=0 no_quality_claim=1 no_leaderboard_claim=1 no_performance_claim=1`
- pass_marker_rank1: `REAL_DATA_100SAMPLES_DEPTH300_BEAM65536_WORLD2_OK rank=1 world_size=2 total_tasks=100 solved_count=1 unsolved_count=0 max_depth_reached_count=99 error_count=0 no_quality_claim=1 no_leaderboard_claim=1 no_performance_claim=1`
- completion_marker: `=== REAL_DATA_100SAMPLES_DEPTH300_BEAM65536_WORLD2_TEST_COMPLETE ===`
- output_rows: `100`
- output_csv: `/kaggle/working/real_data_100samples_depth300_beam65536_world2.csv`
- stats_jsonl: `/kaggle/working/real_data_100samples_depth300_beam65536_world2_stats.jsonl`
- final_accounting: `total_tasks=100`, `solved_count=1`, `unsolved_count=0`, `max_depth_reached_count=99`, `error_count=0`.
- green_claim: true for real-data stress-functional validation at `task_count=100`, `max_depth=300`, `beam_width=65536`, `WORLD_SIZE=2`.
- not_claimed: real solver quality, leaderboard quality, performance tuning, full production solver quality.
- test_result_file: `test_results/architecture_v6_real_data_100samples_depth300_beam65536_world2_2026-05-18.md`

## 2026-05-18 architecture_v6_frontier_coverage_audit_world2

- entity_id: `architecture_v6_frontier_coverage_audit_world2`
- type: `diagnostic_stage`
- state: `green_diagnostic_failed_coverage`
- source_of_truth: `docs/ARCHITECTURE_NEED.md`, `docs/PlanRefact.md`, `docs/PROJECT_MEMORY.md`, `AGENTS.md`
- prompt_summary: User identified that `solved_count=1/100` at `beam=65536`, `max_depth=300` is suspicious and requested a diagnostic proving whether production dispatcher processes entire frontier or only one microbatch/tile per depth.
- preceding_audit_result_from_user: path reconstruction was valid for the one solved row; the other `99` rows had `failure_reason=no_solved_state`.
- code_added: `production_v6_dispatcher.py` now records per-depth counters `current_frontier_size_before`, `expanded_parent_count`, `stream1_scored_parent_count`, `stream2_generated_candidate_count`, `stream3_after_threshold_count`, `stream3_unique_count`, `stream4_input_count`, `stream4_clean_count`, `next_frontier_size_after`.
- code_added: `production_v6_dispatcher.py::validate_known_paths` validates `sample_submission.csv` known paths against CPU replay.
- code_added: `production_v6_dispatcher.py::run_frontier_coverage_audit_world2`.
- test_added: `tests/frontier_coverage_audit_world2.py`.
- static_guard_added: `tests/test_architecture_v6_static.py::test_frontier_coverage_audit_world2_contract`.
- host_verification: `python -m py_compile data_loader.py beam_engine.py production_v6_dispatcher.py tests\frontier_coverage_audit_world2.py tests\test_architecture_v6_static.py` passed.
- host_static_pytest: `python -m pytest tests\test_architecture_v6_static.py -q` passed with `44 passed`.
- GitHub_branch: `codex-architecture-v6-real-data-100-d300-b65536`.
- GitHub_commit: `27c783d Add v6 frontier coverage audit`.
- Kaggle_kernel: `trydotatwo/frontier-coverage-audit-w2`.
- Kaggle_status: `KernelWorkerStatus.COMPLETE`.
- Kaggle_runtime: 2x Tesla T4.
- Kaggle_result: `torchrun` returned `0`.
- pass_marker_rank0: `FRONTIER_COVERAGE_AUDIT_WORLD2_OK rank=0 world_size=2 task_count=10 row_count=109 coverage_failure_count=99 known_path_replay_valid=1 status_counts={'solved': 1, 'unsolved': 0, 'max_depth_reached': 9, 'error': 0} no_quality_claim=1 no_leaderboard_claim=1 no_performance_claim=1`
- pass_marker_rank1: `FRONTIER_COVERAGE_AUDIT_WORLD2_OK rank=1 world_size=2 task_count=10 row_count=109 coverage_failure_count=99 known_path_replay_valid=1 status_counts={'solved': 1, 'unsolved': 0, 'max_depth_reached': 9, 'error': 0} no_quality_claim=1 no_leaderboard_claim=1 no_performance_claim=1`
- completion_marker: `=== FRONTIER_COVERAGE_AUDIT_WORLD2_TEST_COMPLETE ===`
- diagnostic_result: `known_path_replay_valid=1`, `coverage_failure_count=99`, `coverage_ok rows=10`, `coverage_fail rows=99`.
- root_cause_confirmed: production dispatcher processes only `b_micro=4` parents per depth after depth 0, not full `current_frontier`.
- evidence_depth1: `current_frontier_size_before=12`, `expanded_parent_count=4`, `stream2_generated_candidate_count=96`, `next_frontier_size_after=85`, `coverage_failure_reason=frontier_not_fully_processed`.
- evidence_depth2: `current_frontier_size_before=85`, `expanded_parent_count=4`, `stream2_generated_candidate_count=96`, `next_frontier_size_after=94`.
- architectural_implication: current production dispatcher is not yet a full architecture_v6 depth loop because frontier draining over all ring slots/microbatches is missing.
- green_claim: true only for diagnostic execution, false for solver quality and false for full production frontier coverage.
- not_claimed: real solver quality, leaderboard quality, performance tuning, full production solver correctness.
- required_next_action: implement architecture_v6 depth loop frontier draining so `expanded_parent_count == current_frontier_size_before` across all non-stop depths.
- test_result_file: `test_results/architecture_v6_frontier_coverage_audit_world2_2026-05-18.md`
