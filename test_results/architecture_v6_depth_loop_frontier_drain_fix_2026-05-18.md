# architecture_v6_depth_loop_frontier_drain_fix

entity_id=architecture_v6_depth_loop_frontier_drain_fix; type=test_result_record; state=host_green_kaggle_pending

## Context

- task_id=architecture_v6_depth_loop_frontier_drain_fix
- date=2026-05-18
- source_of_truth=[docs/ARCHITECTURE_NEED.md, docs/PROJECT_MEMORY.md, AGENTS.md]
- constraints=[no_architecture_expansion, no_performance_tuning, no_legacy_solver_substitution, no_runtime_120_slice, no_separate_nn_input_120_buffer, no_fallback_backend]

## Changes

- production_v6_dispatcher.py: dispatcher bucket capacity now uses `pow2_ceil(max(131072, b_micro * MOVE_COUNT))`; expected `B_MICRO=8192`, `MOVE_COUNT=24`, `K_EXPAND_TILE=196608`, `BUCKET_CAP_PER_PEER=262144`.
- production_v6_dispatcher.py: `_run_stream5` capacity path uses only `self.cfg["bucket_cap_per_peer"]`; `recv_count` and `recv_offset` exist before synchronization; `v6_stream5_exchange_candidate_meta` remains before synchronization; return dict keys are exact and non-duplicated.
- tests/frontier_coverage_audit_world2.py: default `FRONTIER_COVERAGE_B_MICRO` changed to `8192`.
- tests/test_architecture_v6_static.py: static guards added for Stream5 capacity, frontier drain loop, capacity derivation, and required depth counters.

## Host Checks

- command=`python -m py_compile production_v6_dispatcher.py beam_engine.py tests\frontier_coverage_audit_world2.py tests\test_architecture_v6_static.py`; result=pass
- command=`python -m pytest tests\test_architecture_v6_static.py -q`; result=pass; summary=`46 passed in 0.34s`

## Hard Invariant Update

- invariant=`B_MICRO = 8192`
- invariant=`K_EXPAND_TILE = 8192 * 24 = 196608`
- forbidden=[B_MICRO_less_than_8192,B_MICRO_4,K_EXPAND_TILE_not_196608,tiny_microbatch_frontier_dispatcher_stream5_test,green_claim_after_tiny_path]
- code_guard=`production_v6_dispatcher.py::require_production_microbatch`
- code_guard_behavior=`invalid_config` raised when `B_MICRO != 8192` or `K_EXPAND_TILE != 196608`
- test_guard=`tests/test_architecture_v6_static.py::test_architecture_v6_production_microbatch_hard_invariant`
- touched_paths=[production_v6_dispatcher.py,tests/frontier_coverage_audit_world2.py,tests/production_dispatcher_path_world2_smoke.py,tests/real_data_100samples_depth300_beam65536_world2.py,tests/real_data_100samples_depth300_beam65536_path_audit_world2.py,tests/full_test_csv_depth300_beam65536_world2.py,tests/stream5_exchange_smoke.py,tests/stream5_2gpu_nccl_explicit_smoke.py,kaggle_frontier_coverage_audit_world2_stage/frontier_coverage_audit_world2.ipynb]
- command=`python -m py_compile production_v6_dispatcher.py tests\frontier_coverage_audit_world2.py tests\production_dispatcher_path_world2_smoke.py tests\real_data_100samples_depth300_beam65536_world2.py tests\real_data_100samples_depth300_beam65536_path_audit_world2.py tests\full_test_csv_depth300_beam65536_world2.py tests\stream5_exchange_smoke.py tests\stream5_2gpu_nccl_explicit_smoke.py tests\test_architecture_v6_static.py`; result=pass
- command=`python -m pytest tests\test_architecture_v6_static.py -q`; result=pass; summary=`47 passed in 0.23s`
- kaggle_retry_status=not_run_after_hard_invariant_update

## Kaggle Retry cc41b40 Failure Parse

- entity_id=kaggle_retry_cc41b40_result
- state=failed_invalid_config_and_collective_mismatch
- green_claim=forbidden
- observed_config={K_EXPAND_TILE=96,BUCKET_CAP_PER_PEER=131072,BUCKET_CAP_PER_PEER_SAFE=131072}
- required_config={K_EXPAND_TILE=196608,BUCKET_CAP_PER_PEER=262144,BUCKET_CAP_PER_PEER_SAFE=262144}
- observed_runtime={returncode=1,NCCL_ALLGATHER_timeout=true}
- root_cause_1=BeamEngine_config_guard_missing_or_not_used_by_Kaggle_runner
- root_cause_2=non_uniform_rank_exit_after_empty_next_frontier

## Correction Patch After cc41b40 Failure

- code_guard=production_v6_dispatcher.py::PRODUCTION_V6_CONFIG_GUARD
- code_guard_behavior=fail_before_BeamEngine_buffers_when_B_MICRO_or_K_EXPAND_TILE_or_BUCKET_CAP_PER_PEER_invalid
- runner_guard=tests/frontier_coverage_audit_world2.py::FRONTIER_COVERAGE_PRE_TORCHRUN_CONFIG
- kaggle_guard=kaggle_frontier_coverage_audit_world2_stage/frontier_coverage_audit_world2.ipynb::FRONTIER_COVERAGE_PRE_TORCHRUN_CONFIG
- distributed_debug=COLLECTIVE_SEQ_TAG_before_collectives
- uniform_exit_patch=rank_uniform_task_barrier_after_each_frontier_audit_task
- command=`python -m py_compile production_v6_dispatcher.py tests\frontier_coverage_audit_world2.py tests\test_architecture_v6_static.py`; result=pass
- command=`python -m pytest tests\test_architecture_v6_static.py -q`; result=pass; summary=`47 passed in 0.35s`
- git_commit=`5c2e723 Guard v6 config and rank-uniform frontier exits`
- github_push=success
- kaggle_retry_after_correction_patch={kernel=trydotatwo/frontier-coverage-audit-w2,version=6,status=KernelWorkerStatus.COMPLETE}
- kaggle_log_status=required_UI_log_excerpt_pending
- kaggle_log_required=[K_EXPAND_TILE_196608,BUCKET_CAP_PER_PEER_262144,returncode_0,rank0_marker,rank1_marker,completion_marker,no_NCCL_ALLGATHER_timeout]

## Pending External Validation

- target=Kaggle_2xT4_frontier_coverage_audit
- required_params={task_count=10,max_depth=12,beam_width=65536,b_micro=8192}
- expected_config_log={K_EXPAND_TILE=196608,BUCKET_CAP_PER_PEER=262144,BUCKET_CAP_PER_PEER_SAFE=262144}
- required_runtime=[torchrun_returncode_0,rank0_marker,rank1_marker,completion_marker,output_csv,jsonl]
- required_coverage=[expanded_parent_count_equals_current_frontier_size_before,stream1_scored_parent_count_equals_expanded_parent_count,stream2_generated_candidate_count_equals_expanded_parent_count_times_24]

## Claim Boundary

- green_claim=false
- reason=Kaggle_2xT4_frontier_coverage_audit_not_run_in_this_record
- forbidden_claims=[real_solver_quality,leaderboard_quality,performance_tuning,full_production_solver_quality]
