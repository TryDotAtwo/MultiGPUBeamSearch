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
