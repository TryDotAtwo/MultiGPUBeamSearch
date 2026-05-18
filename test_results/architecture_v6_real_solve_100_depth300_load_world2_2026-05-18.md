# architecture_v6_real_solve_100_depth300_load_world2

entity_id=architecture_v6_real_solve_100_depth300_load_world2; type=production_load_validation; state=host_green_kaggle_retry_pending

## Scope

- hardware=Kaggle_2xT4
- input=data/test.csv
- task_count=100
- max_depth=300
- beam_width=65536
- hard_invariants={B_MICRO=8192,K_EXPAND_TILE=196608,BUCKET_CAP_PER_PEER=262144}
- forbidden=[tiny_microbatch_path,B_MICRO_4,fallback_backend,runtime_120_slice,quality_claim,leaderboard_claim,per_depth_stdout_spam,hidden_subprocess_output]

## Implementation

- runner=tests/real_solve_100_depth300_load_world2.py
- kaggle_stage=kaggle_real_solve_100_depth300_load_world2_stage
- notebook_runner=live_subprocess_Popen_stdout_stream
- cuda_graphs=USE_CUDA_GRAPHS=1
- sparse_logs=[RUN_START,CONFIG_GUARD_OK,CUDA_GRAPHS_ENABLED,TASK_SOLVED,TASK_DONE,HEARTBEAT,RUN_SUMMARY]
- output_csv=/kaggle/working/real_solve_100_depth300_load_world2.csv
- stats_jsonl=/kaggle/working/real_solve_100_depth300_load_world2_stats.jsonl

## Host Checks

- command=`python -m py_compile production_v6_dispatcher.py tests\real_solve_100_depth300_load_world2.py tests\test_architecture_v6_static.py`; result=pass
- command=`python -m pytest tests\test_architecture_v6_static.py -q`; result=pass; summary=`48 passed in 0.44s`

## Kaggle Validation

- status=failed_runtime_errors_hidden_by_runner_v1
- observed_log={RUN_START=true,CUDA_GRAPHS_ENABLED=true,CONFIG_GUARD_OK_rank0=true,CONFIG_GUARD_OK_rank1=true,K_EXPAND_TILE=196608,BUCKET_CAP_PER_PEER=262144,BUCKET_CAP_PER_PEER_SAFE=262144,TASK_DONE_rows=100,TASK_SOLVED_rows=1,error_count=198,RUN_SUMMARY=false,returncode=nonzero_expected_after_assert}
- failure_class=runner_exception_path_hidden
- patch_pending=runner_v2_prints_TASK_ERROR_with_exception_note_and_RUN_ABORT_on_first_error
- status_v2=failed_invalid_score_ring_depth_and_nameerror
- observed_log_v2={SCORE_RING_DEPTH=1,TASK_ERROR_task_idx=0,note="NameError: name 'send_request_total' is not defined",RUN_ABORT=true,output_rows=1,RUN_SUMMARY=false}
- patch_v3={production_dispatcher_score_ring_depth=2,beam_engine_cpp_auto_min=2,beam_engine_py_auto_min=2,send_request_total_removed,rank_ok_after_abort_removed,error_count_assert_before_output_rows_assert=true}
- host_check_v3=`python -m py_compile production_v6_dispatcher.py beam_engine.py tests\real_solve_100_depth300_load_world2.py tests\test_architecture_v6_static.py`; result=pass
- host_static_pytest_v3=`python -m pytest tests\test_architecture_v6_static.py -q`; result=pass; summary=`48 passed in 0.23s`
- required=[Kaggle_status_COMPLETE,torchrun_returncode_0,runtime_B_MICRO_8192,runtime_K_EXPAND_TILE_196608,runtime_BUCKET_CAP_PER_PEER_262144,CUDA_GRAPHS_ENABLED_true,no_NCCL_timeout,output_rows_100,error_count_0,RUN_SUMMARY_present]
- green_claim=false
