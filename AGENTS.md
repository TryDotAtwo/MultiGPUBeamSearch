# Project Agent Rules

entity_id=project_rules; type=agent_protocol; state=active

## Required Startup

- action=read_first; target=project_rules; params={files=[AGENTS.md, docs/PROJECT_MEMORY.md, docs/KAGGLE_T4_DEBUG.md]}
- action=update_memory; target=docs/PROJECT_MEMORY.md; params={when=each_task, include=[prompt_summary, decisions, commands, constraints, outcomes]}
- action=preserve_history; target=docs/PROJECT_MEMORY.md; params={include=[prompt_history, change_history]}

## Communication Protocol

- protocol=AML-HIP; requirements=[explicit_entities, key_value_lines, low_ambiguity, high_information_density]
- avoid=[implicit_references, vague_actions, hidden_dependencies]

## Code Change Gate

- rule=no_logic_change_without_user_approval; scope=[cpp, cuda, python_algorithm, notebook_algorithm, distributed_runtime]
- before_logic_change_required_fields=[problem, reason, proposed_change, expected_effect, risk, verification_plan]
- allowed_without_user_approval=[read_files, run_diagnostics, update_project_memory, update_documentation_without_algorithm_change]
- rule=static_arrays_only; scope=[gpu_data_plane, distributed_buffers, hot_path_runtime]; requirement=all primary arrays are fixed-size and allocated before search/program hot path starts; forbidden=[dynamic_device_allocation, growing_buffers, runtime_container_growth_for_candidates, unbounded_queues]
- allowed_static_array_pattern=[preallocated_score_ring, preallocated_send_recv_buckets, preallocated_hash_table, preallocated_histograms, preallocated_counters, fixed_capacity_overflow_counters]

## Work Stages

- stage=1; target=Kaggle_2xT4; access=Kaggle_CLI; notebook_url=https://www.kaggle.com/code/trydotatwo/notebookaafc902d8e/edit; purpose=debug_code_correctness
- stage=2; target=cluster_2xH100; access=SSH; purpose=final_debug_before_scale
- stage=3; target=cluster_100xH100; access=cluster_runtime; purpose=production_scale

## Current Technical Scope

- project=CayleyBeam100H100; domain=GPU_resident_distributed_beam_search; languages=[C++, CUDA, Python]
- priority_order=[correctness, reproducibility, GPU_residency, distributed_behavior, performance]
- key_files=[beam_engine.cpp, beam_kernels.cu, beam_engine.py, scripts/kaggle_correctness_check.py, notebooks/kaggle_2xt4_debug.ipynb]
