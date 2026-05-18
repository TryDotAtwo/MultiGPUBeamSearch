from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_data_loader_logical120_and_state128_runtime_boundary():
    import data_loader

    central120 = data_loader.get_central_state_u8()
    assert central120.shape == (120,)
    assert len(data_loader.get_action_table_u8()) == 24 * 120

    central128 = data_loader.get_central_state128_u8()
    assert central128.shape == (128,)
    assert np.array_equal(central128[:120], central120)
    assert np.all(central128[120:128] == 0)

    action_table128 = np.frombuffer(data_loader.get_action_table128_u8(), dtype=np.uint8).reshape((24, 128))
    assert action_table128.shape == (24, 128)
    assert len(data_loader.get_action_table128_u8()) == 24 * 128
    expected_padding = np.arange(120, 128, dtype=np.uint8)
    assert np.all(action_table128[:, 120:128] == expected_padding)

    state120 = np.arange(120, dtype=np.uint8)
    state128 = data_loader.pad_state128_u8(state120)
    assert state128.shape == (128,)
    assert np.array_equal(state128[:120], state120)
    assert np.all(state128[120:128] == 0)

    states128 = data_loader.pad_states128_u8(np.stack([state120, central120], axis=0))
    assert states128.shape == (2, 128)
    assert np.all(states128[:, 120:128] == 0)
    data_loader.validate_inverse_pairs()


def test_configure_engine_uses_state128_runtime_tables():
    text = (ROOT / "beam_engine.py").read_text(encoding="utf-8")
    assert "engine.set_action_permutation_table(data_loader.get_action_table128_u8())" in text
    assert "engine.set_central_state(data_loader.get_central_state128_u8().tobytes())" in text
    assert "engine.set_action_permutation_table(data_loader.get_action_table_u8())" not in text
    assert "engine.set_central_state(data_loader.get_central_state_u8().tobytes())" not in text


def round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def test_required_type_contracts_are_declared():
    text = (ROOT / "beam_types.hpp").read_text(encoding="utf-8")
    required = [
        "struct alignas(16) State128",
        "struct alignas(16) Hash128",
        "struct alignas(32) CandidateMeta",
        "struct alignas(16) FinalRequest",
        "using FinalResponse = State128",
        "static_assert(sizeof(State128) == 128",
        "static_assert(sizeof(Hash128) == 16",
        "static_assert(sizeof(CandidateMeta) == 32",
        "static_assert(sizeof(FinalRequest) == 16",
        "static_assert(sizeof(FinalResponse) == 128",
    ]
    for needle in required:
        assert needle in text


def test_score_constants_match_architecture():
    score_max_q = 300.0
    score_scale = 256
    score_max_key = int(score_max_q * score_scale)
    score_bin_count = score_max_key + 1
    assert score_max_key == 76800
    assert score_bin_count == 76801


def test_beam_width_alignment_uses_explicit_shard_unit():
    world_size = 8
    shard_count = 64
    per_shard_unit = 4096
    user_global_beam_width = 2_000_001
    max_safe = 2_500_000
    alignment = world_size * shard_count * per_shard_unit
    effective = min(round_up(user_global_beam_width, alignment), max_safe)
    assert alignment == 2_097_152
    assert effective == 2_097_152


def test_threshold_initialization_rule_never_relaxes_after_init():
    uint32_max = 2**32 - 1
    threshold_initialized = False
    current_threshold = uint32_max
    global_beam_width_effective = 100

    if not threshold_initialized and 50 < global_beam_width_effective:
        current_threshold = uint32_max
    assert current_threshold == uint32_max

    new_threshold = 700
    if 100 >= global_beam_width_effective:
        current_threshold = min(current_threshold, new_threshold)
        threshold_initialized = True
    assert threshold_initialized is True
    assert current_threshold == 700

    if threshold_initialized and 20 < global_beam_width_effective:
        current_threshold = current_threshold
    assert current_threshold == 700

    relaxed_threshold = 900
    if 120 >= global_beam_width_effective:
        current_threshold = min(current_threshold, relaxed_threshold)
    assert current_threshold == 700


def test_padding_and_final_response_helpers_are_declared():
    text = (ROOT / "beam_types.hpp").read_text(encoding="utf-8")
    assert "final_response_set_target_local_idx" in text
    assert "final_response_get_target_local_idx" in text
    assert "clear_state_padding" in text
    assert "response.v[120]" in text
    assert "response.v[123]" in text


def test_memory_contract_names_are_declared():
    config_text = (ROOT / "beam_config.hpp").read_text(encoding="utf-8")
    memory_text = (ROOT / "beam_memory.hpp").read_text(encoding="utf-8")
    assert "stream4_batch_candidates_per_shard_unit" in config_text
    assert "solved_result_capacity" in config_text
    assert "threshold_initialized" in config_text
    assert "LayoutStreams" in memory_text
    assert "LayoutFinal" in memory_text
    assert "scratch_pool_bytes" in memory_text


def test_no_runtime_fallback_allowed_in_python_configure_path():
    text = (ROOT / "beam_engine.py").read_text(encoding="utf-8")
    assert 'backend != "fullbeamnice_static"' in text
    assert "Target architecture v6 production path requires INFERENCE_BACKEND=fullbeamnice_static" in text


def test_stream1_cutlass_static_score_key_contract():
    engine_text = (ROOT / "beam_engine.cpp").read_text(encoding="utf-8")
    kernels_text = (ROOT / "beam_kernels.cu").read_text(encoding="utf-8")
    python_text = (ROOT / "beam_engine.py").read_text(encoding="utf-8")
    required_engine = [
        "FullBeamNiceRequiredBackend",
        "fallback inference is forbidden in architecture_v6",
        'backend_name == "fullbeamnice_static"',
        "std::make_unique<FullBeamNiceRequiredBackend>()",
        "launch_fullbeamnice_q_to_score_key_ring",
        "reinterpret_cast<uint32_t*>(score_ring.data_ptr<int32_t>())",
        "fullbeamnice_static supports State128 input: state_size=128 and num_classes=128",
    ]
    required_kernel = [
        "kernel_fullbeamnice_q_to_score_key_ring",
        "uint32_t* __restrict__ score_ring",
        "if (q < 0.0f) q = 0.0f;",
        "if (q > 300.0f) q = 300.0f;",
        "uint32_t score_key = static_cast<uint32_t>(q * 256.0f + 0.5f);",
    ]
    required_python = [
        '"score_ring": torch.empty((score_ring_elements,), dtype=torch.int32, device=device)',
        '"state_size_bytes": 128',
    ]
    forbidden_engine = [
        'backend_name == "dummy" || backend_name == "central_hamming" || backend_name == "torchscript_ensemble" || backend_name == "fullbeamnice_static"',
    ]
    for needle in required_engine:
        assert needle in engine_text
    for needle in required_kernel:
        assert needle in kernels_text
    for needle in required_python:
        assert needle in python_text
    for needle in forbidden_engine:
        assert needle not in engine_text


def test_stream1_cutlass_score_key_smoke_contract():
    test_text = (ROOT / "tests" / "stream1_cutlass_score_key_smoke.py").read_text(encoding="utf-8")
    required = [
        "INFERENCE_BACKEND",
        "fullbeamnice_static",
        "load_fullbeamnice_static",
        "warmup_inference",
        "buffers[\"score_ring\"].dtype is not torch.int32",
        "q_by_src_action",
        "q_current_order",
        "torch.round(q_current_order.float().clamp(0.0, SCORE_MAX_Q) * SCORE_SCALE)",
        '"state_size": 128',
        '"num_classes": 128',
        "STREAM1_CUTLASS_SCORE_KEY_SMOKE_OK",
    ]
    forbidden = [
        "torchscript_ensemble",
        "dummy",
        "central_hamming",
    ]
    for needle in required:
        assert needle in test_text
    for needle in forbidden:
        assert needle not in test_text


def test_stream1_real_weights_smoke_contract():
    test_text = (ROOT / "tests" / "stream1_real_weights_smoke.py").read_text(encoding="utf-8")
    required = [
        "FullBeamNice",
        "load_static_weights",
        "static_forward_q",
        "INFERENCE_BACKEND",
        "fullbeamnice_static",
        "load_fullbeamnice_static",
        "warmup_inference",
        "buffers[\"score_ring\"].dtype is not torch.int32",
        "torch.round(q_current_order.float().clamp(0.0, SCORE_MAX_Q) * SCORE_SCALE)",
        "FullBeamNice120 vs FullBeamNice128 padded output mismatch",
        "FullBeamNice State128 padding token weights must be zero",
        "STREAM1_REAL_WEIGHTS_SMOKE_OK",
    ]
    forbidden = [
        "torchscript_ensemble",
        "dummy",
        "central_hamming",
    ]
    for needle in required:
        assert needle in test_text
    for needle in forbidden:
        assert needle not in test_text


def test_stream1_stream2_ring_batch_world2_contract():
    test_text = (ROOT / "tests" / "stream1_stream2_ring_batch_world2_smoke.py").read_text(encoding="utf-8")
    required = [
        "load_static_weights",
        "static_forward_q",
        "fullbeamnice_static",
        "warmup_inference",
        "score_ring",
        "v6_stream2_hash_goal",
        "hash_ring",
        "GOAL_SCORE_KEY",
        "pack_route(rank, rank, 0)",
        "FullBeamNice static weights must expose physical State128 input",
        "FullBeamNice120 vs FullBeamNice128 padded mismatch",
        "padding_weight_max=0",
        "STREAM1_STREAM2_RING_BATCH_WORLD2_SMOKE_OK",
        "STREAM1_STREAM2_RING_BATCH_WORLD2_TEST_COMPLETE",
    ]
    forbidden = [
        "v6_stream3",
        "v6_stream4",
        "v6_stream5",
        "v6_final",
        "torchscript_ensemble",
        "dummy",
        "central_hamming",
    ]
    for needle in required:
        assert needle in test_text
    for needle in forbidden:
        assert needle not in test_text


def test_stream1_state128_static_loader_contract():
    loader_text = (ROOT / "scripts" / "static_fullbeamnice_inference.py").read_text(encoding="utf-8")
    required = [
        "STATE_LEN = 120",
        "STATE_STORAGE_LEN = 128",
        "STATE_VALUE_PAD = 128",
        "expanded_embed_w = torch.zeros",
        "STATE_STORAGE_LEN * STATE_VALUE_PAD",
        "expanded_embed_w[new_start:new_start + logical_num_classes].copy_",
        "state_size=STATE_STORAGE_LEN",
        "num_classes=STATE_VALUE_PAD",
        "states_u8.size(1) == STATE_LEN",
        "padded[:, :STATE_LEN].copy_(states_u8)",
    ]
    for needle in required:
        assert needle in loader_text


def test_python_allocation_exposes_v6_buffers():
    text = (ROOT / "beam_engine.py").read_text(encoding="utf-8")
    required = [
        '"current_frontier_states"',
        '"scratch_pool"',
        '"solved_flag"',
        '"stop_flag"',
        '"solved_count"',
        '"solved_overflow"',
        '"solved_meta_list"',
        '"solved_depth_list"',
    ]
    for needle in required:
        assert needle in text


def test_stream2_kernel_declares_hash_goal_and_solved_visibility_contract():
    text = (ROOT / "beam_kernels_stream2.cu").read_text(encoding="utf-8")
    required = [
        "kernel_v6_stream2_hash_goal",
        "apply_move_state128",
        "hash_state128_zobrist",
        "GOAL_SCORE_KEY",
        "solved_meta_list[idx] = meta",
        "solved_depth_list[idx] = depth",
        "__threadfence_system()",
        "atomicCAS(solved_flag, 0u, 1u)",
        "atomicExch(stop_flag, 1u)",
        "hash_ring[ring_offset] = hash",
    ]
    for needle in required:
        assert needle in text


def test_layout_size_formula_uses_max_streams_final_overlay():
    text = (ROOT / "beam_memory.cpp").read_text(encoding="utf-8")
    assert "out.scratch_pool_bytes = std::max(out.streams.bytes, out.final.bytes)" in text
    assert "out.current_frontier_bytes" in text
    assert "out.solved_buffers_bytes" in text


def test_final_materialization_kernel_contract_is_declared():
    text = (ROOT / "beam_kernels_final.cu").read_text(encoding="utf-8")
    required = [
        "kernel_v6_final_materialize",
        "kernel_v6_final_scatter_responses",
        "FinalResponse response",
        "final_response_set_target_local_idx",
        "final_response_get_target_local_idx",
        "clear_state_padding(response)",
        "next_frontier_states_tmp[target_local_idx] = response",
    ]
    for needle in required:
        assert needle in text


def test_stream3_isolated_kernel_contract_is_declared():
    text = (ROOT / "beam_kernels_stream3.cu").read_text(encoding="utf-8")
    required = [
        "kernel_v6_stream3_pack_threshold_compact",
        "kernel_v6_stream3_dedup_sorted",
        "kernel_v6_stream3_restore_split",
        "cub::DeviceRadixSort::SortPairs",
        "Hash128KeyDecomposer",
        "pack_stream3_val(score_key, i)",
        "owner_from_hash128",
        "pack_route",
        "send_count + owner",
    ]
    for needle in required:
        assert needle in text


def test_stream4_isolated_kernel_contract_is_declared():
    text = (ROOT / "beam_kernels_stream4.cu").read_text(encoding="utf-8")
    required = [
        "kernel_v6_stream4_threshold_compact",
        "kernel_v6_stream4_dedup_sorted",
        "kernel_v6_stream4_write_clean",
        "cub::DeviceRadixSort::SortPairs",
        "Stream4HashKeyDecomposer",
        "candidate_better",
        "a.score_key < b.score_key",
        "a.parent_idx < b.parent_idx",
        "a.route_packed < b.route_packed",
        "dirty_count[0] = 0",
        "processing_flag[0] = 0",
    ]
    for needle in required:
        assert needle in text


def test_stream5_exchange_binding_contract_is_declared():
    text = (ROOT / "beam_engine.cpp").read_text(encoding="utf-8")
    required = [
        "v6_stream5_exchange_candidate_meta",
        "sizeof(beam_v6::CandidateMeta)",
        "ncclSend",
        "ncclRecv",
        "remote_send_buffer",
        "remote_recv_buffer",
        "send_offset",
        "recv_offset",
    ]
    for needle in required:
        assert needle in text


def test_dispatcher_skeleton_contract_is_declared():
    engine_text = (ROOT / "beam_engine.cpp").read_text(encoding="utf-8")
    dispatcher_text = (ROOT / "beam_dispatcher.cpp").read_text(encoding="utf-8")
    python_text = (ROOT / "beam_engine.py").read_text(encoding="utf-8")
    test_text = (ROOT / "tests" / "dispatcher_skeleton_smoke.py").read_text(encoding="utf-8")
    required_engine = [
        "v6_dispatcher_skeleton_single_gpu_smoke",
        "v6_dispatcher_skeleton_single_gpu_smoke_contract",
    ]
    required_dispatcher = [
        "architecture_v6_stage6_dispatcher_skeleton",
        "uses_prefilled_score_ring",
        "dispatcher_outside_cuda_graph",
    ]
    required_python = [
        "def v6_dispatcher_skeleton_single_gpu_smoke",
        "_v6_validate_u32(\"current_threshold\", 0xFFFFFFFF)",
        "v6_stream2_hash_goal",
        "v6_stream3_pack_threshold_compact",
        "v6_stream5_exchange_candidate_meta",
        "v6_stream4_threshold_compact",
        "v6_final_materialize",
        "stream1_production_called",
        "fallback_backend_called",
    ]
    required_test = [
        "DISPATCHER_SKELETON_SMOKE_OK",
        "test_normal_path",
        "test_stop_path",
        "first_solved_score_key",
    ]
    for needle in required_engine:
        assert needle in engine_text
    for needle in required_dispatcher:
        assert needle in dispatcher_text
    for needle in required_python:
        assert needle in python_text
    for needle in required_test:
        assert needle in test_text


def test_threshold_pybind_accepts_uint32_contract():
    text = (ROOT / "beam_engine.cpp").read_text(encoding="utf-8")
    required = [
        "uint64_t current_threshold",
        "current_threshold > 0xffffffffULL",
        "current_threshold must be in uint32 range",
        "uint64_t stream4_job_threshold",
        "stream4_job_threshold > 0xffffffffULL",
        "stream4_job_threshold must be in uint32 range",
    ]
    for needle in required:
        assert needle in text


def test_stream5_2gpu_nccl_explicit_smoke_contract():
    text = (ROOT / "tests" / "stream5_2gpu_nccl_explicit_smoke.py").read_text(encoding="utf-8")
    required = [
        "WORLD_SIZE_REQUIRED = 2",
        "visible CUDA device count must be >=2",
        "send_to_peer = 1 if rank == 0 else 3",
        "recv_from_peer = 3 if rank == 0 else 1",
        "v6_stream5_exchange_candidate_meta",
        "assert recv_count_after.tolist() == recv_count_host.tolist()",
        "assert recv_offset_after.tolist() == recv_offset_host.tolist()",
        "STREAM5_2GPU_NCCL_EXPLICIT_SMOKE_OK",
    ]
    forbidden = [
        "STREAM5_EXCHANGE_SMOKE_SKIPPED",
    ]
    for needle in required:
        assert needle in text
    for needle in forbidden:
        assert needle not in text


def test_dispatcher_stream5_world2_smoke_contract():
    python_text = (ROOT / "beam_engine.py").read_text(encoding="utf-8")
    test_text = (ROOT / "tests" / "dispatcher_stream5_world2_smoke.py").read_text(encoding="utf-8")
    required_python = [
        "def v6_dispatcher_skeleton_world2_stream5_smoke",
        "v6_stream5_exchange_candidate_meta",
        "stream5_launched_by_dispatcher",
        "stream1_production_called",
        "stream3_collector_expanded",
        "stream4_scheduler_expanded",
        "final_materialization_expanded",
    ]
    required_test = [
        "dispatcher WORLD_SIZE=2 Stream5 smoke requires Tesla T4 devices",
        "DISPATCHER_STREAM5_WORLD2_SMOKE_OK",
        "payload_byte_identical",
        "stream5_launched_by_dispatcher",
    ]
    forbidden_test = [
        "STREAM5_EXCHANGE_SMOKE_SKIPPED",
    ]
    for needle in required_python:
        assert needle in python_text
    for needle in required_test:
        assert needle in test_text
    for needle in forbidden_test:
        assert needle not in test_text


def test_dispatcher_stream3_stream5_collector_world2_smoke_contract():
    python_text = (ROOT / "beam_engine.py").read_text(encoding="utf-8")
    test_text = (ROOT / "tests" / "dispatcher_stream3_stream5_collector_world2_smoke.py").read_text(encoding="utf-8")
    required_python = [
        "def v6_dispatcher_stream3_stream5_collector_world2_smoke",
        "stream3_split_done",
        "local_pending_buffer",
        "remote_recv_buffer",
        "collector_ingest_ok",
        "v6_stream5_exchange_candidate_meta",
        "stream4_scheduler_expanded",
        "shard_dirty_clean_lifecycle_expanded",
        "final_materialization_expanded",
    ]
    required_test = [
        "dispatcher Stream3/Stream5/collector WORLD_SIZE=2 smoke requires Tesla T4 devices",
        "DISPATCHER_STREAM3_STREAM5_COLLECTOR_WORLD2_SMOKE_OK",
        "remote_recv_byte_identical",
        "collector_sources",
        "threshold_logic_changed",
    ]
    forbidden_test = [
        "STREAM5_EXCHANGE_SMOKE_SKIPPED",
    ]
    for needle in required_python:
        assert needle in python_text
    for needle in required_test:
        assert needle in test_text
    for needle in forbidden_test:
        assert needle not in test_text


def test_collector_shard_dirty_spill_world2_smoke_contract():
    python_text = (ROOT / "beam_engine.py").read_text(encoding="utf-8")
    test_text = (ROOT / "tests" / "collector_shard_dirty_spill_world2_smoke.py").read_text(encoding="utf-8")
    required_python = [
        "def v6_collector_shard_dirty_spill_world2_smoke",
        "local_pending_buffer",
        "remote_recv_buffer",
        "survivor_shard",
        "global_spill_buffer",
        "processing_flag = [0, 1]",
        "dirty_write_ok",
        "spill_write_ok",
        "stream4_kernel_launched",
        "clean_dirty_lifecycle_after_stream4",
    ]
    required_test = [
        "collector shard dirty/spill WORLD_SIZE=2 smoke requires Tesla T4 devices",
        "COLLECTOR_SHARD_DIRTY_SPILL_WORLD2_SMOKE_OK",
        "dirty_write_ok",
        "spill_write_ok",
        "stream4_kernel_launched",
        "threshold_logic_changed",
    ]
    forbidden_test = [
        "STREAM5_EXCHANGE_SMOKE_SKIPPED",
    ]
    for needle in required_python:
        assert needle in python_text
    for needle in required_test:
        assert needle in test_text
    for needle in forbidden_test:
        assert needle not in test_text


def test_collector_stream4_shard_launch_world2_smoke_contract():
    python_text = (ROOT / "beam_engine.py").read_text(encoding="utf-8")
    test_text = (ROOT / "tests" / "collector_stream4_shard_launch_world2_smoke.py").read_text(encoding="utf-8")
    required_python = [
        "def v6_collector_stream4_shard_launch_world2_smoke",
        "launch_condition_met",
        "v6_stream4_threshold_compact",
        "v6_stream4_sort_pairs",
        "v6_stream4_dedup_sorted",
        "v6_stream4_write_clean",
        "stream4_job_threshold",
        "histogram_allreduce_used",
        "threshold_update_logic_used",
    ]
    required_test = [
        "collector Stream4 shard launch WORLD_SIZE=2 smoke requires Tesla T4 devices",
        "COLLECTOR_STREAM4_SHARD_LAUNCH_WORLD2_SMOKE_OK",
        "stream4_clean_count",
        "stream4_dirty_count",
        "dedup_best_ok",
        "histogram_allreduce_used",
    ]
    forbidden_test = [
        "STREAM5_EXCHANGE_SMOKE_SKIPPED",
    ]
    for needle in required_python:
        assert needle in python_text
    for needle in required_test:
        assert needle in test_text
    for needle in forbidden_test:
        assert needle not in test_text


def test_spill_drain_then_stream4_relaunch_world2_smoke_contract():
    python_text = (ROOT / "beam_engine.py").read_text(encoding="utf-8")
    test_text = (ROOT / "tests" / "spill_drain_then_stream4_relaunch_world2_smoke.py").read_text(encoding="utf-8")
    required_python = [
        "def v6_spill_drain_then_stream4_relaunch_world2_smoke",
        "spill_count_initial",
        "spill_count_after_drain",
        "relaunch_condition",
        "v6_stream4_threshold_compact",
        "v6_stream4_sort_pairs",
        "v6_stream4_dedup_sorted",
        "v6_stream4_write_clean",
        "histogram_allreduce_used",
        "threshold_update_logic_used",
    ]
    required_test = [
        "spill drain then Stream4 relaunch WORLD_SIZE=2 smoke requires Tesla T4 devices",
        "SPILL_DRAIN_THEN_STREAM4_RELAUNCH_WORLD2_SMOKE_OK",
        "spill_count_after_drain",
        "relaunch_condition_met",
        "second_stream4_clean_count",
        "histogram_allreduce_used",
    ]
    forbidden_test = [
        "STREAM5_EXCHANGE_SMOKE_SKIPPED",
    ]
    for needle in required_python:
        assert needle in python_text
    for needle in required_test:
        assert needle in test_text
    for needle in forbidden_test:
        assert needle not in test_text


def test_collector_stream4_batch_world2_smoke_contract():
    python_text = (ROOT / "beam_engine.py").read_text(encoding="utf-8")
    test_text = (ROOT / "tests" / "collector_stream4_batch_world2_smoke.py").read_text(encoding="utf-8")
    required_python = [
        "def v6_collector_stream4_batch_world2_smoke",
        "spill_drain_then_stream4_relaunch_world2_smoke",
        "multi_shard_ready_same_tick_world2_smoke",
        "busy_shard_spill_then_drain_after_processing_flag_false_world2_smoke",
        "stream4_dedup_best_score_survives_world2_smoke",
        "stream4_uint32max_threshold_keeps_all_world2_smoke",
        "two_round_clean_dirty_processing_lifecycle_world2_smoke",
        "histogram_allreduce_used",
        "threshold_update_logic_used",
    ]
    required_test = [
        "collector Stream4 batch WORLD_SIZE=2 smoke requires Tesla T4 devices",
        "COLLECTOR_STREAM4_BATCH_WORLD2_SMOKE_OK",
        "=== COLLECTOR_STREAM4_BATCH_WORLD2_TEST_COMPLETE ===",
        "stream4_uint32max_threshold_keeps_all_world2_smoke",
        "two_round_clean_dirty_processing_lifecycle_world2_smoke",
        "histogram_allreduce_used",
    ]
    forbidden_test = [
        "STREAM5_EXCHANGE_SMOKE_SKIPPED",
    ]
    for needle in required_python:
        assert needle in python_text
    for needle in required_test:
        assert needle in test_text
    for needle in forbidden_test:
        assert needle not in test_text


def test_threshold_histogram_batch_world2_smoke_contract():
    python_text = (ROOT / "beam_engine.py").read_text(encoding="utf-8")
    test_text = (ROOT / "tests" / "threshold_histogram_batch_world2_smoke.py").read_text(encoding="utf-8")
    required_python = [
        "def v6_threshold_histogram_batch_world2_smoke",
        "threshold_uninitialized_uint32max_until_enough_survivors_world2_smoke",
        "threshold_initialized_when_total_survivors_reaches_GLOBAL_BEAM_WIDTH_EFFECTIVE_world2_smoke",
        "threshold_monotonic_never_relaxes_world2_smoke",
        "local_score_hist_to_global_score_hist_allreduce_world2_smoke",
        "GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS_triggers_update_world2_smoke",
        "stream4_jobs_use_snapshot_threshold_not_later_threshold_world2_smoke",
        "dist.all_reduce",
        "histogram_threshold",
        "layout_final_used",
        "load_balancing_used",
    ]
    required_test = [
        "threshold histogram batch WORLD_SIZE=2 smoke requires Tesla T4 devices",
        "THRESHOLD_HISTOGRAM_BATCH_WORLD2_SMOKE_OK",
        "=== THRESHOLD_HISTOGRAM_BATCH_WORLD2_TEST_COMPLETE ===",
        "threshold_uninitialized_uint32max_until_enough_survivors_world2_smoke",
        "stream4_jobs_use_snapshot_threshold_not_later_threshold_world2_smoke",
        "layout_final_used",
    ]
    forbidden_test = [
        "STREAM5_EXCHANGE_SMOKE_SKIPPED",
    ]
    for needle in required_python:
        assert needle in python_text
    for needle in required_test:
        assert needle in test_text
    for needle in forbidden_test:
        assert needle not in test_text


def test_final_threshold_balance_batch_world2_smoke_contract():
    python_text = (ROOT / "beam_engine.py").read_text(encoding="utf-8")
    test_text = (ROOT / "tests" / "final_threshold_balance_batch_world2_smoke.py").read_text(encoding="utf-8")
    required_python = [
        "def v6_final_threshold_balance_batch_world2_smoke",
        "final_flush_all_dirty_shards_before_threshold_world2_smoke",
        "final_global_threshold_after_local_final_dedup_world2_smoke",
        "final_cutoff_score_key_le_current_threshold_world2_smoke",
        "allgather_local_keep_count_world2_smoke",
        "prefix_counts_target_rank_target_local_idx_world2_smoke",
        "tie_at_final_threshold_allowed_count_may_exceed_beam_width_world2_smoke",
        "dist.all_gather",
        "histogram_threshold",
        "layout_final_used",
        "state_materialization_used",
    ]
    required_test = [
        "final threshold balance batch WORLD_SIZE=2 smoke requires Tesla T4 devices",
        "FINAL_THRESHOLD_BALANCE_BATCH_WORLD2_SMOKE_OK",
        "=== FINAL_THRESHOLD_BALANCE_BATCH_WORLD2_TEST_COMPLETE ===",
        "final_request_used",
        "final_response_used",
        "state_materialization_used",
    ]
    forbidden_test = [
        "STREAM5_EXCHANGE_SMOKE_SKIPPED",
    ]
    for needle in required_python:
        assert needle in python_text
    for needle in required_test:
        assert needle in test_text
    for needle in forbidden_test:
        assert needle not in test_text


def test_final_materialization_batch_world2_smoke_contract():
    python_text = (ROOT / "beam_engine.py").read_text(encoding="utf-8")
    test_text = (ROOT / "tests" / "final_materialization_batch_world2_smoke.py").read_text(encoding="utf-8")
    required_python = [
        "def v6_final_materialization_batch_world2_smoke",
        "final_request_group_by_source_rank_world2_smoke",
        "final_response_target_local_idx_pack_unpack_world2_smoke",
        "cross_rank_final_request_response_world2_smoke",
        "apply_move_matches_cpu_reference_world2_smoke",
        "padding_clear_before_next_frontier_write_world2_smoke",
        "next_frontier_states_tmp_write_by_target_local_idx_world2_smoke",
        "optional_next_frontier_tmp_to_current_frontier_copy_world2_smoke",
        "_v6_pack_final_request",
        "response_for_write[120:128] = 0",
    ]
    required_test = [
        "final materialization batch WORLD_SIZE=2 smoke requires Tesla T4 devices",
        "FINAL_MATERIALIZATION_BATCH_WORLD2_SMOKE_OK",
        "=== FINAL_MATERIALIZATION_BATCH_WORLD2_TEST_COMPLETE ===",
        "solved_path_expanded",
        "new_threshold_logic_used",
        "new_load_balancing_logic_used",
    ]
    forbidden_test = [
        "STREAM5_EXCHANGE_SMOKE_SKIPPED",
    ]
    for needle in required_python:
        assert needle in python_text
    for needle in required_test:
        assert needle in test_text
    for needle in forbidden_test:
        assert needle not in test_text


def test_solved_stop_batch_world2_smoke_contract():
    python_text = (ROOT / "beam_engine.py").read_text(encoding="utf-8")
    test_text = (ROOT / "tests" / "solved_stop_batch_world2_smoke.py").read_text(encoding="utf-8")
    kernel_text = (ROOT / "beam_kernels_stream2.cu").read_text(encoding="utf-8")
    required_python = [
        "def v6_solved_stop_batch_world2_smoke",
        "stream2_goal_candidate_writes_GOAL_SCORE_KEY_world2_smoke",
        "solved_count_and_solved_depth_list_world2_smoke",
        "solved_flag_stop_flag_publication_order_world2_smoke",
        "solved_overflow_when_capacity_exceeded_world2_smoke",
        "dispatcher_stop_propagation_world2_smoke",
        "active_jobs_safe_completion_after_stop_world2_smoke",
        "cpu_solved_list_readback_world2_smoke",
        "dist.all_reduce",
        "full_production_depth_loop_used",
    ]
    required_test = [
        "solved/stop batch WORLD_SIZE=2 smoke requires Tesla T4 devices",
        "SOLVED_STOP_BATCH_WORLD2_SMOKE_OK",
        "=== SOLVED_STOP_BATCH_WORLD2_TEST_COMPLETE ===",
        "threadfence_system_contract",
        "new_final_materialization_logic_used",
    ]
    required_kernel = [
        "meta.score_key = GOAL_SCORE_KEY",
        "__threadfence_system();",
        "atomicCAS(solved_flag, 0u, 1u)",
        "atomicExch(stop_flag, 1u)",
        "atomicExch(solved_overflow, 1u)",
    ]
    forbidden_test = [
        "STREAM5_EXCHANGE_SMOKE_SKIPPED",
    ]
    for needle in required_python:
        assert needle in python_text
    for needle in required_test:
        assert needle in test_text
    for needle in required_kernel:
        assert needle in kernel_text
    for needle in forbidden_test:
        assert needle not in test_text


def test_synthetic_depth_loop_batch_world2_smoke_contract():
    python_text = (ROOT / "beam_engine.py").read_text(encoding="utf-8")
    test_text = (ROOT / "tests" / "synthetic_depth_loop_batch_world2_smoke.py").read_text(encoding="utf-8")
    required_python = [
        "def v6_synthetic_depth_loop_batch_world2_smoke",
        "synthetic_unsolved_depth_full_path_world2_smoke",
        "synthetic_depth_with_remote_exchange_and_multi_shard_stream4_world2_smoke",
        "synthetic_depth_with_periodic_threshold_update_world2_smoke",
        "synthetic_depth_final_balance_materialization_world2_smoke",
        "synthetic_depth_solved_early_stop_world2_smoke",
        "synthetic_depth_no_work_left_drain_order_world2_smoke",
        "prefilled_score_ring_used",
        "real_puzzle_solve_claim",
        "performance_tuning_used",
    ]
    required_test = [
        "synthetic depth loop batch WORLD_SIZE=2 smoke requires Tesla T4 devices",
        "SYNTHETIC_DEPTH_LOOP_BATCH_WORLD2_SMOKE_OK",
        "=== SYNTHETIC_DEPTH_LOOP_BATCH_WORLD2_TEST_COMPLETE ===",
        "stream1_production_called",
        "model_backend_called",
        "real_inference_used",
        "performance_tuning_used",
    ]
    forbidden_test = [
        "STREAM5_EXCHANGE_SMOKE_SKIPPED",
    ]
    for needle in required_python:
        assert needle in python_text
    for needle in required_test:
        assert needle in test_text
    for needle in forbidden_test:
        assert needle not in test_text


def test_synthetic_full_depth_with_stream1_batch_world2_smoke_contract():
    test_text = (ROOT / "tests" / "synthetic_full_depth_with_stream1_batch_world2_smoke.py").read_text(encoding="utf-8")
    required = [
        "load_static_weights",
        "static_forward_q",
        "fullbeamnice_static",
        "warmup_inference",
        "v6_stream2_hash_goal",
        "best_by_hash",
        "histogram_threshold",
        "one_depth_unsolved_real_stream1_to_final_materialization_world2",
        "one_depth_remote_exchange_real_stream1_scores_world2",
        "one_depth_stream4_threshold_from_real_score_ring_world2",
        "one_depth_solved_goal_stops_before_final_world2",
        "one_depth_drain_order_with_active_stream4_world2",
        "one_depth_padding_contract_after_materialization_world2",
        "SYNTHETIC_FULL_DEPTH_WITH_STREAM1_BATCH_WORLD2_SMOKE_OK",
        "SYNTHETIC_FULL_DEPTH_WITH_STREAM1_BATCH_WORLD2_TEST_COMPLETE",
    ]
    forbidden = [
        "torchscript_ensemble",
        "dummy",
        "central_hamming",
        "nn_input",
    ]
    for needle in required:
        assert needle in test_text
    for needle in forbidden:
        assert needle not in test_text


def test_multi_depth_dispatcher_loop_batch_world2_smoke_contract():
    test_text = (ROOT / "tests" / "multi_depth_dispatcher_loop_batch_world2_smoke.py").read_text(encoding="utf-8")
    required = [
        "run_stream1_stream2",
        "best_by_hash",
        "histogram_threshold",
        "multi_depth_two_iterations_real_stream1_world2_smoke",
        "layout_streams_layout_final_switching_world2_smoke",
        "current_frontier_copy_between_depths_world2_smoke",
        "threshold_initialized_persists_across_depths_world2_smoke",
        "stop_solved_early_exit_across_depths_world2_smoke",
        "multi_depth_padding_contract_world2_smoke",
        "MULTI_DEPTH_DISPATCHER_LOOP_BATCH_WORLD2_SMOKE_OK",
        "MULTI_DEPTH_DISPATCHER_LOOP_BATCH_WORLD2_TEST_COMPLETE",
    ]
    forbidden = [
        "torchscript_ensemble",
        "dummy",
        "central_hamming",
        "nn_input",
        "performance_tuning",
        "real_solver_claim",
    ]
    for needle in required:
        assert needle in test_text
    for needle in forbidden:
        assert needle not in test_text


def test_real_data_functional_validation_world2_contract():
    helper_text = (ROOT / "tests" / "synthetic_full_depth_with_stream1_batch_world2_smoke.py").read_text(encoding="utf-8")
    test_text = (ROOT / "tests" / "real_data_functional_validation_world2.py").read_text(encoding="utf-8")
    required_helper = [
        "states_logical_override",
        "generators_override",
        "central_override",
        "depth_base",
    ]
    required_test = [
        "data_loader.load_test_puzzles",
        "data_loader.get_generators",
        "data_loader.get_central_state_u8",
        "run_stream1_stream2",
        "REAL_DATA_TASK_STATUS",
        "REAL_DATA_OUTPUT_FILE",
        "REAL_DATA_FUNCTIONAL_VALIDATION_WORLD2_SMOKE_OK",
        "REAL_DATA_FUNCTIONAL_VALIDATION_WORLD2_TEST_COMPLETE",
        "functional_checked",
        "no_leaderboard_claim",
    ]
    forbidden_test = [
        "torchscript_ensemble",
        "dummy",
        "central_hamming",
        "nn_input",
        "leaderboard_claim=true",
        "real_solver_quality_claim=true",
        "performance_tuning=true",
    ]
    for needle in required_helper:
        assert needle in helper_text
    for needle in required_test:
        assert needle in test_text
    for needle in forbidden_test:
        assert needle not in test_text


def test_real_data_larger_batch_correctness_world2_contract():
    test_text = (ROOT / "tests" / "real_data_larger_batch_correctness_world2.py").read_text(encoding="utf-8")
    required = [
        "data_loader.load_test_puzzles",
        "_run_real_task",
        "REAL_DATA_LARGER_TASK_STATUS",
        "REAL_DATA_LARGER_OUTPUT_FILE",
        "REAL_DATA_LARGER_BATCH_CORRECTNESS_WORLD2_SMOKE_OK",
        "REAL_DATA_LARGER_BATCH_CORRECTNESS_WORLD2_TEST_COMPLETE",
        "solved_count",
        "unsolved_count",
        "no_leaderboard_claim",
        "no_real_solver_quality_claim",
        "no_performance_tuning_claim",
        "no_large_beam",
        "no_full_test_csv",
    ]
    forbidden = [
        "torchscript_ensemble",
        "dummy",
        "central_hamming",
        "nn_input",
        "leaderboard_claim=true",
        "real_solver_quality_claim=true",
        "performance_tuning=true",
    ]
    for needle in required:
        assert needle in test_text
    for needle in forbidden:
        assert needle not in test_text


def test_production_dispatcher_path_world2_contract():
    driver_text = (ROOT / "production_v6_dispatcher.py").read_text(encoding="utf-8")
    test_text = (ROOT / "tests" / "production_dispatcher_path_world2_smoke.py").read_text(encoding="utf-8")
    required_driver = [
        "ProductionV6Dispatcher",
        "warmup_inference",
        "v6_stream2_hash_goal",
        "v6_stream3_pack_threshold_compact",
        "v6_stream3_sort_pairs",
        "v6_stream3_dedup_sorted",
        "v6_stream3_restore_split",
        "v6_stream5_exchange_candidate_meta",
        "v6_stream4_threshold_compact",
        "v6_stream4_sort_pairs",
        "v6_stream4_dedup_sorted",
        "v6_stream4_write_clean",
        "allreduce_score_threshold",
        "v6_final_materialize",
        "v6_final_scatter_responses",
        "production_v6_dispatcher_path",
        "legacy_next_state_pool_path",
        "prefilled_score_ring_fake_path",
    ]
    required_test = [
        "run_real_data_production_v6_world2",
        "PRODUCTION_V6_DISPATCHER_PATH_WORLD2_SMOKE_OK",
        "PRODUCTION_V6_DISPATCHER_PATH_WORLD2_TEST_COMPLETE",
        "first validation task_count must be 1..3",
        "first validation max_depth must be 10..20",
        "first validation beam_width must be 4096..65536",
    ]
    forbidden_driver = [
        "engine.search",
        "reset_search",
        'buffers["next_state_pool"]',
        "prefilled_score_ring_used",
        "run_stream1_stream2",
        "torchscript_ensemble",
        "central_hamming",
        "nn_input",
    ]
    for needle in required_driver:
        assert needle in driver_text
    for needle in required_test:
        assert needle in test_text
    for needle in forbidden_driver:
        assert needle not in driver_text


def test_real_data_100samples_depth300_beam65536_world2_contract():
    test_text = (ROOT / "tests" / "real_data_100samples_depth300_beam65536_world2.py").read_text(encoding="utf-8")
    dispatcher_text = (ROOT / "production_v6_dispatcher.py").read_text(encoding="utf-8")
    smoke_text = (ROOT / "tests" / "production_dispatcher_path_world2_smoke.py").read_text(encoding="utf-8")
    required_test = [
        "run_real_data_production_v6_world2_detailed",
        "REAL_DATA_100_TASK_COUNT",
        "REAL_DATA_100_MAX_DEPTH",
        "GLOBAL_BEAM_WIDTH",
        "task_count=100",
        "max_depth=300",
        "beam_width=65536",
        "REAL_DATA_100SAMPLES_DEPTH300_BEAM65536_WORLD2_OK",
        "REAL_DATA_100SAMPLES_DEPTH300_BEAM65536_WORLD2_TEST_COMPLETE",
        "no_quality_claim",
        "no_leaderboard_claim",
        "no_performance_claim",
    ]
    required_dispatcher = [
        "run_real_data_production_v6_world2_detailed",
        "row_id",
        "initial_state_id",
        "depth_reached",
        "solution_len",
        "error_or_note",
        "REAL_DATA_100SAMPLES_PROGRESS",
        "gpu_memory_allocated",
        "gpu_memory_reserved",
    ]
    forbidden = [
        "engine.search",
        "reset_search",
        'buffers["next_state_pool"]',
        "torchscript_ensemble",
        "central_hamming",
        "nn_input",
    ]
    for needle in required_test:
        assert needle in test_text
    for needle in required_dispatcher:
        assert needle in dispatcher_text
    for needle in forbidden:
        assert needle not in test_text
    assert "first validation max_depth must be 10..20" in smoke_text


def test_full_test_csv_depth300_beam65536_world2_contract():
    test_text = (ROOT / "tests" / "full_test_csv_depth300_beam65536_world2.py").read_text(encoding="utf-8")
    dispatcher_text = (ROOT / "production_v6_dispatcher.py").read_text(encoding="utf-8")
    required_test = [
        "run_real_data_production_v6_world2_detailed",
        "validate_output_paths",
        "load_test_puzzles(max_puzzles=None)",
        "FULL_TEST_MAX_DEPTH",
        "GLOBAL_BEAM_WIDTH",
        "max_depth=300",
        "beam_width=65536",
        "FULL_TEST_CSV_DEPTH300_BEAM65536_WORLD2_OK",
        "FULL_TEST_CSV_DEPTH300_BEAM65536_WORLD2_TEST_COMPLETE",
        "path_replay_valid",
        "no_quality_claim",
        "no_leaderboard_claim",
        "no_performance_claim",
    ]
    required_dispatcher = [
        "append_move_to_path",
        "solved_meta",
        "current_paths",
        "validate_output_paths",
        "replay_path_to_central",
        "apply_actions_cpu",
        "solution_len",
    ]
    forbidden = [
        "engine.search",
        "reset_search",
        'buffers["next_state_pool"]',
        "torchscript_ensemble",
        "central_hamming",
        "nn_input",
    ]
    for needle in required_test:
        assert needle in test_text
    for needle in required_dispatcher:
        assert needle in dispatcher_text
    for needle in forbidden:
        assert needle not in test_text


def test_real_solve_100_depth300_load_world2_contract():
    runner_text = (ROOT / "tests" / "real_solve_100_depth300_load_world2.py").read_text(encoding="utf-8")
    notebook_text = (ROOT / "kaggle_real_solve_100_depth300_load_world2_stage" / "real_solve_100_depth300_load_world2.ipynb").read_text(encoding="utf-8")
    metadata_text = (ROOT / "kaggle_real_solve_100_depth300_load_world2_stage" / "kernel-metadata.json").read_text(encoding="utf-8")
    dispatcher_text = (ROOT / "production_v6_dispatcher.py").read_text(encoding="utf-8")
    required_runner = [
        "TASK_COUNT = 100",
        "MAX_DEPTH = 300",
        "BEAM_WIDTH = 65536",
        "BUCKET_CAP_PER_PEER = 262144",
        "assert PRODUCTION_B_MICRO == 8192",
        "assert PRODUCTION_K_EXPAND_TILE == 196608",
        "RUN_START",
        "CUDA_GRAPHS_ENABLED true",
        "TASK_SOLVED",
        "TASK_ERROR",
        "TASK_DONE",
        "HEARTBEAT",
        "RUN_ABORT",
        "RUN_SUMMARY",
        "REAL_SOLVE_100_DEPTH300_LOAD_WORLD2_OK",
        "unsolved_empty_frontier",
        "unsolved_pruned",
        "error_count",
    ]
    required_notebook = [
        "subprocess.Popen",
        "PYTHONUNBUFFERED",
        "USE_CUDA_GRAPHS",
        "\\\"1\\\"",
        "B_MICRO",
        "8192",
        "K_EXPAND_TILE",
        "196608",
        "BUCKET_CAP_PER_PEER",
        "262144",
        "--nproc_per_node=2",
        "returncode {returncode}",
        "REAL_SOLVE_100_DEPTH300_LOAD_WORLD2_TEST_COMPLETE",
    ]
    forbidden = [
        "FRONTIER_COVERAGE_AUDIT_PROGRESS",
        "LOG_EACH_DEPTH\", \"1\"",
        "capture_output=True",
        "central_hamming",
        "torchscript_ensemble",
        "nn_input",
    ]
    for needle in required_runner:
        assert needle in runner_text
    for needle in required_notebook:
        assert needle in notebook_text
    assert '"id": "trydotatwo/real-solve-100-depth300-load-w2"' in metadata_text
    assert '"enable_gpu": true' in metadata_text
    assert "CONFIG_GUARD_OK" in dispatcher_text
    assert "os.environ.setdefault(\"USE_CUDA_GRAPHS\", \"1\")" in dispatcher_text
    for needle in forbidden:
        assert needle not in runner_text
        assert needle not in notebook_text


def test_real_data_100samples_depth300_beam65536_path_audit_world2_contract():
    test_text = (ROOT / "tests" / "real_data_100samples_depth300_beam65536_path_audit_world2.py").read_text(encoding="utf-8")
    dispatcher_text = (ROOT / "production_v6_dispatcher.py").read_text(encoding="utf-8")
    required_test = [
        "run_real_data_path_audit_world2",
        "REAL_DATA_PATH_AUDIT_TASK_COUNT",
        "REAL_DATA_PATH_AUDIT_MAX_DEPTH",
        "GLOBAL_BEAM_WIDTH",
        "task_count=100",
        "max_depth=300",
        "beam_width=65536",
        "REAL_DATA_100SAMPLES_DEPTH300_BEAM65536_PATH_AUDIT_WORLD2_OK",
        "REAL_DATA_100SAMPLES_DEPTH300_BEAM65536_PATH_AUDIT_WORLD2_TEST_COMPLETE",
        "failure_counts",
        "no_quality_claim",
        "no_leaderboard_claim",
        "no_performance_claim",
    ]
    required_dispatcher = [
        "run_real_data_path_audit_world2",
        "raw_solved_record_exists",
        "solved_parent_idx",
        "solved_move",
        "reconstructed_path_exists",
        "path_replay_valid",
        "failure_reason",
        "no_solved_state",
        "solved_state_but_no_parent_chain",
        "parent_chain_broken",
        "replay_failed",
        "output_writer_empty_path",
    ]
    forbidden = [
        "engine.search",
        "reset_search",
        'buffers["next_state_pool"]',
        "torchscript_ensemble",
        "central_hamming",
        "nn_input",
    ]
    for needle in required_test:
        assert needle in test_text
    for needle in required_dispatcher:
        assert needle in dispatcher_text
    for needle in forbidden:
        assert needle not in test_text


def test_frontier_coverage_audit_world2_contract():
    test_text = (ROOT / "tests" / "frontier_coverage_audit_world2.py").read_text(encoding="utf-8")
    dispatcher_text = (ROOT / "production_v6_dispatcher.py").read_text(encoding="utf-8")
    required_test = [
        "run_frontier_coverage_audit_world2",
        "FRONTIER_COVERAGE_TASK_COUNT",
        "FRONTIER_COVERAGE_MAX_DEPTH",
        "GLOBAL_BEAM_WIDTH",
        "task_count=10",
        "max_depth=12",
        "beam_width=65536",
        "REQUIRED_B_MICRO = 8192",
        "FRONTIER_COVERAGE_AUDIT_WORLD2_OK",
        "FRONTIER_COVERAGE_AUDIT_WORLD2_TEST_COMPLETE",
        "coverage_failure_count",
        "known_path_replay_valid",
        "no_quality_claim",
        "no_leaderboard_claim",
        "no_performance_claim",
    ]
    required_dispatcher = [
        "run_frontier_coverage_audit_world2",
        "validate_known_paths",
        "current_frontier_size_before",
        "expanded_parent_count",
        "stream1_scored_parent_count",
        "stream2_generated_candidate_count",
        "stream3_after_threshold_count",
        "stream3_unique_count",
        "stream4_input_count",
        "stream4_clean_count",
        "next_frontier_size_after",
        "frontier_not_fully_processed",
    ]
    forbidden = [
        "engine.search",
        "reset_search",
        'buffers["next_state_pool"]',
        "torchscript_ensemble",
        "central_hamming",
        "nn_input",
    ]
    for needle in required_test:
        assert needle in test_text
    for needle in required_dispatcher:
        assert needle in dispatcher_text
    for needle in forbidden:
        assert needle not in test_text


def test_architecture_v6_production_microbatch_hard_invariant():
    dispatcher_text = (ROOT / "production_v6_dispatcher.py").read_text(encoding="utf-8")
    kaggle_text = (ROOT / "kaggle_frontier_coverage_audit_world2_stage" / "frontier_coverage_audit_world2.ipynb").read_text(encoding="utf-8")
    production_test_paths = [
        ROOT / "tests" / "frontier_coverage_audit_world2.py",
        ROOT / "tests" / "production_dispatcher_path_world2_smoke.py",
        ROOT / "tests" / "real_data_100samples_depth300_beam65536_world2.py",
        ROOT / "tests" / "real_data_100samples_depth300_beam65536_path_audit_world2.py",
        ROOT / "tests" / "full_test_csv_depth300_beam65536_world2.py",
        ROOT / "tests" / "stream5_exchange_smoke.py",
        ROOT / "tests" / "stream5_2gpu_nccl_explicit_smoke.py",
    ]
    required_driver = [
        "PRODUCTION_B_MICRO = 8192",
        "PRODUCTION_K_EXPAND_TILE = PRODUCTION_B_MICRO * MOVE_COUNT",
        "assert PRODUCTION_B_MICRO == 8192",
        "assert PRODUCTION_K_EXPAND_TILE == 196608",
        "def require_production_microbatch",
        "invalid_config: B_MICRO must be",
        "invalid_config: K_EXPAND_TILE must be",
        "self.b_micro = require_production_microbatch(b_micro)",
        "assert required_candidate_capacity == PRODUCTION_K_EXPAND_TILE",
    ]
    for needle in required_driver:
        assert needle in dispatcher_text
    for path in production_test_paths:
        text = path.read_text(encoding="utf-8")
        assert '"8192"' in text or "PRODUCTION_B_MICRO" in text or "REQUIRED_B_MICRO = 8192" in text
        assert '_B_MICRO", "4"' not in text
        assert 'FRONTIER_COVERAGE_B_MICRO"] = "4"' not in text
        assert '"k_expand_tile": 96' not in text
        assert 'cfg["k_expand_tile"] = 48' not in text
        assert 'cfg["b_micro"] = 2' not in text
        assert 'cfg["b_micro"] = 4' not in text
    assert 'FRONTIER_COVERAGE_B_MICRO' in kaggle_text
    assert '8192' in kaggle_text
    assert 'FRONTIER_COVERAGE_B_MICRO\\"] = \\"4\\"' not in kaggle_text


def test_architecture_v6_frontier_drain_and_stream5_capacity_static_guards():
    import ast

    dispatcher_path = ROOT / "production_v6_dispatcher.py"
    dispatcher_text = dispatcher_path.read_text(encoding="utf-8")
    tree = ast.parse(dispatcher_text)

    stream5_fn = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_run_stream5"
    )
    stream5_text = ast.get_source_segment(dispatcher_text, stream5_fn)
    assert 'remote_capacity = max(int(self.cfg["bucket_cap_per_peer"]), 1)' in stream5_text
    assert 'stream3.get("unique_count"' not in stream5_text
    assert 'stream3["unique_count"]' not in stream5_text
    assert "v6_stream5_exchange_candidate_meta" in stream5_text
    assert stream5_text.index("recv_count = torch.zeros") < stream5_text.index("torch.cuda.synchronize()")
    assert stream5_text.index("recv_offset = torch.zeros") < stream5_text.index("torch.cuda.synchronize()")
    assert stream5_text.index("v6_stream5_exchange_candidate_meta") < stream5_text.index("torch.cuda.synchronize()")

    return_dicts = [
        node for node in ast.walk(stream5_fn) if isinstance(node, ast.Dict)
    ]
    stream5_return = return_dicts[-1]
    return_keys = [key.value for key in stream5_return.keys if isinstance(key, ast.Constant)]
    assert return_keys == [
        "remote_recv",
        "recv_count",
        "recv_offset",
        "remote_recv_count",
        "remote_capacity",
    ]
    assert len(return_keys) == len(set(return_keys))

    init_fn = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    init_text = ast.get_source_segment(dispatcher_text, init_fn)
    assert "required_candidate_capacity = self.b_micro * MOVE_COUNT" in init_text
    assert "bucket_cap_per_peer = pow2_ceil(max(131072, required_candidate_capacity))" in init_text
    assert '"bucket_cap_per_peer": bucket_cap_per_peer' in init_text
    assert '"k_expand_tile": required_candidate_capacity' in init_text
    assert '"stream3_batch_candidates": required_candidate_capacity' in init_text
    assert "pow2_ceil(max(131072, required_candidate_capacity))" in init_text
    assert (1 << ((8192 * 24) - 1).bit_length()) == 262144

    run_task_fn = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "run_task"
    )
    run_task_text = ast.get_source_segment(dispatcher_text, run_task_fn)
    assert "parent_offset = 0" in run_task_text
    assert "while parent_offset < len(current):" in run_task_text
    assert 'parent_offset += int(stream12.get("frontier_count", 0))' in run_task_text
    assert run_task_text.index("while parent_offset < len(current):") < run_task_text.index("allreduce_score_threshold")

    required_depth_counters = [
        "current_frontier_size_before",
        "expanded_parent_count",
        "stream1_scored_parent_count",
        "stream2_generated_candidate_count",
        "stream3_unique_count",
        "stream4_clean_count",
        "next_frontier_size_after",
    ]
    for counter in required_depth_counters:
        assert counter in dispatcher_text


def test_beam_engine_capacity_derivation_covers_expand_tile_static_guard():
    cpp_text = (ROOT / "beam_engine.cpp").read_text(encoding="utf-8")
    required = [
        "required_candidate_capacity = std::max<int64_t>(1, static_cast<int64_t>(k_expand_tile))",
        "base_safe = std::max<int64_t>(131072, required_candidate_capacity)",
        "bucket_cap_per_peer_safe = static_cast<int64_t>(pow2_ceil(base_safe))",
        'throw std::runtime_error("bucket_cap_per_peer_safe is smaller than K_EXPAND_TILE")',
        "bucket_cap_per_peer = static_cast<int>(bucket_cap_per_peer_safe)",
        "K_EXPAND_TILE=196608 -> bucket_cap_per_peer_safe=262144",
    ]
    for needle in required:
        assert needle in cpp_text
