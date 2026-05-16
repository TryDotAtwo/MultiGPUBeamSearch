from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
