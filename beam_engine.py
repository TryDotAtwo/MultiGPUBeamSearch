"""
beam_engine.py

Python control-plane for the CUDA/C++ multi-GPU beam-search engine.
Data-plane remains GPU-resident: inference scores, candidate routing, dedup,
thresholding, pruning and compaction are executed inside the extension.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

import data_loader

PROJECT_DIR = Path(__file__).resolve().parent


def history_backend() -> str:
    value = os.environ.get("HISTORY_BACKEND", "gpu").strip().lower()
    if value not in {"gpu", "cpu"}:
        raise ValueError(f"HISTORY_BACKEND must be gpu or cpu, got {value!r}")
    return value


def _flag_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"", "0", "false", "no", "off"}


def _pow2_ceil(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _auto_stream2_params(global_beam_width: int, world_size: int, b_micro: int, fanout: int) -> Dict[str, int]:
    target_rounds = max(1, int(os.environ.get("TARGET_STREAM2_ROUNDS", "16")))
    n_local = (global_beam_width + world_size - 1) // world_size
    k_expand_tile = _pow2_ceil((global_beam_width * fanout + world_size * target_rounds - 1) // (world_size * target_rounds))
    score_ring_depth = _pow2_ceil((k_expand_tile + b_micro * fanout - 1) // (b_micro * fanout))
    bucket_cap_fast = _pow2_ceil((k_expand_tile + world_size - 1) // world_size)
    bucket_cap_safe = _pow2_ceil((k_expand_tile * 3 + 3) // 4)
    bucket_mode = os.environ.get("AUTO_BUCKET_CAP_MODE", "safe").strip().lower()
    bucket_cap = bucket_cap_safe if bucket_mode == "safe" else bucket_cap_fast
    return {
        "n_local": n_local,
        "k_expand_tile": k_expand_tile,
        "score_ring_depth": max(1, score_ring_depth),
        "bucket_cap_per_peer": max(4096, bucket_cap),
    }


def _env_int_or_auto(name: str, default: str, auto_value: int) -> int:
    raw = os.environ.get(name, default).strip().lower()
    if raw in {"", "auto"}:
        return int(auto_value)
    return int(raw)


def make_default_config() -> Dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    b_micro = int(os.environ.get("B_MICRO", "32768"))
    global_beam_width = int(os.environ.get("GLOBAL_BEAM_WIDTH", str(1 << 16)))
    fanout = 24
    auto = _auto_stream2_params(global_beam_width, max(world_size, 1), b_micro, fanout)
    return {
        "world_size": world_size,
        "rank": rank,
        "global_beam_width": global_beam_width,
        "fanout": fanout,
        "state_size_bytes": 120,
        "b_micro": b_micro,
        "score_ring_depth": _env_int_or_auto("SCORE_RING_DEPTH", "auto", auto["score_ring_depth"]),
        "net_ring_depth": int(os.environ.get("NET_RING_DEPTH", "2")),
        "probe_limit": int(os.environ.get("PROBE_LIMIT", "64")),
        "bucket_cap_per_peer": _env_int_or_auto("BUCKET_CAP_PER_PEER", "auto", auto["bucket_cap_per_peer"]),
        "inference_parallelism": int(os.environ.get("INFERENCE_PARALLELISM", "1")),
        "k_expand_tile": _env_int_or_auto("K_EXPAND_TILE", "auto", auto["k_expand_tile"]),
        "torchscript_scorer_paths": os.environ.get("TORCHSCRIPT_SCORER_PATHS", ""),
        "fullbeamnice_dir": os.environ.get("FULLBEAMNICE_DIR", str(PROJECT_DIR / "FullBeamNice")),
        "nn_score_scale": float(os.environ.get("NN_SCORE_SCALE", "1.0")),
        "nn_score_bias": float(os.environ.get("NN_SCORE_BIAS", "0.0")),
        "gamma": float(os.environ.get("GAMMA", "1.05")),
        "beta": float(os.environ.get("BETA", "1.15")),
        "hash_load_factor": float(os.environ.get("HASH_LOAD_FACTOR", "0.55")),
        "inference_backend": os.environ.get("INFERENCE_BACKEND", "fullbeamnice_static"),
        "max_depth": int(os.environ.get("MAX_DEPTH", "4")),
        "histogram_period_micro": int(os.environ.get("HISTOGRAM_PERIOD_MICRO", "4")),
        "history_backend": history_backend(),
        "cpu_history_checkpoint": _flag_enabled("CPU_HISTORY_CHECKPOINT"),
        "stream3_batch_candidates": int(os.environ.get("STREAM3_BATCH_CANDIDATES", "0")),
        "stream4_batch_candidates": int(os.environ.get("STREAM4_BATCH_CANDIDATES", "0")),
        "stream4_batch_candidates_per_shard_unit": int(os.environ.get("STREAM4_BATCH_CANDIDATES_PER_SHARD_UNIT", "0")),
        "ring_count": int(os.environ.get("RING_COUNT", "2")),
        "shard_count": int(os.environ.get("SHARD_COUNT", "1")),
        "global_spill_capacity": int(os.environ.get("GLOBAL_SPILL_CAPACITY", "0")),
        "solved_result_capacity": int(os.environ.get("SOLVED_RESULT_CAPACITY", "256")),
        "global_beam_width_max_safe": int(os.environ.get("GLOBAL_BEAM_WIDTH_MAX_SAFE", "0")),
        "global_threshold_update_period_shards": int(os.environ.get("GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS", "16")),
    }


def build_extension(verbose: bool = True):
    sources = [
        str(PROJECT_DIR / "beam_engine.cpp"),
        str(PROJECT_DIR / "beam_kernels.cu"),
        str(PROJECT_DIR / "beam_config.cpp"),
        str(PROJECT_DIR / "beam_memory.cpp"),
        str(PROJECT_DIR / "beam_kernels_stream2.cu"),
        str(PROJECT_DIR / "beam_kernels_final.cu"),
        str(PROJECT_DIR / "beam_kernels_stream3.cu"),
        str(PROJECT_DIR / "beam_kernels_stream4.cu"),
        str(PROJECT_DIR / "beam_dispatcher.cpp"),
    ]
    hist_cpu = history_backend() == "cpu"
    checkpoint_on = _flag_enabled("CPU_HISTORY_CHECKPOINT")
    debug_on = _flag_enabled("BEAM_DEBUG") or _flag_enabled("ENGINE_DEBUG")
    variant = f"h{'cpu' if hist_cpu else 'gpu'}_c{int(checkpoint_on)}_d{int(debug_on)}"
    macros = [
        f"-DBEAM_HISTORY_CPU={1 if hist_cpu else 0}",
        f"-DBEAM_CHECKPOINT_ON={1 if checkpoint_on else 0}",
        f"-DBEAM_DEBUG_ON={1 if debug_on else 0}",
    ]
    extra_cflags = ["-O3", "-std=c++17", *macros]
    extra_cuda_cflags = ["-O3", "--use_fast_math", "-lineinfo", "-std=c++17", *macros]
    cutlass_include = PROJECT_DIR / "third_party" / "cutlass" / "include"
    cute_include = PROJECT_DIR / "third_party" / "cutlass" / "tools" / "util" / "include"
    extra_include_paths = []
    if cutlass_include.exists():
        extra_include_paths.append(str(cutlass_include))
    if cute_include.exists():
        extra_include_paths.append(str(cute_include))
    extra_ldflags = ["-lnccl"]
    if os.name != "nt" and os.path.exists("/kaggle/working") and "TMPDIR" not in os.environ:
        os.environ["TMPDIR"] = "/kaggle/working/.tmp"
        os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    if "TORCH_EXTENSIONS_DIR" not in os.environ:
        os.environ["TORCH_EXTENSIONS_DIR"] = str(PROJECT_DIR / "runtime" / "torch_extensions")
        
    return load(
        name=f"beam_engine_ext_{variant}",
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        verbose=verbose,
    )


def init_distributed_if_needed(cfg: Dict[str, Any]) -> None:
    if cfg["world_size"] <= 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")) % torch.cuda.device_count())
        return
    local_rank = int(os.environ.get("LOCAL_RANK", str(cfg["rank"] % torch.cuda.device_count())))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=cfg["rank"], world_size=cfg["world_size"])


def allocate_buffers(ext, cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    sizes = ext.derive_sizes(cfg)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", str(cfg["rank"] % torch.cuda.device_count()))))

    n_local = int(sizes["n_local"])
    k_work = int(sizes["k_work"])
    hash_capacity = int(sizes["hash_capacity"])
    score_ring_elements = int(sizes["score_ring_elements"])
    candidate_record_bytes = int(sizes["candidate_record_bytes"])
    beam_meta_bytes = int(sizes["beam_meta_bytes"])
    hash_slot_bytes = int(sizes["hash_slot_bytes"])
    send_recv_records = int(sizes["send_recv_records"])
    state_size = int(cfg["state_size_bytes"])
    history_records = int(sizes["history_records"])
    state128_bytes = int(sizes.get("state128_bytes", 128))
    candidate_meta_bytes = int(sizes.get("candidate_meta_bytes", 32))
    scratch_pool_bytes = int(sizes.get("scratch_pool_bytes", 0))
    solved_result_capacity = int(sizes.get("solved_result_capacity", cfg.get("solved_result_capacity", 256)))

    buffers = {
        "beam_current": torch.empty((n_local, state_size), dtype=torch.uint8, device=device),
        "current_active_flags": torch.empty((n_local,), dtype=torch.uint8, device=device),
        "next_state_pool": torch.empty((k_work, state_size), dtype=torch.uint8, device=device),
        "next_meta": torch.empty((k_work * beam_meta_bytes,), dtype=torch.uint8, device=device),
        "hash_table": torch.empty((hash_capacity * hash_slot_bytes,), dtype=torch.uint8, device=device),
        "active_flags": torch.empty((k_work,), dtype=torch.uint8, device=device),
        "free_indices": torch.empty((k_work,), dtype=torch.int32, device=device),
        "free_count": torch.empty((1,), dtype=torch.int32, device=device),
        "score_ring": torch.empty((score_ring_elements,), dtype=torch.int16, device=device),
        "send_buckets": torch.empty((send_recv_records * candidate_record_bytes,), dtype=torch.uint8, device=device),
        "recv_buckets": torch.empty((send_recv_records * candidate_record_bytes,), dtype=torch.uint8, device=device),
        "send_counts": torch.empty((cfg["net_ring_depth"] * cfg["world_size"],), dtype=torch.int32, device=device),
        "recv_counts": torch.empty((cfg["net_ring_depth"] * cfg["world_size"],), dtype=torch.int32, device=device),
        "local_hist": torch.empty((65536,), dtype=torch.int32, device=device),
        "global_hist": torch.empty((65536,), dtype=torch.int32, device=device),
        "threshold_cell": torch.empty((2,), dtype=torch.int32, device=device),
        "counters": torch.empty((8,), dtype=torch.int32, device=device),
        "beam_status": torch.empty((8,), dtype=torch.int32, device=device),
        "history_parent_idx": torch.empty((history_records,), dtype=torch.int32, device=device),
        "history_parent_rank": torch.empty((history_records,), dtype=torch.uint8, device=device),
        "history_action": torch.empty((history_records,), dtype=torch.uint8, device=device),
        "history_valid": torch.empty((history_records,), dtype=torch.uint8, device=device),
        "history_depth_cell": torch.empty((1,), dtype=torch.int32, device=device),
        "current_frontier_states": torch.empty((n_local * state128_bytes,), dtype=torch.uint8, device=device),
        "scratch_pool": torch.empty((scratch_pool_bytes,), dtype=torch.uint8, device=device),
        "solved_flag": torch.empty((1,), dtype=torch.int32, device=device),
        "stop_flag": torch.empty((1,), dtype=torch.int32, device=device),
        "solved_count": torch.empty((1,), dtype=torch.int32, device=device),
        "solved_overflow": torch.empty((1,), dtype=torch.int32, device=device),
        "solved_meta_list": torch.empty((solved_result_capacity * candidate_meta_bytes,), dtype=torch.uint8, device=device),
        "solved_depth_list": torch.empty((solved_result_capacity,), dtype=torch.int32, device=device),
    }
    if str(cfg.get("inference_backend", "")).strip().lower() == "fullbeamnice_static":
        b_micro = int(cfg["b_micro"])
        lanes = int(cfg["inference_parallelism"])
        buffers.update({
            "fb_act1": torch.empty((lanes, b_micro, 1536), dtype=torch.float16, device=device),
            "fb_act2": torch.empty((lanes, b_micro, 512), dtype=torch.float16, device=device),
            "fb_act3": torch.empty((lanes, b_micro, 512), dtype=torch.float16, device=device),
            "fb_out": torch.empty((lanes, b_micro, 24), dtype=torch.float16, device=device),
        })
    for tensor in buffers.values():
        tensor.zero_()
    return buffers


def _v6_pack_meta(lo: int, hi: int, parent_idx: int, score_key: int, route: int) -> bytes:
    return struct.pack("<QQQII", lo, hi, parent_idx, score_key, route)


def _v6_unpack_meta(raw: bytes, idx: int) -> Dict[str, int]:
    lo, hi, parent_idx, score_key, route = struct.unpack_from("<QQQII", raw, idx * 32)
    return {
        "lo": lo,
        "hi": hi,
        "parent_idx": parent_idx,
        "score_key": score_key,
        "route": route,
        "source_rank": route >> 16,
        "owner": (route >> 8) & 0xFF,
        "move": route & 0xFF,
    }


def _v6_pack_final_request(parent_idx: int, target_local_idx: int, return_rank: int, move: int) -> bytes:
    return struct.pack("<QIHBB", parent_idx, target_local_idx, return_rank, move, 0)


def _v6_make_identity_generators() -> np.ndarray:
    generators = np.zeros((24, 128), dtype=np.uint8)
    for move in range(24):
        generators[move] = np.arange(128, dtype=np.uint8)
    return generators


def _v6_make_zobrist() -> np.ndarray:
    zobrist = np.zeros((128, 128, 16), dtype=np.uint8)
    for pos in range(120):
        for value in range(128):
            lo = (0x9E3779B97F4A7C15 * (pos + 1) + 0xD1B54A32D192ED03 * (value + 1)) & 0xFFFFFFFFFFFFFFFF
            hi = (0x94D049BB133111EB * (pos + 3) + 0x2545F4914F6CDD1D * (value + 5)) & 0xFFFFFFFFFFFFFFFF
            zobrist[pos, value] = np.frombuffer(struct.pack("<QQ", lo, hi), dtype=np.uint8)
    return zobrist.reshape(-1)


def v6_dispatcher_skeleton_single_gpu_smoke(stop_path: bool = False, verbose: bool = False) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v6 dispatcher skeleton smoke")
    os.environ.setdefault("INFERENCE_BACKEND", "fullbeamnice_static")
    ext = build_extension(verbose=verbose)
    contract = ext.v6_dispatcher_skeleton_single_gpu_smoke()
    if int(contract["world_size"]) != 1:
        raise RuntimeError("v6 dispatcher skeleton contract must stay WORLD_SIZE=1")

    device = torch.device("cuda", 0)
    cfg = make_default_config()
    cfg.update({
        "world_size": 1,
        "rank": 0,
        "global_beam_width": 64,
        "b_micro": 2,
        "score_ring_depth": 1,
        "net_ring_depth": 1,
        "bucket_cap_per_peer": 8,
        "k_expand_tile": 48,
        "inference_parallelism": 1,
        "max_depth": 1,
        "inference_backend": "fullbeamnice_static",
        "stream3_batch_candidates": 48,
        "stream4_batch_candidates": 2,
        "stream4_batch_candidates_per_shard_unit": 2,
        "shard_count": 1,
    })
    buffers = allocate_buffers(ext, cfg)
    engine = ext.BeamEngine(cfg, buffers, "fullbeamnice_static")

    b_micro = 2
    move_count = 24
    stream3_batch_candidates = b_micro * move_count
    current_threshold = 100
    parent_base = torch.tensor([0], dtype=torch.int64, device=device)
    count = torch.tensor([b_micro], dtype=torch.int32, device=device)
    score = np.full((stream3_batch_candidates,), 1000, dtype=np.int32)
    score[0] = 10
    score[1] = 20
    score[24] = 5
    score_ring = torch.tensor(score, dtype=torch.int32, device=device)

    states = np.zeros((b_micro, 128), dtype=np.uint8)
    states[0, :120] = np.arange(120, dtype=np.uint8) % 128
    states[1, :120] = (np.arange(120, dtype=np.uint8) + 7) % 128
    current_frontier_states = torch.tensor(states.reshape(-1), dtype=torch.uint8, device=device)
    generators_np = _v6_make_identity_generators()
    generators = torch.tensor(generators_np.reshape(-1), dtype=torch.uint8, device=device)
    central_np = np.zeros((128,), dtype=np.uint8)
    central_np[:120] = states[0, :120] if stop_path else 127
    central = torch.tensor(central_np, dtype=torch.uint8, device=device)
    zobrist = torch.tensor(_v6_make_zobrist(), dtype=torch.uint8, device=device)

    hash_ring = torch.zeros((stream3_batch_candidates * 16,), dtype=torch.uint8, device=device)
    solved_flag = torch.zeros((1,), dtype=torch.int32, device=device)
    stop_flag = torch.zeros((1,), dtype=torch.int32, device=device)
    solved_count = torch.zeros((1,), dtype=torch.int32, device=device)
    solved_overflow = torch.zeros((1,), dtype=torch.int32, device=device)
    solved_meta_list = torch.zeros((8 * 32,), dtype=torch.uint8, device=device)
    solved_depth_list = torch.zeros((8,), dtype=torch.int32, device=device)

    ext.v6_stream2_hash_goal(
        current_frontier_states,
        parent_base,
        count,
        score_ring,
        hash_ring,
        generators,
        central,
        zobrist,
        solved_flag,
        stop_flag,
        solved_count,
        solved_overflow,
        solved_meta_list,
        solved_depth_list,
        8,
        0,
        0,
        0,
        0,
        1,
        b_micro,
    )
    torch.cuda.synchronize()
    if int(solved_flag.cpu()[0]) != 0:
        solved_raw = solved_meta_list.cpu().numpy().tobytes()
        first_meta = _v6_unpack_meta(solved_raw, 0) if int(solved_count.cpu()[0]) > 0 else {}
        return {
            "path": "stop",
            "stream2_hash_written": bool(hash_ring[:16].any().item()),
            "solved_flag": int(solved_flag.cpu()[0]),
            "stop_flag": int(stop_flag.cpu()[0]),
            "solved_count": int(solved_count.cpu()[0]),
            "solved_overflow": int(solved_overflow.cpu()[0]),
            "first_solved_score_key": int(first_meta.get("score_key", -1)),
            "stream3_launched": False,
            "stream4_launched": False,
            "final_launched": False,
            "solved_list_copied_to_cpu": True,
            "stream1_production_called": False,
            "fallback_backend_called": False,
        }

    stream3_key_a = torch.zeros((stream3_batch_candidates * 16,), dtype=torch.uint8, device=device)
    stream3_key_b = torch.zeros_like(stream3_key_a)
    stream3_val_a = torch.zeros((stream3_batch_candidates,), dtype=torch.int64, device=device)
    stream3_val_b = torch.zeros_like(stream3_val_a)
    compact_count = torch.zeros((1,), dtype=torch.int32, device=device)
    ext.v6_stream3_pack_threshold_compact(
        score_ring, hash_ring, parent_base, count, stream3_key_a, stream3_val_a, compact_count,
        current_threshold, 0, 1, b_micro, stream3_batch_candidates,
    )
    torch.cuda.synchronize()
    compact_n = int(compact_count.cpu()[0])
    temp_storage = torch.empty((int(ext.v6_stream3_sort_temp_bytes(compact_n)),), dtype=torch.uint8, device=device)
    ext.v6_stream3_sort_pairs(temp_storage, stream3_key_a, stream3_key_b, stream3_val_a, stream3_val_b, compact_n)
    unique_key = torch.zeros((stream3_batch_candidates * 16,), dtype=torch.uint8, device=device)
    unique_val = torch.zeros((stream3_batch_candidates,), dtype=torch.int64, device=device)
    unique_count = torch.zeros((1,), dtype=torch.int32, device=device)
    ext.v6_stream3_dedup_sorted(stream3_key_b, stream3_val_b, unique_key, unique_val, unique_count, compact_n)
    torch.cuda.synchronize()
    unique_n = int(unique_count.cpu()[0])

    local_pending_buffer = torch.zeros((stream3_batch_candidates * 32,), dtype=torch.uint8, device=device)
    remote_send_buffer = torch.zeros_like(local_pending_buffer)
    local_count = torch.zeros((1,), dtype=torch.int32, device=device)
    send_count = torch.zeros((1,), dtype=torch.int32, device=device)
    send_offset = torch.zeros((2,), dtype=torch.int32, device=device)
    ext.v6_stream3_restore_split(
        unique_key, unique_val, parent_base, local_pending_buffer, remote_send_buffer,
        local_count, send_count, send_offset, unique_n, 0, 1, 0, 1, b_micro,
    )
    torch.cuda.synchronize()
    local_n = int(local_count.cpu()[0])

    remote_recv_buffer = torch.zeros_like(local_pending_buffer)
    recv_count = torch.empty((1,), dtype=torch.int32, device=device)
    recv_offset = torch.empty((2,), dtype=torch.int32, device=device)
    local_bytes = local_pending_buffer[:local_n * 32].clone()
    send_count_for_copy = torch.tensor([local_n], dtype=torch.int32, device=device)
    send_offset_for_copy = torch.tensor([0, local_n], dtype=torch.int32, device=device)
    engine.v6_stream5_exchange_candidate_meta(
        local_pending_buffer, remote_recv_buffer,
        send_count_for_copy, send_offset_for_copy,
        recv_count, recv_offset,
    )
    torch.cuda.synchronize()
    stream5_byte_identical = bool(torch.equal(local_bytes, remote_recv_buffer[:local_n * 32]))

    survivor_shard = torch.zeros((max(2 * local_n, 1) * 32,), dtype=torch.uint8, device=device)
    if local_n > 0:
        survivor_shard[:local_n * 32].copy_(local_pending_buffer[:local_n * 32])
    dirty_count = torch.tensor([local_n], dtype=torch.int32, device=device)
    clean_count = torch.tensor([0], dtype=torch.int32, device=device)
    processing_flag = torch.tensor([0], dtype=torch.uint8, device=device)

    input_count = local_n
    stream4_key_a = torch.zeros((max(input_count, 1) * 16,), dtype=torch.uint8, device=device)
    stream4_key_b = torch.zeros_like(stream4_key_a)
    stream4_val_a = torch.zeros((max(input_count, 1) * 32,), dtype=torch.uint8, device=device)
    stream4_val_b = torch.zeros_like(stream4_val_a)
    stream4_compact_count = torch.zeros((1,), dtype=torch.int32, device=device)
    ext.v6_stream4_threshold_compact(survivor_shard, stream4_key_a, stream4_val_a, stream4_compact_count, input_count, current_threshold)
    torch.cuda.synchronize()
    stream4_compact_n = int(stream4_compact_count.cpu()[0])
    stream4_temp = torch.empty((int(ext.v6_stream4_sort_temp_bytes(stream4_compact_n)),), dtype=torch.uint8, device=device)
    ext.v6_stream4_sort_pairs(stream4_temp, stream4_key_a, stream4_key_b, stream4_val_a, stream4_val_b, stream4_compact_n)
    clean_tmp = torch.zeros((max(input_count, 1) * 32,), dtype=torch.uint8, device=device)
    new_clean_count = torch.zeros((1,), dtype=torch.int32, device=device)
    ext.v6_stream4_dedup_sorted(stream4_key_b, stream4_val_b, clean_tmp, new_clean_count, stream4_compact_n)
    torch.cuda.synchronize()
    clean_n = int(new_clean_count.cpu()[0])
    ext.v6_stream4_write_clean(survivor_shard, clean_tmp, clean_count, dirty_count, processing_flag, clean_n)
    torch.cuda.synchronize()

    clean_raw = survivor_shard.cpu().numpy().tobytes()
    final_requests = []
    clean_metas = []
    for i in range(clean_n):
        meta = _v6_unpack_meta(clean_raw, i)
        clean_metas.append(meta)
        final_requests.append(_v6_pack_final_request(meta["parent_idx"], i, 0, meta["move"]))
    final_request_buffer = torch.tensor(np.frombuffer(b"".join(final_requests), dtype=np.uint8).copy(), dtype=torch.uint8, device=device)
    final_response_buffer = torch.zeros((max(clean_n, 1) * 128,), dtype=torch.uint8, device=device)
    next_frontier_states_tmp = torch.zeros((max(clean_n, 1) * 128,), dtype=torch.uint8, device=device)
    ext.v6_final_materialize(current_frontier_states, final_request_buffer, generators, final_response_buffer, clean_n)
    torch.cuda.synchronize()
    response_raw_before_scatter = final_response_buffer.cpu().numpy().tobytes()
    ext.v6_final_scatter_responses(final_response_buffer, next_frontier_states_tmp, clean_n)
    torch.cuda.synchronize()
    current_frontier_states[:clean_n * 128].copy_(next_frontier_states_tmp[:clean_n * 128])
    torch.cuda.synchronize()
    next_raw = next_frontier_states_tmp.cpu().numpy().tobytes()
    current_raw = current_frontier_states.cpu().numpy().tobytes()

    return {
        "path": "normal",
        "stream2_hash_written": bool(hash_ring[:16].any().item()),
        "stream3_launched": True,
        "stream4_launched": True,
        "final_launched": True,
        "compact_count": compact_n,
        "unique_count": unique_n,
        "local_count": local_n,
        "stream5_byte_identical": stream5_byte_identical,
        "collector_dirty_count_initial": local_n,
        "clean_count": int(clean_count.cpu()[0]),
        "dirty_count": int(dirty_count.cpu()[0]),
        "processing_flag": int(processing_flag.cpu()[0]),
        "final_count": clean_n,
        "final_response_target0": struct.unpack_from("<I", response_raw_before_scatter, 120)[0] if clean_n else -1,
        "next_padding_zero": all(b == 0 for b in next_raw[120:128]) if clean_n else True,
        "current_frontier_updated": current_raw[:clean_n * 128] == next_raw[:clean_n * 128],
        "first_clean_score_key": clean_metas[0]["score_key"] if clean_metas else -1,
        "first_clean_move": clean_metas[0]["move"] if clean_metas else -1,
        "stream1_production_called": False,
        "fallback_backend_called": False,
    }


def create_nccl_id(ext, cfg: Dict[str, Any]) -> bytes:
    if cfg["world_size"] <= 1:
        return b""
    obj = [None]
    if cfg["rank"] == 0:
        obj[0] = bytes(ext.get_nccl_unique_id())
    dist.broadcast_object_list(obj, src=0)
    return obj[0]


def configure_engine(ext, cfg: Dict[str, Any], buffers: Dict[str, torch.Tensor]):
    backend = str(cfg.get("inference_backend", "")).strip().lower()
    allow_ts = os.environ.get("ALLOW_TORCHSCRIPT_SCORER", "").strip().lower() in {"1", "true", "yes", "on"}
    if backend != "fullbeamnice_static":
        raise ValueError("Target architecture v6 production path requires INFERENCE_BACKEND=fullbeamnice_static")
    if backend == "torchscript_ensemble" and not allow_ts:
        raise ValueError(
            "INFERENCE_BACKEND=torchscript_ensemble is disabled by default (no accidental TorchScript hot path). "
            "Use INFERENCE_BACKEND=fullbeamnice_static for CUTLASS static scorer, or set ALLOW_TORCHSCRIPT_SCORER=1 to load TorchScript."
        )
    engine = ext.BeamEngine(cfg, buffers, cfg["inference_backend"])
    engine.set_action_permutation_table(data_loader.get_action_table_u8())
    engine.set_central_state(data_loader.get_central_state_u8().tobytes())
    if backend == "torchscript_ensemble":
        paths = [p for p in str(cfg.get("torchscript_scorer_paths", "")).split(os.pathsep) if p]
        if not paths:
            raise ValueError("INFERENCE_BACKEND=torchscript_ensemble requires TORCHSCRIPT_SCORER_PATHS")
        if len(paths) > 1:
            print(f"[beam_engine] warning: {len(paths)} TorchScript paths were provided; one path is enough for shared-weight multi-lane inference")
        engine.load_torchscript_ensemble(paths)
    if backend == "fullbeamnice_static":
        from scripts.static_fullbeamnice_inference import load_static_weights

        weights = load_static_weights(Path(cfg["fullbeamnice_dir"]), device=buffers["beam_current"].device, dtype=torch.float16)
        engine.load_fullbeamnice_static({
            "embed_w_t": weights.embed_w_t,
            "embed_bias": weights.embed_bias,
            "hidden_w_t": weights.hidden_w_t,
            "hidden_bias": weights.hidden_bias,
            "res0_fc1_w_t": weights.res0_fc1_w_t,
            "res0_fc1_bias": weights.res0_fc1_bias,
            "res0_fc2_w_t": weights.res0_fc2_w_t,
            "res0_fc2_bias": weights.res0_fc2_bias,
            "res1_fc1_w_t": weights.res1_fc1_w_t,
            "res1_fc1_bias": weights.res1_fc1_bias,
            "res1_fc2_w_t": weights.res1_fc2_w_t,
            "res1_fc2_bias": weights.res1_fc2_bias,
            "out_w_t": weights.out_w_t,
            "out_bias": weights.out_bias,
            "action_perm": weights.action_perm.to(device=buffers["beam_current"].device, dtype=torch.int32),
            "score_scale": float(weights.score_scale),
            "score_bias": float(weights.score_bias),
            "state_size": int(weights.state_size),
            "num_classes": int(weights.num_classes),
        })
    if cfg["world_size"] > 1:
        engine.init_nccl(create_nccl_id(ext, cfg))
    engine.enable_cuda_graphs(os.environ.get("USE_CUDA_GRAPHS", "1") != "0")
    if os.environ.get("ENGINE_DEBUG", "0") != "0":
        engine.enable_debug(True, True, int(os.environ.get("ENGINE_LOG_PERIOD", "1")))
    return engine


def load_puzzle_data() -> Tuple[np.ndarray, Dict[str, np.ndarray], list]:
    return data_loader.get_central_state(), data_loader.get_generators(), data_loader.load_test_puzzles()


def print_puzzle_summary() -> None:
    central_state, generators, test_puzzles = load_puzzle_data()
    print("=" * 60)
    print("Puzzle Data Loaded Successfully")
    print("=" * 60)
    print(f"Central state size: {len(central_state)}")
    print(f"Central state first 10: {central_state[:10].tolist()}")
    print(f"Number of generators: {len(generators)}")
    print(f"Generator names: {sorted(generators.keys())}")
    print(f"Total test puzzles: {len(test_puzzles)}")
    if test_puzzles:
        puzzle_id, state = test_puzzles[0]
        print(f"First test puzzle ID: {puzzle_id}")
        print(f"First test puzzle state first 10: {state[:10].tolist()}")
    print("=" * 60)


def make_shallow_scramble(actions: list[str]) -> np.ndarray:
    state = data_loader.get_central_state_u8()
    for action in actions:
        state = data_loader.apply_action_cpu(state, action)
    return state.astype(np.uint8)


def run_one_search(engine, cfg: Dict[str, Any], initial_state: np.ndarray, max_depth: int, active_owner: bool = True) -> Dict[str, Any]:
    initial_u8 = np.asarray(initial_state, dtype=np.uint8)
    engine.reset_search(initial_u8.tobytes(), bool(active_owner))
    result = engine.search(int(max_depth), int(cfg["histogram_period_micro"]))
    return json.loads(json.dumps(result, default=lambda x: x))


def main() -> None:
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    if os.path.exists("/kaggle/working"):
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    cfg = make_default_config()
    init_distributed_if_needed(cfg)
    print_puzzle_summary()

    ext = build_extension(verbose=os.environ.get("BUILD_VERBOSE", "1") != "0")
    buffers = allocate_buffers(ext, cfg)
    engine = configure_engine(ext, cfg, buffers)

    # Deterministic correctness case: scramble by U, solve by -U within depth 1.
    state = make_shallow_scramble(["U"])
    # Rank ownership: for the initial frontier only the owner rank is active.
    owner = data_loader.owner_rank_for_state(state, cfg["world_size"])
    result = run_one_search(engine, cfg, state, max_depth=max(1, cfg["max_depth"]), active_owner=(cfg["rank"] == owner))
    torch.cuda.synchronize()

    if cfg["world_size"] > 1:
        # Ensure every rank reaches here before process-group teardown.
        dist.barrier()
    print(json.dumps({"rank": cfg["rank"], "owner": owner, "result": result, "sizes": dict(engine.sizes()), "status": dict(engine.status())}, indent=2))

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
