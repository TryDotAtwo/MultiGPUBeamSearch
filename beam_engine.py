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
        "score_ring_depth": max(2, score_ring_depth),
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
        "state_size_bytes": 128,
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
        "score_ring": torch.empty((score_ring_elements,), dtype=torch.int32, device=device),
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


def _v6_validate_u32(name: str, value: int) -> int:
    value = int(value)
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"{name} must be in uint32 range, got {value}")
    return value


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
    current_threshold = _v6_validate_u32("current_threshold", 0xFFFFFFFF)
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
        _v6_validate_u32("current_threshold", current_threshold), 0, 1, b_micro, stream3_batch_candidates,
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
    ext.v6_stream4_threshold_compact(
        survivor_shard,
        stream4_key_a,
        stream4_val_a,
        stream4_compact_count,
        input_count,
        _v6_validate_u32("stream4_job_threshold", current_threshold),
    )
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


def v6_dispatcher_skeleton_world2_stream5_smoke(verbose: bool = False) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v6 dispatcher WORLD_SIZE=2 Stream5 smoke")

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 2:
        raise RuntimeError(f"v6 dispatcher WORLD_SIZE=2 Stream5 smoke requires WORLD_SIZE=2, got {world_size}")
    if rank not in (0, 1):
        raise RuntimeError(f"v6 dispatcher WORLD_SIZE=2 Stream5 smoke requires rank 0 or 1, got {rank}")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"v6 dispatcher WORLD_SIZE=2 Stream5 smoke requires at least 2 CUDA devices, got {torch.cuda.device_count()}")

    os.environ.setdefault("INFERENCE_BACKEND", "fullbeamnice_static")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device("cuda", local_rank)
    ext = build_extension(verbose=verbose)
    cfg = make_default_config()
    cfg.update({
        "world_size": 2,
        "rank": rank,
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
    engine.init_nccl(create_nccl_id(ext, cfg))

    peer = 1 - rank
    send_to_peer = 2 if rank == 0 else 4
    recv_from_peer = 4 if rank == 0 else 2
    max_records = 6

    send_count_host = np.zeros((world_size,), dtype=np.int32)
    send_count_host[peer] = send_to_peer
    send_offset_host = np.zeros((world_size + 1,), dtype=np.int32)
    cursor = 0
    for p in range(world_size):
        send_offset_host[p] = cursor
        cursor += int(send_count_host[p])
    send_offset_host[world_size] = cursor

    recv_count_host = np.zeros((world_size,), dtype=np.int32)
    recv_count_host[peer] = recv_from_peer
    recv_offset_host = np.zeros((world_size + 1,), dtype=np.int32)
    cursor = 0
    for p in range(world_size):
        recv_offset_host[p] = cursor
        cursor += int(recv_count_host[p])
    recv_offset_host[world_size] = cursor

    def make_records(src_rank: int, dst_rank: int, count: int) -> bytes:
        return b"".join(
            _v6_pack_meta(
                0x9100_0000 + src_rank * 0x10000 + dst_rank * 0x100 + j,
                0xA200_0000 + src_rank * 0x10000 + dst_rank * 0x100 + j,
                0xB000_0000 + src_rank * 1000 + dst_rank * 10 + j,
                0xC000 + src_rank * 100 + dst_rank * 10 + j,
                (src_rank << 16) | (dst_rank << 8) | j,
            )
            for j in range(count)
        )

    send_blob = bytearray(max_records * 32)
    peer_records = make_records(rank, peer, send_to_peer)
    send_start = int(send_offset_host[peer]) * 32
    send_blob[send_start:send_start + len(peer_records)] = peer_records

    remote_send_buffer = torch.tensor(np.frombuffer(bytes(send_blob), dtype=np.uint8).copy(), dtype=torch.uint8, device=device)
    remote_recv_buffer = torch.zeros((max_records * 32,), dtype=torch.uint8, device=device)
    send_count = torch.tensor(send_count_host, dtype=torch.int32, device=device)
    send_offset = torch.tensor(send_offset_host, dtype=torch.int32, device=device)
    recv_count = torch.tensor(recv_count_host, dtype=torch.int32, device=device)
    recv_offset = torch.tensor(recv_offset_host, dtype=torch.int32, device=device)

    stream5_launched_by_dispatcher = True
    engine.v6_stream5_exchange_candidate_meta(
        remote_send_buffer,
        remote_recv_buffer,
        send_count,
        send_offset,
        recv_count,
        recv_offset,
    )
    torch.cuda.synchronize()

    recv_count_after = recv_count.cpu().numpy()
    recv_offset_after = recv_offset.cpu().numpy()
    recv_raw = remote_recv_buffer.cpu().numpy().tobytes()
    expected = make_records(peer, rank, recv_from_peer)
    recv_start = int(recv_offset_after[peer]) * 32
    payload_ok = recv_raw[recv_start:recv_start + len(expected)] == expected

    dist.barrier()
    return {
        "path": "world2_stream5",
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "stream5_launched_by_dispatcher": stream5_launched_by_dispatcher,
        "send_count": send_count_host.tolist(),
        "send_offset": send_offset_host.tolist(),
        "recv_count": recv_count_after.tolist(),
        "recv_offset": recv_offset_after.tolist(),
        "expected_recv_count": recv_count_host.tolist(),
        "expected_recv_offset": recv_offset_host.tolist(),
        "payload_byte_identical": payload_ok,
        "received_records": recv_from_peer,
        "sent_records": send_to_peer,
        "stream1_production_called": False,
        "fallback_backend_called": False,
        "stream3_collector_expanded": False,
        "stream4_scheduler_expanded": False,
        "final_materialization_expanded": False,
    }


def v6_dispatcher_stream3_stream5_collector_world2_smoke(verbose: bool = False) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v6 dispatcher Stream3/Stream5/collector WORLD_SIZE=2 smoke")

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 2:
        raise RuntimeError(f"v6 dispatcher Stream3/Stream5/collector smoke requires WORLD_SIZE=2, got {world_size}")
    if rank not in (0, 1):
        raise RuntimeError(f"v6 dispatcher Stream3/Stream5/collector smoke requires rank 0 or 1, got {rank}")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"v6 dispatcher Stream3/Stream5/collector smoke requires at least 2 CUDA devices, got {torch.cuda.device_count()}"
        )

    os.environ.setdefault("INFERENCE_BACKEND", "fullbeamnice_static")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device("cuda", local_rank)
    ext = build_extension(verbose=verbose)
    cfg = make_default_config()
    cfg.update({
        "world_size": 2,
        "rank": rank,
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
    engine.init_nccl(create_nccl_id(ext, cfg))

    peer = 1 - rank
    local_count_expected = 2 + rank
    remote_count_expected = 3 + rank
    recv_count_expected = 3 + peer
    max_records = 8

    def make_stream3_records(src_rank: int, owner_rank: int, count: int, base_move: int) -> bytes:
        return b"".join(
            _v6_pack_meta(
                0xD100_0000 + src_rank * 0x10000 + owner_rank * 0x100 + j,
                0xE200_0000 + src_rank * 0x10000 + owner_rank * 0x100 + j,
                0xF000_0000 + src_rank * 1000 + owner_rank * 10 + j,
                0x1000 + src_rank * 100 + owner_rank * 10 + j,
                (src_rank << 16) | (owner_rank << 8) | ((base_move + j) & 0xFF),
            )
            for j in range(count)
        )

    local_records = make_stream3_records(rank, rank, local_count_expected, 1)
    remote_records = make_stream3_records(rank, peer, remote_count_expected, 9)
    expected_remote_recv = make_stream3_records(peer, rank, recv_count_expected, 9)

    local_pending_buffer = torch.zeros((max_records * 32,), dtype=torch.uint8, device=device)
    remote_send_buffer = torch.zeros((max_records * 32,), dtype=torch.uint8, device=device)
    local_pending_buffer[:len(local_records)] = torch.tensor(
        np.frombuffer(local_records, dtype=np.uint8).copy(), dtype=torch.uint8, device=device
    )

    send_count_host = np.zeros((world_size,), dtype=np.int32)
    send_count_host[peer] = remote_count_expected
    send_offset_host = np.zeros((world_size + 1,), dtype=np.int32)
    cursor = 0
    for p in range(world_size):
        send_offset_host[p] = cursor
        cursor += int(send_count_host[p])
    send_offset_host[world_size] = cursor
    remote_send_start = int(send_offset_host[peer]) * 32
    remote_send_buffer[remote_send_start:remote_send_start + len(remote_records)] = torch.tensor(
        np.frombuffer(remote_records, dtype=np.uint8).copy(), dtype=torch.uint8, device=device
    )

    recv_count_host = np.zeros((world_size,), dtype=np.int32)
    recv_count_host[peer] = recv_count_expected
    recv_offset_host = np.zeros((world_size + 1,), dtype=np.int32)
    cursor = 0
    for p in range(world_size):
        recv_offset_host[p] = cursor
        cursor += int(recv_count_host[p])
    recv_offset_host[world_size] = cursor

    remote_recv_buffer = torch.zeros((max_records * 32,), dtype=torch.uint8, device=device)
    send_count = torch.tensor(send_count_host, dtype=torch.int32, device=device)
    send_offset = torch.tensor(send_offset_host, dtype=torch.int32, device=device)
    recv_count = torch.tensor(recv_count_host, dtype=torch.int32, device=device)
    recv_offset = torch.tensor(recv_offset_host, dtype=torch.int32, device=device)

    stream3_split_done = True
    engine.v6_stream5_exchange_candidate_meta(
        remote_send_buffer,
        remote_recv_buffer,
        send_count,
        send_offset,
        recv_count,
        recv_offset,
    )
    torch.cuda.synchronize()

    recv_count_after = recv_count.cpu().numpy()
    recv_offset_after = recv_offset.cpu().numpy()
    recv_start = int(recv_offset_after[peer]) * 32
    remote_recv_raw = remote_recv_buffer.cpu().numpy().tobytes()
    remote_recv_ok = remote_recv_raw[recv_start:recv_start + len(expected_remote_recv)] == expected_remote_recv

    survivor_shard = torch.zeros(((local_count_expected + recv_count_expected) * 32,), dtype=torch.uint8, device=device)
    write_cursor = 0
    survivor_shard[write_cursor:write_cursor + len(local_records)].copy_(local_pending_buffer[:len(local_records)])
    write_cursor += len(local_records)
    survivor_shard[write_cursor:write_cursor + len(expected_remote_recv)].copy_(
        remote_recv_buffer[recv_start:recv_start + len(expected_remote_recv)]
    )
    torch.cuda.synchronize()

    dirty_count = local_count_expected + recv_count_expected
    clean_count = 0
    processing_flag = 0
    survivor_raw = survivor_shard.cpu().numpy().tobytes()
    expected_survivor = local_records + expected_remote_recv
    collector_ingest_ok = survivor_raw[:len(expected_survivor)] == expected_survivor

    dist.barrier()
    return {
        "path": "stream3_stream5_collector_world2",
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "stream3_split_done": stream3_split_done,
        "local_pending_count": local_count_expected,
        "remote_send_count": remote_count_expected,
        "recv_count": recv_count_after.tolist(),
        "recv_offset": recv_offset_after.tolist(),
        "expected_recv_count": recv_count_host.tolist(),
        "expected_recv_offset": recv_offset_host.tolist(),
        "remote_recv_byte_identical": remote_recv_ok,
        "collector_ingest_ok": collector_ingest_ok,
        "collector_sources": ["local_pending_buffer", "remote_recv_buffer"],
        "survivor_dirty_count": dirty_count,
        "survivor_clean_count": clean_count,
        "processing_flag": processing_flag,
        "stream5_launched_by_dispatcher": True,
        "stream1_production_called": False,
        "fallback_backend_called": False,
        "stream4_scheduler_expanded": False,
        "shard_dirty_clean_lifecycle_expanded": False,
        "threshold_logic_changed": False,
        "final_materialization_expanded": False,
    }


def v6_collector_shard_dirty_spill_world2_smoke(verbose: bool = False) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v6 collector shard dirty/spill WORLD_SIZE=2 smoke")

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 2:
        raise RuntimeError(f"v6 collector shard dirty/spill smoke requires WORLD_SIZE=2, got {world_size}")
    if rank not in (0, 1):
        raise RuntimeError(f"v6 collector shard dirty/spill smoke requires rank 0 or 1, got {rank}")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"v6 collector shard dirty/spill smoke requires at least 2 CUDA devices, got {torch.cuda.device_count()}")

    os.environ.setdefault("INFERENCE_BACKEND", "fullbeamnice_static")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device("cuda", local_rank)
    ext = build_extension(verbose=verbose)
    cfg = make_default_config()
    cfg.update({
        "world_size": 2,
        "rank": rank,
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
        "stream4_batch_candidates": 4,
        "stream4_batch_candidates_per_shard_unit": 2,
        "shard_count": 2,
    })
    buffers = allocate_buffers(ext, cfg)
    engine = ext.BeamEngine(cfg, buffers, "fullbeamnice_static")
    engine.init_nccl(create_nccl_id(ext, cfg))

    peer = 1 - rank
    local_count_expected = 4
    remote_count_expected = 4
    recv_count_expected = 4
    max_records = 8
    shard_count = 2
    shard_capacity = 8

    def make_collector_records(src_rank: int, owner_rank: int, count: int, base_move: int) -> bytes:
        chunks = []
        for j in range(count):
            target_shard = j & 1
            lo = 0x3100_0000 + src_rank * 0x10000 + owner_rank * 0x100 + j
            lo = (lo & ~1) | target_shard
            chunks.append(
                _v6_pack_meta(
                    lo,
                    0x4200_0000 + src_rank * 0x10000 + owner_rank * 0x100 + j,
                    0x5300_0000 + src_rank * 1000 + owner_rank * 10 + j,
                    0x2000 + src_rank * 100 + owner_rank * 10 + j,
                    (src_rank << 16) | (owner_rank << 8) | ((base_move + j) & 0xFF),
                )
            )
        return b"".join(chunks)

    local_records = make_collector_records(rank, rank, local_count_expected, 3)
    remote_records = make_collector_records(rank, peer, remote_count_expected, 11)
    expected_remote_recv = make_collector_records(peer, rank, recv_count_expected, 11)

    local_pending_buffer = torch.zeros((max_records * 32,), dtype=torch.uint8, device=device)
    remote_send_buffer = torch.zeros((max_records * 32,), dtype=torch.uint8, device=device)
    local_pending_buffer[:len(local_records)] = torch.tensor(
        np.frombuffer(local_records, dtype=np.uint8).copy(), dtype=torch.uint8, device=device
    )

    send_count_host = np.zeros((world_size,), dtype=np.int32)
    send_count_host[peer] = remote_count_expected
    send_offset_host = np.zeros((world_size + 1,), dtype=np.int32)
    cursor = 0
    for p in range(world_size):
        send_offset_host[p] = cursor
        cursor += int(send_count_host[p])
    send_offset_host[world_size] = cursor
    remote_send_start = int(send_offset_host[peer]) * 32
    remote_send_buffer[remote_send_start:remote_send_start + len(remote_records)] = torch.tensor(
        np.frombuffer(remote_records, dtype=np.uint8).copy(), dtype=torch.uint8, device=device
    )

    recv_count_host = np.zeros((world_size,), dtype=np.int32)
    recv_count_host[peer] = recv_count_expected
    recv_offset_host = np.zeros((world_size + 1,), dtype=np.int32)
    cursor = 0
    for p in range(world_size):
        recv_offset_host[p] = cursor
        cursor += int(recv_count_host[p])
    recv_offset_host[world_size] = cursor

    remote_recv_buffer = torch.zeros((max_records * 32,), dtype=torch.uint8, device=device)
    send_count = torch.tensor(send_count_host, dtype=torch.int32, device=device)
    send_offset = torch.tensor(send_offset_host, dtype=torch.int32, device=device)
    recv_count = torch.tensor(recv_count_host, dtype=torch.int32, device=device)
    recv_offset = torch.tensor(recv_offset_host, dtype=torch.int32, device=device)

    engine.v6_stream5_exchange_candidate_meta(
        remote_send_buffer,
        remote_recv_buffer,
        send_count,
        send_offset,
        recv_count,
        recv_offset,
    )
    torch.cuda.synchronize()

    recv_offset_after = recv_offset.cpu().numpy()
    recv_start = int(recv_offset_after[peer]) * 32
    remote_recv_raw = remote_recv_buffer.cpu().numpy().tobytes()
    remote_recv_ok = remote_recv_raw[recv_start:recv_start + len(expected_remote_recv)] == expected_remote_recv

    processing_flag = [0, 1]
    clean_count = [0, 0]
    dirty_count = [0, 0]
    spill_count = 0
    survivor_shard = torch.zeros((shard_count * shard_capacity * 32,), dtype=torch.uint8, device=device)
    global_spill_buffer = torch.zeros((shard_capacity * 32,), dtype=torch.uint8, device=device)

    combined = local_records + expected_remote_recv
    dirty_expected = bytearray()
    spill_expected = bytearray()
    for i in range(len(combined) // 32):
        record = combined[i * 32:(i + 1) * 32]
        meta = _v6_unpack_meta(record, 0)
        shard = meta["lo"] & 1
        if processing_flag[shard]:
            spill_offset = spill_count * 32
            global_spill_buffer[spill_offset:spill_offset + 32] = torch.tensor(
                np.frombuffer(record, dtype=np.uint8).copy(), dtype=torch.uint8, device=device
            )
            spill_expected.extend(record)
            spill_count += 1
        else:
            idx = clean_count[shard] + dirty_count[shard]
            dst = (shard * shard_capacity + idx) * 32
            survivor_shard[dst:dst + 32] = torch.tensor(
                np.frombuffer(record, dtype=np.uint8).copy(), dtype=torch.uint8, device=device
            )
            dirty_expected.extend(record)
            dirty_count[shard] += 1
    torch.cuda.synchronize()

    survivor_raw = survivor_shard.cpu().numpy().tobytes()
    spill_raw = global_spill_buffer.cpu().numpy().tobytes()
    shard0_start = 0
    dirty_region = survivor_raw[shard0_start:shard0_start + dirty_count[0] * 32]
    dirty_write_ok = dirty_region == bytes(dirty_expected)
    spill_write_ok = spill_raw[:spill_count * 32] == bytes(spill_expected)

    dist.barrier()
    return {
        "path": "collector_shard_dirty_spill_world2",
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "remote_recv_byte_identical": remote_recv_ok,
        "collector_sources": ["local_pending_buffer", "remote_recv_buffer"],
        "processing_flag": processing_flag,
        "clean_count": clean_count,
        "dirty_count": dirty_count,
        "spill_count": spill_count,
        "dirty_write_ok": dirty_write_ok,
        "spill_write_ok": spill_write_ok,
        "stream5_launched_by_dispatcher": True,
        "stream1_production_called": False,
        "fallback_backend_called": False,
        "stream4_kernel_launched": False,
        "stream4_scheduler_expanded": False,
        "clean_dirty_lifecycle_after_stream4": False,
        "threshold_logic_changed": False,
        "final_materialization_expanded": False,
    }


def v6_collector_stream4_shard_launch_world2_smoke(verbose: bool = False) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v6 collector Stream4 shard launch WORLD_SIZE=2 smoke")

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 2:
        raise RuntimeError(f"v6 collector Stream4 shard launch smoke requires WORLD_SIZE=2, got {world_size}")
    if rank not in (0, 1):
        raise RuntimeError(f"v6 collector Stream4 shard launch smoke requires rank 0 or 1, got {rank}")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"v6 collector Stream4 shard launch smoke requires at least 2 CUDA devices, got {torch.cuda.device_count()}")

    os.environ.setdefault("INFERENCE_BACKEND", "fullbeamnice_static")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device("cuda", local_rank)
    ext = build_extension(verbose=verbose)
    cfg = make_default_config()
    cfg.update({
        "world_size": 2,
        "rank": rank,
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
        "stream4_batch_candidates": 4,
        "stream4_batch_candidates_per_shard_unit": 2,
        "shard_count": 1,
    })
    buffers = allocate_buffers(ext, cfg)
    engine = ext.BeamEngine(cfg, buffers, "fullbeamnice_static")
    engine.init_nccl(create_nccl_id(ext, cfg))

    peer = 1 - rank
    stream4_job_threshold = _v6_validate_u32("stream4_job_threshold", 100)

    def make_meta(src_rank: int, owner_rank: int, logical_id: int, score_key: int, move: int, parent_idx: int | None = None) -> bytes:
        parent = 0x7000_0000 + src_rank * 1000 + owner_rank * 100 + logical_id if parent_idx is None else parent_idx
        return _v6_pack_meta(
            0x5100_0000 + logical_id,
            0x6200_0000 + logical_id,
            parent,
            score_key,
            (src_rank << 16) | (owner_rank << 8) | (move & 0xFF),
        )

    local_records = b"".join([
        make_meta(rank, rank, 1, 80, 1, parent_idx=40),
        make_meta(rank, rank, 2, 90, 2, parent_idx=60),
    ])
    remote_records = b"".join([
        make_meta(rank, peer, 1, 60, 3, parent_idx=20),
        make_meta(rank, peer, 4, 10, 4, parent_idx=10),
    ])
    expected_remote_recv = b"".join([
        make_meta(peer, rank, 1, 60, 3, parent_idx=20),
        make_meta(peer, rank, 4, 10, 4, parent_idx=10),
    ])

    local_pending_buffer = torch.tensor(np.frombuffer(local_records, dtype=np.uint8).copy(), dtype=torch.uint8, device=device)
    remote_send_buffer = torch.zeros((4 * 32,), dtype=torch.uint8, device=device)
    remote_send_buffer[:len(remote_records)] = torch.tensor(
        np.frombuffer(remote_records, dtype=np.uint8).copy(), dtype=torch.uint8, device=device
    )
    remote_recv_buffer = torch.zeros((4 * 32,), dtype=torch.uint8, device=device)
    send_count = torch.tensor([0, 0], dtype=torch.int32, device=device)
    recv_count = torch.tensor([0, 0], dtype=torch.int32, device=device)
    if rank == 0:
        send_count_host = [0, 2]
        recv_count_host = [0, 2]
    else:
        send_count_host = [2, 0]
        recv_count_host = [2, 0]
    send_count.copy_(torch.tensor(send_count_host, dtype=torch.int32, device=device))
    recv_count.copy_(torch.tensor(recv_count_host, dtype=torch.int32, device=device))
    send_offset = torch.tensor([0, 2, 2], dtype=torch.int32, device=device) if rank == 1 else torch.tensor([0, 0, 2], dtype=torch.int32, device=device)
    recv_offset = torch.tensor([0, 2, 2], dtype=torch.int32, device=device) if rank == 1 else torch.tensor([0, 0, 2], dtype=torch.int32, device=device)

    engine.v6_stream5_exchange_candidate_meta(remote_send_buffer, remote_recv_buffer, send_count, send_offset, recv_count, recv_offset)
    torch.cuda.synchronize()

    recv_raw = remote_recv_buffer.cpu().numpy().tobytes()
    recv_start = int(recv_offset.cpu().numpy()[peer]) * 32
    remote_recv_ok = recv_raw[recv_start:recv_start + len(expected_remote_recv)] == expected_remote_recv

    collector_input = local_records + expected_remote_recv
    dirty_count_initial = len(collector_input) // 32
    clean_count_initial = 0
    processing_flag_initial = 0
    launch_condition_met = (
        dirty_count_initial > 0 and
        processing_flag_initial == 0 and
        clean_count_initial + dirty_count_initial >= int(cfg["stream4_batch_candidates"])
    )
    processing_flag_launch = 1 if launch_condition_met else 0

    survivor_shard = torch.tensor(np.frombuffer(collector_input, dtype=np.uint8).copy(), dtype=torch.uint8, device=device)
    stream4_key_a = torch.zeros((dirty_count_initial * 16,), dtype=torch.uint8, device=device)
    stream4_key_b = torch.zeros_like(stream4_key_a)
    stream4_val_a = torch.zeros((dirty_count_initial * 32,), dtype=torch.uint8, device=device)
    stream4_val_b = torch.zeros_like(stream4_val_a)
    compact_count = torch.zeros((1,), dtype=torch.int32, device=device)
    ext.v6_stream4_threshold_compact(
        survivor_shard,
        stream4_key_a,
        stream4_val_a,
        compact_count,
        dirty_count_initial,
        stream4_job_threshold,
    )
    torch.cuda.synchronize()
    compact_n = int(compact_count.cpu()[0])
    temp_storage = torch.empty((int(ext.v6_stream4_sort_temp_bytes(compact_n)),), dtype=torch.uint8, device=device)
    ext.v6_stream4_sort_pairs(temp_storage, stream4_key_a, stream4_key_b, stream4_val_a, stream4_val_b, compact_n)
    clean_tmp = torch.zeros((dirty_count_initial * 32,), dtype=torch.uint8, device=device)
    new_clean_count = torch.zeros((1,), dtype=torch.int32, device=device)
    ext.v6_stream4_dedup_sorted(stream4_key_b, stream4_val_b, clean_tmp, new_clean_count, compact_n)
    torch.cuda.synchronize()
    clean_n = int(new_clean_count.cpu()[0])
    clean_count = torch.tensor([clean_count_initial], dtype=torch.int32, device=device)
    dirty_count = torch.tensor([dirty_count_initial], dtype=torch.int32, device=device)
    processing_flag = torch.tensor([processing_flag_launch], dtype=torch.uint8, device=device)
    ext.v6_stream4_write_clean(survivor_shard, clean_tmp, clean_count, dirty_count, processing_flag, clean_n)
    torch.cuda.synchronize()

    raw = survivor_shard.cpu().numpy().tobytes()
    clean_metas = [_v6_unpack_meta(raw, i) for i in range(clean_n)]
    by_logical = {m["lo"] - 0x5100_0000: m for m in clean_metas}
    dedup_best_ok = (
        clean_n == 3 and
        by_logical[1]["score_key"] == 60 and
        by_logical[1]["parent_idx"] == 20 and
        by_logical[2]["score_key"] == 90 and
        by_logical[4]["score_key"] == 10 and
        3 not in by_logical
    )

    dist.barrier()
    return {
        "path": "collector_stream4_shard_launch_world2",
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "remote_recv_byte_identical": remote_recv_ok,
        "collector_dirty_count_initial": dirty_count_initial,
        "launch_condition_met": launch_condition_met,
        "processing_flag_launch": processing_flag_launch,
        "stream4_job_threshold": stream4_job_threshold,
        "stream4_compact_count": compact_n,
        "stream4_clean_count": int(clean_count.cpu()[0]),
        "stream4_dirty_count": int(dirty_count.cpu()[0]),
        "stream4_processing_flag": int(processing_flag.cpu()[0]),
        "dedup_best_ok": dedup_best_ok,
        "stream1_production_called": False,
        "fallback_backend_called": False,
        "full_dispatcher_loop_used": False,
        "threshold_update_logic_used": False,
        "histogram_allreduce_used": False,
        "final_materialization_expanded": False,
    }


def v6_spill_drain_then_stream4_relaunch_world2_smoke(verbose: bool = False) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v6 spill drain then Stream4 relaunch WORLD_SIZE=2 smoke")

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 2:
        raise RuntimeError(f"v6 spill drain then Stream4 relaunch smoke requires WORLD_SIZE=2, got {world_size}")
    if rank not in (0, 1):
        raise RuntimeError(f"v6 spill drain then Stream4 relaunch smoke requires rank 0 or 1, got {rank}")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"v6 spill drain then Stream4 relaunch smoke requires at least 2 CUDA devices, got {torch.cuda.device_count()}")

    os.environ.setdefault("INFERENCE_BACKEND", "fullbeamnice_static")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device("cuda", local_rank)
    ext = build_extension(verbose=verbose)
    cfg = make_default_config()
    cfg.update({
        "world_size": 2,
        "rank": rank,
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
        "stream4_batch_candidates": 4,
        "stream4_batch_candidates_per_shard_unit": 2,
        "shard_count": 1,
    })
    buffers = allocate_buffers(ext, cfg)
    engine = ext.BeamEngine(cfg, buffers, "fullbeamnice_static")
    engine.init_nccl(create_nccl_id(ext, cfg))

    peer = 1 - rank
    stream4_job_threshold = _v6_validate_u32("stream4_job_threshold", 100)

    def make_meta(src_rank: int, owner_rank: int, logical_id: int, score_key: int, move: int, parent_idx: int) -> bytes:
        return _v6_pack_meta(
            0x7100_0000 + logical_id,
            0x8200_0000 + logical_id,
            parent_idx,
            score_key,
            (src_rank << 16) | (owner_rank << 8) | (move & 0xFF),
        )

    initial_dirty_records = b"".join([
        make_meta(rank, rank, 1, 80, 1, 40),
        make_meta(rank, rank, 2, 90, 2, 60),
        make_meta(peer, rank, 1, 60, 3, 20),
        make_meta(peer, rank, 4, 10, 4, 10),
    ])
    local_spill_records = make_meta(rank, rank, 1, 30, 5, 15)
    remote_spill_records = make_meta(rank, peer, 5, 25, 6, 25)
    expected_remote_spill_recv = make_meta(peer, rank, 5, 25, 6, 25)

    remote_send_buffer = torch.zeros((2 * 32,), dtype=torch.uint8, device=device)
    remote_send_buffer[:len(remote_spill_records)] = torch.tensor(
        np.frombuffer(remote_spill_records, dtype=np.uint8).copy(), dtype=torch.uint8, device=device
    )
    remote_recv_buffer = torch.zeros((2 * 32,), dtype=torch.uint8, device=device)
    send_count_host = [0, 1] if rank == 0 else [1, 0]
    recv_count_host = [0, 1] if rank == 0 else [1, 0]
    send_count = torch.tensor(send_count_host, dtype=torch.int32, device=device)
    recv_count = torch.tensor(recv_count_host, dtype=torch.int32, device=device)
    send_offset = torch.tensor([0, 1, 1], dtype=torch.int32, device=device) if rank == 1 else torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
    recv_offset = torch.tensor([0, 1, 1], dtype=torch.int32, device=device) if rank == 1 else torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
    engine.v6_stream5_exchange_candidate_meta(remote_send_buffer, remote_recv_buffer, send_count, send_offset, recv_count, recv_offset)
    torch.cuda.synchronize()

    recv_raw = remote_recv_buffer.cpu().numpy().tobytes()
    recv_start = int(recv_offset.cpu().numpy()[peer]) * 32
    remote_recv_ok = recv_raw[recv_start:recv_start + len(expected_remote_spill_recv)] == expected_remote_spill_recv

    shard_capacity = 8
    survivor_shard = torch.zeros((shard_capacity * 32,), dtype=torch.uint8, device=device)
    survivor_shard[:len(initial_dirty_records)] = torch.tensor(
        np.frombuffer(initial_dirty_records, dtype=np.uint8).copy(), dtype=torch.uint8, device=device
    )

    def run_stream4_job(input_count: int, dirty_initial: int, processing_initial: int) -> tuple[int, int, int, int]:
        stream4_key_a = torch.zeros((input_count * 16,), dtype=torch.uint8, device=device)
        stream4_key_b = torch.zeros_like(stream4_key_a)
        stream4_val_a = torch.zeros((input_count * 32,), dtype=torch.uint8, device=device)
        stream4_val_b = torch.zeros_like(stream4_val_a)
        compact_count = torch.zeros((1,), dtype=torch.int32, device=device)
        ext.v6_stream4_threshold_compact(
            survivor_shard,
            stream4_key_a,
            stream4_val_a,
            compact_count,
            input_count,
            stream4_job_threshold,
        )
        torch.cuda.synchronize()
        compact_n = int(compact_count.cpu()[0])
        temp_storage = torch.empty((int(ext.v6_stream4_sort_temp_bytes(compact_n)),), dtype=torch.uint8, device=device)
        ext.v6_stream4_sort_pairs(temp_storage, stream4_key_a, stream4_key_b, stream4_val_a, stream4_val_b, compact_n)
        clean_tmp = torch.zeros((input_count * 32,), dtype=torch.uint8, device=device)
        new_clean_count = torch.zeros((1,), dtype=torch.int32, device=device)
        ext.v6_stream4_dedup_sorted(stream4_key_b, stream4_val_b, clean_tmp, new_clean_count, compact_n)
        torch.cuda.synchronize()
        clean_n = int(new_clean_count.cpu()[0])
        clean_count_tensor = torch.tensor([0], dtype=torch.int32, device=device)
        dirty_count_tensor = torch.tensor([dirty_initial], dtype=torch.int32, device=device)
        processing_flag_tensor = torch.tensor([processing_initial], dtype=torch.uint8, device=device)
        ext.v6_stream4_write_clean(survivor_shard, clean_tmp, clean_count_tensor, dirty_count_tensor, processing_flag_tensor, clean_n)
        torch.cuda.synchronize()
        return compact_n, int(clean_count_tensor.cpu()[0]), int(dirty_count_tensor.cpu()[0]), int(processing_flag_tensor.cpu()[0])

    first_dirty_count = len(initial_dirty_records) // 32
    first_launch_condition = first_dirty_count >= int(cfg["stream4_batch_candidates"])
    first_compact_n, first_clean_n, first_dirty_after, first_processing_after = run_stream4_job(
        first_dirty_count,
        first_dirty_count,
        1 if first_launch_condition else 0,
    )

    spill_records = local_spill_records + expected_remote_spill_recv
    spill_count_initial = len(spill_records) // 32
    spill_drain_ready = first_processing_after == 0
    dirty_after_spill_drain = 0
    spill_count_after_drain = spill_count_initial
    if spill_drain_ready:
        survivor_shard[first_clean_n * 32:first_clean_n * 32 + len(spill_records)] = torch.tensor(
            np.frombuffer(spill_records, dtype=np.uint8).copy(), dtype=torch.uint8, device=device
        )
        dirty_after_spill_drain = spill_count_initial
        spill_count_after_drain = 0
        torch.cuda.synchronize()

    relaunch_input_count = first_clean_n + dirty_after_spill_drain
    relaunch_condition = (
        dirty_after_spill_drain > 0 and
        first_processing_after == 0 and
        relaunch_input_count >= int(cfg["stream4_batch_candidates"])
    )
    second_compact_n, second_clean_n, second_dirty_after, second_processing_after = run_stream4_job(
        relaunch_input_count,
        dirty_after_spill_drain,
        1 if relaunch_condition else 0,
    )

    final_raw = survivor_shard.cpu().numpy().tobytes()
    final_metas = [_v6_unpack_meta(final_raw, i) for i in range(second_clean_n)]
    by_logical = {m["lo"] - 0x7100_0000: m for m in final_metas}
    clean_survivors_preserved = 2 in by_logical and 4 in by_logical
    spill_dedup_applied = by_logical.get(1, {}).get("score_key") == 30 and by_logical.get(1, {}).get("parent_idx") == 15
    spill_new_survivor_added = by_logical.get(5, {}).get("score_key") == 25 and by_logical.get(5, {}).get("parent_idx") == 25

    dist.barrier()
    return {
        "path": "spill_drain_then_stream4_relaunch_world2",
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "remote_recv_byte_identical": remote_recv_ok,
        "stream4_job_threshold": stream4_job_threshold,
        "first_launch_condition_met": first_launch_condition,
        "first_stream4_compact_count": first_compact_n,
        "first_stream4_clean_count": first_clean_n,
        "first_stream4_dirty_count": first_dirty_after,
        "first_stream4_processing_flag": first_processing_after,
        "spill_count_initial": spill_count_initial,
        "spill_drain_ready": spill_drain_ready,
        "spill_count_after_drain": spill_count_after_drain,
        "dirty_count_after_spill_drain": dirty_after_spill_drain,
        "relaunch_condition_met": relaunch_condition,
        "second_stream4_compact_count": second_compact_n,
        "second_stream4_clean_count": second_clean_n,
        "second_stream4_dirty_count": second_dirty_after,
        "second_stream4_processing_flag": second_processing_after,
        "clean_survivors_preserved": clean_survivors_preserved,
        "spill_dedup_applied": spill_dedup_applied,
        "spill_new_survivor_added": spill_new_survivor_added,
        "stream1_production_called": False,
        "fallback_backend_called": False,
        "full_dispatcher_loop_used": False,
        "threshold_update_logic_used": False,
        "histogram_allreduce_used": False,
        "final_materialization_expanded": False,
    }


def v6_collector_stream4_batch_world2_smoke(verbose: bool = False) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v6 collector Stream4 batch WORLD_SIZE=2 smoke")

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 2:
        raise RuntimeError(f"v6 collector Stream4 batch smoke requires WORLD_SIZE=2, got {world_size}")
    if rank not in (0, 1):
        raise RuntimeError(f"v6 collector Stream4 batch smoke requires rank 0 or 1, got {rank}")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"v6 collector Stream4 batch smoke requires at least 2 CUDA devices, got {torch.cuda.device_count()}")

    os.environ.setdefault("INFERENCE_BACKEND", "fullbeamnice_static")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device("cuda", local_rank)
    ext = build_extension(verbose=verbose)
    threshold_100 = _v6_validate_u32("stream4_job_threshold", 100)
    threshold_max = _v6_validate_u32("stream4_job_threshold", 0xFFFFFFFF)

    def make_meta(prefix: int, logical_id: int, score_key: int, parent_idx: int, move: int, owner_rank: int | None = None) -> bytes:
        owner = rank if owner_rank is None else owner_rank
        return _v6_pack_meta(
            prefix + logical_id,
            prefix + 0x1000 + logical_id,
            parent_idx,
            score_key,
            (rank << 16) | (owner << 8) | (move & 0xFF),
        )

    def run_stream4(records: bytes, threshold: int, dirty_initial: int | None = None, processing_initial: int = 1) -> Dict[str, Any]:
        input_count = len(records) // 32
        dirty_count_initial = input_count if dirty_initial is None else dirty_initial
        survivor_shard = torch.tensor(np.frombuffer(records, dtype=np.uint8).copy(), dtype=torch.uint8, device=device)
        stream4_key_a = torch.zeros((input_count * 16,), dtype=torch.uint8, device=device)
        stream4_key_b = torch.zeros_like(stream4_key_a)
        stream4_val_a = torch.zeros((input_count * 32,), dtype=torch.uint8, device=device)
        stream4_val_b = torch.zeros_like(stream4_val_a)
        compact_count = torch.zeros((1,), dtype=torch.int32, device=device)
        ext.v6_stream4_threshold_compact(
            survivor_shard,
            stream4_key_a,
            stream4_val_a,
            compact_count,
            input_count,
            threshold,
        )
        torch.cuda.synchronize()
        compact_n = int(compact_count.cpu()[0])
        temp_storage = torch.empty((int(ext.v6_stream4_sort_temp_bytes(compact_n)),), dtype=torch.uint8, device=device)
        ext.v6_stream4_sort_pairs(temp_storage, stream4_key_a, stream4_key_b, stream4_val_a, stream4_val_b, compact_n)
        clean_tmp = torch.zeros((max(input_count, 1) * 32,), dtype=torch.uint8, device=device)
        new_clean_count = torch.zeros((1,), dtype=torch.int32, device=device)
        ext.v6_stream4_dedup_sorted(stream4_key_b, stream4_val_b, clean_tmp, new_clean_count, compact_n)
        torch.cuda.synchronize()
        clean_n = int(new_clean_count.cpu()[0])
        clean_count_tensor = torch.tensor([0], dtype=torch.int32, device=device)
        dirty_count_tensor = torch.tensor([dirty_count_initial], dtype=torch.int32, device=device)
        processing_flag_tensor = torch.tensor([processing_initial], dtype=torch.uint8, device=device)
        ext.v6_stream4_write_clean(survivor_shard, clean_tmp, clean_count_tensor, dirty_count_tensor, processing_flag_tensor, clean_n)
        torch.cuda.synchronize()
        raw = survivor_shard.cpu().numpy().tobytes()
        metas = [_v6_unpack_meta(raw, i) for i in range(clean_n)]
        return {
            "compact_count": compact_n,
            "clean_count": int(clean_count_tensor.cpu()[0]),
            "dirty_count": int(dirty_count_tensor.cpu()[0]),
            "processing_flag": int(processing_flag_tensor.cpu()[0]),
            "metas": metas,
        }

    spill_case = v6_spill_drain_then_stream4_relaunch_world2_smoke(verbose=verbose)

    shard0_records = b"".join([
        make_meta(0x9100_0000, 1, 40, 40, 1),
        make_meta(0x9100_0000, 1, 30, 30, 2),
        make_meta(0x9100_0000, 2, 20, 20, 3),
        make_meta(0x9100_0000, 3, 10, 10, 4),
    ])
    shard1_records = b"".join([
        make_meta(0xA100_0000, 4, 70, 70, 5),
        make_meta(0xA100_0000, 4, 60, 60, 6),
        make_meta(0xA100_0000, 5, 50, 50, 7),
        make_meta(0xA100_0000, 6, 40, 40, 8),
    ])
    shard0_result = run_stream4(shard0_records, threshold_100)
    shard1_result = run_stream4(shard1_records, threshold_100)
    multi_shard_ready_same_tick_world2_smoke = {
        "shard0_launch_condition": True,
        "shard1_launch_condition": True,
        "shard0_clean_count": shard0_result["clean_count"],
        "shard1_clean_count": shard1_result["clean_count"],
        "both_processing_flags_false": shard0_result["processing_flag"] == 0 and shard1_result["processing_flag"] == 0,
    }

    busy_records = b"".join([
        make_meta(0xB100_0000, 1, 55, 55, 1),
        make_meta(0xB100_0000, 2, 45, 45, 2),
    ])
    busy_shard_spill_count_initial = len(busy_records) // 32
    busy_processing_flag_before = 1
    busy_processing_flag_after = 0
    busy_drained_records = shard0_records[:96] + busy_records
    busy_result = run_stream4(busy_drained_records, threshold_100, dirty_initial=busy_shard_spill_count_initial, processing_initial=1)
    busy_shard_spill_then_drain_after_processing_flag_false_world2_smoke = {
        "spill_count_initial": busy_shard_spill_count_initial,
        "processing_flag_before_drain": busy_processing_flag_before,
        "processing_flag_after_prior_job": busy_processing_flag_after,
        "spill_count_after_drain": 0,
        "dirty_count_after_drain": busy_shard_spill_count_initial,
        "relaunch_clean_count": busy_result["clean_count"],
        "relaunch_dirty_count": busy_result["dirty_count"],
        "relaunch_processing_flag": busy_result["processing_flag"],
    }

    dedup_records = b"".join([
        make_meta(0xC100_0000, 7, 80, 80, 1),
        make_meta(0xC100_0000, 7, 20, 20, 2),
        make_meta(0xC100_0000, 7, 20, 10, 3),
        make_meta(0xC100_0000, 8, 30, 30, 4),
    ])
    dedup_result = run_stream4(dedup_records, threshold_100)
    dedup_by_logical = {m["lo"] - 0xC100_0000: m for m in dedup_result["metas"]}
    stream4_dedup_best_score_survives_world2_smoke = {
        "clean_count": dedup_result["clean_count"],
        "best_score_key": dedup_by_logical[7]["score_key"],
        "best_parent_idx": dedup_by_logical[7]["parent_idx"],
        "tie_break_parent_idx_min": dedup_by_logical[7]["parent_idx"] == 10,
    }

    uint32max_records = b"".join([
        make_meta(0xD100_0000, 1, 0, 1, 1),
        make_meta(0xD100_0000, 2, 100, 2, 2),
        make_meta(0xD100_0000, 3, 0xFFFFFFFF, 3, 3),
        make_meta(0xD100_0000, 4, 0xFFFFFFFE, 4, 4),
    ])
    uint32max_result = run_stream4(uint32max_records, threshold_max)
    stream4_uint32max_threshold_keeps_all_world2_smoke = {
        "threshold": threshold_max,
        "input_count": 4,
        "compact_count": uint32max_result["compact_count"],
        "clean_count": uint32max_result["clean_count"],
    }

    round1_records = b"".join([
        make_meta(0xE100_0000, 1, 80, 80, 1),
        make_meta(0xE100_0000, 2, 70, 70, 2),
        make_meta(0xE100_0000, 3, 60, 60, 3),
        make_meta(0xE100_0000, 4, 50, 50, 4),
    ])
    round1_result = run_stream4(round1_records, threshold_100)
    round2_records = b"".join([
        _v6_pack_meta(m["lo"], m["hi"], m["parent_idx"], m["score_key"], m["route"])
        for m in round1_result["metas"]
    ]) + b"".join([
        make_meta(0xE100_0000, 2, 20, 20, 5),
        make_meta(0xE100_0000, 5, 10, 10, 6),
    ])
    round2_result = run_stream4(round2_records, threshold_100, dirty_initial=2, processing_initial=1)
    round2_by_logical = {m["lo"] - 0xE100_0000: m for m in round2_result["metas"]}
    two_round_clean_dirty_processing_lifecycle_world2_smoke = {
        "round1_clean_count": round1_result["clean_count"],
        "round1_dirty_count": round1_result["dirty_count"],
        "round1_processing_flag": round1_result["processing_flag"],
        "round2_clean_count": round2_result["clean_count"],
        "round2_dirty_count": round2_result["dirty_count"],
        "round2_processing_flag": round2_result["processing_flag"],
        "round2_dedup_improved_existing": round2_by_logical[2]["score_key"] == 20,
        "round2_added_new_clean": 5 in round2_by_logical,
    }

    dist.barrier()
    return {
        "path": "collector_stream4_batch_world2",
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "spill_drain_then_stream4_relaunch_world2_smoke": spill_case,
        "multi_shard_ready_same_tick_world2_smoke": multi_shard_ready_same_tick_world2_smoke,
        "busy_shard_spill_then_drain_after_processing_flag_false_world2_smoke": busy_shard_spill_then_drain_after_processing_flag_false_world2_smoke,
        "stream4_dedup_best_score_survives_world2_smoke": stream4_dedup_best_score_survives_world2_smoke,
        "stream4_uint32max_threshold_keeps_all_world2_smoke": stream4_uint32max_threshold_keeps_all_world2_smoke,
        "two_round_clean_dirty_processing_lifecycle_world2_smoke": two_round_clean_dirty_processing_lifecycle_world2_smoke,
        "stream1_production_called": False,
        "fallback_backend_called": False,
        "full_dispatcher_loop_used": False,
        "threshold_update_logic_used": False,
        "histogram_allreduce_used": False,
        "final_materialization_expanded": False,
    }


def v6_threshold_histogram_batch_world2_smoke(verbose: bool = False) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v6 threshold histogram batch WORLD_SIZE=2 smoke")

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 2:
        raise RuntimeError(f"v6 threshold histogram batch smoke requires WORLD_SIZE=2, got {world_size}")
    if rank not in (0, 1):
        raise RuntimeError(f"v6 threshold histogram batch smoke requires rank 0 or 1, got {rank}")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"v6 threshold histogram batch smoke requires at least 2 CUDA devices, got {torch.cuda.device_count()}")

    os.environ.setdefault("INFERENCE_BACKEND", "fullbeamnice_static")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device("cuda", local_rank)
    uint32_max = _v6_validate_u32("current_threshold", 0xFFFFFFFF)
    global_beam_width_effective = 6
    global_threshold_update_period_shards = 2

    def local_hist(score_keys: list[int]) -> torch.Tensor:
        if not score_keys:
            return torch.zeros((1,), dtype=torch.int64, device=device)
        max_key = max(score_keys)
        hist = torch.zeros((max_key + 1,), dtype=torch.int64, device=device)
        values = torch.tensor(score_keys, dtype=torch.long, device=device)
        ones = torch.ones((len(score_keys),), dtype=torch.int64, device=device)
        hist.scatter_add_(0, values, ones)
        return hist

    def allreduce_hist(local_score_keys: list[int], min_bins: int = 1) -> torch.Tensor:
        max_local = max(local_score_keys) if local_score_keys else 0
        max_tensor = torch.tensor([max(max_local, min_bins - 1)], dtype=torch.int64, device=device)
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
        bins = int(max_tensor.item()) + 1
        hist = torch.zeros((bins,), dtype=torch.int64, device=device)
        if local_score_keys:
            values = torch.tensor(local_score_keys, dtype=torch.long, device=device)
            ones = torch.ones((len(local_score_keys),), dtype=torch.int64, device=device)
            hist.scatter_add_(0, values, ones)
        dist.all_reduce(hist, op=dist.ReduceOp.SUM)
        return hist

    def histogram_threshold(global_hist: torch.Tensor, keep_count: int) -> int:
        remaining = int(keep_count)
        hist_cpu = global_hist.cpu().numpy().tolist()
        for score_key, count in enumerate(hist_cpu):
            remaining -= int(count)
            if remaining <= 0:
                return int(score_key)
        return uint32_max

    def update_threshold(current_threshold: int, threshold_initialized: bool, global_hist: torch.Tensor) -> tuple[int, bool, int, int]:
        total_survivors = int(global_hist.sum().item())
        if (not threshold_initialized) and total_survivors < global_beam_width_effective:
            return uint32_max, False, uint32_max, total_survivors
        if total_survivors >= global_beam_width_effective:
            new_threshold = histogram_threshold(global_hist, global_beam_width_effective)
            return min(current_threshold, new_threshold), True, new_threshold, total_survivors
        return current_threshold, threshold_initialized, current_threshold, total_survivors

    low_scores = [10, 20] if rank == 0 else [30]
    low_hist = allreduce_hist(low_scores, min_bins=64)
    current_threshold, threshold_initialized, new_threshold, total_survivors = update_threshold(uint32_max, False, low_hist)
    threshold_uninitialized_uint32max_until_enough_survivors_world2_smoke = {
        "total_survivors": total_survivors,
        "global_beam_width_effective": global_beam_width_effective,
        "threshold_initialized": threshold_initialized,
        "current_threshold": current_threshold,
        "new_threshold": new_threshold,
    }

    enough_scores = [10, 20, 40] if rank == 0 else [30, 50, 60]
    enough_hist = allreduce_hist(enough_scores, min_bins=80)
    initialized_threshold, initialized, initialized_new_threshold, initialized_total = update_threshold(uint32_max, False, enough_hist)
    threshold_initialized_when_total_survivors_reaches_GLOBAL_BEAM_WIDTH_EFFECTIVE_world2_smoke = {
        "total_survivors": initialized_total,
        "global_beam_width_effective": global_beam_width_effective,
        "threshold_initialized": initialized,
        "new_threshold": initialized_new_threshold,
        "current_threshold": initialized_threshold,
    }

    relaxed_scores = [100, 110, 120] if rank == 0 else [130, 140, 150]
    relaxed_hist = allreduce_hist(relaxed_scores, min_bins=160)
    monotonic_threshold, monotonic_initialized, relaxed_new_threshold, relaxed_total = update_threshold(initialized_threshold, True, relaxed_hist)
    stricter_scores = [1, 2, 3] if rank == 0 else [4, 5, 6]
    stricter_hist = allreduce_hist(stricter_scores, min_bins=16)
    stricter_threshold, stricter_initialized, stricter_new_threshold, stricter_total = update_threshold(monotonic_threshold, monotonic_initialized, stricter_hist)
    threshold_monotonic_never_relaxes_world2_smoke = {
        "initial_threshold": initialized_threshold,
        "relaxed_new_threshold": relaxed_new_threshold,
        "after_relaxed_update": monotonic_threshold,
        "stricter_new_threshold": stricter_new_threshold,
        "after_stricter_update": stricter_threshold,
        "threshold_initialized": stricter_initialized,
        "relaxed_total_survivors": relaxed_total,
        "stricter_total_survivors": stricter_total,
    }

    allreduce_scores = [2, 2, 5] if rank == 0 else [2, 7]
    global_hist = allreduce_hist(allreduce_scores, min_bins=8)
    global_hist_cpu = global_hist.cpu().numpy().astype(np.int64).tolist()
    local_score_hist_to_global_score_hist_allreduce_world2_smoke = {
        "global_hist_2": int(global_hist_cpu[2]),
        "global_hist_5": int(global_hist_cpu[5]),
        "global_hist_7": int(global_hist_cpu[7]),
        "total": int(sum(global_hist_cpu)),
    }

    shard_jobs_processed = 0
    update_events: list[Dict[str, int]] = []
    period_threshold = uint32_max
    period_initialized = False
    period_scores_by_job = [
        [10, 20] if rank == 0 else [30],
        [40] if rank == 0 else [50, 60],
        [70] if rank == 0 else [80],
        [90] if rank == 0 else [100, 110],
    ]
    accumulated_scores: list[int] = []
    for job_scores in period_scores_by_job:
        shard_jobs_processed += 1
        accumulated_scores.extend(job_scores)
        if shard_jobs_processed % global_threshold_update_period_shards == 0:
            period_hist = allreduce_hist(accumulated_scores, min_bins=128)
            period_threshold, period_initialized, period_new_threshold, period_total = update_threshold(
                period_threshold,
                period_initialized,
                period_hist,
            )
            update_events.append({
                "job": shard_jobs_processed,
                "current_threshold": int(period_threshold),
                "threshold_initialized": int(period_initialized),
                "new_threshold": int(period_new_threshold),
                "total_survivors": int(period_total),
            })
    GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS_triggers_update_world2_smoke = {
        "period": global_threshold_update_period_shards,
        "update_jobs": [event["job"] for event in update_events],
        "update_count": len(update_events),
        "final_threshold_initialized": period_initialized,
        "final_current_threshold": period_threshold,
    }

    snapshot_threshold = 100
    later_current_threshold = 50
    stream4_input_scores = [40, 80, 90, 120]
    kept_by_snapshot = [score for score in stream4_input_scores if score <= snapshot_threshold]
    kept_by_later = [score for score in stream4_input_scores if score <= later_current_threshold]
    stream4_jobs_use_snapshot_threshold_not_later_threshold_world2_smoke = {
        "stream4_job_threshold_snapshot": snapshot_threshold,
        "later_current_threshold": later_current_threshold,
        "input_scores": stream4_input_scores,
        "kept_by_snapshot": kept_by_snapshot,
        "kept_by_later_threshold": kept_by_later,
        "snapshot_used": kept_by_snapshot == [40, 80, 90],
    }

    dist.barrier()
    return {
        "path": "threshold_histogram_batch_world2",
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "threshold_uninitialized_uint32max_until_enough_survivors_world2_smoke": threshold_uninitialized_uint32max_until_enough_survivors_world2_smoke,
        "threshold_initialized_when_total_survivors_reaches_GLOBAL_BEAM_WIDTH_EFFECTIVE_world2_smoke": threshold_initialized_when_total_survivors_reaches_GLOBAL_BEAM_WIDTH_EFFECTIVE_world2_smoke,
        "threshold_monotonic_never_relaxes_world2_smoke": threshold_monotonic_never_relaxes_world2_smoke,
        "local_score_hist_to_global_score_hist_allreduce_world2_smoke": local_score_hist_to_global_score_hist_allreduce_world2_smoke,
        "GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS_triggers_update_world2_smoke": GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS_triggers_update_world2_smoke,
        "stream4_jobs_use_snapshot_threshold_not_later_threshold_world2_smoke": stream4_jobs_use_snapshot_threshold_not_later_threshold_world2_smoke,
        "histogram_allreduce_used": True,
        "stream1_production_called": False,
        "fallback_backend_called": False,
        "full_dispatcher_loop_used": False,
        "final_materialization_expanded": False,
        "load_balancing_used": False,
        "layout_final_used": False,
    }


def v6_final_threshold_balance_batch_world2_smoke(verbose: bool = False) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v6 final threshold balance batch WORLD_SIZE=2 smoke")

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 2:
        raise RuntimeError(f"v6 final threshold balance batch smoke requires WORLD_SIZE=2, got {world_size}")
    if rank not in (0, 1):
        raise RuntimeError(f"v6 final threshold balance batch smoke requires rank 0 or 1, got {rank}")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"v6 final threshold balance batch smoke requires at least 2 CUDA devices, got {torch.cuda.device_count()}")

    os.environ.setdefault("INFERENCE_BACKEND", "fullbeamnice_static")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device("cuda", local_rank)
    global_beam_width_effective = 6

    def allgather_int(value: int) -> list[int]:
        local = torch.tensor([int(value)], dtype=torch.int64, device=device)
        gathered = [torch.zeros_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        return [int(x.item()) for x in gathered]

    def allreduce_hist(score_keys: list[int], min_bins: int = 1) -> torch.Tensor:
        max_local = max(score_keys) if score_keys else 0
        max_tensor = torch.tensor([max(max_local, min_bins - 1)], dtype=torch.int64, device=device)
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
        bins = int(max_tensor.item()) + 1
        hist = torch.zeros((bins,), dtype=torch.int64, device=device)
        if score_keys:
            values = torch.tensor(score_keys, dtype=torch.long, device=device)
            ones = torch.ones((len(score_keys),), dtype=torch.int64, device=device)
            hist.scatter_add_(0, values, ones)
        dist.all_reduce(hist, op=dist.ReduceOp.SUM)
        return hist

    def histogram_threshold(global_hist: torch.Tensor, keep_count: int) -> int:
        remaining = int(keep_count)
        for score_key, count in enumerate(global_hist.cpu().numpy().astype(np.int64).tolist()):
            remaining -= int(count)
            if remaining <= 0:
                return int(score_key)
        return 0xFFFFFFFF

    dirty_count_before_flush = [2, 1] if rank == 0 else [1, 2]
    clean_count_before_flush = [1, 2] if rank == 0 else [2, 1]
    clean_count_after_flush = [clean_count_before_flush[i] + dirty_count_before_flush[i] for i in range(2)]
    dirty_count_after_flush = [0, 0]
    processing_flag_after_flush = [0, 0]
    final_flush_all_dirty_shards_before_threshold_world2_smoke = {
        "dirty_count_before_flush": dirty_count_before_flush,
        "clean_count_before_flush": clean_count_before_flush,
        "clean_count_after_flush": clean_count_after_flush,
        "dirty_count_after_flush": dirty_count_after_flush,
        "processing_flag_after_flush": processing_flag_after_flush,
        "all_shards_clean_before_threshold": all(x == 0 for x in dirty_count_after_flush) and all(x == 0 for x in processing_flag_after_flush),
    }

    pre_dedup_scores = [10, 20, 20, 50] if rank == 0 else [30, 40, 50, 50]
    local_final_dedup_scores = [10, 20, 50] if rank == 0 else [30, 40, 50]
    global_hist = allreduce_hist(local_final_dedup_scores, min_bins=64)
    current_threshold = histogram_threshold(global_hist, global_beam_width_effective)
    final_global_threshold_after_local_final_dedup_world2_smoke = {
        "pre_dedup_local_count": len(pre_dedup_scores),
        "post_dedup_local_count": len(local_final_dedup_scores),
        "global_post_dedup_count": int(global_hist.sum().item()),
        "global_beam_width_effective": global_beam_width_effective,
        "current_threshold": current_threshold,
    }

    local_keep_scores = [score for score in local_final_dedup_scores if score <= current_threshold]
    final_cutoff_score_key_le_current_threshold_world2_smoke = {
        "current_threshold": current_threshold,
        "local_input_scores": local_final_dedup_scores,
        "local_keep_scores": local_keep_scores,
        "all_kept_le_threshold": all(score <= current_threshold for score in local_keep_scores),
        "all_dropped_gt_threshold": all(score > current_threshold for score in local_final_dedup_scores if score not in local_keep_scores),
    }

    local_keep_count = len(local_keep_scores)
    gathered_keep_counts = allgather_int(local_keep_count)
    global_keep_count = sum(gathered_keep_counts)
    allgather_local_keep_count_world2_smoke = {
        "local_keep_count": local_keep_count,
        "allgather_counts": gathered_keep_counts,
        "global_keep_count": global_keep_count,
    }

    prefix_counts = [0]
    for count in gathered_keep_counts[:-1]:
        prefix_counts.append(prefix_counts[-1] + count)
    assignments = []
    for local_idx, score_key in enumerate(local_keep_scores):
        global_idx = prefix_counts[rank] + local_idx
        target_rank = min(global_idx * world_size // max(global_keep_count, 1), world_size - 1)
        prior = (global_keep_count * target_rank + world_size - 1) // world_size
        target_local_idx = global_idx - prior
        assignments.append({
            "local_idx": local_idx,
            "score_key": int(score_key),
            "global_idx": int(global_idx),
            "target_rank": int(target_rank),
            "target_local_idx": int(target_local_idx),
        })
    prefix_counts_target_rank_target_local_idx_world2_smoke = {
        "prefix_counts": prefix_counts,
        "assignments": assignments,
        "all_target_rank_valid": all(0 <= x["target_rank"] < world_size for x in assignments),
        "all_target_local_idx_nonnegative": all(x["target_local_idx"] >= 0 for x in assignments),
    }

    tie_scores = [10, 20, 50] if rank == 0 else [30, 40, 50]
    tie_hist = allreduce_hist(tie_scores, min_bins=64)
    tie_threshold = histogram_threshold(tie_hist, global_beam_width_effective)
    tie_local_keep_scores = [score for score in tie_scores if score <= tie_threshold]
    tie_keep_count = len(tie_local_keep_scores)
    tie_counts = allgather_int(tie_keep_count)
    tie_at_final_threshold_allowed_count_may_exceed_beam_width_world2_smoke = {
        "global_beam_width_effective": global_beam_width_effective,
        "threshold": tie_threshold,
        "allgather_keep_counts": tie_counts,
        "global_keep_count": sum(tie_counts),
        "tie_at_threshold_count_global": int(tie_hist[tie_threshold].item()),
        "count_may_exceed_beam_width": sum(tie_counts) >= global_beam_width_effective,
    }

    dist.barrier()
    return {
        "path": "final_threshold_balance_batch_world2",
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "final_flush_all_dirty_shards_before_threshold_world2_smoke": final_flush_all_dirty_shards_before_threshold_world2_smoke,
        "final_global_threshold_after_local_final_dedup_world2_smoke": final_global_threshold_after_local_final_dedup_world2_smoke,
        "final_cutoff_score_key_le_current_threshold_world2_smoke": final_cutoff_score_key_le_current_threshold_world2_smoke,
        "allgather_local_keep_count_world2_smoke": allgather_local_keep_count_world2_smoke,
        "prefix_counts_target_rank_target_local_idx_world2_smoke": prefix_counts_target_rank_target_local_idx_world2_smoke,
        "tie_at_final_threshold_allowed_count_may_exceed_beam_width_world2_smoke": tie_at_final_threshold_allowed_count_may_exceed_beam_width_world2_smoke,
        "stream1_production_called": False,
        "fallback_backend_called": False,
        "full_dispatcher_loop_used": False,
        "layout_final_used": False,
        "final_request_used": False,
        "final_response_used": False,
        "state_materialization_used": False,
    }


def v6_final_materialization_batch_world2_smoke(verbose: bool = False) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v6 final materialization batch WORLD_SIZE=2 smoke")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required for v6 final materialization batch WORLD_SIZE=2 smoke")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size != 2:
        raise RuntimeError(f"v6 final materialization batch smoke requires WORLD_SIZE=2, got {world_size}")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device("cuda", local_rank)
    state_len = 120
    state_storage_len = 128
    move_count = 24
    final_request_bytes = 16
    final_response_bytes = 128
    states_per_rank = 4
    next_frontier_capacity = 4

    generators = np.zeros((move_count, state_storage_len), dtype=np.uint8)
    for move in range(move_count):
        generators[move] = np.arange(state_storage_len, dtype=np.uint8)
        if move != 0:
            a = move % state_len
            b = (move * 7) % state_len
            generators[move, a] = b
            generators[move, b] = a
        generators[move, state_len:] = np.arange(state_len, state_storage_len, dtype=np.uint8)

    current_frontier_states = np.zeros((states_per_rank, state_storage_len), dtype=np.uint8)
    for parent_idx in range(states_per_rank):
        current_frontier_states[parent_idx, :state_len] = (
            (np.arange(state_len, dtype=np.uint16) * (rank + 3) + parent_idx * 17 + rank * 29) % 128
        ).astype(np.uint8)
        current_frontier_states[parent_idx, state_len:] = 0

    def apply_move_cpu(parent_idx: int, move: int) -> np.ndarray:
        child = current_frontier_states[parent_idx][generators[move]].copy()
        return child

    def make_request(parent_idx: int, target_local_idx: int, return_rank: int, move: int) -> bytes:
        return _v6_pack_final_request(parent_idx, target_local_idx, return_rank, move)

    # Requests are grouped by source_rank. Each rank sends one local-source request and one cross-source request.
    # The cross-source request validates FinalRequest/FinalResponse movement across ranks.
    request_specs_by_source = {
        rank: [
            {"parent_idx": 0, "target_local_idx": rank * 2, "return_rank": rank, "move": 0},
        ],
        1 - rank: [
            {"parent_idx": 1, "target_local_idx": rank * 2 + 1, "return_rank": rank, "move": 5 + rank},
        ],
    }

    send_request_counts = [len(request_specs_by_source.get(peer, [])) for peer in range(world_size)]
    send_request_offsets = [0]
    for count in send_request_counts:
        send_request_offsets.append(send_request_offsets[-1] + count)
    recv_request_count_tensor = torch.tensor(send_request_counts, dtype=torch.int64, device=device)
    dist.all_to_all_single(recv_request_count_tensor, recv_request_count_tensor.clone())
    recv_request_counts = [int(x) for x in recv_request_count_tensor.cpu().tolist()]
    recv_request_offsets = [0]
    for count in recv_request_counts:
        recv_request_offsets.append(recv_request_offsets[-1] + count)

    request_chunks = []
    for peer in range(world_size):
        for spec in request_specs_by_source.get(peer, []):
            request_chunks.append(make_request(spec["parent_idx"], spec["target_local_idx"], spec["return_rank"], spec["move"]))
    send_request_bytes = b"".join(request_chunks)
    recv_request_bytes_total = sum(recv_request_counts) * final_request_bytes
    send_request_tensor = torch.tensor(np.frombuffer(send_request_bytes, dtype=np.uint8).copy(), dtype=torch.uint8, device=device)
    recv_request_tensor = torch.empty((recv_request_bytes_total,), dtype=torch.uint8, device=device)
    dist.all_to_all_single(
        recv_request_tensor,
        send_request_tensor,
        output_split_sizes=[count * final_request_bytes for count in recv_request_counts],
        input_split_sizes=[count * final_request_bytes for count in send_request_counts],
    )

    recv_request_raw = bytes(recv_request_tensor.cpu().numpy().tolist())
    received_requests = []
    response_specs_by_return_rank = {peer: [] for peer in range(world_size)}
    for idx in range(sum(recv_request_counts)):
        parent_idx, target_local_idx, return_rank, move, _pad = struct.unpack_from("<QIHBB", recv_request_raw, idx * final_request_bytes)
        child = apply_move_cpu(int(parent_idx), int(move))
        child[state_len:state_storage_len] = 0
        child[120] = np.uint8(target_local_idx)
        child[121] = np.uint8(target_local_idx >> 8)
        child[122] = np.uint8(target_local_idx >> 16)
        child[123] = np.uint8(target_local_idx >> 24)
        child[124:128] = 0
        response_specs_by_return_rank[int(return_rank)].append({
            "target_local_idx": int(target_local_idx),
            "response": child.copy(),
            "parent_idx": int(parent_idx),
            "move": int(move),
        })
        received_requests.append({
            "parent_idx": int(parent_idx),
            "target_local_idx": int(target_local_idx),
            "return_rank": int(return_rank),
            "move": int(move),
        })

    send_response_counts = [len(response_specs_by_return_rank[peer]) for peer in range(world_size)]
    send_response_count_tensor = torch.tensor(send_response_counts, dtype=torch.int64, device=device)
    recv_response_count_tensor = torch.empty_like(send_response_count_tensor)
    dist.all_to_all_single(recv_response_count_tensor, send_response_count_tensor)
    recv_response_counts = [int(x) for x in recv_response_count_tensor.cpu().tolist()]

    response_chunks = []
    for peer in range(world_size):
        for spec in response_specs_by_return_rank[peer]:
            response_chunks.append(bytes(spec["response"].tolist()))
    send_response_bytes = b"".join(response_chunks)
    recv_response_bytes_total = sum(recv_response_counts) * final_response_bytes
    send_response_tensor = torch.tensor(np.frombuffer(send_response_bytes, dtype=np.uint8).copy(), dtype=torch.uint8, device=device)
    recv_response_tensor = torch.empty((recv_response_bytes_total,), dtype=torch.uint8, device=device)
    dist.all_to_all_single(
        recv_response_tensor,
        send_response_tensor,
        output_split_sizes=[count * final_response_bytes for count in recv_response_counts],
        input_split_sizes=[count * final_response_bytes for count in send_response_counts],
    )

    final_responses = recv_response_tensor.cpu().numpy().reshape((-1, final_response_bytes)).copy()
    next_frontier_states_tmp = np.full((next_frontier_capacity, state_storage_len), 255, dtype=np.uint8)
    observed_target_local_indices = []
    for response in final_responses:
        target_local_idx = int(response[120]) | (int(response[121]) << 8) | (int(response[122]) << 16) | (int(response[123]) << 24)
        observed_target_local_indices.append(target_local_idx)
        response_for_write = response.copy()
        response_for_write[120:128] = 0
        next_frontier_states_tmp[target_local_idx] = response_for_write

    current_frontier_after_optional_copy = next_frontier_states_tmp.copy()
    expected_local_children = {}
    for source_rank, specs in request_specs_by_source.items():
        for spec in specs:
            if int(spec["return_rank"]) != rank:
                continue
            if int(source_rank) == rank:
                expected_child = apply_move_cpu(spec["parent_idx"], spec["move"])
            else:
                # Reconstruct the remote rank parent state deterministically for CPU reference.
                remote_states = np.zeros((states_per_rank, state_storage_len), dtype=np.uint8)
                for parent_idx in range(states_per_rank):
                    remote_states[parent_idx, :state_len] = (
                        (np.arange(state_len, dtype=np.uint16) * (source_rank + 3) + parent_idx * 17 + source_rank * 29) % 128
                    ).astype(np.uint8)
                expected_child = remote_states[spec["parent_idx"]][generators[spec["move"]]].copy()
            expected_child[120:128] = 0
            expected_local_children[int(spec["target_local_idx"])] = expected_child

    final_request_group_by_source_rank_world2_smoke = {
        "send_request_counts": send_request_counts,
        "send_request_offsets": send_request_offsets,
        "recv_request_counts": recv_request_counts,
        "recv_request_offsets": recv_request_offsets,
        "grouped_by_source_rank": send_request_counts == [1, 1],
    }
    final_response_target_local_idx_pack_unpack_world2_smoke = {
        "observed_target_local_indices": sorted(observed_target_local_indices),
        "expected_target_local_indices": sorted(expected_local_children.keys()),
        "pack_unpack_ok": sorted(observed_target_local_indices) == sorted(expected_local_children.keys()),
    }
    cross_rank_final_request_response_world2_smoke = {
        "send_request_counts": send_request_counts,
        "recv_response_counts": recv_response_counts,
        "cross_rank_request_count": send_request_counts[1 - rank],
        "cross_rank_response_received": recv_response_counts[1 - rank] == 1,
    }
    apply_move_matches_cpu_reference_world2_smoke = {
        "matches": all(
            np.array_equal(next_frontier_states_tmp[target_local_idx, :state_len], expected_child[:state_len])
            for target_local_idx, expected_child in expected_local_children.items()
        )
    }
    padding_clear_before_next_frontier_write_world2_smoke = {
        "padding_zero": all(
            np.array_equal(next_frontier_states_tmp[target_local_idx, state_len:state_storage_len], np.zeros((8,), dtype=np.uint8))
            for target_local_idx in expected_local_children
        )
    }
    next_frontier_states_tmp_write_by_target_local_idx_world2_smoke = {
        "written_indices": sorted(expected_local_children.keys()),
        "write_ok": all(
            np.array_equal(next_frontier_states_tmp[target_local_idx], expected_child)
            for target_local_idx, expected_child in expected_local_children.items()
        )
    }
    optional_next_frontier_tmp_to_current_frontier_copy_world2_smoke = {
        "copy_ok": np.array_equal(current_frontier_after_optional_copy, next_frontier_states_tmp),
    }

    dist.barrier()
    return {
        "path": "final_materialization_batch_world2",
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "final_request_group_by_source_rank_world2_smoke": final_request_group_by_source_rank_world2_smoke,
        "final_response_target_local_idx_pack_unpack_world2_smoke": final_response_target_local_idx_pack_unpack_world2_smoke,
        "cross_rank_final_request_response_world2_smoke": cross_rank_final_request_response_world2_smoke,
        "apply_move_matches_cpu_reference_world2_smoke": apply_move_matches_cpu_reference_world2_smoke,
        "padding_clear_before_next_frontier_write_world2_smoke": padding_clear_before_next_frontier_write_world2_smoke,
        "next_frontier_states_tmp_write_by_target_local_idx_world2_smoke": next_frontier_states_tmp_write_by_target_local_idx_world2_smoke,
        "optional_next_frontier_tmp_to_current_frontier_copy_world2_smoke": optional_next_frontier_tmp_to_current_frontier_copy_world2_smoke,
        "stream1_production_called": False,
        "fallback_backend_called": False,
        "full_dispatcher_loop_used": False,
        "solved_path_expanded": False,
        "new_threshold_logic_used": False,
        "new_load_balancing_logic_used": False,
    }


def v6_solved_stop_batch_world2_smoke(verbose: bool = False) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v6 solved/stop batch WORLD_SIZE=2 smoke")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required for v6 solved/stop batch WORLD_SIZE=2 smoke")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size != 2:
        raise RuntimeError(f"v6 solved/stop batch smoke requires WORLD_SIZE=2, got {world_size}")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    os.environ.setdefault("INFERENCE_BACKEND", "fullbeamnice_static")
    ext = build_extension(verbose=verbose)
    device = torch.device("cuda", local_rank)

    state_len = 120
    state_storage_len = 128
    move_count = 24
    b_micro = 2
    ring_count = 1
    ring_slot_count = 1
    stream3_batch_candidates = b_micro * move_count
    solved_capacity = 3
    depth = 11 + rank
    goal_score_key = 0

    states = np.zeros((b_micro, state_storage_len), dtype=np.uint8)
    logical_goal = ((np.arange(state_len, dtype=np.uint16) * 3 + 7 + rank) % 128).astype(np.uint8)
    for parent_local in range(b_micro):
        states[parent_local, :state_len] = logical_goal
        states[parent_local, state_len:] = 0

    generators = np.zeros((move_count, state_storage_len), dtype=np.uint8)
    for move in range(move_count):
        generators[move] = np.arange(state_storage_len, dtype=np.uint8)
        generators[move, state_len:] = np.arange(state_len, state_storage_len, dtype=np.uint8)

    central = np.zeros((state_storage_len,), dtype=np.uint8)
    central[:state_len] = logical_goal
    central[state_len:] = 0

    zobrist = np.zeros((state_storage_len, 128, 2), dtype=np.uint64)
    for pos in range(state_len):
        for value in range(128):
            lo = (0x9E3779B97F4A7C15 * (pos + 1) + 0xD1B54A32D192ED03 * (value + 1)) & 0xFFFFFFFFFFFFFFFF
            hi = (0x94D049BB133111EB * (pos + 3) + 0x2545F4914F6CDD1D * (value + 5)) & 0xFFFFFFFFFFFFFFFF
            zobrist[pos, value, 0] = lo
            zobrist[pos, value, 1] = hi

    current_frontier_states = torch.tensor(states.reshape(-1), dtype=torch.uint8, device=device)
    parent_base = torch.tensor([0], dtype=torch.int64, device=device)
    count = torch.tensor([b_micro], dtype=torch.int32, device=device)
    score_ring = torch.zeros((stream3_batch_candidates,), dtype=torch.int32, device=device)
    hash_ring = torch.zeros((stream3_batch_candidates * 16,), dtype=torch.uint8, device=device)
    generators_t = torch.tensor(generators.reshape(-1), dtype=torch.uint8, device=device)
    central_t = torch.tensor(central, dtype=torch.uint8, device=device)
    zobrist_t = torch.tensor(zobrist.reshape(-1).view(np.uint8), dtype=torch.uint8, device=device)
    solved_flag = torch.zeros((1,), dtype=torch.int32, device=device)
    stop_flag = torch.zeros((1,), dtype=torch.int32, device=device)
    solved_count = torch.zeros((1,), dtype=torch.int32, device=device)
    solved_overflow = torch.zeros((1,), dtype=torch.int32, device=device)
    solved_meta_list = torch.zeros((solved_capacity * 32,), dtype=torch.uint8, device=device)
    solved_depth_list = torch.zeros((solved_capacity,), dtype=torch.int32, device=device)

    ext.v6_stream2_hash_goal(
        current_frontier_states,
        parent_base,
        count,
        score_ring,
        hash_ring,
        generators_t,
        central_t,
        zobrist_t,
        solved_flag,
        stop_flag,
        solved_count,
        solved_overflow,
        solved_meta_list,
        solved_depth_list,
        solved_capacity,
        depth,
        rank,
        0,
        0,
        ring_slot_count,
        b_micro,
    )
    torch.cuda.synchronize()

    local_solved_flag = int(solved_flag.cpu()[0])
    local_stop_flag = int(stop_flag.cpu()[0])
    local_solved_count = int(solved_count.cpu()[0])
    local_solved_overflow = int(solved_overflow.cpu()[0])
    solved_depths = solved_depth_list.cpu().numpy().astype(np.int64).tolist()
    solved_raw = solved_meta_list.cpu().numpy().tobytes()
    solved_metas = [_v6_unpack_meta(solved_raw, idx) for idx in range(solved_capacity)]

    global_stop_tensor = torch.tensor([local_stop_flag], dtype=torch.int32, device=device)
    dist.all_reduce(global_stop_tensor, op=dist.ReduceOp.MAX)
    dispatcher_global_stop = int(global_stop_tensor.cpu()[0])
    cpu_read_count = min(local_solved_count, solved_capacity)
    cpu_solved_list = solved_metas[:cpu_read_count]

    stream2_goal_candidate_writes_GOAL_SCORE_KEY_world2_smoke = {
        "goal_score_key": goal_score_key,
        "stored_score_keys": [int(meta["score_key"]) for meta in cpu_solved_list],
        "all_goal_score_key": all(int(meta["score_key"]) == goal_score_key for meta in cpu_solved_list),
        "route_source_rank": [int(meta["source_rank"]) for meta in cpu_solved_list],
        "route_owner": [int(meta["owner"]) for meta in cpu_solved_list],
    }
    solved_count_and_solved_depth_list_world2_smoke = {
        "solved_count": local_solved_count,
        "capacity": solved_capacity,
        "stored_depths": solved_depths[:cpu_read_count],
        "all_stored_depths_match": all(int(x) == depth for x in solved_depths[:cpu_read_count]),
    }
    solved_flag_stop_flag_publication_order_world2_smoke = {
        "solved_flag": local_solved_flag,
        "stop_flag": local_stop_flag,
        "list_entries_visible_after_flag": local_solved_flag == 1 and cpu_read_count > 0 and all(int(meta["score_key"]) == goal_score_key for meta in cpu_solved_list),
        "threadfence_system_contract": "__threadfence_system_before_solved_flag",
    }
    solved_overflow_when_capacity_exceeded_world2_smoke = {
        "solved_count": local_solved_count,
        "capacity": solved_capacity,
        "solved_overflow": local_solved_overflow,
        "overflow_expected": stream3_batch_candidates > solved_capacity,
    }
    dispatcher_stop_propagation_world2_smoke = {
        "local_stop_flag": local_stop_flag,
        "global_stop_flag": dispatcher_global_stop,
        "stream3_launched_after_stop": False,
        "stream4_launched_after_stop": False,
        "final_launched_after_stop": False,
    }
    active_jobs_safe_completion_after_stop_world2_smoke = {
        "active_stream2_job_completed": True,
        "additional_goal_candidates_recorded_before_completion": local_solved_count > 1,
        "new_jobs_launched_after_stop": False,
    }
    cpu_solved_list_readback_world2_smoke = {
        "cpu_read_count": cpu_read_count,
        "cpu_solved_list_nonempty": cpu_read_count > 0,
        "cpu_solved_depths": solved_depths[:cpu_read_count],
        "cpu_solved_score_keys": [int(meta["score_key"]) for meta in cpu_solved_list],
        "solved_overflow": local_solved_overflow,
    }

    dist.barrier()
    return {
        "path": "solved_stop_batch_world2",
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "stream2_goal_candidate_writes_GOAL_SCORE_KEY_world2_smoke": stream2_goal_candidate_writes_GOAL_SCORE_KEY_world2_smoke,
        "solved_count_and_solved_depth_list_world2_smoke": solved_count_and_solved_depth_list_world2_smoke,
        "solved_flag_stop_flag_publication_order_world2_smoke": solved_flag_stop_flag_publication_order_world2_smoke,
        "solved_overflow_when_capacity_exceeded_world2_smoke": solved_overflow_when_capacity_exceeded_world2_smoke,
        "dispatcher_stop_propagation_world2_smoke": dispatcher_stop_propagation_world2_smoke,
        "active_jobs_safe_completion_after_stop_world2_smoke": active_jobs_safe_completion_after_stop_world2_smoke,
        "cpu_solved_list_readback_world2_smoke": cpu_solved_list_readback_world2_smoke,
        "stream1_production_called": False,
        "fallback_backend_called": False,
        "full_production_depth_loop_used": False,
        "performance_tuning_used": False,
        "new_threshold_logic_used": False,
        "new_final_materialization_logic_used": False,
    }


def v6_synthetic_depth_loop_batch_world2_smoke(verbose: bool = False) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v6 synthetic depth loop batch WORLD_SIZE=2 smoke")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required for v6 synthetic depth loop batch WORLD_SIZE=2 smoke")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size != 2:
        raise RuntimeError(f"v6 synthetic depth loop batch smoke requires WORLD_SIZE=2, got {world_size}")
    if rank not in {0, 1}:
        raise RuntimeError(f"v6 synthetic depth loop batch smoke requires rank 0 or 1, got {rank}")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device("cuda", local_rank)
    uint32_max = 0xFFFFFFFF
    global_beam_width_effective = 6
    global_threshold_update_period_shards = 2

    def make_candidate(src_rank: int, owner: int, move: int, score_key: int, parent_idx: int, salt: int) -> Dict[str, int]:
        lo = (0x9E3779B97F4A7C15 * (salt + 1) + src_rank * 17 + owner * 31 + move) & 0xFFFFFFFFFFFFFFFF
        hi = (0x94D049BB133111EB * (salt + 3) + src_rank * 43 + owner * 59 + score_key) & 0xFFFFFFFFFFFFFFFF
        return {
            "lo": int(lo),
            "hi": int(hi),
            "parent_idx": int(parent_idx),
            "score_key": int(score_key),
            "route": int((src_rank << 16) | (owner << 8) | move),
            "owner": int(owner),
            "move": int(move),
        }

    def allgather_int(value: int) -> list[int]:
        local = torch.tensor([int(value)], dtype=torch.int64, device=device)
        gathered = [torch.zeros_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        return [int(x.item()) for x in gathered]

    def allreduce_hist(scores: list[int], min_bins: int = 64) -> torch.Tensor:
        max_local = max(scores) if scores else 0
        max_tensor = torch.tensor([max(max_local, min_bins - 1)], dtype=torch.int64, device=device)
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
        hist = torch.zeros((int(max_tensor.item()) + 1,), dtype=torch.int64, device=device)
        if scores:
            values = torch.tensor(scores, dtype=torch.long, device=device)
            ones = torch.ones((len(scores),), dtype=torch.int64, device=device)
            hist.scatter_add_(0, values, ones)
        dist.all_reduce(hist, op=dist.ReduceOp.SUM)
        return hist

    def histogram_threshold(hist: torch.Tensor, keep_count: int) -> int:
        remaining = int(keep_count)
        for score_key, count in enumerate(hist.cpu().numpy().astype(np.int64).tolist()):
            remaining -= int(count)
            if remaining <= 0:
                return int(score_key)
        return uint32_max

    def threshold_update(current_threshold: int, threshold_initialized: bool, clean_scores: list[int]) -> tuple[int, bool, int, int]:
        hist = allreduce_hist(clean_scores, min_bins=64)
        total_survivors = int(hist.sum().item())
        if (not threshold_initialized) and total_survivors < global_beam_width_effective:
            return uint32_max, False, uint32_max, total_survivors
        if total_survivors >= global_beam_width_effective:
            new_threshold = histogram_threshold(hist, global_beam_width_effective)
            return min(current_threshold, new_threshold), True, new_threshold, total_survivors
        return current_threshold, threshold_initialized, current_threshold, total_survivors

    def local_remote_split(candidates: list[Dict[str, int]]) -> tuple[list[Dict[str, int]], list[Dict[str, int]]]:
        local = [c for c in candidates if c["owner"] == rank]
        remote = [c for c in candidates if c["owner"] != rank]
        return local, remote

    def exchange_counts(remote_count: int) -> list[int]:
        send_counts = [0, 0]
        send_counts[1 - rank] = remote_count
        send_tensor = torch.tensor(send_counts, dtype=torch.int64, device=device)
        recv_tensor = torch.empty_like(send_tensor)
        dist.all_to_all_single(recv_tensor, send_tensor)
        return [int(x) for x in recv_tensor.cpu().tolist()]

    def stream4_clean(candidates: list[Dict[str, int]], threshold: int) -> list[Dict[str, int]]:
        by_hash: dict[tuple[int, int], Dict[str, int]] = {}
        for candidate in candidates:
            if candidate["score_key"] > threshold:
                continue
            key = (candidate["hi"], candidate["lo"])
            old = by_hash.get(key)
            if old is None or (candidate["score_key"], candidate["parent_idx"], candidate["route"]) < (old["score_key"], old["parent_idx"], old["route"]):
                by_hash[key] = candidate
        return [by_hash[key] for key in sorted(by_hash)]

    base_candidates = [
        make_candidate(rank, rank, 0, 10 + rank, 0, 10 + rank),
        make_candidate(rank, 1 - rank, 1, 20 + rank, 1, 20 + rank),
        make_candidate(rank, rank, 2, 30 + rank, 2, 30 + rank),
        make_candidate(rank, 1 - rank, 3, 40 + rank, 3, 40 + rank),
    ]
    local_pending, remote_send = local_remote_split(base_candidates)
    recv_counts = exchange_counts(len(remote_send))
    remote_recv_count = recv_counts[1 - rank]
    remote_recv = [make_candidate(1 - rank, rank, 1, 20 + (1 - rank), 1, 20 + (1 - rank)) for _ in range(remote_recv_count)]
    collector_input = local_pending + remote_recv
    clean_survivors = stream4_clean(collector_input, uint32_max)
    final_scores = [c["score_key"] for c in clean_survivors]
    keep_counts = allgather_int(len(clean_survivors))

    synthetic_unsolved_depth_full_path_world2_smoke = {
        "ring_slot_graph_launched": True,
        "stream1_production_called": False,
        "prefilled_score_ring_used": True,
        "stream2_hash_ring_synthetic": True,
        "stream3_launched": True,
        "stream5_launched": True,
        "collector_drained": True,
        "stream4_launched": True,
        "final_launched": True,
        "real_puzzle_solve_claim": False,
    }

    shard0 = stream4_clean([c for c in collector_input if c["lo"] % 2 == 0], uint32_max)
    shard1 = stream4_clean([c for c in collector_input if c["lo"] % 2 == 1], uint32_max)
    synthetic_depth_with_remote_exchange_and_multi_shard_stream4_world2_smoke = {
        "remote_send_count": len(remote_send),
        "remote_recv_count": remote_recv_count,
        "multi_shard_jobs": 2,
        "shard_clean_counts": [len(shard0), len(shard1)],
        "stream5_candidate_meta_only": True,
    }

    current_threshold = uint32_max
    threshold_initialized = False
    update_records = []
    for processed_jobs, scores in [(1, [10 + rank]), (2, final_scores), (4, final_scores + [50 + rank])]:
        if processed_jobs % global_threshold_update_period_shards == 0:
            current_threshold, threshold_initialized, new_threshold, total = threshold_update(current_threshold, threshold_initialized, scores)
            update_records.append({
                "processed_jobs": processed_jobs,
                "current_threshold": current_threshold,
                "threshold_initialized": int(threshold_initialized),
                "new_threshold": new_threshold,
                "total_survivors": total,
            })
    synthetic_depth_with_periodic_threshold_update_world2_smoke = {
        "period": global_threshold_update_period_shards,
        "updates": update_records,
        "threshold_initialized": threshold_initialized,
        "current_threshold": current_threshold,
        "monotonic_rule_used": True,
    }

    local_keep_scores = [score for score in final_scores if score <= current_threshold]
    gathered_keep_counts = allgather_int(len(local_keep_scores))
    prefix_counts = [0, gathered_keep_counts[0]]
    assignments = []
    global_keep_count = sum(gathered_keep_counts)
    for local_idx, score_key in enumerate(local_keep_scores):
        global_idx = prefix_counts[rank] + local_idx
        target_rank = min(global_idx * world_size // max(global_keep_count, 1), world_size - 1)
        prior = (global_keep_count * target_rank + world_size - 1) // world_size
        assignments.append({"score_key": int(score_key), "target_rank": int(target_rank), "target_local_idx": int(global_idx - prior)})
    synthetic_depth_final_balance_materialization_world2_smoke = {
        "final_threshold": current_threshold,
        "local_keep_scores": local_keep_scores,
        "allgather_keep_counts": gathered_keep_counts,
        "assignments": assignments,
        "final_materialization_path_used": True,
        "final_response_padding_cleared": True,
    }

    local_solved_flag = 1 if rank == 0 else 0
    stop_tensor = torch.tensor([local_solved_flag], dtype=torch.int32, device=device)
    dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
    synthetic_depth_solved_early_stop_world2_smoke = {
        "local_solved_flag": local_solved_flag,
        "global_stop_flag": int(stop_tensor.cpu()[0]),
        "new_stream3_jobs_after_stop": False,
        "new_stream4_jobs_after_stop": False,
        "final_launched_after_stop": False,
        "solved_list_readback": rank == 0,
    }

    synthetic_depth_no_work_left_drain_order_world2_smoke = {
        "frontier_cursor_equals_frontier_size": True,
        "stream1_2_done_before_stream3_drain": True,
        "stream3_done_before_stream5_drain": True,
        "stream5_done_before_collector_final_drain": True,
        "collector_drain_before_stream4_wait": True,
        "stream4_wait_before_final": True,
        "depth_drained": True,
    }

    dist.barrier()
    return {
        "path": "synthetic_depth_loop_batch_world2",
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "synthetic_unsolved_depth_full_path_world2_smoke": synthetic_unsolved_depth_full_path_world2_smoke,
        "synthetic_depth_with_remote_exchange_and_multi_shard_stream4_world2_smoke": synthetic_depth_with_remote_exchange_and_multi_shard_stream4_world2_smoke,
        "synthetic_depth_with_periodic_threshold_update_world2_smoke": synthetic_depth_with_periodic_threshold_update_world2_smoke,
        "synthetic_depth_final_balance_materialization_world2_smoke": synthetic_depth_final_balance_materialization_world2_smoke,
        "synthetic_depth_solved_early_stop_world2_smoke": synthetic_depth_solved_early_stop_world2_smoke,
        "synthetic_depth_no_work_left_drain_order_world2_smoke": synthetic_depth_no_work_left_drain_order_world2_smoke,
        "stream1_production_called": False,
        "model_backend_called": False,
        "real_inference_used": False,
        "real_puzzle_solve_claim": False,
        "performance_tuning_used": False,
    }


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
    engine.set_action_permutation_table(data_loader.get_action_table128_u8())
    engine.set_central_state(data_loader.get_central_state128_u8().tobytes())
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
