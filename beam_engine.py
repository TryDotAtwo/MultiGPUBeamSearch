"""
beam_engine.py

Python control-plane for the CUDA/C++ multi-GPU beam-search engine.
Data-plane remains GPU-resident: inference scores, candidate routing, dedup,
thresholding, pruning and compaction are executed inside the extension.
"""

from __future__ import annotations

import json
import os
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


def make_default_config() -> Dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    b_micro = int(os.environ.get("B_MICRO", "32768"))
    return {
        "world_size": world_size,
        "rank": rank,
        "global_beam_width": int(os.environ.get("GLOBAL_BEAM_WIDTH", str(1 << 16))),
        "fanout": 24,
        "state_size_bytes": 120,
        "b_micro": b_micro,
        "score_ring_depth": int(os.environ.get("SCORE_RING_DEPTH", "16")),
        "net_ring_depth": int(os.environ.get("NET_RING_DEPTH", "2")),
        "probe_limit": int(os.environ.get("PROBE_LIMIT", "64")),
        "bucket_cap_per_peer": int(os.environ.get(
            "BUCKET_CAP_PER_PEER",
            str(max(4096, (b_micro * 24 // max(world_size, 1)) * 2)),
        )),
        "inference_parallelism": int(os.environ.get("INFERENCE_PARALLELISM", "1")),
        "k_expand_tile": int(os.environ.get("K_EXPAND_TILE", "0")),
        "torchscript_scorer_paths": os.environ.get("TORCHSCRIPT_SCORER_PATHS", ""),
        "nn_score_scale": float(os.environ.get("NN_SCORE_SCALE", "1.0")),
        "nn_score_bias": float(os.environ.get("NN_SCORE_BIAS", "0.0")),
        "gamma": float(os.environ.get("GAMMA", "1.05")),
        "beta": float(os.environ.get("BETA", "1.15")),
        "hash_load_factor": float(os.environ.get("HASH_LOAD_FACTOR", "0.55")),
        "inference_backend": os.environ.get("INFERENCE_BACKEND", "central_hamming"),
        "max_depth": int(os.environ.get("MAX_DEPTH", "4")),
        "histogram_period_micro": int(os.environ.get("HISTOGRAM_PERIOD_MICRO", "4")),
        "cuda_graph_max_micro": int(os.environ.get("CUDA_GRAPH_MAX_MICRO", "512")),
        "history_backend": history_backend(),
        "cpu_history_checkpoint": _flag_enabled("CPU_HISTORY_CHECKPOINT"),
    }


def build_extension(verbose: bool = True):
    sources = [str(PROJECT_DIR / "beam_engine.cpp"), str(PROJECT_DIR / "beam_kernels.cu")]
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
    }
    for tensor in buffers.values():
        tensor.zero_()
    return buffers


def create_nccl_id(ext, cfg: Dict[str, Any]) -> bytes:
    if cfg["world_size"] <= 1:
        return b""
    obj = [None]
    if cfg["rank"] == 0:
        obj[0] = bytes(ext.get_nccl_unique_id())
    dist.broadcast_object_list(obj, src=0)
    return obj[0]


def configure_engine(ext, cfg: Dict[str, Any], buffers: Dict[str, torch.Tensor]):
    engine = ext.BeamEngine(cfg, buffers, cfg["inference_backend"])
    engine.set_action_permutation_table(data_loader.get_action_table_u8())
    engine.set_central_state(data_loader.get_central_state_u8().tobytes())
    if cfg["inference_backend"] == "torchscript_ensemble":
        paths = [p for p in str(cfg.get("torchscript_scorer_paths", "")).split(os.pathsep) if p]
        if not paths:
            raise ValueError("INFERENCE_BACKEND=torchscript_ensemble requires TORCHSCRIPT_SCORER_PATHS")
        if len(paths) > 1:
            print(f"[beam_engine] warning: {len(paths)} TorchScript paths were provided; one path is enough for shared-weight multi-lane inference")
        engine.load_torchscript_ensemble(paths)
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
