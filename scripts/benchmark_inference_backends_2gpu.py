#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import beam_engine
import data_loader


def _init_cfg(backend: str) -> dict:
    os.environ["INFERENCE_BACKEND"] = backend
    if backend == "torchscript_ensemble":
        os.environ["ALLOW_TORCHSCRIPT_SCORER"] = "1"
    cfg = beam_engine.make_default_config()
    cfg["inference_backend"] = backend
    cfg["global_beam_width"] = int(os.environ.get("BENCH_GLOBAL_BEAM_WIDTH", "65536"))
    cfg["max_depth"] = 2
    return cfg


def _export_one_torchscript(cfg: dict) -> None:
    out_dir = PROJECT_DIR / "runtime" / "bench_fullbeamnice_scorer" / f"rank{cfg['rank']}"
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "scripts" / "export_fullbeamnice_scorer.py"),
        "--copies",
        "1",
        "--out-dir",
        str(out_dir),
    ]
    out = subprocess.check_output(cmd, cwd=str(PROJECT_DIR), text=True)
    paths = [line for line in out.splitlines() if line.startswith("TORCHSCRIPT_SCORER_PATHS=")]
    if not paths:
        raise RuntimeError("TorchScript scorer export did not print TORCHSCRIPT_SCORER_PATHS")
    cfg["torchscript_scorer_paths"] = paths[0].split("=", 1)[1]


def _prepare_frontier(buffers: dict[str, torch.Tensor], micro_size: int, device: torch.device) -> None:
    central = data_loader.get_central_state_u8()
    host = np.repeat(central.reshape(1, -1), micro_size, axis=0).astype(np.uint8, copy=True)
    # Deterministic non-constant perturbation keeps embedding path realistic while preserving valid uint8 tokens.
    offsets = (np.arange(micro_size, dtype=np.uint16)[:, None] + np.arange(host.shape[1], dtype=np.uint16)[None, :]) % 120
    host = ((host.astype(np.uint16) + offsets) % 120).astype(np.uint8)
    buffers["beam_current"][:micro_size].copy_(torch.from_numpy(host).to(device=device))
    buffers["current_active_flags"][:micro_size].fill_(1)
    torch.cuda.synchronize(device)


def _summarize(backend: str, rank: int, world_size: int, micro_size: int, lanes: int, timings: list[float]) -> dict:
    timings = [float(x) for x in timings]
    mean_ms = statistics.fmean(timings)
    sorted_ms = sorted(timings)
    p50_ms = sorted_ms[len(sorted_ms) // 2]
    p95_ms = sorted_ms[min(len(sorted_ms) - 1, int(len(sorted_ms) * 0.95))]
    states_per_iter = micro_size * lanes
    return {
        "backend": backend,
        "rank": rank,
        "world_size": world_size,
        "batch_per_lane": micro_size,
        "inference_parallelism": lanes,
        "states_per_iter": states_per_iter,
        "iterations": len(timings),
        "mean_ms": round(mean_ms, 4),
        "p50_ms": round(p50_ms, 4),
        "p95_ms": round(p95_ms, 4),
        "states_per_sec": round((states_per_iter * 1000.0) / mean_ms, 2) if mean_ms > 0 else 0.0,
    }


def _run_backend(backend: str, ext, device: torch.device, micro_size: int, repeats: int, warmup: int) -> dict:
    cfg = _init_cfg(backend)
    if backend == "torchscript_ensemble":
        _export_one_torchscript(cfg)
    buffers = beam_engine.allocate_buffers(ext, cfg)
    _prepare_frontier(buffers, micro_size, device)
    engine = beam_engine.configure_engine(ext, cfg, buffers)
    engine.enable_cuda_graphs(False)
    timings = [float(x) for x in engine.benchmark_inference(micro_size, repeats, warmup)]
    torch.cuda.synchronize(device)
    return _summarize(backend, int(cfg["rank"]), int(cfg["world_size"]), micro_size, int(cfg["inference_parallelism"]), timings)


def main() -> None:
    os.environ.setdefault("USE_CUDA_GRAPHS", "0")
    os.environ["GLOBAL_BEAM_WIDTH"] = os.environ.get("BENCH_GLOBAL_BEAM_WIDTH", "65536")
    os.environ.setdefault("B_MICRO", os.environ.get("BENCH_B_MICRO", os.environ.get("B_MICRO", "8192")))
    os.environ.setdefault("INFERENCE_PARALLELISM", os.environ.get("BENCH_INFERENCE_PARALLELISM", os.environ.get("INFERENCE_PARALLELISM", "2")))
    os.environ.setdefault("HISTORY_BACKEND", "cpu")
    os.environ.setdefault("CPU_HISTORY_CHECKPOINT", "0")
    os.environ.setdefault("BEAM_DEBUG", "0")

    cfg0 = beam_engine.make_default_config()
    beam_engine.init_distributed_if_needed(cfg0)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    ext = beam_engine.build_extension(verbose=os.environ.get("BUILD_VERBOSE", "0") == "1")
    micro_size = min(int(os.environ.get("BENCH_MICRO", str(cfg0["b_micro"]))), int(cfg0["b_micro"]))
    repeats = int(os.environ.get("BENCH_ITERS", "50"))
    warmup = int(os.environ.get("BENCH_WARMUP", "10"))
    backends = [x.strip() for x in os.environ.get("BENCH_INFERENCE_BACKENDS", "torchscript_ensemble,fullbeamnice_static").split(",") if x.strip()]

    local_results = []
    for backend in backends:
        local_results.append(_run_backend(backend, ext, device, micro_size, repeats, warmup))

    gathered = [None for _ in range(dist.get_world_size())] if dist.is_available() and dist.is_initialized() else None
    if gathered is not None:
        dist.all_gather_object(gathered, local_results)
        flat = [item for rank_items in gathered for item in rank_items]
    else:
        flat = local_results

    if int(cfg0["rank"]) == 0:
        by_backend = {}
        for item in flat:
            by_backend.setdefault(item["backend"], []).append(item)
        summary = {"entity_id": "stream1_inference_benchmark", "type": "benchmark", "results": flat}
        if "torchscript_ensemble" in by_backend and "fullbeamnice_static" in by_backend:
            ts_mean = statistics.fmean(x["mean_ms"] for x in by_backend["torchscript_ensemble"])
            static_mean = statistics.fmean(x["mean_ms"] for x in by_backend["fullbeamnice_static"])
            summary["speedup_static_vs_torchscript"] = round(ts_mean / static_mean, 4) if static_mean > 0 else None
        print("STREAM1_BENCHMARK " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
