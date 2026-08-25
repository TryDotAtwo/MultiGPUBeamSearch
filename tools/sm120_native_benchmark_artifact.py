#!/usr/bin/env python3
"""Convert completed native Cube4 runner logs into fail-closed tuner evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FATAL_MARKERS = (
    "CUDA error", "out of memory", "OutOfMemory", "overflow=1", "fatal=1",
    "SIGABRT", "SIGSEGV", "illegal memory access", "ChildFailedError",
)


def _last_value(text: str, key: str) -> str:
    matches = re.findall(rf"(?:^|\n)(?:\[[^\]]+\]:)?{re.escape(key)}=([^\r\n]+)", text)
    if not matches:
        raise ValueError(f"native runner log is missing {key}")
    return matches[-1].strip()


def parse_native_runner_log(
    text: str,
    *,
    name: str,
    expected_beam: int = 2**25,
    expected_depth: int = 8,
    expected_stream3_jobs: int = 391,
    expected_world_size: int = 1,
) -> dict[str, Any]:
    failures = [marker for marker in FATAL_MARKERS if marker.lower() in text.lower()]
    if failures:
        raise ValueError("native runner log contains fatal markers: " + ", ".join(failures))
    if int(_last_value(text, "cuda_device_sm")) != 120:
        raise ValueError("native benchmark was not produced on sm_120")
    if _last_value(text, "stream1_backend") != "piece_transformer":
        raise ValueError("native benchmark must use piece_transformer Stream1")
    if int(_last_value(text, "GLOBAL_BEAM_WIDTH_EFFECTIVE")) != expected_beam:
        raise ValueError("native benchmark effective beam does not match the acceptance workload")
    world_size = int(_last_value(text, "WORLD_SIZE"))
    if world_size != expected_world_size or world_size <= 0:
        raise ValueError("native benchmark world size does not match the acceptance workload")
    if int(_last_value(text, "B_MICRO")) != 3584:
        raise ValueError("native benchmark outer B_MICRO must be 3584")
    if int(_last_value(text, "stream1_transformer_micro")) != 896:
        raise ValueError("native benchmark Transformer microbatch must be 896")
    if _last_value(text, "stream1_transformer_activation") != "relu":
        raise ValueError("native Cube4 benchmark must preserve manifest ReLU activation")
    if int(_last_value(text, "STREAM1_CONCURRENCY")) < 2:
        raise ValueError("native benchmark must keep at least two inference lanes")
    dims = re.findall(r"stream1_transformer_dims[^\n]*output_dim=(\d+)", text)
    if not dims or int(dims[-1]) != 24:
        raise ValueError("native benchmark must use output_dim=24")
    depth_rows = re.findall(
        r"depth_done=(\d+)\s+depth_sec=([0-9.eE+-]+)[^\n]*?"
        r"stream3_jobs=(\d+)[^\n]*?next_frontier_size=(\d+)",
        text,
    )
    matches = [row for row in depth_rows if int(row[0]) == expected_depth]
    if len(matches) != 1:
        raise ValueError(f"native runner log must contain exactly one depth_done={expected_depth}")
    _, seconds, stream3_jobs, frontier_size = matches[0]
    if int(stream3_jobs) != expected_stream3_jobs:
        raise ValueError("native benchmark Stream3 job count is not comparable")
    expected_frontier = expected_beam // expected_world_size
    if expected_beam % expected_world_size != 0 or int(frontier_size) != expected_frontier:
        raise ValueError("per-rank frontier does not match effective beam and world size")
    backend = _last_value(text, "stream1_transformer_fp16_gemm_backend")
    if backend not in {"cutlass", "cublaslt"}:
        raise ValueError("native runner log contains an unsupported FP16 GEMM backend")
    return {
        "name": name,
        "latency_ms": float(seconds) * 1000.0,
        "depth": expected_depth,
        "effective_beam": expected_beam,
        "per_rank_frontier": int(frontier_size),
        "world_size": world_size,
        "stream3_jobs": int(stream3_jobs),
        "device_name": _last_value(text, "cuda_device_name"),
        "native_execution": {
            "fp16_gemm_backend": backend,
            "target_sm": 120,
            "workspace_bytes": 0,
        },
        "model_activation": "relu",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True,
                        help="NAME=PATH to a completed combined runner log")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for item in args.candidate:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("--candidate must be NAME=PATH")
        path = Path(raw_path)
        rows.append(parse_native_runner_log(path.read_text(encoding="utf-8"), name=name))
    payload = {"schema_version": 1, "candidates": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
