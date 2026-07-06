from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

GITHUB_REPO_URL = "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git"
GITHUB_BRANCH = "codex/stream1-piece-transformer"
EXPECTED_COMMIT_PREFIX = os.environ.get("EXPECTED_COMMIT_PREFIX", "")

WORK_DIR = Path("/kaggle/working")
TMP_DIR = Path("/tmp")
REPO_DIR = TMP_DIR / "beam_solver_pipeline_smoke"
CUTLASS_DIR = TMP_DIR / "cutlass"
BUILD_DIR = TMP_DIR / "beam_build_pipeline_smoke"
LOG_DIR = WORK_DIR / "stream_pipeline_smoke_logs"
ROWS_CSV = WORK_DIR / "stream_pipeline_smoke_rows.csv"
SUMMARY_JSON = WORK_DIR / "stream_pipeline_smoke_summary.json"
SUMMARY_MD = WORK_DIR / "stream_pipeline_smoke_summary.md"

CUDA_ARCHITECTURES = "75"
MODES = ["stream12", "stream123"]
WINDOWS = [16, 32]
B_MICRO = 512
CONCURRENCY = 2
RING_SLOTS = 1
RINGS = 4
SHARD_COUNT = 8
SHARD_CAPACITY = 65536
STREAM4_BATCH = 65536
STREAM4_TRIGGER = 65536
GLOBAL_SPILL = 65536
FINAL_CHUNK = 8192
FINAL_EXCHANGE_SCALE_PPM = 1000000
PUZZLE_ID = 990

LOG_DIR.mkdir(parents=True, exist_ok=True)

row_pattern = re.compile(r"stream_pipeline_benchmark\s+(?P<pairs>.+)$")


def run_checked(cmd, cwd=None, env=None):
    cmd = [str(part) for part in cmd]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def run_capture(cmd, cwd=None, env=None, check=True):
    cmd = [str(part) for part in cmd]
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end="", flush=True)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout)
    return result


def parse_pairs(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for part in text.split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value
    return values


def cleanup_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def disk_line(path: str) -> str:
    usage = shutil.disk_usage(path)
    return f"{path}: free={usage.free} total={usage.total}"


def preflight() -> None:
    print("PIPELINE_SMOKE_GITHUB_BRANCH=", GITHUB_BRANCH, flush=True)
    print("PIPELINE_SMOKE_EXPECTED_COMMIT_PREFIX=", EXPECTED_COMMIT_PREFIX, flush=True)
    print("disk_tmp=", disk_line("/tmp"), flush=True)
    print("disk_working=", disk_line("/kaggle/working"), flush=True)
    run_capture(["nvidia-smi"], check=False)
    import torch

    gpu_count = torch.cuda.device_count()
    print("torch_version=", torch.__version__, flush=True)
    print("torch_cuda_device_count=", gpu_count, flush=True)
    for index in range(gpu_count):
        print(f"torch_cuda_device_{index}={torch.cuda.get_device_name(index)}", flush=True)
    if gpu_count < 1:
        raise RuntimeError("pipeline smoke requires at least one CUDA device")


def prepare_repo_and_build() -> str:
    for path in (REPO_DIR, BUILD_DIR):
        cleanup_path(path)
    run_checked(["git", "clone", "--branch", GITHUB_BRANCH, "--depth", "1", GITHUB_REPO_URL, REPO_DIR])
    actual = run_capture(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR).stdout.strip()
    print("checked_out_commit=", actual, flush=True)
    if EXPECTED_COMMIT_PREFIX and not actual.startswith(EXPECTED_COMMIT_PREFIX):
        raise RuntimeError(f"expected commit prefix {EXPECTED_COMMIT_PREFIX}, got {actual}")

    if not (CUTLASS_DIR / "include").exists():
        cleanup_path(CUTLASS_DIR)
        run_checked(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/cutlass.git", CUTLASS_DIR])

    run_checked(
        [
            "cmake",
            "-S",
            REPO_DIR,
            "-B",
            BUILD_DIR,
            "-GNinja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DBEAM_CUDA_ARCHITECTURES={CUDA_ARCHITECTURES}",
            f"-DCUTLASS_DIR={CUTLASS_DIR}",
        ],
        cwd=REPO_DIR,
    )
    run_checked(["cmake", "--build", BUILD_DIR, "--target", "stream_pipeline_benchmark", "contract_tests", "-j", "2"])
    run_checked([BUILD_DIR / "contract_tests"])
    return actual


def run_smoke(mode: str, window: int) -> dict[str, str]:
    log_path = LOG_DIR / f"pipeline_{mode}_w{window}.log"
    env = os.environ.copy()
    env.update(
        {
            "BEAM_PIPELINE_BENCH_MODE": mode,
            "BEAM_RING_GRAPH_EXECS_PER_LANE": str(window),
            "BEAM_B_MICRO": str(B_MICRO),
            "BEAM_STREAM1_CONCURRENCY": str(CONCURRENCY),
            "BEAM_STREAM3_RING_SLOTS": str(RING_SLOTS),
            "BEAM_PIPELINE_SMOKE_RINGS": str(RINGS),
            "BEAM_SHARD_COUNT": str(SHARD_COUNT),
            "BEAM_SHARD_BUFFER_COUNT": "2",
            "BEAM_STREAM4_BATCH_CANDIDATES": str(STREAM4_BATCH),
            "STREAM4_BATCH_CANDIDATES": str(STREAM4_BATCH),
            "BEAM_STREAM4_TRIGGER_CANDIDATES": str(STREAM4_TRIGGER),
            "STREAM4_TRIGGER_CANDIDATES": str(STREAM4_TRIGGER),
            "BEAM_GLOBAL_SPILL_CAPACITY": str(GLOBAL_SPILL),
            "BEAM_SHARD_CAPACITY_CANDIDATES": str(SHARD_CAPACITY),
            "BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES": str(FINAL_CHUNK),
            "BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM": str(FINAL_EXCHANGE_SCALE_PPM),
        }
    )
    start = time.time()
    result = run_capture([BUILD_DIR / "stream_pipeline_benchmark", str(PUZZLE_ID)], cwd=REPO_DIR, env=env, check=False)
    elapsed = time.time() - start
    log_path.write_text(result.stdout, encoding="utf-8")
    match = None
    for line in result.stdout.splitlines():
        row_match = row_pattern.search(line)
        if row_match:
            match = row_match
    row: dict[str, str] = {
        "mode": mode,
        "window": str(window),
        "return_code": str(result.returncode),
        "elapsed_wall_sec": f"{elapsed:.3f}",
        "log": str(log_path),
    }
    if match:
        row.update(parse_pairs(match.group("pairs")))
    row.setdefault("status", "OK" if result.returncode == 0 else "FAILED")
    if result.returncode != 0:
        row["status"] = "FAILED"
    print("PIPELINE_SMOKE_ROW " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    return row


def write_outputs(commit: str, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "mode",
        "window",
        "return_code",
        "status",
        "b_micro",
        "concurrency",
        "ring_slots",
        "stream3_batch",
        "graph_window_jobs",
        "physical_jobs",
        "frontier_size",
        "ring_slot_jobs",
        "stream3_jobs",
        "stream4_jobs",
        "candidates",
        "depth_like_ms",
        "candidates_per_sec",
        "shard_capacity",
        "allocation_bytes",
        "elapsed_wall_sec",
        "log",
    ]
    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    failures = [row for row in rows if row.get("return_code") != "0" or row.get("status") != "OK"]
    summary = {
        "commit": commit,
        "rows_csv": str(ROWS_CSV),
        "summary_md": str(SUMMARY_MD),
        "modes": MODES,
        "windows": WINDOWS,
        "b_micro": B_MICRO,
        "concurrency": CONCURRENCY,
        "failures": failures,
        "rows": rows,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Stream Pipeline Smoke",
        "",
        f"- commit: `{commit}`",
        f"- b_micro: `{B_MICRO}`",
        f"- concurrency: `{CONCURRENCY}`",
        f"- failures: `{len(failures)}`",
        "",
        "| mode | window | status | cand/s | depth_like_ms | ring_jobs | stream3_jobs | log |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {window} | {status} | {candidates_per_sec} | {depth_like_ms} | {ring_slot_jobs} | {stream3_jobs} | `{log}` |".format(
                mode=row.get("mode", ""),
                window=row.get("window", ""),
                status=row.get("status", ""),
                candidates_per_sec=row.get("candidates_per_sec", ""),
                depth_like_ms=row.get("depth_like_ms", ""),
                ring_slot_jobs=row.get("ring_slot_jobs", ""),
                stream3_jobs=row.get("stream3_jobs", ""),
                log=row.get("log", ""),
            )
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    preflight()
    commit = prepare_repo_and_build()
    rows = []
    for mode in MODES:
        for window in WINDOWS:
            rows.append(run_smoke(mode, window))
    write_outputs(commit, rows)
    failures = [row for row in rows if row.get("return_code") != "0" or row.get("status") != "OK"]
    print("PIPELINE_SMOKE_SUMMARY_JSON=", SUMMARY_JSON, flush=True)
    print("PIPELINE_SMOKE_ROWS_CSV=", ROWS_CSV, flush=True)
    print("PIPELINE_SMOKE_FAILURES=", len(failures), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
