from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

GITHUB_REPO_URL = "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git"
GITHUB_REF = "stream1-transformer-sm75-gemm-v21-fe3cb95"
EXPECTED_COMMIT_PREFIX = "fe3cb95"

WORK_DIR = Path("/kaggle/working")
TMP_DIR = Path("/tmp")
REPO_DIR = TMP_DIR / "beam_solver_pipeline_gate"
CUTLASS_DIR = TMP_DIR / "cutlass"
BUILD_DIR = TMP_DIR / "beam_build_pipeline_gate"
LOG_DIR = WORK_DIR / "stream_pipeline_gate_logs"
ROWS_CSV = WORK_DIR / "stream_pipeline_gate_rows.csv"
SUMMARY_JSON = WORK_DIR / "stream_pipeline_gate_summary.json"
SUMMARY_MD = WORK_DIR / "stream_pipeline_gate_summary.md"

CUDA_ARCHITECTURES = "75"
GPUS = [0, 1]
REPEATS = 20
B_MICRO = 384
TRANSFORMER_MICRO = 384
CONCURRENCY = 1
RING_SLOTS = 1
RINGS = 8
WINDOW = 8
SHARD_COUNT = 4
SHARD_CAPACITY = 1_048_576
STREAM4_BATCH = 262_144
STREAM4_TRIGGER = 524_288
GLOBAL_SPILL = 1_048_576
FINAL_CHUNK = 98_304
FINAL_EXCHANGE_SCALE_PPM = 2_000_000
PUZZLE_ID = 990

COMMON_TRANSFORMER_ENV = {
    "BEAM_STREAM1_TRANSFORMER_BLOCK51": "1",
    "BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY": "1",
    "BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION": "0",
}
SELECTED_POLICY = {
    "BEAM_STREAM1_TRANSFORMER_QKV_POLICY": "m128n128",
    "BEAM_STREAM1_TRANSFORMER_QKV_SWIZZLE": "8",
    "BEAM_STREAM1_TRANSFORMER_FF1_POLICY": "m128n128w64n32",
    "BEAM_STREAM1_TRANSFORMER_FF1_STAGES": "2",
    "BEAM_STREAM1_TRANSFORMER_ATTN_OUT_POLICY": "m128n128",
    "BEAM_STREAM1_TRANSFORMER_ATTN_OUT_EPILOGUE": "fused",
    "BEAM_STREAM1_TRANSFORMER_ATTN_OUT_SWIZZLE": "2",
    "BEAM_STREAM1_TRANSFORMER_FF2_POLICY": "m128n128",
    "BEAM_STREAM1_TRANSFORMER_FF2_EPILOGUE": "fused",
    "BEAM_STREAM1_TRANSFORMER_FF2_SWIZZLE": "2",
}
POLICIES = [
    ("sm75_selected", SELECTED_POLICY.copy()),
    ("sm75_selected_attention", {
        **SELECTED_POLICY,
        "BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION": "1",
        "BEAM_STREAM1_TRANSFORMER_CLS_ATTENTION_POLICY": "q32k64",
        "BEAM_STREAM1_TRANSFORMER_ATTENTION_MAX_K_POLICY": "exact32",
    }),
]
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


def parse_pairs(text):
    values = {}
    for part in text.split():
        if "=" in part:
            key, value = part.split("=", 1)
            values[key] = value
    return values


def cleanup_path(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path) if path.is_dir() else path.unlink()


def prepare_repo_and_build():
    for path in (REPO_DIR, BUILD_DIR):
        cleanup_path(path)
    run_capture(["nvidia-smi"], check=False)
    run_checked(["git", "clone", "--branch", GITHUB_REF, "--depth", "1", GITHUB_REPO_URL, REPO_DIR])
    actual = run_capture(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR).stdout.strip()
    if not actual.startswith(EXPECTED_COMMIT_PREFIX):
        raise RuntimeError(f"expected commit {EXPECTED_COMMIT_PREFIX}, got {actual}")
    if not (CUTLASS_DIR / "include").exists():
        cleanup_path(CUTLASS_DIR)
        run_checked(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/cutlass.git", CUTLASS_DIR])
    run_checked([
        "cmake", "-S", REPO_DIR, "-B", BUILD_DIR, "-GNinja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DBEAM_CUDA_ARCHITECTURES={CUDA_ARCHITECTURES}",
        f"-DCUTLASS_DIR={CUTLASS_DIR}",
    ], cwd=REPO_DIR)
    run_checked(["cmake", "--build", BUILD_DIR, "--target", "stream_pipeline_benchmark", "contract_tests", "-j", "2"])
    run_checked([BUILD_DIR / "contract_tests"], cwd=REPO_DIR)
    return actual


def run_one(gpu, label, repeat, policy_env):
    log_path = LOG_DIR / f"gpu{gpu}_{label}_r{repeat}.log"
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "BEAM_CUDA_DEVICE": "0",
        "BEAM_PIPELINE_BENCH_MODE": "stream123",
        "BEAM_RING_GRAPH_EXECS_PER_LANE": str(WINDOW),
        "BEAM_B_MICRO": str(B_MICRO),
        "BEAM_STREAM1_TRANSFORMER_MICRO": str(TRANSFORMER_MICRO),
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
    })
    env.update(COMMON_TRANSFORMER_ENV)
    env.update(policy_env)
    start = time.time()
    result = run_capture([BUILD_DIR / "stream_pipeline_benchmark", str(PUZZLE_ID)], cwd=REPO_DIR, env=env, check=False)
    elapsed = time.time() - start
    log_path.write_text(result.stdout, encoding="utf-8")
    match = None
    for line in result.stdout.splitlines():
        candidate = row_pattern.search(line)
        if candidate:
            match = candidate
    row = {
        "gpu": str(gpu), "label": label, "repeat": str(repeat),
        "return_code": str(result.returncode), "elapsed_wall_sec": f"{elapsed:.3f}", "log": str(log_path),
    }
    if match:
        row.update(parse_pairs(match.group("pairs")))
    row.setdefault("status", "OK" if result.returncode == 0 else "FAILED")
    if result.returncode != 0 or row.get("status") != "OK":
        raise RuntimeError(f"pipeline row failed gpu={gpu} label={label} repeat={repeat} log={log_path}")
    print("PIPELINE_GATE_ROW " + " ".join(f"{key}={value}" for key, value in row.items()), flush=True)
    return row


def write_outputs(commit, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    results = []
    for gpu in GPUS:
        baseline = sorted((row for row in rows if row["gpu"] == str(gpu) and row["label"] == "sm75_selected"), key=lambda row: int(row["repeat"]))
        selected = sorted((row for row in rows if row["gpu"] == str(gpu) and row["label"] == "sm75_selected_attention"), key=lambda row: int(row["repeat"]))
        if len(baseline) != REPEATS or len(selected) != REPEATS:
            raise RuntimeError(f"missing pipeline repeats on GPU {gpu}")
        baseline_cps = [float(row["candidates_per_sec"]) for row in baseline]
        selected_cps = [float(row["candidates_per_sec"]) for row in selected]
        results.append({
            "gpu": gpu,
            "baseline_median_cps": statistics.median(baseline_cps),
            "selected_median_cps": statistics.median(selected_cps),
            "speedup": statistics.median(selected_cps) / statistics.median(baseline_cps),
            "paired_wins": sum(candidate > reference for candidate, reference in zip(selected_cps, baseline_cps)),
            "pairs": REPEATS,
        })
    summary = {"commit": commit, "github_ref": GITHUB_REF, "config": {"b_micro": B_MICRO, "transformer_micro": TRANSFORMER_MICRO, "concurrency": CONCURRENCY, "ring_slots": RING_SLOTS, "rings": RINGS, "window": WINDOW}, "results": results}
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# SM75 attention Stream1->2->3 integration gate", "", f"- commit: `{commit}`", f"- ref: `{GITHUB_REF}`", "", "| GPU | baseline cand/s | selected cand/s | speedup | paired wins |", "|---:|---:|---:|---:|---:|"]
    for item in results:
        lines.append(f"| {item['gpu']} | {item['baseline_median_cps']:.1f} | {item['selected_median_cps']:.1f} | {item['speedup']:.4f}x | {item['paired_wins']}/{item['pairs']} |")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("PIPELINE_GATE_RESULTS", json.dumps(results), flush=True)


def main():
    commit = prepare_repo_and_build()
    rows = []
    for gpu in GPUS:
        for repeat in range(REPEATS):
            configs = POLICIES if repeat % 2 == 0 else list(reversed(POLICIES))
            for label, env in configs:
                rows.append(run_one(gpu, label, repeat, env))
    write_outputs(commit, rows)


if __name__ == "__main__":
    main()
