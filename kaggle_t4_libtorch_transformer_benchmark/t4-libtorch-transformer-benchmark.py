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
KAGGLE_MODEL_SOURCE = "vladkuznetsov266/megaminx-qtransformer-1782210824/PyTorch/default/1"
MODEL_SOURCE_SLUG = "megaminx-qtransformer-1782210824"
MODEL_INPUT_ROOTS = [
    "/kaggle/input/megaminx-qtransformer-1782210824/PyTorch/default/1",
    "/kaggle/input/megaminx-qtransformer-1782210824/pytorch/default/1",
]
WORK_DIR = Path("/kaggle/working")
TMP_DIR = Path("/tmp")
REPO_DIR = TMP_DIR / "beam_solver_libtorch_transformer_bench"
CUTLASS_DIR = TMP_DIR / "cutlass"
BUILD_DIR = TMP_DIR / "beam_build_libtorch_transformer_bench"
WEIGHT_OUT_DIR = WORK_DIR / "stream1_transformer_weights_fp16"
BENCH_LOG_DIR = WORK_DIR / "stream1_libtorch_transformer_logs"
ROWS_CSV = WORK_DIR / "stream1_libtorch_transformer_rows.csv"
SUMMARY_JSON = WORK_DIR / "stream1_libtorch_transformer_summary.json"
CUDA_ARCHITECTURES = "75"
BENCH_GPUS = [0, 1]
BENCH_BATCHES = [128, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 1536, 2048]
BENCH_WARMUP = 20
BENCH_ITERS = 100

BENCH_LOG_DIR.mkdir(parents=True, exist_ok=True)

row_pattern = re.compile(
    r"stream1_transformer_libtorch_micro\s+"
    r"batch=(?P<batch>\d+)\s+"
    r"iters=(?P<iters>\d+)\s+"
    r"elapsed_ms=(?P<elapsed>[0-9.eE+-]+)\s+"
    r"parents_per_sec=(?P<parents>[0-9.eE+-]+)\s+"
    r"candidates_per_sec=(?P<candidates>[0-9.eE+-]+)\s+"
    r"checksum=(?P<checksum>-?\d+)"
)


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


def cleanup_path(path: Path):
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def disk_line(path):
    usage = shutil.disk_usage(path)
    return f"{path}: free={usage.free} total={usage.total}"


def find_model_checkpoint() -> Path:
    input_root = Path("/kaggle/input")
    all_pth = sorted(input_root.rglob("*.pth")) if input_root.exists() else []
    candidate_roots = [Path(path) for path in MODEL_INPUT_ROOTS if Path(path).exists()]
    if not candidate_roots and input_root.exists():
        candidate_roots = sorted({path for path in input_root.rglob("*") if path.is_dir() and MODEL_SOURCE_SLUG in str(path)})
    matches = []
    for root in candidate_roots:
        matches.extend(root.rglob("*.pth"))
    matches = sorted(set(matches))
    if len(matches) != 1:
        message = [
            "expected exactly one transformer .pth under the configured Kaggle model source",
            f"configured_model_source={KAGGLE_MODEL_SOURCE}",
            "candidate_roots=" + json.dumps([str(path) for path in candidate_roots], indent=2),
            "matched_pth=" + json.dumps([str(path) for path in matches], indent=2),
            "all_discovered_pth=" + json.dumps([str(path) for path in all_pth], indent=2),
        ]
        raise RuntimeError("\n".join(message))
    return matches[0]


def preflight():
    print("KAGGLE_MODEL_SOURCE=", KAGGLE_MODEL_SOURCE, flush=True)
    print("GITHUB_BRANCH=", GITHUB_BRANCH, flush=True)
    print("EXPECTED_COMMIT_PREFIX=", EXPECTED_COMMIT_PREFIX, flush=True)
    print("disk_tmp=", disk_line("/tmp"), flush=True)
    print("disk_working=", disk_line("/kaggle/working"), flush=True)
    run_capture(["nvidia-smi"], check=False)
    import torch
    print("torch_version=", torch.__version__, flush=True)
    print("torch_cmake_prefix_path=", torch.utils.cmake_prefix_path, flush=True)
    gpu_count = torch.cuda.device_count()
    print("torch_cuda_device_count=", gpu_count, flush=True)
    if gpu_count < len(BENCH_GPUS):
        raise RuntimeError(f"expected at least {len(BENCH_GPUS)} GPUs, found {gpu_count}")
    return torch.utils.cmake_prefix_path


def prepare_repo_and_weights(torch_cmake_prefix_path: str):
    for path in (REPO_DIR, BUILD_DIR, WEIGHT_OUT_DIR):
        cleanup_path(path)
    run_checked(["git", "clone", "--branch", GITHUB_BRANCH, "--depth", "1", GITHUB_REPO_URL, REPO_DIR])
    actual = run_capture(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR).stdout.strip()
    print("checked_out_commit=", actual, flush=True)
    if EXPECTED_COMMIT_PREFIX and not actual.startswith(EXPECTED_COMMIT_PREFIX):
        raise RuntimeError(f"expected commit prefix {EXPECTED_COMMIT_PREFIX}, got {actual}")
    checkpoint_path = find_model_checkpoint()
    print("selected_transformer_checkpoint=", checkpoint_path, flush=True)
    run_checked([
        sys.executable, REPO_DIR / "tools" / "export_stream1.py",
        "--weights", checkpoint_path,
        "--out", WEIGHT_OUT_DIR,
        "--format", "piece-transformer",
        "--dtype", "fp16",
        "--num-classes", "120",
    ], cwd=REPO_DIR)
    manifest = json.loads((WEIGHT_OUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    summary = {
        "backend": manifest.get("backend"),
        "dtype": manifest.get("dtype"),
        "seq_len": manifest.get("seq_len"),
        "d_model": manifest.get("d_model"),
        "nhead": manifest.get("nhead"),
        "layers": manifest.get("num_layers"),
        "output_dim": manifest.get("output_dim"),
    }
    print("exported_manifest_summary=", summary, flush=True)
    if summary["backend"] != "piece_transformer" or summary["dtype"] != "fp16":
        raise RuntimeError(f"bad exported manifest summary: {summary}")

    if not (CUTLASS_DIR / "include").exists():
        cleanup_path(CUTLASS_DIR)
        run_checked(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/cutlass.git", CUTLASS_DIR])

    run_checked([
        "cmake", "-S", REPO_DIR, "-B", BUILD_DIR, "-GNinja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_PREFIX_PATH={torch_cmake_prefix_path}",
        "-DBEAM_ENABLE_LIBTORCH_STREAM1=ON",
        f"-DBEAM_CUDA_ARCHITECTURES={CUDA_ARCHITECTURES}",
        f"-DCUTLASS_DIR={CUTLASS_DIR}",
    ], cwd=REPO_DIR)
    run_checked(["cmake", "--build", BUILD_DIR, "--target", "stream1_transformer_libtorch_benchmark", "-j", "2"])


def run_one_gpu(gpu: int):
    log_path = BENCH_LOG_DIR / f"stream1_libtorch_transformer_gpu{gpu}.log"
    csv_path = BENCH_LOG_DIR / f"stream1_libtorch_transformer_gpu{gpu}.csv"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        BUILD_DIR / "stream1_transformer_libtorch_benchmark",
        "--weight-dir", WEIGHT_OUT_DIR,
        "--device", "cuda:0",
        "--batches", ",".join(str(x) for x in BENCH_BATCHES),
        "--warmup", str(BENCH_WARMUP),
        "--iters", str(BENCH_ITERS),
        "--csv", csv_path,
    ]
    print("RUN_STREAM1_LIBTORCH_BENCH_START", {"gpu": gpu, "log": str(log_path), "csv": str(csv_path)}, flush=True)
    rows = []
    start = time.time()
    with log_path.open("w", buffering=1, encoding="utf-8") as log:
        proc = subprocess.Popen([str(x) for x in cmd], cwd=REPO_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            print(line, end="", flush=True)
            match = row_pattern.search(line)
            if match:
                group = match.groupdict()
                rows.append({
                    "gpu": gpu,
                    "batch": int(group["batch"]),
                    "iters": int(group["iters"]),
                    "elapsed_ms": float(group["elapsed"]),
                    "parents_per_sec": float(group["parents"]),
                    "candidates_per_sec": float(group["candidates"]),
                    "checksum": int(group["checksum"]),
                })
        rc = proc.wait()
    elapsed = time.time() - start
    print("RUN_STREAM1_LIBTORCH_BENCH_DONE", {"gpu": gpu, "rc": rc, "seconds": elapsed, "rows": len(rows)}, flush=True)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, [str(x) for x in cmd])
    if not rows:
        raise RuntimeError(f"no libtorch benchmark rows parsed for gpu {gpu}")
    return rows


def main():
    torch_cmake_prefix_path = preflight()
    prepare_repo_and_weights(torch_cmake_prefix_path)
    all_rows = []
    for gpu in BENCH_GPUS:
        all_rows.extend(run_one_gpu(gpu))
    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["gpu", "batch", "iters", "elapsed_ms", "parents_per_sec", "candidates_per_sec", "checksum"])
        writer.writeheader()
        writer.writerows(all_rows)
    best_by_gpu = {}
    for row in all_rows:
        gpu = row["gpu"]
        if gpu not in best_by_gpu or row["candidates_per_sec"] > best_by_gpu[gpu]["candidates_per_sec"]:
            best_by_gpu[gpu] = row
    aggregate = sum(row["candidates_per_sec"] for row in best_by_gpu.values())
    summary = {
        "rows_csv": str(ROWS_CSV),
        "best_by_gpu": {str(k): v for k, v in sorted(best_by_gpu.items())},
        "aggregate_candidates_per_sec": aggregate,
        "pytorch_batch_process_per_t4_reference": 630697.0,
        "pytorch_batch_process_2xt4_reference": 1261394.0,
        "libtorch_over_pytorch_2xt4": aggregate / 1261394.0,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cleanup_path(WEIGHT_OUT_DIR)
    print("STREAM1_LIBTORCH_WEIGHTS_CLEANED=", WEIGHT_OUT_DIR, flush=True)
    print("STREAM1_LIBTORCH_BENCH_CSV=", ROWS_CSV, flush=True)
    for gpu, row in sorted(best_by_gpu.items()):
        print("STREAM1_LIBTORCH_BEST_GPU", gpu, row, flush=True)
    print("STREAM1_LIBTORCH_BEST_2XT4_AGG_CANDIDATES_PER_SEC=", aggregate, flush=True)
    print("STREAM1_LIBTORCH_OVER_PYTORCH_2XT4=", summary["libtorch_over_pytorch_2xt4"], flush=True)
    print("STREAM1_LIBTORCH_SUMMARY_JSON=", SUMMARY_JSON, flush=True)


if __name__ == "__main__":
    main()