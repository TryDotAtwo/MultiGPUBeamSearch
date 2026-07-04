from __future__ import annotations

import argparse
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
MODEL_ARTIFACT_NAME = "megaminx_qtransformer_1782210824_e99997"
MODEL_ID = 1782210824
GROUP_ID = 900
TARGET_ID = 0

WORK_DIR = Path("/kaggle/working")
TMP_DIR = Path("/tmp")
REPO_DIR = TMP_DIR / "beam_solver_transformer_backend_compare"
CUTLASS_DIR = TMP_DIR / "cutlass"
BUILD_DIR = TMP_DIR / "beam_build_transformer_backend_compare"
WEIGHT_OUT_DIR = WORK_DIR / "stream1_transformer_weights_fp16"
LOG_DIR = WORK_DIR / "stream1_transformer_backend_compare_logs"
ROWS_CSV = WORK_DIR / "stream1_transformer_backend_compare_rows.csv"
SUMMARY_JSON = WORK_DIR / "stream1_transformer_backend_compare_summary.json"
SUMMARY_MD = WORK_DIR / "stream1_transformer_backend_compare_summary.md"

CUDA_ARCHITECTURES = "75"
BENCH_GPUS = [0, 1]
LIBTORCH_BATCHES = [128, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 1536, 2048]
TORCH_EXPORTED_BATCHES = [128, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 1536, 2048, 3072, 4096]
ORIGINAL_TORCH_TOTAL_ROWS = 65536
ORIGINAL_TORCH_EVAL_BATCHES = [2048, 4096, 8192, 16384]
BENCH_WARMUP = 8
BENCH_ITERS = 50
LIBTORCH_WARMUP = 20
LIBTORCH_ITERS = 100
ORIGINAL_TORCH_WARMUP = 3
ORIGINAL_TORCH_ITERS = 20

LOG_DIR.mkdir(parents=True, exist_ok=True)

libtorch_row_pattern = re.compile(
    r"stream1_transformer_libtorch_micro\s+"
    r"mode=(?P<mode>\w+)\s+"
    r"batch=(?P<batch>\d+)\s+"
    r"iters=(?P<iters>\d+)\s+"
    r"elapsed_ms=(?P<elapsed>[0-9.eE+-]+)\s+"
    r"parents_per_sec=(?P<parents>[0-9.eE+-]+)\s+"
    r"candidates_per_sec=(?P<candidates>[0-9.eE+-]+)\s+"
    r"checksum=(?P<checksum>-?\d+)"
)

native_row_pattern = re.compile(
    r"stream1_transformer_micro\s+"
    r"b_micro=(?P<b_micro>\d+)\s+"
    r"concurrency=(?P<concurrency>\d+)\s+"
    r"rows_per_launch_group=(?P<rows>\d+)\s+"
    r"ms_per_launch_group=(?P<ms>[0-9.eE+-]+)\s+"
    r"parents_per_sec=(?P<parents>[0-9.eE+-]+)\s+"
    r"candidates_per_sec=(?P<candidates>[0-9.eE+-]+)\s+"
    r"scratch_bytes=(?P<scratch>\d+)"
)

child_row_pattern = re.compile(r"BACKEND_COMPARE_CHILD_ROW\s+(?P<pairs>.+)$")


def parse_pairs(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for part in text.split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value
    return values


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


def disk_line(path: str) -> str:
    usage = shutil.disk_usage(path)
    return f"{path}: free={usage.free} total={usage.total}"


def find_model_checkpoint() -> Path:
    input_root = Path("/kaggle/input")
    all_pth = sorted(input_root.rglob("*.pth")) if input_root.exists() else []
    candidate_roots = [
        Path("/kaggle/input/megaminx-qtransformer-1782210824/PyTorch/default/1"),
        Path("/kaggle/input/megaminx-qtransformer-1782210824/pytorch/default/1"),
    ]
    candidate_roots = [path for path in candidate_roots if path.exists()]
    if not candidate_roots and input_root.exists():
        candidate_roots = sorted({path for path in input_root.rglob("*") if path.is_dir() and MODEL_SOURCE_SLUG in str(path)})
    matches: list[Path] = []
    for root in candidate_roots:
        matches.extend(root.rglob("*.pth"))
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise RuntimeError(
            "\n".join(
                [
                    "expected exactly one transformer .pth under the configured Kaggle model source",
                    f"configured_model_source={KAGGLE_MODEL_SOURCE}",
                    "candidate_roots=" + json.dumps([str(path) for path in candidate_roots], indent=2),
                    "matched_pth=" + json.dumps([str(path) for path in matches], indent=2),
                    "all_discovered_pth=" + json.dumps([str(path) for path in all_pth], indent=2),
                ]
            )
        )
    return matches[0]


def find_model_root() -> Path:
    input_root = Path("/kaggle/input")
    candidates: list[Path] = []
    for manifest in input_root.rglob("manifest.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("name") == MODEL_ARTIFACT_NAME:
            candidates.append(manifest.parent)
    if not candidates:
        raise FileNotFoundError(f"Add the Kaggle Model artifact {MODEL_ARTIFACT_NAME}.")
    return candidates[0] / "megaminx-transformer"


def preflight() -> str:
    print("BACKEND_COMPARE_MODEL_SOURCE=", KAGGLE_MODEL_SOURCE, flush=True)
    print("BACKEND_COMPARE_GITHUB_BRANCH=", GITHUB_BRANCH, flush=True)
    print("BACKEND_COMPARE_EXPECTED_COMMIT_PREFIX=", EXPECTED_COMMIT_PREFIX, flush=True)
    print("disk_tmp=", disk_line("/tmp"), flush=True)
    print("disk_working=", disk_line("/kaggle/working"), flush=True)
    run_capture(["nvidia-smi"], check=False)
    import torch

    print("torch_version=", torch.__version__, flush=True)
    print("torch_cuda=", torch.version.cuda, flush=True)
    print("torch_cmake_prefix_path=", torch.utils.cmake_prefix_path, flush=True)
    gpu_count = torch.cuda.device_count()
    print("torch_cuda_device_count=", gpu_count, flush=True)
    for index in range(gpu_count):
        print(f"torch_cuda_device_{index}={torch.cuda.get_device_name(index)}", flush=True)
    if gpu_count < len(BENCH_GPUS):
        raise RuntimeError(f"expected at least {len(BENCH_GPUS)} GPUs, found {gpu_count}")
    return torch.utils.cmake_prefix_path


def prepare_repo_weights_and_build(torch_cmake_prefix_path: str) -> None:
    for path in (REPO_DIR, BUILD_DIR, WEIGHT_OUT_DIR):
        cleanup_path(path)
    run_checked(["git", "clone", "--branch", GITHUB_BRANCH, "--depth", "1", GITHUB_REPO_URL, REPO_DIR])
    actual = run_capture(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR).stdout.strip()
    print("checked_out_commit=", actual, flush=True)
    if EXPECTED_COMMIT_PREFIX and not actual.startswith(EXPECTED_COMMIT_PREFIX):
        raise RuntimeError(f"expected commit prefix {EXPECTED_COMMIT_PREFIX}, got {actual}")

    checkpoint_path = find_model_checkpoint()
    print("selected_transformer_checkpoint=", checkpoint_path, flush=True)
    run_checked(
        [
            sys.executable,
            REPO_DIR / "tools" / "export_stream1.py",
            "--weights",
            checkpoint_path,
            "--out",
            WEIGHT_OUT_DIR,
            "--format",
            "piece-transformer",
            "--dtype",
            "fp16",
            "--num-classes",
            "120",
        ],
        cwd=REPO_DIR,
    )
    manifest = json.loads((WEIGHT_OUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest_summary = {
        "backend": manifest.get("backend"),
        "dtype": manifest.get("dtype"),
        "seq_len": manifest.get("seq_len"),
        "d_model": manifest.get("d_model"),
        "nhead": manifest.get("nhead"),
        "layers": manifest.get("num_layers"),
        "output_dim": manifest.get("output_dim"),
    }
    print("exported_manifest_summary=", manifest_summary, flush=True)
    if manifest_summary["backend"] != "piece_transformer" or manifest_summary["dtype"] != "fp16":
        raise RuntimeError(f"bad exported manifest summary: {manifest_summary}")

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
            f"-DCMAKE_PREFIX_PATH={torch_cmake_prefix_path}",
            "-DBEAM_ENABLE_LIBTORCH_STREAM1=ON",
            f"-DBEAM_CUDA_ARCHITECTURES={CUDA_ARCHITECTURES}",
            f"-DCUTLASS_DIR={CUTLASS_DIR}",
        ],
        cwd=REPO_DIR,
    )
    run_checked(
        [
            "cmake",
            "--build",
            BUILD_DIR,
            "--target",
            "stream_benchmark",
            "stream1_transformer_libtorch_benchmark",
            "-j",
            "2",
        ]
    )


def child_run_torch_exported(args: argparse.Namespace) -> None:
    import torch

    sys.path.insert(0, str(REPO_DIR))
    from tools.stream1_transformer_torch_benchmark import PieceTransformerTorch, benchmark_batch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    model = PieceTransformerTorch(WEIGHT_OUT_DIR, device)
    for batch in TORCH_EXPORTED_BATCHES:
        row = benchmark_batch(model, batch, BENCH_WARMUP, BENCH_ITERS)
        print(
            "BACKEND_COMPARE_CHILD_ROW"
            " backend=torch_exported"
            " mode=eager"
            f" gpu={args.gpu_label}"
            f" batch={batch}"
            f" iters={row['iters']}"
            f" elapsed_ms={row['elapsed_ms']:.6f}"
            f" parents_per_sec={row['parents_per_s']:.3f}"
            f" candidates_per_sec={row['candidates_per_s']:.3f}"
            f" peak_mem_gib={row['peak_mem_gib']:.6f}",
            flush=True,
        )


def child_run_original_torch(args: argparse.Namespace) -> None:
    import torch

    model_root = find_model_root()
    if str(model_root) not in sys.path:
        sys.path.insert(0, str(model_root))
    from pilgrim import build_model_from_info, configure_inference_backend, generate_inverse_moves, parse_generator_spec
    from pilgrim.model import batch_process
    from pilgrim.utils import load_torch_file

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    base = f"model_p{GROUP_ID:03d}-t{TARGET_ID:03d}"
    hits = sorted((model_root / "logs").glob(f"{base}*_{MODEL_ID}.json"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one metadata file for model_id={MODEL_ID}, got {hits}")
    info_path = hits[0]
    info = json.loads(info_path.read_text(encoding="utf-8"))
    with (model_root / "generators" / f"p{GROUP_ID:03d}.json").open("r", encoding="utf-8") as f:
        moves, names = parse_generator_spec(json.load(f))
    all_moves = torch.tensor(moves, dtype=torch.int64, device=device)
    _inverse_moves = torch.tensor(generate_inverse_moves(names), dtype=torch.int64, device=device)
    target = load_torch_file(
        model_root / "targets" / f"p{GROUP_ID:03d}-t{TARGET_ID:03d}.pt",
        weights_only=True,
        map_location=device,
    )
    q_model = str(info.get("model_name", "")).endswith("-q") or str(info.get("training_mode", "")).startswith("q_")
    output_dim = int(all_moves.size(0) if q_model else 1)
    model = build_model_from_info(
        info,
        num_classes=torch.unique(target).numel(),
        state_size=all_moves.size(1),
        output_dim=output_dim,
    )
    model_name = info.get("model_name", f"p{GROUP_ID:03d}-t{TARGET_ID:03d}")
    weights = model_root / "weights" / f"{model_name}_{MODEL_ID}_best.pth"
    model.load_state_dict(load_torch_file(weights, weights_only=False, map_location="cpu"), strict=True)
    model.eval()
    model.half()
    model.dtype = torch.float16
    if target.min() < 0:
        model.z_add = -target.min().item()
    model.to(device)
    backend = configure_inference_backend(model, "fast")
    print(
        "BACKEND_COMPARE_ORIGINAL_TORCH_ENV"
        f" gpu={args.gpu_label}"
        f" backend={backend}"
        f" output_dim={output_dim}"
        f" metadata={info_path}"
        f" weights={weights}",
        flush=True,
    )
    base_state = target.to(device=device, dtype=target.dtype).reshape(1, -1)
    states = base_state.repeat(ORIGINAL_TORCH_TOTAL_ROWS, 1).contiguous()
    for eval_batch_size in ORIGINAL_TORCH_EVAL_BATCHES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            with torch.inference_mode():
                for _ in range(ORIGINAL_TORCH_WARMUP):
                    logits = batch_process(model, states, states.device, eval_batch_size)
                torch.cuda.synchronize(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(ORIGINAL_TORCH_ITERS):
                    logits = batch_process(model, states, states.device, eval_batch_size)
                end.record()
                torch.cuda.synchronize(device)
        except RuntimeError as exc:
            torch.cuda.empty_cache()
            print(
                "BACKEND_COMPARE_CHILD_SKIP"
                " backend=torch_original_batch_process"
                f" gpu={args.gpu_label}"
                f" eval_batch_size={eval_batch_size}"
                f" status={type(exc).__name__}:{str(exc).splitlines()[0][:160]}",
                flush=True,
            )
            continue
        elapsed_ms = float(start.elapsed_time(end))
        parents_per_sec = (ORIGINAL_TORCH_TOTAL_ROWS * ORIGINAL_TORCH_ITERS) / (elapsed_ms / 1000.0)
        candidates_per_sec = parents_per_sec * output_dim
        sample = float(logits.reshape(-1)[0].detach().float().item())
        print(
            "BACKEND_COMPARE_CHILD_ROW"
            " backend=torch_original_batch_process"
            " mode=batch_process"
            f" gpu={args.gpu_label}"
            f" batch={ORIGINAL_TORCH_TOTAL_ROWS}"
            f" eval_batch_size={eval_batch_size}"
            f" iters={ORIGINAL_TORCH_ITERS}"
            f" elapsed_ms={elapsed_ms:.6f}"
            f" parents_per_sec={parents_per_sec:.3f}"
            f" candidates_per_sec={candidates_per_sec:.3f}"
            f" peak_mem_gib={torch.cuda.max_memory_allocated(device) / (1024.0**3):.6f}"
            f" sample={sample:.8f}",
            flush=True,
        )


def run_child_backend(gpu: int, child_kind: str) -> list[dict[str, object]]:
    log_path = LOG_DIR / f"{child_kind}_gpu{gpu}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [sys.executable, Path(__file__), "--child", child_kind, "--gpu-label", str(gpu)]
    print("BACKEND_COMPARE_CHILD_START", {"gpu": gpu, "child": child_kind, "log": str(log_path)}, flush=True)
    rows: list[dict[str, object]] = []
    with log_path.open("w", buffering=1, encoding="utf-8") as log:
        proc = subprocess.Popen([str(x) for x in cmd], cwd=REPO_DIR if child_kind == "torch-exported" else WORK_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            print(line, end="", flush=True)
            match = child_row_pattern.search(line)
            if match:
                values = parse_pairs(match.group("pairs"))
                row: dict[str, object] = {
                    "backend": values["backend"],
                    "mode": values["mode"],
                    "gpu": int(values["gpu"]),
                    "batch": int(values["batch"]),
                    "iters": int(values["iters"]),
                    "elapsed_ms": float(values["elapsed_ms"]),
                    "parents_per_sec": float(values["parents_per_sec"]),
                    "candidates_per_sec": float(values["candidates_per_sec"]),
                    "peak_mem_gib": float(values.get("peak_mem_gib", "0")),
                    "source": child_kind,
                }
                if "eval_batch_size" in values:
                    row["eval_batch_size"] = int(values["eval_batch_size"])
                rows.append(row)
        rc = proc.wait()
    print("BACKEND_COMPARE_CHILD_DONE", {"gpu": gpu, "child": child_kind, "rc": rc, "rows": len(rows)}, flush=True)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, [str(x) for x in cmd])
    if not rows:
        raise RuntimeError(f"no rows parsed for child={child_kind} gpu={gpu}")
    return rows


def run_libtorch(gpu: int, mode: str) -> list[dict[str, object]]:
    log_path = LOG_DIR / f"libtorch_{mode}_gpu{gpu}.log"
    csv_path = LOG_DIR / f"libtorch_{mode}_gpu{gpu}.csv"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        BUILD_DIR / "stream1_transformer_libtorch_benchmark",
        "--weight-dir",
        WEIGHT_OUT_DIR,
        "--device",
        "cuda:0",
        "--batches",
        ",".join(str(x) for x in LIBTORCH_BATCHES),
        "--warmup",
        str(LIBTORCH_WARMUP),
        "--iters",
        str(LIBTORCH_ITERS),
        "--csv",
        csv_path,
    ]
    if mode == "cuda_graph":
        cmd.append("--cuda-graph")
    rows: list[dict[str, object]] = []
    with log_path.open("w", buffering=1, encoding="utf-8") as log:
        proc = subprocess.Popen([str(x) for x in cmd], cwd=REPO_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            print(line, end="", flush=True)
            match = libtorch_row_pattern.search(line)
            if match:
                values = match.groupdict()
                rows.append(
                    {
                        "backend": "libtorch",
                        "mode": values["mode"],
                        "gpu": gpu,
                        "batch": int(values["batch"]),
                        "iters": int(values["iters"]),
                        "elapsed_ms": float(values["elapsed"]),
                        "parents_per_sec": float(values["parents"]),
                        "candidates_per_sec": float(values["candidates"]),
                        "checksum": int(values["checksum"]),
                        "source": "libtorch_cli",
                    }
                )
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, [str(x) for x in cmd])
    if not rows:
        raise RuntimeError(f"no libtorch rows parsed for gpu={gpu} mode={mode}")
    return rows


def run_native(gpu: int, mode: str) -> list[dict[str, object]]:
    log_path = LOG_DIR / f"native_cutlass_{mode}_gpu{gpu}.log"
    report_path = LOG_DIR / f"native_cutlass_{mode}_gpu{gpu}.md"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["BEAM_WEIGHT_DIR"] = str(WEIGHT_OUT_DIR)
    env["BEAM_STREAM_BENCH_REPORT"] = str(report_path)
    env["BEAM_STREAM1_TRANSFORMER_BLOCK51"] = "1"
    if mode == "graph":
        env["BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH"] = "1"
    rows: list[dict[str, object]] = []
    with log_path.open("w", buffering=1, encoding="utf-8") as log:
        proc = subprocess.Popen([str(BUILD_DIR / "stream_benchmark"), "991"], cwd=REPO_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            print(line, end="", flush=True)
            match = native_row_pattern.search(line)
            if match:
                values = match.groupdict()
                rows.append(
                    {
                        "backend": "native_cutlass",
                        "mode": mode,
                        "gpu": gpu,
                        "b_micro": int(values["b_micro"]),
                        "concurrency": int(values["concurrency"]),
                        "rows_per_launch_group": int(values["rows"]),
                        "batch": int(values["rows"]),
                        "iters": 0,
                        "elapsed_ms": float(values["ms"]),
                        "parents_per_sec": float(values["parents"]),
                        "candidates_per_sec": float(values["candidates"]),
                        "scratch_bytes": int(values["scratch"]),
                        "source": "stream_benchmark",
                    }
                )
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, [str(BUILD_DIR / "stream_benchmark"), "991"])
    if not rows:
        raise RuntimeError(f"no native rows parsed for gpu={gpu} mode={mode}")
    return rows


def best_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for row in rows:
        key = f"{row['backend']}/{row['mode']}/gpu{row['gpu']}"
        if key not in best or float(row["candidates_per_sec"]) > float(best[key]["candidates_per_sec"]):
            best[key] = row
    return best


def aggregate_best(best: dict[str, dict[str, object]]) -> dict[str, float]:
    aggregates: dict[str, float] = {}
    for row in best.values():
        key = f"{row['backend']}/{row['mode']}"
        aggregates[key] = aggregates.get(key, 0.0) + float(row["candidates_per_sec"])
    return aggregates


def write_outputs(rows: list[dict[str, object]], started: float) -> None:
    fieldnames = [
        "backend",
        "mode",
        "gpu",
        "batch",
        "eval_batch_size",
        "b_micro",
        "concurrency",
        "rows_per_launch_group",
        "iters",
        "elapsed_ms",
        "parents_per_sec",
        "candidates_per_sec",
        "peak_mem_gib",
        "scratch_bytes",
        "checksum",
        "source",
    ]
    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    best = best_rows(rows)
    aggregates = aggregate_best(best)
    fastest = max(aggregates.items(), key=lambda item: item[1])
    summary = {
        "rows_csv": str(ROWS_CSV),
        "log_dir": str(LOG_DIR),
        "best_by_backend_mode_gpu": best,
        "aggregate_best_candidates_per_sec": aggregates,
        "fastest_aggregate": {"backend_mode": fastest[0], "candidates_per_sec": fastest[1]},
        "seconds": time.time() - started,
        "checked_out_commit": run_capture(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR).stdout.strip(),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Stream1 Transformer Backend Compare 2xT4",
        "",
        f"- checked_out_commit={summary['checked_out_commit']}",
        f"- rows_csv={ROWS_CSV}",
        f"- log_dir={LOG_DIR}",
        f"- fastest={fastest[0]} {fastest[1]:.1f} candidates/s aggregate",
        "",
        "| backend/mode | aggregate best candidates/s |",
        "|---|---:|",
    ]
    for key, value in sorted(aggregates.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {key} | {value:.1f} |")
    lines.extend(["", "| backend/mode/gpu | best candidates/s | config |", "|---|---:|---|"])
    for key, row in sorted(best.items(), key=lambda item: float(item[1]["candidates_per_sec"]), reverse=True):
        config = {k: row.get(k) for k in ("batch", "eval_batch_size", "b_micro", "concurrency", "rows_per_launch_group", "iters") if row.get(k) not in (None, "", 0)}
        lines.append(f"| {key} | {float(row['candidates_per_sec']):.1f} | `{config}` |")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("BACKEND_COMPARE_ROWS_CSV=", ROWS_CSV, flush=True)
    print("BACKEND_COMPARE_SUMMARY_JSON=", SUMMARY_JSON, flush=True)
    print("BACKEND_COMPARE_SUMMARY_MD=", SUMMARY_MD, flush=True)
    print("BACKEND_COMPARE_FASTEST=", summary["fastest_aggregate"], flush=True)
    for key, value in sorted(aggregates.items(), key=lambda item: item[1], reverse=True):
        print("BACKEND_COMPARE_AGGREGATE", key, value, flush=True)


def parent_main() -> None:
    started = time.time()
    try:
        torch_cmake_prefix_path = preflight()
        prepare_repo_weights_and_build(torch_cmake_prefix_path)
        rows: list[dict[str, object]] = []
        for gpu in BENCH_GPUS:
            rows.extend(run_child_backend(gpu, "original-torch"))
        for gpu in BENCH_GPUS:
            rows.extend(run_child_backend(gpu, "torch-exported"))
        for mode in ("eager", "cuda_graph"):
            for gpu in BENCH_GPUS:
                rows.extend(run_libtorch(gpu, mode))
        for mode in ("eager", "graph"):
            for gpu in BENCH_GPUS:
                rows.extend(run_native(gpu, mode))
        write_outputs(rows, started)
    finally:
        cleanup_path(WEIGHT_OUT_DIR)
        print("BACKEND_COMPARE_WEIGHTS_CLEANED=", WEIGHT_OUT_DIR, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=["torch-exported", "original-torch"])
    parser.add_argument("--gpu-label", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.child == "torch-exported":
        child_run_torch_exported(args)
    elif args.child == "original-torch":
        child_run_original_torch(args)
    else:
        parent_main()


if __name__ == "__main__":
    main()
