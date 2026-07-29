#!/usr/bin/env python3
"""Build and validate the private real-2xT4 CayleyPy Task 5 acceptance gate."""

from __future__ import annotations

import ast
import base64
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any
import zlib


BASE_GIT_REV = "6f95bd6bdb32b5f6ef7cca32b96967bce6036503"
REVIEWED_COMMIT = "6830401ed2086921d2563c2bc3c11faf6c5a0741"
CUTLASS_GIT_REV = "afa1772203677c5118fcd82537a9c8fefbcc7008"
KERNEL_SLUG = "trydotatwo/cayleypy-public-task-5-2xt4-gate"
KERNEL_VERSION = 3
OUT_DIR = Path("kaggle_cayleypy_task5_gate")
OUT_NOTEBOOK = OUT_DIR / "cayleypy-task5-2xt4-gate.ipynb"
SOURCE_PATH = Path(__file__).resolve().with_name("production_runner.cu")
PUZZLE_INFO_PATH = SOURCE_PATH.parents[1] / "data" / "puzzle_info.json"
TEST_CSV_PATH = SOURCE_PATH.parents[1] / "data" / "test.csv"
PUZZLE_ID = 1
BEAM_WIDTH = 65536
DEPTH_LIMIT = 8
MAX_UNIQUE_SOLUTIONS = 16
SOLVED_RESULT_CAPACITY = 786432
K1_RADIUS = 0
K2_RADIUS = 0
MEASURED_PROFILE = {
    "hardware": "kaggle_2xt4",
    "model_class": "output_move_count",
    "profile_power": 16,
    "requested_beam": 65536,
    "effective_beam": 65536,
    "local_beam": 32768,
    "b_micro": 2048,
    "stream1_concurrency": 4,
    "stream3_ring_slots": 4,
    "stream3_batch_candidates": 196608,
    "shard_count": 4,
    "shard_capacity_candidates": 196608,
    "shard_capacity_scale_ppm": 1050000,
    "stream4_batch_candidates": 98304,
    "stream4_trigger_candidates": 98304,
    "stream4_active_sort_slots": 4,
    "evidence_kernel_slug": "trydotatwo/cayley-beam-2xt4-mlp-autoprofiles",
    "evidence_kernel_version": 5,
    "evidence_config_id": "depth8_output_move_count_p16_sh4_b2048_1",
    "evidence_depth_sec": 0.127134,
    "evidence_elapsed_sec": 5.127961788999983,
}

_TSV_FIELDS = (
    "puzzle_id", "depth_index", "found_depth", "total_depth",
    "known_length", "delta", "owner_rank", "solution_path",
)
_FIRST_RELEASE_RE = re.compile(
    r"^\[default0\]:puzzle_solved=1 puzzle_id=1 "
    r"seconds=(?P<seconds>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?) "
    r"solution_length=(?P<solution_length>\d+) "
    r"found_depth=(?P<found_depth>\d+) "
    r"touch_depth=(?P<touch_depth>\d+) solution=(?P<solution>.*)$"
)
_FATAL_PATTERNS = (
    "out of memory", "cudaerror", "nccl error", "stream fatal",
    "childfailederror", "timed out", "solve bucket overflow", "collective hang",
)
_PRIVATE_PATTERNS = (
    "c:\\users\\", "d:\\100xh100", "codex/public-cayleypy-notebook",
    "ghp_", "github_pat_", "sk-proj-", "kaggle_key",
)


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _code_cell(source: str, cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "code", "id": cell_id, "execution_count": None,
        "metadata": {}, "outputs": [], "source": source.strip().splitlines(keepends=True),
    }


def _markdown_cell(source: str, cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown", "id": cell_id, "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def _config_cell(source: bytes) -> str:
    payload = base64.b64encode(zlib.compress(source, level=9)).decode("ascii")
    return f'''from pathlib import Path

KERNEL_SLUG = {KERNEL_SLUG!r}
BASE_REPO_URL = "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git"
BASE_GIT_REV = {BASE_GIT_REV!r}
REVIEWED_COMMIT = {REVIEWED_COMMIT!r}
CUTLASS_REPO_URL = "https://github.com/NVIDIA/cutlass.git"
CUTLASS_GIT_REV = {CUTLASS_GIT_REV!r}
SOURCE_SHA256 = {_digest(source)!r}
SOURCE_SIZE = {len(source)}
SOURCE_PAYLOAD_B64 = {payload!r}
PUZZLE_ID = {PUZZLE_ID}
BEAM_WIDTH = {BEAM_WIDTH}
DEPTH_LIMIT = {DEPTH_LIMIT}
MAX_UNIQUE_SOLUTIONS = {MAX_UNIQUE_SOLUTIONS}
SOLVED_RESULT_CAPACITY = {SOLVED_RESULT_CAPACITY}
K1_RADIUS = {K1_RADIUS}
K2_RADIUS = {K2_RADIUS}
RUN_TIMEOUT_SEC = 240
MAX_COMBINED_LOG_BYTES = 8 * 1024**2
PROCESS_RSS_MAX_SAMPLES = 512
GPU_HEADROOM_BYTES = 768 * 1024**2
HISTORY_RAM_BYTES = 28 * 1024**3
HISTORY_DISK_BYTES = 32 * 1024**3
PROFILE = {json.dumps(MEASURED_PROFILE, sort_keys=True)}
'''


SETUP_CELL = r'''
import base64
from hashlib import sha256
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import zlib

WORK_DIR = Path("/kaggle/working")
REPO_DIR = Path("/tmp/cayleypy_task5_gate_repo")
CUTLASS_DIR = Path("/tmp/cayleypy_task5_gate_cutlass")
BUILD_DIR = Path("/tmp/cayleypy_task5_gate_build")
BUILD_LOG = WORK_DIR / "build.log"


def sha256_file(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(cmd):
    return subprocess.check_output(list(map(str, cmd)), text=True).strip()


def run_logged(cmd, cwd=None):
    started = time.perf_counter()
    completed = subprocess.run(
        list(map(str, cmd)), cwd=cwd, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    )
    elapsed = time.perf_counter() - started
    block = "+ " + " ".join(map(str, cmd)) + f"\nreturn_code={completed.returncode} elapsed_sec={elapsed:.6f}\n" + completed.stdout
    with BUILD_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(block)
        if not block.endswith("\n"):
            handle.write("\n")
    print(block[-4000:], flush=True)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, cmd, output=completed.stdout)
    return elapsed


WORK_DIR.mkdir(parents=True, exist_ok=True)
BUILD_LOG.write_text("", encoding="utf-8")
for path in (REPO_DIR, CUTLASS_DIR, BUILD_DIR, Path("/tmp/beam_history_task5_gate")):
    if path.exists():
        shutil.rmtree(path)

gpu_rows = capture([
    "nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
    "--format=csv,noheader,nounits",
]).splitlines()
gpu_names = [row.split(",")[1].strip() for row in gpu_rows]
if len(gpu_rows) != 2 or any(name not in {"Tesla T4", "NVIDIA T4"} for name in gpu_names):
    raise RuntimeError(f"expected exactly two NVIDIA T4 GPUs; observed={gpu_rows!r}")
(WORK_DIR / "environment.json").write_text(json.dumps({
    "gpu_rows": gpu_rows,
    "python": sys.version,
    "torch": __import__("torch").__version__,
    "psutil": __import__("psutil").__version__,
    "nvcc": capture(["nvcc", "--version"]),
    "cmake": capture(["cmake", "--version"]).splitlines()[0],
    "ninja": capture(["ninja", "--version"]),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

timings = {}
timings["base_clone"] = run_logged([
    "git", "clone", "--no-checkout", "--depth=1", "--branch", "main",
    BASE_REPO_URL, REPO_DIR,
])
observed_origin = capture(["git", "-C", REPO_DIR, "rev-parse", "origin/main"])
if observed_origin != BASE_GIT_REV:
    raise RuntimeError(f"public origin/main drift: {observed_origin} != {BASE_GIT_REV}")
run_logged(["git", "-C", REPO_DIR, "checkout", "--detach", BASE_GIT_REV])
if capture(["git", "-C", REPO_DIR, "rev-parse", "HEAD"]) != BASE_GIT_REV:
    raise RuntimeError("base checkout mismatch")

source_bytes = zlib.decompress(base64.b64decode(SOURCE_PAYLOAD_B64, validate=True))
if len(source_bytes) != SOURCE_SIZE or sha256(source_bytes).hexdigest() != SOURCE_SHA256:
    raise RuntimeError("embedded source failed pre-write attestation")
source_target = REPO_DIR / "tools/production_runner.cu"
base_source_sha256 = sha256_file(source_target)
source_target.write_bytes(source_bytes)
if sha256_file(source_target) != SOURCE_SHA256:
    raise RuntimeError("source overlay failed post-write attestation")

weights_dir = REPO_DIR / "stream1_weights"
weight_manifest = json.loads((weights_dir / "manifest.json").read_text(encoding="utf-8"))
model = {"state_len": 120, "num_classes": 120, "output_dim": 24, "dtype": "fp16"}
if any(weight_manifest.get(key) != value for key, value in model.items()):
    raise RuntimeError(f"tracked Stream1 model mismatch: {model} vs {weight_manifest}")
weight_files = {
    path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    for path in sorted(weights_dir.iterdir()) if path.is_file()
}
(WORK_DIR / "weights_manifest.json").write_text(
    json.dumps({"model": model, "files": weight_files}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

timings["cutlass_clone"] = run_logged([
    "git", "clone", "--no-checkout", "--depth=1", "--branch", "v3.8.0",
    CUTLASS_REPO_URL, CUTLASS_DIR,
])
run_logged(["git", "-C", CUTLASS_DIR, "checkout", "--detach", CUTLASS_GIT_REV])
if capture(["git", "-C", CUTLASS_DIR, "rev-parse", "HEAD"]) != CUTLASS_GIT_REV:
    raise RuntimeError("CUTLASS checkout mismatch")
timings["configure"] = run_logged([
    "cmake", "-S", REPO_DIR, "-B", BUILD_DIR, "-GNinja",
    "-DCMAKE_BUILD_TYPE=Release", "-DBEAM_CUDA_ARCHITECTURES=75",
    f"-DCUTLASS_DIR={CUTLASS_DIR}", "-DBEAM_ENABLE_DEBUG=ON",
    "-DBEAM_ENABLE_DEPTH_LOGS=ON", "-DBEAM_ENABLE_DEBUG_LOGS=OFF",
    "-DBEAM_DEBUG_PIPELINE_STATS=OFF",
], cwd=REPO_DIR)
timings["compile"] = run_logged([
    "cmake", "--build", BUILD_DIR, "--target", "production_runner", "-j", "2",
], cwd=REPO_DIR)
binary = BUILD_DIR / "production_runner"
if not binary.is_file():
    raise RuntimeError("Release production_runner binary is missing")
ldd = capture(["ldd", binary])
run_logged(["ldd", binary], cwd=REPO_DIR)
shutil.copy2(BUILD_DIR / "CMakeCache.txt", WORK_DIR / "CMakeCache.txt")
source_manifest = {
    "kernel_slug": KERNEL_SLUG, "base_git_rev": BASE_GIT_REV,
    "reviewed_commit": REVIEWED_COMMIT,
    "base_production_runner_sha256": base_source_sha256,
    "production_runner_sha256": sha256_file(source_target),
    "production_runner_bytes": source_target.stat().st_size,
    "binary_sha256": sha256_file(binary), "binary_bytes": binary.stat().st_size,
    "cutlass_git_rev": CUTLASS_GIT_REV, "build_type": "Release",
    "cuda_architectures": "75", "nccl_linked": "libnccl" in ldd.lower(),
    "timings_sec": timings,
}
if source_manifest["production_runner_sha256"] != SOURCE_SHA256 or not source_manifest["nccl_linked"]:
    raise RuntimeError(f"source/build attestation failed: {source_manifest}")
(WORK_DIR / "source_manifest.json").write_text(
    json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print("TASK5_GATE_SETUP_COMPLETE", json.dumps(source_manifest, sort_keys=True), flush=True)
'''


RUN_CELL = r'''
from collections import deque
import csv
from hashlib import sha256
import json
import os
import re
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil

RUNS_DIR = WORK_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_ROOT = Path("/tmp/beam_history_task5_gate")
FATAL_PATTERNS = (
    "out of memory", "cudaerror", "nccl error", "stream fatal",
    "childfailederror", "timed out", "solve bucket overflow", "collective hang",
)
RELEASE_RE = re.compile(
    r"^\[default0\]:puzzle_solved=1 puzzle_id=1 "
    r"seconds=(?P<seconds>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?) "
    r"solution_length=(?P<solution_length>\d+) found_depth=(?P<found_depth>\d+) "
    r"touch_depth=(?P<touch_depth>\d+) solution=(?P<solution>.*)$",
    re.MULTILINE,
)
TSV_FIELDS = (
    "puzzle_id", "depth_index", "found_depth", "total_depth",
    "known_length", "delta", "owner_rank", "solution_path",
)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def gpu_snapshot():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()


def process_tree_rss(pid):
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0, 0
    total = 0
    count = 0
    for process in processes:
        try:
            total += process.memory_info().rss
            count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total, count


def base_env(run_name, history_dir):
    reserved = {
        "WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
        "ROLE_RANK", "ROLE_NAME", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
        "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID", "TORCHELASTIC_ERROR_FILE", "PYTHON_EXEC",
    }
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("BEAM_") and key not in reserved
    }
    env.update({
        "BEAM_WEIGHT_DIR": str(REPO_DIR / "stream1_weights"),
        "BEAM_PUZZLE_INFO_JSON": str(REPO_DIR / "data/puzzle_info.json"),
        "BEAM_GENERATOR_PATH": str(REPO_DIR / "FullBeamNice/generators/p900.json"),
        "BEAM_TEST_CSV": str(REPO_DIR / "data/test.csv"),
        "BEAM_RUNTIME_CONFIG_MODE": "manual",
        "BEAM_B_MICRO": str(PROFILE["b_micro"]),
        "BEAM_STREAM1_CONCURRENCY": str(PROFILE["stream1_concurrency"]),
        "BEAM_STREAM3_RING_SLOTS": str(PROFILE["stream3_ring_slots"]),
        "BEAM_STREAM3_BATCH_CANDIDATES": str(PROFILE["stream3_batch_candidates"]),
        "BEAM_SHARD_COUNT": str(PROFILE["shard_count"]),
        "BEAM_SHARD_BUFFER_COUNT": "2",
        "BEAM_SHARD_CAPACITY_CANDIDATES": str(PROFILE["shard_capacity_candidates"]),
        "BEAM_SHARD_CAPACITY_SCALE_PPM": str(PROFILE["shard_capacity_scale_ppm"]),
        "BEAM_STREAM4_BATCH_CANDIDATES": str(PROFILE["stream4_batch_candidates"]),
        "BEAM_STREAM4_TRIGGER_CANDIDATES": str(PROFILE["stream4_trigger_candidates"]),
        "BEAM_STREAM4_ACTIVE_SORT_SLOTS": str(PROFILE["stream4_active_sort_slots"]),
        "BEAM_GLOBAL_SPILL_CAPACITY": "0",
        "BEAM_STREAM5_RECV_CAPACITY_SCALE_PPM": "1000000",
        "BEAM_GPU_HEADROOM_BYTES": str(GPU_HEADROOM_BYTES),
        "BEAM_HISTORY_MODE": "static_hybrid",
        "BEAM_HISTORY_SLOT_COUNT": "2", "BEAM_HISTORY_WORKERS": "1",
        "BEAM_HISTORY_RAM_BYTES": str(HISTORY_RAM_BYTES),
        "BEAM_HISTORY_DISK_BYTES": str(HISTORY_DISK_BYTES),
        "BEAM_HISTORY_DIR": str(history_dir), "BEAM_HISTORY_DISK_PATH": str(history_dir),
        "BEAM_NCCL_ID_FILE": str(history_dir / f"{run_name}-nccl-id.bin"),
        "BEAM_SOLVED_NEIGHBORHOOD_RADIUS": str(K1_RADIUS),
        "BEAM_REPAIR_K1_RADIUS": str(K1_RADIUS), "BEAM_REPAIR_K2_RADIUS": str(K2_RADIUS),
        "BEAM_STREAM2_SUFFIX_RADIUS": str(K2_RADIUS), "BEAM_DEPTH_LOG_EVERY": "1",
    })
    return env


def discover_rank_logs(torchrun_dir, run_dir):
    copied = {"stdout": {}, "stderr": {}}
    for stream in ("stdout", "stderr"):
        for source in sorted(torchrun_dir.rglob(f"{stream}.log")):
            rank = next((int(part) for part in reversed(source.parts[:-1]) if part in {"0", "1"}), None)
            if rank is None:
                continue
            target = run_dir / f"rank{rank}.{stream}.log"
            shutil.copy2(source, target)
            copied[stream][rank] = target
        if set(copied[stream]) != {0, 1}:
            raise RuntimeError(f"missing redirected {stream} logs: {copied[stream]}")
    return copied


def append_bounded_output(combined, combined_bytes, line):
    next_bytes = combined_bytes + len(line.encode("utf-8"))
    if next_bytes > MAX_COMBINED_LOG_BYTES:
        raise RuntimeError("combined rank-0 tee log exceeded bounded capture")
    combined.append(line)
    return next_bytes


def stop_process_group_and_reap(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        return proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return proc.wait(timeout=10)


def run_solver(name, mode):
    run_dir = RUNS_DIR / name
    history_dir = HISTORY_ROOT / name
    for path in (run_dir, history_dir):
        if path.exists():
            shutil.rmtree(path)
    run_dir.mkdir(parents=True)
    history_dir.mkdir(parents=True)
    torchrun_dir = run_dir / "torchrun"
    result_tsv = run_dir / "solutions.tsv"
    env = base_env(name, history_dir)
    if mode == "collect":
        env.update({
            "BEAM_SOLVE_BUCKET_MODE": "1",
            "BEAM_SOLVE_BUCKET_STOP_DEPTH": str(DEPTH_LIMIT),
            "BEAM_SOLVE_BUCKET_MAX_SOLUTIONS": str(MAX_UNIQUE_SOLUTIONS),
            "BEAM_SOLVED_RESULT_CAPACITY": str(SOLVED_RESULT_CAPACITY),
            "BEAM_SOLVE_BUCKET_KNOWN_LENGTH": "1",
            "BEAM_SOLVE_BUCKET_RESULT_TSV": str(result_tsv),
        })
    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--no-python",
        "--nnodes=1", "--nproc-per-node=2", "--node-rank=0",
        "--rdzv-backend=c10d", f"--rdzv-endpoint=127.0.0.1:{free_port()}",
        f"--rdzv-id=task5-{name}", f"--log-dir={torchrun_dir}",
        "--redirects=3", "--tee=0:3", str(BUILD_DIR / "production_runner"),
        str(PUZZLE_ID), str(DEPTH_LIMIT), str(BEAM_WIDTH),
    ]
    started = time.perf_counter()
    deadline = time.monotonic() + RUN_TIMEOUT_SEC
    proc = subprocess.Popen(
        list(map(str, cmd)), cwd=REPO_DIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True,
    )
    try:
        assert proc.stdout is not None
        selector = selectors.DefaultSelector()
        try:
            selector.register(proc.stdout, selectors.EVENT_READ)
            combined = []
            combined_bytes = 0
            timed_out = False
            peak_rss = 0
            sample_count = 0
            samples = deque(maxlen=PROCESS_RSS_MAX_SAMPLES)
            gpu_before = gpu_snapshot()
            while proc.poll() is None:
                for key, _ in selector.select(timeout=0.10):
                    line = key.fileobj.readline()
                    if line:
                        combined_bytes = append_bounded_output(
                            combined, combined_bytes, line
                        )
                        if any(marker in line for marker in (
                            "depth_done=", "puzzle_solved=", "collection_status=", "solve_bucket_stop=",
                        )):
                            print(line, end="", flush=True)
                rss, process_count = process_tree_rss(proc.pid)
                peak_rss = max(peak_rss, rss)
                sample_count += 1
                samples.append({
                    "elapsed_sec": time.perf_counter() - started,
                    "rss_bytes": rss,
                    "process_count": process_count,
                })
                if time.monotonic() >= deadline:
                    timed_out = True
                    stop_process_group_and_reap(proc)
                    break
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                combined_bytes = append_bounded_output(combined, combined_bytes, line)
            return_code = proc.wait(timeout=10)
        finally:
            selector.close()
    except BaseException as error:
        try:
            stop_process_group_and_reap(proc)
        except BaseException as cleanup_error:
            error.add_note(f"process-group cleanup failed: {cleanup_error!r}")
        raise
    elapsed = time.perf_counter() - started
    combined_text = "".join(combined)
    (run_dir / "combined.log").write_text(
        combined_text, encoding="utf-8", newline="\n"
    )
    copied = discover_rank_logs(torchrun_dir, run_dir)
    rank_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for group in copied.values() for path in group.values()
    )
    diagnostic_text = (combined_text + "\n" + rank_text).lower()
    fatal_hits = [pattern for pattern in FATAL_PATTERNS if pattern in diagnostic_text]
    result = {
        "name": name, "mode": mode, "command": list(map(str, cmd)),
        "return_code": return_code,
        "rank_return_codes": {"0": 0 if return_code == 0 else None, "1": 0 if return_code == 0 else None},
        "rank_return_code_basis": "torchrun rc=0 requires every local worker to exit zero",
        "timed_out": timed_out, "elapsed_sec": elapsed,
        "peak_process_tree_rss_bytes": peak_rss,
        "rss_sample_count": sample_count, "rss_samples_retained": len(samples),
        "rss_samples": list(samples),
        "combined_log_bytes": len(combined_text.encode("utf-8")),
        "combined_log_sha256": sha256((run_dir / "combined.log").read_bytes()).hexdigest(),
        "rank_stdout_count": len(copied["stdout"]),
        "rank_stderr_count": len(copied["stderr"]),
        "gpu_before": gpu_before, "gpu_after": gpu_snapshot(),
        "fatal_hits": fatal_hits,
        "normal_completion_both_ranks": return_code == 0 and not timed_out and not fatal_hits,
    }
    (run_dir / "run_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if timed_out or return_code != 0 or fatal_hits or peak_rss <= 0:
        raise RuntimeError(f"{name} failed acceptance: {result}")
    if history_dir.exists():
        shutil.rmtree(history_dir)
    return result


def load_contract():
    puzzle_info = json.loads((REPO_DIR / "data/puzzle_info.json").read_text(encoding="utf-8"))
    with (REPO_DIR / "data/test.csv").open(newline="", encoding="utf-8") as handle:
        rows = {int(row["initial_state_id"]): row for row in csv.DictReader(handle)}
    initial = tuple(map(int, rows[PUZZLE_ID]["initial_state"].split(",")))
    central = tuple(puzzle_info["central_state"])
    generators = {
        name: tuple(permutation)
        for name, permutation in puzzle_info["generators"].items()
    }
    return initial, central, generators


def validate_path(path, initial, central, generators):
    current = tuple(initial)
    tokens = () if path == "" else tuple(path.split("."))
    for token in tokens:
        if token not in generators:
            return False
        current = tuple(current[index] for index in generators[token])
    return current == central


first = run_solver("first", "first")
collect_a = run_solver("collect_a", "collect")
collect_b = run_solver("collect_b", "collect")

first_combined = (RUNS_DIR / "first/combined.log").read_text(encoding="utf-8")
release_match = RELEASE_RE.search(first_combined)
if release_match is None:
    raise RuntimeError("real [default0]: first-mode release line with depth fields is missing")
release = release_match.groupdict()
release.update({
    key: int(release[key])
    for key in ("solution_length", "found_depth", "touch_depth")
})
release["seconds"] = float(release["seconds"])
if release["solution_length"] != release["found_depth"] + release["touch_depth"]:
    raise RuntimeError(f"first-mode depth decomposition mismatch: {release}")

initial, central, generators = load_contract()
cpu_solution_valid = validate_path(release["solution"], initial, central, generators)
if not cpu_solution_valid:
    raise RuntimeError(f"first-mode CPU solution validation failed: {release}")

collect_path_a = RUNS_DIR / "collect_a/solutions.tsv"
collect_path_b = RUNS_DIR / "collect_b/solutions.tsv"
collect_bytes_a = collect_path_a.read_bytes()
collect_bytes_b = collect_path_b.read_bytes()
if collect_bytes_a != collect_bytes_b:
    raise RuntimeError("repeat collect TSV files are not byte-identical")
collect_sha256 = sha256(collect_bytes_a).hexdigest()
with collect_path_a.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    if tuple(reader.fieldnames or ()) != TSV_FIELDS:
        raise RuntimeError(f"unexpected collect TSV schema: {reader.fieldnames}")
    collect_rows = list(reader)
if not (1 <= len(collect_rows) <= MAX_UNIQUE_SOLUTIONS):
    raise RuntimeError(f"collect row count outside 1..{MAX_UNIQUE_SOLUTIONS}: {len(collect_rows)}")
paths = [row["solution_path"] for row in collect_rows]
if len(paths) != len(set(paths)):
    raise RuntimeError("collect TSV contains duplicate solution paths")
if any(int(row["puzzle_id"]) != PUZZLE_ID for row in collect_rows):
    raise RuntimeError("collect TSV contains the wrong puzzle id")
if not all(validate_path(path, initial, central, generators) for path in paths):
    raise RuntimeError("collect TSV contains a CPU-invalid solution")

collection_status = {}
for name in ("collect_a", "collect_b"):
    text = (RUNS_DIR / name / "combined.log").read_text(encoding="utf-8")
    match = re.search(
        r"^\[default0\]:collection_status=(depth_reached|capacity_reached)$",
        text,
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"{name} has no normal collection_status line")
    collection_status[name] = match.group(1)
if collection_status["collect_a"] != collection_status["collect_b"]:
    raise RuntimeError(f"repeat collection statuses differ: {collection_status}")

source_manifest = json.loads((WORK_DIR / "source_manifest.json").read_text(encoding="utf-8"))
gate_summary = {
    "status": "ok",
    "kernel_slug": KERNEL_SLUG,
    "base_git_rev": BASE_GIT_REV,
    "reviewed_commit": REVIEWED_COMMIT,
    "source_sha256": SOURCE_SHA256,
    "binary_sha256": source_manifest["binary_sha256"],
    "cutlass_git_rev": CUTLASS_GIT_REV,
    "hardware": json.loads(
        (WORK_DIR / "environment.json").read_text(encoding="utf-8")
    )["gpu_rows"],
    "build": source_manifest,
    "parameters": {
        "puzzle_id": PUZZLE_ID, "beam_width": BEAM_WIDTH,
        "depth_limit": DEPTH_LIMIT, "max_unique_solutions": MAX_UNIQUE_SOLUTIONS,
        "solved_result_capacity": SOLVED_RESULT_CAPACITY,
        "k1_radius": K1_RADIUS, "k2_radius": K2_RADIUS, "profile": PROFILE,
    },
    "first_release_line": release_match.group(0),
    "first_release": release,
    "cpu_solution_valid": cpu_solution_valid,
    "collection_status": collection_status,
    "collect_tsv_schema": list(TSV_FIELDS),
    "collect_tsv_rows": len(collect_rows),
    "collect_tsv_unique_paths": len(set(paths)),
    "collect_tsv_bytes": len(collect_bytes_a),
    "collect_tsv_sha256": collect_sha256,
    "collect_tsv_byte_identical": True,
    "runs": {"first": first, "collect_a": collect_a, "collect_b": collect_b},
    "no_overflow_oom_timeout_or_collective_hang": True,
    "injected_rank0_failure": "source-test-only-no-safe-runtime-hook",
}
(WORK_DIR / "gate_summary.json").write_text(
    json.dumps(gate_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print("TASK5_GATE_GREEN", json.dumps(gate_summary, sort_keys=True), flush=True)
'''


def build_notebook(
    out_dir: Path = OUT_DIR,
    source_path: Path = SOURCE_PATH,
) -> tuple[Path, Path]:
    source = Path(source_path).read_bytes()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = out_dir / OUT_NOTEBOOK.name
    metadata_path = out_dir / "kernel-metadata.json"
    notebook = {
        "cells": [
            _markdown_cell(
                "# CayleyPy Task 5 private real-2xT4 acceptance gate\n\n"
                "Pinned public-base build with a compressed, SHA-256-attested reviewed "
                "`production_runner.cu` overlay. This private gate runs first mode once "
                "and collect mode twice on exactly two T4 GPUs.",
                "intro",
            ),
            _code_cell(_config_cell(source), "config"),
            _code_cell(SETUP_CELL, "setup-build"),
            _code_cell(RUN_CELL, "run-validate"),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    metadata = {
        "id": KERNEL_SLUG,
        "title": "CayleyPy Public Task 5 2xT4 Gate",
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if decode_embedded_source(notebook_path) != source:
        raise RuntimeError("generated notebook source payload did not roundtrip")
    return notebook_path, metadata_path


def decode_embedded_source(notebook_path: Path) -> bytes:
    notebook = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    config_source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("id") == "config"
    )
    values: dict[str, object] = {}
    for node in ast.parse(config_source).body:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        try:
            values[node.targets[0].id] = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            pass
    payload = values.get("SOURCE_PAYLOAD_B64")
    if not isinstance(payload, str):
        raise ValueError("generated notebook has no literal source payload")
    source = zlib.decompress(base64.b64decode(payload, validate=True))
    if len(source) != values.get("SOURCE_SIZE") or _digest(source) != values.get("SOURCE_SHA256"):
        raise ValueError("generated notebook source payload failed attestation")
    return source


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _scan_text_artifacts(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".json", ".log", ".tsv", ".txt", ".ipynb",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        hits = [pattern for pattern in _PRIVATE_PATTERNS if pattern in text]
        if hits:
            raise ValueError(f"private or secret marker in {path}: {hits}")


def _expected_parameters() -> dict[str, Any]:
    return {
        "puzzle_id": PUZZLE_ID,
        "beam_width": BEAM_WIDTH,
        "depth_limit": DEPTH_LIMIT,
        "max_unique_solutions": MAX_UNIQUE_SOLUTIONS,
        "solved_result_capacity": SOLVED_RESULT_CAPACITY,
        "k1_radius": K1_RADIUS,
        "k2_radius": K2_RADIUS,
        "profile": MEASURED_PROFILE,
    }


def _load_local_contract() -> tuple[
    tuple[int, ...], tuple[int, ...], dict[str, tuple[int, ...]]
]:
    info = json.loads(PUZZLE_INFO_PATH.read_text(encoding="utf-8"))
    with TEST_CSV_PATH.open(newline="", encoding="utf-8") as handle:
        rows = {int(row["initial_state_id"]): row for row in csv.DictReader(handle)}
    initial = tuple(map(int, rows[PUZZLE_ID]["initial_state"].split(",")))
    central = tuple(map(int, info["central_state"]))
    generators = {
        name: tuple(map(int, permutation))
        for name, permutation in info["generators"].items()
    }
    return initial, central, generators


def _path_tokens(path: str) -> tuple[str, ...]:
    return () if path == "" else tuple(path.split("."))


def _solution_path_is_valid(
    path: str,
    initial: tuple[int, ...],
    central: tuple[int, ...],
    generators: dict[str, tuple[int, ...]],
) -> bool:
    current = initial
    for token in _path_tokens(path):
        permutation = generators.get(token)
        if permutation is None:
            return False
        current = tuple(current[index] for index in permutation)
    return current == central


def _parse_release_line(line: str) -> dict[str, Any]:
    match = _FIRST_RELEASE_RE.fullmatch(line)
    if match is None:
        raise ValueError("gate lacks the real [default0]: release line with depth fields")
    release: dict[str, Any] = match.groupdict()
    for key in ("solution_length", "found_depth", "touch_depth"):
        release[key] = int(release[key])
    release["seconds"] = float(release["seconds"])
    return release


def _validate_remote_attestation(root: Path) -> dict[str, Any]:
    remote = root / "remote"
    raw_names = (
        "push_receipt.txt", "status.txt", "kernel-metadata.json",
        "pulled-notebook.ipynb",
    )
    raw = {name: (remote / name).read_bytes() for name in raw_names}
    capture = _read_json(remote / "capture_manifest.json")
    expected_hashes = {name: _digest(payload) for name, payload in raw.items()}
    if capture.get("sha256") != expected_hashes:
        raise ValueError("raw remote evidence hashes do not match the capture manifest")

    push_lines = [
        line for line in raw["push_receipt.txt"].decode("utf-8").splitlines()
        if line
    ]
    push_pattern = re.compile(
        r"Kernel version (?P<version>\d+) successfully pushed\.\s+Please check progress at "
        rf"https://www\.kaggle\.com/code/{re.escape(KERNEL_SLUG)}",
    )
    warning_pattern = re.compile(
        r"Warning: Looks like you're using an outdated `kaggle` version "
        r"\(installed: [0-9.]+\), please consider upgrading to the latest version \([0-9.]+\)"
    )
    push_matches = [push_pattern.fullmatch(line) for line in push_lines]
    push_matches = [match for match in push_matches if match is not None]
    non_success = [line for line in push_lines if push_pattern.fullmatch(line) is None]
    if (
        len(push_matches) != 1
        or int(push_matches[0].group("version")) != KERNEL_VERSION
        or any(warning_pattern.fullmatch(line) is None for line in non_success)
    ):
        raise ValueError("push receipt lacks the exact slug and pushed version")

    expected_status = f'{KERNEL_SLUG} has status "KernelWorkerStatus.COMPLETE"'
    status_lines = [
        line for line in raw["status.txt"].decode("utf-8").splitlines()
        if line
    ]
    status_matches = [line for line in status_lines if line == expected_status]
    status_other = [line for line in status_lines if line != expected_status]
    if (
        status_matches != [expected_status]
        or any(warning_pattern.fullmatch(line) is None for line in status_other)
    ):
        raise ValueError("status receipt is not exact COMPLETE for the expected slug")

    pulled = json.loads(raw["kernel-metadata.json"].decode("utf-8"))
    expected_pulled = {
        "id": KERNEL_SLUG,
        "title": "CayleyPy Public Task 5 2xT4 Gate",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
    }
    if any(pulled.get(key) != value for key, value in expected_pulled.items()):
        raise ValueError("pulled metadata lacks the exact private 2xT4 slug contract")
    if not str(pulled.get("code_file", "")).endswith(".ipynb"):
        raise ValueError("pulled metadata lacks a notebook code_file")

    pushed_path = SOURCE_PATH.parents[1] / OUT_NOTEBOOK
    pushed_notebook = json.loads(pushed_path.read_text(encoding="utf-8"))
    pulled_notebook = json.loads(raw["pulled-notebook.ipynb"].decode("utf-8"))

    def normalize_notebook(notebook):
        normalized = json.loads(json.dumps(notebook))
        for cell in normalized.get("cells", []):
            source = cell.get("source")
            if isinstance(source, list):
                cell["source"] = "".join(source)
        return normalized

    if normalize_notebook(pulled_notebook) != normalize_notebook(pushed_notebook):
        raise ValueError("pulled notebook is not semantically equal to the pushed package")
    pushed_notebook_sha = _digest(pushed_path.read_bytes())
    pulled_notebook_sha = _digest(raw["pulled-notebook.ipynb"])

    observed: dict[str, datetime] = {}
    for key in ("push_observed_at_utc", "completion_observed_at_utc"):
        try:
            value = datetime.fromisoformat(str(capture[key]).replace("Z", "+00:00"))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"capture manifest lacks {key}") from exc
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"capture manifest {key} is not timezone-aware")
        observed[key] = value.astimezone(timezone.utc)
    if observed["push_observed_at_utc"] > observed["completion_observed_at_utc"]:
        raise ValueError("push observation is later than completion observation")
    return {
        "slug": KERNEL_SLUG,
        "private": True,
        "pushed_version": KERNEL_VERSION,
        "status": "COMPLETE",
        "completion_observed_at_utc": observed["completion_observed_at_utc"].isoformat(),
        "pushed_notebook_sha256": pushed_notebook_sha,
        "pulled_notebook_sha256": pulled_notebook_sha,
        "pulled_notebook_semantic_match": True,
    }


def validate_gate_output(root: Path) -> dict[str, Any]:
    """Independently validate downloaded raw Kaggle evidence."""
    root = Path(root)
    try:
        remote_attestation = _validate_remote_attestation(root)
    except Exception as exc:
        raise ValueError(f"remote Kaggle attestation is invalid: {exc}") from exc
    manifest = _read_json(root / "source_manifest.json")
    summary = _read_json(root / "gate_summary.json")
    expected_source_sha = _digest(SOURCE_PATH.read_bytes())
    expected_manifest = {
        "kernel_slug": KERNEL_SLUG,
        "base_git_rev": BASE_GIT_REV,
        "reviewed_commit": REVIEWED_COMMIT,
        "production_runner_sha256": expected_source_sha,
        "production_runner_bytes": SOURCE_PATH.stat().st_size,
        "cutlass_git_rev": CUTLASS_GIT_REV,
        "build_type": "Release",
        "cuda_architectures": "75",
        "nccl_linked": True,
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise ValueError("downloaded source/build provenance does not match the gate contract")
    for key in ("base_production_runner_sha256", "binary_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(manifest.get(key, ""))) is None:
            raise ValueError(f"downloaded {key} is malformed")
    if int(manifest.get("binary_bytes", 0)) <= 0:
        raise ValueError("downloaded binary size attestation is invalid")
    timings = manifest.get("timings_sec")
    if not isinstance(timings, dict) or set(timings) != {
        "base_clone", "cutlass_clone", "configure", "compile"
    }:
        raise ValueError("downloaded build timings are incomplete")
    if any(float(value) < 0 for value in timings.values()):
        raise ValueError("downloaded build timing is negative")

    for artifact in ("build.log", "CMakeCache.txt", "environment.json", "weights_manifest.json"):
        artifact_path = root / artifact
        if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
            raise ValueError(f"downloaded build artifact is missing: {artifact}")
    build_log = (root / "build.log").read_text(encoding="utf-8", errors="replace")
    for marker in ("-DCMAKE_BUILD_TYPE=Release", "-DBEAM_CUDA_ARCHITECTURES=75", "libnccl"):
        if marker.lower() not in build_log.lower():
            raise ValueError(f"build log lacks required attestation: {marker}")
    cache = (root / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
    if "CMAKE_BUILD_TYPE:STRING=Release" not in cache:
        raise ValueError("CMake cache does not attest a Release build")
    weights = _read_json(root / "weights_manifest.json")
    if weights.get("model") != {
        "state_len": 120,
        "num_classes": 120,
        "output_dim": 24,
        "dtype": "fp16",
    }:
        raise ValueError("downloaded weights are not the tracked output_dim=24 fp16 model")

    expected_summary = {
        "status": "ok",
        "kernel_slug": KERNEL_SLUG,
        "base_git_rev": BASE_GIT_REV,
        "reviewed_commit": REVIEWED_COMMIT,
        "source_sha256": expected_source_sha,
        "binary_sha256": manifest["binary_sha256"],
        "cutlass_git_rev": CUTLASS_GIT_REV,
        "build": manifest,
        "parameters": _expected_parameters(),
        "cpu_solution_valid": True,
        "collect_tsv_byte_identical": True,
        "no_overflow_oom_timeout_or_collective_hang": True,
        "injected_rank0_failure": "source-test-only-no-safe-runtime-hook",
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise ValueError("gate summary does not match the exact source/build/run contract")

    environment = _read_json(root / "environment.json")
    hardware = summary.get("hardware")
    if hardware != environment.get("gpu_rows") or not isinstance(hardware, list):
        raise ValueError("summary hardware does not match the raw environment evidence")
    if len(hardware) != 2 or any(
        not any(name in str(row) for name in ("Tesla T4", "NVIDIA T4"))
        for row in hardware
    ):
        raise ValueError(f"gate hardware is not exactly two T4 GPUs: {hardware}")

    results: dict[str, dict[str, Any]] = {}
    combined_logs: dict[str, str] = {}
    for name, expected_mode in (
        ("first", "first"), ("collect_a", "collect"), ("collect_b", "collect")
    ):
        run_dir = root / "runs" / name
        result = _read_json(run_dir / "run_result.json")
        results[name] = result
        combined_path = run_dir / "combined.log"
        combined = combined_path.read_text(encoding="utf-8", errors="replace")
        combined_logs[name] = combined
        if summary.get("runs", {}).get(name) != result:
            raise ValueError(f"{name} summary result does not match raw evidence")
        if (
            result.get("name") != name
            or result.get("mode") != expected_mode
            or result.get("return_code") != 0
            or result.get("rank_return_codes") != {"0": 0, "1": 0}
            or result.get("timed_out") is not False
            or result.get("normal_completion_both_ranks") is not True
            or result.get("fatal_hits") != []
        ):
            raise ValueError(f"{name} did not complete normally on both ranks: {result}")
        command = [str(value) for value in result.get("command", [])]
        if (
            "--nproc-per-node=2" not in command
            or "--redirects=3" not in command
            or "--tee=0:3" not in command
            or command[-3:] != [str(PUZZLE_ID), str(DEPTH_LIMIT), str(BEAM_WIDTH)]
        ):
            raise ValueError(f"{name} command does not attest the exact 2-rank run contract")
        if int(result.get("peak_process_tree_rss_bytes", 0)) <= 0:
            raise ValueError(f"{name} lacks bounded process-tree RSS evidence")
        retained = int(result.get("rss_samples_retained", 0))
        samples = result.get("rss_samples")
        if not isinstance(samples, list) or retained != len(samples) or not (1 <= retained <= 512):
            raise ValueError(f"{name} RSS capture is missing or unbounded")
        if not any(int(sample.get("rss_bytes", 0)) > 0 for sample in samples):
            raise ValueError(f"{name} RSS samples contain no live process-tree observation")
        if result.get("rank_stdout_count") != 2 or result.get("rank_stderr_count") != 2:
            raise ValueError(f"{name} lacks both redirected rank logs")
        for rank in (0, 1):
            for stream in ("stdout", "stderr"):
                if not (run_dir / f"rank{rank}.{stream}.log").is_file():
                    raise ValueError(f"{name} is missing rank {rank} {stream}")
        if result.get("combined_log_sha256") != _digest(combined_path.read_bytes()):
            raise ValueError(f"{name} combined log SHA-256 does not match raw evidence")
        for gpu_key in ("gpu_before", "gpu_after"):
            gpu_rows = result.get(gpu_key)
            if not isinstance(gpu_rows, list) or len(gpu_rows) != 2 or any(
                "T4" not in str(row) for row in gpu_rows
            ):
                raise ValueError(f"{name} lacks exact two-T4 {gpu_key} evidence")
        diagnostic = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in [combined_path, *run_dir.glob("rank*.log")]
        ).lower()
        hits = [pattern for pattern in _FATAL_PATTERNS if pattern in diagnostic]
        if hits:
            raise ValueError(f"{name} contains fatal runtime markers: {hits}")

    release_line = str(summary.get("first_release_line", ""))
    release = _parse_release_line(release_line)
    raw_release_lines = [
        line for line in combined_logs["first"].splitlines()
        if _FIRST_RELEASE_RE.fullmatch(line) is not None
    ]
    if raw_release_lines != [release_line]:
        raise ValueError("first raw log does not contain exactly the summarized release line")
    if summary.get("first_release") != release:
        raise ValueError("parsed first release fields do not match the summary")
    if release["solution_length"] != release["found_depth"] + release["touch_depth"]:
        raise ValueError("first solution depth decomposition is invalid")
    if len(_path_tokens(release["solution"])) != release["solution_length"]:
        raise ValueError("first solution_length does not match its token count")
    initial, central, generators = _load_local_contract()
    if not _solution_path_is_valid(release["solution"], initial, central, generators):
        raise ValueError("first solution is CPU-invalid under the local puzzle contract")

    collect_a = (root / "runs/collect_a/solutions.tsv").read_bytes()
    collect_b = (root / "runs/collect_b/solutions.tsv").read_bytes()
    if collect_a != collect_b:
        raise ValueError("repeat collect TSV output is not byte-identical")
    with (root / "runs/collect_a/solutions.tsv").open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != _TSV_FIELDS:
            raise ValueError(f"unexpected collect TSV schema: {reader.fieldnames}")
        rows = list(reader)
    if not (1 <= len(rows) <= MAX_UNIQUE_SOLUTIONS):
        raise ValueError(f"collect TSV row count outside 1..{MAX_UNIQUE_SOLUTIONS}: {len(rows)}")
    paths = [row["solution_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("collect TSV solution paths are not unique")
    for row in rows:
        try:
            puzzle_id = int(row["puzzle_id"])
            depth_index = int(row["depth_index"])
            found_depth = int(row["found_depth"])
            total_depth = int(row["total_depth"])
            known_length = int(row["known_length"])
            delta = int(row["delta"])
            owner_rank = int(row["owner_rank"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"collect TSV has a non-integer contract field: {row}") from exc
        if (
            puzzle_id != PUZZLE_ID
            or not (0 <= depth_index <= DEPTH_LIMIT)
            or not (0 <= found_depth <= total_depth)
            or known_length != 1
            or delta != total_depth - known_length
            or owner_rank not in (0, 1)
        ):
            raise ValueError(f"collect TSV row violates the configured contract: {row}")
        if len(_path_tokens(row["solution_path"])) != total_depth:
            raise ValueError("collect solution_length does not match its token count")
        if not _solution_path_is_valid(row["solution_path"], initial, central, generators):
            raise ValueError("collect TSV contains a CPU-invalid solution path")

    collect_sha = _digest(collect_a)
    expected_collect_summary = {
        "collect_tsv_schema": list(_TSV_FIELDS),
        "collect_tsv_rows": len(rows),
        "collect_tsv_unique_paths": len(paths),
        "collect_tsv_bytes": len(collect_a),
        "collect_tsv_sha256": collect_sha,
    }
    if any(summary.get(key) != value for key, value in expected_collect_summary.items()):
        raise ValueError("collect TSV summary does not match the raw deterministic artifact")
    observed_status: dict[str, str] = {}
    for name in ("collect_a", "collect_b"):
        matches = re.findall(
            r"^\[default0\]:collection_status=(depth_reached|capacity_reached)$",
            combined_logs[name],
            re.MULTILINE,
        )
        if len(matches) != 1:
            raise ValueError(f"{name} lacks exactly one normal collection completion status")
        observed_status[name] = matches[0]
    if observed_status["collect_a"] != observed_status["collect_b"]:
        raise ValueError("repeat collect runs ended with different normal statuses")
    if summary.get("collection_status") != observed_status:
        raise ValueError("collection statuses do not match the raw logs")

    _scan_text_artifacts(root)
    return {
        "status": "ok",
        "collect_tsv_sha256": collect_sha,
        "collect_tsv_rows": len(rows),
        "source_sha256": expected_source_sha,
        "binary_sha256": manifest["binary_sha256"],
        "remote_attestation": remote_attestation,
    }


def main() -> None:
    notebook_path, metadata_path = build_notebook()
    print(f"wrote {notebook_path}")
    print(f"wrote {metadata_path}")
    print(f"embedded_source_sha256={_digest(decode_embedded_source(notebook_path))}")


if __name__ == "__main__":
    main()
