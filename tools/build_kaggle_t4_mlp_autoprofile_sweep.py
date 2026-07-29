#!/usr/bin/env python3
"""Build the private Kaggle 2xT4 MLP profile-replication notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


OUT_DIR = Path("kaggle_t4_mlp_autoprofile_sweep")
OUT_NOTEBOOK = OUT_DIR / "t4-mlp-autoprofile-sweep.ipynb"
PROFILE_REGISTRY_PATH = Path("configs/kaggle_t4_mlp_profiles.json")
# The replication gate pins the exact solver revision used by accepted v4/v5
# evidence; Kaggle must not silently benchmark whatever ``main`` became later.
PINNED_SOLVER_COMMIT = "6f95bd6bdb32b5f6ef7cca32b96967bce6036503"


CONFIG_CELL = r'''
import hashlib
import json
from pathlib import Path

# Fixed real-hardware replication contract. This is deliberately not a broad
# optimizer: it adjudicates two suspicious alternatives and measures controls.
OUTPUT_CLASSES = ('output1', 'output_move_count')
BEAM_POWERS = tuple(range(16, 26))
DEPTH_LIMIT = 9  # runner emits 0..8; metric is depth_done=8
WARMUP_DEPTH_LIMIT = 4
MEASURED_REPETITIONS = 3
MATERIAL_IMPROVEMENT_RATIO = 0.95  # alternate median must be >=5% faster
INTER_ATTEMPT_COOLDOWN_SEC = 10  # let torchrun/NCCL teardown finish before next pair
# Only canonical/alternate pairs may change selected_profiles. Controls are
# evidence-only and never modify a tuple.
REPLICATION_TARGETS = (
    ('output_move_count', 16, 'canonical', 4),
    ('output_move_count', 16, 'alternate', 2),
    ('output_move_count', 19, 'canonical', 4),
    ('output_move_count', 19, 'alternate', 8),
    ('output1', 19, 'control_same_tuple', 16),
    ('output_move_count', 17, 'control_same_tuple', 2),
    ('output_move_count', 20, 'control_same_tuple', 8),
    ('output_move_count', 21, 'control_same_tuple', 8),
)
TORCHRUN_NNODES = 1
TORCHRUN_NPROC_PER_NODE = 2
TORCHRUN_NODE_RANK = 0
TORCHRUN_RDZV_BACKEND = "c10d"
TORCHRUN_RDZV_ENDPOINT = "127.0.0.1:29500"

GITHUB_REPO_URL = "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git"
GITHUB_BRANCH = "main"
PINNED_SOLVER_COMMIT = "__PINNED_SOLVER_COMMIT__"
CUDA_ARCHITECTURES = "75"
PUZZLE_ID = 0
RUN_TIMEOUT_SEC = 120  # hard external cap per depth-8 torchrun attempt
PROCESS_GROUP_TERM_GRACE_SEC = 10
STREAM4_BATCH_ALIGNMENT = 1024
SHARD_CAPACITY_SCALE_PPM = 1050000
GPU_HEADROOM_BYTES = 224 * 1024**2
HISTORY_RAM_BYTES = 28 * 1024**3
HISTORY_DISK_BYTES = 32 * 1024**3
OUTPUT1_CHECKPOINT = Path(
    "/kaggle/input/models/arabidopsisthalian/"
    "megaminx2048-512-8-e4000/pytorch/default/1/"
    "weights_megaminx2048_512_8_e4000.pth"
)
'''


SETUP_CELL = r'''
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

WORK_DIR = Path("/kaggle/working")
TMP_DIR = Path("/tmp")
REPO_DIR = TMP_DIR / "beam_autoprofile_repo"
CUTLASS_DIR = TMP_DIR / "cutlass"
BUILD_DIR = TMP_DIR / "beam_autoprofile_build"
OUTPUT1_DIR = TMP_DIR / "stream1_output1"
OUTPUT24_DIR = TMP_DIR / "stream1_output24"


def checked(cmd, cwd=None, env=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, check=True)


gpu_rows = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ],
    text=True,
).strip().splitlines()
gpu_names = [row.split(",", 1)[0].strip() for row in gpu_rows]
if len(gpu_rows) != 2 or any(name not in {"Tesla T4", "NVIDIA T4"} for name in gpu_names):
    raise RuntimeError(f"expected exactly two NVIDIA T4 GPUs; observed={gpu_rows!r}")
print("validated_hardware=", gpu_rows)

for path in (REPO_DIR, BUILD_DIR, OUTPUT1_DIR, OUTPUT24_DIR):
    if path.exists():
        shutil.rmtree(path)
checked(["git", "clone", "--branch", GITHUB_BRANCH, "--depth", "1", GITHUB_REPO_URL, REPO_DIR])
checked(["git", "fetch", "--depth", "1", "origin", PINNED_SOLVER_COMMIT], cwd=REPO_DIR)
checked(["git", "checkout", "--detach", PINNED_SOLVER_COMMIT], cwd=REPO_DIR)
actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True).strip()
if actual_commit != PINNED_SOLVER_COMMIT:
    raise RuntimeError(f"pinned solver commit mismatch: {actual_commit} != {PINNED_SOLVER_COMMIT}")
if not OUTPUT1_CHECKPOINT.is_file():
    raise FileNotFoundError(f"output1 checkpoint not found: {OUTPUT1_CHECKPOINT}")
checked(
    [
        "python3",
        REPO_DIR / "tools/export_stream1_mlp.py",
        "--weights",
        OUTPUT1_CHECKPOINT,
        "--out",
        OUTPUT1_DIR,
        "--format",
        "batchnorm-folded",
        "--dtype",
        "fp16",
        "--num-classes",
        "120",
    ],
    cwd=REPO_DIR,
)
shutil.copytree(REPO_DIR / "stream1_weights", OUTPUT24_DIR)
MODEL_DIRS = {
    "output1": OUTPUT1_DIR,
    "output_move_count": OUTPUT24_DIR,
}
for model_class, model_dir in MODEL_DIRS.items():
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    output_dim = int(manifest["output_dim"])
    if model_class == "output1" and output_dim != 1:
        raise RuntimeError(f"output1 manifest mismatch: {manifest}")
    if model_class == "output_move_count" and output_dim != 24:
        raise RuntimeError(f"output_move_count manifest mismatch: {manifest}")
    print("validated_model=", model_class, manifest)

def tree_sha256(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()

MODEL_ARTIFACTS = {model_class: {"tree_sha256": tree_sha256(model_dir), "manifest": json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))} for model_class, model_dir in MODEL_DIRS.items()}
print("model_artifacts=", json.dumps(MODEL_ARTIFACTS, sort_keys=True))

if not (CUTLASS_DIR / "include").exists():
    if CUTLASS_DIR.exists():
        shutil.rmtree(CUTLASS_DIR)
    checked(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/cutlass.git", CUTLASS_DIR])
checked(
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
        "-DBEAM_ENABLE_DEBUG=ON",
        "-DBEAM_ENABLE_DEPTH_LOGS=ON",
        "-DBEAM_ENABLE_DEBUG_LOGS=OFF",
        "-DBEAM_DEBUG_PIPELINE_STATS=OFF",
    ],
    cwd=REPO_DIR,
)
checked(["cmake", "--build", BUILD_DIR, "--target", "production_runner", "-j", "2"])
'''


RUNNER_CELL = r'''
import csv
import json
import math
import os
import re
import selectors
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEPTH_RE = re.compile(r"depth_done=(\d+).*?depth_sec=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")
RANK_PREFIX_RE = re.compile(r"^\[[^]]*\]:")
OUTPUT_DIM_RE = re.compile(r"stream1_model_output_dim=(\d+)")
STATIC_RE = re.compile(r"(?:static_allocation_bytes|static_bytes)=(\d+)")
FREE_RE = re.compile(r"(?:free_after_all_allocations_bytes|free_bytes)=(\d+)")


def round_up(value, alignment):
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def runtime_seed(model_class, shard_count):
    if model_class == "output1":
        b_micro = 49152
        concurrency = 4
        ring_slots = 4
    else:
        b_micro = 2048
        concurrency = 4
        ring_slots = 4
    return {
        "b_micro": b_micro,
        "stream1_concurrency": concurrency,
        "stream3_ring_slots": ring_slots,
        "stream3_batch_candidates": 196608,
        "shard_count": shard_count,
        "shard_capacity_scale_ppm": SHARD_CAPACITY_SCALE_PPM,
        "stream4_batch_candidates": 98304,
        "stream4_trigger_candidates": 98304,
        "stream4_active_sort_slots": 4,
    }


def classify_error(return_code, output, timed_out):
    low = output.lower()
    if timed_out:
        return "timeout"
    if "code=3002" in low or "stream3_double_buffer_overflow" in low:
        return "stream3_overflow"
    if "cuda graph" in low and "out of memory" in low:
        return "cuda_graph_oom"
    if "out of memory" in low:
        return "oom"
    if "invalid config" in low or "invalid runtime" in low:
        return "invalid_config"
    if return_code:
        return "rank_failure"
    return ""


def parse_measurements(output):
    by_depth = {}
    for match in DEPTH_RE.finditer(output):
        depth = int(match.group(1))
        seconds = float(match.group(2))
        by_depth[depth] = max(seconds, by_depth.get(depth, 0.0))
    measured = sorted(by_depth.items())
    steady = measured[-1][1] if measured else None
    output_dims = [int(value) for value in OUTPUT_DIM_RE.findall(output)]
    static_values = [int(value) for value in STATIC_RE.findall(output)]
    free_values = [int(value) for value in FREE_RE.findall(output)]
    return {
        "depth_timings": measured,
        "steady_state_sec": steady,
        "max_depth_completed": measured[-1][0] if measured else -1,
        "observed_output_dim": output_dims[-1] if output_dims else None,
        "static_vram_bytes": max(static_values) if static_values else None,
        "free_vram_bytes": min(free_values) if free_values else None,
    }


def derived_layout(requested_beam, runtime):
    world_size = TORCHRUN_NNODES * TORCHRUN_NPROC_PER_NODE
    shard_count = int(runtime["shard_count"])
    quantum = world_size * shard_count * STREAM4_BATCH_ALIGNMENT
    effective_beam = round_up(requested_beam, quantum)
    local_beam = effective_beam // world_size
    logical_shard = (local_beam + shard_count - 1) // shard_count
    capacity = round_up(
        (logical_shard * int(runtime["shard_capacity_scale_ppm"]) + 999999) // 1000000,
        STREAM4_BATCH_ALIGNMENT,
    )
    min_capacity = max(
        int(runtime["stream3_batch_candidates"]),
        int(runtime["stream4_batch_candidates"]),
        int(runtime["stream4_trigger_candidates"]),
    )
    capacity = max(capacity, round_up(min_capacity, STREAM4_BATCH_ALIGNMENT))
    return effective_beam, local_beam, capacity


def make_env(model_class, runtime, history_path):
    target = REPO_DIR / "stream1_weights"
    source = MODEL_DIRS[model_class]
    if target.exists() and target.resolve() != source.resolve():
        shutil.rmtree(target)
    if not target.exists():
        shutil.copytree(source, target)
    env = os.environ.copy()
    env.update(
        {
            "BEAM_RUNTIME_CONFIG_MODE": "manual",
            "BEAM_B_MICRO": str(runtime["b_micro"]),
            "BEAM_STREAM1_CONCURRENCY": str(runtime["stream1_concurrency"]),
            "BEAM_STREAM3_RING_SLOTS": str(runtime["stream3_ring_slots"]),
            "BEAM_STREAM3_BATCH_CANDIDATES": str(runtime["stream3_batch_candidates"]),
            "BEAM_SHARD_COUNT": str(runtime["shard_count"]),
            "BEAM_SHARD_BUFFER_COUNT": "2",
            "BEAM_SHARD_CAPACITY_CANDIDATES": str(runtime["shard_capacity_candidates"]),
            "BEAM_SHARD_CAPACITY_SCALE_PPM": str(runtime["shard_capacity_scale_ppm"]),
            "BEAM_STREAM4_BATCH_CANDIDATES": str(runtime["stream4_batch_candidates"]),
            "BEAM_STREAM4_TRIGGER_CANDIDATES": str(runtime["stream4_trigger_candidates"]),
            "BEAM_STREAM4_ACTIVE_SORT_SLOTS": str(runtime["stream4_active_sort_slots"]),
            "BEAM_GLOBAL_SPILL_CAPACITY": "0",
            "BEAM_STREAM5_RECV_CAPACITY_SCALE_PPM": "1000000",
            "BEAM_GPU_HEADROOM_BYTES": str(GPU_HEADROOM_BYTES),
            "BEAM_HISTORY_MODE": "ram",
            "BEAM_HISTORY_SLOT_COUNT": "2",
            "BEAM_HISTORY_WORKERS": "1",
            "BEAM_HISTORY_RAM_BYTES": str(HISTORY_RAM_BYTES),
            "BEAM_HISTORY_DISK_BYTES": str(HISTORY_DISK_BYTES),
            "BEAM_HISTORY_DISK_PATH": str(history_path),
            "BEAM_SOLVED_NEIGHBORHOOD_RADIUS": "0",
            "BEAM_STREAM2_SUFFIX_RADIUS": "0",
            "BEAM_DEPTH_LOG_EVERY": "1",
        }
    )
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        env.pop(key, None)
    return env


def run_attempt(model_class, beam_power, runtime, phase, ordinal):
    requested_beam = 2**beam_power
    effective_beam, local_beam, capacity = derived_layout(requested_beam, runtime)
    runtime = dict(runtime)
    runtime["shard_capacity_candidates"] = capacity
    config_id = (
        f"{phase}_{model_class}_p{beam_power}_sh{runtime['shard_count']}_"
        f"b{runtime['b_micro']}_{ordinal}"
    )
    history_path = Path("/tmp/beam_history_autoprofile") / config_id
    if history_path.exists():
        shutil.rmtree(history_path)
    history_path.mkdir(parents=True, exist_ok=True)
    log_path = WORK_DIR / f"{config_id}.log"
    env = make_env(model_class, runtime, history_path)
    depth_limit = WARMUP_DEPTH_LIMIT if phase == "warmup" else DEPTH_LIMIT
    rendezvous_port = 29500 + (int(hashlib.sha256(config_id.encode("utf-8")).hexdigest()[:4], 16) % 1000)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--no-python",
        f"--nnodes={TORCHRUN_NNODES}",
        f"--nproc-per-node={TORCHRUN_NPROC_PER_NODE}",
        f"--node-rank={TORCHRUN_NODE_RANK}",
        f"--rdzv-backend={TORCHRUN_RDZV_BACKEND}",
        f"--rdzv-endpoint=127.0.0.1:{rendezvous_port}",
        f"--rdzv-id={config_id}",
        f"--log-dir={WORK_DIR / (config_id + '_ranks')}",
        "--redirects=3",
        "--tee=0:3",
        str(BUILD_DIR / "production_runner"),
        str(PUZZLE_ID),
        str(depth_limit),
        str(requested_beam),
    ]
    print(
        "attempt_start",
        config_id,
        "requested_beam=",
        requested_beam,
        "effective_beam=",
        effective_beam,
        flush=True,
    )
    started = time.perf_counter()
    timed_out = False
    return_code = -999
    output = ""
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert proc.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + RUN_TIMEOUT_SEC
        lines = []
        while proc.poll() is None:
            for key, _ in selector.select(timeout=1.0):
                line = key.fileobj.readline()
                if line:
                    lines.append(line)
                    if "depth_done=" in line or "error" in line.lower():
                        print(line, end="", flush=True)
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(cmd, RUN_TIMEOUT_SEC, output="".join(lines))
        lines.extend(proc.stdout.readlines())
        return_code = proc.wait()
        output = "".join(lines)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        return_code = -200
        output = exc.output or ""
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                tail, _ = proc.communicate(timeout=PROCESS_GROUP_TERM_GRACE_SEC)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                tail, _ = proc.communicate()
            output += tail or ""
        print("attempt_timeout_cleanup", config_id, "process_group_terminated=1", flush=True)
    finally:
        elapsed = time.perf_counter() - started
        log_path.write_text(output, encoding="utf-8")
        if history_path.exists():
            shutil.rmtree(history_path)
    parsed = parse_measurements(output)
    error_class = classify_error(return_code, output, timed_out)
    expected_last_depth = depth_limit - 1
    status = (
        "ok"
        if return_code == 0 and parsed["max_depth_completed"] >= expected_last_depth
        else "failed"
    )
    steady = parsed["steady_state_sec"]
    throughput = local_beam / steady if steady and steady > 0 else None
    row = {
        "config_id": config_id,
        "phase": phase,
        "model_class": model_class,
        "output_dim": 1 if model_class == "output1" else 24,
        "beam_power": beam_power,
        "requested_beam": requested_beam,
        "effective_beam": effective_beam,
        "local_beam": local_beam,
        **runtime,
        **parsed,
        "elapsed_sec": elapsed,
        "throughput_local_frontier_per_sec": throughput,
        "return_code": return_code,
        "status": status,
        "error_class": error_class,
        "log_path": str(log_path),
    }
    print("attempt_done", json.dumps(row, default=str), flush=True)
    # v8 saw intermittent rank teardown aborts. Keep each measurement a fresh
    # process pair and give launch/NCCL cleanup a bounded quiet interval.
    time.sleep(INTER_ATTEMPT_COOLDOWN_SEC)
    return row
'''


SWEEP_CELL = r'''
from statistics import median

canonical_registry_bytes = CANONICAL_PROFILE_REGISTRY_JSON.encode("utf-8")
actual_registry_sha256 = hashlib.sha256(canonical_registry_bytes).hexdigest()
if actual_registry_sha256 != CANONICAL_PROFILE_REGISTRY_SHA256:
    raise RuntimeError("embedded canonical profile registry SHA256 mismatch")
canonical_payload = json.loads(CANONICAL_PROFILE_REGISTRY_JSON)
if canonical_payload.get("hardware") != "kaggle_2xt4":
    raise RuntimeError(f"unexpected canonical hardware: {canonical_payload.get('hardware')!r}")
canonical_profiles = canonical_payload.get("profiles", {})
expected_anchor_keys = {str(power) for power in BEAM_POWERS}
if set(canonical_profiles) != set(OUTPUT_CLASSES):
    raise RuntimeError("canonical profile model classes mismatch")
for model_class in OUTPUT_CLASSES:
    if set(canonical_profiles[model_class]) != expected_anchor_keys:
        raise RuntimeError(f"canonical profile anchors mismatch for {model_class}")
    for power in BEAM_POWERS:
        row = canonical_profiles[model_class][str(power)]
        if row.get("validation_status") != "measured" or not isinstance(row.get("runtime"), dict):
            raise RuntimeError(f"canonical profile is not measured: {model_class}/p{power}")

def canonical_runtime(model_class, beam_power, shard_count):
    runtime = dict(canonical_profiles[model_class][str(beam_power)]["runtime"])
    runtime["shard_count"] = int(shard_count)
    # The v4/v5 registry deliberately leaves stream3 sizing out of the public
    # profile. The established safe 2xT4 floor is fixed here, identically for
    # every compared tuple.
    runtime["stream3_batch_candidates"] = 196608
    return runtime

attempts = []
for model_class, beam_power, label, shard_count in REPLICATION_TARGETS:
    runtime = canonical_runtime(model_class, beam_power, shard_count)
    warmup = run_attempt(model_class, beam_power, runtime, phase="warmup", ordinal=f"{label}_warmup")
    warmup["tuple_label"] = label
    warmup["replication_index"] = -1
    attempts.append(warmup)
    for repetition in range(MEASURED_REPETITIONS):
        row = run_attempt(model_class, beam_power, runtime, phase="replication", ordinal=f"{label}_r{repetition}")
        row["tuple_label"] = label
        row["replication_index"] = repetition
        attempts.append(row)

fieldnames = sorted({key for row in attempts for key in row})
attempts_csv = WORK_DIR / "autoprofile_replication_attempts.csv"
with attempts_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in attempts:
        serialized = dict(row)
        serialized["depth_timings"] = json.dumps(serialized.get("depth_timings"))
        writer.writerow(serialized)

def tuple_rows(model_class, beam_power, label, phase):
    return [row for row in attempts if row.get("model_class") == model_class
            and row.get("beam_power") == beam_power and row.get("tuple_label") == label
            and row.get("phase") == phase]

def tuple_evidence(model_class, beam_power, label):
    warmups = tuple_rows(model_class, beam_power, label, "warmup")
    rows = tuple_rows(model_class, beam_power, label, "replication")
    timings = [row.get("steady_state_sec") for row in rows if row.get("status") == "ok"]
    all_ok = (len(warmups) == 1 and warmups[0].get("status") == "ok"
              and len(rows) == MEASURED_REPETITIONS
              and len(timings) == MEASURED_REPETITIONS)
    return {"warmup_ok": len(warmups) == 1 and warmups[0].get("status") == "ok",
            "all_runs_ok": all_ok, "timings_sec": timings,
            "median_sec": median(timings) if len(timings) == MEASURED_REPETITIONS else None,
            "attempt_ids": [row.get("config_id") for row in warmups + rows]}

selected = {"schema_version": 1, "hardware": "kaggle_2xt4",
            "selection_metric": "depth_done=8 median of three post-warmup repetitions",
            "profile_source": "canonical_v4_v5_plus_v8_bounded_replication_gate",
            "profiles": {model_class: dict(canonical_profiles[model_class]) for model_class in OUTPUT_CLASSES}}
decisions = {"schema_version": 1, "canonical_profile_sha256": CANONICAL_PROFILE_REGISTRY_SHA256,
             "pinned_solver_commit": PINNED_SOLVER_COMMIT,
             "measured_repetitions": MEASURED_REPETITIONS,
             "material_improvement_ratio": MATERIAL_IMPROVEMENT_RATIO,
             "inter_attempt_cooldown_sec": INTER_ATTEMPT_COOLDOWN_SEC,
             "per_attempt_timeout_sec": RUN_TIMEOUT_SEC,
             "process_group_term_grace_sec": PROCESS_GROUP_TERM_GRACE_SEC,
             "targets": []}
for model_class, beam_power in (("output_move_count", 16), ("output_move_count", 19)):
    base = tuple_evidence(model_class, beam_power, "canonical")
    alternate = tuple_evidence(model_class, beam_power, "alternate")
    accepted = bool(base["all_runs_ok"] and alternate["all_runs_ok"]
                    and alternate["median_sec"] <= base["median_sec"] * MATERIAL_IMPROVEMENT_RATIO)
    decision = {"model_class": model_class, "beam_power": beam_power,
                "canonical": base, "alternate": alternate,
                "accepted_alternate": accepted,
                "reason": "all runs succeeded and alternate median is materially faster" if accepted
                          else "canonical retained: incomplete/failed evidence or insufficient median improvement"}
    decisions["targets"].append(decision)
    if accepted:
        canonical = canonical_profiles[model_class][str(beam_power)]
        runtime = canonical_runtime(model_class, beam_power, 2 if beam_power == 16 else 8)
        selected["profiles"][model_class][str(beam_power)] = {
            "validation_status": "measured", "runtime": {key: runtime[key] for key in (
                "b_micro", "stream1_concurrency", "stream3_ring_slots", "shard_count",
                "shard_capacity_scale_ppm", "stream4_batch_candidates",
                "stream4_trigger_candidates", "stream4_active_sort_slots")},
            "evidence": {"config_id": alternate["attempt_ids"][-1], "depth": 8,
                         "depth_sec_median": alternate["median_sec"],
                         "replication_timings_sec": alternate["timings_sec"],
                         "selection": "v8 bounded replication gate"},
        }
for model_class, beam_power, label, shard_count in REPLICATION_TARGETS:
    if label == "control_same_tuple":
        decisions["targets"].append({"model_class": model_class, "beam_power": beam_power,
            "control_same_tuple": tuple_evidence(model_class, beam_power, label),
            "selected_change": False, "reason": "variance control only; canonical tuple cannot change"})

selected_path = WORK_DIR / "selected_profiles.json"
decisions_path = WORK_DIR / "replication_decisions.json"
selected_path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
decisions_path.write_text(json.dumps(decisions, indent=2) + "\n", encoding="utf-8")
summary = {"hardware": gpu_rows, "torchrun_world_size": TORCHRUN_NNODES * TORCHRUN_NPROC_PER_NODE,
    "selection_depth": 8, "attempt_count": len(attempts),
    "successful_count": sum(row.get("status") == "ok" for row in attempts),
    "failed_count": sum(row.get("status") != "ok" for row in attempts),
    "canonical_profile_source": f"embedded_sha256:{CANONICAL_PROFILE_REGISTRY_SHA256}",
    "repository_commit": actual_commit, "pinned_solver_commit": PINNED_SOLVER_COMMIT,
    "model_artifacts": MODEL_ARTIFACTS,
    "outputs": {"attempts_csv": str(attempts_csv), "selected_profiles": str(selected_path),
                "replication_decisions": str(decisions_path)}}
(WORK_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
'''

def _code_cell(source: str, cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


def build_notebook(out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = out_dir / OUT_NOTEBOOK.name
    metadata_path = out_dir / "kernel-metadata.json"
    registry_text = PROFILE_REGISTRY_PATH.read_text(encoding="utf-8")
    registry_payload = json.loads(registry_text)
    if registry_payload.get("hardware") != "kaggle_2xt4":
        raise ValueError("profile registry hardware must be kaggle_2xt4")
    registry_profiles = registry_payload.get("profiles", {})
    if set(registry_profiles) != {"output1", "output_move_count"}:
        raise ValueError("profile registry must contain exactly both model classes")
    expected_anchor_keys = {str(power) for power in range(16, 26)}
    for model_class in ("output1", "output_move_count"):
        if set(registry_profiles[model_class]) != expected_anchor_keys:
            raise ValueError(f"profile registry anchors mismatch for {model_class}")
    registry_sha256 = hashlib.sha256(registry_text.encode("utf-8")).hexdigest()
    config_source = CONFIG_CELL.rstrip() + "\n"
    config_source += f"PINNED_SOLVER_COMMIT = {PINNED_SOLVER_COMMIT!r}\n"
    config_source = config_source.replace("__PINNED_SOLVER_COMMIT__", PINNED_SOLVER_COMMIT)
    config_source += f"CANONICAL_PROFILE_REGISTRY_JSON = {registry_text!r}\n"
    config_source += f"CANONICAL_PROFILE_REGISTRY_SHA256 = {registry_sha256!r}\n"
    notebook = {
        "cells": [
            _code_cell(config_source, "config"),
            _code_cell(SETUP_CELL, "setup"),
            _code_cell(RUNNER_CELL, "runner"),
            _code_cell(SWEEP_CELL, "sweep"),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "id": "trydotatwo/cayley-beam-2xt4-mlp-autoprofiles",
        "title": "Cayley Beam 2xT4 MLP Autoprofiles",
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
        "model_sources": [
            "arabidopsisthalian/megaminx2048-512-8-e4000/PyTorch/default/1"
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return notebook_path, metadata_path


def main() -> None:
    notebook_path, metadata_path = build_notebook()
    print(f"wrote {notebook_path}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
