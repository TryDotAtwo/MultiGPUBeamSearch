"""Build paired private Kaggle notebooks for the paper's single-GPU sweep.

The notebooks are intentionally thin orchestrators.  The attached benchmark
bundle owns the two unmodified implementations and their checkpoint loaders;
this file owns only hardware validation, the sweep/retry policy, telemetry,
and durable reviewer-facing artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path


OUT = Path("paper/kaggle_review_sweeps")
BEAMS = [2**power for power in range(10, 26)]

# The broad MultiGPU v11 run established that very small frontiers are a
# pathological compute regime for the production runner (32,768 took almost
# six hours while using only 417 MiB).  Capacity is already evidenced at
# 25,165,824 for the scalar head, so the follow-up starts there and uses one
# broad midpoint before the reproduced 29,360,128 failure.  The output-24 head
# has no valid WORLD_SIZE=1 boundary yet, hence a sparse logarithmic grid plus
# three near-limit probes.  This is a boundary confirmation, not a repeat of
# the small-width performance curve.
MULTIGPU_BOUNDARY_BEAMS = {
    1: [25_165_824, 27_262_976, 29_360_128],
    24: [1_048_576, 4_194_304, 8_388_608, 16_777_216,
         20_971_520, 25_165_824, 29_360_128, 33_554_432],
}


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source.splitlines(keepends=True),
    }


def notebook(gpu_token: str, methods: tuple[str, ...], beams_by_output: dict[int, list[int]] | None = None) -> dict:
    title = "Tesla T4" if gpu_token == "T4" else "Tesla P100"
    method_label = "Pilgrim" if methods == ("pilgrim",) else "MultiGPUBeamSearch"
    header = f"""# Paired Megaminx beam-capacity sweep — 1×{gpu_token}

Reviewer-facing, private {method_label} benchmark notebook. It runs on one
visible {title}, fixed Megaminx state and two real
checkpoint contracts: scalar `output_dim=1` and Q `output_dim=24` (the number
of generators). It does not convert, average, or otherwise substitute heads.

Methods:

- `pilgrim`: scalar Searcher for output-1 and QSearcher for output-24;
- `multigpu_beam_search`: the unchanged `WORLD_SIZE=1` production runner.

Every point is a fresh child process. A failure cannot poison later CUDA state.
After the first failure, the notebook estimates the memory boundary from the
last completed samples and runs exactly one confirmation point below it. It
does not finely bisect the boundary. `RUNNING` is never a result.
"""
    beam_map = beams_by_output or {1: BEAMS, 24: BEAMS}
    config = f'''from pathlib import Path

EXPECTED_GPU_TOKEN = {gpu_token!r}
BEAM_WIDTHS_BY_OUTPUT = {beam_map!r}
DEPTH_LIMIT = 8
BOUNDARY_ALIGNMENT = 2**20
BOUNDARY_SAFETY = 0.95

    # Private attached bundle contract. It contains pinned source snapshots,
# the fixed p900 state/generators, provenance, and the two thin workers.
BUNDLE_GLOB = "paper-multigpu-megaminx-benchmark-bundle-v3"
EXPECTED_BUNDLE_MANIFEST_VERSION = 1
EXPECTED_GENERATORS = 24
EXPECTED_OUTPUT_DIMS = (1, 24)
METHODS = {methods!r}
OUTPUT_DIMS = EXPECTED_OUTPUT_DIMS

WORK = Path("/kaggle/working/paper_paired_sweep")
RESULTS_JSONL = WORK / "results.jsonl"
SUMMARY_JSON = WORK / "summary.json"
RESULTS_CSV = WORK / "results.csv"
'''
    setup = r'''import csv, hashlib, json, os, platform, shutil, subprocess, sys, time

# Hide Kaggle's second T4 before either worker imports torch.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
WORK.mkdir(parents=True, exist_ok=True)

gpu_rows = subprocess.check_output([
    "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader,nounits"], text=True).strip().splitlines()
if len(gpu_rows) < 1 or EXPECTED_GPU_TOKEN not in gpu_rows[0]:
    raise RuntimeError(f"requires 1x{EXPECTED_GPU_TOKEN}; physical rows={gpu_rows!r}")

bundle_matches = [p for p in Path("/kaggle/input").rglob("benchmark_manifest.json")
                  if BUNDLE_GLOB in str(p)]
if len(bundle_matches) != 1:
    raise RuntimeError(f"expected one private benchmark bundle manifest, got {bundle_matches}")
manifest = json.loads(bundle_matches[0].read_text(encoding="utf-8"))
dataset_root = bundle_matches[0].parent
payload_zip = dataset_root / "bundle_payload.zip"
if payload_zip.is_file():
    unpacked = WORK / "verified_bundle"
    if unpacked.exists():
        shutil.rmtree(unpacked)
    shutil.unpack_archive(payload_zip, unpacked)
    BUNDLE = unpacked
elif (dataset_root / "bundle_payload").is_dir():
    BUNDLE = dataset_root / "bundle_payload"
else:
    BUNDLE = dataset_root
if manifest.get("manifest_version") != EXPECTED_BUNDLE_MANIFEST_VERSION:
    raise RuntimeError("benchmark bundle manifest version mismatch")
if manifest.get("move_count") != EXPECTED_GENERATORS:
    raise RuntimeError("benchmark bundle is not the fixed 24-generator Megaminx contract")
if sorted(manifest.get("output_dims", [])) != list(EXPECTED_OUTPUT_DIMS):
    raise RuntimeError("both real output_dim=1 and output_dim=24 checkpoints are required")

# The manifest must prove two independently trained native heads. Merely naming
# files output1/output24 is insufficient for a reviewer-facing measurement.
checkpoints = manifest.get("checkpoints", {})
for output_dim in EXPECTED_OUTPUT_DIMS:
    spec = checkpoints.get(str(output_dim), {})
    required = {"model_glob", "sha256", "architecture", "output_dim", "training_run_id", "native_head"}
    missing = sorted(required - set(spec))
    if missing:
        raise RuntimeError(f"checkpoint output_dim={output_dim} lacks provenance fields: {missing}")
    if int(spec["output_dim"]) != output_dim or spec["native_head"] is not True:
        raise RuntimeError(f"checkpoint is not a native output_dim={output_dim} head: {spec}")
    if checkpoints.get("1", {}).get("training_run_id") == checkpoints.get("24", {}).get("training_run_id"):
        raise RuntimeError("output-1 and output-24 must identify distinct native training runs")
    found = list(Path("/kaggle/input/models").glob(spec["model_glob"]))
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one attached model for output_dim={output_dim}: {found}")
    checkpoint_path = found[0]
    actual = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if actual != spec["sha256"]:
        raise RuntimeError(f"checkpoint SHA256 mismatch for output_dim={output_dim}")

for rel, expected in manifest["sha256"].items():
    path = BUNDLE / rel
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch: {rel}: {actual} != {expected}")

hardware = {
    "gpu_visible": gpu_rows[0], "physical_gpu_rows": gpu_rows,
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "python": sys.version, "platform": platform.platform(),
    "bundle_manifest": manifest,
}
(WORK / "hardware.json").write_text(json.dumps(hardware, indent=2), encoding="utf-8")
print(json.dumps(hardware, indent=2), flush=True)
'''
    run = r'''def classify(returncode, timed_out, result_path, log_text):
    if timed_out:
        return "timeout"
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status"):
            return payload["status"]
    low = log_text.lower()
    if "cuda out of memory" in low or "cuda_oom" in low:
        return "cuda_oom"
    if returncode in (-9, 137) or "sigkill" in low:
        return "host_oom_or_sigkill"
    return "completed" if returncode == 0 else "process_exit"


def run_point(method, output_dim, beam, attempt):
    tag = f"{method}_out{output_dim}_beam{beam}_attempt{attempt}"
    point_dir = WORK / tag
    point_dir.mkdir(parents=True, exist_ok=True)
    result_path = point_dir / "result.json"
    log_path = point_dir / "combined.log"
    worker = BUNDLE / "workers" / f"{method}.py"
    cmd = [sys.executable, str(worker), "--bundle", str(BUNDLE),
           "--output-dim", str(output_dim), "--beam", str(beam),
           "--depth", str(DEPTH_LIMIT), "--output", str(result_path)]
    started = time.time()
    timed_out = False
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, env=os.environ.copy())
    returncode, log_text = proc.returncode, proc.stdout
    wall = time.time() - started
    log_path.write_text(log_text, encoding="utf-8", errors="replace")
    status = classify(returncode, timed_out, result_path, log_text)
    payload = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    row = {
        "method": method, "output_dim": output_dim, "beam_width": beam,
        "attempt": attempt, "status": status, "returncode": returncode,
        "orchestrator_wall_s": wall, "log_path": str(log_path), **payload,
    }
    expected_checkpoint_sha = checkpoints[str(output_dim)]["sha256"]
    # A hard timeout can kill the child before it writes result.json. The
    # checkpoint and native-head contract were already verified in preflight;
    # preserve that provenance without pretending the worker attested it.
    if timed_out and not payload:
        row.update({"checkpoint_sha256": expected_checkpoint_sha,
                    "observed_output_dim": output_dim,
                    "move_count": EXPECTED_GENERATORS,
                    "attestation_source": "orchestrator_preflight"})
    if row.get("checkpoint_sha256") != expected_checkpoint_sha:
        raise RuntimeError(f"worker did not attest the selected native checkpoint: {row}")
    if int(row.get("observed_output_dim", -1)) != output_dim:
        raise RuntimeError(f"worker observed a non-native head shape: {row}")
    if int(row.get("move_count", -1)) != EXPECTED_GENERATORS:
        raise RuntimeError(f"worker changed the generator contract: {row}")
    with RESULTS_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print("POINT_RESULT " + json.dumps(row, sort_keys=True), flush=True)
    return row


def confirmation_width(completed, failure_width):
    usable = [r for r in completed if r.get("peak_device_used_mib") is not None]
    estimate = float(failure_width)
    if len(usable) >= 2:
        left, right = usable[-2], usable[-1]
        db = right["beam_width"] - left["beam_width"]
        dm = right["peak_device_used_mib"] - left["peak_device_used_mib"]
        if db > 0 and dm > 0:
            slope = dm / db
            total_mib = float(str(hardware["gpu_visible"]).split(",")[1].strip())
            predicted = right["beam_width"] + (0.97 * total_mib - right["peak_device_used_mib"]) / slope
            estimate = min(float(failure_width), max(float(right["beam_width"]), predicted))
    candidate = int(estimate * BOUNDARY_SAFETY) // BOUNDARY_ALIGNMENT * BOUNDARY_ALIGNMENT
    last = max((r["beam_width"] for r in completed), default=0)
    return candidate if last < candidate < failure_width else None


if RESULTS_JSONL.exists():
    RESULTS_JSONL.unlink()
rows = []
for method in METHODS:
    for output_dim in OUTPUT_DIMS:
        first_failure = None
        completed = []
        for beam in BEAM_WIDTHS_BY_OUTPUT[output_dim]:
            row = run_point(method, output_dim, beam, 1)
            rows.append(row)
            if row["status"] == "completed":
                completed.append(row)
            else:
                first_failure = row
                break
        if first_failure is not None:
            confirm = confirmation_width(completed, first_failure["beam_width"])
            if confirm is not None:
                rows.append(run_point(method, output_dim, confirm, 1))
'''
    summarize = r'''keys = sorted({key for row in rows for key in row})
with RESULTS_CSV.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows)

series = []
for method in METHODS:
    for output_dim in OUTPUT_DIMS:
        subset = [r for r in rows if r["method"] == method and r["output_dim"] == output_dim]
        completed = [r for r in subset if r["status"] == "completed" and r["attempt"] == 1]
        failed_widths = sorted({r["beam_width"] for r in subset if r["status"] != "completed"})
        series.append({
            "method": method, "output_dim": output_dim,
            "max_tested_complete": max((r["beam_width"] for r in completed), default=None),
            "first_failure": min(failed_widths, default=None),
            "boundary_policy": "powers_of_two_then_one_memory_trend_confirmation",
            "rows": len(subset),
        })
summary = {"hardware": hardware, "series": series, "rows": rows}
SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({"series": series}, indent=2), flush=True)
if any(item["max_tested_complete"] is None for item in series):
    raise RuntimeError("benchmark invalid: every method/head series must complete its minimum beam")

try:
    import pandas as pd
    display(pd.DataFrame(rows))
    display(pd.DataFrame(series))
except Exception:
    pass
'''
    return {
        "cells": [md(header), code(config), code(setup), code(run), code(summarize)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for method, label in (("pilgrim", "pilgrim"), ("multigpu_beam_search", "multigpu")):
        token, shape, slug = "T4", "NvidiaTeslaT4", f"t4_{label}"
        package = OUT / slug
        package.mkdir(parents=True, exist_ok=True)
        nb_name = f"paper-megaminx-{label}-sweep-1xt4.ipynb"
        beam_map = MULTIGPU_BOUNDARY_BEAMS if method == "multigpu_beam_search" else None
        (package / nb_name).write_text(
            json.dumps(notebook(token, (method,), beam_map), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "id": f"trydotatwo/paper-megaminx-{label}-sweep-1xt4",
            "title": f"Paper Megaminx {label} Sweep 1xT4",
            "code_file": nb_name, "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "machine_shape": shape,
            "enable_internet": True,
            "dataset_sources": ["trydotatwo/paper-multigpu-megaminx-benchmark-bundle-v3"],
            "competition_sources": ["cayley-py-megaminx"], "kernel_sources": [],
            "model_sources": [
                "arabidopsisthalian/megaminx2048-512-8-e4000/PyTorch/default/1",
                "trydotatwo/megaminx-output24-p900-t000-q-sym/PyTorch/default/1",
            ],
        }
        (package / "kernel-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    boundary_package = OUT / "t4_pilgrim_boundary"
    boundary_package.mkdir(parents=True, exist_ok=True)
    boundary_name = "paper-megaminx-pilgrim-boundary-1xt4.ipynb"
    boundary_beams = {
        1: [5_636_096, 5_767_168, 5_898_240],
        24: [20_971_520, 23_068_672, 25_165_824],
    }
    (boundary_package / boundary_name).write_text(
        json.dumps(notebook("T4", ("pilgrim",), boundary_beams), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    boundary_metadata = {
        "id": "trydotatwo/paper-megaminx-pilgrim-boundary-1xt4",
        "title": "Paper Megaminx Pilgrim Boundary 1xT4",
        "code_file": boundary_name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": ["trydotatwo/paper-multigpu-megaminx-benchmark-bundle-v3"],
        "competition_sources": ["cayley-py-megaminx"], "kernel_sources": [],
        "model_sources": [
            "arabidopsisthalian/megaminx2048-512-8-e4000/PyTorch/default/1",
            "trydotatwo/megaminx-output24-p900-t000-q-sym/PyTorch/default/1",
        ],
    }
    (boundary_package / "kernel-metadata.json").write_text(
        json.dumps(boundary_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
