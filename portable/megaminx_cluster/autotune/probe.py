"""Isolated solver probes and strict stability classification."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
import subprocess
import time
from types import MappingProxyType
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class ProbeRequest:
    archive_root: Path
    candidate_dir: Path
    world_size: int
    puzzle_id: int
    depth: int
    requested_beam: int
    runtime: Mapping[str, int]
    timeout_seconds: int
    rendezvous_id: str
    total_vram_mib: tuple[int, ...]
    required_scratch_bytes: int
    provenance: Mapping[str, str]


@dataclass(frozen=True)
class ProbeResult:
    stable: bool
    status: str
    metrics: Mapping[str, object]
    command: tuple[str, ...]


def build_probe_command(request: ProbeRequest) -> list[str]:
    return [
        "python3", "-m", "portable.megaminx_cluster.workflow",
        "--archive-root", str(request.archive_root.resolve()),
        "--run-dir", str(request.candidate_dir.resolve()),
        "--world-size", str(request.world_size),
        "--job-id", request.rendezvous_id,
        "--puzzle", str(request.puzzle_id),
        "--depth", str(request.depth),
        "--beam", str(request.requested_beam),
        "--reflect", "off",
    ]


def _positive_number(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value > 0


def classify_metrics(
    payload: Mapping[str, object], world_size: int, required_scratch_bytes: int
) -> tuple[bool, str]:
    scalar = (
        "requested_beam", "effective_beam", "wall_us", "solve_us", "throughput",
        "host_ram_bytes", "scratch_bytes",
    )
    if any(not _positive_number(payload, key) for key in scalar):
        return False, "missing_metric"
    if not isinstance(payload.get("setup_us"), (int, float)) or payload["setup_us"] < 0:
        return False, "missing_metric"
    peaks = payload.get("peak_vram_mib")
    totals = payload.get("total_vram_mib")
    if not isinstance(peaks, list) or not isinstance(totals, list):
        return False, "missing_metric"
    if len(peaks) != world_size or len(totals) != world_size:
        return False, "missing_metric"
    if any(not isinstance(value, int) or value <= 0 for value in peaks + totals):
        return False, "missing_metric"
    if payload.get("replay_ok") is not True:
        return False, "replay_failed"
    digest = payload.get("exactness_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        return False, "missing_metric"
    if payload.get("cuda_ok") is not True:
        return False, "cuda_error"
    if payload.get("nccl_ok") is not True:
        return False, "nccl_error"
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        return False, "missing_metric"
    if int(payload["scratch_bytes"]) < required_scratch_bytes:
        return False, "scratch_capacity"
    if any(peak * 10 > total * 9 for peak, total in zip(peaks, totals)):
        return False, "vram_margin"
    return True, "stable"


def _runtime_env(runtime: Mapping[str, int]) -> dict[str, str]:
    names = {
        "b_micro": "BEAM_B_MICRO",
        "stream1_concurrency": "BEAM_STREAM1_CONCURRENCY",
        "stream3_ring_slots": "BEAM_STREAM3_RING_SLOTS",
        "shard_count": "BEAM_SHARD_COUNT",
        "stream4_batch_candidates": "BEAM_STREAM4_BATCH_CANDIDATES",
        "stream4_trigger_candidates": "BEAM_STREAM4_TRIGGER_CANDIDATES",
        "stream4_active_sort_slots": "BEAM_STREAM4_ACTIVE_SORT_SLOTS",
        "final_materialize_chunk_candidates": "BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES",
    }
    values = {"BEAM_RUNTIME_CONFIG_MODE": "manual"}
    values.update({target: str(runtime[source]) for source, target in names.items()})
    values["BEAM_SHARD_CAPACITY_SCALE_PPM"] = str(runtime["shard_capacity_scale_ppm"])
    return values


def _write(path: Path, value: str | bytes | None) -> None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    path.write_text(value or "", encoding="utf-8")


def _failure_status(returncode: int, text: str) -> str:
    lower = text.lower()
    if "out of memory" in lower or "cuda_error_out_of_memory" in lower:
        return "oom"
    if "nccl" in lower:
        return "nccl_error"
    if "cuda" in lower:
        return "cuda_error"
    return "process_error" if returncode else "stable"


def _run_with_vram_monitor(command: Sequence[str], **kwargs):
    """Run a probe while sampling per-visible-GPU memory usage."""
    timeout = float(kwargs["timeout"])
    world_size = int(kwargs["world_size"])
    process = subprocess.Popen(
        list(command), env=kwargs.get("env"), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    peaks = [0] * world_size
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.kill()
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr)
        try:
            stdout, stderr = process.communicate(timeout=min(0.2, remaining))
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr), tuple(peaks)
        except subprocess.TimeoutExpired:
            sample = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, capture_output=True, check=False,
            )
            if sample.returncode == 0:
                values = [line.strip() for line in sample.stdout.splitlines() if line.strip()]
                if len(values) == world_size and all(value.isdigit() for value in values):
                    peaks = [max(old, int(value)) for old, value in zip(peaks, values)]

def run_probe(
    request: ProbeRequest,
    *,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    monotonic: Callable[[], float] = time.monotonic,
    peak_vram_mib: Sequence[int] | None = None,
    effective_beam: int | None = None,
    throughput: int | float | None = None,
) -> ProbeResult:
    candidate = request.candidate_dir.resolve()
    candidate.mkdir(parents=True, exist_ok=False)
    (candidate / "logs").mkdir()
    (candidate / "history").mkdir()
    command = build_probe_command(request)
    env = dict(os.environ)
    env.update(_runtime_env(request.runtime))
    env["BEAM_HISTORY_DISK_PATH"] = str(candidate / "history")
    env["BEAM_RANK_LOG_DIR"] = str(candidate / "logs" / "ranks")
    env["BEAM_NCCL_ID_FILE"] = str(candidate / f"nccl-{request.rendezvous_id}.bin")
    started = monotonic()
    try:
        if run_command is subprocess.run and peak_vram_mib is None:
            completed, monitored_peaks = _run_with_vram_monitor(
                command, env=env, text=True, capture_output=True, check=False,
                timeout=request.timeout_seconds, world_size=request.world_size,
            )
            peak_vram_mib = monitored_peaks
        else:
            completed = run_command(
                command, env=env, text=True, capture_output=True, check=False,
                timeout=request.timeout_seconds,
            )
    except subprocess.TimeoutExpired as exc:
        _write(candidate / "stdout.log", exc.output)
        _write(candidate / "stderr.log", exc.stderr)
        return ProbeResult(False, "timeout", MappingProxyType({}), tuple(command))
    wall_us = max(1, int((monotonic() - started) * 1_000_000))
    _write(candidate / "stdout.log", completed.stdout)
    _write(candidate / "stderr.log", completed.stderr)
    combined = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        return ProbeResult(
            False, _failure_status(completed.returncode, combined),
            MappingProxyType({"returncode": completed.returncode, "wall_us": wall_us}),
            tuple(command),
        )
    validated_path = candidate / "validated_results.json"
    if not validated_path.is_file():
        return ProbeResult(False, "missing_metric", MappingProxyType({}), tuple(command))
    validated = json.loads(validated_path.read_text(encoding="utf-8"))
    replay_ok = bool(validated.get("results")) and all(
        item.get("valid") is True for item in validated.get("results", [])
    )
    canonical = json.dumps(validated, sort_keys=True, separators=(",", ":")).encode("utf-8")
    peaks = tuple(int(value) for value in (peak_vram_mib or ()))
    actual_effective = int(effective_beam or request.requested_beam)
    actual_throughput = throughput or actual_effective * 1_000_000 / wall_us
    metrics: dict[str, object] = {
        "requested_beam": request.requested_beam,
        "effective_beam": actual_effective,
        "wall_us": wall_us,
        "solve_us": wall_us,
        "setup_us": 0,
        "throughput": actual_throughput,
        "host_ram_bytes": max(1, len(canonical)),
        "scratch_bytes": shutil.disk_usage(candidate).free,
        "peak_vram_mib": list(peaks),
        "total_vram_mib": list(request.total_vram_mib),
        "replay_ok": replay_ok,
        "exactness_digest": sha256(canonical).hexdigest(),
        "cuda_ok": "cuda error" not in combined.lower(),
        "nccl_ok": "nccl error" not in combined.lower(),
        "provenance": dict(request.provenance),
    }
    stable, status = classify_metrics(metrics, request.world_size, request.required_scratch_bytes)
    (candidate / "probe_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ProbeResult(stable, status, MappingProxyType(metrics), tuple(command))
