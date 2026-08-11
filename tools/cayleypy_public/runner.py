"""Deterministic host-side orchestration for the existing production runner."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
import csv
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from typing import Callable, Literal, Mapping

import pandas as pd

from tools.cayleypy_public.config import PublicRunConfig
from tools.cayleypy_public.data import PuzzleContract
from tools.cayleypy_public.model import ExportedModel
from tools.cayleypy_public.paths import (
    SolutionRecord, deduplicate_solutions, invert_path, make_reflected_state, validate_original_solution,
)
from tools.cayleypy_public.profile import RuntimePlan


Variant = Literal["original", "reflected"]
CollectionStatus = Literal["first_solution", "depth_reached", "capacity_reached", "not_collected"]
LogSanitizer = Callable[[str], str]
_UINT32_MAX = 2**32 - 1
_SOLVED_RECORD_BYTES = 32 + 4 + 4
_T4_DEVICE_BYTES = 16 * 1024**3
_T4_HEADROOM_BYTES = 768 * 1024**2
_MAX_GATHER_RECORDS_PER_CHUNK = 65_536
_PROCESS_STOP_TIMEOUT_SECONDS = 5.0
_DIAGNOSTIC_TAIL_BYTES = 4 * 1024 * 1024
_TORCHRUN_RESERVED_ENV_KEYS = frozenset({
    "LOCAL_RANK",
    "RANK",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_NAME",
    "LOCAL_WORLD_SIZE",
    "WORLD_SIZE",
    "GROUP_WORLD_SIZE",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_USE_AGENT_STORE",
    "TORCHELASTIC_ERROR_FILE",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "PYTHON_EXEC",
})
_RELEASE_SOLUTION_RE = re.compile(
    r"puzzle_solved=1 puzzle_id=(?P<puzzle_id>\d+) seconds=(?P<seconds>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?) "
    r"solution_length=(?P<solution_length>\d+) found_depth=(?P<found_depth>\d+) "
    r"touch_depth=(?P<touch_depth>\d+) solution=(?P<solution>.*)"
)
_RELEASE_SOLUTION_LEGACY_RE = re.compile(
    r"puzzle_solved=1 puzzle_id=(?P<puzzle_id>\d+) seconds=(?P<seconds>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?) "
    r"solution_length=(?P<solution_length>\d+) solution=(?P<solution>.*)"
)
_DEBUG_SOLUTION_PATH_RE = re.compile(r"solution_path=(?P<solution>.*)")
_TORCHRUN_TEE_PREFIX_RE = re.compile(r"\[default0\]:(?P<line>.*)")
_HISTORY_MODE = "static_hybrid"
_HISTORY_SLOT_COUNT = 2
_HISTORY_WORKERS = 1
_HISTORY_RAM_BYTES = 28 * 1024**3
_HISTORY_DISK_BYTES = 32 * 1024**3
_HISTORY_RAM_ENV = "CAYLEYPY_HISTORY_RAM_BYTES"
_HISTORY_DISK_ENV = "CAYLEYPY_HISTORY_DISK_BYTES"
_HISTORY_ENTRY_BYTES = 16
_CANDIDATE_META_BYTES = 32
_HISTORY_STAGING_ENTRIES = 1 << 20
_HISTORY_ROOT = PurePosixPath("/tmp/beam_history_public")


@dataclass(frozen=True)
class RunnerInvocation:
    command: tuple[str, ...]
    env: dict[str, str]
    result_tsv: Path | None
    combined_log: Path
    rank_logs: tuple[Path, ...]
    torchrun_log_dir: Path
    history_dir: Path
    puzzle_id: int
    variant: Variant
    source_solution_sha256: str | None = None
    reflected_source_path: str | None = None


@dataclass(frozen=True)
class GatherChunkPlan:
    capacity: int
    records_per_chunk: int
    chunk_count: int


@dataclass(frozen=True)
class HistoryRuntimePreflight:
    required_history_bytes: int
    usable_history_bytes: int
    pinned_slot_bytes: int
    staging_bytes: int


@dataclass(frozen=True)
class InvocationExecution:
    invocation: RunnerInvocation
    return_code: int
    elapsed_seconds: float
    output: str


@dataclass(frozen=True)
class ParsedRunnerOutput:
    records: tuple[SolutionRecord, ...]
    collection_status: CollectionStatus


@dataclass(frozen=True)
class RunArtifacts:
    solution_records: tuple[SolutionRecord, ...]
    submission: pd.DataFrame
    combined_logs: tuple[Path, ...]
    rank_logs: tuple[tuple[Path, Path], ...]
    return_codes: tuple[int, ...]
    timing_summaries: tuple[float, ...]
    collection_statuses: tuple[CollectionStatus, ...]

    @property
    def collection_status(self) -> CollectionStatus:
        if "capacity_reached" in self.collection_statuses:
            return "capacity_reached"
        if "depth_reached" in self.collection_statuses:
            return "depth_reached"
        if "first_solution" in self.collection_statuses:
            return "first_solution"
        return "not_collected"


class PublicSearchRunError(RuntimeError):
    """Hard process failure with earlier artifacts retained for diagnosis only."""

    def __init__(self, message: str, partial_artifacts: RunArtifacts):
        super().__init__(message)
        self.partial_artifacts = partial_artifacts


class InvocationLogCaptureError(RuntimeError):
    """Internal subprocess result whose retained logs or scratch cleanup are incomplete."""

    def __init__(self, execution: InvocationExecution, message: str):
        super().__init__(message)
        self.execution = execution


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _path_depth(path: str) -> int:
    return 0 if path == "" else len(path.split("."))


def derive_solved_result_capacity(plan: RuntimePlan, move_count: int) -> int:
    """Bound one rank's worst-case Stream2 hits for a completed depth."""
    if isinstance(move_count, bool) or not isinstance(move_count, int) or move_count <= 0:
        raise ValueError("move_count must be a positive integer")
    if plan.local_beam <= 0:
        raise ValueError("local_beam must be positive")
    capacity = plan.local_beam * move_count
    if capacity > _UINT32_MAX:
        raise ValueError("BEAM_SOLVED_RESULT_CAPACITY exceeds uint32")
    snapshot_bytes = capacity * _SOLVED_RECORD_BYTES
    frontier_bytes = plan.local_beam * 128
    required_bytes = snapshot_bytes + frontier_bytes
    available_bytes = _T4_DEVICE_BYTES - _T4_HEADROOM_BYTES
    if required_bytes > available_bytes:
        raise ValueError(
            "BEAM_SOLVED_RESULT_CAPACITY snapshot plus current frontier exceeds available T4 device memory; "
            "production_runner performs the exact StaticMemoryPlan and non-static budget gate "
            f"({required_bytes} > {available_bytes} bytes)"
        )
    return capacity


def derive_gather_chunk_plan(
    frontier_states: int, capacity: int, world_size: int = 2, packet_words: int = 8,
) -> GatherChunkPlan:
    """Plan fixed-scratch host gather chunks without a capacity-sized packet buffer."""
    values = {
        "frontier_states": frontier_states,
        "capacity": capacity,
        "world_size": world_size,
        "packet_words": packet_words,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values.values()):
        raise ValueError(f"gather chunk inputs must be positive integers: {values}")
    scratch_words = frontier_states * 128 // 8
    records_per_chunk = min(
        scratch_words // (packet_words * (world_size + 1)), _MAX_GATHER_RECORDS_PER_CHUNK
    )
    if records_per_chunk <= 0:
        raise ValueError("solve bucket gather scratch cannot fit one record per rank")
    return GatherChunkPlan(
        capacity=capacity,
        records_per_chunk=records_per_chunk,
        chunk_count=(capacity + records_per_chunk - 1) // records_per_chunk,
    )


def derive_collection_batch_limit(max_solutions: int, accepted_count: int) -> int:
    """Return the next deterministic host-selection limit, capped at one fixed chunk."""
    values = {"max_solutions": max_solutions, "accepted_count": accepted_count}
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values()):
        raise ValueError(f"collection batch inputs must be non-negative integers: {values}")
    if max_solutions == 0:
        return _MAX_GATHER_RECORDS_PER_CHUNK
    return min(max(max_solutions - accepted_count, 0), _MAX_GATHER_RECORDS_PER_CHUNK)
def _positive_budget_from_env(name: str, default: int, environ: Mapping[str, str] | None = None) -> int:
    values = os.environ if environ is None else environ
    raw = values.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer byte count") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer byte count")
    return value


def history_runtime_budgets(environ: Mapping[str, str] | None = None) -> tuple[int, int]:
    """Resolve bounded static-history RAM/disk budgets before a runner launch."""
    return (
        _positive_budget_from_env(_HISTORY_RAM_ENV, _HISTORY_RAM_BYTES, environ),
        _positive_budget_from_env(_HISTORY_DISK_ENV, _HISTORY_DISK_BYTES, environ),
    )


def _resolved_history_path(history_dir: Path) -> Path:
    candidate = PurePosixPath(history_dir.as_posix())
    if ".." in candidate.parts or _HISTORY_ROOT not in candidate.parents:
        raise ValueError("history path must be unique and under /tmp/beam_history_public")
    resolved_root = Path(_HISTORY_ROOT.as_posix()).resolve(strict=False)
    resolved_candidate = Path(candidate.as_posix()).resolve(strict=False)
    if resolved_candidate == resolved_root or resolved_root not in resolved_candidate.parents:
        raise ValueError("resolved history path must be a descendant of /tmp/beam_history_public")
    return resolved_candidate


def cleanup_history_runtime(history_dir: Path) -> None:
    """Remove only a validated per-invocation scratch history tree."""
    resolved = _resolved_history_path(history_dir)
    if not resolved.exists():
        return
    if not resolved.is_dir():
        raise RuntimeError(f"history scratch path is not a directory: {resolved}")
    shutil.rmtree(resolved)


def maximum_history_depth(
    plan: RuntimePlan,
    move_count: int,
    touch_bfs_radius: int,
    history_ram_bytes: int,
    history_disk_bytes: int,
) -> int:
    """Return the largest MAX_DEPTH admitted by the static-history budget."""
    values = (move_count, touch_bfs_radius, history_ram_bytes, history_disk_bytes)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("history depth inputs must be integers")
    if move_count <= 0 or touch_bfs_radius < 0:
        raise ValueError("move_count must be positive and touch_bfs_radius nonnegative")
    if history_ram_bytes <= 0 or history_disk_bytes <= 0:
        raise ValueError("history budgets must be positive")
    per_rank_ram = history_ram_bytes // 2
    per_rank_disk = history_disk_bytes // 2
    pinned_slot_bytes = _HISTORY_SLOT_COUNT * plan.local_beam * _CANDIDATE_META_BYTES
    staging_bytes = (
        _HISTORY_SLOT_COUNT
        * min(plan.local_beam, _HISTORY_STAGING_ENTRIES)
        * _HISTORY_ENTRY_BYTES
    )
    if per_rank_ram <= pinned_slot_bytes + staging_bytes:
        raise ValueError("history RAM budget cannot fit pinned slots and staging")
    usable_entries = (
        per_rank_ram - pinned_slot_bytes - staging_bytes + per_rank_disk
    ) // _HISTORY_ENTRY_BYTES
    frontier_bound = 1
    states_before_target = 0
    saturation_depth = 0
    while frontier_bound < plan.local_beam:
        frontier_bound = min(plan.local_beam, frontier_bound * move_count)
        if frontier_bound >= plan.local_beam:
            break
        states_before_target += frontier_bound
        saturation_depth += 1
    if frontier_bound < plan.local_beam:
        max_effective_depth = usable_entries
    elif usable_entries < states_before_target:
        max_effective_depth = saturation_depth
    else:
        full_depths = (usable_entries - states_before_target) // plan.local_beam
        max_effective_depth = saturation_depth + full_depths
    return int(min(max_effective_depth + touch_bfs_radius, 2**32 - 1))

def preflight_history_runtime(
    config: PublicRunConfig,
    plan: RuntimePlan,
    move_count: int,
    history_dir: Path,
    *,
    tmp_free_bytes: int | None = None,
    history_ram_bytes: int | None = None,
    history_disk_bytes: int | None = None,
) -> HistoryRuntimePreflight:
    """Fail closed on the bounded two-rank static-hybrid history contract."""
    if isinstance(move_count, bool) or not isinstance(move_count, int) or move_count <= 0:
        raise ValueError("move_count must be a positive integer")
    _resolved_history_path(history_dir)
    default_ram_bytes, default_disk_bytes = history_runtime_budgets()
    history_ram_bytes = default_ram_bytes if history_ram_bytes is None else history_ram_bytes
    history_disk_bytes = default_disk_bytes if history_disk_bytes is None else history_disk_bytes
    if isinstance(history_ram_bytes, bool) or not isinstance(history_ram_bytes, int) or history_ram_bytes <= 0:
        raise ValueError("history_ram_bytes must be a positive integer")
    if isinstance(history_disk_bytes, bool) or not isinstance(history_disk_bytes, int) or history_disk_bytes <= 0:
        raise ValueError("history_disk_bytes must be a positive integer")
    if tmp_free_bytes is None:
        tmp_free_bytes = shutil.disk_usage(str(_HISTORY_ROOT.parent)).free
    if isinstance(tmp_free_bytes, bool) or not isinstance(tmp_free_bytes, int) or tmp_free_bytes < 0:
        raise ValueError("tmp_free_bytes must be a nonnegative integer")
    if tmp_free_bytes < history_disk_bytes:
        raise ValueError(
            f"/tmp free disk is smaller than the history disk budget "
            f"({tmp_free_bytes} < {history_disk_bytes} bytes)"
        )

    per_rank_ram = history_ram_bytes // 2
    per_rank_disk = history_disk_bytes // 2
    pinned_slot_bytes = _HISTORY_SLOT_COUNT * plan.local_beam * _CANDIDATE_META_BYTES
    staging_bytes = (
        _HISTORY_SLOT_COUNT * min(plan.local_beam, _HISTORY_STAGING_ENTRIES) * _HISTORY_ENTRY_BYTES
    )
    if per_rank_ram <= pinned_slot_bytes + staging_bytes:
        raise ValueError("history RAM budget cannot fit pinned slots and staging")

    effective_depth = max(config.max_depth - config.touch_bfs_radius, 0)
    frontier_bound = 1
    states_before_target = 0
    required_entries = 0
    for depth in range(effective_depth):
        frontier_bound = min(plan.local_beam, frontier_bound * move_count)
        if frontier_bound >= plan.local_beam:
            required_entries = states_before_target + (effective_depth - depth) * plan.local_beam
            break
        states_before_target += frontier_bound
    else:
        required_entries = states_before_target
    required_history_bytes = required_entries * _HISTORY_ENTRY_BYTES
    usable_history_bytes = per_rank_ram - pinned_slot_bytes - staging_bytes + per_rank_disk
    if required_history_bytes > usable_history_bytes:
        raise ValueError(
            "static hybrid history budget is too small for the requested depth/beam/move count "
            f"({required_history_bytes} > {usable_history_bytes} bytes per rank)"
        )
    return HistoryRuntimePreflight(
        required_history_bytes=required_history_bytes,
        usable_history_bytes=usable_history_bytes,
        pinned_slot_bytes=pinned_slot_bytes,
        staging_bytes=staging_bytes,
    )


def _runtime_env(
    config: PublicRunConfig,
    plan: RuntimePlan,
    weights_dir: Path,
    model: ExportedModel | None = None,
) -> dict[str, str]:
    generator_path = str(config.puzzle_info_json)
    if model is not None and model.backend == "piece_transformer":
        source_generators = model.manifest.get("source_generators")
        if not isinstance(source_generators, str) or not source_generators:
            raise ValueError("piece-transformer export manifest is missing source_generators")
        generator_path = source_generators
    env = {
        "BEAM_WEIGHT_DIR": str(weights_dir),
        "BEAM_PUZZLE_INFO_JSON": str(config.puzzle_info_json),
        "BEAM_GENERATOR_PATH": generator_path,
        "BEAM_TEST_CSV": str(config.test_csv),
        "BEAM_RUNTIME_CONFIG_MODE": "manual",
        "BEAM_B_MICRO": str(plan.runtime["b_micro"]),
        "BEAM_STREAM1_CONCURRENCY": str(plan.runtime["stream1_concurrency"]),
        "BEAM_STREAM3_RING_SLOTS": str(plan.runtime["stream3_ring_slots"]),
        "BEAM_SHARD_COUNT": str(plan.runtime["shard_count"]),
        "BEAM_SHARD_BUFFER_COUNT": "2",
        "BEAM_SHARD_CAPACITY_CANDIDATES": str(plan.shard_capacity_candidates),
        "BEAM_STREAM4_BATCH_CANDIDATES": str(plan.runtime["stream4_batch_candidates"]),
        "BEAM_STREAM4_TRIGGER_CANDIDATES": str(plan.runtime["stream4_trigger_candidates"]),
        "BEAM_STREAM4_ACTIVE_SORT_SLOTS": str(plan.runtime["stream4_active_sort_slots"]),
        "BEAM_GLOBAL_SPILL_CAPACITY": "0",
        "BEAM_STREAM5_RECV_CAPACITY_SCALE_PPM": "1000000",
        "BEAM_GPU_HEADROOM_BYTES": str(_T4_HEADROOM_BYTES),
        "BEAM_SOLVED_NEIGHBORHOOD_RADIUS": str(config.touch_bfs_radius),
        "BEAM_REPAIR_K1_RADIUS": str(config.touch_bfs_radius),
        "BEAM_REPAIR_K2_RADIUS": "0",
        "BEAM_STREAM2_SUFFIX_RADIUS": "0",
        "BEAM_DEPTH_LOG_EVERY": str(config.depth_log_every),
        "BEAM_PUZZLE_LOG_EVERY": str(config.puzzle_log_every),
    }
    if "ring_count" in plan.runtime:
        env["BEAM_RING_COUNT"] = str(plan.runtime["ring_count"])
    if "final_materialize_chunk_candidates" in plan.runtime:
        env["BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES"] = str(
            plan.runtime["final_materialize_chunk_candidates"]
        )
    if model is not None and model.backend == "piece_transformer":
        env.update({
            "BEAM_STREAM1_EXECUTOR": "libtorch_eager",
            "BEAM_STREAM1_TRANSFORMER_MICRO": str(plan.runtime["b_micro"]),
        })
    return env


def build_runner_invocation(
    config: PublicRunConfig,
    plan: RuntimePlan,
    move_count: int,
    puzzle_id: int,
    variant: Variant,
    weights_dir: Path,
    artifact_dir: Path,
    *,
    runner_path: str = "production_runner",
    model: ExportedModel | None = None,
    source_solution_sha256: str | None = None,
    reflected_source_path: str | None = None,
) -> RunnerInvocation:
    """Build a hermetic torchrun invocation for exactly one variant."""
    run_id = uuid.uuid4().hex
    run_root = artifact_dir / f"puzzle-{puzzle_id}" / variant / run_id
    torchrun_log_dir = run_root / "torchrun"
    history_dir = Path(_HISTORY_ROOT.as_posix()) / run_id / str(puzzle_id) / variant
    result_tsv = run_root / "solve_bucket.tsv" if config.solution_mode == "collect" else None
    history_ram_bytes, history_disk_bytes = history_runtime_budgets()
    preflight_history_runtime(
        config,
        plan,
        move_count,
        history_dir,
        history_ram_bytes=history_ram_bytes,
        history_disk_bytes=history_disk_bytes,
    )
    run_root.mkdir(parents=True, exist_ok=False)
    env = _runtime_env(config, plan, weights_dir, model)
    env.update({
        "BEAM_HISTORY_MODE": _HISTORY_MODE,
        "BEAM_HISTORY_SLOT_COUNT": str(_HISTORY_SLOT_COUNT),
        "BEAM_HISTORY_WORKERS": str(_HISTORY_WORKERS),
        "BEAM_HISTORY_RAM_BYTES": str(history_ram_bytes),
        "BEAM_HISTORY_DISK_BYTES": str(history_disk_bytes),
        "BEAM_HISTORY_DIR": history_dir.as_posix(),
        "BEAM_HISTORY_DISK_PATH": history_dir.as_posix(),
    })
    env["BEAM_NCCL_ID_FILE"] = str(run_root / "nccl-id.bin")
    if result_tsv is not None:
        snapshot_capacity = derive_solved_result_capacity(plan, move_count)
        derive_gather_chunk_plan(
            plan.local_beam, snapshot_capacity, world_size=plan.world_size,
        )
        env.update({
            "BEAM_SOLVE_BUCKET_MODE": "1",
            "BEAM_SOLVE_BUCKET_STOP_DEPTH": str(config.collect_until_depth),
            "BEAM_SOLVE_BUCKET_MAX_SOLUTIONS": str(config.max_collected_solutions),
            "BEAM_SOLVED_RESULT_CAPACITY": str(snapshot_capacity),
            "BEAM_SOLVE_BUCKET_RESULT_TSV": str(result_tsv),
        })
    command = (
        "python", "-m", "torch.distributed.run", f"--nproc-per-node={plan.world_size}",
        "--rdzv-backend=c10d", f"--rdzv-endpoint=127.0.0.1:{_free_port()}", f"--rdzv-id={run_id}",
        f"--log-dir={torchrun_log_dir}", "--redirects=3", "--tee=0:3", "--no-python", runner_path,
        str(puzzle_id), str(config.max_depth), str(config.beam_width),
    )
    return RunnerInvocation(
        command=command,
        env=env,
        result_tsv=result_tsv,
        combined_log=run_root / "combined.log",
        rank_logs=tuple(run_root / f"rank-{rank}.log" for rank in range(plan.world_size)),
        torchrun_log_dir=torchrun_log_dir,
        history_dir=history_dir,
        puzzle_id=puzzle_id,
        variant=variant,
        source_solution_sha256=source_solution_sha256,
        reflected_source_path=reflected_source_path,
    )


def _solver_log_lines(output: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw_line in output.splitlines():
        match = _TORCHRUN_TEE_PREFIX_RE.fullmatch(raw_line)
        normalized.append(match.group("line") if match is not None else raw_line)
    return tuple(normalized)


def _parse_uint(row: dict[str, str], name: str) -> int:
    try:
        value = int(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"solve bucket TSV has invalid {name}") from error
    if value < 0:
        raise ValueError(f"solve bucket TSV has negative {name}")
    return value


def parse_runner_output(
    output: str, result_tsv: Path | None, puzzle_id: int, variant: Variant,
) -> ParsedRunnerOutput:
    status: CollectionStatus = "not_collected"
    if "collection_status=capacity_reached" in output:
        status = "capacity_reached"
    elif "collection_status=depth_reached" in output:
        status = "depth_reached"
    records: list[SolutionRecord] = []
    if result_tsv is not None and result_tsv.exists():
        with result_tsv.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if reader.fieldnames is None or "solution_path" not in reader.fieldnames:
                raise ValueError("solve bucket TSV missing solution_path")
            for row in reader:
                if _parse_uint(row, "puzzle_id") != puzzle_id:
                    continue
                path = row["solution_path"]
                found_depth = _parse_uint(row, "found_depth")
                total_depth = _parse_uint(row, "total_depth")
                if total_depth < found_depth:
                    raise ValueError("solve bucket TSV total_depth is smaller than found_depth")
                if _path_depth(path) != total_depth:
                    raise ValueError("solve bucket TSV solution_path length does not match total_depth")
                records.append(SolutionRecord(
                    puzzle_id=puzzle_id,
                    variant=variant,
                    path=path,
                    original_oriented_path=path,
                    found_depth=found_depth,
                    touch_depth=total_depth - found_depth,
                    source_solution_sha256=None,
                    reflected_source_path=None,
                    valid=True,
                    reached_state=(),
                ))
    if not records:
        release_rows: list[tuple[str, int, int]] = []
        for line in _solver_log_lines(output):
            if not line.startswith("puzzle_solved=1"):
                continue
            match = _RELEASE_SOLUTION_RE.fullmatch(line)
            legacy = False
            if match is None:
                match = _RELEASE_SOLUTION_LEGACY_RE.fullmatch(line)
                legacy = match is not None
            if match is None:
                raise ValueError("malformed puzzle_solved=1 release line")
            seconds = float(match.group("seconds"))
            if not math.isfinite(seconds):
                raise ValueError("release solution seconds must be finite")
            release_puzzle_id = int(match.group("puzzle_id"))
            if release_puzzle_id != puzzle_id:
                raise ValueError(
                    f"release solution puzzle id {release_puzzle_id} does not match requested puzzle id {puzzle_id}"
                )
            path = match.group("solution")
            solution_length = int(match.group("solution_length"))
            found_depth = solution_length if legacy else int(match.group("found_depth"))
            touch_depth = 0 if legacy else int(match.group("touch_depth"))
            if _path_depth(path) != solution_length or found_depth + touch_depth != solution_length:
                raise ValueError("release solution length/depth fields do not match solution path")
            release_rows.append((path, found_depth, touch_depth))
        if release_rows:
            for path, found_depth, touch_depth in release_rows:
                records.append(SolutionRecord(
                    puzzle_id=puzzle_id,
                    variant=variant,
                    path=path,
                    original_oriented_path=path,
                    found_depth=found_depth,
                    touch_depth=touch_depth,
                    source_solution_sha256=None,
                    reflected_source_path=None,
                    valid=True,
                    reached_state=(),
                ))
        else:
            debug_paths = [
                match.group("solution")
                for line in _solver_log_lines(output)
                if (match := _DEBUG_SOLUTION_PATH_RE.fullmatch(line)) is not None
            ]
            if debug_paths:
                path = debug_paths[-1]
                records.append(SolutionRecord(
                    puzzle_id=puzzle_id,
                    variant=variant,
                    path=path,
                    original_oriented_path=path,
                    found_depth=_path_depth(path),
                    touch_depth=0,
                    source_solution_sha256=None,
                    reflected_source_path=None,
                    valid=True,
                    reached_state=(),
                ))
        if records and status == "not_collected":
            status = "first_solution"
    return ParsedRunnerOutput(tuple(records), status)


def _read_rank_streams(log_dir: Path, rank: int) -> tuple[str, str]:
    stdout_files = sorted(path for path in log_dir.rglob("stdout.log") if path.parent.name == str(rank))
    stderr_files = sorted(path for path in log_dir.rglob("stderr.log") if path.parent.name == str(rank))
    if not stdout_files or not stderr_files:
        raise RuntimeError(f"torchrun did not retain stdout/stderr for rank {rank}")
    stdout = "".join(path.read_text(encoding="utf-8") for path in stdout_files)
    stderr = "".join(path.read_text(encoding="utf-8") for path in stderr_files)
    return stdout, stderr


def _withhold_raw_log(path: Path) -> None:
    """Replace a suspect log atomically; never leave raw diagnostics public."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.withheld")
    try:
        temporary.write_text("[log withheld: sanitization failed]\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _sanitize_log_file(path: Path, sanitizer: LogSanitizer | None) -> None:
    """Sanitize a redirect log line-by-line and atomically replace the raw source."""
    if sanitizer is None or not path.exists():
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.sanitized")
    try:
        with path.open("r", encoding="utf-8", errors="replace") as source, temporary.open("w", encoding="utf-8", newline="") as destination:
            for line in source:
                destination.write(sanitizer(line))
        os.replace(temporary, path)
    except Exception as error:
        try:
            _withhold_raw_log(path)
        except Exception as withhold_error:
            raise RuntimeError(f"log sanitization failed and raw log could not be withheld: {withhold_error}") from error
        raise RuntimeError("log sanitization failed; raw log was withheld") from error
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def sanitize_torchrun_redirect_logs(invocation: RunnerInvocation, sanitizer: LogSanitizer | None) -> None:
    """Remove private paths from torchrun redirect files before public copies exist."""
    if sanitizer is None or not invocation.torchrun_log_dir.exists():
        return
    for path in sorted(invocation.torchrun_log_dir.rglob("stdout.log")):
        _sanitize_log_file(path, sanitizer)
    for path in sorted(invocation.torchrun_log_dir.rglob("stderr.log")):
        _sanitize_log_file(path, sanitizer)


def collect_torchrun_rank_logs(invocation: RunnerInvocation, sanitizer: LogSanitizer | None = None) -> None:
    """Materialize readable per-rank files exclusively from torchrun redirects."""
    captured: list[tuple[Path, str]] = []
    for rank, output_path in enumerate(invocation.rank_logs):
        stdout, stderr = _read_rank_streams(invocation.torchrun_log_dir, rank)
        if stdout and not stdout.endswith("\n"):
            stdout += "\n"
        if stderr and not stderr.endswith("\n"):
            stderr += "\n"
        text = f"[stdout]\n{stdout}[stderr]\n{stderr}"
        captured.append((output_path, sanitizer(text) if sanitizer is not None else text))
    for output_path, text in captured:
        output_path.write_text(text, encoding="utf-8")


def collect_available_torchrun_rank_logs(invocation: RunnerInvocation, sanitizer: LogSanitizer | None = None) -> tuple[str, ...]:
    """Best-effort materialization for ranks whose redirected streams are already complete."""
    errors: list[str] = []
    for rank, output_path in enumerate(invocation.rank_logs):
        try:
            stdout, stderr = _read_rank_streams(invocation.torchrun_log_dir, rank)
            if stdout and not stdout.endswith("\n"):
                stdout += "\n"
            if stderr and not stderr.endswith("\n"):
                stderr += "\n"
            text = f"[stdout]\n{stdout}[stderr]\n{stderr}"
            output_path.write_text(sanitizer(text) if sanitizer is not None else text, encoding="utf-8")
        except (OSError, RuntimeError, UnicodeError) as error:
            errors.append(f"rank {rank}: {error}")
    return tuple(errors)


def _bounded_log_tail(path: Path, max_bytes: int = _DIAGNOSTIC_TAIL_BYTES) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(size - max_bytes, 0), os.SEEK_SET)
            return handle.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _wait_process_bounded(process: object, timeout_seconds: float) -> int:
    wait = getattr(process, "wait", None)
    if not callable(wait):
        raise RuntimeError("torchrun process does not expose wait()")
    try:
        return int(wait(timeout=timeout_seconds))
    except TypeError:
        # Minimal test doubles may not accept the timeout keyword; real Popen always does.
        return int(wait())


def _stop_process(process: object, timeout_seconds: float = _PROCESS_STOP_TIMEOUT_SECONDS) -> int:
    """Reap a child, escalating terminate to kill with bounded real-Popen waits."""
    poll = getattr(process, "poll", None)
    if callable(poll):
        return_code = poll()
        if return_code is not None:
            return int(return_code)
    terminate = getattr(process, "terminate", None)
    if callable(terminate):
        terminate()
    try:
        return _wait_process_bounded(process, timeout_seconds)
    except subprocess.TimeoutExpired:
        kill = getattr(process, "kill", None)
        if not callable(kill):
            raise RuntimeError("torchrun process timed out and does not expose kill()")
        kill()
        return _wait_process_bounded(process, timeout_seconds)


def _add_exception_note(error: BaseException, note: str) -> None:
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)


def stream_process_output(
    process: object,
    combined_log: Path,
    *,
    live_stream: object | None = None,
    max_tail_chars: int = 4 * 1024 * 1024,
    log_sanitizer: LogSanitizer | None = None,
) -> tuple[int, str]:
    """Stream rank-0 tee output live and retain only a bounded parser tail in RAM."""
    if isinstance(max_tail_chars, bool) or not isinstance(max_tail_chars, int) or max_tail_chars <= 0:
        raise ValueError("max_tail_chars must be a positive integer")
    if live_stream is None:
        live_stream = sys.stdout
    stdout = getattr(process, "stdout", None)
    if stdout is None:
        raise RuntimeError("torchrun stdout pipe is unavailable")
    tail = ""
    with combined_log.open("w", encoding="utf-8") as log:
        for chunk in stdout:
            safe_chunk = log_sanitizer(chunk) if log_sanitizer is not None else chunk
            live_stream.write(safe_chunk)
            live_stream.flush()
            log.write(safe_chunk)
            log.flush()
            tail = (tail + safe_chunk)[-max_tail_chars:]
    return int(process.wait()), tail


def _run_one(invocation: RunnerInvocation, *, log_sanitizer: LogSanitizer | None = None) -> InvocationExecution:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("BEAM_")
        and not key.startswith("TORCHELASTIC_")
        and key not in _TORCHRUN_RESERVED_ENV_KEYS
    }
    environment.update(invocation.env)
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            invocation.command,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except BaseException as primary_error:
        try:
            cleanup_history_runtime(invocation.history_dir)
        except Exception as cleanup_error:
            _add_exception_note(
                primary_error, f"history scratch cleanup failed after spawn error: {cleanup_error}"
            )
        raise

    execution: InvocationExecution | None = None
    try:
        return_code, output_tail = stream_process_output(process, invocation.combined_log, log_sanitizer=log_sanitizer)
        elapsed = time.monotonic() - started
        execution = InvocationExecution(invocation, return_code, elapsed, output_tail)
        try:
            sanitize_torchrun_redirect_logs(invocation, log_sanitizer)
            collect_torchrun_rank_logs(invocation, log_sanitizer)
        except (OSError, RuntimeError, UnicodeError) as error:
            raise InvocationLogCaptureError(
                execution, f"torchrun log capture failed: {error}"
            ) from error
    except BaseException as primary_error:
        process_stopped = False
        return_code = -1
        try:
            return_code = _stop_process(process)
            process_stopped = True
        except Exception as stop_error:
            _add_exception_note(primary_error, f"torchrun process cleanup failed: {stop_error}")
        if execution is None:
            execution = InvocationExecution(
                invocation,
                return_code,
                time.monotonic() - started,
                _bounded_log_tail(invocation.combined_log),
            )
        try:
            setattr(primary_error, "execution", execution)
        except (AttributeError, TypeError):
            _add_exception_note(primary_error, "partial invocation execution could not be attached")
        try:
            try:
                sanitize_torchrun_redirect_logs(invocation, log_sanitizer)
            except Exception as sanitize_error:
                _add_exception_note(primary_error, f"partial torchrun log sanitization failed: {sanitize_error}")
            diagnostic_errors = collect_available_torchrun_rank_logs(invocation, log_sanitizer)
        except Exception as diagnostic_failure:
            diagnostic_errors = ()
            _add_exception_note(
                primary_error, f"partial torchrun log capture failed: {diagnostic_failure}"
            )
        for diagnostic_error in diagnostic_errors:
            _add_exception_note(primary_error, f"partial torchrun log capture failed: {diagnostic_error}")
        if process_stopped:
            try:
                cleanup_history_runtime(invocation.history_dir)
            except Exception as cleanup_error:
                _add_exception_note(
                    primary_error, f"history scratch cleanup failed: {cleanup_error}"
                )
        else:
            _add_exception_note(
                primary_error,
                f"history scratch retained because torchrun may still be alive: {invocation.history_dir}",
            )
        raise

    try:
        cleanup_history_runtime(invocation.history_dir)
    except Exception as cleanup_error:
        raise InvocationLogCaptureError(
            execution, f"history scratch cleanup failed: {cleanup_error}"
        ) from cleanup_error
    return execution


def _reflection_sources(
    config: PublicRunConfig, contract: PuzzleContract,
) -> dict[int, tuple[tuple[str, str], ...]]:
    if config.reflect_source_csv is None:
        return {puzzle_id: () for puzzle_id in config.puzzle_ids}
    frame = pd.read_csv(config.reflect_source_csv, keep_default_na=False)
    id_columns = [name for name in ("initial_state_id", "puzzle_id", "id") if name in frame.columns]
    path_columns = [name for name in ("path", "moves", "solution_path", "solution") if name in frame.columns]
    if len(id_columns) != 1 or len(path_columns) != 1:
        raise ValueError("reflection source CSV must have one puzzle id and one solution path column")
    selected = set(config.puzzle_ids)
    sources: dict[int, list[tuple[str, str]]] = {puzzle_id: [] for puzzle_id in config.puzzle_ids}
    for index, row in frame.iterrows():
        try:
            puzzle_id = int(row[id_columns[0]])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid reflection source puzzle id at row {index}") from error
        if puzzle_id not in selected:
            raise ValueError(f"reflection source puzzle {puzzle_id} is outside selected range")
        path_value = row[path_columns[0]]
        if not isinstance(path_value, str):
            raise ValueError(f"invalid reflection source path for puzzle {puzzle_id}")
        if not validate_original_solution(
            contract.initial_states[puzzle_id], contract.central_state, path_value, contract.generators
        ):
            raise ValueError(f"invalid reflection source path for puzzle {puzzle_id}")
        sources[puzzle_id].append((path_value, sha256(path_value.encode("utf-8")).hexdigest()))
    if config.reflect_mode == "only":
        missing = [puzzle_id for puzzle_id, rows in sources.items() if not rows]
        if missing:
            raise ValueError(f"reflection source CSV missing selected puzzle ids: {missing}")
    return {
        puzzle_id: tuple(sorted(set(rows), key=lambda item: (_path_depth(item[0]), item[0], item[1])))
        for puzzle_id, rows in sources.items()
    }


def _write_reflected_test_row(invocation: RunnerInvocation, puzzle_id: int, state: tuple[int, ...]) -> Path:
    path = invocation.combined_log.parent / "reflected_test.csv"
    pd.DataFrame({
        "initial_state_id": [puzzle_id],
        "initial_state": [",".join(str(value) for value in state)],
    }).to_csv(path, index=False)
    invocation.env["BEAM_TEST_CSV"] = str(path)
    return path


def _submission_column(frame: pd.DataFrame) -> str:
    matches = [name for name in ("path", "moves", "solution_path", "solution") if name in frame.columns]
    if len(matches) != 1:
        raise ValueError("sample submission must have exactly one solution path column")
    return matches[0]


def _build_artifacts(
    records: list[SolutionRecord], submission: pd.DataFrame, executions: list[InvocationExecution],
    statuses: list[CollectionStatus],
) -> RunArtifacts:
    return RunArtifacts(
        solution_records=tuple(records),
        submission=submission.copy(deep=True),
        combined_logs=tuple(item.invocation.combined_log for item in executions),
        rank_logs=tuple(item.invocation.rank_logs for item in executions),
        return_codes=tuple(item.return_code for item in executions),
        timing_summaries=tuple(item.elapsed_seconds for item in executions),
        collection_statuses=tuple(statuses),
    )


def _validated_records(
    parsed: ParsedRunnerOutput,
    contract: PuzzleContract,
    puzzle_id: int,
    variant: Variant,
    reflected_state: tuple[int, ...] | None = None,
    source_sha256: str | None = None,
    reflected_source_path: str | None = None,
) -> list[SolutionRecord]:
    validated: list[SolutionRecord] = []
    for raw in parsed.records:
        if variant == "original":
            original_path = raw.path
            valid = validate_original_solution(
                contract.initial_states[puzzle_id], contract.central_state, original_path, contract.generators
            )
        else:
            if reflected_state is None:
                raise RuntimeError("reflected validation requires reflected_state")
            reflected_valid = validate_original_solution(
                reflected_state, contract.central_state, raw.path, contract.generators
            )
            original_path = invert_path(raw.path, contract.generators)
            valid = reflected_valid and validate_original_solution(
                contract.initial_states[puzzle_id], contract.central_state, original_path, contract.generators
            )
        if not valid:
            raise RuntimeError(f"runner returned an invalid {variant} solution for puzzle {puzzle_id}")
        validated.append(SolutionRecord(
            puzzle_id=puzzle_id,
            variant=variant,
            path=raw.path,
            original_oriented_path=original_path,
            found_depth=raw.found_depth,
            touch_depth=raw.touch_depth,
            source_solution_sha256=source_sha256,
            reflected_source_path=reflected_source_path,
            valid=True,
            reached_state=contract.central_state,
        ))
    return validated


def run_public_search(
    config: PublicRunConfig,
    contract: PuzzleContract,
    model: ExportedModel,
    plan: RuntimePlan,
    weights_dir: Path,
    artifact_dir: Path,
    *,
    runner_path: str = "production_runner",
    log_sanitizer: LogSanitizer | None = None,
) -> RunArtifacts:
    """Run requested variants and retain only CPU-validated original-oriented paths."""
    if config.reflect_mode != "off":
        invert_path("", contract.generators)  # Validate inverse closure before any GPU launch.
    external_sources = _reflection_sources(config, contract)  # Must finish before any GPU launch.
    if config.solution_mode == "collect":
        derive_solved_result_capacity(plan, contract.move_count)
    submission = contract.sample_submission.copy(deep=True)
    submission_column = _submission_column(submission)
    records: list[SolutionRecord] = []
    if config.reflect_mode != "off":
        for source_puzzle_id in sorted(external_sources):
            for source_path, source_sha256 in external_sources[source_puzzle_id]:
                records.append(SolutionRecord(
                    puzzle_id=source_puzzle_id,
                    variant="source",
                    path=source_path,
                    original_oriented_path=source_path,
                    found_depth=_path_depth(source_path),
                    touch_depth=0,
                    source_solution_sha256=source_sha256,
                    reflected_source_path=source_path,
                    valid=True,
                    reached_state=contract.central_state,
                ))
    executions: list[InvocationExecution] = []
    statuses: list[CollectionStatus] = []

    def execute(invocation: RunnerInvocation) -> ParsedRunnerOutput:
        try:
            execution = _run_one(invocation, log_sanitizer=log_sanitizer) if log_sanitizer is not None else _run_one(invocation)
        except Exception as error:
            execution = getattr(error, "execution", None)
            if not isinstance(execution, InvocationExecution):
                raise
            executions.append(execution)
            partial = _build_artifacts(records, submission, executions, statuses)
            raise PublicSearchRunError(
                f"production runner failed for puzzle {invocation.puzzle_id} {invocation.variant} "
                f"with exit code {execution.return_code}; runner postprocess failed: {error}",
                partial,
            ) from error
        executions.append(execution)
        if execution.return_code != 0:
            partial = _build_artifacts(records, submission, executions, statuses)
            raise PublicSearchRunError(
                f"production runner failed for puzzle {invocation.puzzle_id} {invocation.variant} "
                f"with exit code {execution.return_code}",
                partial,
            )
        parsed = parse_runner_output(
            execution.output, invocation.result_tsv, invocation.puzzle_id, invocation.variant
        )
        statuses.append(parsed.collection_status)
        return parsed

    for puzzle_id in config.puzzle_ids:
        discovered_sources: list[tuple[str, str]] = list(external_sources[puzzle_id])
        if config.reflect_mode != "only":
            invocation = build_runner_invocation(
                config, plan, contract.move_count, puzzle_id, "original", weights_dir, artifact_dir,
                runner_path=runner_path, model=model,
            )
            original_records = _validated_records(execute(invocation), contract, puzzle_id, "original")
            records.extend(original_records)
            if config.reflect_mode == "after_original":
                discovered_sources.extend(
                    (record.original_oriented_path, sha256(record.original_oriented_path.encode("utf-8")).hexdigest())
                    for record in original_records
                )
        if config.reflect_mode in {"after_original", "only"}:
            sources = sorted(set(discovered_sources), key=lambda item: (_path_depth(item[0]), item[0], item[1]))
            for source_path, source_sha256 in sources:
                reflected_state = make_reflected_state(contract.central_state, source_path, contract.generators)
                invocation = build_runner_invocation(
                    config, plan, contract.move_count, puzzle_id, "reflected", weights_dir, artifact_dir,
                    runner_path=runner_path, model=model, source_solution_sha256=source_sha256,
                    reflected_source_path=source_path,
                )
                _write_reflected_test_row(invocation, puzzle_id, reflected_state)
                records.extend(_validated_records(
                    execute(invocation), contract, puzzle_id, "reflected",
                    reflected_state=reflected_state, source_sha256=source_sha256,
                    reflected_source_path=source_path,
                ))

        records = deduplicate_solutions(records)
        puzzle_records = [record for record in records if record.puzzle_id == puzzle_id]
        if puzzle_records:
            best = min(
                puzzle_records,
                key=lambda record: (
                    _path_depth(record.original_oriented_path),
                    record.original_oriented_path,
                    {"source": 0, "original": 1, "reflected": 2}[record.variant],
                    record.source_solution_sha256 or "",
                ),
            )
            submission.loc[
                submission["initial_state_id"] == puzzle_id, submission_column
            ] = best.original_oriented_path

    return _build_artifacts(records, submission, executions, statuses)
