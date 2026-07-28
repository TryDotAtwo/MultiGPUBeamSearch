"""Deterministic host-side orchestration for the existing production runner."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
import csv
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from typing import Literal

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
_UINT32_MAX = 2**32 - 1
_SOLVED_RECORD_BYTES = 32 + 4 + 4
_T4_DEVICE_BYTES = 16 * 1024**3
_T4_HEADROOM_BYTES = 768 * 1024**2
_MAX_GATHER_RECORDS_PER_CHUNK = 65_536
_RANK_ENV_KEYS = frozenset({"WORLD_SIZE", "RANK", "LOCAL_RANK"})
_HISTORY_MODE = "static_hybrid"
_HISTORY_SLOT_COUNT = 2
_HISTORY_WORKERS = 1
_HISTORY_RAM_BYTES = 28 * 1024**3
_HISTORY_DISK_BYTES = 32 * 1024**3
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
    rank_logs: tuple[Path, Path]
    torchrun_log_dir: Path
    history_dir: Path
    puzzle_id: int
    variant: Variant
    source_solution_sha256: str | None = None


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


def preflight_history_runtime(
    config: PublicRunConfig,
    plan: RuntimePlan,
    move_count: int,
    history_dir: Path,
    *,
    tmp_free_bytes: int | None = None,
) -> HistoryRuntimePreflight:
    """Fail closed on the bounded two-rank static-hybrid history contract."""
    if isinstance(move_count, bool) or not isinstance(move_count, int) or move_count <= 0:
        raise ValueError("move_count must be a positive integer")
    _resolved_history_path(history_dir)
    if tmp_free_bytes is None:
        tmp_free_bytes = shutil.disk_usage(str(_HISTORY_ROOT.parent)).free
    if isinstance(tmp_free_bytes, bool) or not isinstance(tmp_free_bytes, int) or tmp_free_bytes < 0:
        raise ValueError("tmp_free_bytes must be a nonnegative integer")
    if tmp_free_bytes < _HISTORY_DISK_BYTES:
        raise ValueError(
            f"/tmp free disk is smaller than the history disk budget "
            f"({tmp_free_bytes} < {_HISTORY_DISK_BYTES} bytes)"
        )

    per_rank_ram = _HISTORY_RAM_BYTES // 2
    per_rank_disk = _HISTORY_DISK_BYTES // 2
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


def _runtime_env(config: PublicRunConfig, plan: RuntimePlan, weights_dir: Path) -> dict[str, str]:
    return {
        "BEAM_WEIGHT_DIR": str(weights_dir),
        "BEAM_PUZZLE_INFO_JSON": str(config.puzzle_info_json),
        "BEAM_GENERATOR_PATH": str(config.puzzle_info_json),
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
        "BEAM_DEPTH_LOG_EVERY": "1",
    }


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
    source_solution_sha256: str | None = None,
) -> RunnerInvocation:
    """Build a hermetic two-rank torchrun invocation for exactly one variant."""
    run_id = uuid.uuid4().hex
    run_root = artifact_dir / f"puzzle-{puzzle_id}" / variant / run_id
    torchrun_log_dir = run_root / "torchrun"
    history_dir = Path(_HISTORY_ROOT.as_posix()) / run_id / str(puzzle_id) / variant
    result_tsv = run_root / "solve_bucket.tsv" if config.solution_mode == "collect" else None
    preflight_history_runtime(config, plan, move_count, history_dir)
    run_root.mkdir(parents=True, exist_ok=False)
    env = _runtime_env(config, plan, weights_dir)
    env.update({
        "BEAM_HISTORY_MODE": _HISTORY_MODE,
        "BEAM_HISTORY_SLOT_COUNT": str(_HISTORY_SLOT_COUNT),
        "BEAM_HISTORY_WORKERS": str(_HISTORY_WORKERS),
        "BEAM_HISTORY_RAM_BYTES": str(_HISTORY_RAM_BYTES),
        "BEAM_HISTORY_DISK_BYTES": str(_HISTORY_DISK_BYTES),
        "BEAM_HISTORY_DIR": history_dir.as_posix(),
        "BEAM_HISTORY_DISK_PATH": history_dir.as_posix(),
    })
    env["BEAM_NCCL_ID_FILE"] = str(run_root / "nccl-id.bin")
    if result_tsv is not None:
        snapshot_capacity = derive_solved_result_capacity(plan, move_count)
        derive_gather_chunk_plan(plan.local_beam, snapshot_capacity)
        env.update({
            "BEAM_SOLVE_BUCKET_MODE": "1",
            "BEAM_SOLVE_BUCKET_STOP_DEPTH": str(config.collect_until_depth),
            "BEAM_SOLVE_BUCKET_MAX_SOLUTIONS": str(config.max_collected_solutions),
            "BEAM_SOLVED_RESULT_CAPACITY": str(snapshot_capacity),
            "BEAM_SOLVE_BUCKET_RESULT_TSV": str(result_tsv),
        })
    command = (
        "python", "-m", "torch.distributed.run", "--nproc-per-node=2",
        "--rdzv-backend=c10d", f"--rdzv-endpoint=127.0.0.1:{_free_port()}", f"--rdzv-id={run_id}",
        f"--log-dir={torchrun_log_dir}", "--redirects=3", "--tee=0:3", "--no-python", runner_path,
        str(puzzle_id), str(config.max_depth), str(config.beam_width),
    )
    return RunnerInvocation(
        command=command,
        env=env,
        result_tsv=result_tsv,
        combined_log=run_root / "combined.log",
        rank_logs=(run_root / "rank-0.log", run_root / "rank-1.log"),
        torchrun_log_dir=torchrun_log_dir,
        history_dir=history_dir,
        puzzle_id=puzzle_id,
        variant=variant,
        source_solution_sha256=source_solution_sha256,
    )


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
                    valid=True,
                    reached_state=(),
                ))
    if not records:
        matches = re.findall(r"(?<![A-Za-z0-9_])solution_path=([^\r\n]*)", output)
        if matches:
            path = matches[-1].strip()
            records.append(SolutionRecord(
                puzzle_id=puzzle_id,
                variant=variant,
                path=path,
                original_oriented_path=path,
                found_depth=_path_depth(path),
                touch_depth=0,
                source_solution_sha256=None,
                valid=True,
                reached_state=(),
            ))
            if status == "not_collected":
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


def collect_torchrun_rank_logs(invocation: RunnerInvocation) -> None:
    """Materialize readable per-rank files exclusively from torchrun redirects."""
    captured: list[tuple[Path, str]] = []
    for rank, output_path in enumerate(invocation.rank_logs):
        stdout, stderr = _read_rank_streams(invocation.torchrun_log_dir, rank)
        if stdout and not stdout.endswith("\n"):
            stdout += "\n"
        if stderr and not stderr.endswith("\n"):
            stderr += "\n"
        captured.append((output_path, f"[stdout]\n{stdout}[stderr]\n{stderr}"))
    for output_path, text in captured:
        output_path.write_text(text, encoding="utf-8")


def stream_process_output(
    process: object,
    combined_log: Path,
    *,
    live_stream: object | None = None,
    max_tail_chars: int = 4 * 1024 * 1024,
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
            live_stream.write(chunk)
            live_stream.flush()
            log.write(chunk)
            log.flush()
            tail = (tail + chunk)[-max_tail_chars:]
    return int(process.wait()), tail


def _run_one(invocation: RunnerInvocation, extra_env: dict[str, str]) -> InvocationExecution:
    environment = {key: value for key, value in os.environ.items() if key not in _RANK_ENV_KEYS}
    environment.update(invocation.env)
    environment.update(extra_env)
    started = time.monotonic()
    execution: InvocationExecution | None = None
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
        return_code, output_tail = stream_process_output(process, invocation.combined_log)
        elapsed = time.monotonic() - started
        execution = InvocationExecution(invocation, return_code, elapsed, output_tail)
        try:
            collect_torchrun_rank_logs(invocation)
        except RuntimeError as error:
            raise InvocationLogCaptureError(
                execution, f"torchrun log capture failed: {error}"
            ) from error
        return execution
    finally:
        try:
            cleanup_history_runtime(invocation.history_dir)
        except (OSError, RuntimeError) as error:
            if execution is not None:
                raise InvocationLogCaptureError(
                    execution, f"history scratch cleanup failed: {error}"
                ) from error
            raise


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
) -> RunArtifacts:
    """Run requested variants and retain only CPU-validated original-oriented paths."""
    del model
    external_sources = _reflection_sources(config, contract)  # Must finish before any GPU launch.
    if config.solution_mode == "collect":
        derive_solved_result_capacity(plan, contract.move_count)
    submission = contract.sample_submission.copy(deep=True)
    submission_column = _submission_column(submission)
    records: list[SolutionRecord] = []
    executions: list[InvocationExecution] = []
    statuses: list[CollectionStatus] = []

    def execute(invocation: RunnerInvocation) -> ParsedRunnerOutput:
        try:
            execution = _run_one(invocation, {})
        except InvocationLogCaptureError as error:
            execution = error.execution
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
                runner_path=runner_path,
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
                    runner_path=runner_path, source_solution_sha256=source_sha256,
                )
                _write_reflected_test_row(invocation, puzzle_id, reflected_state)
                records.extend(_validated_records(
                    execute(invocation), contract, puzzle_id, "reflected",
                    reflected_state=reflected_state, source_sha256=source_sha256,
                ))

        records = deduplicate_solutions(records)
        puzzle_records = [record for record in records if record.puzzle_id == puzzle_id]
        if puzzle_records:
            best = min(
                puzzle_records,
                key=lambda record: (
                    _path_depth(record.original_oriented_path),
                    record.original_oriented_path,
                    0 if record.variant == "original" else 1,
                    record.source_solution_sha256 or "",
                ),
            )
            submission.loc[
                submission["initial_state_id"] == puzzle_id, submission_column
            ] = best.original_oriented_path

    return _build_artifacts(records, submission, executions, statuses)
