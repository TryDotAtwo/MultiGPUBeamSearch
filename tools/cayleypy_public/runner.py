"""Deterministic host-side orchestration for the existing production runner."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import csv
import os
import socket
import subprocess
import time
import uuid
from typing import Literal, Sequence

from tools.cayleypy_public.config import PublicRunConfig
from tools.cayleypy_public.data import PuzzleContract
from tools.cayleypy_public.model import ExportedModel
from tools.cayleypy_public.paths import (
    SolutionRecord, invert_path, make_reflected_state, validate_original_solution,
)
from tools.cayleypy_public.profile import RuntimePlan


CollectionStatus = Literal["first_solution", "depth_reached", "capacity_reached", "not_collected"]


@dataclass(frozen=True)
class RunnerInvocation:
    command: tuple[str, ...]
    env: dict[str, str]
    result_tsv: Path | None
    combined_log: Path
    rank_logs: tuple[Path, Path]


@dataclass(frozen=True)
class ParsedRunnerOutput:
    records: tuple[SolutionRecord, ...]
    collection_status: CollectionStatus


@dataclass(frozen=True)
class RunArtifacts:
    solution_records: tuple[SolutionRecord, ...]
    combined_log: Path
    rank_logs: tuple[Path, Path]
    return_codes: tuple[int, ...]
    timing_summaries: tuple[float, ...]
    collection_status: CollectionStatus


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _runtime_env(config: PublicRunConfig, plan: RuntimePlan, weights_dir: Path) -> dict[str, str]:
    env = {
        "BEAM_WEIGHT_DIR": str(weights_dir),
        "BEAM_PUZZLE_INFO_JSON": str(config.puzzle_info_json),
        "BEAM_GENERATOR_PATH": str(config.puzzle_info_json),
        "BEAM_TEST_CSV": str(config.test_csv),
        "BEAM_B_MICRO": str(plan.runtime["b_micro"]),
        "BEAM_STREAM1_CONCURRENCY": str(plan.runtime["stream1_concurrency"]),
        "BEAM_STREAM3_RING_SLOTS": str(plan.runtime["stream3_ring_slots"]),
        "BEAM_SHARD_COUNT": str(plan.runtime["shard_count"]),
        "BEAM_SHARD_CAPACITY_CANDIDATES": str(plan.shard_capacity_candidates),
        "BEAM_STREAM4_BATCH_CANDIDATES": str(plan.runtime["stream4_batch_candidates"]),
        "BEAM_STREAM4_TRIGGER_CANDIDATES": str(plan.runtime["stream4_trigger_candidates"]),
        "BEAM_STREAM4_ACTIVE_SORT_SLOTS": str(plan.runtime["stream4_active_sort_slots"]),
        "BEAM_SOLVED_NEIGHBORHOOD_RADIUS": str(config.touch_bfs_radius),
        "BEAM_REPAIR_K1_RADIUS": str(config.touch_bfs_radius),
        "BEAM_REPAIR_K2_RADIUS": "0",
    }
    return env


def build_runner_invocation(
    config: PublicRunConfig, plan: RuntimePlan, puzzle_id: int, variant: Literal["original", "reflected"],
    weights_dir: Path, artifact_dir: Path, *, runner_path: str = "production_runner",
) -> RunnerInvocation:
    """Build a hermetic two-rank torchrun invocation for exactly one puzzle."""
    run_id = uuid.uuid4().hex
    run_root = artifact_dir / f"puzzle-{puzzle_id}" / variant / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    history_dir = Path("/tmp/beam_history_public") / run_id / str(puzzle_id) / variant
    result_tsv = run_root / "solve_bucket.tsv" if config.solution_mode == "collect" else None
    env = _runtime_env(config, plan, weights_dir)
    env["BEAM_HISTORY_DIR"] = history_dir.as_posix()
    if result_tsv is not None:
        snapshot_capacity = max(plan.stream3_batch_candidates, plan.runtime["stream4_batch_candidates"], plan.runtime["stream4_trigger_candidates"])
        env.update({
            "BEAM_SOLVE_BUCKET_MODE": "1",
            "BEAM_SOLVE_BUCKET_STOP_DEPTH": str(config.collect_until_depth),
            "BEAM_SOLVE_BUCKET_MAX_SOLUTIONS": str(config.max_collected_solutions),
            # Per-depth device snapshot capacity is independent from the host collection limit.
            "BEAM_SOLVED_RESULT_CAPACITY": str(snapshot_capacity),
            "BEAM_SOLVE_BUCKET_RESULT_TSV": str(result_tsv),
        })
    command = (
        "python", "-m", "torch.distributed.run", "--nproc-per-node=2",
        "--rdzv-backend=c10d", f"--rdzv-endpoint=127.0.0.1:{_free_port()}", f"--rdzv-id={run_id}",
        "--redirects=3", "--tee=0:3", "--no-python", runner_path,
        str(puzzle_id), str(config.max_depth), str(config.beam_width),
    )
    return RunnerInvocation(command, env, result_tsv, run_root / "combined.log",
                            (run_root / "rank-0.log", run_root / "rank-1.log"))


def parse_runner_output(output: str, result_tsv: Path | None, puzzle_id: int,
                        variant: Literal["original", "reflected"]) -> ParsedRunnerOutput:
    status: CollectionStatus = "not_collected"
    if "collection_status=capacity_reached" in output:
        status = "capacity_reached"
    elif "collection_status=depth_reached" in output:
        status = "depth_reached"
    records: list[SolutionRecord] = []
    if result_tsv is not None and result_tsv.exists():
        with result_tsv.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if int(row["puzzle_id"]) != puzzle_id:
                    continue
                path = row["solution"]
                records.append(SolutionRecord(puzzle_id, variant, path, path, int(row["total_depth"]),
                                              int(row["found_depth"]), None, True, ()))
    if not records and "solution_path=" in output:
        path = output.rsplit("solution_path=", 1)[1].splitlines()[0].strip()
        records.append(SolutionRecord(puzzle_id, variant, path, path, len(path.split(".")) if path else 0,
                                      0, None, True, ()))
        status = "first_solution"
    return ParsedRunnerOutput(tuple(records), status)


def _run_one(invocation: RunnerInvocation, extra_env: dict[str, str]) -> tuple[int, float, str]:
    environment = {key: value for key, value in os.environ.items() if key not in {"WORLD_SIZE", "RANK", "LOCAL_RANK"}}
    environment.update(invocation.env)
    environment.update(extra_env)
    started = time.monotonic()
    completed = subprocess.run(invocation.command, env=environment, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    invocation.combined_log.write_text(completed.stdout, encoding="utf-8")
    for rank_log in invocation.rank_logs:
        if not rank_log.exists():
            rank_log.write_text(completed.stdout if rank_log.name == "rank-0.log" else "", encoding="utf-8")
    return completed.returncode, elapsed, completed.stdout


def run_public_search(config: PublicRunConfig, contract: PuzzleContract, model: ExportedModel,
                      plan: RuntimePlan, weights_dir: Path, artifact_dir: Path,
                      *, runner_path: str = "production_runner") -> RunArtifacts:
    """Run originals/reflections and return only CPU-validated original-oriented paths."""
    del model  # Export validation is complete before orchestration.
    all_records: list[SolutionRecord] = []
    return_codes: list[int] = []
    timings: list[float] = []
    combined_logs: list[Path] = []
    rank_logs: list[Path] = []
    statuses: list[CollectionStatus] = []
    for puzzle_id in config.puzzle_ids:
        variants: list[tuple[Literal["original", "reflected"], dict[str, str]]] = []
        if config.reflect_mode != "only":
            variants.append(("original", {}))
        for variant, extras in variants:
            invocation = build_runner_invocation(config, plan, puzzle_id, variant, weights_dir, artifact_dir, runner_path=runner_path)
            code, elapsed, output = _run_one(invocation, extras)
            parsed = parse_runner_output(output, invocation.result_tsv, puzzle_id, variant)
            return_codes.append(code); timings.append(elapsed); combined_logs.append(invocation.combined_log); rank_logs.extend(invocation.rank_logs); statuses.append(parsed.collection_status)
            for record in parsed.records:
                if validate_original_solution(contract.initial_states[puzzle_id], contract.central_state, record.path, contract.generators):
                    all_records.append(SolutionRecord(record.puzzle_id, record.variant, record.path, record.path,
                                                       record.found_depth, record.touch_depth, record.source_solution_sha256,
                                                       True, contract.central_state))
        # Reflection is intentionally gated on validated original results; a caller may provide sources in a later task.
    final_status = "capacity_reached" if "capacity_reached" in statuses else ("depth_reached" if "depth_reached" in statuses else ("first_solution" if "first_solution" in statuses else "not_collected"))
    root = artifact_dir / f"puzzle-{config.puzzle_id_start}"
    return RunArtifacts(tuple(all_records), combined_logs[0] if combined_logs else root / "combined.log",
                        tuple(rank_logs[:2]) if rank_logs else (root / "rank-0.log", root / "rank-1.log"),
                        tuple(return_codes), tuple(timings), final_status)
