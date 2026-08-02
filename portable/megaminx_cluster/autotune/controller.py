"""Deadline-aware adaptive successive-halving controller."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import statistics
import time
from typing import Callable, Mapping

from portable.megaminx_cluster.autotune.contracts import TrialScore
from portable.megaminx_cluster.autotune.evidence import EvidenceStore, SessionIdentity
from portable.megaminx_cluster.autotune.probe import ProbeResult, ProbeRequest, run_probe
from portable.megaminx_cluster.autotune.search_space import (
    beam_anchors, candidate_grid, retain_round, round_schedule,
)
from portable.megaminx_cluster.profile import round_half_up_log2


@dataclass(frozen=True)
class TrialRequest:
    phase: str
    beam: int
    config_id: str
    runtime: Mapping[str, int]
    puzzle_id: int
    repetition: int
    bfs_radius: int

    @property
    def key(self) -> str:
        return f"{self.phase}:{self.beam}:{self.config_id}:{self.puzzle_id}:{self.repetition}"


@dataclass(frozen=True)
class ControllerConfig:
    identity: SessionIdentity
    seed_runtime: Mapping[str, int]
    max_beam_limit: int


@dataclass(frozen=True)
class ControllerResult:
    maximum_stable_beam: int
    bfs_radius: int
    complete: bool
    registry_fragment: Mapping[str, object]


class BudgetController:
    def __init__(self, clock: Callable[[], float], total_seconds: int, initial_estimate_seconds: float):
        self.clock = clock
        self.started = clock()
        self.deadline = self.started + total_seconds
        self.initial_estimate = initial_estimate_seconds
        self.observed: list[float] = []

    @property
    def estimate(self) -> float:
        return statistics.median(self.observed) if self.observed else self.initial_estimate

    def observe(self, seconds: float) -> None:
        if seconds > 0:
            self.observed.append(seconds)

    def can_launch(self, reserve_jobs: int = 0) -> bool:
        return self.clock() + self.estimate * (1 + reserve_jobs) <= self.deadline


def _peak_scalar(result: ProbeResult) -> int | None:
    value = result.metrics.get("peak_vram_mib")
    if isinstance(value, list) and value and all(isinstance(item, int) for item in value):
        return max(value)
    if isinstance(value, int):
        return value
    return None


def run_session(
    config: ControllerConfig,
    probe: Callable[[TrialRequest], ProbeResult],
    clock: Callable[[], float],
    store: EvidenceStore,
) -> ControllerResult:
    identity = config.identity
    budget = BudgetController(clock, identity.time_budget_seconds, initial_estimate_seconds=1.0)
    checkpoint = store.read_checkpoint()
    completed = set(checkpoint.get("completed_keys", []))
    cached = {str(row.get("key")): row for row in store.read_rows() if row.get("key")}
    complete = True

    def execute(trial: TrialRequest, reserve_jobs: int = 0) -> ProbeResult | None:
        nonlocal complete
        if trial.key in completed:
            row = cached.get(trial.key)
            if row is None:
                raise ValueError(f"checkpoint key has no evidence row: {trial.key}")
            return ProbeResult(bool(row.get("stable")), str(row.get("status")), row, ())
        if not budget.can_launch(reserve_jobs):
            complete = False
            return None
        before = clock()
        result = probe(trial)
        elapsed = max(0.000001, clock() - before)
        budget.observe(elapsed)
        row = {
            "key": trial.key, "phase": trial.phase, "beam": trial.beam,
            "profile_power": round_half_up_log2(trial.beam),
            "config_id": trial.config_id, "runtime": dict(trial.runtime),
            "puzzle_id": trial.puzzle_id, "repetition": trial.repetition,
            "bfs_radius": trial.bfs_radius, "stable": result.stable,
            "status": result.status, "wall_us": result.metrics.get("wall_us", int(elapsed * 1e6)),
            "peak_vram_mib": _peak_scalar(result),
        }
        store.append_trial(row)
        cached[trial.key] = row
        completed.add(trial.key)
        store.write_checkpoint({"completed_keys": sorted(completed), "last_phase": trial.phase})
        return result

    seed = dict(config.seed_runtime)
    first_puzzle = identity.puzzle_ids[0]
    low = identity.min_beam
    high: int | None = None
    beam = low
    while True:
        trial = TrialRequest("max_beam", beam, "seed", seed, first_puzzle, 0, identity.bfs_radius)
        result = execute(trial)
        if result is None:
            break
        if not result.stable:
            if beam == identity.min_beam:
                raise RuntimeError("minimum beam is not stable")
            high = beam
            break
        low = beam
        if beam == config.max_beam_limit:
            break
        beam = min(config.max_beam_limit, beam * 2)
    if high is not None:
        while high - low > 1:
            candidate = (low + high) // 2
            result = execute(TrialRequest("max_beam", candidate, "seed", seed, first_puzzle, 0, identity.bfs_radius))
            if result is None:
                break
            if result.stable:
                low = candidate
            else:
                high = candidate

    maximum = low
    anchors = beam_anchors(maximum, identity.min_beam)
    chosen: dict[int, dict[str, object]] = {}
    candidates = candidate_grid(seed)
    schedule = round_schedule(identity.puzzle_ids)
    for anchor in anchors:
        survivors = candidates
        for spec in schedule:
            scores = []
            for candidate in survivors:
                results = []
                for puzzle in spec.puzzle_ids:
                    for repetition in range(spec.repetitions):
                        result = execute(TrialRequest(
                            "halving", anchor, candidate.config_id, candidate.runtime,
                            puzzle, repetition, identity.bfs_radius,
                        ), reserve_jobs=9)
                        if result is None:
                            break
                        results.append(result)
                    if not complete:
                        break
                stable = bool(results) and all(item.stable for item in results)
                times = [int(item.metrics.get("wall_us", 0)) for item in results if item.stable]
                peaks = [_peak_scalar(item) for item in results if item.stable]
                scores.append(TrialScore(
                    candidate.config_id, stable,
                    int(statistics.median(times)) if times else None,
                    int(max(item for item in peaks if item is not None)) if any(item is not None for item in peaks) else None,
                ))
                if not complete:
                    break
            if not scores:
                break
            ranked = retain_round(tuple(scores), spec.retain_fraction)
            by_id = {item.config_id: item for item in survivors}
            survivors = tuple(by_id[item.config_id] for item in ranked)
            if not complete:
                break
        winner = survivors[0] if survivors else candidates[0]
        power = round_half_up_log2(anchor)
        chosen[power] = {"runtime": dict(winner.runtime), "evidence_id": f"autotune-{winner.config_id}-p{power}"}
        if not complete:
            break
        for puzzle in identity.puzzle_ids:
            for repetition in range(3):
                result = execute(TrialRequest(
                    "final", anchor, winner.config_id, winner.runtime,
                    puzzle, repetition, identity.bfs_radius,
                ), reserve_jobs=max(0, 8 - (identity.puzzle_ids.index(puzzle) * 3 + repetition)))
                if result is None:
                    break
            if not complete:
                break
        if not complete:
            break

    if not chosen:
        power = round_half_up_log2(maximum)
        chosen[power] = {"runtime": seed, "evidence_id": f"autotune-seed-p{power}"}
    store.write_leaderboard()
    fragment = store.emit_registry_fragment(chosen)
    statuses = [anchor["status"] for anchor in fragment["profiles"][0]["anchors"].values()]
    complete = complete and bool(statuses) and all(status == "measured" for status in statuses)
    store.write_checkpoint({"completed_keys": sorted(completed), "last_phase": "emit", "complete": complete})
    return ControllerResult(maximum, identity.bfs_radius, complete, fragment)


_BOOTSTRAP_RUNTIME = {
    "b_micro": 8192, "stream1_concurrency": 1, "stream3_ring_slots": 1,
    "shard_count": 8, "shard_capacity_scale_ppm": 2500000,
    "stream4_batch_candidates": 262144, "stream4_trigger_candidates": 524288,
    "stream4_active_sort_slots": 1, "final_materialize_chunk_candidates": 65536,
}


def main() -> int:
    root = Path(os.environ["MEGAMINX_ARCHIVE_ROOT"]).resolve()
    run_dir = Path(os.environ["MEGAMINX_AUTOTUNE_RUN_DIR"]).resolve()
    manifest_bytes = (root / "MANIFEST.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
    puzzles = tuple(int(item) for item in os.environ["MEGAMINX_AUTOTUNE_PUZZLES"].split(":"))
    move_count = int(manifest["move_count"])
    hash_bytes = int(manifest.get("hash_bytes", 16))
    budget_bytes = int(os.environ["MEGAMINX_AUTOTUNE_BFS_HASH_BUDGET_MIB"]) * 2**20
    from portable.megaminx_cluster.autotune.contracts import PuzzleContract
    from portable.megaminx_cluster.autotune.search_space import derive_bfs_boundary
    boundary = derive_bfs_boundary(PuzzleContract(move_count, hash_bytes, 32), budget_bytes)
    first_gpu = preflight["gpus"][0]
    identity = SessionIdentity(
        str(first_gpu["family"]), int(first_gpu["vram_mib"]), int(first_gpu["sm"]),
        int(preflight["world_size"]), str(first_gpu["driver_major"]),
        str(manifest["solver_commit"]), str(manifest["model_sha256"]),
        __import__("hashlib").sha256(manifest_bytes).hexdigest(), str(manifest["backend"]),
        str(manifest["model_class"]), puzzles, int(os.environ["MEGAMINX_AUTOTUNE_MIN_BEAM"]),
        int(os.environ["MEGAMINX_AUTOTUNE_TIME_BUDGET_SECONDS"]), boundary.radius,
        move_count, hash_bytes, budget_bytes, boundary.cumulative_states,
    )
    store = EvidenceStore.create_or_resume(run_dir, identity)
    sequence = {"value": 0}

    def real_probe(trial: TrialRequest) -> ProbeResult:
        sequence["value"] += 1
        request = ProbeRequest(
            root, run_dir / "candidates" / f"{sequence['value']:06d}-{trial.key.replace(':', '-')}",
            identity.world_size, trial.puzzle_id, 8, trial.beam, trial.runtime,
            max(60, int(min(1800, identity.time_budget_seconds / 6))),
            f"{os.environ.get('SLURM_JOB_ID', 'manual')}-{sequence['value']}",
            (identity.vram_mib,) * identity.world_size,
            int(manifest.get("history_disk_bytes", 1)),
            {"solver_commit": identity.solver_commit, "manifest_digest": identity.release_manifest_digest},
        )
        return run_probe(request)

    max_beam = int(os.environ.get("MEGAMINX_AUTOTUNE_MAX_BEAM", str(2**40)))
    if max_beam < identity.min_beam:
        raise ValueError("MEGAMINX_AUTOTUNE_MAX_BEAM must be >= minimum beam")
    result = run_session(ControllerConfig(identity, _BOOTSTRAP_RUNTIME, max_beam), real_probe, time.monotonic, store)
    print(f"maximum_stable_beam={result.maximum_stable_beam} complete={int(result.complete)}")
    return 0 if result.complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
