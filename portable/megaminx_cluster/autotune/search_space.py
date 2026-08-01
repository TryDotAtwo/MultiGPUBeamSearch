"""Deterministic search primitives for cluster profile tuning."""
from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import math
from types import MappingProxyType

from portable.megaminx_cluster.autotune.contracts import (
    BeamSearchResult,
    BfsBoundary,
    PuzzleContract,
    RoundSpec,
    RuntimeCandidate,
    TrialScore,
)

_U64_MAX = 2**64 - 1


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return value


def derive_bfs_boundary(contract: PuzzleContract, budget_bytes: int) -> BfsBoundary:
    moves = _positive_int(contract.move_count, "move_count")
    width = _positive_int(contract.hash_bytes, "hash_bytes")
    cap = _positive_int(contract.max_bfs_radius, "max_bfs_radius", allow_zero=True)
    budget = _positive_int(budget_bytes, "budget_bytes")
    if width > _U64_MAX or budget > _U64_MAX:
        raise OverflowError("BFS byte calculation exceeds uint64")
    if width > budget:
        raise ValueError("BFS budget cannot fit radius zero hash")

    radius = 0
    layer = 1
    cumulative = 1
    while radius < cap:
        if layer > _U64_MAX // moves:
            raise OverflowError("BFS layer count exceeds uint64")
        next_layer = layer * moves
        if cumulative > _U64_MAX - next_layer:
            raise OverflowError("BFS cumulative state count exceeds uint64")
        next_cumulative = cumulative + next_layer
        if next_cumulative > budget // width:
            break
        radius += 1
        layer = next_layer
        cumulative = next_cumulative
    return BfsBoundary(radius, cumulative, cumulative * width, budget)


def beam_anchors(max_beam: int, min_beam: int) -> tuple[int, ...]:
    maximum = _positive_int(max_beam, "max_beam")
    minimum = _positive_int(min_beam, "min_beam")
    if maximum < minimum:
        raise ValueError("max_beam must be at least min_beam")
    anchors = {minimum, maximum}
    power = 1
    while power < minimum:
        power <<= 1
    while power <= maximum:
        anchors.add(power)
        power <<= 1
    return tuple(sorted(anchors))


def refine_max_stable(
    probe: Callable[[int], bool],
    min_beam: int,
    max_beam_limit: int,
) -> BeamSearchResult:
    minimum = _positive_int(min_beam, "min_beam")
    limit = _positive_int(max_beam_limit, "max_beam_limit")
    if limit < minimum:
        raise ValueError("max_beam_limit must be at least min_beam")
    attempts: list[tuple[int, bool]] = []

    def attempt(beam: int) -> bool:
        stable = bool(probe(beam))
        attempts.append((beam, stable))
        return stable

    if not attempt(minimum):
        raise RuntimeError("minimum beam is not stable")
    low = minimum
    high: int | None = None
    while low < limit:
        candidate = min(limit, low * 2)
        if attempt(candidate):
            low = candidate
            if candidate == limit:
                return BeamSearchResult(low, tuple(attempts))
        else:
            high = candidate
            break
    if high is None:
        return BeamSearchResult(low, tuple(attempts))
    while high - low > 1:
        candidate = (low + high) // 2
        if attempt(candidate):
            low = candidate
        else:
            high = candidate
    return BeamSearchResult(low, tuple(attempts))
_TUNED_KEYS = (
    "b_micro",
    "stream1_concurrency",
    "stream3_ring_slots",
    "shard_count",
    "shard_capacity_scale_ppm",
    "stream4_batch_candidates",
    "stream4_trigger_candidates",
    "stream4_active_sort_slots",
)


def _candidate_id(runtime: Mapping[str, int]) -> str:
    payload = json.dumps(dict(sorted(runtime.items())), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


def candidate_grid(seed_runtime: Mapping[str, int]) -> tuple[RuntimeCandidate, ...]:
    from portable.megaminx_cluster.profile import RUNTIME_KEYS

    if set(seed_runtime) != set(RUNTIME_KEYS):
        raise ValueError("runtime keys must exactly match the production contract")
    seed = {key: _positive_int(seed_runtime[key], f"runtime.{key}") for key in sorted(RUNTIME_KEYS)}
    configs: list[dict[str, int]] = [seed]
    for key in _TUNED_KEYS:
        value = seed[key]
        values = {max(1, value // 2), value, value * 2}
        for candidate_value in sorted(values):
            if candidate_value == value:
                continue
            config = dict(seed)
            config[key] = candidate_value
            if config["stream4_trigger_candidates"] < config["stream4_batch_candidates"]:
                continue
            configs.append(config)
    unique = {_candidate_id(config): config for config in configs}
    seed_id = _candidate_id(seed)
    ordered_ids = (seed_id,) + tuple(sorted(item for item in unique if item != seed_id))
    return tuple(
        RuntimeCandidate(config_id, MappingProxyType(unique[config_id]))
        for config_id in ordered_ids
    )


def retain_round(scores: tuple[TrialScore, ...], fraction: float) -> tuple[TrialScore, ...]:
    if not scores:
        raise ValueError("scores must not be empty")
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    count = max(1, math.ceil(len(scores) * fraction))
    ordered = sorted(
        scores,
        key=lambda item: (
            not item.stable,
            item.median_wall_us if item.median_wall_us is not None else math.inf,
            item.peak_vram_mib if item.peak_vram_mib is not None else math.inf,
            item.config_id,
        ),
    )
    return tuple(ordered[:count])


def round_schedule(puzzle_ids: tuple[int, ...]) -> tuple[RoundSpec, ...]:
    if len(puzzle_ids) != 3 or len(set(puzzle_ids)) != 3 or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in puzzle_ids
    ):
        raise ValueError("exactly three distinct nonnegative puzzle ids are required")
    return (
        RoundSpec(puzzle_ids[:1], warmups=1, repetitions=1, retain_fraction=0.5),
        RoundSpec(puzzle_ids[:2], warmups=0, repetitions=1, retain_fraction=0.5),
        RoundSpec(puzzle_ids, warmups=0, repetitions=3, retain_fraction=1.0),
    )
