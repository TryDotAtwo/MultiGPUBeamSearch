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
    return tuple(sorted(anchors, reverse=True))


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
    "final_materialize_chunk_candidates",
)

_FINAL_CHUNK_VALUES = (32768, 65536, 98304, 131072)

def _candidate_id(runtime: Mapping[str, int]) -> str:
    payload = json.dumps(dict(sorted(runtime.items())), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


def candidate_grid(seed_runtime: Mapping[str, int]) -> tuple[RuntimeCandidate, ...]:
    """Broad deterministic mixed grid; measurements remain hardware-tuple specific."""
    from portable.megaminx_cluster.profile import RUNTIME_KEYS

    if set(seed_runtime) != set(RUNTIME_KEYS):
        raise ValueError("runtime keys must exactly match the production contract")
    seed = {key: _positive_int(seed_runtime[key], f"runtime.{key}") for key in sorted(RUNTIME_KEYS)}
    if seed["final_materialize_chunk_candidates"] not in _FINAL_CHUNK_VALUES:
        raise ValueError("final materialize chunk must stay in the bounded small range")

    domains = {
        "b_micro": (1024, 2048, 4096, 8192, 16384, 32768, 65536),
        "stream1_concurrency": (1, 2, 4),
        "stream3_ring_slots": (1, 2, 4),
        "shard_count": (2, 4, 8, 16, 32, 64),
        "shard_capacity_scale_ppm": (1000000, 1250000, 1500000, 2000000, 2500000),
        "stream4_batch_candidates": (65536, 98304, 131072, 196608, 262144, 524288),
        "stream4_active_sort_slots": (1, 2, 4),
        "final_materialize_chunk_candidates": _FINAL_CHUNK_VALUES,
    }
    configs: list[dict[str, int]] = [dict(seed)]
    # Deterministic mixed designs expose cross-parameter interactions without a huge Cartesian grid.
    keys = tuple(domains)
    strides = (1, 5, 7, 11, 13, 17, 19, 23)
    for index in range(24):
        config = dict(seed)
        for key, stride in zip(keys, strides):
            values = domains[key]
            config[key] = values[(index * stride + stride // 2) % len(values)]
        batch = config["stream4_batch_candidates"]
        config["stream4_trigger_candidates"] = batch * (1 + (index % 2))
        configs.append(config)

    valid = []
    for config in configs:
        if config["stream4_trigger_candidates"] < config["stream4_batch_candidates"]:
            continue
        if config["final_materialize_chunk_candidates"] > 131072:
            continue
        valid.append(config)
    unique = {_candidate_id(config): config for config in valid}
    seed_id = _candidate_id(seed)
    ordered_ids = (seed_id,) + tuple(sorted(item for item in unique if item != seed_id))
    return tuple(RuntimeCandidate(item, MappingProxyType(unique[item])) for item in ordered_ids)

def retain_round(scores: tuple[TrialScore, ...], fraction: float) -> tuple[TrialScore, ...]:
    if not scores:
        raise ValueError("scores must not be empty")
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    count = max(1, math.ceil(len(scores) * fraction))
    stable_times = [
        item.median_wall_us for item in scores
        if item.stable and item.median_wall_us is not None
    ]
    fastest = min(stable_times) if stable_times else None
    fast_limit = None if fastest is None else fastest * 1.03
    ordered = sorted(
        scores,
        key=lambda item: (
            not item.stable,
            fast_limit is None or item.median_wall_us is None or item.median_wall_us > fast_limit,
            item.peak_vram_mib if fast_limit is not None and item.median_wall_us is not None and item.median_wall_us <= fast_limit and item.peak_vram_mib is not None else math.inf,
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
        RoundSpec(puzzle_ids[:1], warmups=0, repetitions=1, retain_fraction=0.375),
        RoundSpec(puzzle_ids[:2], warmups=0, repetitions=1, retain_fraction=0.34),
        RoundSpec(puzzle_ids, warmups=0, repetitions=3, retain_fraction=1.0),
    )
