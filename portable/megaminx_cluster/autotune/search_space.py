"""Deterministic search primitives for cluster profile tuning."""
from __future__ import annotations

from collections.abc import Callable

from portable.megaminx_cluster.autotune.contracts import (
    BeamSearchResult,
    BfsBoundary,
    PuzzleContract,
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
