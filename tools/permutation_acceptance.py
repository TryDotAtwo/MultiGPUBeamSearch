"""Independent CPU oracle and strict validator for saturated permutation acceptance.

The oracle intentionally models the existing Hash128 owner/shard routing and
stable equal-score cutoff. It does not call GraphContract.replay or production
search code.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Iterable, Sequence


_MASK64 = (1 << 64) - 1
_HASH_SEED = 0xC0DEC0DE
_TRACE_MARKER = "depth_flow_trace"
_KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
_HEX_UUID = re.compile(r"(?:GPU-)?[0-9a-fA-F]{32}\Z")
_LABEL = re.compile(r"[A-Za-z0-9_.-]+\Z")


class MT19937_64:
    """Small, independent implementation of the C++ std::mt19937_64 engine."""

    _N = 312
    _M = 156
    _MATRIX_A = 0xB5026F5AA96619E9
    _UPPER_MASK = 0xFFFFFFFF80000000
    _LOWER_MASK = 0x7FFFFFFF

    def __init__(self, seed: int):
        state = [0] * self._N
        state[0] = seed & _MASK64
        for index in range(1, self._N):
            previous = state[index - 1]
            state[index] = (
                6364136223846793005 * (previous ^ (previous >> 62)) + index
            ) & _MASK64
        self._state = state
        self._index = self._N

    def _twist(self) -> None:
        for index in range(self._N):
            joined = (
                (self._state[index] & self._UPPER_MASK)
                | (self._state[(index + 1) % self._N] & self._LOWER_MASK)
            )
            shifted = joined >> 1
            if joined & 1:
                shifted ^= self._MATRIX_A
            self._state[index] = self._state[(index + self._M) % self._N] ^ shifted
        self._index = 0

    def next(self) -> int:
        if self._index >= self._N:
            self._twist()
        value = self._state[self._index]
        self._index += 1
        value ^= (value >> 29) & 0x5555555555555555
        value ^= (value << 17) & 0x71D67FFFEDA60000
        value ^= (value << 37) & 0xFFF7EEE000000000
        value ^= value >> 43
        return value & _MASK64


@dataclass(frozen=True)
class Candidate:
    state: tuple[int, ...]
    hash128: tuple[int, int]  # (lo, hi), matching the C++ storage layout
    source_rank: int
    owner_rank: int
    parent_idx: int
    move_id: int
    payload_id: int
    score_key: int
    route_packed: int


@dataclass(frozen=True)
class StepReference:
    input_counts: tuple[int, ...]
    eligible_distinct: int
    keep_count: int
    cross_source_duplicate_count: int
    all_scores_tied: bool
    ordered_survivors: tuple[Candidate, ...]
    frontier_by_rank: tuple[tuple[Candidate, ...], ...]


@dataclass(frozen=True)
class WorldReference:
    world_size: int
    steps: tuple[StepReference, ...]


@dataclass(frozen=True)
class SaturatedFixture:
    start: tuple[int, ...]
    distance: int
    scramble: tuple[int, ...]
    worlds: dict[int, WorldReference]


@lru_cache(maxsize=16)
def _zobrist_rows(state_len: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    value_pad = 256 if state_len > 128 else 128
    rng = MT19937_64(_HASH_SEED)
    return tuple(
        tuple((rng.next(), rng.next()) for _ in range(value_pad))
        for _ in range(state_len)
    )


def hash_state(state: Sequence[int]) -> tuple[int, int]:
    rows = _zobrist_rows(len(state))
    lo = hi = 0
    for position, raw_value in enumerate(state):
        value = int(raw_value)
        if not 0 <= value < len(rows[position]):
            raise ValueError(f"state value {value} at {position} exceeds Hash128 table")
        zlo, zhi = rows[position][value]
        lo ^= zlo
        hi ^= zhi
    return lo & _MASK64, hi & _MASK64


def _mix64(value: int) -> int:
    value &= _MASK64
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _MASK64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _MASK64
    value ^= value >> 31
    return value & _MASK64


def _rotl64(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (64 - shift))) & _MASK64


def _distribution_key(hash128: tuple[int, int], salt: int) -> int:
    lo, hi = hash128
    value = lo ^ _rotl64(hi, 32) ^ salt
    value ^= _mix64((hi + 0x9E3779B97F4A7C15) & _MASK64)
    return _mix64(value)


def _owner(hash128: tuple[int, int], world_size: int) -> int:
    return _distribution_key(hash128, 0x243F6A8885A308D3) % world_size


def _shard(hash128: tuple[int, int], shard_count: int) -> int:
    return _distribution_key(hash128, 0x13198A2E03707344) % shard_count


def _hash_sort_key(hash128: tuple[int, int]) -> tuple[int, int]:
    lo, hi = hash128
    return hi, lo


def _move(state: tuple[int, ...], generator: Sequence[int]) -> tuple[int, ...]:
    return tuple(state[int(source)] for source in generator)


def _route(source_rank: int, owner_rank: int, move_id: int) -> int:
    return (source_rank << 16) | (owner_rank << 8) | move_id


def _candidate_key(candidate: Candidate) -> tuple[int, int, int]:
    # src/stream4.cpp::candidate_better, applied after full Hash128 equality.
    return candidate.score_key, candidate.parent_idx, candidate.route_packed


def _partition(
    ordered: Sequence[Candidate], world_size: int
) -> tuple[tuple[Candidate, ...], ...]:
    count = len(ordered)
    result = []
    for rank in range(world_size):
        begin = (rank * count + world_size - 1) // world_size
        end = ((rank + 1) * count + world_size - 1) // world_size
        result.append(tuple(ordered[begin:end]))
    return tuple(result)


def _step(
    frontier_by_rank: tuple[tuple[Candidate, ...], ...] | tuple[tuple[int, ...], ...],
    generators: tuple[tuple[int, ...], ...],
    *,
    world_size: int,
    beam_width: int,
    shard_count: int,
    initial: bool = False,
) -> StepReference:
    input_counts = tuple(len(frontier) for frontier in frontier_by_rank)
    by_source: list[dict[tuple[int, int], Candidate]] = [dict() for _ in range(world_size)]
    hash_sources: dict[tuple[int, int], set[int]] = defaultdict(set)
    hash_states: dict[tuple[int, int], tuple[int, ...]] = {}

    for source_rank, frontier in enumerate(frontier_by_rank):
        for parent_idx, parent in enumerate(frontier):
            parent_state = tuple(parent) if initial else parent.state
            for move_id, generator in enumerate(generators):
                child = _move(parent_state, generator)
                h = hash_state(child)
                existing_state = hash_states.setdefault(h, child)
                if existing_state != child:
                    raise ValueError("fixture has a full Hash128 collision between distinct states")
                hash_sources[h].add(source_rank)
                owner_rank = _owner(h, world_size)
                payload_id = parent_idx * len(generators) + move_id
                candidate = Candidate(
                    state=child,
                    hash128=h,
                    source_rank=source_rank,
                    owner_rank=owner_rank,
                    parent_idx=parent_idx,
                    move_id=move_id,
                    payload_id=payload_id,
                    score_key=0,
                    route_packed=_route(source_rank, owner_rank, move_id),
                )
                current = by_source[source_rank].get(h)
                if current is None or (candidate.score_key, candidate.payload_id) < (
                    current.score_key,
                    current.payload_id,
                ):
                    by_source[source_rank][h] = candidate

    # Stream 4 routes by Hash128 owner, reduces full-hash matches, then keeps
    # the established per-shard (Hash128 hi,lo) order.
    by_owner: list[dict[tuple[int, int], Candidate]] = [dict() for _ in range(world_size)]
    for source_candidates in by_source:
        for h, candidate in source_candidates.items():
            current = by_owner[candidate.owner_rank].get(h)
            if current is None or _candidate_key(candidate) < _candidate_key(current):
                by_owner[candidate.owner_rank][h] = candidate

    owner_ordered: list[tuple[Candidate, ...]] = []
    for owner_rank in range(world_size):
        candidates = list(by_owner[owner_rank].values())
        candidates.sort(
            key=lambda item: (
                _shard(item.hash128, shard_count),
                *_hash_sort_key(item.hash128),
            )
        )
        owner_ordered.append(tuple(candidates))

    global_order = tuple(candidate for owner in owner_ordered for candidate in owner)
    keep_count = min(beam_width, len(global_order))
    survivors = global_order[:keep_count]
    cross_source_duplicates = sum(
        1 for sources in hash_sources.values() if len(sources) > 1
    )
    return StepReference(
        input_counts=input_counts,
        eligible_distinct=len(global_order),
        keep_count=keep_count,
        cross_source_duplicate_count=cross_source_duplicates,
        all_scores_tied=len({candidate.score_key for candidate in global_order}) <= 1,
        ordered_survivors=survivors,
        frontier_by_rank=_partition(survivors, world_size),
    )


def _distances_from(
    center: tuple[int, ...], generators: tuple[tuple[int, ...], ...]
) -> dict[tuple[int, ...], int]:
    distances = {center: 0}
    queue = deque([center])
    while queue:
        state = queue.popleft()
        for generator in generators:
            child = _move(state, generator)
            if child not in distances:
                distances[child] = distances[state] + 1
                queue.append(child)
    return distances


def _world_reference(
    start: tuple[int, ...],
    generators: tuple[tuple[int, ...], ...],
    *,
    world_size: int,
    beam_width: int,
    shard_count: int,
) -> WorldReference:
    empty_candidate = tuple()
    if world_size == 1:
        frontier: tuple[tuple[Candidate, ...], ...] = ((start,),)
    else:
        frontier = ((start,),) + tuple(() for _ in range(world_size - 1))
    steps = []
    for _depth in range(2):
        step = _step(
            frontier,
            generators,
            world_size=world_size,
            beam_width=beam_width,
            shard_count=shard_count,
            initial=True,
        )
        steps.append(step)
        frontier = step.frontier_by_rank
    return WorldReference(world_size, tuple(steps))


def build_saturated_reference_fixture(
    center: Sequence[int],
    generators: Iterable[Sequence[int]],
    *,
    beam_width: int,
    shard_count: int = 64,
) -> SaturatedFixture:
    """Choose the first deterministic exact-distance>=3 LRX state that saturates.

    The first beam creates three adjacent states, partitioned 2+1 for two
    ranks. Their next expansion necessarily sends the initial state back from
    both ranks; full Hash128 owner/shard routing decides which three survive.
    """
    center = tuple(map(int, center))
    generators = tuple(tuple(map(int, generator)) for generator in generators)
    if not center or not generators or beam_width <= 0 or shard_count <= 0:
        raise ValueError("center, generators, beam width and shard count must be positive")
    if any(len(generator) != len(center) for generator in generators):
        raise ValueError("every generator must match the logical state length")
    if any(sorted(generator) != list(range(len(center))) for generator in generators):
        raise ValueError("permutation fixture generators must be permutations")
    distances = _distances_from(center, generators)
    seen: set[tuple[int, ...]] = set()
    max_length = min(5, max(3, len(center) + 1))
    for length in range(3, max_length + 1):
        for sequence in __import__("itertools").product(range(len(generators)), repeat=length):
            start = center
            for move_id in sequence:
                start = _move(start, generators[move_id])
            if start in seen:
                continue
            seen.add(start)
            distance = distances.get(start)
            if distance is None or distance < 3:
                continue
            worlds = {
                world_size: _world_reference(
                    start,
                    generators,
                    world_size=world_size,
                    beam_width=beam_width,
                    shard_count=shard_count,
                )
                for world_size in (1, 2)
            }
            two_gpu = worlds[2].steps[1]
            if (
                two_gpu.input_counts == (2, 1)
                and all(worlds[w].steps[1].eligible_distinct > beam_width for w in (1, 2))
                and two_gpu.cross_source_duplicate_count > 0
                and two_gpu.keep_count == beam_width
                and all(two_gpu.frontier_by_rank)
            ):
                return SaturatedFixture(
                    start=start,
                    distance=distance,
                    scramble=tuple(sequence),
                    worlds=worlds,
                )
    raise AssertionError("could not find a deterministic saturated LRX-style fixture")


def decode_frontier_snapshot(
    raw: bytes | bytearray | memoryview, *, state_len: int, storage_len: int
) -> tuple[tuple[int, ...], ...]:
    """Decode bounded StatePacked records and enforce zero persistent padding."""
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise TypeError("frontier snapshot must be bytes-like")
    if type(state_len) is not int or type(storage_len) is not int:
        raise TypeError("state_len and storage_len must be integers")
    if state_len <= 0 or storage_len < state_len:
        raise ValueError("invalid frontier snapshot shape")
    view = memoryview(raw)
    if len(view) % storage_len:
        raise ValueError("frontier snapshot byte count is not a whole number of states")
    states = []
    for offset in range(0, len(view), storage_len):
        record = bytes(view[offset : offset + storage_len])
        if any(record[state_len:]):
            raise ValueError("persistent frontier padding must be zero")
        states.append(tuple(record[:state_len]))
    return tuple(states)


def _parse_trace(log: str) -> list[dict[str, str]]:
    records = []
    for line_number, line in enumerate(log.splitlines(), 1):
        marker = line.find(_TRACE_MARKER)
        if marker < 0:
            continue
        payload = line[marker + len(_TRACE_MARKER) :].strip()
        values: dict[str, str] = {}
        for key, value in _KV.findall(payload):
            if key in values:
                raise ValueError(f"duplicate {key} in depth-flow trace line {line_number}")
            values[key] = value
        if values:
            records.append(values)
    return records


def validate_depth_flow_trace(
    log: str,
    *,
    run_id: str,
    case_id: str,
    call_id: int,
    world_size: int,
    depth: int,
    beam_width: int,
    source_sha: str | None = None,
    build_key: str | None = None,
    binary_sha256: str | None = None,
) -> tuple[dict[str, object], ...]:
    """Validate one complete, saturated trace call without merging other runs."""
    for name, value in (("run_id", run_id), ("case_id", case_id)):
        if not isinstance(value, str) or not _LABEL.fullmatch(value):
            raise ValueError(f"{name} must be a nonempty portable label")
    if world_size not in (1, 2) or beam_width <= 0 or depth < 0 or call_id <= 0:
        raise ValueError("invalid trace topology, width, depth or call identity")
    selected = [
        record
        for record in _parse_trace(log)
        if record.get("run_id") == run_id
        and record.get("case_id") == case_id
        and record.get("call_id") == str(call_id)
        and record.get("depth") == str(depth)
    ]
    if len(selected) != world_size:
        raise ValueError(
            f"expected exactly {world_size} rank traces for run={run_id} case={case_id} "
            f"call={call_id} depth={depth}; found {len(selected)}"
        )

    int_fields = (
        "call_id", "rank", "world_size", "depth", "pid", "shard_count",
        "frontier_size", "generated", "stream3_unique", "final_global_less",
        "final_global_equal", "final_total_available", "final_global_keep",
        "final_candidate_count", "final_request_count", "next_frontier_size",
    )
    rows: list[dict[str, object]] = []
    for record in selected:
        row: dict[str, object] = dict(record)
        for key in int_fields:
            try:
                row[key] = int(record[key])
            except (KeyError, ValueError) as error:
                raise ValueError(f"depth-flow trace is missing a valid {key}") from error
        if row["world_size"] != world_size or row["depth"] != depth or row["call_id"] != call_id:
            raise ValueError("depth-flow trace has a mismatched world, depth or call")
        if row["rank"] < 0 or row["rank"] >= world_size or row["pid"] <= 0:
            raise ValueError("depth-flow trace has an invalid rank or worker PID")
        if row["shard_count"] <= 0 or row["frontier_size"] <= 0 or row["generated"] <= 0:
            raise ValueError("rank did no measurable expansion work")
        if row["stream3_unique"] <= 0 or row["next_frontier_size"] <= 0:
            raise ValueError("rank did not produce or retain frontier work")
        gpu_uuid = record.get("gpu_uuid", "")
        if not _HEX_UUID.fullmatch(gpu_uuid):
            raise ValueError("rank trace lacks an observed CUDA GPU UUID")
        row["gpu_uuid"] = gpu_uuid.lower().removeprefix("gpu-")
        if source_sha is not None and record.get("source_sha") != source_sha:
            raise ValueError("rank trace source SHA differs from this acceptance case")
        if build_key is not None and record.get("build_key") != build_key:
            raise ValueError("rank trace native build key differs from the accepted binary")
        if binary_sha256 is not None and record.get("binary_sha256") != binary_sha256:
            raise ValueError("rank trace binary SHA256 differs from the accepted binary")
        rows.append(row)

    rows.sort(key=lambda row: int(row["rank"]))
    ranks = [int(row["rank"]) for row in rows]
    if ranks != list(range(world_size)):
        raise ValueError("rank trace set is incomplete or contains a duplicate rank")
    if len({int(row["pid"]) for row in rows}) != world_size:
        raise ValueError("rank traces do not identify distinct worker processes")
    if len({str(row["gpu_uuid"]) for row in rows}) != world_size:
        raise ValueError("rank traces do not identify distinct physical GPUs")
    shared = (
        "final_global_less", "final_global_equal", "final_total_available",
        "final_global_keep", "shard_count", "source_sha", "build_key", "binary_sha256",
    )
    for key in shared:
        if len({row.get(key) for row in rows}) != 1:
            raise ValueError(f"rank traces disagree on shared field {key}")
    less = int(rows[0]["final_global_less"])
    equal = int(rows[0]["final_global_equal"])
    available = int(rows[0]["final_total_available"])
    keep = int(rows[0]["final_global_keep"])
    if less != 0 or equal <= beam_width or available <= beam_width or keep != beam_width:
        raise ValueError("depth is not an exact, saturated equal-score global cutoff")
    if sum(int(row["next_frontier_size"]) for row in rows) != beam_width:
        raise ValueError("rank-local output counts do not sum to the requested global width")
    return tuple(rows)
