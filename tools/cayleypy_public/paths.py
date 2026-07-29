"""Fail-closed CayleyPy path replay, reflection, and solution deduplication."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, Mapping, Sequence


State = tuple[int, ...]
Generators = Mapping[str, Sequence[int]]


def tokenize_path(path: str) -> tuple[str, ...]:
    if not isinstance(path, str):
        raise ValueError("path must be a dot-separated string")
    if path == "":
        return ()
    tokens = tuple(path.split("."))
    if any(not token for token in tokens):
        raise ValueError("path must not contain empty move tokens")
    if any(
        len(token) > 128 or any(character < "!" or character > "~" for character in token)
        for token in tokens
    ):
        raise ValueError("path move tokens must be printable ASCII without spaces and at most 128 characters")
    return tokens


def _validated_generators(generators: Generators, state_len: int | None = None) -> dict[str, tuple[int, ...]]:
    if not generators:
        raise ValueError("generators must not be empty")
    normalized: dict[str, tuple[int, ...]] = {}
    expected_len = state_len
    for name, raw_permutation in generators.items():
        if not isinstance(name, str) or not name:
            raise ValueError("generator names must be non-empty strings")
        permutation = tuple(raw_permutation)
        if expected_len is None:
            expected_len = len(permutation)
        if len(permutation) != expected_len or set(permutation) != set(range(expected_len)):
            raise ValueError(f"generator {name!r} must be a permutation of range(state_len)")
        normalized[name] = permutation
    return normalized


def _inverse_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    inverse = [0] * len(permutation)
    for destination, source in enumerate(permutation):
        inverse[source] = destination
    return tuple(inverse)


def _inverse_names(generators: Generators) -> dict[str, str]:
    normalized = _validated_generators(generators)
    names_by_permutation: dict[tuple[int, ...], list[str]] = {}
    for name, permutation in normalized.items():
        names_by_permutation.setdefault(permutation, []).append(name)

    result: dict[str, str] = {}
    for name, permutation in normalized.items():
        matches = names_by_permutation.get(_inverse_permutation(permutation), [])
        if len(matches) != 1:
            raise ValueError(f"generator {name!r} does not have a unique inverse generator")
        result[name] = matches[0]
    return result


def apply_path(state: Sequence[int], path: str, generators: Generators) -> State:
    """Replay a standard dot-separated CayleyPy path on ``state``."""
    current = tuple(state)
    normalized = _validated_generators(generators, len(current))
    for move in tokenize_path(path):
        try:
            permutation = normalized[move]
        except KeyError as error:
            raise ValueError(f"unknown move token {move!r}") from error
        current = tuple(current[index] for index in permutation)
    return current


def invert_path(path: str, generators: Generators) -> str:
    """Return the algebraic inverse without relying on generator-name syntax."""
    inverse_names = _inverse_names(generators)
    moves = tokenize_path(path)
    for move in moves:
        if move not in inverse_names:
            raise ValueError(f"unknown move token {move!r}")
    return ".".join(inverse_names[move] for move in reversed(moves))


def make_reflected_state(central: Sequence[int], original_solution: str, generators: Generators) -> State:
    """Create the reflected start state whose solved path inverts to the original."""
    return apply_path(central, original_solution, generators)


def validate_original_solution(initial: Sequence[int], central: Sequence[int], path: str, generators: Generators) -> bool:
    """Return whether a complete, valid path takes the original state to center."""
    try:
        return apply_path(initial, path, generators) == tuple(central)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class SolutionRecord:
    puzzle_id: int
    variant: Literal["original", "reflected", "source"]
    path: str
    original_oriented_path: str
    found_depth: int
    touch_depth: int
    source_solution_sha256: str | None
    valid: bool
    reached_state: State
    reflected_source_path: str | None = None

    def __post_init__(self) -> None:
        try:
            normalized_state = tuple(self.reached_state)
        except TypeError as error:
            raise ValueError("reached_state must be an iterable state") from error
        object.__setattr__(self, "reached_state", normalized_state)


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _state_digest(state: Sequence[int]) -> str:
    return _digest(",".join(str(value) for value in state))


def _provenance_key(record: SolutionRecord) -> tuple[int, int, int, str, str, str]:
    return (
        1 if record.variant == "source" else 0,
        record.found_depth,
        record.touch_depth,
        record.source_solution_sha256 or "",
        record.variant,
        record.path,
    )


def deduplicate_solutions(records: Sequence[SolutionRecord]) -> list[SolutionRecord]:
    """Keep one deterministic earliest-provenance record per semantic solution."""
    selected: dict[tuple[int, str, str], SolutionRecord] = {}
    for record in records:
        if not record.valid:
            continue
        key = (record.puzzle_id, _digest(record.original_oriented_path), _state_digest(record.reached_state))
        previous = selected.get(key)
        if previous is None or _provenance_key(record) < _provenance_key(previous):
            selected[key] = record
    return list(selected.values())
