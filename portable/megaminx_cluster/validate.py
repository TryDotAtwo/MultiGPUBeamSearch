"""Independent CPU replay helpers for Megaminx results."""

from __future__ import annotations

from typing import Mapping, Sequence


def apply_path(
    state: Sequence[int], path: Sequence[str], generators: Mapping[str, Sequence[int]]
) -> list[int]:
    current = list(state)
    for token in path:
        if token not in generators:
            raise ValueError(f"unknown move token: {token}")
        permutation = generators[token]
        if len(permutation) != len(current):
            raise ValueError(f"generator length mismatch for move: {token}")
        current = [current[index] for index in permutation]
    return current


def invert_token(token: str) -> str:
    return token[1:] if token.startswith("-") else f"-{token}"


def invert_path(path: Sequence[str]) -> list[str]:
    return [invert_token(token) for token in reversed(path)]


def validate_solution(
    initial: Sequence[int],
    central: Sequence[int],
    path: Sequence[str],
    generators: Mapping[str, Sequence[int]],
) -> bool:
    return apply_path(initial, path, generators) == list(central)
