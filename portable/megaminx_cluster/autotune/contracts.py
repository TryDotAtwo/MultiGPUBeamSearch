"""Immutable contracts shared by the cluster autotuner."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PuzzleContract:
    move_count: int
    hash_bytes: int
    max_bfs_radius: int


@dataclass(frozen=True)
class BfsBoundary:
    radius: int
    cumulative_states: int
    raw_bytes: int
    budget_bytes: int


@dataclass(frozen=True)
class BeamSearchResult:
    maximum_stable: int
    attempts: tuple[tuple[int, bool], ...]
