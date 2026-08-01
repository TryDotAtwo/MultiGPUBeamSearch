"""Immutable contracts shared by the cluster autotuner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


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
@dataclass(frozen=True)
class RuntimeCandidate:
    config_id: str
    runtime: Mapping[str, int]


@dataclass(frozen=True)
class TrialScore:
    config_id: str
    stable: bool
    median_wall_us: int | None
    peak_vram_mib: int | None


@dataclass(frozen=True)
class RoundSpec:
    puzzle_ids: tuple[int, ...]
    warmups: int
    repetitions: int
    retain_fraction: float
