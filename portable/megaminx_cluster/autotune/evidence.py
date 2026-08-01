"""Atomic evidence and fail-closed profile fragment emission."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import os
import tempfile
from typing import Mapping


@dataclass(frozen=True)
class SessionIdentity:
    gpu_family: str
    vram_mib: int
    sm: int
    world_size: int
    driver: str
    solver_commit: str
    model_digest: str
    release_manifest_digest: str
    backend: str
    model_class: str
    puzzle_ids: tuple[int, int, int]
    min_beam: int
    time_budget_seconds: int
    bfs_radius: int
    move_count: int
    hash_bytes: int
    bfs_hash_budget_bytes: int
    bfs_cumulative_states: int


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class EvidenceStore:
    def __init__(self, run_dir: Path, identity: SessionIdentity):
        self.run_dir = run_dir
        self.identity = identity
        self.results_path = run_dir / "autotune_results.jsonl"

    @classmethod
    def create_or_resume(cls, run_dir: Path, identity: SessionIdentity) -> "EvidenceStore":
        run_dir = run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        session_path = run_dir / "session.json"
        expected = asdict(identity)
        if session_path.exists():
            observed = json.loads(session_path.read_text(encoding="utf-8"))
            if _canonical(observed) != _canonical(expected):
                raise ValueError("session identity mismatch")
        else:
            _atomic_json(session_path, expected)
        store = cls(run_dir, identity)
        store._read_rows()
        return store

    def _read_rows(self) -> list[dict[str, object]]:
        if not self.results_path.exists():
            return []
        rows = []
        for line_number, line in enumerate(
            self.results_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid evidence JSONL line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"invalid evidence JSONL line {line_number}")
            rows.append(value)
        return rows

    def read_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(self._read_rows())

    def read_checkpoint(self) -> dict[str, object]:
        path = self.run_dir / "resume.json"
        if not path.exists():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("resume checkpoint must be an object")
        return value
    def append_trial(self, row: Mapping[str, object]) -> None:
        payload = _canonical(dict(row))
        with self.results_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def write_checkpoint(self, state: Mapping[str, object]) -> None:
        _atomic_json(self.run_dir / "resume.json", dict(state))

    def write_leaderboard(self) -> None:
        rows = self._read_rows()
        rows.sort(key=lambda row: (
            row.get("stable") is not True,
            row.get("wall_us") if isinstance(row.get("wall_us"), (int, float)) else float("inf"),
            row.get("peak_vram_mib") if isinstance(row.get("peak_vram_mib"), (int, float)) else float("inf"),
            str(row.get("config_id", "")),
        ))
        path = self.run_dir / "leaderboard.tsv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("config_id", "stable", "wall_us", "peak_vram_mib"))
            for row in rows:
                writer.writerow((
                    row.get("config_id", ""), str(row.get("stable", False)).lower(),
                    row.get("wall_us", ""), row.get("peak_vram_mib", ""),
                ))

    def _anchor_measured(self, power: int) -> bool:
        rows = [
            row for row in self._read_rows()
            if row.get("phase") == "final" and row.get("profile_power") == power
        ]
        if any(row.get("stable") is not True for row in rows):
            return False
        expected = {
            (puzzle, repetition)
            for puzzle in self.identity.puzzle_ids
            for repetition in range(3)
        }
        observed = {
            (row.get("puzzle_id"), row.get("repetition"))
            for row in rows if row.get("stable") is True
        }
        return expected.issubset(observed)

    def emit_registry_fragment(
        self, anchors: Mapping[int, Mapping[str, object]]
    ) -> dict[str, object]:
        if not anchors:
            raise ValueError("at least one anchor is required")
        bfs = {
            "radius": self.identity.bfs_radius,
            "move_count": self.identity.move_count,
            "hash_bytes": self.identity.hash_bytes,
            "hash_budget_bytes": self.identity.bfs_hash_budget_bytes,
            "cumulative_states": self.identity.bfs_cumulative_states,
        }
        anchor_records = {}
        for power, item in sorted(anchors.items()):
            runtime = item.get("runtime")
            evidence_id = item.get("evidence_id")
            if not isinstance(runtime, Mapping) or not isinstance(evidence_id, str) or not evidence_id:
                raise ValueError(f"invalid anchor {power}")
            anchor_records[str(power)] = {
                "status": "measured" if self._anchor_measured(power) else "unverified",
                "evidence_id": evidence_id,
                "runtime": dict(runtime),
                "bfs": bfs,
            }
        powers = tuple(sorted(anchors))
        fragment: dict[str, object] = {
            "schema_version": 1,
            "profiles": [{
                "hardware": {
                    "gpu_family": self.identity.gpu_family,
                    "vram_mib": self.identity.vram_mib,
                    "sm": self.identity.sm,
                    "world_size": self.identity.world_size,
                },
                "backend": self.identity.backend,
                "model_class": self.identity.model_class,
                "min_beam_power": powers[0],
                "max_beam_power": powers[-1],
                "anchors": anchor_records,
            }],
        }
        _atomic_json(self.run_dir / "registry.fragment.json", fragment)
        _atomic_json(self.run_dir / "profile_candidate.json", fragment["profiles"][0])
        return fragment
