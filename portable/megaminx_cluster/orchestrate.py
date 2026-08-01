"""One-puzzle original/reflected workflow primitives."""

from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Mapping, Sequence

from portable.megaminx_cluster.validate import apply_path


def build_reflected_state(
    central: Sequence[int],
    original_solution: Sequence[str],
    generators: Mapping[str, Sequence[int]],
) -> list[int]:
    if not original_solution:
        raise ValueError("original solution must not be empty")
    return apply_path(central, original_solution, generators)


def plan_steps(mode: str, original_solution: Sequence[str] | None) -> tuple[str, ...]:
    if mode == "off":
        return ("original",)
    if mode == "after":
        return ("original", "reflected")
    if mode == "only":
        if not original_solution:
            raise ValueError("reflect only requires an original solution")
        return ("reflected",)
    raise ValueError(f"unsupported reflection mode: {mode}")


def parse_solution_line(log_text: str, puzzle_id: int) -> list[str]:
    pattern = re.compile(
        rf"^.*puzzle_solved=1 puzzle_id={puzzle_id} .*?solution_length=([0-9]+) solution=([^\s]+)\s*$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(log_text))
    if not matches:
        raise ValueError(f"no valid solution line for puzzle {puzzle_id}")
    length = int(matches[-1].group(1))
    path = matches[-1].group(2).split(".")
    if len(path) != length:
        raise ValueError(f"solution length mismatch for puzzle {puzzle_id}")
    return path


def build_torchrun_command(
    archive_root: Path,
    world_size: int,
    rendezvous_id: str,
    puzzle_id: int,
    depth: int,
    beam: int,
) -> list[str]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", rendezvous_id):
        raise ValueError("unsafe rendezvous id")
    return [
        "python3", "-m", "torch.distributed.run",
        "--nnodes=1", f"--nproc-per-node={world_size}", "--node-rank=0",
        "--rdzv-backend=c10d", "--rdzv-endpoint=127.0.0.1:29500",
        f"--rdzv-id={rendezvous_id}", "--no-python",
        str(archive_root / "bin" / "production_runner"),
        str(puzzle_id), str(depth), str(beam),
    ]


def write_synthetic_puzzle(
    source_csv: Path, target_csv: Path, synthetic_id: int, state: Sequence[int]
) -> None:
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    with source_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = handle.seek(0) or next(csv.reader(handle))
    if not rows:
        raise ValueError("source test CSV contains no puzzles")
    if any(int(row["initial_state_id"]) == synthetic_id for row in rows):
        raise ValueError(f"synthetic puzzle id already exists: {synthetic_id}")
    rows.append({"initial_state_id": str(synthetic_id), "initial_state": ",".join(str(x) for x in state)})
    with target_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)