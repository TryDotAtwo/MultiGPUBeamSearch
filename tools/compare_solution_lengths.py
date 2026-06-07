#!/usr/bin/env python3
"""Compare puzzle solution lengths without validating moves."""

from __future__ import annotations

import argparse
import csv
import glob
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Solution:
    source: str
    text: str

    @property
    def length(self) -> int:
        if not self.text:
            return 0
        return len([move for move in self.text.split(".") if move])


def read_submission_csv(path: Path, source: str) -> dict[int, Solution]:
    rows: dict[int, Solution] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return rows
        id_field = "initial_state_id" if "initial_state_id" in reader.fieldnames else reader.fieldnames[0]
        solution_field = "path" if "path" in reader.fieldnames else reader.fieldnames[-1]
        for row in reader:
            raw_id = (row.get(id_field) or "").strip()
            solution = (row.get(solution_field) or "").strip()
            if not raw_id:
                continue
            rows[int(raw_id)] = Solution(source, solution)
    return rows


def read_solve_reflect_logs(log_dir: Path) -> dict[int, Solution]:
    rows: dict[int, Solution] = {}
    for path_text in sorted(glob.glob(str(log_dir / "slurm-*.out"))):
        path = Path(path_text)
        text = path.read_text(encoding="utf-8", errors="ignore")
        puzzle_ids = re.findall(r"original_puzzle_id=(\d+)", text)
        if not puzzle_ids:
            continue
        puzzle_id = int(puzzle_ids[-1])

        candidates: list[Solution] = []
        original_solutions = re.findall(r"original_solution=([^\s]+)", text)
        if original_solutions:
            candidates.append(Solution(f"{path.name}:original", original_solutions[-1]))

        reflected_ok = re.findall(r"candidate_solution_solves_original=(\d+)", text)
        reflected_solutions = re.findall(r"candidate_solution_for_original=([^\s]+)", text)
        if reflected_solutions and reflected_ok and reflected_ok[-1] == "1":
            candidates.append(Solution(f"{path.name}:reflected", reflected_solutions[-1]))

        if candidates:
            rows[puzzle_id] = min(candidates, key=lambda item: item.length)
    return rows


def format_markdown(
    baseline: dict[int, Solution],
    candidate: dict[int, Solution],
    start: int | None,
    end: int | None,
) -> str:
    ids = sorted(set(baseline) | set(candidate))
    if start is not None:
        ids = [item for item in ids if item >= start]
    if end is not None:
        ids = [item for item in ids if item <= end]

    lines = [
        "| Семпл | Было | Стало | Дельта | Источник | Решение |",
        "|---:|---:|---:|---:|---|---|",
    ]
    total_delta = 0
    improved = 0
    worsened = 0
    equal = 0

    for puzzle_id in ids:
        old = baseline.get(puzzle_id)
        new = candidate.get(puzzle_id)
        old_len = old.length if old else None
        new_len = new.length if new else None
        delta = None if old_len is None or new_len is None else new_len - old_len
        if delta is not None:
            total_delta += delta
            if delta < 0:
                improved += 1
            elif delta > 0:
                worsened += 1
            else:
                equal += 1
        lines.append(
            "| {puzzle_id} | {old_len} | {new_len} | {delta} | {source} | `{solution}` |".format(
                puzzle_id=puzzle_id,
                old_len="" if old_len is None else old_len,
                new_len="" if new_len is None else new_len,
                delta="" if delta is None else delta,
                source="" if new is None else new.source,
                solution="" if new is None else new.text,
            )
        )

    lines.append("")
    lines.append(
        f"Итого: delta={total_delta}, improved={improved}, equal={equal}, worsened={worsened}"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare solution lengths only; no puzzle validation is performed."
    )
    parser.add_argument("--baseline-csv", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate-csv", type=Path)
    source.add_argument("--logs-dir", type=Path)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()

    baseline = read_submission_csv(args.baseline_csv, "baseline")
    if args.candidate_csv:
        candidate = read_submission_csv(args.candidate_csv, args.candidate_csv.name)
    else:
        candidate = read_solve_reflect_logs(args.logs_dir)

    print(format_markdown(baseline, candidate, args.start, args.end))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
