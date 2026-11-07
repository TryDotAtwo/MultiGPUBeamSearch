#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def split_path(path: str) -> list[str]:
    path = path.strip()
    return [] if not path else path.split(".")


def load_test_csv(path: Path) -> dict[int, list[int]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        id_field = "initial_state_id" if "initial_state_id" in (reader.fieldnames or []) else (reader.fieldnames or [""])[0]
        state_field = "initial_state" if "initial_state" in (reader.fieldnames or []) else (reader.fieldnames or ["", ""])[-1]
        return {
            int(row[id_field]): [int(x) for x in row[state_field].split(",")]
            for row in reader
            if row.get(id_field) and row.get(state_field)
        }


def apply_move(state: list[int], generators: dict[str, list[int]], move: str) -> list[int]:
    return [state[i] for i in generators[move]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solutions", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--puzzle-info", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--length", type=int, default=24)
    args = parser.parse_args()

    solutions_obj = json.loads(args.solutions.read_text())
    info = json.loads(args.puzzle_info.read_text())
    generators = info["generators"]
    initial_states = load_test_csv(args.test_csv)

    by_puzzle: dict[int, list[str]] = defaultdict(list)
    for key, paths in solutions_obj.items():
        puzzle_id = int(key)
        for path in paths:
            moves = split_path(path)
            if len(moves) == args.length:
                by_puzzle[puzzle_id].append(path)

    rows = []
    for puzzle_id in sorted(by_puzzle.keys(), reverse=True):
        if puzzle_id not in initial_states:
            continue
        paths = by_puzzle[puzzle_id]
        first_counts = Counter(split_path(path)[0] for path in paths if split_path(path))
        if not first_counts:
            continue
        rare_count = min(first_counts.values())
        rare_moves = sorted(move for move, count in first_counts.items() if count == rare_count)
        first_move = rare_moves[0]
        synthetic_id = 9_300_000 + puzzle_id
        synthetic_state = apply_move(initial_states[puzzle_id], generators, first_move)
        rows.append({
            "plan_index": len(rows),
            "puzzle_id": puzzle_id,
            "synthetic_id": synthetic_id,
            "original_length": args.length,
            "prefix_move": first_move,
            "prefix_count": first_counts[first_move],
            "length24_solution_count": len(paths),
            "synthetic_state": ",".join(str(x) for x in synthetic_state),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        fieldnames = [
            "plan_index",
            "puzzle_id",
            "synthetic_id",
            "original_length",
            "prefix_move",
            "prefix_count",
            "length24_solution_count",
            "synthetic_state",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"prefix_solve_plan_rows={len(rows)}")
    print(f"prefix_solve_plan={args.out}")


if __name__ == "__main__":
    main()
