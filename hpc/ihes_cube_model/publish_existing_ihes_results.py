#!/usr/bin/env python3
"""Publish existing IHES bucket TSV logs into the public results repository."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

FIELDS = [
    "puzzle_id",
    "variant",
    "source_index",
    "known_length",
    "found_length",
    "delta",
    "run_puzzle_id",
    "depth_index",
    "found_depth",
    "owner_rank",
    "solution_path",
    "source_solution",
]
INDEX_FIELDS = [
    "puzzle_id",
    "known_length",
    "best_length",
    "delta",
    "records",
    "unique_solutions",
    "variants",
    "source_files",
]


def read_rows(paths: list[Path]) -> dict[str, list[dict[str, str]]]:
    by_puzzle: dict[str, list[dict[str, str]]] = {}
    seen_records: set[tuple[str, str, str, str, str, str, str]] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                puzzle_id = row.get("puzzle_id", "").strip()
                solution = row.get("solution_path", "").strip()
                if not puzzle_id or not solution:
                    continue
                normalized = {field: row.get(field, "") for field in FIELDS}
                normalized["source_file"] = path.name
                key = (
                    normalized["puzzle_id"],
                    normalized["variant"],
                    normalized["source_index"],
                    normalized["found_length"],
                    normalized["solution_path"],
                    normalized["source_solution"],
                    normalized["source_file"],
                )
                if key in seen_records:
                    continue
                seen_records.add(key)
                by_puzzle.setdefault(puzzle_id, []).append(normalized)
    return by_puzzle


def write_table(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def int_or_default(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def publish_tables(results_dir: Path, by_puzzle: dict[str, list[dict[str, str]]]) -> tuple[int, int, int]:
    base = results_dir / "data" / "ihes_cube"
    puzzles_dir = base / "puzzles"
    index_rows: list[dict[str, str]] = []
    improved = 0
    total_records = 0

    for puzzle_id in sorted(by_puzzle, key=lambda x: int_or_default(x, 10**9)):
        rows = sorted(
            by_puzzle[puzzle_id],
            key=lambda row: (
                int_or_default(row.get("found_length", "999999"), 999999),
                row.get("variant", ""),
                int_or_default(row.get("source_index", "0")),
                row.get("solution_path", ""),
            ),
        )
        puzzle_dir = puzzles_dir / f"p{int(puzzle_id):04d}"
        write_table(puzzle_dir / "solutions.tsv", rows, FIELDS + ["source_file"])

        unique_by_path: dict[str, dict[str, str]] = {}
        for row in rows:
            path = row.get("solution_path", "")
            if path and path not in unique_by_path:
                unique_by_path[path] = row
        unique_rows = list(unique_by_path.values())
        write_table(puzzle_dir / "unique_solutions.tsv", unique_rows, FIELDS + ["source_file"])

        best = min(rows, key=lambda row: int_or_default(row.get("found_length", "999999"), 999999))
        known_length = int_or_default(best.get("known_length", "0"))
        best_length = int_or_default(best.get("found_length", "-1"), -1)
        delta = best_length - known_length if best_length >= 0 else 0
        if delta < 0:
            improved += 1
        variants = ",".join(sorted({row.get("variant", "") for row in rows if row.get("variant", "")}))
        source_files = ",".join(sorted({row.get("source_file", "") for row in rows if row.get("source_file", "")}))
        metadata = {
            "puzzle_id": puzzle_id,
            "known_length": str(known_length),
            "best_length": str(best_length),
            "delta": str(delta),
            "records": str(len(rows)),
            "unique_solutions": str(len(unique_rows)),
            "variants": variants,
            "source_files": source_files,
        }
        with (puzzle_dir / "metadata.env").open("w", encoding="utf-8") as out:
            for key, value in metadata.items():
                out.write(f"{key}={value}\n")

        length_counts: dict[str, int] = {}
        variant_counts: dict[str, int] = {}
        for row in rows:
            length_counts[row.get("found_length", "")] = length_counts.get(row.get("found_length", ""), 0) + 1
            variant_counts[row.get("variant", "")] = variant_counts.get(row.get("variant", ""), 0) + 1
        summary = [
            f"# IHES Cube Puzzle {puzzle_id}",
            "",
            f"known_length: {known_length}",
            f"best_length: {best_length}",
            f"delta: {delta}",
            f"records: {len(rows)}",
            f"unique_solutions: {len(unique_rows)}",
            f"source_files: {source_files}",
            "",
            "## Variant Counts",
            "",
        ]
        for key in sorted(variant_counts):
            summary.append(f"- {key}: {variant_counts[key]}")
        summary += ["", "## Length Counts", ""]
        for key in sorted(length_counts, key=lambda x: int_or_default(x, -1)):
            summary.append(f"- {key}: {length_counts[key]}")
        (puzzle_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

        index_rows.append(metadata)
        total_records += len(rows)

    write_table(base / "index.tsv", index_rows, INDEX_FIELDS)
    write_table(
        base / "improvements.tsv",
        [row for row in index_rows if int_or_default(row.get("delta", "0")) < 0],
        INDEX_FIELDS,
    )
    return len(index_rows), total_records, improved


def git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, text=True, check=check)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--glob", default="solve_bucket_fresh_*.tsv")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    paths = sorted(args.logs_dir.glob(args.glob))
    if not paths:
        raise SystemExit(f"no TSV files matched {args.logs_dir / args.glob}")
    by_puzzle = read_rows(paths)
    puzzles, records, improved = publish_tables(args.results_dir, by_puzzle)
    print(f"published_existing_ihes_results puzzles={puzzles} records={records} improved={improved}")

    if args.push:
        git(["config", "user.name", "beam-results-bot"], args.results_dir)
        git(["config", "user.email", "beam-results-bot@users.noreply.github.com"], args.results_dir)
        git(["pull", "--rebase", "origin", "main"], args.results_dir, check=False)
        git(["add", "data/ihes_cube", "README.md"], args.results_dir)
        diff = git(["diff", "--cached", "--quiet"], args.results_dir, check=False)
        if diff.returncode == 0:
            print("published_existing_ihes_results_no_changes=1")
        else:
            git(["commit", "-m", "Backfill existing IHES bucket results"], args.results_dir)
            git(["push", "origin", "main"], args.results_dir)
            print("published_existing_ihes_results_pushed=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
