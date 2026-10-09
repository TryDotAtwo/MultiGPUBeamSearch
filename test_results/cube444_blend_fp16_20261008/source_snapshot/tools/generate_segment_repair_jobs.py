#!/usr/bin/env python3
"""Generate synthetic start/target jobs for segment and suffix repair.

The solver still runs the normal production runner. This helper only prepares
states and splice metadata:

  start_state = state after prefix i
  target_state = state after prefix j

Run production_runner for start_state with BEAM_TARGET_STATE_TEXT=target_state.
If the found segment is shorter than j-i, splice:

  prefix_to_i + found_segment + suffix_from_j
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


State = list[int]


@dataclass(frozen=True)
class RepairJob:
    repair_id: int
    solution_job_id: int
    puzzle_id: int
    solution_index: int
    search_depth: int
    window: int
    start_step: int
    target_step: int
    old_segment_len: int
    original_length: int
    prefix_path: str
    old_segment_path: str
    suffix_path: str
    target_state: State


@dataclass(frozen=True)
class SolutionJob:
    solution_job_id: int
    puzzle_id: int
    solution_index: int
    original_length: int
    original_path: str


def split_path(path: str) -> list[str]:
    path = path.strip()
    if not path:
        return []
    return [item for item in path.split(".") if item]


def path_text(moves: list[str]) -> str:
    return ".".join(moves)


def state_text(state: State) -> str:
    return ",".join(str(item) for item in state)


def load_test_csv(path: Path) -> dict[int, State]:
    rows: dict[int, State] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        id_field = "initial_state_id" if "initial_state_id" in (reader.fieldnames or []) else (reader.fieldnames or [""])[0]
        state_field = "initial_state" if "initial_state" in (reader.fieldnames or []) else (reader.fieldnames or ["", ""])[-1]
        for row in reader:
            raw_id = (row.get(id_field) or "").strip()
            raw_state = (row.get(state_field) or "").strip()
            if raw_id and raw_state:
                rows[int(raw_id)] = [int(item) for item in raw_state.split(",") if item != ""]
    return rows


def load_puzzle_info(path: Path) -> tuple[State, dict[str, list[int]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    central = [int(item) for item in data["central_state"]]
    raw_generators = data.get("generators", data.get("actions"))
    if not isinstance(raw_generators, dict):
        raise ValueError("puzzle_info.json must contain object field 'generators' or 'actions'")
    generators = {str(name): [int(item) for item in perm] for name, perm in raw_generators.items()}
    return central, generators


def apply_move(state: State, perm: list[int]) -> State:
    return [state[idx] for idx in perm]


def states_along_path(initial: State, moves: list[str], generators: dict[str, list[int]]) -> list[State]:
    states = [initial]
    current = initial
    for move in moves:
        if move not in generators:
            raise ValueError(f"unknown move token {move!r}")
        current = apply_move(current, generators[move])
        states.append(current)
    return states


def load_solutions(path: Path) -> dict[int, list[str]]:
    text = path.read_text(encoding="utf-8-sig")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        raw = json.loads(text)
        return {int(key): [str(item) for item in value] for key, value in raw.items()}

    rows: dict[int, list[str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        id_field = "initial_state_id" if "initial_state_id" in (reader.fieldnames or []) else (reader.fieldnames or [""])[0]
        path_field = "path" if "path" in (reader.fieldnames or []) else (reader.fieldnames or ["", ""])[-1]
        for row in reader:
            raw_id = (row.get(id_field) or "").strip()
            solution = (row.get(path_field) or "").strip()
            if raw_id:
                rows.setdefault(int(raw_id), []).append(solution)
    return rows


def make_jobs(
    initial_states: dict[int, State],
    generators: dict[str, list[int]],
    solutions: dict[int, list[str]],
    k1_radius: int,
    search_depths: list[int],
    block_step: bool,
    only_puzzle_ids: set[int] | None,
    include_suffix: bool,
) -> tuple[list[SolutionJob], list[RepairJob], dict[int, State]]:
    solution_jobs: list[SolutionJob] = []
    jobs: list[RepairJob] = []
    synthetic_states: dict[int, State] = {}
    repair_id = 9_100_000
    solution_job_id = 9_200_000

    for puzzle_id in sorted(solutions):
        if only_puzzle_ids is not None and puzzle_id not in only_puzzle_ids:
            continue
        if puzzle_id not in initial_states:
            raise ValueError(f"puzzle {puzzle_id} is missing from test csv")
        for solution_index, solution_text in enumerate(solutions[puzzle_id]):
            moves = split_path(solution_text)
            if not moves:
                continue
            solution_job_id += 1
            solution_jobs.append(
                SolutionJob(
                    solution_job_id=solution_job_id,
                    puzzle_id=puzzle_id,
                    solution_index=solution_index,
                    original_length=len(moves),
                    original_path=path_text(moves),
                )
            )
            states = states_along_path(initial_states[puzzle_id], moves, generators)
            for search_depth in search_depths:
                window = k1_radius + search_depth
                start_step = 0
                while start_step < len(moves):
                    target_step = min(len(moves), start_step + window)
                    if target_step <= start_step:
                        break
                    if include_suffix or target_step != len(moves):
                        repair_id += 1
                        synthetic_states[repair_id] = states[start_step]
                        jobs.append(
                            RepairJob(
                                repair_id=repair_id,
                                solution_job_id=solution_job_id,
                                puzzle_id=puzzle_id,
                                solution_index=solution_index,
                                search_depth=search_depth,
                                window=window,
                                start_step=start_step,
                                target_step=target_step,
                                old_segment_len=target_step - start_step,
                                original_length=len(moves),
                                prefix_path=path_text(moves[:start_step]),
                                old_segment_path=path_text(moves[start_step:target_step]),
                                suffix_path=path_text(moves[target_step:]),
                                target_state=states[target_step],
                            )
                        )
                    if block_step:
                        start_step = target_step
                    else:
                        start_step += 1
    return solution_jobs, jobs, synthetic_states


def write_test_csv(path: Path, states: dict[int, State]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["initial_state_id", "initial_state"])
        for puzzle_id in sorted(states):
            writer.writerow([puzzle_id, state_text(states[puzzle_id])])


def write_solution_jobs_csv(path: Path, solution_jobs: list[SolutionJob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "solution_job_id",
            "puzzle_id",
            "solution_index",
            "original_length",
            "original_path",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for job in solution_jobs:
            writer.writerow(
                {
                    "solution_job_id": job.solution_job_id,
                    "puzzle_id": job.puzzle_id,
                    "solution_index": job.solution_index,
                    "original_length": job.original_length,
                    "original_path": job.original_path,
                }
            )


def write_jobs_csv(
    path: Path,
    jobs: list[RepairJob],
    k1_radius: int,
    beam_width: int | None,
    move_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "repair_id",
            "solution_job_id",
            "puzzle_id",
            "solution_index",
            "search_depth",
            "window",
            "start_step",
            "target_step",
            "old_segment_len",
            "original_length",
            "k1_radius",
            "beam_width",
            "move_count",
            "prefix_path",
            "old_segment_path",
            "suffix_path",
            "target_state",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "repair_id": job.repair_id,
                    "solution_job_id": job.solution_job_id,
                    "puzzle_id": job.puzzle_id,
                    "solution_index": job.solution_index,
                    "search_depth": job.search_depth,
                    "window": job.window,
                    "start_step": job.start_step,
                    "target_step": job.target_step,
                    "old_segment_len": job.old_segment_len,
                    "original_length": job.original_length,
                    "k1_radius": k1_radius,
                    "beam_width": "" if beam_width is None else beam_width,
                    "move_count": move_count,
                    "prefix_path": job.prefix_path,
                    "old_segment_path": job.old_segment_path,
                    "suffix_path": job.suffix_path,
                    "target_state": state_text(job.target_state),
                }
            )


def parse_ids(text: str | None) -> set[int] | None:
    if not text:
        return None
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            begin, end = part.split("-", 1)
            out.update(range(int(begin), int(end) + 1))
        else:
            out.add(int(part))
    return out


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            begin, end = part.split("-", 1)
            values.extend(range(int(begin), int(end) + 1))
        else:
            values.append(int(part))
    return sorted(set(values))


def full_frontier_depth(move_count: int, beam_width: int) -> int:
    if move_count <= 1:
        raise ValueError("move_count must be > 1")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    depth = 0
    frontier = 1
    while frontier <= beam_width // move_count:
        frontier *= move_count
        depth += 1
    return depth


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzle-info", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--solutions", required=True, type=Path)
    parser.add_argument("--out-test-csv", required=True, type=Path)
    parser.add_argument("--out-jobs-csv", required=True, type=Path)
    parser.add_argument("--out-solution-jobs-csv", required=True, type=Path)
    parser.add_argument("--k1-radius", type=int, default=4)
    parser.add_argument(
        "--search-depth",
        type=int,
        help="Explicit runner depth before K1 target hit; overrides --beam-width auto depth.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        help="Auto-select search depth as max d where move_count**d <= beam_width.",
    )
    parser.add_argument(
        "--search-depths",
        help="Comma/range list of repair depths, for diversity, e.g. 1-7. Overrides --search-depth/--beam-width.",
    )
    parser.add_argument(
        "--sliding",
        action="store_true",
        help="Use every start step. Default is block stepping by window size.",
    )
    parser.add_argument("--puzzle-ids", help="Comma/range list, for example 312,668,748-778")
    parser.add_argument("--no-suffix", action="store_true", help="Skip windows ending at the final center state")
    args = parser.parse_args()

    if args.k1_radius < 0:
        raise ValueError("k1 radius must be non-negative")

    _, generators = load_puzzle_info(args.puzzle_info)
    move_count = len(generators)
    if args.search_depths:
        search_depths = parse_int_list(args.search_depths)
    elif args.search_depth is None:
        if args.beam_width is None:
            raise ValueError("set --beam-width for auto BFS-like depth or pass --search-depth explicitly")
        search_depths = [full_frontier_depth(move_count, args.beam_width)]
    else:
        search_depths = [args.search_depth]
    if not search_depths or min(search_depths) < 0:
        raise ValueError("search depth must be non-negative")

    initial_states = load_test_csv(args.test_csv)
    solutions = load_solutions(args.solutions)
    solution_jobs, jobs, synthetic_states = make_jobs(
        initial_states=initial_states,
        generators=generators,
        solutions=solutions,
        k1_radius=args.k1_radius,
        search_depths=search_depths,
        block_step=not args.sliding,
        only_puzzle_ids=parse_ids(args.puzzle_ids),
        include_suffix=not args.no_suffix,
    )
    write_test_csv(args.out_test_csv, synthetic_states)
    write_solution_jobs_csv(args.out_solution_jobs_csv, solution_jobs)
    write_jobs_csv(args.out_jobs_csv, jobs, args.k1_radius, args.beam_width, move_count)
    print(f"solution_jobs={len(solution_jobs)}")
    print(f"segment_jobs={len(jobs)}")
    print(f"move_count={move_count}")
    print(f"search_depths={','.join(str(item) for item in search_depths)}")
    print(f"windows={','.join(str(args.k1_radius + item) for item in search_depths)}")
    print(f"out_test_csv={args.out_test_csv}")
    print(f"out_solution_jobs_csv={args.out_solution_jobs_csv}")
    print(f"out_jobs_csv={args.out_jobs_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
