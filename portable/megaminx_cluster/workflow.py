"""Execute exactly one original/reflected Megaminx workflow."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys

from portable.megaminx_cluster.orchestrate import (
    build_reflected_state,
    build_torchrun_command,
    parse_solution_line,
    plan_steps,
    write_synthetic_puzzle,
)
from portable.megaminx_cluster.validate import invert_path, validate_solution


def _load_puzzle(test_csv: Path, puzzle_id: int) -> list[int]:
    with test_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["initial_state_id"]) == puzzle_id:
                return [int(value) for value in row["initial_state"].split(",")]
    raise ValueError(f"puzzle id not found: {puzzle_id}")


def _load_definition(path: Path) -> tuple[list[int], dict[str, list[int]]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return list(data["central_state"]), {name: list(perm) for name, perm in data["generators"].items()}


def _read_supplied(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("original solution file is empty")
    return text.split(".")


def _clear_history(path: Path, run_dir: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != run_dir.resolve():
        raise ValueError(f"refusing to clear history outside run directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir()


def _run_once(
    archive_root: Path,
    run_dir: Path,
    world_size: int,
    job_id: str,
    tag: str,
    puzzle_id: int,
    depth: int,
    beam: int,
    test_csv: Path,
) -> list[str]:
    command = build_torchrun_command(
        archive_root, world_size, f"megaminx-{job_id}-{tag}", puzzle_id, depth, beam
    )
    env = dict(os.environ)
    env["BEAM_TEST_CSV"] = str(test_csv)
    env["BEAM_PUZZLE_INFO_JSON"] = str(archive_root / "data" / "puzzle_info.json")
    env["BEAM_WEIGHT_DIR"] = str(archive_root / "weights")
    env["BEAM_RANK_LOG_DIR"] = str(run_dir / "logs" / f"ranks-{tag}")
    env["BEAM_NCCL_ID_FILE"] = str(run_dir / f"nccl-{job_id}-{tag}.bin")
    Path(env["BEAM_RANK_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    log_path = run_dir / "logs" / f"{tag}.log"
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"torchrun {tag} failed rc={result.returncode}; log={log_path}")
    return parse_solution_line(result.stdout + result.stderr, puzzle_id)


def _run_benchmark(
    archive_root: Path,
    run_dir: Path,
    world_size: int,
    job_id: str,
    puzzle_id: int,
    depth: int,
    beam: int,
    test_csv: Path,
) -> dict[str, object]:
    command = build_torchrun_command(
        archive_root, world_size, f"megaminx-{job_id}-benchmark", puzzle_id, depth, beam
    )
    env = dict(os.environ)
    env["BEAM_TEST_CSV"] = str(test_csv)
    env["BEAM_PUZZLE_INFO_JSON"] = str(archive_root / "data" / "puzzle_info.json")
    env["BEAM_WEIGHT_DIR"] = str(archive_root / "weights")
    env["BEAM_RANK_LOG_DIR"] = str(run_dir / "logs" / "ranks-benchmark")
    env["BEAM_NCCL_ID_FILE"] = str(run_dir / f"nccl-{job_id}-benchmark.bin")
    env["BEAM_AUTOTUNE_DEPTH_METRICS"] = "1"
    Path(env["BEAM_RANK_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    text = result.stdout + result.stderr
    log_path = run_dir / "logs" / "benchmark.log"
    log_path.write_text(text, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"torchrun benchmark failed rc={result.returncode}; log={log_path}")
    pattern = re.compile(
        r"autotune_depth_done=([0-9]+) depth_sec=([0-9.eE+-]+) next_frontier_size=([0-9]+)"
    )
    rows = [
        (int(match.group(1)), float(match.group(2)), int(match.group(3)))
        for match in pattern.finditer(text)
        if int(match.group(1)) == depth - 1
    ]
    if not rows:
        raise RuntimeError(f"benchmark depth {depth} did not complete on rank 0; log={log_path}")
    local_frontier = min(row[2] for row in rows)
    frontier = local_frontier * world_size
    if frontier < beam:
        raise RuntimeError(
            f"benchmark depth {depth} frontier is not full: global_frontier={frontier} beam={beam}; log={log_path}"
        )
    depth_sec = max(row[1] for row in rows)
    if depth_sec <= 0:
        raise RuntimeError("benchmark depth time must be positive")
    return {
        "benchmark_depth": depth,
        "depth_sec": depth_sec,
        "frontier_size": frontier,
        "local_frontier_size": local_frontier,
        "rank_samples": len(rows),
        "frontier_full": True,
    }

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--puzzle", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--beam", type=int, required=True)
    parser.add_argument("--reflect", choices=("off", "after", "only"), required=True)
    parser.add_argument("--original-solution", type=Path)
    parser.add_argument("--benchmark-depth", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = args.archive_root.resolve()
        run_dir = args.run_dir.resolve()
        source_csv = root / "data" / "test.csv"
        run_csv = run_dir / "data" / "test.csv"
        run_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_csv, run_csv)
        central, generators = _load_definition(root / "data" / "puzzle_info.json")
        initial = _load_puzzle(run_csv, args.puzzle)
        supplied = _read_supplied(args.original_solution)
        steps = plan_steps(args.reflect, supplied)
        original = supplied
        if args.benchmark_depth:
            metrics = _run_benchmark(
                root, run_dir, args.world_size, args.job_id,
                args.puzzle, args.depth, args.beam, run_csv,
            )
            (run_dir / "benchmark_metrics.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return 0
        results: list[dict[str, object]] = []
        if "original" in steps:
            original = _run_once(root, run_dir, args.world_size, args.job_id, "original", args.puzzle, args.depth, args.beam, run_csv)
            if not validate_solution(initial, central, original, generators):
                raise ValueError("CPU replay rejected original solution")
            results.append({"search": "original", "path": original, "valid": True})
        if "reflected" in steps:
            if original is None or not validate_solution(initial, central, original, generators):
                raise ValueError("original solution is invalid for reflected search")
            synthetic_id = 9_000_000 + args.puzzle
            reflected_state = build_reflected_state(central, original, generators)
            write_synthetic_puzzle(source_csv, run_csv, synthetic_id, reflected_state)
            _clear_history(run_dir / "history", run_dir)
            reflected = _run_once(root, run_dir, args.world_size, args.job_id, "reflected", synthetic_id, args.depth, args.beam, run_csv)
            candidate = invert_path(reflected)
            if not validate_solution(initial, central, candidate, generators):
                raise ValueError("CPU replay rejected reflected candidate")
            results.append({"search": "reflected", "searched_path": reflected, "path": candidate, "valid": True})
        (run_dir / "validated_results.json").write_text(
            json.dumps({"puzzle_id": args.puzzle, "reflect": args.reflect, "results": results}, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"workflow_failed={exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
