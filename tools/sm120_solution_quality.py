#!/usr/bin/env python3
"""Primary solution-quality comparison for SM120 precision experiments.

Logit and frontier diagnostics decide what is cheap enough to try.  This module
implements the acceptance metric: replay-valid solutions at an identical beam
and depth limit, with solution length compared to the FP32/FP16 control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_QUALITY_COHORTS = {2**16: 10, 2**18: 30, 2**20: 100}


def _load(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
        state_id = str(record.get("state_id", ""))
        if not state_id or state_id in records:
            raise ValueError(f"state_id must be unique and non-empty at {path}:{line_number}")
        solved = record.get("solved")
        replay_valid = record.get("replay_valid")
        length = record.get("solution_length")
        if not isinstance(solved, bool) or not isinstance(replay_valid, bool):
            raise ValueError(f"solved/replay_valid must be booleans at {path}:{line_number}")
        if replay_valid and not solved:
            raise ValueError(f"replay_valid requires solved=true at {path}:{line_number}")
        if solved and (not isinstance(length, int) or length < 0):
            raise ValueError(f"solved record requires non-negative solution_length at {path}:{line_number}")
        if not solved and length is not None:
            raise ValueError(f"unsolved record requires null solution_length at {path}:{line_number}")
        if not isinstance(record.get("beam"), int) or record["beam"] <= 0:
            raise ValueError(f"beam must be positive at {path}:{line_number}")
        if not isinstance(record.get("depth_limit"), int) or record["depth_limit"] <= 0:
            raise ValueError(f"depth_limit must be positive at {path}:{line_number}")
        records[state_id] = record
    if not records:
        raise ValueError(f"solution run is empty: {path}")
    return records


def compare_solution_runs(baseline_path: Path, candidate_path: Path) -> dict[str, float | int]:
    baseline = _load(Path(baseline_path))
    candidate = _load(Path(candidate_path))
    if set(baseline) != set(candidate):
        raise ValueError("solution runs must contain identical state_id sets")
    state_ids = sorted(baseline)
    for state_id in state_ids:
        left = baseline[state_id]
        right = candidate[state_id]
        if (left["beam"], left["depth_limit"]) != (right["beam"], right["depth_limit"]):
            raise ValueError("solution runs must use identical beam and depth_limit")

    def valid_solved(record: dict[str, Any]) -> bool:
        return bool(record["solved"] and record["replay_valid"])

    baseline_solved = sum(valid_solved(baseline[key]) for key in state_ids)
    candidate_solved = sum(valid_solved(candidate[key]) for key in state_ids)
    joint = [
        key for key in state_ids
        if valid_solved(baseline[key]) and valid_solved(candidate[key])
    ]
    deltas = [
        int(candidate[key]["solution_length"]) - int(baseline[key]["solution_length"])
        for key in joint
    ]
    count = len(state_ids)
    return {
        "state_count": count,
        "beam": int(baseline[state_ids[0]]["beam"]),
        "depth_limit": int(baseline[state_ids[0]]["depth_limit"]),
        "baseline_replay_valid_solved_rate": baseline_solved / count,
        "candidate_replay_valid_solved_rate": candidate_solved / count,
        "candidate_solved_rate_delta": (candidate_solved - baseline_solved) / count,
        "joint_solved_count": len(joint),
        "same_length_rate_on_joint_solved": (
            sum(delta == 0 for delta in deltas) / len(deltas) if deltas else 0.0
        ),
        "mean_length_delta_on_joint_solved": (
            sum(deltas) / len(deltas) if deltas else 0.0
        ),
        "max_length_regression_on_joint_solved": max(deltas) if deltas else 0,
    }


def evaluate_quality_cohorts(cohorts: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the plan's solution-based FP16 -> NVFP4 promotion gate."""
    by_beam: dict[int, dict[str, Any]] = {}
    for cohort in cohorts:
        beam = cohort.get("beam")
        if not isinstance(beam, int) or beam in by_beam:
            raise ValueError("quality gate cohorts require unique integer beam values")
        by_beam[beam] = cohort
    if set(by_beam) != set(REQUIRED_QUALITY_COHORTS):
        raise ValueError("quality gate requires exact cohorts 2^16/10, 2^18/30, 2^20/100")

    reports: list[dict[str, Any]] = []
    reasons: list[str] = []
    for beam, expected_count in REQUIRED_QUALITY_COHORTS.items():
        cohort = by_beam[beam]
        metrics = compare_solution_runs(
            Path(cohort["baseline"]), Path(cohort["candidate"])
        )
        if metrics["beam"] != beam or metrics["state_count"] != expected_count:
            raise ValueError(
                f"beam {beam} cohort must contain exactly {expected_count} states"
            )
        if metrics["candidate_replay_valid_solved_rate"] < metrics["baseline_replay_valid_solved_rate"]:
            reasons.append("solved_count_regression")
        if metrics["max_length_regression_on_joint_solved"] > 0:
            reasons.append("solution_length_regression")
        reports.append(metrics)
    reasons = sorted(set(reasons))
    return {
        "schema_version": 1,
        "status": "accepted" if not reasons else "rejected",
        "reasons": reasons,
        "total_state_count": sum(int(item["state_count"]) for item in reports),
        "cohorts": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_solution_runs(args.baseline, args.candidate)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
