from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.sm120_solution_quality import compare_solution_runs, evaluate_quality_cohorts


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_compare_solution_runs_uses_replay_valid_same_length_as_primary_metric(tmp_path: Path) -> None:
    baseline = tmp_path / "fp32.jsonl"
    candidate = tmp_path / "nvfp4.jsonl"
    common = {"beam": 65536, "depth_limit": 100}
    _write(baseline, [
        {**common, "state_id": "s0", "solved": True, "replay_valid": True, "solution_length": 8},
        {**common, "state_id": "s1", "solved": True, "replay_valid": True, "solution_length": 10},
        {**common, "state_id": "s2", "solved": False, "replay_valid": False, "solution_length": None},
    ])
    _write(candidate, [
        {**common, "state_id": "s0", "solved": True, "replay_valid": True, "solution_length": 8},
        {**common, "state_id": "s1", "solved": True, "replay_valid": True, "solution_length": 11},
        {**common, "state_id": "s2", "solved": True, "replay_valid": True, "solution_length": 12},
    ])
    metrics = compare_solution_runs(baseline, candidate)
    assert metrics["state_count"] == 3
    assert metrics["candidate_replay_valid_solved_rate"] == pytest.approx(1.0)
    assert metrics["same_length_rate_on_joint_solved"] == pytest.approx(0.5)
    assert metrics["mean_length_delta_on_joint_solved"] == pytest.approx(0.5)
    assert metrics["candidate_solved_rate_delta"] == pytest.approx(1 / 3)


def test_compare_solution_runs_fails_closed_on_workload_or_state_mismatch(tmp_path: Path) -> None:
    baseline = tmp_path / "fp32.jsonl"
    candidate = tmp_path / "nvfp4.jsonl"
    _write(baseline, [{"state_id": "s0", "beam": 1024, "depth_limit": 20, "solved": False, "replay_valid": False, "solution_length": None}])
    _write(candidate, [{"state_id": "s0", "beam": 2048, "depth_limit": 20, "solved": False, "replay_valid": False, "solution_length": None}])
    with pytest.raises(ValueError, match="identical beam and depth_limit"):
        compare_solution_runs(baseline, candidate)

    _write(candidate, [{"state_id": "other", "beam": 1024, "depth_limit": 20, "solved": False, "replay_valid": False, "solution_length": None}])
    with pytest.raises(ValueError, match="identical state_id sets"):
        compare_solution_runs(baseline, candidate)


def test_quality_cohort_gate_requires_exact_plan_and_accepts_equal_solutions(tmp_path: Path) -> None:
    cohorts: list[dict] = []
    for beam, count in ((2**16, 10), (2**18, 30), (2**20, 100)):
        baseline = tmp_path / f"fp16-{beam}.jsonl"
        candidate = tmp_path / f"nvfp4-{beam}.jsonl"
        rows = [
            {"state_id": f"s{index}", "beam": beam, "depth_limit": 100,
             "solved": True, "replay_valid": True, "solution_length": 8 + index % 3}
            for index in range(count)
        ]
        _write(baseline, rows)
        _write(candidate, rows)
        cohorts.append({"beam": beam, "baseline": baseline, "candidate": candidate})
    report = evaluate_quality_cohorts(cohorts)
    assert report["status"] == "accepted"
    assert report["total_state_count"] == 140


def test_quality_cohort_gate_rejects_length_or_solve_regression(tmp_path: Path) -> None:
    cohorts: list[dict] = []
    for beam, count in ((2**16, 10), (2**18, 30), (2**20, 100)):
        baseline = tmp_path / f"fp16-{beam}.jsonl"
        candidate = tmp_path / f"nvfp4-{beam}.jsonl"
        rows = [
            {"state_id": f"s{index}", "beam": beam, "depth_limit": 100,
             "solved": True, "replay_valid": True, "solution_length": 8}
            for index in range(count)
        ]
        changed = [dict(row) for row in rows]
        if beam == 2**18:
            changed[0]["solution_length"] = 9
        _write(baseline, rows)
        _write(candidate, changed)
        cohorts.append({"beam": beam, "baseline": baseline, "candidate": candidate})
    report = evaluate_quality_cohorts(cohorts)
    assert report["status"] == "rejected"
    assert "solution_length_regression" in report["reasons"]


def test_quality_cohort_gate_rejects_noncanonical_cohort() -> None:
    with pytest.raises(ValueError, match="exact cohorts"):
        evaluate_quality_cohorts([])
