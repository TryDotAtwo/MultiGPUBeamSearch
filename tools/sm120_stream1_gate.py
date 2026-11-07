"""Deterministic promotion gate for the canonical SM120 Stream1 benchmark."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Mapping, Sequence

DENSE_EQUIVALENT_FLOPS_PER_PARENT = 371_933_184
CANONICAL_BATCH = 896
CANONICAL_CONCURRENCY = 1
MIN_ALTERNATING_TRIALS = 7
MEDIAN_LIMIT_MS = 0.333252
P95_LIMIT_MS = 0.350
STAGE_SUM_LIMIT_MS = 0.300


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def evaluate_stream1_gate(
    *, trials: Sequence[Mapping[str, object]], stage_ms: Mapping[str, float],
    batch: int, concurrency: int,
) -> dict[str, object]:
    if batch != CANONICAL_BATCH or concurrency != CANONICAL_CONCURRENCY:
        raise ValueError("promotion gate requires canonical B=896, concurrency=1")
    if len(trials) < MIN_ALTERNATING_TRIALS:
        raise ValueError("promotion gate requires at least seven alternating trials")
    candidate: list[float] = []
    baseline: list[float] = []
    expected = ("baseline", "candidate")
    for index, trial in enumerate(trials):
        order = tuple(trial.get("order", ()))
        if order != expected:
            raise ValueError(f"trial order must alternate strictly; mismatch at pair {index}")
        expected = tuple(reversed(expected))
        control_ms = float(trial["baseline_ms"])
        candidate_ms = float(trial["candidate_ms"])
        if not math.isfinite(control_ms) or not math.isfinite(candidate_ms) or min(control_ms, candidate_ms) <= 0:
            raise ValueError("trial latencies must be finite positive values")
        baseline.append(control_ms)
        candidate.append(candidate_ms)
    stages = {name: float(value) for name, value in stage_ms.items()}
    if not stages or any(not math.isfinite(value) or value < 0 for value in stages.values()):
        raise ValueError("stage latencies must be a non-empty map of finite non-negative values")

    median_ms = statistics.median(candidate)
    p95_ms = _percentile(candidate, 0.95)
    stage_sum_ms = sum(stages.values())
    reasons: list[str] = []
    if median_ms > MEDIAN_LIMIT_MS:
        reasons.append("median_latency")
    if p95_ms > P95_LIMIT_MS:
        reasons.append("p95_latency")
    if stage_sum_ms > STAGE_SUM_LIMIT_MS:
        reasons.append("stage_budget")
    dense_tflops = (
        DENSE_EQUIVALENT_FLOPS_PER_PARENT * batch / median_ms / 1_000_000_000.0
    )
    return {
        "schema_version": 1,
        "status": "accepted" if not reasons else "rejected",
        "reasons": reasons,
        "batch": batch,
        "concurrency": concurrency,
        "trial_count": len(trials),
        "candidate_median_ms": median_ms,
        "candidate_p95_ms": p95_ms,
        "baseline_median_ms": statistics.median(baseline),
        "stage_sum_ms": stage_sum_ms,
        "stage_ms": stages,
        "dense_equivalent_flops_per_parent": DENSE_EQUIVALENT_FLOPS_PER_PARENT,
        "dense_equivalent_tflops": dense_tflops,
        "limits": {
            "median_ms": MEDIAN_LIMIT_MS,
            "p95_ms": P95_LIMIT_MS,
            "stage_sum_ms": STAGE_SUM_LIMIT_MS,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = evaluate_stream1_gate(
        trials=payload["trials"], stage_ms=payload["stage_ms"],
        batch=int(payload["batch"]), concurrency=int(payload["concurrency"]),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if report["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())

