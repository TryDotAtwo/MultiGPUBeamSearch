#!/usr/bin/env python3
"""Compare reconstructed frontier states for an SM120 quality gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.sm120_quant_calibrate import (
    _load_initial_state,
    generator_actions,
    reconstruct_frontiers,
)


def _state_set(rows) -> set[bytes]:
    return {memoryview(row).tobytes() for row in rows}


def compare_frontiers(
    baseline_dir: Path,
    candidate_dir: Path,
    generator_json: Path,
    test_csv: Path,
) -> list[dict[str, object]]:
    payload = json.loads(Path(generator_json).read_text(encoding="utf-8"))
    generators = generator_actions(payload)
    initial = _load_initial_state(Path(test_csv))
    baseline = reconstruct_frontiers(Path(baseline_dir), initial, generators)
    candidate = reconstruct_frontiers(Path(candidate_dir), initial, generators)
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate history depths differ")
    rows: list[dict[str, object]] = []
    for depth in sorted(baseline):
        left = _state_set(baseline[depth])
        right = _state_set(candidate[depth])
        union = left | right
        intersection = left & right
        rows.append({
            "depth": depth,
            "baseline_states": len(left),
            "candidate_states": len(right),
            "intersection_states": len(intersection),
            "jaccard": len(intersection) / len(union) if union else 1.0,
            "baseline_sha256": hashlib.sha256(b"".join(sorted(left))).hexdigest(),
            "candidate_sha256": hashlib.sha256(b"".join(sorted(right))).hexdigest(),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-history", type=Path, required=True)
    parser.add_argument("--candidate-history", type=Path, required=True)
    parser.add_argument("--generator-json", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--minimum-jaccard", type=float, default=0.995)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_jaccard <= 1.0:
        raise ValueError("minimum_jaccard must be in [0, 1]")
    depths = compare_frontiers(
        args.baseline_history, args.candidate_history,
        args.generator_json, args.test_csv,
    )
    failures = [row["depth"] for row in depths if float(row["jaccard"]) < args.minimum_jaccard]
    payload = {
        "schema_version": 1,
        "minimum_jaccard": args.minimum_jaccard,
        "accepted": not failures,
        "failed_depths": failures,
        "depths": depths,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
