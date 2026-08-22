#!/usr/bin/env python3
"""Fail-closed selector for a native SM120 mixed-precision profile.

The selector intentionally consumes separate artifacts.  Ranking must be
measured against the original FP32 checkpoint, frontier quality must come from
reconstructed states, and latency must come from the native Stream1 path.
Missing evidence rejects a candidate instead of silently substituting a fake
quantization or PyTorch timing result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.sm120_quant_tuner import (
    CORE_OPERATORS,
    QualityThresholds,
    build_profile,
    pareto_select,
    write_immutable_profile,
)


def _candidate_map(rows: Sequence[Mapping[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        name = str(row.get("name", ""))
        if not name:
            raise ValueError(f"{source} contains an unnamed candidate")
        if name in result:
            raise ValueError(f"{source} contains duplicate candidate {name}")
        result[name] = row
    return result


def _minimum_frontier_jaccard(payload: Mapping[str, Any]) -> float:
    depths = payload.get("depths")
    if not isinstance(depths, list) or not depths:
        raise ValueError("frontier artifact must contain non-empty depths")
    values = [float(row["jaccard"]) for row in depths]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("frontier Jaccard must be in [0, 1]")
    return min(values)


def select_profile(
    ranking_rows: Sequence[Mapping[str, Any]],
    frontier_rows: Sequence[Mapping[str, Any]],
    benchmark_rows: Sequence[Mapping[str, Any]],
    thresholds: QualityThresholds,
) -> dict[str, Any]:
    ranking = _candidate_map(ranking_rows, "ranking")
    frontier = _candidate_map(frontier_rows, "frontier")
    benchmark = _candidate_map(benchmark_rows, "benchmark")
    names = sorted(set(ranking) | set(frontier) | set(benchmark))
    complete: list[dict[str, Any]] = []
    missing: dict[str, list[str]] = {}
    for name in names:
        absent = [
            source for source, rows in (
                ("ranking", ranking), ("frontier", frontier), ("benchmark", benchmark)
            ) if name not in rows
        ]
        if absent:
            missing[name] = absent
            continue
        policy = ranking[name].get("operator_precision")
        if not isinstance(policy, Mapping) or set(policy) != set(CORE_OPERATORS):
            missing[name] = ["operator_precision"]
            continue
        row = dict(ranking[name])
        row["frontier_jaccard"] = _minimum_frontier_jaccard(frontier[name])
        row["latency_ms"] = float(benchmark[name]["latency_ms"])
        row["operator_precision"] = dict(policy)
        complete.append(row)
    decision = pareto_select(complete, thresholds)
    decision["missing_evidence"] = missing
    decision["quality_thresholds"] = {
        "top1_agreement": thresholds.top1_agreement,
        "topk_set_overlap": thresholds.topk_set_overlap,
        "frontier_jaccard": thresholds.frontier_jaccard,
        "threshold_band_agreement": thresholds.threshold_band_agreement,
    }
    return decision


def _load_rows(path: Path, key: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get(key) if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a {key} array")
    return [dict(row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--frontiers", type=Path, required=True)
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--model-metadata-sha256", required=True)
    parser.add_argument("--calibration-sha256", required=True)
    parser.add_argument("--gpu-identity", required=True)
    parser.add_argument("--cutlass-commit", required=True)
    parser.add_argument("--solver-commit", required=True)
    parser.add_argument("--cuda-version", required=True)
    parser.add_argument("--top1", type=float, default=0.999)
    parser.add_argument("--topk", type=float, default=0.999)
    parser.add_argument("--frontier-jaccard", type=float, default=0.995)
    parser.add_argument("--threshold-band", type=float, default=0.999)
    args = parser.parse_args()
    ranking_rows = _load_rows(args.ranking, "observations")
    frontier_rows = _load_rows(args.frontiers, "candidates")
    benchmark_rows = _load_rows(args.benchmarks, "candidates")
    thresholds = QualityThresholds(
        top1_agreement=args.top1,
        topk_set_overlap=args.topk,
        frontier_jaccard=args.frontier_jaccard,
        threshold_band_agreement=args.threshold_band,
    )
    decision = select_profile(ranking_rows, frontier_rows, benchmark_rows, thresholds)
    selected = decision.get("selected")
    if selected is None:
        raise SystemExit("no candidate passed all ranking, frontier, and native latency gates")
    profile = build_profile(
        checkpoint_sha256=args.checkpoint_sha256,
        model_metadata_sha256=args.model_metadata_sha256,
        calibration_sha256=args.calibration_sha256,
        gpu_identity=args.gpu_identity,
        cutlass_commit=args.cutlass_commit,
        operator_precision=selected["operator_precision"],
        solver_commit=args.solver_commit,
        cuda_version=args.cuda_version,
    )
    write_immutable_profile(
        args.output_dir,
        profile,
        artifacts={
            "selection.json": decision,
            "ranking_metrics.json": json.loads(args.ranking.read_text(encoding="utf-8")),
            "frontier_metrics.json": json.loads(args.frontiers.read_text(encoding="utf-8")),
            "native_benchmarks.json": json.loads(args.benchmarks.read_text(encoding="utf-8")),
        },
    )
    print(json.dumps({"selected": selected["name"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
