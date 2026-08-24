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
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.sm120_quant_tuner import (
    CORE_OPERATORS,
    FP16_GEMM_BACKENDS,
    NATIVE_LOW_PRECISION_OPERATORS,
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
    fp16_rows = [dict(row) for row in ranking_rows if str(row.get("name", "")) == "current_fp16"]
    if len(fp16_rows) != 1:
        raise ValueError("ranking must contain exactly one current_fp16 quality reference")
    fp16_reference = fp16_rows[0]
    quality_metrics = ("top1_agreement", "topk_set_overlap", "threshold_band_agreement")
    for metric in quality_metrics:
        value = float(fp16_reference.get(metric, -1.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"current_fp16 has invalid {metric}")
    if not 0.0 <= thresholds.fp32_regression_budget <= 1.0:
        raise ValueError("fp32_regression_budget must be in [0, 1]")
    fp16_floor = {
        metric: max(0.0, float(fp16_reference[metric]) - thresholds.fp32_regression_budget)
        for metric in quality_metrics
    }
    ranking = _candidate_map(
        [row for row in ranking_rows if str(row.get("name", "")) not in {"current_fp16", "fp32_reference"}],
        "ranking",
    )
    frontier = _candidate_map(frontier_rows, "frontier")
    benchmark = _candidate_map(benchmark_rows, "benchmark")
    names = sorted(set(ranking) | set(frontier) | set(benchmark))
    complete: list[dict[str, Any]] = []
    missing: dict[str, list[str]] = {}
    quality_rejected: dict[str, list[str]] = {}
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
        unsupported_precision = [
            operator for operator, precision in policy.items()
            if precision != "fp16" and (
                operator not in NATIVE_LOW_PRECISION_OPERATORS
                or precision != "sm120_block_fp8"
            )
        ]
        if unsupported_precision:
            missing[name] = ["operator_precision.native_runtime"]
            continue
        if ranking[name].get("operator_corrections"):
            missing[name] = ["operator_corrections.native_runtime"]
            continue
        transforms = ranking[name].get("operator_transforms")
        if transforms is None:
            transforms = {operator: [] for operator in CORE_OPERATORS}
        if not isinstance(transforms, Mapping) or set(transforms) != set(CORE_OPERATORS):
            missing[name] = ["operator_transforms"]
            continue
        versus_fp32 = ranking[name].get("vs_fp32")
        versus_fp16 = ranking[name].get("vs_fp16")
        if not isinstance(versus_fp32, Mapping) or not isinstance(versus_fp16, Mapping):
            missing[name] = ["vs_fp32", "vs_fp16"]
            continue
        invalid_quality = False
        for source_name, values in (("vs_fp32", versus_fp32), ("vs_fp16", versus_fp16)):
            for metric in quality_metrics:
                value = float(values.get(metric, -1.0))
                if not 0.0 <= value <= 1.0:
                    missing[name] = [f"{source_name}.{metric}"]
                    invalid_quality = True
                    break
            if invalid_quality:
                break
        if invalid_quality:
            continue
        regressions = [
            f"fp32_{metric.removesuffix('_agreement')}_regression"
            for metric in quality_metrics
            if float(versus_fp32[metric]) < fp16_floor[metric]
        ]
        if regressions:
            quality_rejected[name] = regressions
            continue
        native_execution = benchmark[name].get("native_execution")
        if not isinstance(native_execution, Mapping):
            missing[name] = ["native_execution"]
            continue
        if set(native_execution) != {"fp16_gemm_backend", "target_sm", "workspace_bytes"}:
            missing[name] = ["native_execution"]
            continue
        if native_execution.get("fp16_gemm_backend") not in FP16_GEMM_BACKENDS:
            missing[name] = ["native_execution.fp16_gemm_backend"]
            continue
        if native_execution.get("target_sm") != 120:
            missing[name] = ["native_execution.target_sm"]
            continue
        workspace_bytes = native_execution.get("workspace_bytes")
        if not isinstance(workspace_bytes, int) or isinstance(workspace_bytes, bool) or workspace_bytes < 0:
            missing[name] = ["native_execution.workspace_bytes"]
            continue
        row = dict(ranking[name])
        # Pareto quality thresholds describe incremental deviation from the
        # accepted FP16 execution.  FP32-relative floors were enforced above.
        for metric in quality_metrics:
            row[metric] = float(versus_fp16[metric])
        row["frontier_jaccard"] = _minimum_frontier_jaccard(frontier[name])
        row["latency_ms"] = float(benchmark[name]["latency_ms"])
        row["operator_precision"] = dict(policy)
        row["operator_transforms"] = {key: list(value) for key, value in transforms.items()}
        row["native_execution"] = dict(native_execution)
        complete.append(row)
    decision = pareto_select(complete, thresholds)
    decision["rejected"].update(quality_rejected)
    decision["missing_evidence"] = missing
    decision["fp16_quality_reference"] = {
        metric: float(fp16_reference[metric]) for metric in quality_metrics
    }
    decision["fp16_quality_floor"] = fp16_floor
    decision["quality_thresholds"] = {
        "top1_agreement": thresholds.top1_agreement,
        "topk_set_overlap": thresholds.topk_set_overlap,
        "frontier_jaccard": thresholds.frontier_jaccard,
        "threshold_band_agreement": thresholds.threshold_band_agreement,
        "fp32_regression_budget": thresholds.fp32_regression_budget,
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
    parser.add_argument("--fp32-regression-budget", type=float, default=0.001)
    args = parser.parse_args()
    ranking_rows = _load_rows(args.ranking, "observations")
    frontier_rows = _load_rows(args.frontiers, "candidates")
    benchmark_rows = _load_rows(args.benchmarks, "candidates")
    thresholds = QualityThresholds(
        top1_agreement=args.top1,
        topk_set_overlap=args.topk,
        frontier_jaccard=args.frontier_jaccard,
        threshold_band_agreement=args.threshold_band,
        fp32_regression_budget=args.fp32_regression_budget,
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
        operator_transforms=selected["operator_transforms"],
        solver_commit=args.solver_commit,
        cuda_version=args.cuda_version,
        fp16_gemm_backend=selected["native_execution"]["fp16_gemm_backend"],
        target_sm=selected["native_execution"]["target_sm"],
        workspace_bytes=selected["native_execution"]["workspace_bytes"],
    )
    profile_bytes = (json.dumps(
        profile, ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n").encode("utf-8")
    runtime_execution = (
        "schema_version=3\n"
        f"profile_sha256={hashlib.sha256(profile_bytes).hexdigest()}\n"
        f"fp16_gemm_backend={profile['native_execution']['fp16_gemm_backend']}\n"
        f"target_sm={profile['native_execution']['target_sm']}\n"
        f"workspace_bytes={profile['native_execution']['workspace_bytes']}\n"
        "weights=offline_immutable\n"
    ).encode("utf-8")
    write_immutable_profile(
        args.output_dir,
        profile,
        artifacts={
            "selection.json": decision,
            "ranking_metrics.json": json.loads(args.ranking.read_text(encoding="utf-8")),
            "frontier_metrics.json": json.loads(args.frontiers.read_text(encoding="utf-8")),
            "native_benchmarks.json": json.loads(args.benchmarks.read_text(encoding="utf-8")),
            "runtime_execution.txt": runtime_execution,
        },
    )
    print(json.dumps({"selected": selected["name"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
