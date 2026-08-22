#!/usr/bin/env python3
"""Ranking-aware, fail-closed contracts for Cube4 SM120 mixed-precision tuning.

This module deliberately separates policy selection and immutable artifact
generation from CUDA execution.  GPU observations are supplied by Molab tools;
production may consume only a validated profile with matching fingerprints.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
CORE_OPERATORS = tuple(
    f"blocks.{block}.{suffix}"
    for block in range(4)
    for suffix in (
        "attn.in_proj_weight",
        "attn.out_proj.weight",
        "ff.0.weight",
        "ff.3.weight",
    )
)
PRECISIONS = frozenset(("fp16", "sm120_block_fp8"))


@dataclass(frozen=True)
class QualityThresholds:
    top1_agreement: float = 0.999
    topk_set_overlap: float = 0.999
    frontier_jaccard: float = 0.995
    threshold_band_agreement: float = 0.999


def _validate_scores(baseline: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if baseline.shape != candidate.shape or baseline.ndim != 2 or baseline.shape[1] < 2:
        raise ValueError("score arrays must have identical [rows, outputs>=2] shapes")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("score arrays must contain only finite values")
    return baseline, candidate


def ranking_metrics(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    top_k: int = 24,
    threshold_band: float = 1.0,
) -> dict[str, float]:
    """Return ranking metrics without collapsing the 24-output head to MSE."""
    baseline, candidate = _validate_scores(baseline, candidate)
    rows, outputs = baseline.shape
    if not 1 <= top_k <= outputs:
        raise ValueError("top_k must be within the output dimension")
    if threshold_band < 0 or not np.isfinite(threshold_band):
        raise ValueError("threshold_band must be finite and non-negative")

    baseline_order = np.argsort(-baseline, axis=1, kind="stable")
    candidate_order = np.argsort(-candidate, axis=1, kind="stable")
    top1 = float(np.mean(baseline_order[:, 0] == candidate_order[:, 0]))
    overlaps = []
    inversion_rates = []
    band_agreements = []
    for row in range(rows):
        base_top = set(baseline_order[row, :top_k].tolist())
        cand_top = set(candidate_order[row, :top_k].tolist())
        overlaps.append(len(base_top & cand_top) / top_k)
        inversions = 0
        pairs = 0
        for left in range(outputs):
            for right in range(left + 1, outputs):
                base_sign = np.sign(baseline[row, left] - baseline[row, right])
                cand_sign = np.sign(candidate[row, left] - candidate[row, right])
                if base_sign != 0:
                    pairs += 1
                    inversions += int(cand_sign != base_sign)
        inversion_rates.append(inversions / pairs if pairs else 0.0)

        base_cutoff = baseline[row, baseline_order[row, min(top_k, outputs) - 1]]
        near = np.flatnonzero(np.abs(baseline[row] - base_cutoff) <= threshold_band)
        agree = 0
        comparisons = 0
        for i, left in enumerate(near):
            for right in near[i + 1 :]:
                base_sign = np.sign(baseline[row, left] - baseline[row, right])
                if base_sign != 0:
                    comparisons += 1
                    agree += int(np.sign(candidate[row, left] - candidate[row, right]) == base_sign)
        band_agreements.append(agree / comparisons if comparisons else 1.0)

    error = candidate - baseline
    return {
        "rows": float(rows),
        "outputs": float(outputs),
        "top1_agreement": top1,
        "topk_set_overlap": float(np.mean(overlaps)),
        "pair_inversion_rate": float(np.mean(inversion_rates)),
        "threshold_band_agreement": float(np.mean(band_agreements)),
        "logit_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "logit_max_abs_error": float(np.max(np.abs(error))),
    }


def frontier_jaccard(baseline_ids: Sequence[int], candidate_ids: Sequence[int]) -> float:
    baseline = set(int(value) for value in baseline_ids)
    candidate = set(int(value) for value in candidate_ids)
    union = baseline | candidate
    return len(baseline & candidate) / len(union) if union else 1.0


def _operator_contract(precision: str) -> dict[str, Any]:
    if precision not in PRECISIONS:
        raise ValueError(f"unsupported operator precision: {precision}")
    if precision == "fp16":
        return {
            "weight_dtype": "fp16",
            "activation_dtype": "fp16",
            "accumulator_dtype": "fp32",
            "output_dtype": "fp16",
            "scale_dtype": None,
            "scale_granularity": None,
            "fallback_precision": "fp16",
            "folded_transforms": [],
        }
    return {
        "weight_dtype": "e4m3",
        "activation_dtype": "e4m3",
        "accumulator_dtype": "fp32",
        "output_dtype": "fp16",
        "scale_dtype": "fp32",
        "scale_granularity": {"m": 1, "n": 128, "k": 128},
        "fallback_precision": "fp16",
        "folded_transforms": [],
    }


def build_profile(
    *,
    checkpoint_sha256: str,
    model_metadata_sha256: str,
    calibration_sha256: str,
    gpu_identity: str,
    cutlass_commit: str,
    operator_precision: Mapping[str, str],
    solver_commit: str = "unknown",
    cuda_version: str = "unknown",
) -> dict[str, Any]:
    if set(operator_precision) != set(CORE_OPERATORS):
        raise ValueError("operator_precision must contain exactly the 16 core operators")
    profile = {
        "schema_version": SCHEMA_VERSION,
        "fingerprints": {
            "checkpoint_sha256": checkpoint_sha256,
            "model_metadata_sha256": model_metadata_sha256,
            "calibration_sha256": calibration_sha256,
            "solver_commit": solver_commit,
            "cutlass_commit": cutlass_commit,
            "cuda_version": cuda_version,
            "gpu_identity": gpu_identity,
        },
        "fixed_high_precision": [
            "frontend", "layernorm", "bias", "residual", "softmax", "output_layer"
        ],
        "operators": {
            name: _operator_contract(str(operator_precision[name])) for name in CORE_OPERATORS
        },
    }
    return validate_profile(profile)


def validate_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, Mapping) or profile.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported SM120 quant profile schema")
    fingerprints = profile.get("fingerprints")
    required_fingerprints = {
        "checkpoint_sha256", "model_metadata_sha256", "calibration_sha256",
        "solver_commit", "cutlass_commit", "cuda_version", "gpu_identity",
    }
    if not isinstance(fingerprints, Mapping) or set(fingerprints) != required_fingerprints:
        raise ValueError("profile fingerprint set is incomplete")
    operators = profile.get("operators")
    if not isinstance(operators, Mapping) or set(operators) != set(CORE_OPERATORS):
        raise ValueError("profile must contain exactly the 16 core operators")
    for name, contract in operators.items():
        if not isinstance(contract, Mapping):
            raise ValueError(f"operator {name} contract must be an object")
        activation = contract.get("activation_dtype")
        weight = contract.get("weight_dtype")
        if activation not in ("fp16", "e4m3"):
            raise ValueError(f"unsupported activation dtype for {name}: {activation}")
        if weight not in ("fp16", "e4m3"):
            raise ValueError(f"unsupported weight dtype for {name}: {weight}")
        if activation != weight:
            raise ValueError(f"mixed operand dtypes are not supported for {name}")
        if contract.get("fallback_precision") != "fp16":
            raise ValueError(f"operator {name} must fail closed to fp16")
    return json.loads(json.dumps(profile, sort_keys=True))


def pareto_select(
    observations: Sequence[Mapping[str, Any]], thresholds: QualityThresholds
) -> dict[str, Any]:
    required = ("top1_agreement", "topk_set_overlap", "frontier_jaccard", "threshold_band_agreement")
    accepted: list[dict[str, Any]] = []
    rejected: dict[str, list[str]] = {}
    threshold_values = asdict(thresholds)
    for raw in observations:
        row = dict(raw)
        name = str(row.get("name", "unnamed"))
        failures = [metric for metric in required if float(row.get(metric, -1.0)) < threshold_values[metric]]
        latency = float(row.get("latency_ms", float("inf")))
        if not np.isfinite(latency) or latency <= 0:
            failures.append("latency_ms")
        if failures:
            rejected[name] = failures
        else:
            accepted.append(row)
    if not accepted:
        return {"selected": None, "pareto": [], "rejected": rejected}
    selected = min(accepted, key=lambda row: (float(row["latency_ms"]), str(row["name"])))
    pareto = []
    for row in accepted:
        dominated = any(
            other is not row
            and float(other["latency_ms"]) <= float(row["latency_ms"])
            and all(float(other[key]) >= float(row[key]) for key in required)
            and (
                float(other["latency_ms"]) < float(row["latency_ms"])
                or any(float(other[key]) > float(row[key]) for key in required)
            )
            for other in accepted
        )
        if not dominated:
            pareto.append(row)
    return {
        "selected": selected,
        "pareto": sorted(pareto, key=lambda row: float(row["latency_ms"])),
        "rejected": rejected,
    }


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_immutable_profile(
    output_dir: Path,
    profile: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Any],
) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"profile directory already exists: {output_dir}")
    validated = validate_profile(profile)
    output_dir.mkdir(parents=True)
    files: dict[str, str] = {}
    payloads = {"profile.json": validated, **dict(artifacts)}
    for relative, value in sorted(payloads.items()):
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = value if isinstance(value, bytes) else _canonical_json(value)
        path.write_bytes(data)
        files[relative] = hashlib.sha256(data).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "files": files,
        "profile_sha256": files["profile.json"],
    }
    (output_dir / "manifest.json").write_bytes(_canonical_json(manifest))
    return output_dir
