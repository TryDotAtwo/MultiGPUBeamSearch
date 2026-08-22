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
import torch


SCHEMA_VERSION = 2
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
PRECISIONS = frozenset(("fp16", "sm120_block_fp8", "sm120_block_int8"))


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

        # Per-state sensitivity is measured around the best move.  The global
        # beam cutoff is evaluated separately from candidate/frontier records.
        base_cutoff = baseline[row, baseline_order[row, 0]]
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


def _positive_scales(scales: torch.Tensor, expected: int) -> torch.Tensor:
    scales = torch.as_tensor(scales)
    if scales.ndim != 1 or scales.numel() != expected:
        raise ValueError(f"equalization scales must contain exactly {expected} values")
    if not torch.isfinite(scales).all() or torch.any(scales <= 0):
        raise ValueError("equalization scales must be finite and positive")
    return scales


def smoothquant_scales(
    activation_amax: torch.Tensor,
    weight_amax: torch.Tensor,
    *,
    alpha: float,
    minimum: float = 1.0 / 16.0,
    maximum: float = 16.0,
) -> torch.Tensor:
    """Compute bounded positive channel equalization scales.

    Zero activation channels collapse to the lower bound.  A missing/zero
    weight range is neutralized to one rather than producing infinity.
    """
    activation_amax = torch.as_tensor(activation_amax)
    weight_amax = torch.as_tensor(weight_amax, device=activation_amax.device)
    if activation_amax.shape != weight_amax.shape or activation_amax.ndim != 1:
        raise ValueError("activation and weight amax must be equal 1D shapes")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("SmoothQuant alpha must be within [0, 1]")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("invalid SmoothQuant scale bounds")
    if torch.any(activation_amax < 0) or torch.any(weight_amax < 0):
        raise ValueError("amax statistics must be non-negative")
    safe_weight = torch.where(weight_amax > 0, weight_amax, torch.ones_like(weight_amax))
    raw = torch.pow(activation_amax, alpha) / torch.pow(safe_weight, 1.0 - alpha)
    return torch.nan_to_num(raw, nan=minimum, posinf=maximum, neginf=minimum).clamp(minimum, maximum)


def fold_layernorm_linear_equalization(
    gamma: torch.Tensor,
    beta: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if gamma.ndim != 1 or beta.shape != gamma.shape or weight.ndim != 2 or weight.shape[1] != gamma.numel():
        raise ValueError("LayerNorm-to-linear fold shape mismatch")
    scales = _positive_scales(scales.to(device=gamma.device, dtype=gamma.dtype), gamma.numel())
    return gamma * scales, beta * scales, weight / scales.to(weight.dtype).unsqueeze(0)


def fold_ffn_equalization(
    ff1_weight: torch.Tensor,
    ff1_bias: torch.Tensor,
    ff2_weight: torch.Tensor,
    scales: torch.Tensor,
    *,
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if activation.lower() != "relu":
        raise ValueError("FFN diagonal folding is only graph-preserving for ReLU")
    hidden = ff1_weight.shape[0] if ff1_weight.ndim == 2 else -1
    if hidden <= 0 or ff1_bias.shape != (hidden,) or ff2_weight.ndim != 2 or ff2_weight.shape[1] != hidden:
        raise ValueError("FF1-ReLU-FF2 fold shape mismatch")
    scales = _positive_scales(scales.to(device=ff1_weight.device, dtype=ff1_weight.dtype), hidden)
    return (
        ff1_weight * scales.unsqueeze(1),
        ff1_bias * scales,
        ff2_weight / scales.to(ff2_weight.dtype).unsqueeze(0),
    )


def fold_qk_reciprocal_equalization(
    query: torch.Tensor, key: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.shape != key.shape or query.ndim < 2:
        raise ValueError("Q/K equalization requires matching tensors")
    scales = _positive_scales(scales.to(device=query.device, dtype=query.dtype), query.shape[-1])
    return query * scales, key / scales.to(key.dtype)


def fold_v_output_equalization(
    value: torch.Tensor, output_weight: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim < 2 or output_weight.ndim != 2 or output_weight.shape[1] != value.shape[-1]:
        raise ValueError("V-to-output equalization shape mismatch")
    scales = _positive_scales(scales.to(device=value.device, dtype=value.dtype), value.shape[-1])
    return value * scales, output_weight / scales.to(output_weight.dtype).unsqueeze(0)


def _operator_contract(precision: str) -> dict[str, Any]:
    if precision not in PRECISIONS:
        raise ValueError(f"unsupported operator precision: {precision}")
    if precision == "fp16":
        return {
            "weight_dtype": "fp16",
            "weight_encoding": "offline_immutable",
            "activation_dtype": "fp16",
            "activation_encoding": "native_dynamic_values",
            "accumulator_dtype": "fp32",
            "output_dtype": "fp16",
            "scale_dtype": None,
            "scale_granularity": None,
            "fallback_precision": "fp16",
            "folded_transforms": [],
        }
    low_dtype = "e4m3" if precision == "sm120_block_fp8" else "int8"
    return {
        "weight_dtype": low_dtype,
        "weight_encoding": "offline_immutable",
        "activation_dtype": low_dtype,
        "activation_encoding": "dynamic_per_batch",
        "accumulator_dtype": "fp32" if low_dtype == "e4m3" else "int32",
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
    operator_transforms: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    solver_commit: str = "unknown",
    cuda_version: str = "unknown",
) -> dict[str, Any]:
    if set(operator_precision) != set(CORE_OPERATORS):
        raise ValueError("operator_precision must contain exactly the 16 core operators")
    transforms = operator_transforms or {name: [] for name in CORE_OPERATORS}
    if set(transforms) != set(CORE_OPERATORS):
        raise ValueError("operator_transforms must contain exactly the 16 core operators")
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
            name: {
                **_operator_contract(str(operator_precision[name])),
                "folded_transforms": [dict(item) for item in transforms[name]],
            }
            for name in CORE_OPERATORS
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
        if activation not in ("fp16", "e4m3", "int8"):
            raise ValueError(f"unsupported activation dtype for {name}: {activation}")
        if weight not in ("fp16", "e4m3", "int8"):
            raise ValueError(f"unsupported weight dtype for {name}: {weight}")
        if activation != weight:
            raise ValueError(f"mixed operand dtypes are not supported for {name}")
        if contract.get("weight_encoding") != "offline_immutable":
            raise ValueError(f"operator {name} weights must be encoded offline")
        expected_activation_encoding = (
            "native_dynamic_values" if activation == "fp16" else "dynamic_per_batch"
        )
        if contract.get("activation_encoding") != expected_activation_encoding:
            raise ValueError(f"operator {name} activation encoding is inconsistent")
        if contract.get("fallback_precision") != "fp16":
            raise ValueError(f"operator {name} must fail closed to fp16")
        transforms = contract.get("folded_transforms")
        if not isinstance(transforms, list):
            raise ValueError(f"operator {name} folded_transforms must be an array")
        for transform in transforms:
            if not isinstance(transform, Mapping) or set(transform) != {"type", "alpha", "scales"}:
                raise ValueError(f"operator {name} has an invalid folded transform")
            if transform["type"] != "layernorm_linear_smoothquant":
                raise ValueError(f"operator {name} has an unsupported folded transform")
            alpha = float(transform["alpha"])
            scales = transform["scales"]
            if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
                raise ValueError(f"operator {name} folded transform alpha is invalid")
            if not isinstance(scales, list) or len(scales) != 256:
                raise ValueError(f"operator {name} folded transform requires 256 scales")
            if any(not np.isfinite(float(value)) or float(value) <= 0.0 for value in scales):
                raise ValueError(f"operator {name} folded transform scales must be positive")
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
