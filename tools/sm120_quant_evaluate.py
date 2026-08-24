#!/usr/bin/env python3
"""Evaluate SM120 mixed-precision candidates on real Cube4 frontier states."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from tools.sm120_quant_calibrate import (
    activation_block_statistics,
    fake_sm120_activation_quant,
    fake_sm120_int8_activation_quant,
    fake_sm120_int8_weight_quant,
    fake_sm120_weight_quant,
)
from tools.sm120_quant_tuner import CORE_OPERATORS, ranking_metrics, smoothquant_scales
from tools.stream1_transformer_torch_benchmark import PieceTransformerTorch


def activation_weighted_low_rank_factors(
    reference_weight: torch.Tensor,
    quantized_weight: torch.Tensor,
    activation_rms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return L,R with L@R approximating W_ref-W_quant under RMS weighting."""
    if reference_weight.ndim != 2 or quantized_weight.shape != reference_weight.shape:
        raise ValueError("low-rank residual weight shape mismatch")
    rms = activation_rms.to(
        device=reference_weight.device, dtype=torch.float32
    ).reshape(-1).clamp_min(1.0e-6)
    if rms.numel() != reference_weight.shape[0]:
        raise ValueError("low-rank residual activation RMS shape mismatch")
    error = reference_weight.float() - quantized_weight.float()
    u, singular, vh = torch.linalg.svd(rms[:, None] * error, full_matrices=False)
    return (u * singular.unsqueeze(0)) / rms[:, None], vh


def select_stratified_states(
    states: np.ndarray,
    depths: np.ndarray,
    *,
    max_states: int,
    split: str = "all",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select deterministic, near-equal quotas from every represented depth."""
    states = np.asarray(states)
    depths = np.asarray(depths)
    if states.shape[0] != depths.shape[0] or states.ndim != 2 or depths.ndim != 1:
        raise ValueError("states/depths shape mismatch")
    if max_states <= 0:
        raise ValueError("max_states must be positive")
    if split not in ("all", "calibration", "holdout"):
        raise ValueError("split must be all, calibration, or holdout")
    unique = np.unique(depths)
    if unique.size == 0:
        raise ValueError("empty calibration corpus")
    eligible_mask = np.ones(states.shape[0], dtype=bool)
    if split != "all":
        eligible_mask = np.arange(states.shape[0]) % 2 == (0 if split == "calibration" else 1)
    target = min(max_states, int(np.count_nonzero(eligible_mask)))
    base, remainder = divmod(target, unique.size)
    selected: list[np.ndarray] = []
    for position, depth in enumerate(unique):
        available = np.flatnonzero(depths == depth)
        if split != "all":
            parity = 0 if split == "calibration" else 1
            available = available[available % 2 == parity]
        quota = min(available.size, base + int(position < remainder))
        if quota:
            # Even spacing avoids a prefix-only bias while remaining reproducible.
            offsets = np.linspace(0, available.size - 1, quota, dtype=np.int64)
            selected.append(available[offsets])
    indices = np.sort(np.concatenate(selected))
    if indices.size < target:
        missing = np.setdiff1d(np.flatnonzero(eligible_mask), indices, assume_unique=True)
        indices = np.sort(np.concatenate((indices, missing[: target - indices.size])))
    return states[indices], depths[indices], indices


def build_initial_mixed_precision_policies() -> list[tuple[str, dict[str, str]]]:
    all_fp8 = {name: "sm120_block_fp8" for name in CORE_OPERATORS}
    policies: list[tuple[str, dict[str, str]]] = [("all_core_fp8", all_fp8)]
    for name in CORE_OPERATORS:
        policy = dict(all_fp8)
        policy[name] = "fp16"
        policies.append(("rollback_" + name, policy))
        only = {other: "fp16" for other in CORE_OPERATORS}
        only[name] = "sm120_block_fp8"
        policies.append(("only_fp8_" + name, only))
    return policies


def build_initial_int8_policies() -> list[tuple[str, dict[str, str]]]:
    all_int8 = {name: "sm120_block_int8" for name in CORE_OPERATORS}
    policies: list[tuple[str, dict[str, str]]] = [("all_core_int8", all_int8)]
    for name in CORE_OPERATORS:
        only = {other: "fp16" for other in CORE_OPERATORS}
        only[name] = "sm120_block_int8"
        policies.append(("only_int8_" + name, only))
    return policies


def _quality_order(rows: list[Mapping[str, Any]], prefix: str) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["top1_agreement"]),
            float(row["threshold_band_agreement"]),
            float(row["global_top_per_state_overlap"]),
            float(row["topk_set_overlap"]),
            str(row["name"]),
        ),
        reverse=True,
    )


def build_incremental_fp8_policies(
    single_fp8_rows: list[Mapping[str, Any]],
) -> list[tuple[str, dict[str, str]]]:
    expected = {"only_fp8_" + name for name in CORE_OPERATORS}
    if {str(row.get("name")) for row in single_fp8_rows} != expected:
        raise ValueError("single FP8 observations must cover all core operators")
    policy = {name: "fp16" for name in CORE_OPERATORS}
    result: list[tuple[str, dict[str, str]]] = []
    for count, row in enumerate(_quality_order(single_fp8_rows, "only_fp8_"), start=1):
        operator = str(row["name"])[len("only_fp8_") :]
        policy[operator] = "sm120_block_fp8"
        if count >= 2:
            result.append((f"incremental_fp8_{count:02d}_{operator}", dict(policy)))
    return result


def build_cumulative_rollback_policies(
    single_rollback_rows: list[Mapping[str, Any]],
) -> list[tuple[str, dict[str, str]]]:
    """Build a deterministic quality-first path from all-FP8 to all-FP16."""
    expected = {"rollback_" + name for name in CORE_OPERATORS}
    if {str(row.get("name")) for row in single_rollback_rows} != expected:
        raise ValueError("single rollback observations must cover all core operators")
    ordered = _quality_order(single_rollback_rows, "rollback_")
    policy = {name: "sm120_block_fp8" for name in CORE_OPERATORS}
    result: list[tuple[str, dict[str, str]]] = []
    for count, row in enumerate(ordered, start=1):
        operator = str(row["name"])[len("rollback_") :]
        policy[operator] = "fp16"
        if count >= 2:
            result.append((f"cumulative_{count:02d}_{operator}", dict(policy)))
    return result


class QuantObservedPieceTransformer(PieceTransformerTorch):
    def __init__(self, weight_dir: Path, device: torch.device) -> None:
        super().__init__(weight_dir, device, projection_mode="matmul", qkv_contiguous=True)
        self.precision = {name: "fp16" for name in CORE_OPERATORS}
        self.quantized_weights: dict[str, torch.Tensor] = {}
        self.quantized_int8_weights: dict[str, torch.Tensor] = {}
        self.capture_statistics = False
        self.statistics: dict[str, list[dict[str, float | int]]] = {}
        self.channel_amax: dict[str, torch.Tensor] = {}
        self.channel_sumsq: dict[str, torch.Tensor] = {}
        self.channel_count: dict[str, int] = {}
        self.low_rank_bases: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.low_rank_corrections: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for block_index, block in enumerate(self.blocks):
            for suffix, key in (
                ("attn.in_proj_weight", "qkv_weight"),
                ("attn.out_proj.weight", "attn_out_weight"),
                ("ff.0.weight", "ff1_weight"),
                ("ff.3.weight", "ff2_weight"),
            ):
                name = f"blocks.{block_index}.{suffix}"
                quantized, _ = fake_sm120_weight_quant(block[key].float())
                self.quantized_weights[name] = quantized.to(dtype=self.dtype)
                quantized_int8, _ = fake_sm120_int8_weight_quant(block[key].float())
                self.quantized_int8_weights[name] = quantized_int8.to(dtype=self.dtype)

    @staticmethod
    def _operator_weight(model: "QuantObservedPieceTransformer", name: str) -> torch.Tensor:
        parts = name.split(".")
        block = model.blocks[int(parts[1])]
        suffix = ".".join(parts[2:])
        keys = {
            "attn.in_proj_weight": "qkv_weight",
            "attn.out_proj.weight": "attn_out_weight",
            "ff.0.weight": "ff1_weight",
            "ff.3.weight": "ff2_weight",
        }
        return block[keys[suffix]]

    def install_fp32_quantization_source(
        self, reference: "QuantObservedPieceTransformer"
    ) -> None:
        """Encode every candidate weight from FP32 truth, never from FP16."""
        for name in CORE_OPERATORS:
            self.refresh_quantized_weight(name, self._operator_weight(reference, name).float())

    def prepare_activation_weighted_low_rank_residuals(
        self,
        reference: "QuantObservedPieceTransformer",
        channel_rms: Mapping[str, torch.Tensor],
    ) -> None:
        """Factor D*(W_fp32-W_fp8) once; D is real-frontier activation RMS."""
        supported = tuple(
            name for name in CORE_OPERATORS
            if name.endswith("attn.in_proj_weight") or name.endswith("ff.0.weight")
        )
        self.low_rank_bases.clear()
        for name in supported:
            if name not in channel_rms:
                raise ValueError(f"missing activation RMS for low-rank residual {name}")
            reference_weight = self._operator_weight(reference, name).float()
            quantized_weight = self.quantized_weights[name].float()
            self.low_rank_bases[name] = activation_weighted_low_rank_factors(
                reference_weight, quantized_weight, channel_rms[name]
            )

    def select_low_rank_residual(self, rank: int) -> None:
        if rank <= 0:
            raise ValueError("low-rank residual rank must be positive")
        self.low_rank_corrections = {
            name: (
                left[:, : min(rank, left.shape[1])].to(dtype=self.dtype),
                right[: min(rank, right.shape[0]), :].to(dtype=self.dtype),
            )
            for name, (left, right) in self.low_rank_bases.items()
        }

    def refresh_quantized_weight(self, name: str, weight_hxk: torch.Tensor) -> None:
        quantized, _ = fake_sm120_weight_quant(weight_hxk.float())
        self.quantized_weights[name] = quantized.to(dtype=self.dtype)
        quantized_int8, _ = fake_sm120_int8_weight_quant(weight_hxk.float())
        self.quantized_int8_weights[name] = quantized_int8.to(dtype=self.dtype)

    def apply_layernorm_smoothquant(
        self,
        activation_amax: Mapping[str, torch.Tensor],
        *,
        alpha: float,
    ) -> dict[str, list[float]]:
        """Fold exact positive scales into LN affine and adjacent HxK weights."""
        applied: dict[str, list[float]] = {}
        for block_index, block in enumerate(self.blocks):
            for suffix, weight_key, gamma_key, beta_key in (
                ("attn.in_proj_weight", "qkv_weight", "ln1_gamma", "ln1_beta"),
                ("ff.0.weight", "ff1_weight", "ln2_gamma", "ln2_beta"),
            ):
                name = f"blocks.{block_index}.{suffix}"
                if name not in activation_amax:
                    raise ValueError(f"missing activation amax for {name}")
                weight = block[weight_key]
                scales = smoothquant_scales(
                    activation_amax[name].to(device=weight.device, dtype=torch.float32),
                    weight.float().abs().amax(dim=1),
                    alpha=alpha,
                ).to(dtype=self.dtype)
                block[gamma_key] = block[gamma_key] * scales
                block[beta_key] = block[beta_key] * scales
                block[weight_key] = weight / scales.unsqueeze(1)
                self.refresh_quantized_weight(name, block[weight_key])
                applied[name] = scales.float().cpu().tolist()
        return applied

    def set_precision(self, policy: Mapping[str, str]) -> None:
        if set(policy) != set(CORE_OPERATORS):
            raise ValueError("precision policy must contain all 16 core operators")
        if any(value not in ("fp16", "sm120_block_fp8", "sm120_block_int8") for value in policy.values()):
            raise ValueError("unknown precision policy value")
        self.precision = dict(policy)

    def project_named(
        self,
        name: str,
        x: torch.Tensor,
        weight_hxk: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        if self.capture_statistics:
            self.statistics.setdefault(name, []).append(activation_block_statistics(x))
            batch_amax = x.detach().float().reshape(-1, x.shape[-1]).abs().amax(dim=0)
            previous = self.channel_amax.get(name)
            self.channel_amax[name] = batch_amax if previous is None else torch.maximum(previous, batch_amax)
            flat = x.detach().float().reshape(-1, x.shape[-1])
            batch_sumsq = torch.sum(flat * flat, dim=0)
            self.channel_sumsq[name] = self.channel_sumsq.get(name, torch.zeros_like(batch_sumsq)) + batch_sumsq
            self.channel_count[name] = self.channel_count.get(name, 0) + flat.shape[0]
        original_x = x
        if self.precision[name] == "sm120_block_fp8":
            x, _ = fake_sm120_activation_quant(x.float())
            x = x.to(dtype=self.dtype)
            weight_hxk = self.quantized_weights[name]
        elif self.precision[name] == "sm120_block_int8":
            x, _ = fake_sm120_int8_activation_quant(x.float())
            x = x.to(dtype=self.dtype)
            weight_hxk = self.quantized_int8_weights[name]
        result = x.matmul(weight_hxk) + bias
        correction = self.low_rank_corrections.get(name)
        if correction is not None:
            left, right = correction
            result = result + original_x.matmul(left).matmul(right)
        return result

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(self.build_tokens(states), self.input_ln_gamma, self.input_ln_beta)
        batch = x.shape[0]
        for block_index, block in enumerate(self.blocks):
            y = self.layer_norm(x, block["ln1_gamma"], block["ln1_beta"])
            qkv = self.project_named(
                f"blocks.{block_index}.attn.in_proj_weight",
                y, block["qkv_weight"], block["qkv_bias"],
            )
            qkv = qkv.reshape(batch, self.seq_len, 3, self.nhead, self.head_dim)
            q = qkv[:, :, 0].permute(0, 2, 1, 3).contiguous()
            k = qkv[:, :, 1].permute(0, 2, 1, 3).contiguous()
            v = qkv[:, :, 2].permute(0, 2, 1, 3).contiguous()
            attention = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            context = attention.permute(0, 2, 1, 3).reshape(batch, self.seq_len, self.d_model)
            x = x + self.project_named(
                f"blocks.{block_index}.attn.out_proj.weight",
                context, block["attn_out_weight"], block["attn_out_bias"],
            )
            y = self.layer_norm(x, block["ln2_gamma"], block["ln2_beta"])
            y = self.activate(self.project_named(
                f"blocks.{block_index}.ff.0.weight",
                y, block["ff1_weight"], block["ff1_bias"],
            ))
            x = x + self.project_named(
                f"blocks.{block_index}.ff.3.weight",
                y, block["ff2_weight"], block["ff2_bias"],
            )
        cls = self.layer_norm(x[:, 0], self.output_ln_gamma, self.output_ln_beta)
        return cls.matmul(self.output_weight) + self.output_bias


def _run_logits(
    model: QuantObservedPieceTransformer,
    states: np.ndarray,
    *,
    batch_size: int,
    capture_statistics: bool,
) -> tuple[np.ndarray, float]:
    outputs = []
    model.capture_statistics = capture_statistics
    model.statistics.clear()
    if capture_statistics:
        model.channel_amax.clear()
        model.channel_sumsq.clear()
        model.channel_count.clear()
    torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, states.shape[0], batch_size):
            batch = torch.from_numpy(states[begin : begin + batch_size]).to(model.device)
            outputs.append(model.forward(batch).float().cpu())
    torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - started
    model.capture_statistics = False
    return torch.cat(outputs).numpy(), elapsed


def _aggregate_statistics(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    total_rows = sum(int(row["rows"]) for row in rows)
    result: dict[str, float | int] = {
        "rows": total_rows,
        "features": int(rows[0]["features"]),
        "blocks_per_row": int(rows[0]["blocks_per_row"]),
        "abs_max": max(float(row["abs_max"]) for row in rows),
        "scale_exponent_min": min(float(row["scale_exponent_min"]) for row in rows),
        "scale_exponent_max": max(float(row["scale_exponent_max"]) for row in rows),
    }
    for key in ("rms", "zero_block_fraction", "quantized_zero_fraction", "saturation_fraction", "qdq_nmse"):
        result[key] = sum(float(row[key]) * int(row["rows"]) for row in rows) / total_rows
    return result


def _global_overlap(baseline: np.ndarray, candidate: np.ndarray, keep: int) -> float:
    keep = min(max(1, keep), baseline.size)
    baseline_indices = np.argpartition(baseline.reshape(-1), keep - 1)[:keep]
    candidate_indices = np.argpartition(candidate.reshape(-1), keep - 1)[:keep]
    return len(set(baseline_indices.tolist()) & set(candidate_indices.tolist())) / keep


def _metrics(baseline: np.ndarray, candidate: np.ndarray, elapsed: float) -> dict[str, float]:
    metrics = ranking_metrics(
        -baseline, -candidate, top_k=min(8, baseline.shape[1]), threshold_band=0.25
    )
    metrics["global_top_per_state_overlap"] = _global_overlap(baseline, candidate, baseline.shape[0])
    metrics["wall_seconds"] = elapsed
    return metrics


def _three_way_metrics(
    fp32: np.ndarray, fp16: np.ndarray, candidate: np.ndarray, elapsed: float
) -> dict[str, Any]:
    """Keep FP32 as truth while isolating incremental mixed-vs-FP16 loss."""
    versus_fp32 = _metrics(fp32, candidate, elapsed)
    return {
        # Keep the flat FP32-relative fields for the Pareto selector and old
        # reports, but make both comparison axes explicit for human review.
        **versus_fp32,
        "vs_fp32": versus_fp32,
        "vs_fp16": _metrics(fp16, candidate, elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-dir", type=Path, required=True)
    parser.add_argument("--fp32-weight-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-states", type=int, default=4096)
    parser.add_argument("--split", choices=("all", "calibration", "holdout"), default="all")
    parser.add_argument(
        "--smoothquant-alpha-grid",
        default="0.0,0.25,0.5,0.75,1.0",
        help="comma-separated exact LN-to-QKV/FF1 equalization alpha candidates",
    )
    parser.add_argument(
        "--lowrank-ranks", default="4,8,16,32",
        help="comma-separated activation-weighted FP16 residual ranks to evaluate",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus = np.load(args.corpus)
    states, selected_depths, selected_indices = select_stratified_states(
        np.asarray(corpus["states"], dtype=np.uint8),
        np.asarray(corpus["depths"], dtype=np.int32),
        max_states=args.max_states,
        split=args.split,
    )
    device = torch.device("cuda")
    fp32_model = QuantObservedPieceTransformer(args.fp32_weight_dir, device)
    fp16_policy = {name: "fp16" for name in CORE_OPERATORS}
    fp32_model.set_precision(fp16_policy)
    fp32_logits, fp32_sec = _run_logits(
        fp32_model, states, batch_size=args.batch_size, capture_statistics=False
    )
    model = QuantObservedPieceTransformer(args.weight_dir, device)
    model.install_fp32_quantization_source(fp32_model)
    fp16 = {name: "fp16" for name in CORE_OPERATORS}
    model.set_precision(fp16)
    fp16_logits, baseline_sec = _run_logits(
        model, states, batch_size=args.batch_size, capture_statistics=True
    )
    activation_stats = {
        name: _aggregate_statistics(rows) for name, rows in model.statistics.items()
    }
    channel_amax = {name: values.detach().clone() for name, values in model.channel_amax.items()}
    channel_rms = {
        name: torch.sqrt(values / model.channel_count[name])
        for name, values in model.channel_sumsq.items()
    }
    np.save(args.output_dir / "fp32_reference_logits.npy", fp32_logits)
    np.save(args.output_dir / "fp16_current_logits.npy", fp16_logits)
    (args.output_dir / "activation_statistics.json").write_text(
        json.dumps(activation_stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    observations: list[dict[str, Any]] = [{
        "name": "fp32_reference", "wall_seconds": fp32_sec,
        "top1_agreement": 1.0, "topk_set_overlap": 1.0,
        "pair_inversion_rate": 0.0, "threshold_band_agreement": 1.0,
        "global_top_per_state_overlap": 1.0,
    }, {
        "name": "current_fp16",
        **_metrics(fp32_logits, fp16_logits, baseline_sec),
    }]
    policies = build_initial_mixed_precision_policies()
    for name, policy in policies:
        model.set_precision(policy)
        candidate, elapsed = _run_logits(
            model, states, batch_size=args.batch_size, capture_statistics=False
        )
        row = {
            "name": name,
            "operator_precision": policy,
            **_three_way_metrics(fp32_logits, fp16_logits, candidate, elapsed),
        }
        observations.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    for name, policy in build_initial_int8_policies():
        model.set_precision(policy)
        candidate, elapsed = _run_logits(
            model, states, batch_size=args.batch_size, capture_statistics=False
        )
        row = {
            "name": name,
            "operator_precision": policy,
            **_three_way_metrics(fp32_logits, fp16_logits, candidate, elapsed),
        }
        observations.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    lowrank_policy = {
        name: (
            "sm120_block_fp8"
            if name.endswith("attn.in_proj_weight") or name.endswith("ff.0.weight")
            else "fp16"
        )
        for name in CORE_OPERATORS
    }
    model.prepare_activation_weighted_low_rank_residuals(fp32_model, channel_rms)
    lowrank_ranks = [int(value) for value in args.lowrank_ranks.split(",") if value.strip()]
    if not lowrank_ranks or any(rank <= 0 for rank in lowrank_ranks):
        raise ValueError("lowrank ranks must be positive")
    for rank in lowrank_ranks:
        model.set_precision(lowrank_policy)
        model.select_low_rank_residual(rank)
        candidate, elapsed = _run_logits(
            model, states, batch_size=args.batch_size, capture_statistics=False
        )
        corrections = {
            name: {"type": "activation_weighted_low_rank_residual", "rank": rank}
            for name in model.low_rank_corrections
        }
        row = {
            "name": f"native_scope_fp8_lowrank_r{rank}",
            "operator_precision": dict(lowrank_policy),
            "operator_corrections": corrections,
            **_three_way_metrics(fp32_logits, fp16_logits, candidate, elapsed),
        }
        observations.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    model.low_rank_corrections.clear()
    rollback_rows = [row for row in observations if str(row["name"]).startswith("rollback_")]
    for name, policy in build_cumulative_rollback_policies(rollback_rows):
        model.set_precision(policy)
        candidate, elapsed = _run_logits(
            model, states, batch_size=args.batch_size, capture_statistics=False
        )
        row = {
            "name": name,
            "fp16_operators": sum(value == "fp16" for value in policy.values()),
            "operator_precision": policy,
            **_three_way_metrics(fp32_logits, fp16_logits, candidate, elapsed),
        }
        observations.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    single_fp8_rows = [row for row in observations if str(row["name"]).startswith("only_fp8_")]
    for name, policy in build_incremental_fp8_policies(single_fp8_rows):
        model.set_precision(policy)
        candidate, elapsed = _run_logits(
            model, states, batch_size=args.batch_size, capture_statistics=False
        )
        row = {
            "name": name,
            "fp8_operators": sum(value == "sm120_block_fp8" for value in policy.values()),
            "operator_precision": policy,
            **_three_way_metrics(fp32_logits, fp16_logits, candidate, elapsed),
        }
        observations.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    alphas = [float(value) for value in args.smoothquant_alpha_grid.split(",") if value.strip()]
    equalization_artifacts: dict[str, Any] = {}
    for alpha in alphas:
        equalized = QuantObservedPieceTransformer(args.weight_dir, device)
        scales = equalized.apply_layernorm_smoothquant(channel_amax, alpha=alpha)
        equalized.set_precision({name: "sm120_block_fp8" for name in CORE_OPERATORS})
        candidate, elapsed = _run_logits(
            equalized, states, batch_size=args.batch_size, capture_statistics=False
        )
        name = f"smoothquant_alpha_{alpha:g}_all_core_fp8"
        transforms = {
            operator: ([{
                "type": "layernorm_linear_smoothquant",
                "alpha": alpha,
                "scales": scales[operator],
            }] if operator in scales else [])
            for operator in CORE_OPERATORS
        }
        row = {
            "name": name,
            "smoothquant_alpha": alpha,
            "folded_operators": sorted(scales),
            "operator_precision": {name: "sm120_block_fp8" for name in CORE_OPERATORS},
            "operator_transforms": transforms,
            **_three_way_metrics(fp32_logits, fp16_logits, candidate, elapsed),
        }
        observations.append(row)
        equalization_artifacts[name] = scales
        print(json.dumps(row, sort_keys=True), flush=True)
    payload = {
        "schema_version": 1,
        "states": int(states.shape[0]),
        "batch_size": args.batch_size,
        "split": args.split,
        "depth_counts": {
            str(int(depth)): int(np.count_nonzero(selected_depths == depth))
            for depth in np.unique(selected_depths)
        },
        "selected_indices_sha256": hashlib.sha256(selected_indices.tobytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "observations": observations,
    }
    (args.output_dir / "ranking_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "equalization_scales.json").write_text(
        json.dumps(equalization_artifacts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "candidates": len(observations)}))


if __name__ == "__main__":
    main()
