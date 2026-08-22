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
    fake_sm120_weight_quant,
)
from tools.sm120_quant_tuner import CORE_OPERATORS, ranking_metrics
from tools.stream1_transformer_torch_benchmark import PieceTransformerTorch


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
    return policies


def build_cumulative_rollback_policies(
    single_rollback_rows: list[Mapping[str, Any]],
) -> list[tuple[str, dict[str, str]]]:
    """Build a deterministic quality-first path from all-FP8 to all-FP16."""
    expected = {"rollback_" + name for name in CORE_OPERATORS}
    if {str(row.get("name")) for row in single_rollback_rows} != expected:
        raise ValueError("single rollback observations must cover all core operators")
    ordered = sorted(
        single_rollback_rows,
        key=lambda row: (
            float(row["top1_agreement"]),
            float(row["threshold_band_agreement"]),
            float(row["global_top_per_state_overlap"]),
            float(row["topk_set_overlap"]),
            str(row["name"]),
        ),
        reverse=True,
    )
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
        self.capture_statistics = False
        self.statistics: dict[str, list[dict[str, float | int]]] = {}
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

    def set_precision(self, policy: Mapping[str, str]) -> None:
        if set(policy) != set(CORE_OPERATORS):
            raise ValueError("precision policy must contain all 16 core operators")
        if any(value not in ("fp16", "sm120_block_fp8") for value in policy.values()):
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
        if self.precision[name] == "sm120_block_fp8":
            x, _ = fake_sm120_activation_quant(x.float())
            x = x.to(dtype=self.dtype)
            weight_hxk = self.quantized_weights[name]
        return x.matmul(weight_hxk) + bias

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
            y = F.silu(self.project_named(
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
    metrics = ranking_metrics(-baseline, -candidate, top_k=8, threshold_band=0.25)
    metrics["global_top_per_state_overlap"] = _global_overlap(baseline, candidate, baseline.shape[0])
    metrics["wall_seconds"] = elapsed
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-states", type=int, default=4096)
    parser.add_argument("--split", choices=("all", "calibration", "holdout"), default="all")
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
    model = QuantObservedPieceTransformer(args.weight_dir, device)
    fp16 = {name: "fp16" for name in CORE_OPERATORS}
    model.set_precision(fp16)
    baseline, baseline_sec = _run_logits(
        model, states, batch_size=args.batch_size, capture_statistics=True
    )
    activation_stats = {
        name: _aggregate_statistics(rows) for name, rows in model.statistics.items()
    }
    np.save(args.output_dir / "baseline_logits.npy", baseline)
    (args.output_dir / "activation_statistics.json").write_text(
        json.dumps(activation_stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    observations: list[dict[str, Any]] = [{
        "name": "all_fp16", "wall_seconds": baseline_sec,
        "top1_agreement": 1.0, "topk_set_overlap": 1.0,
        "pair_inversion_rate": 0.0, "threshold_band_agreement": 1.0,
        "global_top_per_state_overlap": 1.0,
    }]
    policies = build_initial_mixed_precision_policies()
    for name, policy in policies:
        model.set_precision(policy)
        candidate, elapsed = _run_logits(
            model, states, batch_size=args.batch_size, capture_statistics=False
        )
        row = {"name": name, **_metrics(baseline, candidate, elapsed)}
        observations.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
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
            **_metrics(baseline, candidate, elapsed),
        }
        observations.append(row)
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
    print(json.dumps({"output_dir": str(args.output_dir), "candidates": len(observations)}))


if __name__ == "__main__":
    main()
