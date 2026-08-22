from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.sm120_quant_tuner import (
    CORE_OPERATORS,
    QualityThresholds,
    build_profile,
    fold_ffn_equalization,
    fold_layernorm_linear_equalization,
    fold_qk_reciprocal_equalization,
    fold_v_output_equalization,
    pareto_select,
    ranking_metrics,
    smoothquant_scales,
    validate_profile,
    write_immutable_profile,
)


def test_ranking_metrics_detect_top24_and_threshold_changes() -> None:
    baseline = np.stack([np.arange(24, dtype=np.float64), np.arange(24, dtype=np.float64)])
    candidate = baseline.copy()
    candidate[0, [22, 23]] = candidate[0, [23, 22]]
    metrics = ranking_metrics(baseline, candidate, top_k=24, threshold_band=1.5)
    assert metrics["top1_agreement"] == 0.5
    assert metrics["topk_set_overlap"] == 1.0
    assert 0.0 < metrics["pair_inversion_rate"] < 0.01
    assert metrics["threshold_band_agreement"] < 1.0


def test_profile_validation_is_fail_closed() -> None:
    profile = build_profile(
        checkpoint_sha256="a" * 64,
        model_metadata_sha256="b" * 64,
        calibration_sha256="c" * 64,
        gpu_identity="RTX PRO 6000 Blackwell|sm120",
        cutlass_commit="d" * 40,
        operator_precision={name: "sm120_block_fp8" for name in CORE_OPERATORS},
    )
    assert validate_profile(profile) == profile
    broken = json.loads(json.dumps(profile))
    del broken["operators"][CORE_OPERATORS[0]]
    with pytest.raises(ValueError, match="exactly the 16 core operators"):
        validate_profile(broken)
    broken = json.loads(json.dumps(profile))
    broken["operators"][CORE_OPERATORS[0]]["activation_dtype"] = "e5m2"
    with pytest.raises(ValueError, match="unsupported activation dtype"):
        validate_profile(broken)


def test_pareto_selection_applies_quality_as_hard_gate() -> None:
    thresholds = QualityThresholds(
        top1_agreement=0.999,
        topk_set_overlap=0.999,
        frontier_jaccard=0.995,
        threshold_band_agreement=0.999,
    )
    rows = [
        {"name": "fast_bad", "latency_ms": 1.0, "top1_agreement": 0.95,
         "topk_set_overlap": 1.0, "frontier_jaccard": 1.0,
         "threshold_band_agreement": 1.0},
        {"name": "safe", "latency_ms": 1.4, "top1_agreement": 1.0,
         "topk_set_overlap": 1.0, "frontier_jaccard": 0.999,
         "threshold_band_agreement": 1.0},
        {"name": "safe_slow", "latency_ms": 1.8, "top1_agreement": 1.0,
         "topk_set_overlap": 1.0, "frontier_jaccard": 1.0,
         "threshold_band_agreement": 1.0},
    ]
    result = pareto_select(rows, thresholds)
    assert result["selected"]["name"] == "safe"
    assert result["rejected"]["fast_bad"] == ["top1_agreement"]


def test_immutable_writer_hashes_every_artifact(tmp_path: Path) -> None:
    profile = build_profile(
        checkpoint_sha256="a" * 64,
        model_metadata_sha256="b" * 64,
        calibration_sha256="c" * 64,
        gpu_identity="RTX PRO 6000 Blackwell|sm120",
        cutlass_commit="d" * 40,
        operator_precision={name: "fp16" for name in CORE_OPERATORS},
    )
    output = write_immutable_profile(
        tmp_path / "profile",
        profile,
        artifacts={"ranking_metrics.json": {"top1_agreement": 1.0}},
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest["files"]) == {"profile.json", "ranking_metrics.json"}
    with pytest.raises(FileExistsError):
        write_immutable_profile(output, profile, artifacts={})


def test_graph_preserving_equalization_transforms() -> None:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(9, 8, generator=generator)
    gamma = torch.randn(8, generator=generator)
    beta = torch.randn(8, generator=generator)
    weight = torch.randn(12, 8, generator=generator)
    scales = torch.exp(torch.linspace(-0.7, 0.7, 8))
    baseline = (x * gamma + beta) @ weight.T
    gamma_q, beta_q, weight_q = fold_layernorm_linear_equalization(
        gamma, beta, weight, scales
    )
    torch.testing.assert_close((x * gamma_q + beta_q) @ weight_q.T, baseline)

    ff1_weight = torch.randn(16, 8, generator=generator)
    ff1_bias = torch.randn(16, generator=generator)
    ff2_weight = torch.randn(8, 16, generator=generator)
    hidden_scales = torch.exp(torch.linspace(-0.5, 0.5, 16))
    baseline = torch.relu(x @ ff1_weight.T + ff1_bias) @ ff2_weight.T
    w1, b1, w2 = fold_ffn_equalization(
        ff1_weight, ff1_bias, ff2_weight, hidden_scales
    )
    torch.testing.assert_close(torch.relu(x @ w1.T + b1) @ w2.T, baseline, atol=2e-5, rtol=2e-5)

    q = torch.randn(9, 8, generator=generator)
    k = torch.randn(9, 8, generator=generator)
    q_scaled, k_scaled = fold_qk_reciprocal_equalization(q, k, scales)
    torch.testing.assert_close(q_scaled @ k_scaled.T, q @ k.T, atol=2e-5, rtol=2e-5)

    attention = torch.softmax(torch.randn(9, 9, generator=generator), dim=-1)
    value = torch.randn(9, 8, generator=generator)
    output_weight = torch.randn(8, 8, generator=generator)
    baseline = (attention @ value) @ output_weight.T
    value_q, output_q = fold_v_output_equalization(value, output_weight, scales)
    torch.testing.assert_close((attention @ value_q) @ output_q.T, baseline, atol=2e-5, rtol=2e-5)


def test_smoothquant_scales_are_positive_bounded_and_deterministic() -> None:
    activation_amax = torch.tensor([1.0, 16.0, 0.0, 4.0])
    weight_amax = torch.tensor([4.0, 1.0, 2.0, 0.0])
    scales = smoothquant_scales(
        activation_amax, weight_amax, alpha=0.5, minimum=0.25, maximum=4.0
    )
    torch.testing.assert_close(scales, torch.tensor([0.5, 4.0, 0.25, 2.0]))
    assert torch.isfinite(scales).all() and torch.all(scales > 0)
