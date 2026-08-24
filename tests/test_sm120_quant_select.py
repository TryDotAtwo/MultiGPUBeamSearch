from __future__ import annotations

import pytest

from tools.sm120_quant_select import select_profile
from tools.sm120_quant_tuner import CORE_OPERATORS, QualityThresholds


def _fp16_reference(score: float = 0.996) -> dict:
    return {
        "name": "current_fp16",
        "top1_agreement": score,
        "topk_set_overlap": score,
        "threshold_band_agreement": score,
    }


def _ranking(name: str, score: float = 1.0, fp32_score: float = 0.996) -> dict:
    return {
        "name": name,
        "operator_precision": {operator: "fp16" for operator in CORE_OPERATORS},
        "vs_fp16": {
            "top1_agreement": score,
            "topk_set_overlap": score,
            "threshold_band_agreement": score,
        },
        "vs_fp32": {
            "top1_agreement": fp32_score,
            "topk_set_overlap": fp32_score,
            "threshold_band_agreement": fp32_score,
        },
    }


def _frontier(name: str, score: float = 1.0) -> dict:
    return {"name": name, "depths": [{"depth": 8, "jaccard": score}]}


def _benchmark(name: str, latency_ms: float, backend: str = "cublaslt") -> dict:
    return {
        "name": name,
        "latency_ms": latency_ms,
        "native_execution": {
            "fp16_gemm_backend": backend,
            "target_sm": 120,
            "workspace_bytes": 0,
        },
    }


def test_selector_requires_all_three_artifacts_and_uses_native_latency() -> None:
    thresholds = QualityThresholds()
    decision = select_profile(
        [_fp16_reference(), _ranking("fast"), _ranking("missing"), _ranking("bad", 0.9)],
        [_frontier("fast"), _frontier("missing"), _frontier("bad")],
        [_benchmark("fast", 2.0), _benchmark("bad", 1.0)],
        thresholds,
    )
    assert decision["selected"]["name"] == "fast"
    assert decision["selected"]["latency_ms"] == 2.0
    assert decision["missing_evidence"]["missing"] == ["benchmark"]
    assert "bad" in decision["rejected"]


def test_selector_uses_worst_reconstructed_frontier_depth() -> None:
    decision = select_profile(
        [_fp16_reference(), _ranking("candidate")],
        [{"name": "candidate", "depths": [
            {"depth": 7, "jaccard": 1.0}, {"depth": 8, "jaccard": 0.9}
        ]}],
        [_benchmark("candidate", 1.0)],
        QualityThresholds(frontier_jaccard=0.995),
    )
    assert decision["selected"] is None
    assert decision["rejected"]["candidate"] == ["frontier_jaccard"]


def test_selector_rejects_latency_without_native_execution_contract() -> None:
    decision = select_profile(
        [_fp16_reference(), _ranking("candidate")], [_frontier("candidate")],
        [{"name": "candidate", "latency_ms": 1.0}], QualityThresholds(),
    )
    assert decision["selected"] is None
    assert decision["missing_evidence"]["candidate"] == ["native_execution"]


def test_selector_rejects_low_rank_candidate_until_native_runtime_exists() -> None:
    ranking = _ranking("candidate")
    ranking["operator_corrections"] = {
        CORE_OPERATORS[0]: {
            "type": "activation_weighted_low_rank_residual", "rank": 16,
        }
    }
    decision = select_profile(
        [_fp16_reference(), ranking], [_frontier("candidate")], [_benchmark("candidate", 1.0)],
        QualityThresholds(),
    )
    assert decision["selected"] is None
    assert decision["missing_evidence"]["candidate"] == [
        "operator_corrections.native_runtime"
    ]


def test_selector_uses_fp16_as_floor_while_fp32_remains_truth() -> None:
    thresholds = QualityThresholds(fp32_regression_budget=0.001)
    decision = select_profile(
        [
            _fp16_reference(0.996),
            _ranking("within_budget", score=0.9995, fp32_score=0.9951),
            _ranking("below_floor", score=1.0, fp32_score=0.9949),
        ],
        [_frontier("within_budget"), _frontier("below_floor")],
        [_benchmark("within_budget", 2.0), _benchmark("below_floor", 1.0)],
        thresholds,
    )
    assert decision["selected"]["name"] == "within_budget"
    assert decision["fp16_quality_floor"]["top1_agreement"] == pytest.approx(0.995)
    assert "fp32_top1_regression" in decision["rejected"]["below_floor"]


def test_selector_fails_closed_without_fp16_reference() -> None:
    with pytest.raises(ValueError, match="current_fp16"):
        select_profile(
            [_ranking("candidate")], [_frontier("candidate")],
            [_benchmark("candidate", 1.0)], QualityThresholds(),
        )
