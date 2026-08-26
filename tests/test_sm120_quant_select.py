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
    return {
        "name": name,
        "fixture_sha256": "a" * 64,
        "depths": [{"depth": 8, "jaccard": score}],
    }


def _benchmark(name: str, depth8_seconds: float, backend: str = "cublaslt") -> dict:
    return {
        "name": name,
        "workload": {
            "puzzle_family": "cube4",
            "output_dim": 24,
            "depth_start": 0,
            "depth_limit": 8,
            "beam_width": 1 << 25,
            "fixture_sha256": "a" * 64,
        },
        "timing": {
            "depth8_seconds": depth8_seconds,
            "depth8_samples": [depth8_seconds - 0.1, depth8_seconds, depth8_seconds + 0.1],
            "statistic": "median",
        },
        "native_execution": {
            "fp16_gemm_backend": backend,
            "target_sm": 120,
            "workspace_bytes": 0,
            "kernel_contract": (
                "stream1_fp16_control_v1" if name == "current_fp16"
                else "stream1_sm120_nvfp4_fused_ffn_v1"
            ),
            "kernel_sha256": "b" * 64,
        },
    }


def _benchmarks(*rows: dict) -> list[dict]:
    return [_benchmark("current_fp16", 80.2952), *rows]


def test_selector_requires_all_three_artifacts_and_uses_native_latency() -> None:
    thresholds = QualityThresholds()
    decision = select_profile(
        [_fp16_reference(), _ranking("fast"), _ranking("missing"), _ranking("bad", 0.9)],
        [_frontier("fast"), _frontier("missing"), _frontier("bad")],
        _benchmarks(_benchmark("fast", 70.0), _benchmark("bad", 60.0)),
        thresholds,
    )
    assert decision["selected"]["name"] == "fast"
    assert decision["selected"]["depth8_seconds"] == 70.0
    assert decision["missing_evidence"]["missing"] == ["benchmark"]
    assert "bad" in decision["rejected"]


def test_selector_uses_worst_reconstructed_frontier_depth() -> None:
    decision = select_profile(
        [_fp16_reference(), _ranking("candidate")],
        [{"name": "candidate", "fixture_sha256": "a" * 64, "depths": [
            {"depth": 7, "jaccard": 1.0}, {"depth": 8, "jaccard": 0.9}
        ]}],
        _benchmarks(_benchmark("candidate", 70.0)),
        QualityThresholds(frontier_jaccard=0.995),
    )
    assert decision["selected"] is None
    assert decision["rejected"]["candidate"] == ["frontier_jaccard"]


def test_selector_rejects_latency_without_native_execution_contract() -> None:
    decision = select_profile(
        [_fp16_reference(), _ranking("candidate")], [_frontier("candidate")],
        _benchmarks({"name": "candidate", "timing": {}}), QualityThresholds(),
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
        [_fp16_reference(), ranking], [_frontier("candidate")],
        _benchmarks(_benchmark("candidate", 70.0)),
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
        _benchmarks(
            _benchmark("within_budget", 70.0), _benchmark("below_floor", 60.0)
        ),
        thresholds,
    )
    assert decision["selected"]["name"] == "within_budget"
    assert decision["fp16_quality_floor"]["top1_agreement"] == pytest.approx(0.995)
    assert "fp32_top1_regression" in decision["rejected"]["below_floor"]


def test_selector_fails_closed_without_fp16_reference() -> None:
    with pytest.raises(ValueError, match="current_fp16"):
        select_profile(
            [_ranking("candidate")], [_frontier("candidate")],
            _benchmarks(_benchmark("candidate", 70.0)), QualityThresholds(),
        )


def test_selector_rejects_quality_candidate_outside_native_operator_subset() -> None:
    ranking = _ranking("unsupported")
    ranking["operator_precision"]["blocks.0.attn.out_proj.weight"] = "sm120_block_fp8"
    decision = select_profile(
        [_fp16_reference(), ranking], [_frontier("unsupported")],
        _benchmarks(_benchmark("unsupported", 70.0)), QualityThresholds(),
    )
    assert decision["selected"] is None
    assert decision["missing_evidence"]["unsupported"] == [
        "operator_precision.native_runtime"
    ]


def test_selector_rejects_nvfp4_until_fused_native_runtime_is_verified() -> None:
    ranking = _ranking("nvfp4_unverified")
    ranking["operator_precision"]["blocks.0.ff.0.weight"] = "sm120_nvfp4"
    decision = select_profile(
        [_fp16_reference(), ranking], [_frontier("nvfp4_unverified")],
        _benchmarks(_benchmark("nvfp4_unverified", 70.0)), QualityThresholds(),
    )
    assert decision["selected"] is None
    assert decision["missing_evidence"]["nvfp4_unverified"] == [
        "operator_precision.native_runtime"
    ]


def test_selector_rejects_candidate_without_real_depth8_speedup() -> None:
    decision = select_profile(
        [_fp16_reference(), _ranking("slower")], [_frontier("slower")],
        _benchmarks(_benchmark("slower", 80.4)), QualityThresholds(),
    )
    assert decision["selected"] is None
    assert decision["rejected"]["slower"] == ["not_faster_than_current_fp16"]


def test_selector_rejects_different_frontier_fixture() -> None:
    candidate = _benchmark("candidate", 70.0)
    candidate["workload"]["fixture_sha256"] = "c" * 64
    decision = select_profile(
        [_fp16_reference(), _ranking("candidate")], [_frontier("candidate")],
        _benchmarks(candidate), QualityThresholds(),
    )
    assert decision["selected"] is None
    assert decision["missing_evidence"]["candidate"] == ["workload.fixture_sha256"]


def test_selector_rejects_frontier_from_different_fixture() -> None:
    frontier = _frontier("candidate")
    frontier["fixture_sha256"] = "c" * 64
    decision = select_profile(
        [_fp16_reference(), _ranking("candidate")], [frontier],
        _benchmarks(_benchmark("candidate", 70.0)), QualityThresholds(),
    )
    assert decision["selected"] is None
    assert decision["missing_evidence"]["candidate"] == ["frontier.fixture_sha256"]
