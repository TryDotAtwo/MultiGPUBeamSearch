from __future__ import annotations

from tools.sm120_quant_select import select_profile
from tools.sm120_quant_tuner import CORE_OPERATORS, QualityThresholds


def _ranking(name: str, score: float = 1.0) -> dict:
    return {
        "name": name,
        "operator_precision": {operator: "fp16" for operator in CORE_OPERATORS},
        "top1_agreement": score,
        "topk_set_overlap": score,
        "threshold_band_agreement": score,
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
        [_ranking("fast"), _ranking("missing"), _ranking("bad", 0.9)],
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
        [_ranking("candidate")],
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
        [_ranking("candidate")], [_frontier("candidate")],
        [{"name": "candidate", "latency_ms": 1.0}], QualityThresholds(),
    )
    assert decision["selected"] is None
    assert decision["missing_evidence"]["candidate"] == ["native_execution"]
