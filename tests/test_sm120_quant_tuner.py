from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.sm120_quant_tuner import (
    CORE_OPERATORS,
    QualityThresholds,
    build_profile,
    pareto_select,
    ranking_metrics,
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
