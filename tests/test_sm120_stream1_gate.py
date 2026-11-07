from __future__ import annotations

import pytest

from tools.sm120_stream1_gate import evaluate_stream1_gate


def _trials(candidate_ms: list[float], baseline_ms: list[float] | None = None) -> list[dict]:
    baseline = baseline_ms or [1.50] * len(candidate_ms)
    result: list[dict] = []
    for index, (control, candidate) in enumerate(zip(baseline, candidate_ms, strict=True)):
        order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        result.append({"order": list(order), "baseline_ms": control, "candidate_ms": candidate})
    return result


def test_accepts_full_stream1_latency_and_stage_budget() -> None:
    report = evaluate_stream1_gate(
        trials=_trials([0.320, 0.322, 0.324, 0.326, 0.328, 0.330, 0.332]),
        stage_ms={"encode_ln": 0.020, "qkv": 0.060, "fa4": 0.080,
                  "attention_out": 0.030, "ffn": 0.070, "final_head": 0.030},
        batch=896,
        concurrency=1,
    )
    assert report["status"] == "accepted"
    assert report["trial_count"] == 7
    assert report["stage_sum_ms"] == pytest.approx(0.29)
    assert report["dense_equivalent_tflops"] > 1000.0


@pytest.mark.parametrize(
    ("candidate_ms", "stage_ms", "reason"),
    [
        ([0.34] * 7, {"total": 0.29}, "median_latency"),
        ([0.30] * 6 + [0.40], {"total": 0.29}, "p95_latency"),
        ([0.30] * 7, {"total": 0.301}, "stage_budget"),
    ],
)
def test_rejects_quantitative_gate_failures(
    candidate_ms: list[float], stage_ms: dict[str, float], reason: str
) -> None:
    report = evaluate_stream1_gate(
        trials=_trials(candidate_ms), stage_ms=stage_ms, batch=896, concurrency=1
    )
    assert report["status"] == "rejected"
    assert reason in report["reasons"]


def test_requires_seven_strictly_alternating_pairs() -> None:
    with pytest.raises(ValueError, match="at least seven"):
        evaluate_stream1_gate(
            trials=_trials([0.30] * 6), stage_ms={"total": 0.20}, batch=896, concurrency=1
        )
    invalid = _trials([0.30] * 7)
    invalid[3]["order"] = ["baseline", "candidate"]
    with pytest.raises(ValueError, match="alternate"):
        evaluate_stream1_gate(
            trials=invalid, stage_ms={"total": 0.20}, batch=896, concurrency=1
        )


def test_canonical_shape_is_explicit() -> None:
    with pytest.raises(ValueError, match="B=896, concurrency=1"):
        evaluate_stream1_gate(
            trials=_trials([0.30] * 7), stage_ms={"total": 0.20}, batch=1024, concurrency=1
        )

