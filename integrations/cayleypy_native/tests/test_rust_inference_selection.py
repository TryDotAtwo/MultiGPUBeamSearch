import json

import pytest

from multigpubeamsearch import autotune, rust_control
from multigpubeamsearch.calibration_stats import Estimate, Measurement, estimate, select
from multigpubeamsearch.errors import NativeBackendError, NativeUnavailable


def _native_extension():
    try:
        return rust_control.extension()
    except NativeUnavailable:
        pytest.skip("Rust extension is not built in this CPU job")


def _measurements(baseline_values, candidate_values):
    result = []
    for profile, values in (("baseline", baseline_values), ("candidate", candidate_values)):
        for seconds in values:
            result.append(
                Measurement(
                    profile=profile,
                    parents=512,
                    seconds_by_rank=(seconds, seconds * 1.05),
                    correctness_passed=True,
                    memory_admitted=True,
                )
            )
    return result


def _python_reference(samples, baseline):
    rejected = {}
    try:
        estimate(baseline, samples)
    except ValueError as error:
        rejected[baseline] = str(error)
        valid = []
        for profile in sorted({row.profile for row in samples} - {baseline}):
            try:
                valid.append(estimate(profile, samples))
            except ValueError as candidate_error:
                rejected[profile] = str(candidate_error)
        if not valid:
            raise NativeBackendError("no stable complete inference candidate")
        baseline = min(valid, key=lambda row: (row.median, row.profile)).profile
    winner, other_rejected = select(samples, baseline=baseline)
    return winner, dict(rejected, **other_rejected)


@pytest.mark.parametrize(
    "baseline_values,candidate_values,baseline",
    [
        ([0.98, 0.99, 1.00, 1.01, 1.02], [0.68, 0.69, 0.70, 0.71, 0.72], "baseline"),
        ([0.98, 0.99, 1.00, 1.01, 1.02], [0.98, 0.99, 1.00, 1.01, 1.02], "baseline"),
        ([0.50, 0.70, 0.90, 1.20, 1.40], [0.78, 0.79, 0.80, 0.81, 0.82], "baseline"),
        ([0.98, 0.99, 1.00, 1.01, 1.02], [0.70, 0.71, 0.72, 0.73], "baseline"),
    ],
)
def test_rust_inference_selector_matches_python_reference(
    baseline_values, candidate_values, baseline
):
    native = _native_extension()
    builder = getattr(native, "inference_select_compatible", None)
    assert callable(builder), "Rust-compatible inference profile selector is missing"
    samples = _measurements(baseline_values, candidate_values)
    request = {
        "baseline": baseline,
        "samples": [
            {
                "profile": row.profile,
                "parents": row.parents,
                "seconds_by_rank": list(row.seconds_by_rank),
                "correctness_passed": row.correctness_passed,
                "memory_admitted": row.memory_admitted,
                "throttled": row.throttled,
            }
            for row in samples
        ],
    }
    selected, rejected = json.loads(builder(json.dumps(request, allow_nan=False)))
    actual = Estimate(**selected)
    assert (actual, rejected) == _python_reference(samples, baseline)


def test_public_inference_selector_uses_rust_adapter(monkeypatch):
    expected = (Estimate("candidate", 1.0, 0.9, 1.1, 0.01, 5), {"baseline": "unstable"})
    observed = []
    monkeypatch.setattr(
        rust_control,
        "inference_select_compatible",
        lambda samples, baseline: observed.append((samples, baseline)) or expected,
        raising=False,
    )
    samples = _measurements([0.9, 0.95, 1.0, 1.05, 1.1], [0.8, 0.81, 0.82, 0.83, 0.84])
    assert autotune.select_inference(samples, baseline="baseline") == expected
    assert observed == [(samples, "baseline")]


def test_python_inference_selector_remains_fallback(monkeypatch):
    monkeypatch.setattr(
        rust_control,
        "inference_select_compatible",
        lambda samples, baseline: None,
        raising=False,
    )
    samples = _measurements([0.98, 0.99, 1.00, 1.01, 1.02], [0.68, 0.69, 0.70, 0.71, 0.72])
    assert autotune.select_inference(samples, baseline="baseline") == _python_reference(
        samples, "baseline"
    )
