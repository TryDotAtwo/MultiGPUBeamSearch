import random

import pytest

from multigpubeamsearch import autotune, rust_control
from multigpubeamsearch.errors import NativeUnavailable


def _native_extension():
    try:
        return rust_control.extension()
    except NativeUnavailable:
        pytest.skip("Rust extension is not built in this CPU job")


def _legacy_coarse_batches(cap):
    if type(cap) is not int or not 1 <= cap <= 65536:
        raise ValueError("inference cap must be in [1,65536]")
    baseline = min(256, cap)
    values = [x for x in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
                          16384, 32768, 65536) if x <= cap and x != baseline]
    larger = [x for x in values if x >= baseline and x != cap]
    smaller = [x for x in values if x < baseline]
    random.Random(20261009).shuffle(larger)
    random.Random(20261009).shuffle(smaller)
    return [baseline] + ([cap] if cap != baseline else []) + larger + smaller


def _legacy_refinement_batches(winner, cap, visited):
    if type(winner) is not int or not 1 <= winner <= cap:
        raise ValueError("inference winner must fit its cap")
    grain = max(1, min(32, winner // 8))
    values = set()
    for numerator, denominator in ((5, 8), (3, 4), (7, 8), (9, 8), (5, 4), (3, 2)):
        batch = ((winner * numerator + denominator * grain // 2)
                 // (denominator * grain)) * grain
        if 1 <= batch <= cap and batch not in visited:
            values.add(batch)
    result = sorted(values)
    random.Random(20261010).shuffle(result)
    return result


@pytest.mark.parametrize("cap", [1, 17, 256, 1000, 1536, 8192, 65536])
def test_rust_coarse_order_matches_existing_python_policy(cap):
    native = _native_extension()
    builder = getattr(native, "inference_coarse_batches", None)
    assert callable(builder), "Rust coarse inference candidate builder is missing"
    assert builder(cap) == _legacy_coarse_batches(cap)


@pytest.mark.parametrize(
    "winner,cap,visited",
    [
        (1024, 1536, {32, 64, 128, 256, 512, 1024, 1536}),
        (1, 1, {1}),
        (37, 90, {32, 37}),
        (4095, 65536, {4095, 8192}),
    ],
)
def test_rust_refinement_order_matches_existing_python_policy(winner, cap, visited):
    native = _native_extension()
    builder = getattr(native, "inference_refinement_batches", None)
    assert callable(builder), "Rust inference refinement builder is missing"
    assert builder(winner, cap, sorted(visited)) == _legacy_refinement_batches(
        winner, cap, visited
    )


def test_public_coarse_policy_uses_rust_builder(monkeypatch):
    expected = [8192, 256]
    monkeypatch.setattr(
        rust_control, "inference_coarse_batches", lambda cap: expected, raising=False
    )
    assert autotune.coarse_batches(8192) == expected


def test_public_refinement_policy_uses_rust_builder(monkeypatch):
    expected = [800]
    monkeypatch.setattr(
        rust_control, "inference_refinement_batches",
        lambda winner, cap, visited: expected, raising=False,
    )
    assert autotune.refinement_batches(1024, 1536, {32, 64}) == expected


def test_python_candidate_policy_remains_as_fallback(monkeypatch):
    monkeypatch.setattr(
        rust_control, "inference_coarse_batches", lambda cap: None, raising=False
    )
    monkeypatch.setattr(
        rust_control, "inference_refinement_batches",
        lambda winner, cap, visited: None, raising=False,
    )
    assert autotune.coarse_batches(8192) == _legacy_coarse_batches(8192)
    assert autotune.refinement_batches(1024, 1536, {32, 64}) == (
        _legacy_refinement_batches(1024, 1536, {32, 64})
    )