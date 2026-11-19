import json

import pytest

from multigpubeamsearch import autotune, rust_control
from multigpubeamsearch.errors import NativeUnavailable


def _native_extension():
    try:
        return rust_control.extension()
    except NativeUnavailable:
        pytest.skip("Rust extension is not built in this CPU job")


@pytest.mark.parametrize(
    "previous,current,expected",
    [
        ({"gpu-a": 1024}, {"gpu-a": 1056}, True),
        ({"gpu-a": 1024}, {"gpu-a": 1057}, False),
        ({"gpu-a": 2048}, {"gpu-a": 1000}, True),
        ({"gpu-a": 1024, "gpu-b": 2048}, {"gpu-a": 1056, "gpu-b": 2080}, True),
        ({"gpu-a": 1024}, {"gpu-b": 1024}, False),
        ({}, {"gpu-a": 1024}, False),
        ({"gpu-a": 1024}, None, False),
        (None, {"gpu-a": 1024}, False),
    ],
)
def test_rust_capacity_cache_boundary_matches_existing_policy(previous, current, expected):
    native = _native_extension()
    builder = getattr(native, "inference_capacity_cache_reusable", None)
    assert callable(builder), "Rust capacity-cache reuse decision is missing"
    assert builder(json.dumps(previous), json.dumps(current)) is expected


def test_public_capacity_cache_policy_uses_rust_after_memory_snapshot(monkeypatch):
    previous = {"gpu-a": 1024}
    current = {"gpu-a": 1056}
    observed = []

    monkeypatch.setattr(autotune, "free_memory_snapshot", lambda signature: current)

    def rust_decision(old, now):
        observed.append((old, now))
        return True

    monkeypatch.setattr(
        rust_control, "inference_capacity_cache_reusable", rust_decision, raising=False
    )
    data = {"capacity_limited_search": True, "initial_free_memory_mib": previous}
    assert autotune.capacity_cache_reusable(data, {"gpu_uuids": ["gpu-a"]}) is True
    assert observed == [(previous, current)]


def test_non_capacity_limited_cache_keeps_short_circuit(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("capacity data must not be read")

    monkeypatch.setattr(autotune, "free_memory_snapshot", unexpected)
    monkeypatch.setattr(rust_control, "inference_capacity_cache_reusable", unexpected, raising=False)
    assert autotune.capacity_cache_reusable({"capacity_limited_search": False}, {}) is True


def test_python_capacity_cache_policy_remains_fallback(monkeypatch):
    monkeypatch.setattr(
        rust_control,
        "inference_capacity_cache_reusable",
        lambda previous, current: None,
        raising=False,
    )
    previous = {"gpu-a": 1024, "gpu-b": 2048}
    current = {"gpu-a": 1056, "gpu-b": 2049}
    monkeypatch.setattr(autotune, "free_memory_snapshot", lambda signature: current)
    data = {"capacity_limited_search": True, "initial_free_memory_mib": previous}
    assert autotune.capacity_cache_reusable(data, {}) is True

    changed_keys = {"gpu-a": 1056, "gpu-c": 2049}
    monkeypatch.setattr(autotune, "free_memory_snapshot", lambda signature: changed_keys)
    assert autotune.capacity_cache_reusable(data, {}) is False
