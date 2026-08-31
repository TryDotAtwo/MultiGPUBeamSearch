"""Integration with CayleyPy's public hook; other tests also cover legacy releases."""
import pytest
from cayleypy import CayleyGraph, PermutationGroups
import cayleypy
import cayleypy_native.dispatch as dispatch
from cayleypy_native import NativeOptions, NativeUnavailable, NativeFallbackWarning

pytestmark = pytest.mark.skipif(dispatch._registry_api() is None, reason="CayleyPy predates public backend hook")


@pytest.fixture(autouse=True)
def clean_registration():
    dispatch.disable_native()
    yield
    dispatch.disable_native()


def graph():
    return CayleyGraph(PermutationGroups.lrx(5), device="cpu", random_seed=42)


def test_enable_keeps_method_and_restores_previous_backend(tmp_path):
    original = CayleyGraph.beam_search
    sentinel = object()
    cayleypy.register_beam_search_backend("another", lambda g, **kw: sentinel)
    cayleypy.set_default_beam_search_backend("another")
    try:
        dispatch.enable_native(NativeOptions(cache_dir=tmp_path))
        assert CayleyGraph.beam_search is original
        assert cayleypy.get_default_beam_search_backend() == "auto"
        with pytest.warns(NativeFallbackWarning):
            assert graph().beam_search(start_state=[1, 0, 2, 3, 4], return_path=True).path == [2]
        dispatch.disable_native()
        assert graph().beam_search() is sentinel
        assert CayleyGraph.beam_search is original
    finally:
        cayleypy.set_default_beam_search_backend("torch")
        cayleypy.unregister_beam_search_backend("another")


def test_reenable_updates_options_without_recursive_fallback(tmp_path):
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path), default_backend="native")
    with pytest.raises(NativeUnavailable):
        graph().beam_search(start_state=[1, 0, 2, 3, 4])
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path), default_backend="auto")
    with pytest.warns(NativeFallbackWarning):
        assert graph().beam_search(start_state=[1, 0, 2, 3, 4], return_path=True).path == [2]


def test_collision_rolls_back_earlier_registration(tmp_path):
    callback = lambda g, **kw: None
    cayleypy.register_beam_search_backend("auto", callback)
    try:
        with pytest.raises(ValueError, match="already registered"):
            dispatch.enable_native(NativeOptions(cache_dir=tmp_path))
        assert cayleypy.get_beam_search_backend("auto") is callback
        assert cayleypy.get_default_beam_search_backend() == "torch"
        with pytest.raises(ValueError, match="unknown"):
            cayleypy.get_beam_search_backend("native")
    finally:
        cayleypy.unregister_beam_search_backend("auto")


def test_disable_does_not_overwrite_another_default(tmp_path):
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path))
    cayleypy.set_default_beam_search_backend("torch")
    try:
        with pytest.raises(RuntimeError, match="another integration"):
            dispatch.disable_native()
        assert cayleypy.get_default_beam_search_backend() == "torch"
    finally:
        cayleypy.set_default_beam_search_backend("auto")


def test_explicit_torch_uses_only_upstream_contract(tmp_path, monkeypatch):
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path))
    monkeypatch.setattr(dispatch, "runtime_devices", lambda *a: pytest.fail("native preflight entered"))
    assert graph().beam_search(backend="torch", start_state=[1, 0, 2, 3, 4], return_path=True).path == [2]
