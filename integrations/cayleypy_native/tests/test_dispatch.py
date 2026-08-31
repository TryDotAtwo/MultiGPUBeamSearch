import warnings
from types import SimpleNamespace

import pytest
import torch
from cayleypy import CayleyGraph, PermutationGroups, Predictor
from cayleypy.algo.beam_search_result import BeamSearchResult

import cayleypy_native.dispatch as dispatch
from cayleypy_native.errors import NativeBackendError, NativeFallbackWarning, NativeUnavailable
from cayleypy_native.options import NativeOptions, NativeOutcome


@pytest.fixture(autouse=True)
def clean_install():
    dispatch.disable_native()
    yield
    dispatch.disable_native()


def graph():
    return CayleyGraph(PermutationGroups.lrx(5), device="cpu", random_seed=42)


def test_import_has_no_patch_and_enable_is_reversible(tmp_path):
    original = CayleyGraph.beam_search
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path))
    installed = CayleyGraph.beam_search
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path))
    assert CayleyGraph.beam_search is installed
    dispatch.disable_native()
    assert CayleyGraph.beam_search is original


def test_auto_fallback_preserves_original_result_and_reason(tmp_path, monkeypatch):
    def unavailable(*args):
        raise NativeUnavailable("test: no compatible native runtime")
    monkeypatch.setattr(dispatch, "runtime_devices", unavailable)
    g = graph()
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path))
    with pytest.warns(NativeFallbackWarning, match="no compatible native runtime"):
        result = g.beam_search(start_state=[1, 0, 2, 3, 4], return_path=True)
    assert isinstance(result, BeamSearchResult)
    assert result.path == [2]
    assert torch.equal(g.apply_path([1, 0, 2, 3, 4], result.path).reshape(-1), g.central_state)


def test_strict_native_never_falls_back(tmp_path, monkeypatch):
    def unavailable(*args):
        raise NativeUnavailable("no CUDA")
    monkeypatch.setattr(dispatch, "runtime_devices", unavailable)
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path))
    with pytest.raises(NativeUnavailable, match="no CUDA"):
        graph().beam_search(start_state=[1, 0, 2, 3, 4], backend="native")


def test_explicit_torch_never_probes_native(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("should not probe native")
    monkeypatch.setattr(dispatch, "runtime_devices", forbidden)
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("error", NativeFallbackWarning)
        assert graph().beam_search(start_state=[1, 0, 2, 3, 4], backend="torch", return_path=True).path == [2]


def fake_native(monkeypatch, tmp_path, path=(2,)):
    monkeypatch.setattr(dispatch, "runtime_devices", lambda *args: (0,))
    monkeypatch.setattr(dispatch, "prepare_model", lambda *args: SimpleNamespace(artifact_hash="model-sha", manifest={"output_dim": 1}))
    monkeypatch.setattr(dispatch, "prepare_runtime", lambda *args: SimpleNamespace())
    calls = []
    def run(contract, model, options, beam_width, max_steps, run_dir, devices, *, runtime=None):
        calls.append((contract, beam_width, max_steps, devices))
        return NativeOutcome(path, 0.01, 1024, run_dir, {"fixture": True})
    monkeypatch.setattr(dispatch, "run_native", run)
    return calls


def test_native_result_replays_and_is_upstream_compatible(tmp_path, monkeypatch):
    calls = fake_native(monkeypatch, tmp_path)
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path))
    result = graph().beam_search(start_state=[1, 0, 2, 3, 4], beam_width=123, max_steps=4, return_path=True)
    assert isinstance(result, BeamSearchResult)
    assert result.backend == "native"
    assert result.path == [2] and result.get_path_as_string() == "X"
    assert result.native_metadata["replay_valid"] is True
    assert result.native_metadata["requested_beam_width"] == 123
    assert result.native_metadata["effective_beam_width"] == 1024
    assert calls[0][1:] == (123, 4, (0,))


@pytest.mark.parametrize("return_path", [False, True])
def test_bad_native_replay_is_an_error_even_without_return_path(tmp_path, monkeypatch, return_path):
    fake_native(monkeypatch, tmp_path, path=(0,))
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path))
    with pytest.raises(NativeBackendError, match="replay"):
        graph().beam_search(start_state=[1, 0, 2, 3, 4], return_path=return_path)


def test_crash_after_selection_is_not_hidden_by_auto(tmp_path, monkeypatch):
    fake_native(monkeypatch, tmp_path)
    def crash(*args, **kwargs):
        raise NativeBackendError("worker failed")
    monkeypatch.setattr(dispatch, "run_native", crash)
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path))
    with pytest.raises(NativeBackendError, match="worker failed"):
        graph().beam_search(start_state=[1, 0, 2, 3, 4])


@pytest.mark.parametrize("option", [{"history_depth": 1}, {"beam_mode": "advanced"},
                                   {"destination_state": [1, 0, 2, 3, 4]}, {"bfs_result_for_mitm": object()}])
def test_unsupported_modes_reject_before_native(tmp_path, monkeypatch, option):
    calls = fake_native(monkeypatch, tmp_path)
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path))
    with pytest.raises(NativeUnavailable):
        graph().beam_search(start_state=[1, 0, 2, 3, 4], backend="native", **option)
    assert not calls


def test_invalid_selector_rejected(tmp_path):
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path))
    with pytest.raises(ValueError, match="backend"):
        graph().beam_search(start_state=[1, 0, 2, 3, 4], backend="nativ")


@pytest.mark.parametrize("kwargs", [{"beam_width": 0}, {"beam_width": True}, {"max_steps": -1},
                                   {"beam_width": 2**64}, {"max_steps": 2**32}])
def test_invalid_limits_are_not_fallback(tmp_path, monkeypatch, kwargs):
    fake_native(monkeypatch, tmp_path)
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path))
    with pytest.raises(ValueError):
        graph().beam_search(start_state=[1, 0, 2, 3, 4], **kwargs)


def test_bound_method_survives_disable_and_new_session(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "_registry_api", lambda: None)  # Legacy wrapper semantics.
    monkeypatch.setattr(dispatch, "runtime_devices", lambda *args: (_ for _ in ()).throw(NativeUnavailable("no native")))
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path), default_backend="torch")
    saved = graph().beam_search
    dispatch.disable_native()
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path), default_backend="native")
    assert saved(start_state=[1, 0, 2, 3, 4], return_path=True).path == [2]


def test_unsupported_build_falls_back_before_worker(tmp_path, monkeypatch):
    calls = fake_native(monkeypatch, tmp_path)
    def unsupported(*args):
        raise NativeUnavailable("binary shape mismatch")
    monkeypatch.setattr(dispatch, "prepare_runtime", unsupported)
    with pytest.warns(NativeFallbackWarning, match="shape mismatch"):
        result = dispatch.beam_search(graph(), native_options=NativeOptions(cache_dir=tmp_path),
                                      start_state=[1, 0, 2, 3, 4], return_path=True)
    assert result.path == [2] and not calls


def test_native_unavailable_after_launch_is_not_fallback(tmp_path, monkeypatch):
    fake_native(monkeypatch, tmp_path)
    def unexpected(*args, **kwargs):
        raise NativeUnavailable("late failure")
    monkeypatch.setattr(dispatch, "run_native", unexpected)
    with pytest.raises(NativeUnavailable, match="late failure"):
        dispatch.beam_search(graph(), native_options=NativeOptions(cache_dir=tmp_path), start_state=[1, 0, 2, 3, 4])


def test_not_found_is_not_error_or_fallback(tmp_path, monkeypatch):
    calls = fake_native(monkeypatch, tmp_path, path=None)
    result = dispatch.beam_search(graph(), native_options=NativeOptions(cache_dir=tmp_path),
                                  start_state=[1, 0, 2, 3, 4], return_path=True)
    assert calls and result.path_found is False and result.path is None
    assert result.native_metadata["status"] == "not_found_within_budget"


@pytest.mark.parametrize("solved", [False, True])
def test_zero_budget_does_not_build_or_launch(tmp_path, monkeypatch, solved):
    calls = fake_native(monkeypatch, tmp_path)
    monkeypatch.setattr(dispatch, "prepare_runtime", lambda *args: pytest.fail("must not build"))
    result = dispatch.beam_search(graph(), native_options=NativeOptions(cache_dir=tmp_path),
        start_state=[0, 1, 2, 3, 4] if solved else [1, 0, 2, 3, 4], max_steps=0)
    assert not calls and result.path_found is solved
    assert result.path == ([] if solved else None)


def test_auto_preserves_unsupported_upstream_kwargs_and_predictor(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "_registry_api", lambda: None)  # Legacy custom method capture.
    seen = []
    sentinel = object()
    def original(self, **kwargs):
        seen.append(kwargs)
        return sentinel
    monkeypatch.setattr(CayleyGraph, "beam_search", original)
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path, warn_on_fallback=False))
    kwargs = dict(start_state=[1, 0, 2, 3, 4], history_depth=3, beam_mode="advanced", predictor=object(), verbose=100)
    assert graph().beam_search(**kwargs) is sentinel
    assert seen == [kwargs]
    dispatch.disable_native()  # Restore captured function before monkeypatch unwinds.


def test_q_model_requires_explicit_child_scoring(tmp_path, monkeypatch):
    calls = fake_native(monkeypatch, tmp_path)
    monkeypatch.setattr(dispatch, "prepare_model", lambda *args: SimpleNamespace(artifact_hash="q", manifest={"output_dim": 3}))
    opts = NativeOptions(cache_dir=tmp_path)
    with pytest.raises(NativeUnavailable, match="use_child_scores"):
        dispatch.beam_search(graph(), backend="native", native_options=opts, start_state=[1, 0, 2, 3, 4])
    assert not calls
    result = dispatch.beam_search(graph(), backend="native", native_options=opts,
                                  start_state=[1, 0, 2, 3, 4], use_child_scores=True)
    assert calls and result.native_metadata["scoring"] == "parent_q"


def test_native_artifact_fallback_is_explicit(tmp_path, monkeypatch):
    from cayleypy_native import NativeModel
    def unavailable(*args):
        raise NativeUnavailable("no CUDA")
    monkeypatch.setattr(dispatch, "runtime_devices", unavailable)
    g = graph()
    opts = NativeOptions(cache_dir=tmp_path)
    model = NativeModel.for_graph(g, tmp_path / "weights")
    with warnings.catch_warnings():
        warnings.simplefilter("error", NativeFallbackWarning)
        with pytest.raises(NativeUnavailable, match="no CUDA; NativeModel has no fallback"):
            dispatch.beam_search(g, native_options=opts, start_state=[1, 0, 2, 3, 4], predictor=model)
    model = NativeModel.for_graph(g, tmp_path / "weights", fallback=Predictor(g, "hamming"))
    with pytest.warns(NativeFallbackWarning, match="no CUDA"):
        result = dispatch.beam_search(g, native_options=opts, start_state=[1, 0, 2, 3, 4], predictor=model, return_path=True)
    assert result.path == [2]


def test_invalid_falsey_options_are_not_ignored():
    with pytest.raises(TypeError, match="native_options"):
        dispatch.beam_search(graph(), native_options={}, start_state=[1, 0, 2, 3, 4])
