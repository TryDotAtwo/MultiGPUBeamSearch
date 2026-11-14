import json
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
def clean_install(monkeypatch):
    # Source provisioning has its own tests; dispatch mocks must not download.
    monkeypatch.setattr("cayleypy_native.bootstrap.prepare_sources", lambda options: options)
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
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path), default_backend="auto")
    with pytest.warns(NativeFallbackWarning, match="no compatible native runtime"):
        result = g.beam_search(start_state=[1, 0, 2, 3, 4], return_path=True)
    assert isinstance(result, BeamSearchResult)
    assert result.path == [2]
    assert torch.equal(g.apply_path([1, 0, 2, 3, 4], result.path).reshape(-1), g.central_state)


def test_auto_cpu_fallback_does_not_touch_unavailable_cache(tmp_path, monkeypatch):
    blocked_cache = tmp_path / "cache-is-a-file"
    blocked_cache.write_text("native cache must not be touched before CUDA preflight")
    monkeypatch.setattr(
        dispatch, "runtime_devices",
        lambda *args: (_ for _ in ()).throw(NativeUnavailable("test: CPU-only host")),
    )
    result = dispatch.beam_search(
        graph(),
        backend="auto",
        native_options=NativeOptions(cache_dir=blocked_cache, warn_on_fallback=False),
        start_state=[1, 0, 2, 3, 4],
        return_path=True,
    )
    assert result.path == [2]
    assert blocked_cache.is_file()


def test_auto_cache_creation_failure_is_a_capability_fallback(tmp_path, monkeypatch):
    blocked_cache = tmp_path / "cache-is-a-file"
    blocked_cache.write_text("test-only obstruction")
    monkeypatch.setattr(dispatch, "runtime_devices", lambda *args: (0,))
    with pytest.warns(NativeFallbackWarning, match="cache run directory"):
        result = dispatch.beam_search(
            graph(),
            backend="auto",
            native_options=NativeOptions(cache_dir=blocked_cache),
            start_state=[1, 0, 2, 3, 4],
            return_path=True,
        )
    assert result.path == [2]


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
    monkeypatch.setattr(dispatch, "prepare_runtime", lambda *args, **kwargs: SimpleNamespace(build_metadata={}))
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


def test_dispatch_clamps_touch_bfs_to_the_requested_step_budget(tmp_path, monkeypatch):
    fake_native(monkeypatch, tmp_path)
    observed = []

    def prepare(*args, touch_bfs_radius=None):
        observed.append(touch_bfs_radius)
        return SimpleNamespace(build_metadata={})

    monkeypatch.setattr(dispatch, "prepare_runtime", prepare)
    result = dispatch.beam_search(
        graph(),
        backend="native",
        native_options=NativeOptions(cache_dir=tmp_path, touch_bfs_radius=12),
        start_state=[1, 0, 2, 3, 4],
        max_steps=4,
        return_path=True,
    )
    assert result.path == [2]
    assert observed == [3]


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_dispatch_passes_only_opted_in_session_digest_cache(tmp_path, monkeypatch, engine):
    from multigpubeamsearch import NativeSession

    fake_native(monkeypatch, tmp_path)
    observed = []

    def prepare(*args, **kwargs):
        observed.append(kwargs.get("nccl_digest_cache"))
        return SimpleNamespace(build_metadata={})

    monkeypatch.setattr(dispatch, "prepare_runtime", prepare)
    options = NativeOptions(cache_dir=tmp_path)
    args = dict(backend="native", native_options=options, start_state=[1, 0, 2, 3, 4],
                max_steps=4, return_path=True)
    dispatch.beam_search(graph(), **args)
    with NativeSession(engine=engine, experimental_parallel_nccl_identity=True) as session:
        dispatch.beam_search(graph(), **args)
        assert observed == [None, session.nccl_digest_cache]


@pytest.mark.parametrize("invalid", ["closed", "other_process"])
def test_dispatch_rejects_invalid_session_before_digest_preparation(tmp_path, monkeypatch, invalid):
    from multigpubeamsearch import NativeSession

    fake_native(monkeypatch, tmp_path)
    monkeypatch.setattr(dispatch, "prepare_runtime", lambda *args, **kwargs: pytest.fail("must reject before preparation"))
    session = NativeSession(experimental_parallel_nccl_identity=True)
    with session.activate():
        owner = session._owner_pid
        if invalid == "closed":
            session.close()
        else:
            session._owner_pid = -1
        try:
            with pytest.raises(RuntimeError, match="closed|process"):
                dispatch.beam_search(graph(), backend="native", native_options=NativeOptions(cache_dir=tmp_path),
                                     start_state=[1, 0, 2, 3, 4], max_steps=4)
        finally:
            session._owner_pid = owner
            session.close()


def test_session_cannot_close_while_nccl_preparation_is_active(tmp_path, monkeypatch):
    from multigpubeamsearch import NativeSession

    fake_native(monkeypatch, tmp_path)
    with NativeSession(experimental_parallel_nccl_identity=True) as session:
        def prepare(*args, **kwargs):
            with pytest.raises(RuntimeError, match="during a search"):
                session.close()
            return SimpleNamespace(build_metadata={})

        monkeypatch.setattr(dispatch, "prepare_runtime", prepare)
        dispatch.beam_search(graph(), backend="native", native_options=NativeOptions(cache_dir=tmp_path),
                             start_state=[1, 0, 2, 3, 4], max_steps=4)
    assert session.closed


def test_prepared_session_reuses_worker_then_rejects_changed_nccl(tmp_path, monkeypatch):
    import subprocess
    from multigpubeamsearch import NativeSession, backend, build, models, resident_transport
    from multigpubeamsearch.contracts import GraphContract
    from multigpubeamsearch.persistent import current_session

    g = graph()
    start = [1, 0, 2, 3, 4]
    contract = GraphContract.from_graph(g, start)
    runner = tmp_path / "runner"
    runner.write_bytes(b"fixture executable")
    runner.chmod(0o755)
    include = tmp_path / "include"
    include.mkdir()
    (include / "nccl.h").write_bytes(b"header")
    library = tmp_path / "libnccl.so.2"
    library.write_bytes(b"original nccl")
    metadata = {"schema_version": 1, "shape": build.shape_contract(contract), "backend": "mlp",
                "cuda_architectures": [86], "inference_backend": "cutlass",
                "binary_sha256": build.file_sha256(runner), "nccl": build._nccl_identity(include, library)}
    (tmp_path / "native-build.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(dispatch, "runtime_devices", lambda *args: (0,))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (8, 6))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda d: "Fixture GPU")
    monkeypatch.setattr(dispatch, "prepare_model", lambda *args: SimpleNamespace(
        backend="mlp", artifact_hash="fixture-model", manifest={"dtype": "fp16", "output_dim": 1}))
    monkeypatch.setattr(models, "verify_prepared_model", lambda *args: None)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], 0, '{"protocol":"resident-search-v1"}', ""))
    workers = []

    class FixtureWorkers:
        def __init__(self, *args, **kwargs):
            self.pids = [12345]
            self.sequence = 0
            self.closed = False
            workers.append(self)

        def submit(self, start, directory, timeout):
            self.sequence += 1
            (directory / "native.log").write_text(
                "GLOBAL_BEAM_WIDTH_EFFECTIVE=64\n"
                "puzzle_solved=1 puzzle_id=0 seconds=0.1 solution_length=1 solution=m2\n")
            return 0.1

        def close(self):
            self.closed = True

    monkeypatch.setattr(resident_transport, "ResidentWorkers", FixtureWorkers)

    def execute(contract, model, options, beam_width, max_steps, run_dir, devices, *, runtime):
        session = current_session()
        elapsed = session.launch([str(runner)], contract=contract, run_dir=run_dir, env={},
                                 world=len(devices), timeout=5, runtime=runtime)
        return NativeOutcome((2,), elapsed, beam_width, run_dir,
                             {"requested_beam_width": beam_width, "profile": {}, "terminal_records": 1})

    monkeypatch.setattr(backend, "_run_native_once", execute)
    options = NativeOptions(cache_dir=tmp_path / "cache", runner_path=runner, inference_backend="cutlass")
    args = dict(backend="native", native_options=options, start_state=start,
                beam_width=64, max_steps=4, return_path=True)
    with NativeSession(experimental_parallel_nccl_identity=True):
        first = dispatch.beam_search(g, **args)
        second = dispatch.beam_search(g, **args)
        assert first.path == second.path == [2]
        assert len(workers) == 1 and workers[0].sequence == 2
        assert second.native_metadata["persistent_session"]["reused"]
        library.write_bytes(b"modified nccl")
        with pytest.raises(NativeBackendError, match="NCCL.*changed"):
            dispatch.beam_search(g, **args)
        assert workers[0].sequence == 2
    assert workers[0].closed


def test_dispatch_rejects_touch_bfs_bound_before_cuda_or_model(tmp_path, monkeypatch):
    contract = SimpleNamespace(start=(1,), center=(0,), move_count=24)
    monkeypatch.setattr(dispatch.GraphContract, "from_graph", lambda *args: contract)

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported touch-BFS must be rejected before CUDA, model export, build, or launch")

    for name in ("runtime_devices", "prepare_model", "prepare_runtime", "run_native"):
        monkeypatch.setattr(dispatch, name, forbidden)
    with pytest.raises(NativeUnavailable, match="worst-case host allocation"):
        dispatch.beam_search(
            graph(),
            backend="native",
            native_options=NativeOptions(cache_dir=tmp_path, touch_bfs_radius=12),
            start_state=[1, 0, 2, 3, 4],
            max_steps=100,
        )


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
    def unsupported(*args, **kwargs):
        raise NativeUnavailable("binary shape mismatch")
    monkeypatch.setattr(dispatch, "prepare_runtime", unsupported)
    with pytest.warns(NativeFallbackWarning, match="shape mismatch"):
        result = dispatch.beam_search(graph(), backend="auto", native_options=NativeOptions(cache_dir=tmp_path),
                                      start_state=[1, 0, 2, 3, 4], return_path=True)
    assert result.path == [2] and not calls
    assert not list((tmp_path / "runs").glob("*"))


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


@pytest.mark.parametrize("backend", ["native", "auto"])
@pytest.mark.parametrize("solved", [False, True])
def test_trivial_native_result_needs_no_gpu_model_or_cache(
    tmp_path, monkeypatch, solved, backend
):
    from cayleypy_native import NativeModel

    def forbidden(*args, **kwargs):
        pytest.fail("trivial native result must not probe CUDA, prepare a model, build, or launch")

    for name in ("runtime_devices", "prepare_model", "prepare_runtime", "run_native"):
        monkeypatch.setattr(dispatch, name, forbidden)
    g = graph()
    model = NativeModel.for_graph(g, tmp_path / "missing-weights")
    blocked_cache = tmp_path / "cache-is-a-file"
    blocked_cache.write_text("must remain untouched")
    result = dispatch.beam_search(
        g,
        backend=backend,
        native_options=NativeOptions(cache_dir=blocked_cache),
        start_state=[0, 1, 2, 3, 4] if solved else [1, 0, 2, 3, 4],
        predictor=model,
        max_steps=4 if solved else 0,
        return_path=True,
    )
    assert result.path_found is solved
    assert result.path == ([] if solved else None)
    assert result.native_metadata["devices"] == []
    assert result.native_metadata["model_hash"] is None
    assert result.native_metadata["scoring"] == "not_evaluated"
    assert result.native_metadata["run_dir"] is None
    assert blocked_cache.read_text() == "must remain untouched"


def test_auto_preserves_unsupported_upstream_kwargs_and_predictor(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "_registry_api", lambda: None)  # Legacy custom method capture.
    seen = []
    sentinel = object()
    def original(self, **kwargs):
        seen.append(kwargs)
        return sentinel
    monkeypatch.setattr(CayleyGraph, "beam_search", original)
    dispatch.enable_native(NativeOptions(cache_dir=tmp_path, warn_on_fallback=False), default_backend="auto")
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


def test_legacy_cayleypy_accepts_native_q_child_scoring(tmp_path, monkeypatch):
    """Native Q scoring remains available when upstream predates that Torch option."""
    current = dispatch.inspect.signature(dispatch.BeamSearchAlgorithm.search)
    legacy = current.replace(
        parameters=[parameter for parameter in current.parameters.values()
                    if parameter.name != "use_child_scores"]
    )
    monkeypatch.setattr(dispatch.inspect, "signature", lambda _target: legacy)
    calls = fake_native(monkeypatch, tmp_path)
    monkeypatch.setattr(
        dispatch,
        "prepare_model",
        lambda *args: SimpleNamespace(artifact_hash="q", manifest={"output_dim": 3}),
    )

    result = dispatch.beam_search(
        graph(),
        backend="native",
        native_options=NativeOptions(cache_dir=tmp_path),
        start_state=[1, 0, 2, 3, 4],
        use_child_scores=True,
    )

    assert calls
    assert result.native_metadata["scoring"] == "parent_q"


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
            dispatch.beam_search(g, backend="auto", native_options=opts, start_state=[1, 0, 2, 3, 4], predictor=model)
    model = NativeModel.for_graph(g, tmp_path / "weights", fallback=Predictor(g, "hamming"))
    with pytest.warns(NativeFallbackWarning, match="no CUDA"):
        result = dispatch.beam_search(g, backend="auto", native_options=opts, start_state=[1, 0, 2, 3, 4], predictor=model, return_path=True)
    assert result.path == [2]


def test_invalid_falsey_options_are_not_ignored():
    with pytest.raises(TypeError, match="native_options"):
        dispatch.beam_search(graph(), native_options={}, start_state=[1, 0, 2, 3, 4])
