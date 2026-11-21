"""The experimental Rust session owns rank workers behind NativeSession."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.errors import NativeBackendError
from multigpubeamsearch.options import NativeOptions, NativeOutcome
from multigpubeamsearch.persistent import NativeSession


def test_unknown_session_engine_is_rejected():
    with pytest.raises(ValueError, match="engine"):
        NativeSession(engine="unknown")


def test_compiled_rust_preparation_identity_builder_round_trips_fields():
    from multigpubeamsearch import rust_control
    from multigpubeamsearch.errors import NativeUnavailable

    try:
        runtime = rust_control.extension()
    except NativeUnavailable:
        pytest.skip("Rust extension is not built in this CPU-only job")

    build = getattr(runtime, "build_preparation_identity", None)
    assert callable(build), "Rust preparation identity builder is missing"
    runtime_build = {"binary_sha256": "c" * 64, "inference_backend": "cutlass"}
    result = build(
        "a" * 64,
        [4, 4, 2],
        [0, 1],
        [[1, 0]],
        "b" * 64,
        [17],
        "mlp",
        [0, 1],
        False,
        64,
        4,
        json.dumps({"inference_backend": "auto"}, sort_keys=True),
        "0,1",
        "PCI_BUS_ID",
        json.dumps(runtime_build),
    )
    assert json.loads(result) == {
        "graph_hash": "a" * 64,
        "geometry": [4, 4, 2],
        "center": [0, 1],
        "generators": [[1, 0]],
        "model_artifact_hash": "b" * 64,
        "artifact_sizes": [17],
        "model_backend": "mlp",
        "devices": [0, 1],
        "binding_scope": False,
        "beam_width": 64,
        "max_steps": 4,
        "options": {"inference_backend": "auto"},
        "cuda_visible_devices": "0,1",
        "cuda_device_order": "PCI_BUS_ID",
        "runtime_build": runtime_build,
    }


def test_compiled_rust_preparation_identity_uses_structured_options():
    from multigpubeamsearch import rust_control
    from multigpubeamsearch.errors import NativeUnavailable

    try:
        runtime = rust_control.extension()
    except NativeUnavailable:
        pytest.skip("Rust extension is not built in this CPU-only job")

    build = getattr(runtime, "build_preparation_identity", None)
    assert callable(build), "Rust preparation identity builder is missing"
    options = {
        "inference_backend": "cutlass",
        "calibration_frontier_max_states": 1234,
        "source_dir": "/remote/project",
    }
    identity = json.loads(build(
        "a" * 64,
        [4, 4, 2],
        [0, 1],
        [[1, 0]],
        "b" * 64,
        [17],
        "mlp",
        [0],
        False,
        64,
        4,
        json.dumps(options, sort_keys=True, allow_nan=False),
        "0",
        "PCI_BUS_ID",
        json.dumps({"binary_sha256": "c" * 64}),
    ))
    assert identity["options"] == options
    assert "options_repr" not in identity


def test_legacy_preparation_identity_extension_is_not_used_for_typed_identity(monkeypatch):
    from multigpubeamsearch import rust_control

    calls = []
    legacy = SimpleNamespace(build_preparation_identity=lambda *args: calls.append(args))
    monkeypatch.setattr(rust_control, "extension", lambda: legacy)

    assert rust_control.supports_preparation_identity() is False
    result = rust_control.build_preparation_identity(
        graph_hash="a" * 64,
        geometry=(2, 2, 1),
        center=(0, 1),
        generators=((1, 0),),
        model_artifact_hash="b" * 64,
        artifact_sizes=(17,),
        model_backend="mlp",
        devices=(0,),
        binding_scope=False,
        beam_width=64,
        max_steps=4,
        options={"inference_backend": "auto"},
        cuda_visible_devices=None,
        cuda_device_order=None,
        runtime_build=None,
    )
    assert result is None
    assert calls == []


def test_compiled_rust_session_identity_builder_round_trips_fields():
    from multigpubeamsearch import rust_control
    from multigpubeamsearch.errors import NativeUnavailable

    try:
        runtime = rust_control.extension()
    except NativeUnavailable:
        pytest.skip("Rust extension is not built in this CPU-only job")

    build = getattr(runtime, "build_session_identity", None)
    assert callable(build), "Rust session identity builder is missing"
    runtime_build = {
        "binary_sha256": "c" * 64,
        "native_library_sha256": "d" * 64,
        "rust_worker_sha256": "e" * 64,
        "abi_version": 1,
    }
    options = {
        "timeout_seconds": 300,
        "python_session_identity": {"geometry": [2, 2, 1], "binding_scope": False},
    }
    result = build(
        [0, 1], [[1, 0]], "a" * 64, "b" * 64, "mlp", [0, 1], 64, 4,
        json.dumps(runtime_build), json.dumps(options), "0,1", "PCI_BUS_ID",
    )
    assert json.loads(result) == {
        "graph": {"center": [0, 1], "generators": [[1, 0]]},
        "graph_hash": "a" * 64,
        "model_artifact_hash": "b" * 64,
        "model_backend": "mlp",
        "devices": [0, 1],
        "beam_width": 64,
        "max_steps": 4,
        "runtime_build": runtime_build,
        "options": options,
        "cuda_visible_devices": "0,1",
        "cuda_device_order": "PCI_BUS_ID",
    }


def test_rust_session_rejects_stale_terminal_protocol_before_native_build(tmp_path, monkeypatch):
    from multigpubeamsearch import native_library, rust_control
    from multigpubeamsearch.errors import NativeUnavailable
    from multigpubeamsearch.rust_resident import RustResidentWorkers

    monkeypatch.setattr(rust_control, 'extension', lambda: SimpleNamespace(SearchSession=object))
    monkeypatch.setattr(native_library, 'ensure_native_library',
                        lambda *args: pytest.fail('stale runtime must not trigger a native build'))
    with pytest.raises(NativeUnavailable, match='terminal-result v1'):
        RustResidentWorkers(['runner', '0', '4', '64'], contract=None, model=None,
            options=None, max_steps=4, devices=(0,), runtime=object(), run_dir=tmp_path,
            env={}, world=1)


@pytest.mark.parametrize("world", [1, 2])
def test_rust_session_reuses_and_releases_rank_owner(tmp_path, monkeypatch, world):
    from multigpubeamsearch import models, native_library, rust_control

    calls, owners, identity_builds = [], [], []
    session_identity_builds = []
    monkeypatch.setattr(models, "verify_prepared_model", lambda *args: None)
    worker = tmp_path / "beam-runner"
    library = tmp_path / "libbeam_native.so"
    monkeypatch.setattr(rust_control, "worker_executable", lambda: worker)
    monkeypatch.setattr(native_library, "ensure_native_library", lambda *args: (
        calls.append(args) or library,
        {"library_sha256": "d" * 64, "specification": {"worker_sha256": "e" * 64}},
    ))

    class Owner:
        def __init__(self):
            self.pids = list(range(100, 100 + world))
            self.closed = False
            self.identity = None
            self.prepared_identity = None
            self.preparation_checks = []
            self.preparation_begins = []
            self.preparation_resets = 0
            self.preparation_records = []
            self.preparation_results = []
            self.prepared_metadata = None
            self.last_outcome = None
            self.last_outcome_path = None
            self.count = 0
            self.requests = []
            self.fail = False
            owners.append(self)

        def begin_preparation(self, identity):
            self.preparation_begins.append(identity)
            if identity == self.prepared_identity:
                return True
            self.prepared_identity = None
            self.identity = None
            self.count = 0
            self.preparation_resets += 1
            return False

        def preparation_matches(self, identity):
            self.preparation_checks.append(identity)
            return identity == self.prepared_identity

        def remember_preparation(self, identity):
            self.preparation_records.append(identity)
            self.prepared_identity = identity

        def remember_preparation_result(self, identity, metadata):
            prepared = json.loads(metadata)
            self.preparation_results.append((identity, prepared))
            self.prepared_identity = identity
            self.prepared_metadata = prepared
            if self.last_outcome_path is None:
                return False
            self.last_outcome["metadata"] = prepared
            self.last_outcome_path.write_text(json.dumps(self.last_outcome), encoding="utf-8")
            return True

        def run(self, identity, launch, start, run_dir, timeout, metadata):
            if self.fail:
                raise RuntimeError("injected rank failure")
            identity = json.loads(identity)
            reused = identity == self.identity
            if not reused:
                self.identity = identity
                self.count = 0
            self.count += 1
            self.requests.append((identity, json.loads(launch), start, timeout))
            Path(run_dir, "native.log").write_text(
                "GLOBAL_BEAM_WIDTH_EFFECTIVE=64\n"
                "puzzle_solved=1 puzzle_id=0 seconds=0.1 solution_length=1 solution=m0\n"
            )
            result_metadata = dict(self.prepared_metadata or {})
            result_metadata.update({"native_seconds": 0.1, "terminal_records": 1,
                "observed_parent_batch": None, "observed_inference_rows": None,
                "observed_stream3_batch_candidates": None})
            result_metadata["persistent_session"] = {
                "reused": reused, "worker_pids": list(self.pids), "request_count": self.count}
            self.last_outcome = {"path": [0], "effective_beam_width": 64,
                "elapsed_seconds": 0.1, "metadata": result_metadata}
            self.last_outcome_path = Path(run_dir) / "native-outcome.json"
            self.last_outcome_path.write_text(json.dumps(self.last_outcome), encoding="utf-8")
            return json.dumps(self.last_outcome)

        def close(self):
            self.closed = True
            self.pids = []

    def build_preparation_identity(*values):
        identity_builds.append(values)
        names = (
            "graph_hash", "geometry", "center", "generators", "model_artifact_hash",
            "artifact_sizes", "model_backend", "devices", "binding_scope", "beam_width",
            "max_steps", "options_json", "cuda_visible_devices", "cuda_device_order",
            "runtime_build",
        )
        identity = dict(zip(names, values))
        identity["options"] = json.loads(identity.pop("options_json"))
        if identity["runtime_build"] is not None:
            identity["runtime_build"] = json.loads(identity["runtime_build"])
        return json.dumps(identity, sort_keys=True)

    def build_session_identity(*values):
        session_identity_builds.append(values)
        names = (
            "center", "generators", "graph_hash", "model_artifact_hash", "model_backend",
            "devices", "beam_width", "max_steps", "runtime_build", "options",
            "cuda_visible_devices", "cuda_device_order",
        )
        identity = dict(zip(names, values))
        identity["runtime_build"] = json.loads(identity["runtime_build"])
        identity["options"] = json.loads(identity["options"])
        center = identity.pop("center")
        generators = identity.pop("generators")
        identity["graph"] = {"center": center, "generators": generators}
        return json.dumps(identity, sort_keys=True)
    monkeypatch.setattr(rust_control, "extension", lambda: SimpleNamespace(
        SearchSession=Owner, SEARCH_TERMINAL_RESULT_VERSION=1,
        PREPARATION_IDENTITY_SCHEMA_VERSION=2,
        build_preparation_identity=build_preparation_identity,
        build_session_identity=build_session_identity))
    contract = GraphContract(2, 2, ((1, 0),), ("swap",), (0, 1), (1, 0), "a" * 64)
    model = SimpleNamespace(artifact_hash="b" * 64, backend="mlp")
    runtime = SimpleNamespace(runner=tmp_path / "runner", architectures=(86,),
                              build_metadata={"binary_sha256": "c" * 64})
    options = NativeOptions(cache_dir=tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    session = NativeSession(engine="rust")

    def execute(contract, model, options, beam, depth, run_dir, devices, *, runtime):
        session.launch(["old-runner", "0", str(depth), str(beam)], contract=contract,
                       run_dir=run_dir, env={"TEST_ENV": "retained"}, world=len(devices), timeout=5)
        return NativeOutcome((0,), 0.1, beam, run_dir,
                             {"requested_beam_width": beam, "profile": {}, "terminal_records": 1,
                              "callback_metadata_owner": "python-execute"})

    def run(number):
        directory = tmp_path / f"run-{number}"
        directory.mkdir()
        return session.run(execute, contract, model, options, 64, 4, directory,
                           tuple(range(world)), runtime=runtime)

    try:
        first, second = run(1), run(2)
        assert not first.metadata["persistent_session"]["reused"]
        assert second.metadata["persistent_session"]["reused"]
        assert second.metadata["persistent_session"]["request_count"] == 2
        assert second.metadata["callback_metadata_owner"] == "python-execute"
        assert len(owners) == len(calls) == 1
        identity, launch, start, timeout = owners[0].requests[0]
        assert identity["graph"] == {"center": [0, 1], "generators": [[1, 0]]}
        assert identity["devices"] == list(range(world)) and identity["beam_width"] == 64
        assert identity["runtime_build"]["native_library_sha256"] == "d" * 64
        python_identity = identity["options"]["python_session_identity"]
        assert python_identity["geometry"] == [2, 2, 1]
        assert python_identity["artifact_sizes"] == []
        assert python_identity["binding_scope"] is False
        assert launch["command"] == [str(worker), "0", "4", "64"]
        assert launch["protocol_command"] == [str(worker), "--resident-protocol-info"]
        assert launch["environment"]["BEAM_NATIVE_LIBRARY"] == str(library)
        assert launch["environment"]["TEST_ENV"] == "retained"
        assert start == [1, 0] and timeout == 5
        model.artifact_hash = "f" * 64
        changed = run(3)
        assert not changed.metadata["persistent_session"]["reused"]
        assert changed.metadata["persistent_session"]["request_count"] == 1
        assert len(owners) == 1 and len(calls) == 2 and not owners[0].closed
        assert len(owners[0].requests) == 3
        assert owners[0].preparation_checks == []
        assert len(owners[0].preparation_begins) == 2
        assert owners[0].preparation_resets == 1
        assert owners[0].preparation_records == []
        assert len(owners[0].preparation_results) == 2
        assert owners[0].preparation_begins[0] == owners[0].preparation_results[0][0]
        assert owners[0].preparation_begins[1] != owners[0].preparation_results[0][0]
        assert owners[0].preparation_begins[1] == owners[0].preparation_results[1][0]
        assert owners[0].preparation_results[0][1]["callback_metadata_owner"] == "python-execute"
        expected_first_session = {
            "reused": False, "worker_pids": list(range(100, 100 + world)),
            "request_count": 1,
        }
        assert owners[0].preparation_results[0][1]["persistent_session"] == expected_first_session
        assert owners[0].last_outcome["metadata"] == first.metadata
        assert json.loads(owners[0].preparation_begins[0]) == {
            "graph_hash": "a" * 64,
            "geometry": [2, 2, 1],
            "center": [0, 1],
            "generators": [[1, 0]],
            "model_artifact_hash": "b" * 64,
            "artifact_sizes": [],
            "model_backend": "mlp",
            "devices": list(range(world)),
            "binding_scope": False,
            "beam_width": 64,
            "max_steps": 4,
            "options_repr": repr(options),
            "cuda_visible_devices": "0,1",
            "cuda_device_order": "PCI_BUS_ID",
            "runtime_build": {"binary_sha256": "c" * 64},
        }
        assert owners[0].requests[2][0]["model_artifact_hash"] == "f" * 64
        assert len(identity_builds) == 3
        assert len(session_identity_builds) == 2
        assert session_identity_builds[1][3] == "f" * 64
        assert session_identity_builds[0][0:6] == (
            [0, 1], [[1, 0]], "a" * 64, "b" * 64, "mlp", list(range(world)))
        assert json.loads(session_identity_builds[0][9])["python_session_identity"]["geometry"] == [2, 2, 1]
        assert identity_builds[0][0:4] == ("a" * 64, [2, 2, 1], [0, 1], [[1, 0]])
        assert identity_builds[0][4:8] == ("b" * 64, [], "mlp", list(range(world)))
        owners[0].fail = True
        with pytest.raises(NativeBackendError, match="injected rank failure"):
            run(4)
        assert owners[0].closed
    finally:
        session.close()
    assert all(owner.closed for owner in owners)


def test_native_session_defers_outcome_file_only_to_capable_rust_owner(tmp_path, monkeypatch):
    from multigpubeamsearch import backend, persistent

    class Worker:
        def __init__(self, capable):
            self.capable = capable

        def can_finalize_preparation_result(self):
            return self.capable

    rust_session = NativeSession(engine="rust")
    rust_session._workers = Worker(True)
    monkeypatch.setattr(persistent, "current_session", lambda: rust_session)
    outcome = NativeOutcome((2, 3), 0.25, 64, tmp_path, {"owner": "rust"})

    assert rust_session.defer_native_outcome_write() is True
    assert backend._persist_native_outcome(tmp_path, outcome) is False
    assert not (tmp_path / "native-outcome.json").exists()

    rust_session._workers = Worker(False)
    assert rust_session.defer_native_outcome_write() is False
    assert backend._persist_native_outcome(tmp_path, outcome) is True
    target = tmp_path / "native-outcome.json"
    assert target.read_text(encoding="utf-8").endswith("\n")
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved == {
        "path": [2, 3],
        "elapsed_seconds": 0.25,
        "effective_beam_width": 64,
        "metadata": {"owner": "rust"},
    }


def test_python_session_keeps_python_outcome_writer(tmp_path, monkeypatch):
    from multigpubeamsearch import backend, persistent

    session = NativeSession(engine="python")
    session._workers = SimpleNamespace(can_finalize_preparation_result=lambda: True)
    monkeypatch.setattr(persistent, "current_session", lambda: session)
    outcome = NativeOutcome(None, 0.0, None, tmp_path, {"owner": "python"})

    assert session.defer_native_outcome_write() is False
    assert backend._persist_native_outcome(tmp_path, outcome) is True
    assert json.loads((tmp_path / "native-outcome.json").read_text(encoding="utf-8"))["metadata"] == {
        "owner": "python"
    }


def test_rust_session_owner_compares_typed_preparation_identity_when_extension_is_built():
    from multigpubeamsearch import rust_control
    from multigpubeamsearch.errors import NativeUnavailable

    try:
        extension = rust_control.extension()
    except (NativeUnavailable, ImportError):
        pytest.skip("PyO3 extension is not built in this environment")
    identity = json.dumps({
        "graph_hash": "a" * 64,
        "geometry": [2, 2, 1],
        "center": [0, 1],
        "generators": [[1, 0]],
        "model_artifact_hash": "b" * 64,
        "artifact_sizes": [17],
        "model_backend": "mlp",
        "devices": [0, 1],
        "binding_scope": False,
        "beam_width": 64,
        "max_steps": 4,
        "options": {"inference_backend": "auto"},
        "cuda_visible_devices": "0,1",
        "cuda_device_order": "PCI_BUS_ID",
        "runtime_build": {"binary_sha256": "c" * 64, "inference_backend": "cutlass"},
    })
    reordered = json.dumps(dict(reversed(list(json.loads(identity).items()))))
    changed = json.loads(identity)
    changed["model_artifact_hash"] = "d" * 64
    owner = extension.SearchSession()
    try:
        assert owner.begin_preparation(identity) is False
        owner.remember_preparation_result(
            identity,
            json.dumps({"profile": {"calibration_total_wall_seconds": 12.0}}),
        )
        assert owner.begin_preparation(reordered) is True
        assert owner.begin_preparation(json.dumps(changed)) is False
        assert owner.preparation_matches(identity) is False
        with pytest.raises(RuntimeError):
            owner.begin_preparation("{}")
    finally:
        owner.close()


def test_rust_resident_keeps_python_fallback_for_older_search_session_extensions():
    from multigpubeamsearch.rust_resident import RustResidentWorkers

    owner = SimpleNamespace()
    workers = RustResidentWorkers.__new__(RustResidentWorkers)
    workers._owner = owner
    assert workers.begin_preparation("identity") is None
    assert workers.preparation_matches("identity") is None
    assert workers.remember_preparation("identity") is False
    assert workers.remember_preparation_result("identity", {}) is False

    older_owner = SimpleNamespace(preparation_matches=lambda identity: identity == "identity")
    workers._owner = older_owner
    assert workers.begin_preparation("identity") is None
    assert workers.preparation_matches("identity") is True


def test_rust_session_rejects_launch_without_prepared_context(tmp_path):
    session = NativeSession(engine="rust")
    with pytest.raises(NativeBackendError, match="prepared"):
        session.launch(["runner", "0", "4", "64"], contract=None, run_dir=tmp_path,
                       env={}, world=1, timeout=1)
    session.close()


@pytest.mark.parametrize("defer", [False, True])
def test_backend_native_outcome_file_uses_rust_owner_when_available(tmp_path, monkeypatch, defer):
    from multigpubeamsearch import backend, persistent

    session = SimpleNamespace(defer_native_outcome_write=lambda: defer)
    monkeypatch.setattr(persistent, "current_session", lambda: session)
    outcome = NativeOutcome((2, 3), 0.25, 64, tmp_path, {"owner": "rust"})

    written = backend._persist_native_outcome(tmp_path, outcome)

    target = tmp_path / "native-outcome.json"
    assert written is (not defer)
    assert target.exists() is (not defer)
    if not defer:
        saved = json.loads(target.read_text(encoding="utf-8"))
        assert saved == {
            "path": [2, 3],
            "elapsed_seconds": 0.25,
            "effective_beam_width": 64,
            "metadata": {"owner": "rust"},
        }



def test_native_session_cancel_reaches_only_active_rust_owner():
    from multigpubeamsearch.rust_resident import RustResidentWorkers

    calls = []
    owner = SimpleNamespace(cancel=lambda: calls.append("cancel"), close=lambda: calls.append("close"))
    workers = RustResidentWorkers.__new__(RustResidentWorkers)
    workers._owner = owner
    session = NativeSession(engine="rust")
    try:
        assert session.cancel() is False
        session._workers = workers
        session._run_owner = 12345
        assert session.cancel() is True
        assert calls == ["cancel"]
        session._run_owner = None
    finally:
        session.close()
    assert calls == ["cancel", "close"]



def test_rust_resident_cancel_handles_missing_extension_method():
    from multigpubeamsearch.rust_resident import RustResidentWorkers

    workers = RustResidentWorkers.__new__(RustResidentWorkers)
    workers._owner = SimpleNamespace()
    assert workers.cancel() is False


def test_compiled_rust_preparation_identity_builder_accepts_auto_max():
    from multigpubeamsearch import rust_control
    from multigpubeamsearch.errors import NativeUnavailable

    try:
        runtime = rust_control.extension()
    except NativeUnavailable as error:
        if "not installed" in str(error):
            pytest.skip("Rust extension is not built in this CPU-only job")
        raise
    assert runtime.PREPARATION_IDENTITY_AUTO_MAX_VERSION == 1
    identity = rust_control.build_preparation_identity(
        graph_hash="a" * 64,
        geometry=(2, 2, 1),
        center=(0, 1),
        generators=((1, 0),),
        model_artifact_hash="b" * 64,
        artifact_sizes=(17,),
        model_backend="mlp",
        devices=(0, 1),
        binding_scope=False,
        beam_width="max",
        max_steps=4,
        options={"inference_backend": "auto"},
        cuda_visible_devices="0,1",
        cuda_device_order="PCI_BUS_ID",
        runtime_build={"binary_sha256": "c" * 64, "inference_backend": "cutlass"},
    )
    assert isinstance(identity, str)
    assert json.loads(identity)["beam_width"] == "max"

    owner = runtime.SearchSession()
    try:
        assert owner.begin_preparation(identity) is False
        owner.remember_preparation(identity)
        assert owner.preparation_matches(identity) is True
    finally:
        owner.close()


def test_auto_max_identity_builder_requires_explicit_extension_capability(monkeypatch):
    from multigpubeamsearch import rust_control

    builder_calls = []
    legacy_runtime = SimpleNamespace(
        build_preparation_identity=lambda *args: builder_calls.append(args)
    )
    monkeypatch.setattr(rust_control, "extension", lambda: legacy_runtime)

    assert rust_control.supports_auto_max_preparation_identity() is False
    result = rust_control.build_preparation_identity(
        graph_hash="a" * 64,
        geometry=(2, 2, 1),
        center=(0, 1),
        generators=((1, 0),),
        model_artifact_hash="b" * 64,
        artifact_sizes=(17,),
        model_backend="mlp",
        devices=(0,),
        binding_scope=False,
        beam_width="max",
        max_steps=4,
        options={"inference_backend": "auto"},
        cuda_visible_devices=None,
        cuda_device_order=None,
        runtime_build=None,
    )
    assert result is None
    assert builder_calls == []


def test_old_rust_owner_never_receives_auto_max_preparation_json(tmp_path, monkeypatch):
    from multigpubeamsearch import models, persistent, rust_control

    monkeypatch.setattr(rust_control, "supports_auto_max_preparation_identity", lambda: False)
    monkeypatch.setattr(
        rust_control,
        "build_preparation_identity",
        lambda **kwargs: pytest.fail("old extension must not build an auto-max key"),
    )
    monkeypatch.setattr(persistent, "requires_mlp_host_binding", lambda *args, **kwargs: False)
    monkeypatch.setattr(models, "verify_prepared_model", lambda *args, **kwargs: None)

    class NumericOnlyOwner:
        pids = [101]
        sequence = 1
        _last_reused = False

        def configure(self, *args, **kwargs):
            raise AssertionError("the retained owner should not be reconfigured")

        def _reject_typed_preparation(self, identity):
            raise AssertionError("old numeric-only owner received an auto-max key")

        begin_preparation = _reject_typed_preparation
        preparation_matches = _reject_typed_preparation
        remember_preparation = _reject_typed_preparation
        remember_preparation_result = _reject_typed_preparation

        def close(self):
            return None

    contract = SimpleNamespace(
        graph_hash="graph-hash", state_len=1, num_classes=1, move_count=1,
        center=(0,), generators=((0,),), start=(1,),
    )
    model = SimpleNamespace(artifact_hash="model-hash", artifact_sizes=(), backend="mlp")
    options = NativeOptions(cache_dir=tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    session = NativeSession(engine="rust")
    session._workers = NumericOnlyOwner()

    def execute(*args, **kwargs):
        session._pending["launch_started"] = True
        return NativeOutcome(None, 0.01, 64, run_dir, {"terminal_records": 1})

    try:
        outcome = session.run(
            execute, contract, model, options, "max", 4, run_dir, (0,),
            runtime=SimpleNamespace(build_metadata={}),
        )
        assert outcome.effective_beam_width == 64
        assert outcome.metadata["persistent_session"]["worker_pids"] == [101]
    finally:
        session.close()
