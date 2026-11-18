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

    calls, owners = [], []
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
            self.count = 0
            self.requests = []
            self.fail = False
            owners.append(self)

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
            return json.dumps({"path": [0], "effective_beam_width": 64,
                "elapsed_seconds": 0.1, "metadata": {"native_seconds": 0.1,
                "terminal_records": 1, "observed_parent_batch": None,
                "observed_inference_rows": None, "observed_stream3_batch_candidates": None,
                "persistent_session": {"reused": reused, "request_count": self.count}}})

        def close(self):
            self.closed = True
            self.pids = []

    monkeypatch.setattr(rust_control, "extension", lambda: SimpleNamespace(
        SearchSession=Owner, SEARCH_TERMINAL_RESULT_VERSION=1))
    contract = GraphContract(2, 2, ((1, 0),), ("swap",), (0, 1), (1, 0), "a" * 64)
    model = SimpleNamespace(artifact_hash="b" * 64, backend="mlp")
    runtime = SimpleNamespace(runner=tmp_path / "runner", architectures=(86,),
                              build_metadata={"binary_sha256": "c" * 64})
    options = NativeOptions(cache_dir=tmp_path)
    session = NativeSession(engine="rust")

    def execute(contract, model, options, beam, depth, run_dir, devices, *, runtime):
        session.launch(["old-runner", "0", str(depth), str(beam)], contract=contract,
                       run_dir=run_dir, env={"TEST_ENV": "retained"}, world=len(devices), timeout=5)
        return NativeOutcome((0,), 0.1, beam, run_dir,
                             {"requested_beam_width": beam, "profile": {}, "terminal_records": 1})

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
        assert len(owners) == len(calls) == 1
        identity, launch, start, timeout = owners[0].requests[0]
        assert identity["graph"] == {"center": [0, 1], "generators": [[1, 0]]}
        assert identity["devices"] == list(range(world)) and identity["beam_width"] == 64
        assert identity["runtime_build"]["native_library_sha256"] == "d" * 64
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
        assert owners[0].requests[2][0]["model_artifact_hash"] == "f" * 64
        owners[0].fail = True
        with pytest.raises(NativeBackendError, match="injected rank failure"):
            run(4)
        assert owners[0].closed
    finally:
        session.close()
    assert all(owner.closed for owner in owners)


def test_rust_session_rejects_launch_without_prepared_context(tmp_path):
    session = NativeSession(engine="rust")
    with pytest.raises(NativeBackendError, match="prepared"):
        session.launch(["runner", "0", "4", "64"], contract=None, run_dir=tmp_path,
                       env={}, world=1, timeout=1)
    session.close()
