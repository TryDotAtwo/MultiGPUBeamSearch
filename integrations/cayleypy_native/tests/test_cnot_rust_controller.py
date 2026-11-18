"""Compare the Rust controller with the existing CNOT reference search."""
from dataclasses import asdict
from datetime import timedelta
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from multigpubeamsearch import cnot, rust_control
from multigpubeamsearch.errors import NativeUnavailable

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from verify_cnot import WireFixture, verify_layers, wide_case


def require_controller():
    try:
        available = hasattr(rust_control.extension(), "run_cnot")
    except NativeUnavailable:
        available = False
    if not available:
        if os.environ.get("REQUIRE_RUST_CNOT") == "1":
            raise AssertionError("matching Rust CNOT controller is required")
        pytest.skip("matching Rust CNOT controller is unavailable")


def test_engine_resolution_is_explicit_and_old_extensions_fail_closed(monkeypatch):
    monkeypatch.setattr(rust_control, "extension", lambda: SimpleNamespace())
    assert cnot._search_engine("auto", 3) == ("python", None)
    with pytest.raises(NativeUnavailable, match="CNOT search"):
        cnot._search_engine("rust", 3)
    with pytest.raises(ValueError, match="engine"):
        cnot._search_engine("invalid", 3)
    marker = object()
    monkeypatch.setattr(rust_control, "extension", lambda: SimpleNamespace(run_cnot=marker))
    assert cnot._search_engine("auto", 3) == ("rust", marker)
    assert cnot._search_engine("auto", 2**64) == ("python", None)
    with pytest.raises(ValueError, match="depth exceeds"):
        cnot._search_engine("rust", 2**64)


def test_rust_top_level_controller_owns_cnot_admission_and_stage_dispatch():
    require_controller()
    controller = getattr(rust_control.extension(), "run_cnot_native_search", None)
    assert callable(controller), "matching Rust CNOT top-level controller is required"

    events = []
    caller_root = object()
    fingerprints = {
        "upstream_commit": "commit",
        "checkpoint_sha256": "checkpoint",
        "feature_registry_sha256": "registry",
        "training_manifest_sha256": "manifest",
        "training_result_sha256": "result",
    }
    scorer = SimpleNamespace(metadata=fingerprints)
    admitted_root, identity = SimpleNamespace(shape=(1, 3, 1)), object()
    raw = b"packed-root"
    input_matrix = SimpleNamespace(root=admitted_root, raw=raw, n=3)
    result = object()

    class World:
        rank, size, device = 0, 1, "cpu"

        def require_consensus(self, value):
            events.append(("consensus", value))

        def start_transport(self, *, prefer_rust):
            events.append(("transport", prefer_rust))

        def all_ranks_support(self, available):
            events.append(("support", available))
            return True

    world = World()

    def prepare_input(*args):
        events.append(("prepare-input", *args))
        return input_matrix

    def identity_builder(wires, *, device):
        events.append(("identity", wires, device))
        return identity

    def stage_runner(*args):
        events.append(("native-stages", *args))
        return result

    def native_stages():
        events.append(("native-runner",))
        return stage_runner

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected Python CNOT fallback callback")

    options = (5, 9, 4, "qhat", True, "python", "rust")
    callbacks = (prepare_input, identity_builder, native_stages, forbidden, forbidden,
                 cnot.CnotResult)
    assert controller(world, caller_root, scorer, options, callbacks) is result
    import hashlib
    import json

    config = (5, 9, 4, "qhat", True, "python", "rust", (1, 3, 1),
              ("commit", "checkpoint", "registry", "manifest", "result"))
    expected_digest = hashlib.sha256(raw)
    expected_digest.update(json.dumps(config).encode())
    assert events == [
        ("prepare-input", caller_root, 5, 9, 4, "qhat", "python"),
        ("consensus", expected_digest.digest()),
        ("transport", True),
        ("identity", 3, "cpu"),
        ("native-runner",),
        ("support", True),
        ("native-stages", world, admitted_root, identity, input_matrix, scorer,
         "qhat", True, 5, 6, 6, 4),
    ]


def test_rust_controller_prepares_typed_input_without_python_builder(monkeypatch):
    require_controller()
    controller = rust_control.extension().run_cnot_native_search
    root = cnot.pack_rows((1, 2))
    scorer = SimpleNamespace(metadata={})
    identity = object()

    class World:
        rank, size, device = 0, 1, "cpu"

        def require_consensus(self, _digest):
            pass

        def start_transport(self, *, prefer_rust):
            assert not prefer_rust

    def python_input_builder(*_args):
        pytest.fail("Rust admission must construct the typed CNOT input directly")

    def identity_builder(wires, *, device):
        assert wires == 2 and device == "cpu"
        return identity

    def forbidden(*_args, **_kwargs):
        pytest.fail("identity input must return before search callbacks")

    monkeypatch.setattr(torch, "equal", lambda *_args: True)
    callbacks = (python_input_builder, identity_builder, forbidden, forbidden,
                 forbidden, cnot.CnotResult)
    result = controller(
        World(), root, scorer,
        (1, 1, 0, "qhat", False, "auto", "python"), callbacks,
    )

    assert result == cnot.CnotResult("solved", (), [], [])


def test_rust_native_root_goal_uses_typed_matrix_without_torch_equal(monkeypatch):
    require_controller()
    root = cnot.cnot_identity(3)
    scorer = SimpleNamespace(metadata={})

    def forbidden(*_args, **_kwargs):
        pytest.fail("Rust CNOT root admission must use the owned packed matrix")

    monkeypatch.setattr(torch, "equal", forbidden)
    result = cnot.search_cnot(root, scorer, width=4, shortlist=6, max_depth=3,
                              validation="auto", engine="rust")

    assert result.status == "solved"
    assert result.gates == ()


def test_rust_top_level_controller_preserves_compatibility_stage_fallback():
    require_controller()
    controller = getattr(rust_control.extension(), "run_cnot_native_search", None)
    assert callable(controller), "matching Rust CNOT top-level controller is required"

    events = []
    caller_root, scorer = object(), SimpleNamespace(metadata={})
    admitted_root, identity = SimpleNamespace(shape=(1, 4, 1)), object()
    input_matrix = SimpleNamespace(root=admitted_root, raw=b"packed-root", n=4)

    class World:
        rank, size, device = 1, 2, "cpu"

        def longs(self, values):
            events.append(("longs", values))
            return values

        def sum(self, values):
            events.append(("sum", values))
            return SimpleNamespace(item=lambda: 0)

        def require_consensus(self, value):
            events.append(("consensus", value))

        def start_transport(self, *, prefer_rust):
            events.append(("transport", prefer_rust))

        def all_ranks_support(self, available):
            events.append(("support", available))
            return False

    world = World()

    def prepare_input(*args):
        events.append(("prepare-input", *args))
        return input_matrix

    def identity_builder(wires, *, device):
        events.append(("identity", wires, device))
        return identity

    def native_stages():
        events.append(("native-runner",))
        return object()

    class Stages:
        def root_is_goal(self):
            events.append(("root-is-goal",))
            return True

    stages = Stages()

    def make_stages(*args):
        events.append(("make-stages", *args))
        return stages

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected Python reference CNOT search")

    options = (7, 12, 8, "proposal", False, "python", "auto")
    callbacks = (prepare_input, identity_builder, native_stages, make_stages, forbidden,
                 cnot.CnotResult)
    result = controller(world, caller_root, scorer, options, callbacks)
    assert isinstance(result, cnot.CnotResult)
    assert result.status == "solved" and result.gates == ()
    import hashlib
    import json

    config = (7, 12, 8, "proposal", False, "python", "rust", (1, 4, 1),
              (None, None, None, None, None))
    expected_digest = hashlib.sha256(b"packed-root")
    expected_digest.update(json.dumps(config).encode())
    assert events == [
        ("prepare-input", caller_root, 7, 12, 8, "proposal", "python"),
        ("longs", [0]),
        ("sum", [0]),
        ("consensus", expected_digest.digest()),
        ("transport", True),
        ("identity", 4, "cpu"),
        ("native-runner",),
        ("support", True),
        ("make-stages", world, admitted_root, identity, input_matrix, scorer,
         "proposal", False),
        ("root-is-goal",),
    ]


@pytest.mark.parametrize(("engine", "max_depth"), [("python", 2**70), ("auto", 2**64)])
def test_rust_top_level_controller_routes_python_engine_and_unbounded_depth(
        monkeypatch, engine, max_depth):
    require_controller()
    controller = rust_control.extension().run_cnot_native_search
    events = []
    caller_root = object()
    admitted_root = SimpleNamespace(shape=(1, 2, 1))
    scorer = SimpleNamespace(metadata={})
    input_matrix = SimpleNamespace(root=admitted_root, raw=b"packed-root", n=2)
    identity, result = object(), object()

    class World:
        rank, size, device = 0, 1, "cpu"

        def require_consensus(self, digest):
            events.append(("consensus", digest))

        def start_transport(self, *, prefer_rust):
            events.append(("transport", prefer_rust))

    def prepare_input(root, width, shortlist, depth, mode, validation):
        events.append(("prepare-input", root, width, shortlist, depth, mode, validation))
        return input_matrix

    def identity_builder(wires, *, device):
        events.append(("identity", wires, device))
        return identity

    def reference(*args):
        events.append(("reference", *args))
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected Rust CNOT stage")

    monkeypatch.setattr(torch, "equal", lambda *_: False)
    options = (1, 1, max_depth, "qhat", False, "python", engine)
    callbacks = (prepare_input, identity_builder, forbidden, forbidden, reference,
                 cnot.CnotResult)
    world = World()

    assert controller(world, caller_root, scorer, options, callbacks) is result
    import hashlib
    import json

    config = (1, 1, max_depth, "qhat", False, "python", "python", (1, 2, 1),
              (None, None, None, None, None))
    expected_digest = hashlib.sha256(b"packed-root")
    expected_digest.update(json.dumps(config).encode())
    assert events == [
        ("prepare-input", caller_root, 1, 1, max_depth, "qhat", "python"),
        ("consensus", expected_digest.digest()),
        ("transport", False),
        ("identity", 2, "cpu"),
        ("reference", world, admitted_root, identity, input_matrix, scorer,
         1, 1, max_depth, "qhat", False),
    ]


@pytest.mark.parametrize(
    ("engine", "max_depth", "message"),
    [("invalid", 3, "CNOT engine"), ("rust", 2**64, "depth exceeds")],
)
def test_rust_top_level_controller_rejects_invalid_engine_before_consensus(
        engine, max_depth, message):
    require_controller()
    controller = rust_control.extension().run_cnot_native_search
    calls = []
    caller_root = object()
    input_matrix = SimpleNamespace(root=SimpleNamespace(shape=(1, 2, 1)), raw=b"root", n=2)
    scorer = SimpleNamespace(metadata={})

    class World:
        rank, size, device = 0, 1, "cpu"

    def prepare_input(*args):
        calls.append("input")
        return input_matrix

    def forbidden(*args, **kwargs):
        raise AssertionError("search continued after invalid engine configuration")

    options = (1, 1, max_depth, "qhat", False, "python", engine)
    callbacks = (prepare_input, forbidden, forbidden, forbidden, forbidden, cnot.CnotResult)
    with pytest.raises(ValueError, match=message):
        controller(World(), caller_root, scorer, options, callbacks)
    assert calls == ["input"]


@pytest.mark.parametrize("mode,k", [("qhat", 7), ("proposal", 7), ("qhat", 12)])
def test_complete_layers_match_reference_and_independent_oracle(mode, k):
    require_controller()
    root = (8, 4, 2, 1)
    options = dict(width=5, shortlist=k, max_depth=4, mode=mode, audit=True)
    reference = cnot.search_cnot(cnot.pack_rows(root), WireFixture(9), engine="python", **options)
    result = cnot.search_cnot(cnot.pack_rows(root), WireFixture(9), engine="rust", **options)
    assert asdict(result) == asdict(reference)
    assert verify_layers(root, WireFixture(9), mode, 5, k, result.layers, torch.device("cpu"))


@pytest.mark.parametrize("max_depth", [0, 1, 3])
def test_terminal_and_wide_routes_match_reference(max_depth):
    require_controller()
    rows, _ = wide_case()
    options = dict(width=1, shortlist=3, max_depth=max_depth, audit=True)
    before = cnot.search_cnot(cnot.pack_rows(rows), WireFixture(428), engine="python", **options)
    after = cnot.search_cnot(cnot.pack_rows(rows), WireFixture(428), engine="rust", **options)
    assert asdict(after) == asdict(before)
    if after.gates is not None:
        assert cnot.replay_cnot(len(rows), after.gates) == rows


def test_python_scorer_errors_are_preserved():
    require_controller()
    class Invalid(WireFixture):
        def child_values(self, children):
            return torch.full((len(children),), float("nan"), device=children.device)

    for engine in ("python", "rust"):
        with pytest.raises(ValueError, match="nonfinite CNOT model score"):
            cnot.search_cnot(cnot.pack_rows((8, 4, 2, 1)), Invalid(0),
                             width=1, shortlist=2, max_depth=3, engine=engine)


@pytest.mark.parametrize("mode", ["qhat", "proposal"])
def test_rust_search_owns_stages_without_python_stage_object(monkeypatch, mode):
    """Complete audited search must not require the old Python stage owner."""
    require_controller()
    from multigpubeamsearch import cnot_rust_stages

    rows, scorer = wide_case()
    options = dict(width=1, shortlist=3, max_depth=3, mode=mode, audit=True)
    reference = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="python", **options)
    assert reference.gates == (18592, 428)

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy Python CNOT stage owner was instantiated")

    monkeypatch.setattr(cnot_rust_stages.CnotStages, "__init__", forbidden)
    result = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="rust", **options)
    assert asdict(result) == asdict(reference)
    assert cnot.replay_cnot(len(rows), result.gates) == rows


@pytest.mark.parametrize("mode", ["qhat", "proposal"])
def test_rust_search_owns_selection_without_python_stage_object(monkeypatch, mode):
    require_controller()
    from multigpubeamsearch import cnot_rust_selection

    rows, scorer = (8, 4, 2, 1), WireFixture(9)
    options = dict(width=5, shortlist=9, max_depth=4, mode=mode, audit=True)
    reference = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="python", **options)
    assert reference.layers and any(layer["distinct"] > options["width"] for layer in reference.layers)

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy Python CNOT selection owner was instantiated")

    monkeypatch.setattr(cnot_rust_selection.SelectionStages, "__init__", forbidden)
    result = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="rust", **options)
    assert asdict(result) == asdict(reference)
    assert verify_layers(rows, scorer, mode, options["width"], options["shortlist"],
                         result.layers, torch.device("cpu"))


@pytest.mark.parametrize("mode", ["qhat", "proposal"])
def test_rust_search_owns_goal_without_python_coordinator(monkeypatch, mode):
    require_controller()
    rows, scorer = wide_case()
    options = dict(width=1, shortlist=3, max_depth=3, mode=mode, audit=True)
    reference = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="python", **options)
    assert reference.gates == (18592, 428)

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy Python goal coordinator was called")

    monkeypatch.setattr(cnot, "_goal", forbidden)
    result = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="rust", **options)
    assert asdict(result) == asdict(reference)
    assert cnot.replay_cnot(len(rows), result.gates) == rows


@pytest.mark.parametrize("mode", ["qhat", "proposal"])
def test_rust_search_owns_stage_error_coordination(monkeypatch, mode):
    require_controller()
    from multigpubeamsearch.cnot_collectives import World

    rows, scorer = (8, 4, 2, 1), WireFixture(9)
    options = dict(width=5, shortlist=9, max_depth=4, mode=mode, audit=True)
    reference = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="python", **options)
    original = World.checked
    stages = {"initialize CNOT domain", "rank parent actions and expand",
              "deduplicate and evaluate distinct children"}

    def checked(world, stage, action):
        if stage in stages:
            raise AssertionError("legacy Python CNOT stage coordinator was called")
        return original(world, stage, action)

    monkeypatch.setattr(World, "checked", checked)
    result = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="rust", **options)
    assert asdict(result) == asdict(reference)


@pytest.mark.parametrize("mode,k", [("qhat", 9), ("proposal", 9), ("qhat", 12)])
def test_rust_search_owns_batch_stages_without_python_helpers(monkeypatch, mode, k):
    """CPU follows the legacy tensor path, including shortlists and full alphabets."""
    require_controller()
    rows, scorer = (8, 4, 2, 1), WireFixture(9)
    options = dict(width=5, shortlist=k, max_depth=4, mode=mode, audit=True)
    reference = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="python", **options)
    assert reference.layers

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy Python CNOT batch helper was called")

    for name in ("_scores", "_expand", "_distinct"):
        monkeypatch.setattr(cnot, name, forbidden)
    result = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="rust", **options)
    assert asdict(result) == asdict(reference)
    assert verify_layers(rows, scorer, mode, options["width"], k,
                         result.layers, torch.device("cpu"))


@pytest.mark.parametrize("stage,kind", [
    ("parent", "shape"), ("parent", "device"), ("parent", "dtype"), ("parent", "nonfinite"),
    ("child", "shape"), ("child", "device"), ("child", "dtype"), ("child", "nonfinite"),
])
def test_score_validation_error_matches_python_reference(stage, kind):
    require_controller()

    class Invalid(WireFixture):
        def corrupt(self, scores):
            if kind == "shape":
                return scores.flatten()[:1]
            if kind == "dtype":
                return scores.to(torch.int64)
            if kind == "device":
                return torch.empty_like(scores, device="meta")
            return torch.full_like(scores, float("nan"))

        def parent_scores(self, parents, mode):
            scores = super().parent_scores(parents, mode)
            return self.corrupt(scores) if stage == "parent" else scores

        def child_values(self, children):
            scores = super().child_values(children)
            return self.corrupt(scores) if stage == "child" else scores

    errors = []
    for engine in ("python", "rust"):
        with pytest.raises(ValueError) as exc:
            cnot.search_cnot(cnot.pack_rows((8, 4, 2, 1)), Invalid(9),
                             width=5, shortlist=9, max_depth=4, engine=engine)
        errors.append(str(exc.value))
    assert errors[0] == errors[1]
    assert ("nonfinite CNOT model score" if kind == "nonfinite" else
            "expected floating scores on cpu with shape") in errors[0]


@pytest.mark.parametrize("mode", ["qhat", "proposal"])
def test_rust_search_owns_record_exchange_without_python_world(monkeypatch, mode):
    require_controller()
    from multigpubeamsearch.cnot_collectives import World

    rows, scorer = (8, 4, 2, 1), WireFixture(9)
    options = dict(width=5, shortlist=9, max_depth=4, mode=mode, audit=True)
    reference = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="python", **options)
    assert reference.layers

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy Python CNOT record exchange was called")

    monkeypatch.setattr(World, "exchange", forbidden)
    result = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="rust", **options)
    assert asdict(result) == asdict(reference)


def distributed(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        require_controller()
        rows, _ = wide_case()
        for mode in ("qhat", "proposal"):
            options = dict(width=1, shortlist=3, max_depth=3, mode=mode, audit=True)
            old = cnot.search_cnot(cnot.pack_rows(rows), WireFixture(428), engine="python", **options)
            new = cnot.search_cnot(cnot.pack_rows(rows), WireFixture(428), engine="rust", **options)
            assert asdict(old) == asdict(new)
            assert new.gates == (18592, 428)
            assert cnot.replay_cnot(len(rows), new.gates) == rows

        from multigpubeamsearch.cnot_collectives import World
        options = dict(width=1, shortlist=3, max_depth=3, mode="qhat", audit=True)
        exchange_reference = cnot.search_cnot(cnot.pack_rows(rows), WireFixture(428), engine="python", **options)
        original_exchange = World.exchange
        def forbidden_exchange(*args, **kwargs):
            raise AssertionError("legacy Python CNOT record exchange was called")
        World.exchange = forbidden_exchange
        try:
            owned_exchange = cnot.search_cnot(cnot.pack_rows(rows), WireFixture(428), engine="rust", **options)
            assert asdict(owned_exchange) == asdict(exchange_reference)
        finally:
            World.exchange = original_exchange

        from multigpubeamsearch import cnot_rust_stages
        legacy_calls = []
        old_init = cnot_rust_stages.CnotStages.__init__
        old_runner = cnot._native_stages_runner

        def trace_legacy(self, *args, **kwargs):
            legacy_calls.append(True)
            return old_init(self, *args, **kwargs)

        cnot_rust_stages.CnotStages.__init__ = trace_legacy
        cnot._native_stages_runner = lambda: None if rank == 1 else old_runner()
        options = dict(width=1, shortlist=3, max_depth=3, mode="qhat", audit=True)
        fallback_reference = cnot.search_cnot(cnot.pack_rows(rows), WireFixture(428), engine="python", **options)
        fallback = cnot.search_cnot(cnot.pack_rows(rows), WireFixture(428), engine="rust", **options)
        assert asdict(fallback) == asdict(fallback_reference)
        assert legacy_calls, f"rank {rank} did not choose the shared legacy backend"
        cnot._native_stages_runner = old_runner
        cnot_rust_stages.CnotStages.__init__ = old_init

        class RejectOnOne(WireFixture):
            def child_values(self, children):
                if rank == 1:
                    raise RuntimeError("uncovered feature")
                return super().child_values(children)

        with pytest.raises(RuntimeError, match="rank 1: RuntimeError: uncovered feature"):
            cnot.search_cnot(cnot.pack_rows((8, 4, 2, 1)), RejectOnOne(9),
                             width=5, shortlist=9, max_depth=4, engine="rust")
        recovered = cnot.search_cnot(cnot.pack_rows((1, 3)), object(), width=1,
                                     shortlist=1, max_depth=1, engine="rust")
        assert recovered.gates == (0,)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="two CPU ranks require Gloo")
def test_two_gloo_ranks_exact_results_failure_and_recovery(tmp_path):
    require_controller()
    mp.spawn(distributed, args=((tmp_path / "cnot-rust-gloo").resolve().as_uri(),), nprocs=2)
