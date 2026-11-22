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


def test_rust_top_level_controller_owns_cnot_admission_and_stage_dispatch(monkeypatch):
    monkeypatch.delenv("MGBS_CNOT_STRICT_BACKEND", raising=False)
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

        def longs(self, values):
            events.append(("longs", values))
            return torch.tensor(values, dtype=torch.int64)

        def gather_fixed(self, tensor):
            values = tensor.tolist()
            events.append(("gather-fixed", values))
            return torch.stack([tensor for _ in range(self.size)])

        def sum(self, tensor):
            events.append(("sum", tensor.tolist()))
            return tensor.sum()

        def all_ranks_support(self, local_support):
            events.append(("all-ranks-support", bool(local_support)))
            return bool(local_support)

        _stream5_library = None

    world = World()

    def prepare_input(*args):
        events.append(("prepare-input", *args))
        return input_matrix

    def identity_builder(wires, *, device):
        events.append(("identity", wires, device))
        return identity

    def stage_runner(*args, **kwargs):
        events.append(("native-stages", *args, kwargs))
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
    digest_values = list(expected_digest.digest())
    assert events[:-1] == [
        ("prepare-input", caller_root, 5, 9, 4, "qhat", "python"),
        ("longs", digest_values),
        ("gather-fixed", digest_values),
        ("identity", 3, "cpu"),
        ("native-runner",),
        ("all-ranks-support", True),
    ]
    stage_event = events[-1]
    assert stage_event[:12] == (
        "native-stages", world, admitted_root, identity, input_matrix, scorer,
        "qhat", True, 5, 6, 6, 4,
    )
    assert stage_event[-2].strict is False
    assert stage_event[-1] == {"admitted_topology": (0, 1)}


def test_rust_controller_prepares_typed_input_without_python_builder(monkeypatch):
    require_controller()
    controller = rust_control.extension().run_cnot_native_search
    root = cnot.pack_rows((1, 2))
    scorer = SimpleNamespace(metadata={})
    identity = object()

    class World:
        rank, size, device = 0, 1, "cpu"

        def longs(self, values):
            return torch.tensor(values, dtype=torch.int64)

        def gather_fixed(self, tensor):
            return torch.stack([tensor for _ in range(self.size)])

        _stream5_library = None

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
    root = torch.tensor([[[1], [2], [4]]], dtype=torch.int64)
    scorer = SimpleNamespace(metadata={})

    def forbidden(*_args, **_kwargs):
        pytest.fail(
            "Rust CNOT root admission must use the owned packed matrix "
            "without invoking the Python stage callback"
        )

    monkeypatch.setattr(torch, "equal", forbidden)
    from multigpubeamsearch.cnot_rust_stages import CnotStages
    monkeypatch.setattr(CnotStages, "root_is_goal", forbidden)
    monkeypatch.setattr(cnot, "_native_stages_runner", lambda: None)
    result = cnot.search_cnot(root, scorer, width=4, shortlist=6, max_depth=3,
                              validation="auto", engine="rust")

    assert result.status == "solved"
    assert result.gates == ()


def test_rust_compatibility_backend_owns_cnot_goal_dispatch(monkeypatch):
    require_controller()
    from multigpubeamsearch.cnot_rust_stages import CnotStages

    def forbidden(*_args, **_kwargs):
        pytest.fail("Rust compatibility backend must dispatch the CNOT goal directly")

    monkeypatch.setattr(CnotStages, "goal", forbidden)
    monkeypatch.setattr(cnot, "_native_stages_runner", lambda: None)
    root = cnot.pack_rows(cnot.replay_cnot(4, (0,)))
    result = cnot.search_cnot(root, object(), width=1, shortlist=1, max_depth=1,
                              validation="auto", engine="rust")
    assert result.status == "solved"
    assert result.gates == (0,)


def test_rust_compatibility_backend_owns_cnot_history_edge_dispatch(monkeypatch):
    require_controller()
    from multigpubeamsearch.cnot_rust_stages import CnotStages

    def forbidden(*_args, **_kwargs):
        pytest.fail("Rust compatibility backend must broadcast history edges directly")

    monkeypatch.setattr(CnotStages, "history_edge", forbidden)
    monkeypatch.setattr(cnot, "_native_stages_runner", lambda: None)
    rows = cnot.replay_cnot(4, (0, 1))
    result = cnot.search_cnot(
        cnot.pack_rows(rows), WireFixture(9), width=12, shortlist=12, max_depth=2,
        validation="auto", engine="rust",
    )
    assert result.status == "solved"
    assert len(result.gates) == 2
    assert cnot.replay_cnot(4, result.gates) == rows


def test_rust_compatibility_backend_owns_cnot_candidate_exchange_dispatch(monkeypatch):
    require_controller()
    from multigpubeamsearch.cnot_rust_stages import CnotStages

    def forbidden(*_args, **_kwargs):
        pytest.fail(
            "Rust compatibility backend must dispatch candidate exchange directly"
        )

    monkeypatch.setattr(CnotStages, "exchange_candidates", forbidden)
    monkeypatch.setattr(cnot, "_native_stages_runner", lambda: None)
    rows = cnot.replay_cnot(4, (0, 1))
    result = cnot.search_cnot(
        cnot.pack_rows(rows), WireFixture(9), width=12, shortlist=12, max_depth=2,
        validation="auto", engine="rust",
    )
    assert result.status == "solved"
    assert len(result.gates) == 2
    assert cnot.replay_cnot(4, result.gates) == rows


def test_rust_compatibility_backend_owns_cnot_construction_verification_dispatch(monkeypatch):
    require_controller()
    from multigpubeamsearch.cnot_rust_stages import CnotStages

    def forbidden(*_args, **_kwargs):
        pytest.fail(
            "Rust compatibility backend must verify the typed CNOT path directly"
        )

    monkeypatch.setattr(CnotStages, "verifies_construction", forbidden)
    monkeypatch.setattr(cnot, "_native_stages_runner", lambda: None)
    rows = cnot.replay_cnot(4, (0, 1))
    result = cnot.search_cnot(
        cnot.pack_rows(rows), WireFixture(9), width=12, shortlist=12, max_depth=2,
        validation="auto", engine="rust",
    )
    assert result.status == "solved"
    assert len(result.gates) == 2
    assert cnot.replay_cnot(4, result.gates) == rows


def test_rust_compatibility_backend_owns_cnot_initialization_dispatch(monkeypatch):
    require_controller()
    from multigpubeamsearch.cnot_rust_stages import CnotStages

    def forbidden(*_args, **_kwargs):
        pytest.fail(
            "Rust compatibility backend must initialize the typed CNOT stage directly"
        )

    monkeypatch.setattr(CnotStages, "initialize", forbidden)
    monkeypatch.setattr(cnot, "_native_stages_runner", lambda: None)
    rows = cnot.replay_cnot(4, (0, 1))
    result = cnot.search_cnot(
        cnot.pack_rows(rows), WireFixture(9), width=12, shortlist=12, max_depth=2,
        validation="auto", engine="rust",
    )
    assert result.status == "solved"
    assert len(result.gates) == 2
    assert cnot.replay_cnot(4, result.gates) == rows



def test_rust_top_level_controller_preserves_compatibility_stage_fallback(monkeypatch):
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
            return torch.tensor(values, dtype=torch.int64)

        def gather_fixed(self, tensor):
            values = tensor.tolist()
            events.append(("gather-fixed", values))
            return torch.stack([tensor for _ in range(self.size)])

        def sum(self, tensor):
            events.append(("sum", tensor.tolist()))
            return tensor.sum()

        def any_rank_failed(self, local_failed):
            events.append(("gloo-any-rank-failed", bool(local_failed)))
            return bool(local_failed)

        def all_ranks_support(self, local_support):
            events.append(("all-ranks-support", bool(local_support)))
            return next(support_votes)

        _stream5_library = None

    world = World()
    support_votes = iter((False, True, False, True, False, False))
    monkeypatch.delenv("MGBS_CNOT_STRICT_BACKEND", raising=False)

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
    digest_values = list(expected_digest.digest())
    assert [event for event in events if event[0] == "all-ranks-support"] == [
        ("all-ranks-support", False),
        ("all-ranks-support", True),
        ("all-ranks-support", False),
        ("all-ranks-support", True),
        ("all-ranks-support", False),
        ("all-ranks-support", True),
    ]
    assert [event for event in events if event[0] != "all-ranks-support"] == [
        ("prepare-input", caller_root, 7, 12, 8, "proposal", "python"),
        ("gloo-any-rank-failed", False),
        ("longs", digest_values),
        ("gather-fixed", digest_values),
        ("gloo-any-rank-failed", False),
        ("identity", 4, "cpu"),
        ("gloo-any-rank-failed", False),
        ("native-runner",),
        ("gloo-any-rank-failed", False),
        ("make-stages", world, admitted_root, identity, input_matrix, scorer,
         "proposal", False),
        ("gloo-any-rank-failed", False),
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

        def longs(self, values):
            events.append(("longs", values))
            return torch.tensor(values, dtype=torch.int64)

        def gather_fixed(self, tensor):
            values = tensor.tolist()
            events.append(("gather-fixed", values))
            return torch.stack([tensor for _ in range(self.size)])

        _stream5_library = None

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
    digest_values = list(expected_digest.digest())
    assert events == [
        ("prepare-input", caller_root, 1, 1, max_depth, "qhat", "python"),
        ("longs", digest_values),
        ("gather-fixed", digest_values),
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

def distributed_cnot_setup_failure(rank, rendezvous, failure_point):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    old_runner = cnot._native_stages_runner
    old_identity = cnot.native.cnot_identity
    old_compat = cnot._make_compat_stages
    old_input = cnot._input
    old_equal = torch.equal
    message = f"fixture {failure_point} failed on rank one"
    expected_stages = {
        "identity": "build CNOT identity",
        "native_runner": "load CNOT native stage runner",
        "compatibility_stages": "build CNOT compatibility stages",
        "root_goal": "check CNOT root goal",
    }
    try:
        require_controller()
        if failure_point == "identity":
            def build_identity(*args, **kwargs):
                if rank == 1:
                    raise RuntimeError(message)
                return old_identity(*args, **kwargs)
            cnot.native.cnot_identity = build_identity
        elif failure_point == "native_runner":
            def build_runner():
                if rank == 1:
                    raise RuntimeError(message)
                return old_runner()
            cnot._native_stages_runner = build_runner
        elif failure_point == "compatibility_stages":
            cnot._native_stages_runner = lambda: None
            def build_compatibility_stages(*args, **kwargs):
                if rank == 1:
                    raise RuntimeError(message)
                return old_compat(*args, **kwargs)
            cnot._make_compat_stages = build_compatibility_stages
        elif failure_point == "root_goal":
            def build_untyped_input(root, *_args, **_kwargs):
                return cnot._InputMatrix(root.shape[1], b"root-goal-fixture", root)
            cnot._input = build_untyped_input

            def fail_on_rank_one(left, right):
                if rank == 1:
                    raise RuntimeError(message)
                return old_equal(left, right)

            torch.equal = fail_on_rank_one
        else:
            raise AssertionError(f"unknown failure point: {failure_point}")

        with pytest.raises(RuntimeError) as error:
            cnot.search_cnot(
                cnot.pack_rows((8, 4, 2, 1)),
                WireFixture(9),
                width=5,
                shortlist=9,
                max_depth=4,
                mode="qhat",
                validation="python" if failure_point == "root_goal" else "auto",
                engine="rust",
            )
        expected = (
            f"{expected_stages[failure_point]} failed: "
            f"rank 1: RuntimeError: {message}"
        )
        assert str(error.value) == expected, (
            f"{failure_point} returned {str(error.value)!r}; expected {expected!r}"
        )

        # The coordinated preflight error must leave the caller's Gloo world usable.
        probe = torch.tensor([rank + 1], dtype=torch.int64)
        dist.all_reduce(probe)
        assert probe.item() == 3
    finally:
        cnot._native_stages_runner = old_runner
        cnot.native.cnot_identity = old_identity
        cnot._make_compat_stages = old_compat
        cnot._input = old_input
        torch.equal = old_equal
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="two CPU ranks require Gloo")
@pytest.mark.parametrize("failure_point", [
    "identity", "native_runner", "compatibility_stages", "root_goal",
])
def test_two_gloo_ranks_coordinate_cnot_setup_failures_and_keep_the_world(
        tmp_path, failure_point):
    require_controller()
    rendezvous = (tmp_path / f"cnot-setup-{failure_point}").resolve().as_uri()
    mp.spawn(
        distributed_cnot_setup_failure,
        args=(rendezvous, failure_point),
        nprocs=2,
    )


@pytest.mark.parametrize("mode", ["qhat", "proposal"])
def test_rust_compatibility_backend_owns_cnot_expand_dispatch(monkeypatch, mode):
    require_controller()
    from multigpubeamsearch.cnot_rust_stages import CnotStages

    rows = cnot.replay_cnot(4, (0, 1))
    options = dict(width=12, shortlist=12, max_depth=2, mode=mode, audit=True)
    reference = cnot.search_cnot(
        cnot.pack_rows(rows), WireFixture(9), engine="python", **options
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("Rust compatibility backend must dispatch expansion directly")

    monkeypatch.setattr(CnotStages, "expand", forbidden)
    monkeypatch.setattr(cnot, "_native_stages_runner", lambda: None)
    result = cnot.search_cnot(
        cnot.pack_rows(rows), WireFixture(9), engine="rust", **options
    )
    assert asdict(result) == asdict(reference)
    assert result.status == "solved"
    assert cnot.replay_cnot(4, result.gates) == rows

@pytest.mark.parametrize("mode", ["qhat", "proposal"])
def test_rust_compatibility_backend_owns_cnot_deduplicate_score_dispatch(monkeypatch, mode):
    require_controller()
    from multigpubeamsearch.cnot_rust_stages import CnotStages

    rows = cnot.replay_cnot(4, (0, 1))
    options = dict(width=12, shortlist=12, max_depth=2, mode=mode, audit=True)
    reference = cnot.search_cnot(
        cnot.pack_rows(rows), WireFixture(9), engine="python", **options
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("Rust compatibility backend must dispatch deduplication and scoring directly")

    monkeypatch.setattr(CnotStages, "deduplicate_and_score", forbidden)
    monkeypatch.setattr(cnot, "_native_stages_runner", lambda: None)
    result = cnot.search_cnot(
        cnot.pack_rows(rows), WireFixture(9), engine="rust", **options
    )
    assert asdict(result) == asdict(reference)
    assert result.status == "solved"
    assert cnot.replay_cnot(4, result.gates) == rows

@pytest.mark.parametrize("mode", ["qhat", "proposal"])
def test_rust_compatibility_backend_owns_cnot_redistribute_dispatch(monkeypatch, mode):
    require_controller()
    from multigpubeamsearch.cnot_rust_stages import CnotStages

    rows = cnot.replay_cnot(4, (0, 1))
    options = dict(width=12, shortlist=12, max_depth=2, mode=mode, audit=True)
    reference = cnot.search_cnot(
        cnot.pack_rows(rows), WireFixture(9), engine="python", **options
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("Rust compatibility backend must dispatch redistribution directly")

    monkeypatch.setattr(CnotStages, "redistribute", forbidden)
    monkeypatch.setattr(cnot, "_native_stages_runner", lambda: None)
    result = cnot.search_cnot(
        cnot.pack_rows(rows), WireFixture(9), engine="rust", **options
    )
    assert asdict(result) == asdict(reference)
    assert result.status == "solved"
    assert cnot.replay_cnot(4, result.gates) == rows


@pytest.mark.parametrize("mode", ["qhat", "proposal"])
def test_rust_compatibility_backend_owns_cnot_selection_dispatch(monkeypatch, mode):
    require_controller()
    from multigpubeamsearch.cnot_rust_stages import CnotStages

    rows, scorer = (8, 4, 2, 1), WireFixture(9)
    options = dict(width=5, shortlist=9, max_depth=4, mode=mode, audit=True)
    reference = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="python", **options)
    assert reference.layers and any(layer["distinct"] > options["width"] for layer in reference.layers)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Rust compatibility backend must dispatch selection directly")

    monkeypatch.setattr(CnotStages, "prepare_selection", forbidden)
    monkeypatch.setattr(CnotStages, "commit_selection", forbidden)
    monkeypatch.setattr(cnot, "_native_stages_runner", lambda: None)
    result = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="rust", **options)
    assert asdict(result) == asdict(reference)
    assert verify_layers(
        rows, scorer, mode, options["width"], options["shortlist"],
        result.layers, torch.device("cpu"),
    )


@pytest.mark.parametrize("mode", ["qhat", "proposal"])
def test_rust_compatibility_backend_bypasses_cnot_python_expand(monkeypatch, mode):
    require_controller()

    rows = cnot.replay_cnot(4, (0, 1))
    options = dict(width=12, shortlist=12, max_depth=2, mode=mode, audit=True)
    reference = cnot.search_cnot(
        cnot.pack_rows(rows), WireFixture(9), engine="python", **options
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("typed Rust expansion must not call cnot._expand")

    monkeypatch.setattr(cnot, "_expand", forbidden)
    monkeypatch.setattr(cnot, "_scores", forbidden)
    monkeypatch.setattr(cnot, "_native_stages_runner", lambda: None)
    result = cnot.search_cnot(
        cnot.pack_rows(rows), WireFixture(9), engine="rust", **options
    )
    assert asdict(result) == asdict(reference)
    assert result.status == "solved"
    assert cnot.replay_cnot(4, result.gates) == rows


@pytest.mark.parametrize("mode", ["qhat", "proposal"])
def test_rust_compatibility_backend_bypasses_cnot_python_distinct(monkeypatch, mode):
    require_controller()

    rows = cnot.replay_cnot(4, (0, 1))
    options = dict(width=12, shortlist=12, max_depth=2, mode=mode, audit=True)
    reference = cnot.search_cnot(
        cnot.pack_rows(rows), WireFixture(9), engine="python", **options
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("typed Rust deduplication must not call cnot._distinct")

    monkeypatch.setattr(cnot, "_distinct", forbidden)
    monkeypatch.setattr(cnot, "_scores", forbidden)
    monkeypatch.setattr(cnot, "_native_stages_runner", lambda: None)
    result = cnot.search_cnot(
        cnot.pack_rows(rows), WireFixture(9), engine="rust", **options
    )
    assert asdict(result) == asdict(reference)
    assert result.status == "solved"
    assert cnot.replay_cnot(4, result.gates) == rows


@pytest.mark.parametrize("mode", ["qhat", "proposal"])
def test_rust_compatibility_backend_bypasses_cnot_native_route_helper(monkeypatch, mode):
    require_controller()

    rows = cnot.replay_cnot(4, (0, 1))
    options = dict(width=12, shortlist=12, max_depth=2, mode=mode, audit=True)
    reference = cnot.search_cnot(
        cnot.pack_rows(rows), WireFixture(9), engine="python", **options
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("typed Rust expansion must not call cnot_native.pack_cnot_route")

    monkeypatch.setattr(cnot.native, "pack_cnot_route", forbidden)
    monkeypatch.setattr(cnot, "_native_stages_runner", lambda: None)
    result = cnot.search_cnot(
        cnot.pack_rows(rows), WireFixture(9), engine="rust", **options
    )
    assert asdict(result) == asdict(reference)
    assert result.status == "solved"
    assert cnot.replay_cnot(4, result.gates) == rows

def test_rust_cpu_search_applies_children_without_python_tensor_helper(monkeypatch):
    require_controller()
    from multigpubeamsearch import cnot_native

    rows, scorer = (8, 4, 2, 1), WireFixture(9)
    options = dict(width=5, shortlist=9, max_depth=4, mode="qhat", audit=True)
    reference = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="python", **options)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Rust CNOT CPU expansion called the Python tensor child helper")

    monkeypatch.setattr(cnot_native, "cnot_children", forbidden)
    result = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="rust", **options)

    assert asdict(result) == asdict(reference)


def test_rust_cpu_search_deduplicates_without_python_tensor_helper(monkeypatch):
    require_controller()
    from multigpubeamsearch import cnot_native

    rows, scorer = (8, 4, 2, 1), WireFixture(9)
    options = dict(width=5, shortlist=9, max_depth=4, mode="qhat", audit=True)
    reference = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="python", **options)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Rust CNOT deduplication called the Python tensor helper")

    monkeypatch.setattr(cnot_native, "cnot_first_unseen", forbidden)
    result = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="rust", **options)

    assert asdict(result) == asdict(reference)


def test_rust_cpu_search_assigns_owners_without_python_tensor_helper(monkeypatch):
    require_controller()
    from multigpubeamsearch import cnot_native

    rows, scorer = (8, 4, 2, 1), WireFixture(9)
    options = dict(width=5, shortlist=9, max_depth=4, mode="qhat", audit=True)
    reference = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="python", **options)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Rust CNOT owner assignment called the Python tensor helper")

    monkeypatch.setattr(cnot_native, "cnot_owners", forbidden)
    result = cnot.search_cnot(cnot.pack_rows(rows), scorer, engine="rust", **options)

    assert asdict(result) == asdict(reference)


def distributed_cnot_selection_histogram_failure(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    original_bincount = torch.bincount
    message = "fixture CNOT selection histogram failed on rank one"
    try:
        require_controller()
        from multigpubeamsearch.cnot_collectives import World

        selection = rust_control.extension().run_cnot_native_selection
        values = torch.tensor([float(rank), float(rank + 2)], dtype=torch.float32)
        word_keys = torch.tensor([rank * 2, rank * 2 + 1], dtype=torch.int64)
        with World("cpu") as world:
            if rank == 1:
                def fail_selection_histogram(tensor, *args, **kwargs):
                    if kwargs.get("minlength") == 256:
                        raise RuntimeError(message)
                    return original_bincount(tensor, *args, **kwargs)
                torch.bincount = fail_selection_histogram

            with pytest.raises(RuntimeError) as error:
                selection(world, values, word_keys, 1)
            expected = (
                "prepare CNOT selection histogram failed: "
                f"rank 1: RuntimeError: {message}"
            )
            assert str(error.value) == expected, (
                f"selection failure returned {str(error.value)!r}; expected {expected!r}"
            )

            probe = torch.tensor([rank + 1], dtype=torch.int64)
            dist.all_reduce(probe)
            assert probe.item() == 3
    finally:
        torch.bincount = original_bincount
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="two CPU ranks require Gloo")
def test_two_gloo_ranks_coordinate_cnot_selection_histogram_failure_and_keep_world(
        tmp_path):
    require_controller()
    rendezvous = (tmp_path / "cnot-selection-histogram").resolve().as_uri()
    mp.spawn(
        distributed_cnot_selection_histogram_failure,
        args=(rendezvous,),
        nprocs=2,
    )


def distributed_cnot_selection_failure_after_gather(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    original_argsort = torch.argsort
    message = "fixture CNOT selection finish failed on rank one"
    try:
        require_controller()
        if rank == 1:
            def fail_singleton_sort(tensor, *args, **kwargs):
                if tensor.numel() == 1:
                    raise RuntimeError(message)
                return original_argsort(tensor, *args, **kwargs)
            torch.argsort = fail_singleton_sort

        with pytest.raises(RuntimeError) as error:
            cnot.search_cnot(
                cnot.pack_rows((8, 4, 2, 1)),
                WireFixture(9),
                width=1,
                shortlist=3,
                max_depth=3,
                mode="qhat",
                engine="rust",
            )
        expected = (
            "select CNOT frontier failed: "
            f"rank 1: RuntimeError: {message}"
        )
        assert str(error.value) == expected, (
            f"selection failure returned {str(error.value)!r}; expected {expected!r}"
        )

        probe = torch.tensor([rank + 1], dtype=torch.int64)
        dist.all_reduce(probe)
        assert probe.item() == 3
    finally:
        torch.argsort = original_argsort
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="two CPU ranks require Gloo")
def test_two_gloo_ranks_coordinate_cnot_selection_failure_after_gather_and_keep_world(
        tmp_path):
    require_controller()
    rendezvous = (tmp_path / "cnot-selection-finish").resolve().as_uri()
    mp.spawn(
        distributed_cnot_selection_failure_after_gather,
        args=(rendezvous,),
        nprocs=2,
    )


def distributed_cnot_exchange_prepare_failure(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    original_bincount = torch.bincount
    message = "fixture CNOT exchange packet failed on rank one"
    try:
        require_controller()
        if rank == 1:
            def fail_bincount(tensor, *args, **kwargs):
                raise RuntimeError(message)
            torch.bincount = fail_bincount

        with pytest.raises(RuntimeError) as error:
            cnot.search_cnot(
                cnot.pack_rows((8, 4, 2, 1)),
                WireFixture(9),
                width=5,
                shortlist=9,
                max_depth=4,
                mode="qhat",
                engine="rust",
            )
        expected = (
            "prepare CNOT record exchange failed: "
            f"rank 1: RuntimeError: {message}"
        )
        assert str(error.value) == expected, (
            f"exchange preparation returned {str(error.value)!r}; "
            f"expected {expected!r}"
        )

        probe = torch.tensor([rank + 1], dtype=torch.int64)
        dist.all_reduce(probe)
        assert probe.item() == 3
    finally:
        torch.bincount = original_bincount
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="two CPU ranks require Gloo")
def test_two_gloo_ranks_coordinate_cnot_exchange_preparation_failure_and_keep_world(
        tmp_path):
    require_controller()
    rendezvous = (tmp_path / "cnot-exchange-prepare").resolve().as_uri()
    mp.spawn(
        distributed_cnot_exchange_prepare_failure,
        args=(rendezvous,),
        nprocs=2,
    )


def distributed_cnot_exchange_payload_allocation_failure(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    original_new_empty = torch.Tensor.new_empty
    message = "fixture CNOT exchange payload allocation failed on rank one"
    try:
        require_controller()
        if rank == 1:
            def fail_receive_allocation(tensor, size, *args, **kwargs):
                shape = tuple(size)
                if tensor.ndim == 2 and tensor.shape[0] == 0 and len(shape) == 2 and shape[0] > 0:
                    raise RuntimeError(message)
                return original_new_empty(tensor, size, *args, **kwargs)
            torch.Tensor.new_empty = fail_receive_allocation

        with pytest.raises(RuntimeError) as error:
            cnot.search_cnot(
                cnot.pack_rows((8, 4, 2, 1)),
                WireFixture(9),
                width=5,
                shortlist=9,
                max_depth=4,
                mode="qhat",
                engine="rust",
            )
        expected = (
            "prepare CNOT record exchange payload failed: "
            f"rank 1: RuntimeError: {message}"
        )
        assert str(error.value) == expected, (
            f"exchange payload preparation returned {str(error.value)!r}; "
            f"expected {expected!r}"
        )

        probe = torch.tensor([rank + 1], dtype=torch.int64)
        dist.all_reduce(probe)
        assert probe.item() == 3
    finally:
        torch.Tensor.new_empty = original_new_empty
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="two CPU ranks require Gloo")
def test_two_gloo_ranks_coordinate_cnot_exchange_payload_failure_and_keep_world(
        tmp_path):
    require_controller()
    rendezvous = (tmp_path / "cnot-exchange-payload").resolve().as_uri()
    mp.spawn(
        distributed_cnot_exchange_payload_allocation_failure,
        args=(rendezvous,),
        nprocs=2,
    )


def distributed_cnot_keyboard_interrupt(rank, rendezvous):
    import signal

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    old_input = cnot._input
    old_equal = torch.equal
    try:
        require_controller()

        def build_untyped_input(root, *_args, **_kwargs):
            return cnot._InputMatrix(root.shape[1], b"interrupt-fixture", root)
        cnot._input = build_untyped_input

        if rank == 1:
            def interrupt_during_root_goal(left, right):
                signal.raise_signal(signal.SIGINT)
                return old_equal(left, right)

            torch.equal = interrupt_during_root_goal

        with pytest.raises(KeyboardInterrupt) as error:
            cnot.search_cnot(
                cnot.pack_rows((8, 4, 2, 1)),
                WireFixture(9),
                width=5,
                shortlist=9,
                max_depth=4,
                mode="qhat",
                validation="python",
                engine="rust",
            )
        assert "check CNOT root goal" in str(error.value)
        assert "rank 1: KeyboardInterrupt" in str(error.value)

        probe = torch.tensor([rank + 1], dtype=torch.int64)
        dist.all_reduce(probe)
        assert probe.item() == 3
    finally:
        cnot._input = old_input
        torch.equal = old_equal
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="two CPU ranks require Gloo")
def test_two_gloo_ranks_propagate_sigint_as_keyboard_interrupt_and_keep_world(tmp_path):
    require_controller()
    rendezvous = (tmp_path / "cnot-sigint").resolve().as_uri()
    mp.spawn(
        distributed_cnot_keyboard_interrupt,
        args=(rendezvous,),
        nprocs=2,
    )
