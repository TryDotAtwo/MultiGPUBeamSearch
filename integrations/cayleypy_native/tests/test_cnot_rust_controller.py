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
