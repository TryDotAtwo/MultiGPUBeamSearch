"""Rust CNOT goal coordination preserves the reference and error contract."""
from datetime import timedelta
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from multigpubeamsearch import cnot, cnot_native, rust_control
from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.errors import NativeUnavailable


def require_goal():
    try:
        runner = getattr(rust_control.extension(), "run_cnot_native_goal", None)
    except NativeUnavailable:
        runner = None
    if runner is None:
        if os.environ.get("REQUIRE_RUST_CNOT") == "1":
            raise AssertionError("matching Rust CNOT goal coordinator is required")
        pytest.skip("matching Rust CNOT goal coordinator is unavailable")
    return runner


@pytest.mark.parametrize("n", [4, 65, 144])
@pytest.mark.parametrize("count", [0, 1, 4])
def test_native_goal_matches_reference(n, count):
    run = require_goal()
    identity = cnot_native.cnot_identity(n, device="cpu")
    moves = torch.tensor([0, n * (n - 1) - 1, 1, 2])
    parents = cnot_native.cnot_children(identity.expand(4, -1, -1).contiguous(),
                                         torch.arange(4), moves)
    parents[0] = identity[0]
    # Include noncontiguous storage and a lexicographic order unlike row order.
    parents = torch.stack((parents, parents), 1).reshape(8, n, -1)[::2][:count]
    lex = torch.tensor([6, 3, 1, 5], dtype=torch.int64)[:count]
    world = World("cpu")
    assert run(world, parents, lex, 7, identity) == cnot._goal(world, parents, lex, 7, identity)


def test_native_goal_preserves_original_single_rank_exception(monkeypatch):
    run = require_goal()
    parents, identity = cnot.pack_rows((1, 3)), cnot.pack_rows((1, 2))
    lex = torch.tensor([0])

    def failure(*args, **kwargs):
        raise ValueError("goal calculation failed")

    monkeypatch.setattr(torch, "arange", failure)
    with pytest.raises(ValueError, match="goal calculation failed"):
        run(World("cpu"), parents, lex, 0, identity)


def test_native_goal_failure_uses_existing_rank_exchange(monkeypatch):
    run = require_goal()
    parents, identity = cnot.pack_rows((1, 3)), cnot.pack_rows((1, 2))
    lex, exchanged = torch.tensor([0]), []

    class ProbeWorld:
        size, rank, device = 2, 0, torch.device("cpu")

        def longs(self, values):
            return torch.tensor(values, dtype=torch.int64)

        def gather_fixed(self, value):
            exchanged.append(value.tolist())
            return torch.stack((value, torch.tensor([2**63 - 1, -1, 0])))

    def failure(*args, **kwargs):
        raise ValueError("goal calculation failed")

    monkeypatch.setattr(torch, "arange", failure)
    monkeypatch.setattr(dist, "all_gather_object", lambda output, error: output.__setitem__(slice(None), [error, None]))
    with pytest.raises(RuntimeError, match="find CNOT goal failed: rank 0: ValueError: goal calculation failed"):
        run(ProbeWorld(), parents, lex, 0, identity)
    assert len(exchanged) == 1 and exchanged[0][-1] == 1


def test_native_goal_does_not_convert_base_exception_to_collective_error(monkeypatch):
    run = require_goal()
    parents, identity = cnot.pack_rows((1, 3)), cnot.pack_rows((1, 2))

    class ProbeWorld:
        size, rank, device = 2, 0, torch.device("cpu")

        def gather_fixed(self, value):
            raise AssertionError("unexpected collective")

    class Interrupt(BaseException):
        pass

    def interrupt(*args, **kwargs):
        raise Interrupt("stop")

    monkeypatch.setattr(torch, "arange", interrupt)
    with pytest.raises(Interrupt, match="stop"):
        run(ProbeWorld(), parents, torch.tensor([0]), 0, identity)


def distributed(rank, rendezvous, fault_rank):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        run = require_goal()
        world = World("cpu")
        parents, identity = cnot.pack_rows((1, 3)), cnot.pack_rows((1, 2))
        lex = world.longs([rank])
        local = parents[:0] if rank == 0 else parents
        local_lex = lex[:0] if rank == 0 else lex
        assert run(world, local, local_lex, rank, identity) == cnot._goal(
            world, local, local_lex, rank, identity)
        original = torch.arange
        fired = False

        def injected(*args, **kwargs):
            nonlocal fired
            if (rank == fault_rank or fault_rank == 2) and not fired:
                fired = True
                raise ValueError("goal calculation failed")
            return original(*args, **kwargs)

        torch.arange = injected
        try:
            with pytest.raises(RuntimeError, match="find CNOT goal failed") as error:
                run(world, parents, lex, rank, identity)
            for source in ((0, 1) if fault_rank == 2 else (fault_rank,)):
                assert f"rank {source}: ValueError: goal calculation failed" in str(error.value)
        finally:
            torch.arange = original
        assert run(world, parents, lex, rank, identity) == (0, 0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available() or sys.platform == "win32",
                    reason="requires two Gloo ranks")
@pytest.mark.parametrize("fault_rank", [0, 1, 2])
def test_native_goal_failure_is_collective_and_group_is_reusable(tmp_path, fault_rank):
    require_goal()
    mp.spawn(distributed, args=((tmp_path / "goal-gloo").resolve().as_uri(), fault_rank), nprocs=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("missing", [False, True])
def test_native_goal_cuda_and_old_operator_fallback(monkeypatch, missing):
    run = require_goal()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    identity = cnot_native.cnot_identity(65, device=device)
    parents = cnot_native.cnot_children(identity, torch.tensor([0], device=device),
                                         torch.tensor([64], device=device))
    lex, world = torch.tensor([5], device=device), World(device)
    expected = cnot._goal(world, parents, lex, 7, identity)
    ops = cnot_native._cuda_ops()

    class OlderLibrary:
        def __getattr__(self, name):
            if missing and name == "local_goal":
                raise AttributeError(name)
            return getattr(ops, name)

    monkeypatch.setattr(cnot_native, "_cuda_ops", lambda: OlderLibrary())
    assert run(world, parents, lex, 7, identity) == expected
