"""CNOT stage failures are coordinated by Rust with the existing world vote."""
from datetime import timedelta
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from multigpubeamsearch import rust_control
from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.errors import NativeUnavailable


def require_checked():
    try:
        runner = getattr(rust_control.extension(), "run_cnot_native_checked", None)
    except NativeUnavailable:
        runner = None
    if runner is None:
        if os.environ.get("REQUIRE_RUST_CNOT") == "1":
            raise AssertionError("matching Rust CNOT stage error coordinator is required")
        pytest.skip("matching Rust CNOT stage error coordinator is unavailable")
    return runner


def test_single_rank_preserves_result_and_original_error():
    run = require_checked()
    world, value = World("cpu"), object()
    assert run(world, "stage", lambda: value) is value
    error = ValueError("bad score")

    def fail():
        raise error

    with pytest.raises(ValueError, match="bad score") as caught:
        run(world, "stage", fail)
    assert caught.value is error


def test_base_exception_does_not_enter_collective():
    run = require_checked()

    class Interrupt(BaseException):
        pass

    class ProbeWorld:
        size, rank = 2, 0

        def sum(self, value):
            raise AssertionError("unexpected collective")

    def fail():
        raise Interrupt("stop")

    with pytest.raises(Interrupt, match="stop"):
        run(ProbeWorld(), "stage", fail)


def test_error_uses_one_vote_then_existing_error_gather(monkeypatch):
    run = require_checked()
    votes = []

    class ProbeWorld:
        size, rank = 2, 0

        def longs(self, values):
            return torch.tensor(values, dtype=torch.int64)

        def sum(self, value):
            votes.append(value.tolist())
            return value

    monkeypatch.setattr(dist, "all_gather_object", lambda out, error: out.__setitem__(slice(None), [error, None]))

    def fail():
        raise ValueError("bad score")

    with pytest.raises(RuntimeError, match="rank parent actions and expand failed: rank 0: ValueError: bad score"):
        run(ProbeWorld(), "rank parent actions and expand", fail)
    assert votes == [[1]]


def distributed(rank, rendezvous, fault_rank):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        run, world = require_checked(), World("cpu")
        calls = []

        def action():
            calls.append(rank)
            if rank == fault_rank or fault_rank == 2:
                raise ValueError("bad score")
            return rank

        with pytest.raises(RuntimeError, match="stage failed") as error:
            run(world, "stage", action)
        assert calls == [rank]
        expected = (0, 1) if fault_rank == 2 else (fault_rank,)
        assert str(error.value) == "stage failed: " + " | ".join(
            f"rank {source}: ValueError: bad score" for source in expected)
        assert run(world, "next stage", lambda: rank) == rank
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available() or sys.platform == "win32",
                    reason="requires two Gloo ranks")
@pytest.mark.parametrize("fault_rank", [0, 1, 2])
def test_error_vote_is_collective_and_group_is_reusable(tmp_path, fault_rank):
    require_checked()
    mp.spawn(distributed, args=((tmp_path / "checked-gloo").resolve().as_uri(), fault_rank), nprocs=2)
