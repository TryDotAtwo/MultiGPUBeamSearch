"""A local goal-evaluation error must reach every CNOT rank."""
from datetime import timedelta
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from multigpubeamsearch import cnot
from multigpubeamsearch.cnot_collectives import World


def test_goal_local_error_enters_the_existing_collective(monkeypatch):
    parents = cnot.pack_rows((1, 3))
    identity = cnot.pack_rows((1, 2))
    lex = torch.tensor([0], dtype=torch.int64)
    exchanged = []

    class ProbeWorld:
        size, rank, device = 2, 0, torch.device("cpu")

        def longs(self, values):
            return torch.tensor(values, dtype=torch.int64)

        def gather_fixed(self, value):
            exchanged.append(value.tolist())
            return torch.stack((value, torch.tensor([2**63 - 1, -1, 0])))

    def failure(*args, **kwargs):
        raise ValueError("goal calculation failed")

    def gather_errors(output, error):
        output[:] = [error, None]

    monkeypatch.setattr(cnot.torch, "arange", failure)
    monkeypatch.setattr(cnot.dist, "all_gather_object", gather_errors)
    with pytest.raises(RuntimeError, match="find CNOT goal failed: rank 0: ValueError: goal calculation failed"):
        cnot._goal(ProbeWorld(), parents, lex, 0, identity)
    assert len(exchanged) == 1 and exchanged[0][-1] == 1


def test_single_rank_goal_keeps_the_original_exception(monkeypatch):
    parents = cnot.pack_rows((1, 3))
    identity = cnot.pack_rows((1, 2))
    lex = torch.tensor([0], dtype=torch.int64)

    def failure(*args, **kwargs):
        raise ValueError("goal calculation failed")

    monkeypatch.setattr(cnot.torch, "arange", failure)
    with pytest.raises(ValueError, match="goal calculation failed"):
        cnot._goal(World("cpu"), parents, lex, 0, identity)


def _worker(rank, rendezvous, output_dir, fault_rank):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=8),
    )
    try:
        world = World("cpu")
        parents = cnot.pack_rows((1, 3))
        identity = cnot.pack_rows((1, 2))
        lex = world.longs([rank])
        original = torch.arange
        fired = [False]

        def injected(*args, **kwargs):
            if (rank == fault_rank or fault_rank == 2) and not fired[0]:
                fired[0] = True
                raise ValueError("goal calculation failed")
            return original(*args, **kwargs)

        torch.arange = injected
        result = {}
        try:
            cnot._goal(world, parents, lex, rank, identity)
        except Exception as error:
            result["kind"] = type(error).__name__
            result["error"] = str(error)
        else:
            result["kind"] = "accepted"
        finally:
            torch.arange = original

        if result["kind"] == "RuntimeError" and "find CNOT goal failed" in result["error"]:
            # A healthy group can perform the same goal collective immediately.
            result["reused"] = cnot._goal(world, parents, lex, rank, identity) == (0, 0)
        Path(output_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="two CPU ranks require Gloo")
@pytest.mark.parametrize("fault_rank", [0, 1, 2])
def test_rank_local_goal_failure_is_collective_and_group_is_reusable(tmp_path, fault_rank):
    if sys.platform == "win32":
        try:
            dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")
        except RuntimeError as error:
            if "unsupported gloo device" not in str(error):
                raise
            pytest.skip("this Windows torch wheel has no Gloo transport")
    rendezvous = (tmp_path / "rendezvous").resolve().as_uri()
    mp.spawn(_worker, args=(rendezvous, str(tmp_path), fault_rank), nprocs=2)
    results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    for result in results:
        assert result["kind"] == "RuntimeError", results
        expected_ranks = (0, 1) if fault_rank == 2 else (fault_rank,)
        for expected_rank in expected_ranks:
            assert f"rank {expected_rank}: ValueError: goal calculation failed" in result["error"], results
        assert result["reused"] is True
