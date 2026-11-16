"""Native local-goal failures reach every NCCL rank and permit group reuse."""
from datetime import timedelta
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

from multigpubeamsearch import cnot, cnot_native
from multigpubeamsearch import rust_control
from multigpubeamsearch.cnot_collectives import World

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from verify_cnot import WireFixture


device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group("nccl", timeout=timedelta(seconds=60))
rank, size = dist.get_rank(), dist.get_world_size()
original = cnot_native._cuda_ops
try:
    assert size == 2
    ops = original()
    assert hasattr(ops, "local_goal")
    run = rust_control.extension().run_cnot_native_goal
    world = World(device)
    identity = cnot_native.cnot_identity(4, device=device)
    parent = cnot_native.cnot_children(identity, torch.tensor([0], device=device),
                                        torch.tensor([rank], device=device))
    lex = world.longs([1 - rank])
    expected = cnot._goal(world, parent, lex, rank, identity)

    class MixedOperators:
        def __getattr__(self, name):
            if name == "local_goal" and rank == 1:
                raise AttributeError(name)
            return getattr(ops, name)

    cnot_native._cuda_ops = lambda: MixedOperators()
    assert run(world, parent, lex, rank, identity) == expected
    cnot_native._cuda_ops = original

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy Python goal coordinator was called")

    cnot._goal = forbidden
    for fault_rank in range(size):
        fired = [False]

        class FaultOps:
            def __getattr__(self, name):
                if name == "local_goal" and rank == fault_rank:
                    def fail(*args):
                        fired[0] = True
                        raise ValueError("injected native goal failure")
                    return fail
                return getattr(ops, name)

        cnot_native._cuda_ops = lambda: FaultOps()
        try:
            root = cnot.pack_rows((8, 4, 2, 1), device=device)
            cnot.search_cnot(root, WireFixture(9), width=5, shortlist=9, max_depth=4, engine="rust")
        except RuntimeError as error:
            assert f"find CNOT goal failed: rank {fault_rank}: ValueError: injected native goal failure" in str(error)
        else:
            raise AssertionError("native goal failure did not reach every rank")
        finally:
            cnot_native._cuda_ops = original
        if rank == fault_rank:
            assert fired[0]
        result = cnot.search_cnot(cnot.pack_rows((1, 3), device=device), object(),
                                  width=1, shortlist=1, max_depth=1, engine="rust")
        assert result.status == "solved" and result.gates == (0,)
        dist.barrier()
    print(f"CNOT_NATIVE_GOAL_NCCL_OK rank={rank} cases={size} mixed_operators=1", flush=True)
finally:
    cnot_native._cuda_ops = original
    dist.destroy_process_group()
