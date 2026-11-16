"""Rust stage error vote preserves NCCL ordering and permits group reuse."""
from datetime import timedelta
import os

import torch
import torch.distributed as dist

from multigpubeamsearch import rust_control
from multigpubeamsearch.cnot_collectives import World


device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group("nccl", timeout=timedelta(seconds=60))
rank, size = dist.get_rank(), dist.get_world_size()
try:
    assert size == 2
    run, world = rust_control.extension().run_cnot_native_checked, World(device)
    for fault_rank in (0, 1, 2):
        calls = []

        def action():
            calls.append(rank)
            if rank == fault_rank or fault_rank == 2:
                raise ValueError("injected stage failure")
            return rank

        try:
            run(world, "deduplicate and evaluate distinct children", action)
        except RuntimeError as error:
            expected = (0, 1) if fault_rank == 2 else (fault_rank,)
            assert str(error) == "deduplicate and evaluate distinct children failed: " + " | ".join(
                f"rank {source}: ValueError: injected stage failure" for source in expected)
        else:
            raise AssertionError("stage failure did not reach every rank")
        assert calls == [rank]
        assert run(world, "next stage", lambda: rank) == rank
        dist.barrier()
    print(f"CNOT_RUST_CHECKED_NCCL_OK rank={rank} cases=3", flush=True)
finally:
    dist.destroy_process_group()
