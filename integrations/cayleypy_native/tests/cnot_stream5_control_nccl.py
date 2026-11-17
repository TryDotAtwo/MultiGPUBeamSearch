"""Exercise Rust-owned CNOT collectives and record exchange on two NCCL ranks."""
from datetime import timedelta
import os
import time

import torch
import torch.distributed as dist

from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.cnot_stream5 import RustStream5


def main():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    rank, size = dist.get_rank(), dist.get_world_size()
    assert size == 2
    try:
        with World(device) as world:
            world.start_transport(prefer_rust=True)
            assert isinstance(world.transport, RustStream5), (
                "all ranks must select the Rust Stream 5 exchange and collective ABIs"
            )

            reduced = world.longs([rank + 1, rank + 4])
            assert world.sum(reduced).tolist() == [3, 9]

            strided = (torch.arange(8, device=device, dtype=torch.int64).reshape(2, 4) + rank)[:, ::2]
            assert not strided.is_contiguous()
            assert world.sum(strided).tolist() == [[1, 5], [9, 13]]

            gathered = world.gather_fixed(world.longs([rank, rank + 10]))
            assert gathered.tolist() == [[0, 10], [1, 11]]
            empty_gather = world.gather_fixed(torch.empty(0, dtype=torch.int64, device=device))
            assert tuple(empty_gather.shape) == (2, 0)

            broadcast = world.longs([100 + rank])
            world.broadcast(broadcast, 1)
            assert broadcast.tolist() == [101]

            rows = (
                torch.tensor([[20, 21], [22, 23]], dtype=torch.int64, device=device)
                if rank == 1
                else torch.empty((0, 2), dtype=torch.int64, device=device)
            )
            if rank == 0:
                time.sleep(0.15)
            assert world.gather_rows(rows).tolist() == [[20, 21], [22, 23]]

            def fail_on_rank_one():
                if rank == 1:
                    raise ValueError("rank-local control-collective probe")

            try:
                world.checked("control-collective failure probe", fail_on_rank_one)
            except RuntimeError as error:
                assert "rank 1: ValueError: rank-local control-collective probe" in str(error)
            else:
                raise AssertionError("rank-local failure was not coordinated")
            assert world.sum(world.longs([1])).item() == 2

            sent = (
                torch.tensor([[30, 31]], dtype=torch.int64, device=device)
                if rank == 0
                else torch.empty((0, 2), dtype=torch.int64, device=device)
            )
            targets = (
                torch.tensor([1], dtype=torch.int64, device=device)
                if rank == 0
                else torch.empty(0, dtype=torch.int64, device=device)
            )
            if rank == 1:
                time.sleep(0.15)
            received = world.exchange(sent, targets)
            assert received.tolist() == ([[30, 31]] if rank == 1 else [])

            empty = torch.empty((0, 2), dtype=torch.int64, device=device)
            no_targets = torch.empty(0, dtype=torch.int64, device=device)
            assert tuple(world.exchange(empty, no_targets).shape) == (0, 2)

        print(f"CNOT_CONTROL_COLLECTIVES_OK rank={rank} world={size}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
