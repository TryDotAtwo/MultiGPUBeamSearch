"""Rust-owned packed CNOT record exchanges on two physical NCCL ranks."""
from datetime import timedelta
from pathlib import Path
import os
import time

import torch
import torch.distributed as dist

from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.cnot_stream5 import load_stream5
from multigpubeamsearch.rust_control import extension


def main():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    rank, size = dist.get_rank(), dist.get_world_size()
    assert size == 2
    try:
        library = extension().CnotStream5Library(str(Path(load_stream5()._name).resolve(strict=True)))
        identity = library.unique_id()
        payload = torch.tensor(list(identity), dtype=torch.uint8, device=device)
        dist.broadcast(payload, 0)
        communicator = library.init(bytes(payload.tolist()), rank, size)
        world = World(device)
        cases = [
            (([[10, 11], [20, 21], [22, 23]], [1, 2]),
             ([[30, 31]], [1, 0]),
             ([[10, 11], [30, 31]], [[20, 21], [22, 23]])),
            (([], [0, 0]), ([[40, 41]], [1, 0]), ([[40, 41]], [])),
            (([], [0, 0]), ([], [0, 0]), ([], [])),
        ]
        for case, (first, second, expected) in enumerate(cases):
            rows, counts = (first, second)[rank]
            send = torch.tensor(rows, dtype=torch.int64, device=device).reshape(-1, 2)
            counts = torch.tensor(counts, dtype=torch.int64, device=device)
            if case == 1 and rank == 1:
                time.sleep(0.2)
            received = communicator.exchange(send, counts, world)
            assert received.dtype == torch.int64 and received.device == device
            assert received.shape == (len(expected[rank]), 2)
            assert received.tolist() == expected[rank], (rank, case, received.tolist())
        # An inconsistent sender is detected collectively before payload calls;
        # the communicator can then carry another valid exchange.
        send = torch.tensor([[50 + rank, 60 + rank]], dtype=torch.int64, device=device)
        bad = torch.tensor(([1, 0] if rank == 0 else [0, 0]), dtype=torch.int64, device=device)
        try:
            communicator.exchange(send, bad, world)
        except (ValueError, OverflowError):
            pass
        else:
            raise AssertionError("inconsistent send counts were not rejected on every rank")
        dist.barrier()
        counts = torch.tensor(([1, 0] if rank == 0 else [0, 1]), dtype=torch.int64, device=device)
        assert communicator.exchange(send, counts, world).tolist() == [[50 + rank, 60 + rank]]
        communicator.close(False)
        print(f"CNOT_RUST_STREAM5_EXCHANGE_OK rank={rank} cases={len(cases)}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
