"""Exercise Rust-owned Stream 5 communicator lifecycle on two NCCL ranks."""
from datetime import timedelta
import gc
import os
from pathlib import Path

import torch
import torch.distributed as dist

from multigpubeamsearch.cnot_stream5 import load_stream5
from multigpubeamsearch.rust_control import extension


def main():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    try:
        path = Path(load_stream5()._name).resolve(strict=True)
        library = extension().CnotStream5Library(str(path))
        for abort in (False, True, False):
            identity = library.unique_id()
            payload = torch.tensor(list(identity), dtype=torch.uint8, device=device)
            dist.broadcast(payload, 0)
            communicator = library.init(bytes(payload.tolist()), rank, world)
            assert not communicator.closed
            dist.barrier()
            communicator.close(abort)
            assert communicator.closed
            communicator.close(abort)
            del communicator
            gc.collect()
            probe = torch.tensor([rank + 1], device=device)
            dist.all_reduce(probe)
            assert probe.item() == 3
        print(f"CNOT_RUST_STREAM5_LIFECYCLE_OK rank={rank} rounds=3", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
