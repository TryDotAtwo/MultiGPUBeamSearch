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
        for mode in ("close", "abort", "drop"):
            identity = library.unique_id()
            payload = torch.tensor(list(identity), dtype=torch.uint8, device=device)
            dist.broadcast(payload, 0)
            communicator = library.init(bytes(payload.tolist()), rank, world)
            assert not communicator.closed
            dist.barrier()
            # The device on the owning thread can change between initialization
            # and cleanup. Cleanup must use the communicator's original device
            # and return the caller to this deliberately changed device.
            other_device = 1 - device.index
            torch.cuda.set_device(other_device)
            if mode == "drop":
                del communicator
                gc.collect()
            else:
                communicator.close(mode == "abort")
                assert communicator.closed
                communicator.close(mode == "abort")
                del communicator
                gc.collect()
            assert torch.cuda.current_device() == other_device
            torch.cuda.set_device(device)
            probe = torch.tensor([rank + 1], device=device)
            dist.all_reduce(probe)
            assert probe.item() == 3
        print(f"CNOT_RUST_STREAM5_DEVICE_OK rank={rank} rounds=3", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
