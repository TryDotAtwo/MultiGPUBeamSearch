"""Compare Python and Rust record preparation over the same NCCL transport."""
from datetime import timedelta
import os
import time

import torch
import torch.distributed as dist

from multigpubeamsearch import cnot_native
from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.cnot_stream5 import RustStream5
from multigpubeamsearch.rust_control import extension


class OlderOps:
    def __init__(self, wrapped, missing):
        self.wrapped, self.missing = wrapped, missing

    def __getattr__(self, name):
        if name in self.missing:
            raise AttributeError(name)
        return getattr(self.wrapped, name)


def main():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    rank, size = dist.get_rank(), dist.get_world_size()
    assert size == 2
    original_ops = cnot_native._cuda_ops
    try:
        with World(device) as world:
            world.start_transport(prefer_rust=True)
            assert isinstance(world.transport, RustStream5)
            exchange = extension().run_cnot_native_exchange
            cases = [
                (([[10, 11], [20, 21], [22, 23]], [1, 0, 1]), ([[30, 31]], [0])),
                (([], []), ([[40, 41]], [0])),
                (([], []), ([], [])),
                (([[50, 51]], [0]), ([[60, 61]], [1])),
            ]
            expected = [([[20, 21], [30, 31]], [[10, 11], [22, 23]]),
                        ([[40, 41]], []), ([], []), ([[50, 51]], [[60, 61]])]
            for scenario in ("native", "fallback", "mixed"):
                missing = ({"exchange_packet"} if scenario == "fallback" or
                           (scenario == "mixed" and rank == 1) else set())
                cnot_native._cuda_ops = lambda: OlderOps(original_ops(), missing)
                for index, case in enumerate(cases):
                    values, destinations = case[rank]
                    rows = torch.tensor(values, dtype=torch.int64, device=device).reshape(-1, 2)
                    targets = torch.tensor(destinations, dtype=torch.int64, device=device)
                    if index == 1 and rank == 1:
                        time.sleep(0.2)
                    before = world.exchange(rows, targets, native_packet=True)
                    after = exchange(world, rows, targets)
                    assert torch.equal(after, before)
                    assert after.tolist() == expected[index][rank], (rank, scenario, index)
            dist.barrier()
        print(f"CNOT_NATIVE_RECORD_EXCHANGE_OK rank={rank} scenarios=3 cases=4", flush=True)
    finally:
        cnot_native._cuda_ops = original_ops
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
