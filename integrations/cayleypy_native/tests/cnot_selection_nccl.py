"""Compare global CNOT cutoff controllers with real multi-rank NCCL."""
from datetime import timedelta
import os

import torch
import torch.distributed as dist

from multigpubeamsearch import cnot_native
from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.cnot_rust_selection import SelectionStages
from multigpubeamsearch.cnot_selection import global_winners
from multigpubeamsearch.rust_control import extension


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        world = World(device)
        assert world.size == 2
        cases = (
            (([0., -2., 1e300, -0.], [1e-280, -1e300, 1.00001, 0.]),
             ([11, 10, 15, 9], [8, 3, 4, 5])),
            (([], [0., 0., 0.]), ([], [30, 10, 20])),
            (([1., 1., 1.], [1., 1.]), ([9, 3, 8], [2, 1])),
        )
        for values, words in cases:
            scores = torch.tensor(values[world.rank], dtype=torch.float64, device=device)
            keys = torch.tensor(words[world.rank], dtype=torch.int64, device=device)
            for width in range(1, sum(map(len, values)) + 2):
                old = global_winners(world, scores, keys, width)
                new = extension().run_cnot_selection(SelectionStages(world, scores, keys), width)
                native = extension().run_cnot_native_selection(world, scores, keys, width)
                assert all(torch.equal(a, b) for a, b in zip(old, new))
                assert all(torch.equal(a, b) for a, b in zip(old, native))

        original_ops = cnot_native._cuda_ops
        ops = original_ops()

        class OlderLibrary:
            def __getattr__(self, name):
                if name in ("selection_prepare", "selection_records", "selection_order",
                            "radix_histogram", "radix_update"):
                    raise AttributeError(name)
                return getattr(ops, name)

        if world.rank == 1:
            cnot_native._cuda_ops = lambda: OlderLibrary()
        values, words = cases[1]
        scores = torch.tensor(values[world.rank], dtype=torch.float64, device=device)
        keys = torch.tensor(words[world.rank], dtype=torch.int64, device=device)
        for width in range(1, sum(map(len, values)) + 2):
            old = global_winners(world, scores, keys, width)
            native = extension().run_cnot_native_selection(world, scores, keys, width)
            assert all(torch.equal(a, b) for a, b in zip(old, native))
        cnot_native._cuda_ops = original_ops
        print(f"CNOT_RADIX_MIXED_OPS_NCCL_OK rank={world.rank}", flush=True)
        print(f"CNOT_RADIX_NCCL_OK rank={world.rank} cases={len(cases)}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
