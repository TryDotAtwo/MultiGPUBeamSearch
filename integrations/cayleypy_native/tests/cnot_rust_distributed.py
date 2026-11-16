"""Physical GPU/NCCL equivalence and coordinated-failure acceptance fixture."""
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from verify_cnot import WireFixture, wide_case
from multigpubeamsearch import cnot


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    rank = dist.get_rank() if distributed else 0
    size = dist.get_world_size() if distributed else 1
    try:
        cases = []
        wide_rows, _ = wide_case()
        for rows, undo, mode, width, k, depth in (
            ((8, 4, 2, 1), 9, "qhat", 5, 9, 4),
            ((8, 4, 2, 1), 9, "proposal", 5, 9, 4),
            ((8, 4, 2, 1), 9, "qhat", 5, 12, 4),
            (wide_rows, 428, "qhat", 1, 3, 3),
            (wide_rows, 428, "proposal", 1, 3, 3),
        ):
            root = cnot.pack_rows(rows, device=device)
            options = dict(width=width, shortlist=k, max_depth=depth, mode=mode, audit=True)
            old = cnot.search_cnot(root, WireFixture(undo), engine="python", **options)
            new = cnot.search_cnot(root, WireFixture(undo), engine="rust", **options)
            assert asdict(new) == asdict(old), (mode, len(rows), new, old)
            if new.gates is not None:
                assert cnot.replay_cnot(len(rows), new.gates) == rows
            if len(rows) == 144:
                assert new.gates == (18592, 428)
            cases.append(dict(wires=len(rows), mode=mode, options=options,
                              result=asdict(new), replay=new.gates is not None))

        errors = []
        if distributed:
            class RejectOnOne(WireFixture):
                def child_values(self, children):
                    if rank == 1:
                        raise RuntimeError("injected CNOT scorer failure")
                    return super().child_values(children)

            try:
                cnot.search_cnot(cnot.pack_rows((8, 4, 2, 1), device=device), RejectOnOne(9),
                                 width=5, shortlist=9, max_depth=4, engine="rust")
            except RuntimeError as exc:
                assert "rank 1: RuntimeError: injected CNOT scorer failure" in str(exc)
                errors.append(str(exc))
            else:
                raise AssertionError("scorer failure did not reach every rank")
            recovered = cnot.search_cnot(cnot.pack_rows((1, 3), device=device), object(),
                                         width=1, shortlist=1, max_depth=1, engine="rust")
            assert recovered.gates == (0,)

        output = Path(os.environ["CNOT_RUST_EVIDENCE"])
        output.mkdir(parents=True, exist_ok=True)
        (output / f"rank-{rank}.json").write_text(json.dumps(dict(
            rank=rank, world=size, gpu=torch.cuda.get_device_name(device), cases=cases,
            errors=errors, reused_after_failure=bool(errors)), indent=2) + "\n", encoding="utf-8")
        print(f"CNOT_RUST_EQUIVALENCE_OK rank={rank} world={size}", flush=True)
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
