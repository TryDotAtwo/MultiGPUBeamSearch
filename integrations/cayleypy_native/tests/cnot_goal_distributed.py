"""Exercise coordinated goal failures and paired searches on real GPUs."""
from dataclasses import asdict
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from verify_cnot import WireFixture
from multigpubeamsearch import cnot


baseline_path = Path(os.environ["CNOT_GOAL_BASELINE"])
spec = importlib.util.spec_from_file_location("multigpubeamsearch._baseline_cnot", baseline_path)
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)

torch.set_num_threads(1)
device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
torch.cuda.set_device(device)
distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
if distributed:
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
rank = dist.get_rank() if distributed else 0
world_size = dist.get_world_size() if distributed else 1
current_goal = cnot._goal
faults = []
try:
    root_rows = (8, 4, 2, 1)
    root = cnot.pack_rows(root_rows, device=device)
    # This fixture patches the Python reference implementation itself.
    options = dict(width=5, shortlist=9, max_depth=4, mode="qhat", audit=False, engine="python")

    if distributed:
        for fault_rank in (0, 1):
            fired = [False]

            def inject_at_goal(world, parents, lex, begin, identity):
                if rank != fault_rank or not len(parents) or fired[0]:
                    return current_goal(world, parents, lex, begin, identity)
                arange = torch.arange

                def fail_once(*args, **kwargs):
                    if not fired[0]:
                        fired[0] = True
                        raise ValueError("injected goal failure")
                    return arange(*args, **kwargs)

                torch.arange = fail_once
                try:
                    return current_goal(world, parents, lex, begin, identity)
                finally:
                    torch.arange = arange

            cnot._goal = inject_at_goal
            try:
                cnot.search_cnot(root, WireFixture(9), **options)
            except RuntimeError as error:
                message = str(error)
                assert f"find CNOT goal failed: rank {fault_rank}: ValueError: injected goal failure" in message
            else:
                raise AssertionError("goal failure was not coordinated")
            finally:
                cnot._goal = current_goal
            if rank == fault_rank:
                assert fired[0]
            recovered = cnot.search_cnot(cnot.pack_rows((1, 3), device=device), object(),
                                         width=1, shortlist=1, max_depth=1)
            assert recovered.status == "solved" and recovered.gates == (0,)
            faults.append(dict(fault_rank=fault_rank, error=message, reused=True,
                               recovered=asdict(recovered)))

    def execute(goal):
        cnot._goal = goal
        torch.cuda.synchronize(device)
        if distributed:
            dist.barrier()
        start = time.perf_counter()
        result = cnot.search_cnot(root, WireFixture(9), **options)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if distributed:
            durations = [None] * world_size
            dist.all_gather_object(durations, elapsed)
        else:
            durations = [elapsed]
        return asdict(result), max(durations)

    for _ in range(2):
        old, _ = execute(baseline._goal)
        new, _ = execute(current_goal)
        assert old == new
    pairs = []
    for i in range(5):
        order = (("old", baseline._goal), ("new", current_goal))
        if i % 2:
            order = order[::-1]
        results = {name: execute(goal) for name, goal in order}
        assert results["old"][0] == results["new"][0] == old
        pairs.append(dict(old_seconds=results["old"][1], new_seconds=results["new"][1]))
    receipt = dict(rank=rank, world=world_size, device=str(device), gpu=torch.cuda.get_device_name(device),
                   rows=root_rows, options=options, result=old, faults=faults, pairs=pairs,
                   old_median=statistics.median(p["old_seconds"] for p in pairs),
                   new_median=statistics.median(p["new_seconds"] for p in pairs))
    out = Path(os.environ["CNOT_GOAL_EVIDENCE"])
    out.mkdir(parents=True, exist_ok=True)
    (out / f"rank-{rank}.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"CNOT_GOAL_COLLECTIVE_OK rank={rank} world={world_size}", flush=True)
finally:
    cnot._goal = current_goal
    if distributed:
        dist.destroy_process_group()
