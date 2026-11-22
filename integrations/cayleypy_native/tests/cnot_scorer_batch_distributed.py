"""Compare checkpoint-backed whole searches using both scorer ABI paths.

Run under torchrun with MGBS_TQP_LIBRARY, MGBS_TQP_MANIFEST and
MGBS_CNOT_BATCH_EVIDENCE set. This is a bounded GL(4,2) fixture, not a
saturated inference benchmark.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist

from multigpubeamsearch.cnot import pack_rows, replay_cnot, search_cnot
from multigpubeamsearch.cnot_scorer import TQPScorer


device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
torch.cuda.set_device(device)
distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
if distributed:
    dist.init_process_group("nccl")
rank = dist.get_rank() if distributed else 0
world_size = dist.get_world_size() if distributed else 1

try:
    options = dict(n=4, device=device, inference_batch=2)
    with TQPScorer(os.environ["MGBS_TQP_LIBRARY"], os.environ["MGBS_TQP_MANIFEST"], **options) as old, \
         TQPScorer(os.environ["MGBS_TQP_LIBRARY"], os.environ["MGBS_TQP_MANIFEST"],
                   experimental_native_batch=True, **options) as new:
        assert old.metadata["checkpoint_sha256"] == new.metadata["checkpoint_sha256"]
        assert old.metadata["checkpoint_sha256"] == "92f3ec8e6a6ef831916a9d3ba9bbdae9307c9850d61334916a2b3e690c0020c4"
        rows = (1, 7, 4, 10)
        root = pack_rows(rows, device=device)
        kwargs = dict(width=2, shortlist=3, max_depth=3, mode="qhat", audit=False)

        def execute(scorer):
            torch.cuda.synchronize(device)
            if distributed:
                dist.barrier()
            start = time.perf_counter()
            result = search_cnot(root, scorer, **kwargs)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            if distributed:
                durations = [None] * world_size
                dist.all_gather_object(durations, elapsed)
            else:
                durations = [elapsed]
            if result.gates is not None:
                assert replay_cnot(4, result.gates) == rows
            return asdict(result), max(durations)

        for _ in range(2):
            baseline, _ = execute(old)
            candidate, _ = execute(new)
            assert baseline == candidate and baseline["status"] == "solved"
        pairs = []
        for i in range(5):
            order = (("old", old), ("new", new)) if i % 2 == 0 else (("new", new), ("old", old))
            values = {name: execute(scorer) for name, scorer in order}
            assert values["old"][0] == values["new"][0]
            pairs.append(dict(old_seconds=values["old"][1], new_seconds=values["new"][1]))
        receipt = dict(rank=rank, world=world_size, device=str(device), gpu=torch.cuda.get_device_name(device),
                       checkpoint_sha256=old.metadata["checkpoint_sha256"], rows=rows, options=kwargs,
                       result=baseline, pairs=pairs,
                       old_median=statistics.median(p["old_seconds"] for p in pairs),
                       new_median=statistics.median(p["new_seconds"] for p in pairs))
        out = Path(os.environ["MGBS_CNOT_BATCH_EVIDENCE"])
        out.mkdir(parents=True, exist_ok=True)
        (out / f"rank-{rank}.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"CNOT_CHECKPOINT_BATCH_OK rank={rank} world={world_size}", flush=True)
finally:
    if distributed:
        dist.destroy_process_group()
