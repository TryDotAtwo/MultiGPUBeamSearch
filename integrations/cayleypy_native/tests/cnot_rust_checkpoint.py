"""Compare Rust and reference controllers with the same native checkpoint scorer."""
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from multigpubeamsearch.cnot import pack_rows, replay_cnot, search_cnot
from multigpubeamsearch.cnot_scorer import TQPScorer


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    try:
        with TQPScorer(os.environ["MGBS_TQP_LIBRARY"], os.environ["MGBS_TQP_MANIFEST"],
                       n=4, device=device, inference_batch=2) as scorer:
            checkpoint = scorer.metadata["checkpoint_sha256"]
            assert checkpoint == "92f3ec8e6a6ef831916a9d3ba9bbdae9307c9850d61334916a2b3e690c0020c4"
            rows = (1, 7, 4, 10)
            root = pack_rows(rows, device=device)
            results = []
            for mode in ("qhat", "proposal"):
                options = dict(width=2, shortlist=3, max_depth=3, mode=mode, audit=True)
                try:
                    old = search_cnot(root, scorer, engine="python", **options)
                except RuntimeError as error:
                    # This checkpoint has a finite feature domain. Its coverage
                    # rejection is part of the result, not a reason to change it.
                    assert mode == "proposal" and "feature" in str(error)
                    try:
                        search_cnot(root, scorer, engine="rust", **options)
                    except RuntimeError as other:
                        assert str(other) == str(error)
                    else:
                        raise AssertionError("Rust controller missed scorer coverage rejection")
                    results.append(dict(options=options, error=str(error)))
                else:
                    new = search_cnot(root, scorer, engine="rust", **options)
                    assert asdict(new) == asdict(old)
                    if new.gates is not None:
                        assert replay_cnot(4, new.gates) == rows
                    if mode == "qhat":
                        assert new.status == "solved"
                    results.append(dict(options=options, result=asdict(new), replay=new.gates is not None))
            output = Path(os.environ["CNOT_RUST_EVIDENCE"])
            output.mkdir(parents=True, exist_ok=True)
            (output / f"rank-{rank}.json").write_text(json.dumps(dict(
                rank=rank, world=world, gpu=torch.cuda.get_device_name(device),
                checkpoint_sha256=checkpoint, rows=rows, results=results), indent=2) + "\n", encoding="utf-8")
            print(f"CNOT_RUST_CHECKPOINT_OK rank={rank} world={world}", flush=True)
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
