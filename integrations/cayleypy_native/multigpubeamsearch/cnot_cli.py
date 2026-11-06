"""Run the CNOT adapter directly, under torchrun, or as one srun task per GPU."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .cluster import scheduler_rank
from .cnot import pack_rows, search_cnot
from .cnot_collectives import World
from .cnot_scorer import TQPScorer


def initialize_world(backend):
    distributed = "WORLD_SIZE" in os.environ or "SLURM_PROCID" in os.environ
    placement = scheduler_rank() if distributed else None
    device = torch.device("cpu")
    if backend == "nccl":
        ordinal = placement.cuda_ordinal(torch.cuda.device_count()) if placement else 0
        torch.cuda.set_device(ordinal)
        device = torch.device("cuda", ordinal)
    if placement and placement.world_size > 1:
        os.environ["WORLD_SIZE"] = str(placement.world_size)
        os.environ["RANK"] = str(placement.global_rank)
        dist.init_process_group(backend)
    return device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path, help="JSON array of n row integers or 0x-prefixed strings")
    parser.add_argument("--library", required=True, help="compiled TQP scorer bridge")
    parser.add_argument("--manifest", required=True, help="frozen consumer search manifest")
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--shortlist", required=True, type=int)
    parser.add_argument("--max-depth", required=True, type=int)
    parser.add_argument("--mode", choices=("qhat", "proposal"), default="qhat")
    parser.add_argument("--inference-batch", type=int, default=4)
    parser.add_argument("--audit", action="store_true", help="include all candidates in the JSON for small runs")
    args = parser.parse_args()
    device = initialize_world("nccl")
    try:
        group = World(device)
        def load_input():
            data = json.loads(args.matrix.read_text(encoding="utf-8"))
            return pack_rows([int(row, 0) if isinstance(row, str) else row for row in data], device=device)
        root = group.checked("read input matrix", load_input)
        def load_scorer():
            return TQPScorer(args.library, args.manifest, n=root.size(1),
                             device=device, inference_batch=args.inference_batch)
        with group.checked("load 102-feature scorer", load_scorer) as scorer:
            result = search_cnot(root, scorer, width=args.width, shortlist=args.shortlist,
                                 max_depth=args.max_depth, mode=args.mode, audit=args.audit)
            if group.rank == 0:
                print(json.dumps(dict(**asdict(result), length=result.length, model=scorer.metadata)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
