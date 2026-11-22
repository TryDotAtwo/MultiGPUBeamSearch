"""Run under two-rank torchrun; validate actual NCCL through a supervisor."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", action="store_true")
    parser.add_argument("--engine", choices=("python", "rust"))
    parser.add_argument("--shared-dir")
    args = parser.parse_args()
    if args.payload:
        from datetime import timedelta
        import torch
        import torch.distributed as dist
        rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", timeout=timedelta(seconds=60))
        try:
            value = torch.tensor([rank + 1], device="cuda", dtype=torch.int64)
            dist.all_reduce(value)
            torch.cuda.synchronize()
            assert int(value.item()) == 3
            Path("nccl-result.json").write_text(json.dumps({
                "rank": rank, "local_rank": local, "world": dist.get_world_size(),
                "sum": int(value.item()), "device": torch.cuda.get_device_name(local),
                "id_file": os.environ["BEAM_NCCL_ID_FILE"],
            }))
        finally:
            dist.destroy_process_group()
        return
    from multigpubeamsearch.cluster import run_cluster_rank
    directory = run_cluster_rank([sys.executable, str(Path(__file__).resolve()), "--payload"],
                                 shared_dir=args.shared_dir, timeout=90, engine=args.engine)
    print(json.dumps({"rank": int(os.environ["RANK"]), "directory": str(directory)}), flush=True)


if __name__ == "__main__":
    main()
