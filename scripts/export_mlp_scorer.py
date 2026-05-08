"""Export N TorchScript copies of the default 120 -> 1024 -> 256 -> 24 MLP scorer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scorers import make_default_mlp_scorer


def parse_hidden(s: str) -> list[int]:
    return [int(x) for x in s.replace(",", " ").split() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runtime/scorers")
    ap.add_argument("--copies", type=int, default=int(os.environ.get("INFERENCE_PARALLELISM", "2")))
    ap.add_argument("--hidden", default=os.environ.get("MLP_HIDDEN", "1024,256"))
    ap.add_argument("--fanout", type=int, default=24)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out_dir = PROJECT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden = parse_hidden(args.hidden)
    example = torch.zeros((16, 120), dtype=torch.uint8)
    paths: list[str] = []

    for i in range(args.copies):
        torch.manual_seed(args.seed + i)
        model = make_default_mlp_scorer(hidden_sizes=hidden, fanout=args.fanout).eval()
        traced = torch.jit.trace(model, example, strict=False)
        traced = torch.jit.freeze(traced)
        path = out_dir / f"mlp_120_{'_'.join(map(str, hidden))}_{args.fanout}_copy{i:02d}.ts"
        traced.save(str(path))
        paths.append(str(path))

    print("TORCHSCRIPT_SCORER_PATHS=" + os.pathsep.join(paths))
    print("INFERENCE_BACKEND=torchscript_ensemble")
    print(f"INFERENCE_PARALLELISM={args.copies}")


if __name__ == "__main__":
    main()
