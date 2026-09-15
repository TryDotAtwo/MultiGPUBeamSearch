#!/usr/bin/env python3
"""Benchmark-only adapter: Python ATen token plan matching the C++ harness.

No production backend changes. Times GPU-resident uint8 states -> FP16 logits;
score quantization, input upload, loading and graph capture are excluded.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from stream1_transformer_torch_benchmark import PieceTransformerTorch, make_states, score_keys


class StaticTokenModel(PieceTransformerTorch):
    def __init__(self, *args):
        super().__init__(*args)
        self.slots = [
            (self.piece_positions[:, s].contiguous(),
             self.fast_slot_projected[s].contiguous(),
             self.piece_mask[:, s].to(self.dtype).view(1, self.num_pieces, 1).contiguous())
            for s in range(self.max_piece_size)
        ]

    def build_tokens(self, state_u8):
        states = state_u8.narrow(1, 0, self.state_len).long()
        batch = states.shape[0]
        pieces = self.fast_piece_static.unsqueeze(0).expand(batch, -1, -1).clone()
        for positions, table, mask in self.slots:
            values = states.index_select(1, positions).reshape(-1)
            gathered = table.index_select(0, values).view(batch, self.num_pieces, self.d_model)
            pieces = pieces + gathered * mask
        cls = self.cls_token.view(1, 1, self.d_model).expand(batch, 1, -1)
        return torch.cat((cls, pieces), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-dir", type=Path, default=Path("/workspace"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batches", default="128,256,384,512,768,1024,2048,4096")
    args = ap.parse_args()
    device = torch.device("cuda:0")
    old = PieceTransformerTorch(args.weight_dir, device)
    model = StaticTokenModel(args.weight_dir, device)
    torch.manual_seed(20260915)
    with torch.inference_mode():
        varied = torch.randint(0, model.num_classes, (97, model.state_len), device=device, dtype=torch.uint8)
        torch.testing.assert_close(old.build_tokens(varied), model.build_tokens(varied), rtol=0, atol=0)
        torch.testing.assert_close(old.forward(varied), model.forward(varied), rtol=0, atol=0)
        print(json.dumps(dict(validation="static_plan_bitexact_97_varied_states", torch=torch.__version__, gpu=torch.cuda.get_device_name())), flush=True)
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mode", "batch", "pass", "iters", "elapsed_ms", "parents_per_sec", "first_keys"])
            writer.writeheader()
            for batch in map(int, args.batches.split(",")):
                states = make_states(batch, model.state_len, model.num_classes, device)
                expected = model.forward(states)
                for mode in ("legacy_eager", "static_eager", "static_graph"):
                    fn = old.forward if mode == "legacy_eager" else model.forward
                    for _ in range(10):
                        result = fn(states)
                    torch.cuda.synchronize()
                    graph = None
                    if mode == "static_graph":
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            result = fn(states)
                        graph.replay()
                        torch.cuda.synchronize()
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    for rep in range(1, 4):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        for _ in range(100):
                            if graph is None:
                                result = fn(states)
                            else:
                                graph.replay()
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        row = dict(mode=mode, batch=batch, **{"pass": rep}, iters=100,
                                   elapsed_ms=elapsed*1000, parents_per_sec=batch*100/elapsed,
                                   first_keys=score_keys(result)[0].tolist())
                        writer.writerow(row)
                        f.flush()
                        print(json.dumps(row), flush=True)
                    del graph


if __name__ == "__main__":
    main()
