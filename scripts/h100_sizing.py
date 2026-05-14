#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os


def mib(x: int) -> float:
    return x / 1024 / 1024


def gib(x: int) -> float:
    return x / 1024 / 1024 / 1024


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--global-beam-width", type=int, required=True)
    p.add_argument("--bucket-cap-per-peer", type=int, required=True)
    p.add_argument("--b-micro", type=int, default=131072)
    p.add_argument("--k-expand-tile", type=int, default=0)
    p.add_argument("--fanout", type=int, default=24)
    p.add_argument("--score-ring-depth", type=int, default=64)
    p.add_argument("--net-ring-depth", type=int, default=3)
    p.add_argument("--state-size-bytes", type=int, default=120)
    p.add_argument("--max-depth", type=int, default=1)
    p.add_argument("--history-backend", choices=["gpu", "cpu"], default=os.getenv("HISTORY_BACKEND", "gpu").lower())
    p.add_argument("--inference-backend", default=os.getenv("INFERENCE_BACKEND", "fullbeamnice_static"))
    p.add_argument("--gamma", type=float, default=1.05)
    p.add_argument("--beta", type=float, default=1.10)
    p.add_argument("--hash-load-factor", type=float, default=0.60)
    args = p.parse_args()

    n_local = math.ceil(args.global_beam_width / args.world_size)
    k_keep = int(args.gamma * n_local + 0.5)
    k_work = int(args.beta * k_keep + 0.5)
    hash_capacity = int(k_work / args.hash_load_factor + 0.5)

    history_depth = 1 if args.history_backend == "cpu" else args.max_depth
    sizes = {
        "beam_current": n_local * args.state_size_bytes,
        "next_state_pool": k_work * args.state_size_bytes,
        "next_meta": k_work * 32,
        "hash_table": hash_capacity * 32,
        "active_flags": k_work,
        "free_indices": k_work * 4,
        "free_count": 4,
        "score_ring": args.score_ring_depth * args.b_micro * args.fanout * 2,
        "send_buckets": args.net_ring_depth * args.world_size * args.bucket_cap_per_peer * 160,
        "recv_buckets": args.net_ring_depth * args.world_size * args.bucket_cap_per_peer * 160,
        "send_counts": args.net_ring_depth * args.world_size * 4,
        "recv_counts": args.net_ring_depth * args.world_size * 4,
        "history_parent_idx": history_depth * n_local * 4,
        "history_parent_rank": history_depth * n_local,
        "history_action": history_depth * n_local,
        "history_valid": history_depth * n_local,
        "histograms_threshold_counters": 65536 * 4 * 2 + 2 * 4 + 8 * 4,
    }
    if args.inference_backend == "fullbeamnice_static":
        sizes.update({
            "fullbeamnice_static_weights_fp16": 23_978_008 * 2,
            "fullbeamnice_static_act1": args.b_micro * 1536 * 2,
            "fullbeamnice_static_act2": args.b_micro * 512 * 2,
            "fullbeamnice_static_act3": args.b_micro * 512 * 2,
            "fullbeamnice_static_out": args.b_micro * 24 * 2,
        })
    total = sum(sizes.values())
    h100_bytes = 80 * 1024**3

    print("entity_id=h100_sizing; type=memory_model; state=calculated")
    print(f"params: WORLD_SIZE={args.world_size}; GLOBAL_BEAM_WIDTH={args.global_beam_width}; B_MICRO={args.b_micro}; K_EXPAND_TILE={args.k_expand_tile}; FANOUT={args.fanout}; BUCKET_CAP_PER_PEER={args.bucket_cap_per_peer}; MAX_DEPTH={args.max_depth}; HISTORY_BACKEND={args.history_backend}; INFERENCE_BACKEND={args.inference_backend}")
    print(f"derived: N_LOCAL={n_local}; K_KEEP={k_keep}; K_WORK={k_work}; HASH_CAPACITY={hash_capacity}")
    for name, size in sizes.items():
        print(f"buffer={name}; bytes={size}; MiB={mib(size):.2f}; GiB={gib(size):.3f}")
    print(f"total_static_buffers_bytes={total}; total_static_buffers_GiB={gib(total):.3f}")
    print(f"h100_80gb_headroom_GiB={gib(h100_bytes - total):.3f}; memory_ok={total < h100_bytes}")
    print("note=CUDA context, NCCL internal buffers, extension code, and allocator fragmentation are not included")


if __name__ == "__main__":
    main()
