#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os


def gib(x: int) -> float:
    return x / 1024 / 1024 / 1024


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=int(os.getenv("WORLD_SIZE", "2")))
    p.add_argument("--global-beam-width", type=int, default=int(os.getenv("GLOBAL_BEAM_WIDTH", "4194304")))
    p.add_argument("--bucket-cap-per-peer", type=int, default=int(os.getenv("BUCKET_CAP_PER_PEER", "524288")))
    p.add_argument("--b-micro", type=int, default=int(os.getenv("B_MICRO", "32768")))
    p.add_argument("--k-expand-tile", type=int, default=int(os.getenv("K_EXPAND_TILE", "0")))
    p.add_argument("--fanout", type=int, default=int(os.getenv("FANOUT", "24")))
    p.add_argument("--score-ring-depth", type=int, default=int(os.getenv("SCORE_RING_DEPTH", "16")))
    p.add_argument("--inference-parallelism", type=int, default=int(os.getenv("INFERENCE_PARALLELISM", "1")))
    p.add_argument("--net-ring-depth", type=int, default=int(os.getenv("NET_RING_DEPTH", "2")))
    p.add_argument("--state-size-bytes", type=int, default=int(os.getenv("STATE_SIZE_BYTES", "120")))
    p.add_argument("--max-depth", type=int, default=int(os.getenv("MAX_DEPTH", "1")))
    p.add_argument("--gamma", type=float, default=float(os.getenv("GAMMA", "1.05")))
    p.add_argument("--beta", type=float, default=float(os.getenv("BETA", "1.15")))
    p.add_argument("--hash-load-factor", type=float, default=float(os.getenv("HASH_LOAD_FACTOR", "0.55")))
    args = p.parse_args()

    n_local = math.ceil(args.global_beam_width / args.world_size)
    k_keep = int(args.gamma * n_local + 0.5)
    k_work = int(args.beta * k_keep + 0.5)
    hash_capacity = int(k_work / args.hash_load_factor + 0.5)

    sizes = {
        "beam_current": n_local * args.state_size_bytes,
        "next_state_pool": k_work * args.state_size_bytes,
        "next_meta": k_work * 32,
        "hash_table": hash_capacity * 32,
        "current_active_flags": n_local,
        "active_flags": k_work,
        "free_indices": k_work * 4,
        "free_count": 4,
        "score_ring": args.score_ring_depth * args.b_micro * args.fanout * 2,
        "send_buckets": args.net_ring_depth * args.world_size * args.bucket_cap_per_peer * 160,
        "recv_buckets": args.net_ring_depth * args.world_size * args.bucket_cap_per_peer * 160,
        "send_counts": args.net_ring_depth * args.world_size * 4,
        "recv_counts": args.net_ring_depth * args.world_size * 4,
        "history_parent_idx": args.max_depth * n_local * 4,
        "history_parent_rank": args.max_depth * n_local,
        "history_action": args.max_depth * n_local,
        "history_valid": args.max_depth * n_local,
        "histograms_threshold_counters_status": 65536 * 4 * 2 + 2 * 4 + 8 * 4 + 8 * 4,
    }
    total = sum(sizes.values())
    t4_bytes = 15 * 1024**3

    print("entity_id=t4_sizing; type=memory_model; state=calculated")
    print(f"params: WORLD_SIZE={args.world_size}; GLOBAL_BEAM_WIDTH={args.global_beam_width}; B_MICRO={args.b_micro}; K_EXPAND_TILE={args.k_expand_tile}; FANOUT={args.fanout}; BUCKET_CAP_PER_PEER={args.bucket_cap_per_peer}; INFERENCE_PARALLELISM={args.inference_parallelism}; MAX_DEPTH={args.max_depth}")
    print(f"derived: N_LOCAL={n_local}; K_KEEP={k_keep}; K_WORK={k_work}; HASH_CAPACITY={hash_capacity}")
    for name, size in sizes.items():
        print(f"buffer={name}; bytes={size}; GiB={gib(size):.3f}")
    print(f"total_static_buffers_bytes={total}; total_static_buffers_GiB={gib(total):.3f}")
    print(f"t4_15gb_headroom_GiB={gib(t4_bytes - total):.3f}; memory_ok={total < t4_bytes}")
    print("note=Kaggle runtime overhead, CUDA context, NCCL internals, TorchScript model weights, inference output tensors, and PyTorch allocator fragmentation are not included")


if __name__ == "__main__":
    main()
