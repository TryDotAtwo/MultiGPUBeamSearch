#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os


def gib(x: int) -> float:
    return x / 1024 / 1024 / 1024


def pow2_ceil(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def auto_stream2_params(global_beam_width: int, world_size: int, b_micro: int, fanout: int) -> dict[str, int]:
    target_rounds = max(1, int(os.getenv("TARGET_STREAM2_ROUNDS", "16")))
    k_expand_tile = pow2_ceil(math.ceil(global_beam_width * fanout / (world_size * target_rounds)))
    score_ring_depth = pow2_ceil(math.ceil(k_expand_tile / (b_micro * fanout)))
    bucket_cap_fast = pow2_ceil(math.ceil(k_expand_tile / world_size))
    bucket_cap_safe = pow2_ceil(math.ceil(k_expand_tile * 0.75))
    bucket_mode = os.getenv("AUTO_BUCKET_CAP_MODE", "safe").strip().lower()
    return {
        "k_expand_tile": k_expand_tile,
        "score_ring_depth": max(1, score_ring_depth),
        "bucket_cap_per_peer": max(4096, bucket_cap_safe if bucket_mode == "safe" else bucket_cap_fast),
    }


def int_or_auto(value: str, auto_value: int) -> int:
    token = str(value).strip().lower()
    if token in {"", "auto"}:
        return int(auto_value)
    return int(token)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=int(os.getenv("WORLD_SIZE", "2")))
    p.add_argument("--global-beam-width", type=int, default=int(os.getenv("GLOBAL_BEAM_WIDTH", "4194304")))
    p.add_argument("--bucket-cap-per-peer", default=os.getenv("BUCKET_CAP_PER_PEER", "auto"))
    p.add_argument("--b-micro", type=int, default=int(os.getenv("B_MICRO", "32768")))
    p.add_argument("--k-expand-tile", default=os.getenv("K_EXPAND_TILE", "auto"))
    p.add_argument("--fanout", type=int, default=int(os.getenv("FANOUT", "24")))
    p.add_argument("--score-ring-depth", default=os.getenv("SCORE_RING_DEPTH", "auto"))
    p.add_argument("--inference-parallelism", type=int, default=int(os.getenv("INFERENCE_PARALLELISM", "1")))
    p.add_argument("--net-ring-depth", type=int, default=int(os.getenv("NET_RING_DEPTH", "2")))
    p.add_argument("--state-size-bytes", type=int, default=int(os.getenv("STATE_SIZE_BYTES", "120")))
    p.add_argument("--max-depth", type=int, default=int(os.getenv("MAX_DEPTH", "1")))
    p.add_argument("--history-backend", choices=["gpu", "cpu"], default=os.getenv("HISTORY_BACKEND", "gpu").lower())
    p.add_argument("--inference-backend", default=os.getenv("INFERENCE_BACKEND", "fullbeamnice_static"))
    p.add_argument("--gamma", type=float, default=float(os.getenv("GAMMA", "1.05")))
    p.add_argument("--beta", type=float, default=float(os.getenv("BETA", "1.15")))
    p.add_argument("--hash-load-factor", type=float, default=float(os.getenv("HASH_LOAD_FACTOR", "0.55")))
    args = p.parse_args()
    auto = auto_stream2_params(args.global_beam_width, args.world_size, args.b_micro, args.fanout)
    args.k_expand_tile = int_or_auto(args.k_expand_tile, auto["k_expand_tile"])
    args.score_ring_depth = int_or_auto(args.score_ring_depth, auto["score_ring_depth"])
    args.bucket_cap_per_peer = int_or_auto(args.bucket_cap_per_peer, auto["bucket_cap_per_peer"])

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
        "current_active_flags": n_local,
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
        "histograms_threshold_counters_status": 65536 * 4 * 2 + 2 * 4 + 8 * 4 + 8 * 4,
    }
    if args.inference_backend == "fullbeamnice_static":
        sizes.update({
            "fullbeamnice_static_weights_fp16": 23_978_008 * 2,
            "fullbeamnice_static_act1": args.inference_parallelism * args.b_micro * 1536 * 2,
            "fullbeamnice_static_act2": args.inference_parallelism * args.b_micro * 512 * 2,
            "fullbeamnice_static_act3": args.inference_parallelism * args.b_micro * 512 * 2,
            "fullbeamnice_static_out": args.inference_parallelism * args.b_micro * 24 * 2,
        })
    total = sum(sizes.values())
    t4_bytes = 15 * 1024**3

    print("entity_id=t4_sizing; type=memory_model; state=calculated")
    print(f"params: WORLD_SIZE={args.world_size}; GLOBAL_BEAM_WIDTH={args.global_beam_width}; B_MICRO={args.b_micro}; K_EXPAND_TILE={args.k_expand_tile}; FANOUT={args.fanout}; BUCKET_CAP_PER_PEER={args.bucket_cap_per_peer}; INFERENCE_PARALLELISM={args.inference_parallelism}; MAX_DEPTH={args.max_depth}; HISTORY_BACKEND={args.history_backend}; INFERENCE_BACKEND={args.inference_backend}")
    print(f"derived: N_LOCAL={n_local}; K_KEEP={k_keep}; K_WORK={k_work}; HASH_CAPACITY={hash_capacity}")
    for name, size in sizes.items():
        print(f"buffer={name}; bytes={size}; GiB={gib(size):.3f}")
    print(f"total_static_buffers_bytes={total}; total_static_buffers_GiB={gib(total):.3f}")
    print(f"t4_15gb_headroom_GiB={gib(t4_bytes - total):.3f}; memory_ok={total < t4_bytes}")
    print("note=Kaggle runtime overhead, CUDA context, NCCL internals, extension code, and allocator fragmentation are not included")


if __name__ == "__main__":
    main()
