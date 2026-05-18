from __future__ import annotations

import os
import sys
from pathlib import Path

import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from production_v6_dispatcher import run_frontier_coverage_audit_world2

REQUIRED_B_MICRO = 8192
REQUIRED_K_EXPAND_TILE = 196608
REQUIRED_BUCKET_CAP_PER_PEER = 262144

assert REQUIRED_B_MICRO == 8192
assert REQUIRED_K_EXPAND_TILE == 196608
assert REQUIRED_BUCKET_CAP_PER_PEER == 262144


def main() -> None:
    os.environ["INFERENCE_BACKEND"] = "fullbeamnice_static"
    os.environ["USE_CUDA_GRAPHS"] = "0"
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    task_count = int(os.environ.get("FRONTIER_COVERAGE_TASK_COUNT", "10"))
    max_depth = int(os.environ.get("FRONTIER_COVERAGE_MAX_DEPTH", "12"))
    beam_width = int(os.environ.get("GLOBAL_BEAM_WIDTH", "65536"))
    b_micro = int(os.environ.get("FRONTIER_COVERAGE_B_MICRO", str(REQUIRED_B_MICRO)))
    k_expand_tile = b_micro * 24
    bucket_cap_per_peer = 1 << (max(131072, k_expand_tile) - 1).bit_length()
    print(
        "FRONTIER_COVERAGE_PRE_TORCHRUN_CONFIG "
        f"B_MICRO={b_micro} K_EXPAND_TILE={k_expand_tile} BUCKET_CAP_PER_PEER={bucket_cap_per_peer}",
        flush=True,
    )
    if b_micro != REQUIRED_B_MICRO:
        raise RuntimeError(f"invalid_config: B_MICRO must be {REQUIRED_B_MICRO}, got {b_micro}")
    if k_expand_tile != REQUIRED_K_EXPAND_TILE:
        raise RuntimeError(f"invalid_config: K_EXPAND_TILE must be {REQUIRED_K_EXPAND_TILE}, got {k_expand_tile}")
    if bucket_cap_per_peer != REQUIRED_BUCKET_CAP_PER_PEER:
        raise RuntimeError(f"invalid_config: BUCKET_CAP_PER_PEER must be {REQUIRED_BUCKET_CAP_PER_PEER}, got {bucket_cap_per_peer}")
    if task_count != 10:
        raise RuntimeError(f"frontier coverage audit requires task_count=10, got {task_count}")
    if max_depth != 12:
        raise RuntimeError(f"frontier coverage audit requires max_depth=12, got {max_depth}")
    if beam_width != 65536:
        raise RuntimeError(f"frontier coverage audit requires beam_width=65536, got {beam_width}")
    result = run_frontier_coverage_audit_world2(
        task_count=task_count,
        max_depth=max_depth,
        beam_width=beam_width,
        output_path=Path("/kaggle/working/frontier_coverage_audit_world2.csv"),
        audit_path=Path("/kaggle/working/frontier_coverage_audit_world2.jsonl"),
        b_micro=b_micro,
    )
    if result["legacy_next_state_pool_path"] or result["prefilled_score_ring_fake_path"] or result["runtime_120_slice"] or result["fallback_backend"]:
        raise AssertionError(f"forbidden path flag true: {result}")
    print(
        "FRONTIER_COVERAGE_AUDIT_WORLD2_OK "
        f"rank={result['rank']} world_size={result['world_size']} task_count={result['task_count']} "
        f"row_count={result['row_count']} coverage_failure_count={result['coverage_failure_count']} "
        f"known_path_replay_valid={int(result['known_path_result']['known_path_replay_valid'])} "
        f"status_counts={result['status_counts']} no_quality_claim=1 no_leaderboard_claim=1 no_performance_claim=1"
    )
    if int(result["rank"]) == 0:
        print("=== FRONTIER_COVERAGE_AUDIT_WORLD2_TEST_COMPLETE ===")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
