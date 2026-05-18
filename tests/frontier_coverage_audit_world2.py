from __future__ import annotations

import os
import sys
from pathlib import Path

import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from production_v6_dispatcher import run_frontier_coverage_audit_world2


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
        b_micro=int(os.environ.get("FRONTIER_COVERAGE_B_MICRO", "4")),
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
