from __future__ import annotations

import os
import sys
from pathlib import Path

import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from production_v6_dispatcher import run_real_data_path_audit_world2


def main() -> None:
    os.environ["INFERENCE_BACKEND"] = "fullbeamnice_static"
    os.environ["USE_CUDA_GRAPHS"] = "0"
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    task_count = int(os.environ.get("REAL_DATA_PATH_AUDIT_TASK_COUNT", "100"))
    max_depth = int(os.environ.get("REAL_DATA_PATH_AUDIT_MAX_DEPTH", "300"))
    beam_width = int(os.environ.get("GLOBAL_BEAM_WIDTH", "65536"))
    if task_count != 100:
        raise RuntimeError(f"path audit runner requires task_count=100, got {task_count}")
    if max_depth != 300:
        raise RuntimeError(f"path audit runner requires max_depth=300, got {max_depth}")
    if beam_width != 65536:
        raise RuntimeError(f"path audit runner requires beam_width=65536, got {beam_width}")
    result = run_real_data_path_audit_world2(
        task_count=task_count,
        max_depth=max_depth,
        beam_width=beam_width,
        output_path=Path("/kaggle/working/real_data_100samples_depth300_beam65536_path_audit_world2.csv"),
        audit_path=Path("/kaggle/working/real_data_100samples_depth300_beam65536_path_audit_world2.jsonl"),
        b_micro=int(os.environ.get("REAL_DATA_PATH_AUDIT_B_MICRO", "8192")),
    )
    if result["legacy_next_state_pool_path"] or result["prefilled_score_ring_fake_path"] or result["runtime_120_slice"] or result["fallback_backend"]:
        raise AssertionError(f"forbidden path flag true: {result}")
    statuses = result["status_counts"]
    total = int(statuses["solved"] + statuses["unsolved"] + statuses["max_depth_reached"] + statuses["error"])
    if total != task_count:
        raise AssertionError(f"status accounting mismatch: {statuses}")
    print(
        "REAL_DATA_100SAMPLES_DEPTH300_BEAM65536_PATH_AUDIT_WORLD2_OK "
        f"rank={result['rank']} world_size={result['world_size']} total_tasks={task_count} "
        f"solved_count={statuses['solved']} unsolved_count={statuses['unsolved']} "
        f"max_depth_reached_count={statuses['max_depth_reached']} error_count={statuses['error']} "
        f"failure_counts={result['failure_counts']} "
        "no_quality_claim=1 no_leaderboard_claim=1 no_performance_claim=1"
    )
    if int(result["rank"]) == 0:
        print("=== REAL_DATA_100SAMPLES_DEPTH300_BEAM65536_PATH_AUDIT_WORLD2_TEST_COMPLETE ===")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
