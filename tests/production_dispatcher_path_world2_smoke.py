from __future__ import annotations

import os
import sys
from pathlib import Path

import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from production_v6_dispatcher import run_real_data_production_v6_world2


def main() -> None:
    os.environ["INFERENCE_BACKEND"] = "fullbeamnice_static"
    os.environ["USE_CUDA_GRAPHS"] = "0"
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    task_count = int(os.environ.get("PRODUCTION_V6_TASK_COUNT", "2"))
    max_depth = int(os.environ.get("PRODUCTION_V6_MAX_DEPTH", "12"))
    beam_width = int(os.environ.get("GLOBAL_BEAM_WIDTH", "4096"))
    if task_count < 1 or task_count > 3:
        raise RuntimeError(f"first validation task_count must be 1..3, got {task_count}")
    if max_depth < 10 or max_depth > 20:
        raise RuntimeError(f"first validation max_depth must be 10..20, got {max_depth}")
    if beam_width < 4096 or beam_width > 65536:
        raise RuntimeError(f"first validation beam_width must be 4096..65536, got {beam_width}")
    result = run_real_data_production_v6_world2(
        task_count=task_count,
        max_depth=max_depth,
        beam_width=beam_width,
        output_path=Path(os.environ.get("PRODUCTION_V6_OUTPUT_PATH", "/kaggle/working/production_v6_dispatcher_path_world2.csv")),
        stats_path=Path(os.environ.get("PRODUCTION_V6_STATS_PATH", "/kaggle/working/production_v6_dispatcher_path_world2_stats.jsonl")),
        b_micro=int(os.environ.get("PRODUCTION_V6_B_MICRO", "4")),
    )
    allowed_statuses = {"solved", "unsolved", "max_depth_reached"}
    if set(result["status_counts"]) != allowed_statuses:
        raise AssertionError(f"unexpected status keys: {result['status_counts']}")
    if int(result["task_count"]) != task_count:
        raise AssertionError(f"task_count mismatch: {result}")
    if not result["production_v6_dispatcher_path"]:
        raise AssertionError("production_v6_dispatcher_path flag false")
    if result["legacy_next_state_pool_path"] or result["prefilled_score_ring_fake_path"] or result["runtime_120_slice"] or result["fallback_backend"]:
        raise AssertionError(f"forbidden path flag true: {result}")
    print(
        "PRODUCTION_V6_DISPATCHER_PATH_WORLD2_SMOKE_OK "
        f"rank={result['rank']} world_size={result['world_size']} tasks={result['task_count']} "
        f"beam={beam_width} max_depth={max_depth} statuses={result['status_counts']} "
        "legacy_next_state_pool_path=0 prefilled_score_ring_fake_path=0 runtime_120_slice=0 fallback_backend=0"
    )
    if int(result["rank"]) == 0:
        print("=== PRODUCTION_V6_DISPATCHER_PATH_WORLD2_TEST_COMPLETE ===")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
