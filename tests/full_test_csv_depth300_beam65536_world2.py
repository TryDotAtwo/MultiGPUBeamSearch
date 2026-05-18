from __future__ import annotations

import os
import sys
from pathlib import Path

import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import data_loader
from production_v6_dispatcher import run_real_data_production_v6_world2_detailed, validate_output_paths


def main() -> None:
    os.environ["INFERENCE_BACKEND"] = "fullbeamnice_static"
    os.environ["USE_CUDA_GRAPHS"] = "0"
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    task_count = len(data_loader.load_test_puzzles(max_puzzles=None))
    max_depth = int(os.environ.get("FULL_TEST_MAX_DEPTH", "300"))
    beam_width = int(os.environ.get("GLOBAL_BEAM_WIDTH", "65536"))
    if max_depth != 300:
        raise RuntimeError(f"full test runner requires max_depth=300, got {max_depth}")
    if beam_width != 65536:
        raise RuntimeError(f"full test runner requires beam_width=65536, got {beam_width}")
    result = run_real_data_production_v6_world2_detailed(
        task_count=task_count,
        max_depth=max_depth,
        beam_width=beam_width,
        output_path=Path("/kaggle/working/full_test_csv_depth300_beam65536_world2.csv"),
        stats_path=Path("/kaggle/working/full_test_csv_depth300_beam65536_world2_stats.jsonl"),
        b_micro=int(os.environ.get("FULL_TEST_B_MICRO", "4")),
    )
    if result["legacy_next_state_pool_path"] or result["prefilled_score_ring_fake_path"] or result["runtime_120_slice"] or result["fallback_backend"]:
        raise AssertionError(f"forbidden path flag true: {result}")
    statuses = result["status_counts"]
    total = int(statuses["solved"] + statuses["unsolved"] + statuses["max_depth_reached"] + statuses["error"])
    if total != task_count:
        raise AssertionError(f"status accounting mismatch: {statuses}, task_count={task_count}")
    replay = {"solved_rows": 0, "validated_rows": 0, "errors": [], "path_replay_valid": True}
    if int(result["rank"]) == 0:
        replay = validate_output_paths(output_path=Path(result["output_path"]), task_count=task_count)
        if not replay["path_replay_valid"]:
            raise AssertionError(f"path replay validation failed: {replay}")
    print(
        "FULL_TEST_CSV_DEPTH300_BEAM65536_WORLD2_OK "
        f"rank={result['rank']} world_size={result['world_size']} total_tasks={task_count} "
        f"solved_count={statuses['solved']} unsolved_count={statuses['unsolved']} "
        f"max_depth_reached_count={statuses['max_depth_reached']} error_count={statuses['error']} "
        f"path_replay_valid={int(bool(replay['path_replay_valid']))} "
        f"solved_rows={replay['solved_rows']} validated_rows={replay['validated_rows']} "
        "no_quality_claim=1 no_leaderboard_claim=1 no_performance_claim=1"
    )
    if int(result["rank"]) == 0:
        print("=== FULL_TEST_CSV_DEPTH300_BEAM65536_WORLD2_TEST_COMPLETE ===")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
