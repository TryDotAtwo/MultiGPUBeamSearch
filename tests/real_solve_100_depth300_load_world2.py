from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import data_loader
from production_v6_dispatcher import (
    MOVE_COUNT,
    PRODUCTION_B_MICRO,
    PRODUCTION_K_EXPAND_TILE,
    ProductionV6Dispatcher,
    require_production_microbatch,
    require_world2_t4_runtime,
)

TASK_COUNT = 100
MAX_DEPTH = 300
BEAM_WIDTH = 65536
BUCKET_CAP_PER_PEER = 262144
HEARTBEAT_TASK_INTERVAL = 10
HEARTBEAT_SECONDS = 120.0

assert PRODUCTION_B_MICRO == 8192
assert PRODUCTION_K_EXPAND_TILE == 196608
assert BUCKET_CAP_PER_PEER == 262144


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _status_for_output(status: str) -> str:
    if status == "unsolved":
        return "unsolved_empty_frontier"
    return status


def _write_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


def _short_note(note: str, limit: int = 500) -> str:
    note = " ".join(str(note).split())
    return note[:limit]


def main() -> None:
    os.environ["INFERENCE_BACKEND"] = "fullbeamnice_static"
    os.environ["USE_CUDA_GRAPHS"] = "1"
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("COLLECTIVE_SEQ_DEBUG", "0")

    task_count = _env_int("TASK_COUNT", TASK_COUNT)
    max_depth = _env_int("MAX_DEPTH", MAX_DEPTH)
    beam_width = _env_int("GLOBAL_BEAM_WIDTH", BEAM_WIDTH)
    b_micro = require_production_microbatch(_env_int("B_MICRO", PRODUCTION_B_MICRO))
    k_expand_tile = _env_int("K_EXPAND_TILE", b_micro * MOVE_COUNT)
    bucket_cap_per_peer = _env_int("BUCKET_CAP_PER_PEER", BUCKET_CAP_PER_PEER)
    cuda_graphs_enabled = os.environ.get("USE_CUDA_GRAPHS", "0") == "1"

    if task_count != TASK_COUNT:
        raise RuntimeError(f"invalid_config: TASK_COUNT must be {TASK_COUNT}, got {task_count}")
    if max_depth != MAX_DEPTH:
        raise RuntimeError(f"invalid_config: MAX_DEPTH must be {MAX_DEPTH}, got {max_depth}")
    if beam_width != BEAM_WIDTH:
        raise RuntimeError(f"invalid_config: GLOBAL_BEAM_WIDTH must be {BEAM_WIDTH}, got {beam_width}")
    if k_expand_tile != PRODUCTION_K_EXPAND_TILE:
        raise RuntimeError(f"invalid_config: K_EXPAND_TILE must be {PRODUCTION_K_EXPAND_TILE}, got {k_expand_tile}")
    if bucket_cap_per_peer != BUCKET_CAP_PER_PEER:
        raise RuntimeError(f"invalid_config: BUCKET_CAP_PER_PEER must be {BUCKET_CAP_PER_PEER}, got {bucket_cap_per_peer}")
    if not cuda_graphs_enabled:
        raise RuntimeError("invalid_config: USE_CUDA_GRAPHS=1 required")

    rank, world_size, device = require_world2_t4_runtime()
    if rank == 0:
        print(
            "RUN_START "
            f"task_count={task_count} max_depth={max_depth} beam_width={beam_width} "
            f"B_MICRO={b_micro} K_EXPAND_TILE={k_expand_tile} "
            f"BUCKET_CAP_PER_PEER={bucket_cap_per_peer} cuda_graphs=1",
            flush=True,
        )
        print("CUDA_GRAPHS_ENABLED true", flush=True)

    dispatcher = ProductionV6Dispatcher(rank, world_size, device, beam_width=beam_width, b_micro=b_micro)
    puzzles = data_loader.load_test_puzzles(max_puzzles=task_count)
    if len(puzzles) != task_count:
        raise RuntimeError(f"expected {task_count} puzzles, got {len(puzzles)}")

    output_path = Path(os.environ.get("OUTPUT_PATH", "/kaggle/working/real_solve_100_depth300_load_world2.csv"))
    stats_path = Path(os.environ.get("STATS_PATH", "/kaggle/working/real_solve_100_depth300_load_world2_stats.jsonl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_id", "initial_state_id", "status", "found", "depth_used", "path_len", "path", "elapsed_sec", "error_or_note"]
    if rank == 0:
        with output_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        stats_path.write_text("", encoding="utf-8")
    dist.barrier()

    status_counts = {"solved": 0, "max_depth_reached": 0, "unsolved_empty_frontier": 0, "unsolved_pruned": 0, "error": 0}
    completed = 0
    aborted = False
    run_start = time.perf_counter()
    last_heartbeat = run_start

    for task_idx, (initial_state_id, state) in enumerate(puzzles):
        task_start = time.perf_counter()
        path = ""
        note = ""
        try:
            result = dispatcher.run_task(int(initial_state_id), state, max_depth, beam_width)
            status = _status_for_output(result.status)
            depth_used = int(result.solved_depth + 1) if result.status == "solved" else len(result.depth_rows)
            path = result.path
            path_len = len(path.split(".")) if path else 0
            found = result.status == "solved"
            if status not in status_counts:
                status = "unsolved_pruned"
            status_counts[status] += 1
        except Exception as exc:  # noqa: BLE001 - load validation must keep task-indexed failure evidence.
            status = "error"
            found = False
            depth_used = 0
            path_len = 0
            note = f"{type(exc).__name__}: {exc}"
            status_counts["error"] += 1

        elapsed = time.perf_counter() - task_start
        row = {
            "row_id": task_idx,
            "initial_state_id": int(initial_state_id),
            "status": status,
            "found": int(found),
            "depth_used": depth_used,
            "path_len": path_len,
            "path": path,
            "elapsed_sec": f"{elapsed:.3f}",
            "error_or_note": note,
        }
        if rank == 0:
            _write_row(output_path, row, fieldnames)
            with stats_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            if status == "error":
                print(
                    "TASK_ERROR "
                    f"task_idx={task_idx} depth_used={depth_used} elapsed_sec={elapsed:.3f} "
                    f"note={_short_note(note)}",
                    flush=True,
                )
            if found:
                print(
                    "TASK_SOLVED "
                    f"task_idx={task_idx} depth={max(depth_used - 1, 0)} path_len={path_len} "
                    f"elapsed_sec={elapsed:.3f} solved_count={status_counts['solved']}",
                    flush=True,
                )
            print(
                "TASK_DONE "
                f"task_idx={task_idx} status={status} depth_used={depth_used} elapsed_sec={elapsed:.3f}",
                flush=True,
            )
            if status == "error" and ("CUDA error" in note or "AcceleratorError" in note):
                raise RuntimeError(f"task_error_cuda_fault task_idx={task_idx} note={_short_note(note)}")

        completed += 1
        error_flag = torch.tensor([1 if status == "error" else 0], dtype=torch.int32, device=device)
        dist.all_reduce(error_flag, op=dist.ReduceOp.SUM)
        if int(error_flag.cpu()[0]) > 0:
            aborted = True
            if rank == 0:
                print(
                    "RUN_ABORT "
                    f"reason=task_error task_idx={task_idx} error_ranks={int(error_flag.cpu()[0])}",
                    flush=True,
                )
            break
        now = time.perf_counter()
        if rank == 0 and (
            completed % HEARTBEAT_TASK_INTERVAL == 0
            or now - last_heartbeat >= HEARTBEAT_SECONDS
            or completed == task_count
        ):
            gpu_mem = int(torch.cuda.memory_reserved(device)) if torch.cuda.is_available() else 0
            print(
                "HEARTBEAT "
                f"elapsed_sec={now - run_start:.3f} completed={completed}/{task_count} "
                f"solved={status_counts['solved']} errors={status_counts['error']} gpu_mem={gpu_mem}",
                flush=True,
            )
            last_heartbeat = now
        dist.barrier()

    local_counts = torch.tensor(
        [
            completed,
            status_counts["solved"],
            status_counts["max_depth_reached"],
            status_counts["unsolved_empty_frontier"],
            status_counts["unsolved_pruned"],
            status_counts["error"],
        ],
        dtype=torch.int64,
        device=device,
    )
    gathered = [torch.zeros_like(local_counts) for _ in range(world_size)]
    dist.all_gather(gathered, local_counts)
    dist.barrier()

    elapsed_total = time.perf_counter() - run_start
    if rank == 0:
        global_errors = sum(int(item[5].cpu().item()) for item in gathered)
        output_rows = max(sum(1 for _ in output_path.open("r", encoding="utf-8")) - 1, 0)
        if global_errors != 0:
            raise AssertionError(f"error_count={global_errors}")
        if output_rows != task_count:
            raise AssertionError(f"output_rows={output_rows}, expected={task_count}")
        print(
            "RUN_SUMMARY "
            f"elapsed_sec={elapsed_total:.3f} tasks_per_sec={task_count / max(elapsed_total, 1e-9):.6f} "
            f"solved={status_counts['solved']} max_depth_reached={status_counts['max_depth_reached']} "
            f"unsolved={status_counts['unsolved_empty_frontier'] + status_counts['unsolved_pruned']} "
            f"error={status_counts['error']} output_rows={output_rows}",
            flush=True,
        )
        print("REAL_SOLVE_100_DEPTH300_LOAD_WORLD2_OK rank=0", flush=True)
    elif not aborted:
        print("REAL_SOLVE_100_DEPTH300_LOAD_WORLD2_OK rank=1", flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
