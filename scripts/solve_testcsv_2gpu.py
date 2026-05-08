#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import beam_engine
import data_loader
from fullbeamnice_current_solver_2gpu import choose_valid_solution_path, reconstruct_path


def allreduce_i64(values: list[int], device: torch.device) -> list[int]:
    t = torch.tensor(values, dtype=torch.int64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return [int(x) for x in t.cpu().tolist()]


def gather_objects(obj):
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    out = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(out, obj)
    return out


def load_test_rows() -> list[tuple[int, np.ndarray]]:
    known_scramble = os.environ.get("KNOWN_SCRAMBLE", "").strip()
    if known_scramble:
        actions = [x.strip() for x in known_scramble.split(",") if x.strip()]
        state = data_loader.apply_actions_cpu(data_loader.get_central_state_u8(), actions)
        return [(-1, state)]

    rows = data_loader.load_test_puzzles()
    ids_text = os.environ.get("TEST_IDS", "").strip()
    if ids_text:
        wanted = {int(x) for x in ids_text.split(",") if x.strip()}
        rows = [(i, s) for i, s in rows if i in wanted]
    start = int(os.environ.get("TEST_START", "0"))
    count = int(os.environ.get("TEST_COUNT", "0"))
    if start > 0 or count > 0:
        rows = rows[start : (start + count if count > 0 else None)]
    return rows


def export_scorer(cfg: dict) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "scripts" / "export_fullbeamnice_scorer.py"),
        "--copies",
        str(cfg["inference_parallelism"]),
        "--out-dir",
        str(PROJECT_DIR / "runtime" / "fullbeamnice_scorers" / f"rank{cfg['rank']}"),
    ]
    out = subprocess.check_output(cmd, cwd=str(PROJECT_DIR), text=True)
    cfg["torchscript_scorer_paths"] = [line for line in out.splitlines() if line.startswith("TORCHSCRIPT_SCORER_PATHS=")][0].split("=", 1)[1]
    cfg["inference_backend"] = "torchscript_ensemble"
    if cfg["rank"] == 0 and os.environ.get("QUIET", "1") == "0":
        print("SCORER_EXPORT_OUTPUT")
        print(out, flush=True)


def solve_one(engine, cfg: dict, state: np.ndarray, max_depth: int, device: torch.device):
    central = data_loader.get_central_state_u8()
    skip_path_validation = os.environ.get("SKIP_PATH_VALIDATION", "0") != "0"
    owner = data_loader.owner_rank_for_state(state, cfg["world_size"])
    engine.reset_search(np.asarray(state, dtype=np.uint8).tobytes(), cfg["rank"] == owner)
    found_depth = -1
    final_sums = None
    for depth in range(max_depth + 1):
        if depth > 0:
            engine.step(histogram_period_micro=cfg["histogram_period_micro"])
        st = dict(engine.status())
        counters = [int(x) for x in st["counters"]]
        sums = allreduce_i64([int(st["found"]), counters[4], counters[5], int(st["cuda_graph_captured"])], device)
        final_sums = sums
        if sums[1] != 0 or sums[2] != 0:
            raise AssertionError(f"overflow at depth={depth}: bucket={sums[1]} hash={sums[2]}")
        if sums[0] > 0:
            found_depth = depth
            break

    local_found = dict(engine.status())
    reports = gather_objects(
        {
            "rank": cfg["rank"],
            "found": int(local_found.get("local_found", local_found["found"])),
            "found_local_index": int(local_found["found_local_index"]),
        }
    )
    found_rank = -1
    found_local_index = -1
    for item in reports:
        if int(item["found"]) != 0:
            found_rank = int(item["rank"])
            found_local_index = int(item["found_local_index"])
            break

    selected_path: list[str] = []
    restore_ok = False
    restore_distance = -1
    if found_depth > 0 and found_rank >= 0:
        _, raw_path, _ = reconstruct_path(engine, cfg, found_depth, found_rank, found_local_index)
        variant, selected_path, restore_ok, restore_distance, _ = choose_valid_solution_path(state, central, raw_path)
        if not restore_ok:
            if skip_path_validation:
                selected_path = raw_path
                return {
                    "found": False,
                    "depth": found_depth,
                    "path": ".".join(selected_path),
                    "restore_distance": restore_distance,
                    "cuda_graph_captured_sum": int(final_sums[3]) if final_sums is not None else 0,
                    "final_sums": final_sums,
                }
            raise AssertionError(
                "path validation failed; "
                + json.dumps({"variant": variant, "distance": restore_distance, "raw_path": raw_path}, ensure_ascii=False)
            )
    elif found_depth == 0:
        restore_ok = bool(np.array_equal(state, central))

    return {
        "found": found_depth >= 0 and restore_ok,
        "depth": found_depth,
        "path": ".".join(selected_path),
        "restore_distance": restore_distance,
        "cuda_graph_captured_sum": int(final_sums[3]) if final_sums is not None else 0,
        "final_sums": final_sums,
    }


def main() -> None:
    os.environ.setdefault("USE_CUDA_GRAPHS", "1")
    os.environ.setdefault("INFERENCE_BACKEND", "torchscript_ensemble")
    os.environ.setdefault("INFERENCE_PARALLELISM", "1")
    os.environ.setdefault("K_EXPAND_TILE", "32768")
    os.environ.setdefault("GLOBAL_BEAM_WIDTH", str(2**16))
    os.environ.setdefault("B_MICRO", "32768")
    os.environ.setdefault("SCORE_RING_DEPTH", "8")
    os.environ.setdefault("NET_RING_DEPTH", "2")
    os.environ.setdefault("BUCKET_CAP_PER_PEER", "65536")
    os.environ.setdefault("BETA", "1.20")
    os.environ.setdefault("HASH_LOAD_FACTOR", "0.45")
    os.environ.setdefault("PROBE_LIMIT", "256")
    os.environ.setdefault("MAX_DEPTH", "100")
    os.environ.setdefault("HISTOGRAM_PERIOD_MICRO", "2")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    cfg = beam_engine.make_default_config()
    beam_engine.init_distributed_if_needed(cfg)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    export_scorer(cfg)

    ext = beam_engine.build_extension(verbose=os.environ.get("BUILD_VERBOSE", "0") == "1")
    buffers = beam_engine.allocate_buffers(ext, cfg)
    engine = beam_engine.configure_engine(ext, cfg, buffers)
    rows = load_test_rows()
    max_depth = int(os.environ["MAX_DEPTH"])
    output_path = Path(os.environ.get("SUBMISSION_PATH", str(PROJECT_DIR / "submission.csv")))
    log_every = int(os.environ.get("LOG_EVERY", "25"))

    solved_rows: list[dict[str, str]] = []
    t0 = time.time()
    for pos, (sample_id, state) in enumerate(rows):
        result = solve_one(engine, cfg, state, max_depth, device)
        if os.environ.get("USE_CUDA_GRAPHS", "0") != "0" and result["depth"] >= 2 and result["cuda_graph_captured_sum"] < cfg["world_size"]:
            raise AssertionError(
                "CUDA graph was not captured on all ranks; "
                + json.dumps({"cuda_graph_captured_sum": result["cuda_graph_captured_sum"], "world_size": cfg["world_size"]}, ensure_ascii=False)
            )
        if cfg["rank"] == 0:
            solved_rows.append({"initial_state_id": str(sample_id), "path": result["path"]})
            if log_every > 0 and (pos % log_every == 0 or pos + 1 == len(rows)):
                print(
                    "SAMPLE_RESULT "
                    + json.dumps(
                        {
                            "pos": pos,
                            "id": sample_id,
                            "found": result["found"],
                            "depth": result["depth"],
                            "path_len": 0 if result["path"] == "" else len(result["path"].split(".")),
                            "path": result["path"],
                            "cuda_graph_captured_sum": result["cuda_graph_captured_sum"],
                            "elapsed_sec": round(time.time() - t0, 3),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    if cfg["rank"] == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["initial_state_id", "path"])
            writer.writeheader()
            writer.writerows(solved_rows)
        print(
            "SUBMISSION_WRITTEN "
            + json.dumps({"path": str(output_path), "rows": len(solved_rows), "elapsed_sec": round(time.time() - t0, 3)}, ensure_ascii=False),
            flush=True,
        )

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"FATAL_EXIT type={type(exc).__name__}; message={exc}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
