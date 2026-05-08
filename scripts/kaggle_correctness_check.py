"""
Kaggle/torchrun correctness check for stream1/stream2/stream3 and CUDA Graph path.
Run examples:
  WORLD_SIZE=1 USE_CUDA_GRAPHS=1 python scripts/kaggle_correctness_check.py
  USE_CUDA_GRAPHS=1 torchrun --standalone --nproc_per_node=2 scripts/kaggle_correctness_check.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import data_loader
import beam_engine
from scorers import make_default_mlp_scorer


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def running_on_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle").exists()


def should_fast_exit_after_report(cfg: dict) -> bool:
    if int(cfg.get("world_size", 1)) <= 1:
        return False
    default = running_on_kaggle()
    return env_flag("KAGGLE_FAST_TORCHRUN_EXIT", default)


def finalize_after_report(cfg: dict) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    if should_fast_exit_after_report(cfg):
        print(
            f"teardown_fast_exit=true; rank={cfg['rank']}; reason=avoid_kaggle_nccl_destroy_hang_after_success_report",
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if env_flag("DESTROY_PROCESS_GROUP_ON_EXIT", True) and dist.is_available() and dist.is_initialized():
        print(f"destroy_process_group_start=true; rank={cfg['rank']}", flush=True)
        dist.destroy_process_group()
        print(f"destroy_process_group_done=true; rank={cfg['rank']}", flush=True)


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def allreduce_ints(values: list[int], device: torch.device) -> list[int]:
    t = torch.tensor(values, dtype=torch.int32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return [int(x) for x in t.cpu().tolist()]


def export_torchscript_copies(copies: int, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((16, 120), dtype=torch.uint8)
    paths: list[str] = []
    for i in range(copies):
        torch.manual_seed(7000 + i)
        model = make_default_mlp_scorer(hidden_sizes=(1024, 256), fanout=24).eval()
        traced = torch.jit.trace(model, example, strict=False)
        traced = torch.jit.freeze(traced)
        path = out_dir / f"correctness_mlp_copy{i:02d}.ts"
        traced.save(str(path))
        paths.append(str(path))
    return paths


def run_torchscript_ensemble_case(ext, cfg: dict, device: torch.device) -> dict:
    copies = max(1, int(cfg.get("inference_parallelism", 1)))
    paths = export_torchscript_copies(copies, PROJECT_DIR / "runtime" / "scorers" / f"rank{cfg['rank']}")
    cfg_ts = dict(cfg)
    cfg_ts["inference_backend"] = "torchscript_ensemble"
    cfg_ts["torchscript_scorer_paths"] = os.pathsep.join(paths)
    cfg_ts["inference_parallelism"] = copies
    buffers_ts = beam_engine.allocate_buffers(ext, cfg_ts)
    engine_ts = beam_engine.configure_engine(ext, cfg_ts, buffers_ts)
    state = data_loader.apply_actions_cpu(data_loader.get_central_state_u8(), ["U"])
    owner = data_loader.owner_rank_for_state(state, cfg_ts["world_size"])
    engine_ts.reset_search(state.tobytes(), cfg_ts["rank"] == owner)
    result = engine_ts.search(max_depth=1, histogram_period_micro=cfg_ts["histogram_period_micro"])
    st = dict(engine_ts.status())
    counters = [int(x) for x in st["counters"]]
    sums = allreduce_ints([int(st["found"]), counters[1], counters[3], counters[4], counters[5]], device)
    assert_true(sums[0] >= 1, "torchscript ensemble depth1 solution not found")
    assert_true(sums[1] + sums[2] > 0, "torchscript ensemble produced zero candidates")
    assert_true(sums[3] == 0, f"torchscript ensemble bucket overflow detected: {sums[3]}")
    assert_true(sums[4] == 0, f"torchscript ensemble hash overflow detected: {sums[4]}")
    assert_true(bool(engine_ts.cuda_graph_captured()), "torchscript ensemble CUDA Graph was not captured")
    return {
        "paths": paths,
        "copies": copies,
        "local_result": dict(result),
        "local_status": st,
        "global_sums": {
            "found": sums[0],
            "local_inserted": sums[1],
            "remote_packed": sums[2],
            "bucket_overflow": sums[3],
            "hash_overflow": sums[4],
        },
    }


def main() -> None:
    os.environ.setdefault("USE_CUDA_GRAPHS", "1")
    os.environ.setdefault("INFERENCE_BACKEND", "central_hamming")
    os.environ.setdefault("GLOBAL_BEAM_WIDTH", "32768")
    os.environ.setdefault("B_MICRO", "4096")
    os.environ.setdefault("SCORE_RING_DEPTH", "8")
    os.environ.setdefault("NET_RING_DEPTH", "2")
    os.environ.setdefault("BUCKET_CAP_PER_PEER", "65536")
    os.environ.setdefault("INFERENCE_PARALLELISM", "2")
    os.environ.setdefault("MAX_DEPTH", "3")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    cfg = beam_engine.make_default_config()
    beam_engine.init_distributed_if_needed(cfg)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")) % torch.cuda.device_count())

    data_loader.validate_inverse_pairs()
    test_sample = data_loader.load_test_puzzles(max_puzzles=3)
    assert_true(len(test_sample) > 0, "test.csv must contain at least one puzzle")

    ext = beam_engine.build_extension(verbose=os.environ.get("BUILD_VERBOSE", "0") != "0")
    buffers = beam_engine.allocate_buffers(ext, cfg)
    engine = beam_engine.configure_engine(ext, cfg, buffers)

    cases = [
        ("depth0_central", [], 0),
        ("depth1_U", ["U"], 1),
        ("depth2_U_R", ["U", "R"], 2),
    ]

    case_results = []
    for case_name, scramble, max_depth in cases:
        state = data_loader.apply_actions_cpu(data_loader.get_central_state_u8(), scramble)
        owner = data_loader.owner_rank_for_state(state, cfg["world_size"])
        engine.reset_search(state.tobytes(), cfg["rank"] == owner)
        result = engine.search(max_depth=max_depth, histogram_period_micro=cfg["histogram_period_micro"])
        st = engine.status()
        local_found = int(st["found"])
        found_sum = allreduce_ints([local_found], device)[0]
        case_results.append({
            "case": case_name,
            "scramble": scramble,
            "owner": owner,
            "local_result": dict(result),
            "local_status": dict(st),
            "global_found_sum": found_sum,
        })
        assert_true(found_sum >= 1, f"case {case_name}: global solution not found")
        if max_depth > 0:
            assert_true(int(st["cuda_graph_captured"]) == 1 or engine.cuda_graph_captured(), f"case {case_name}: CUDA Graph was not captured")

    # One explicit one-step expansion from a non-central test.csv state validates CSV ingestion.
    puzzle_id, csv_state = test_sample[0]
    owner = data_loader.owner_rank_for_state(csv_state, cfg["world_size"])
    engine.reset_search(csv_state.tobytes(), cfg["rank"] == owner)
    engine.step(histogram_period_micro=cfg["histogram_period_micro"])
    st = dict(engine.status())
    counters = [int(x) for x in st["counters"]]
    local_inserted = counters[1]
    remote_packed = counters[3]
    bucket_overflow = counters[4]
    hash_overflow = counters[5]
    sums = allreduce_ints([local_inserted, remote_packed, bucket_overflow, hash_overflow, int(st["current_size"])], device)
    assert_true(sums[0] + sums[1] > 0, "test.csv one-step expansion produced zero candidates")
    assert_true(sums[2] == 0, f"bucket overflow detected: {sums[2]}")
    assert_true(sums[3] == 0, f"hash overflow detected: {sums[3]}")
    if cfg["world_size"] > 1:
        assert_true(sums[1] > 0, "stream3/NCCL path did not pack remote candidates")

    torchscript_report = None
    if os.environ.get("TEST_TORCHSCRIPT_ENSEMBLE", "1") != "0":
        torchscript_report = run_torchscript_ensemble_case(ext, cfg, device)

    report = {
        "rank": cfg["rank"],
        "world_size": cfg["world_size"],
        "cuda_graph_captured": bool(engine.cuda_graph_captured()),
        "cases": case_results,
        "torchscript_ensemble": torchscript_report,
        "csv_step": {
            "puzzle_id": int(puzzle_id),
            "owner": int(owner),
            "local_status": st,
            "global_sums": {
                "local_inserted": sums[0],
                "remote_packed": sums[1],
                "bucket_overflow": sums[2],
                "hash_overflow": sums[3],
                "current_size": sums[4],
            },
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    finalize_after_report(cfg)


if __name__ == "__main__":
    main()
