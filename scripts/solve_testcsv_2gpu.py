#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
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
import cpu_history_archive
import data_loader
from fullbeamnice_current_solver_2gpu import broadcast_entry_from_owner, choose_valid_solution_path, reconstruct_path


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
    if str(cfg.get("inference_backend", "")).strip().lower() != "torchscript_ensemble":
        return
    if os.environ.get("ALLOW_TORCHSCRIPT_SCORER", "").strip().lower() not in {"1", "true", "yes", "on"}:
        raise ValueError(
            "export_scorer: INFERENCE_BACKEND=torchscript_ensemble requires ALLOW_TORCHSCRIPT_SCORER=1 "
            "(TorchScript is opt-in; use fullbeamnice_static on Kaggle for CUTLASS)."
        )
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


COUNTER_LABELS = (
    "next_pool_size",
    "local_inserted",
    "local_duplicate",
    "remote_packed",
    "bucket_overflow",
    "hash_overflow",
    "pruned",
    "local_updated",
)


def _depth_tuning_log_enabled() -> bool:
    return os.environ.get("DEPTH_TUNING_LOG", "0").strip().lower() in {"1", "true", "yes", "on"}


def _tuning_snapshot(cfg: dict) -> dict:
    return {
        "global_beam_width": int(cfg["global_beam_width"]),
        "world_size": int(cfg["world_size"]),
        "b_micro": int(cfg["b_micro"]),
        "inference_parallelism": int(cfg["inference_parallelism"]),
        "k_expand_tile": int(cfg["k_expand_tile"]),
        "histogram_period_micro": int(cfg["histogram_period_micro"]),
        "bucket_cap_per_peer": int(cfg["bucket_cap_per_peer"]),
        "score_ring_depth": int(cfg["score_ring_depth"]),
        "net_ring_depth": int(cfg["net_ring_depth"]),
        "beta": float(cfg["beta"]),
        "hash_load_factor": float(cfg["hash_load_factor"]),
        "probe_limit": int(cfg["probe_limit"]),
        "inference_backend": str(cfg.get("inference_backend", "")),
        "use_cuda_graphs": os.environ.get("USE_CUDA_GRAPHS", "1") != "0",
    }


def _count_micro_and_tiles(local_frontier: int, cfg: dict, n_local: int) -> tuple[int, int]:
    alim = min(max(1, int(local_frontier)), int(n_local))
    bm = int(cfg["b_micro"])
    if bm < 1:
        bm = 1
    num_micro = (alim + bm - 1) // bm
    fanout = int(cfg.get("fanout", 24))
    kt = int(cfg.get("k_expand_tile", 0))
    total_tiles = 0
    for mb in range(num_micro):
        start = mb * bm
        micro = min(bm, alim - start)
        total_lanes = micro * fanout
        tgt = kt if kt > 0 else total_lanes
        if tgt < 1:
            tgt = total_lanes
        total_tiles += (total_lanes + tgt - 1) // tgt
    return int(num_micro), int(total_tiles)


def estimate_no_inference_prepass_depth(cfg: dict, max_depth: int) -> int:
    if os.environ.get("PREPASS_NO_INFERENCE", "1").strip().lower() in {"", "0", "false", "no", "off"}:
        return 0
    manual = os.environ.get("PREPASS_DEPTH", "").strip()
    if manual:
        return max(0, min(int(manual), max_depth))
    fanout = int(cfg.get("fanout", 24))
    target = int(cfg["global_beam_width"])
    dedup_factor = float(os.environ.get("PREPASS_DEDUP_FACTOR", "0.95"))
    depth = 0
    estimated = 1.0
    while depth < max_depth and estimated < target:
        depth += 1
        estimated = (fanout ** depth) * dedup_factor
    # Ensure prepass_depth does not exceed max_depth - 1 to leave at least one depth for full inference
    prepass_depth = min(depth, max_depth - 1) if max_depth > 0 else 0
    return max(0, prepass_depth)


def prepass_expected_caps() -> list[int]:
    raw = os.environ.get("PREPASS_EXPECTED_CAPS", "1,24,469,7779,104720,1334491").strip()
    caps: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        caps.append(max(1, int(token)))
    if not caps:
        caps = [1]
    if caps[0] != 1:
        caps.insert(0, 1)
    return caps


def prepass_cap_for_depth(caps: list[int], depth: int, fallback: int) -> int:
    if depth < 0:
        return 1
    if depth < len(caps):
        return int(caps[depth])
    return int(fallback)


def reconstruct_path_from_archive(archive: cpu_history_archive.CPUHistoryArchive, cfg: dict, found_depth: int, found_rank: int, found_local_index: int):
    actions: list[int] = []
    owner_rank = int(found_rank)
    local_index = int(found_local_index)
    for history_depth in range(found_depth - 1, -1, -1):
        local_entry = None
        if cfg["rank"] == owner_rank:
            local_entry = archive.history_entry(history_depth, local_index)
        entry = broadcast_entry_from_owner(owner_rank, local_entry)
        if int(entry["valid"]) != 1:
            raise RuntimeError(f"invalid CPU history entry at depth={history_depth}; rank={owner_rank}; index={local_index}")
        actions.append(int(entry["action"]))
        local_index = int(entry["parent_idx"])
        owner_rank = int(entry["parent_rank"])
    actions.reverse()
    current_names = [data_loader.ACTION_NAMES[a] for a in actions]
    fullbeamnice_names = [x[1:] + "'" if x.startswith("-") else x for x in current_names]
    return actions, current_names, fullbeamnice_names


def solve_one(engine, cfg: dict, buffers: dict, sample_id: int, state: np.ndarray, max_depth: int, device: torch.device):
    central = data_loader.get_central_state_u8()
    skip_path_validation = os.environ.get("SKIP_PATH_VALIDATION", "0") != "0"
    owner = data_loader.owner_rank_for_state(state, cfg["world_size"])
    archive = cpu_history_archive.CPUHistoryArchive(cfg, buffers, sample_id) if cfg.get("history_backend") == "cpu" else None
    resume_depth = archive.try_resume(device) if archive is not None else None
    if resume_depth is None:
        engine.reset_search(np.asarray(state, dtype=np.uint8).tobytes(), cfg["rank"] == owner)
        if archive is not None:
            archive.start_new(state, cfg["rank"] == owner)
        start_depth = 0
    else:
        start_depth = int(resume_depth)
    found_depth = -1
    final_sums = None
    beam_debug = os.environ.get("BEAM_DEBUG", os.environ.get("ENGINE_DEBUG", "0")).strip().lower() not in {"", "0", "false", "no", "off"}
    depth_log_every = int(os.environ.get("DEPTH_LOG_EVERY", "0")) if beam_debug else 0
    depth_tuning_log = _depth_tuning_log_enabled()
    n_local_buf = int(buffers["beam_current"].shape[0])
    prepass_depth = estimate_no_inference_prepass_depth(cfg, max_depth) if start_depth == 0 and resume_depth is None else 0
    prepass_hist_micro = int(os.environ.get("PREPASS_HISTOGRAM_PERIOD_MICRO", "1048576"))
    prepass_stop_at_width = os.environ.get("PREPASS_STOP_AT_WIDTH", "1").strip().lower() not in {"", "0", "false", "no", "off"}
    prepass_width_frac = float(os.environ.get("PREPASS_STOP_WIDTH_FRAC", "0.98"))
    target_width_stop = max(1, int(float(cfg["global_beam_width"]) * prepass_width_frac))
    fanout = int(cfg.get("fanout", 24))
    dynamic_prepass = os.environ.get("PREPASS_DYNAMIC_WIDTH", "1").strip().lower() not in {"", "0", "false", "no", "off"}
    prepass_caps = prepass_expected_caps()
    next_limit_buf = int(buffers["next_state_pool"].shape[0])
    uniform_active = False
    if prepass_depth > 0:
        engine.set_prepass_light_solved_scan(True)
        engine.begin_uniform_score(1)
        uniform_active = True
    last_local_frontier: int | None = None
    for depth in range(start_depth, max_depth + 1):
        phase = (
            "init"
            if depth == 0
            else (
                "uniform_fill"
                if (uniform_active and prepass_depth > 0 and depth <= prepass_depth)
                else "full_solver"
            )
        )
        wall_ms: float | None = None
        step_hist_micro: int | None = None
        step_uniform = False
        step_active_limit: int | None = None
        step_next_limit: int | None = None
        num_micro = 0
        total_tiles = 0
        if depth > start_depth:
            step_uniform = bool(uniform_active and depth <= prepass_depth)
            hist_micro = prepass_hist_micro if uniform_active else int(cfg["histogram_period_micro"])
            step_hist_micro = int(hist_micro)
            if last_local_frontier is not None:
                num_micro, total_tiles = _count_micro_and_tiles(last_local_frontier, cfg, n_local_buf)
            if uniform_active:
                if depth <= prepass_depth:
                    engine.set_uniform_score(max(1, min(depth, 65535)))
                    active_limit = max(1, int(last_local_frontier) if last_local_frontier is not None else prepass_cap_for_depth(prepass_caps, depth - 1, n_local_buf))
                    next_limit = min(next_limit_buf, prepass_cap_for_depth(prepass_caps, depth, next_limit_buf))
                    step_active_limit = int(active_limit)
                    step_next_limit = int(next_limit)
                    engine.set_active_limit(active_limit)
                    engine.set_next_limit(next_limit)
                else:
                    engine.set_prepass_light_solved_scan(False)
                    engine.end_uniform_score()
                    engine.clear_logical_limits()
                    uniform_active = False
            if depth_tuning_log:
                torch.cuda.synchronize(device=device)
            t_wall0 = time.perf_counter()
            if uniform_active:
                engine.step_current(histogram_period_micro=hist_micro)
            else:
                engine.clear_logical_limits()
                engine.step(histogram_period_micro=int(cfg["histogram_period_micro"]))
            if depth_tuning_log:
                torch.cuda.synchronize(device=device)
            wall_ms = (time.perf_counter() - t_wall0) * 1000.0
        st = dict(engine.status())
        counters = [int(x) for x in st["counters"]]
        sums = allreduce_i64(
            [
                int(st["found"]),
                counters[4],
                counters[5],
                int(st["cuda_graph_captured"]),
                int(st.get("local_found", 0)),
                int(st["current_size"]),
                int(st["compacted_size"]),
            ],
            device,
        )
        final_sums = sums
        max_wall_ms: float | None = None
        if depth_tuning_log and depth > start_depth and wall_ms is not None:
            tw = torch.tensor([wall_ms], dtype=torch.float64, device=device)
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(tw, op=dist.ReduceOp.MAX)
            max_wall_ms = float(tw.item())
        if cfg["rank"] == 0 and depth_tuning_log and depth > start_depth and wall_ms is not None and max_wall_ms is not None:
            ctr = [int(x) for x in st["counters"]]
            named = {COUNTER_LABELS[i]: ctr[i] for i in range(min(len(COUNTER_LABELS), len(ctr)))}
            print(
                "DEPTH_TUNING "
                + json.dumps(
                    {
                        "depth": depth,
                        "phase": phase,
                        "step_uniform": step_uniform,
                        "histogram_period_micro_used": step_hist_micro,
                        "wall_ms_local": round(wall_ms, 3),
                        "wall_ms_max_rank": round(max_wall_ms, 3),
                        "local_frontier_before_step": last_local_frontier,
                        "logical_active_limit": step_active_limit,
                        "logical_next_limit": step_next_limit,
                        "num_micro_batches": num_micro,
                        "expand_tiles_upper_bound": total_tiles,
                        "local_current_size_after": int(st["current_size"]),
                        "local_compacted_after": int(st["compacted_size"]),
                        "current_size_sum": sums[5],
                        "compacted_size_sum": sums[6],
                        "counters": named,
                        "tuning_params": _tuning_snapshot(cfg),
                        "notes": (
                            "wall_ms = host time with cuda.synchronize before/after engine step (all streams idle); "
                            "Stream1 inference overlaps Stream2/3 in hardware — use max_wall_ms to spot stragglers; "
                            "expand_tiles_upper_bound counts process_score tiles (Stream2+NCCL work driver); "
                            "often smaller b_micro with higher inference_parallelism improves GPU utilization vs few large microbatches."
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        last_local_frontier = int(st["current_size"])
        if sums[1] != 0 or sums[2] != 0:
            raise AssertionError(f"overflow at depth={depth}: bucket={sums[1]} hash={sums[2]}")
        if archive is not None and depth > start_depth:
            archive.after_step(depth, int(st["current_size"]))
        if cfg["rank"] == 0 and depth_log_every > 0 and (depth % depth_log_every == 0 or sums[0] > 0):
            print(
                "DEPTH_RESULT "
                + json.dumps(
                    {
                        "depth": depth,
                        "found_sum": sums[0],
                        "local_found_sum": sums[4],
                        "current_size_sum": sums[5],
                        "compacted_size_sum": sums[6],
                        "bucket_overflow": sums[1],
                        "hash_overflow": sums[2],
                        "cuda_graph_captured_sum": sums[3],
                        "prepass_depth": prepass_depth,
                        "phase": phase,
                        "histogram_period_micro": (prepass_hist_micro if uniform_active and depth <= prepass_depth else int(cfg["histogram_period_micro"])),
                        "logical_active_limit": step_active_limit,
                        "logical_next_limit": step_next_limit,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if sums[0] > 0:
            if sums[4] <= 0:
                raise AssertionError(
                    "global found without local_found owner; "
                    + json.dumps({"depth": depth, "sums": sums}, ensure_ascii=False)
                )
            found_depth = depth
            break
        candidate_upper = int(sums[5]) * fanout
        should_stop_prepass = (
            uniform_active
            and prepass_stop_at_width
            and depth >= start_depth
            and depth <= prepass_depth
            and (
                int(sums[5]) >= target_width_stop
                or (dynamic_prepass and candidate_upper > int(cfg["global_beam_width"]))
            )
        )

        if should_stop_prepass:
            if cfg["rank"] == 0:
                print(
                    "PREPASS_WIDTH_REACHED "
                    + json.dumps(
                        {
                            "depth": depth,
                            "current_size_sum": int(sums[5]),
                            "candidate_upper": candidate_upper,
                            "global_beam_width": int(cfg["global_beam_width"]),
                            "target_width_stop": target_width_stop,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            engine.set_prepass_light_solved_scan(False)
            engine.end_uniform_score()
            engine.clear_logical_limits()
            uniform_active = False
    if uniform_active:
        engine.set_prepass_light_solved_scan(False)
        engine.end_uniform_score()
        engine.clear_logical_limits()

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
    found_reports = []
    for item in reports:
        if int(item["found"]) != 0:
            found_reports.append(item)
    if found_reports:
        found_reports.sort(key=lambda x: int(x["rank"]))
        found_rank = int(found_reports[0]["rank"])
        found_local_index = int(found_reports[0]["found_local_index"])
        if found_local_index < 0:
            raise AssertionError(
                "local_found owner reported invalid found_local_index; "
                + json.dumps({"found_reports": found_reports}, ensure_ascii=False)
            )

    if archive is not None:
        archive.finish()

    selected_path: list[str] = []
    restore_ok = False
    restore_distance = -1
    if found_depth > 0 and found_rank >= 0:
        if archive is not None:
            _, raw_path, _ = reconstruct_path_from_archive(archive, cfg, found_depth, found_rank, found_local_index)
        else:
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


def initialize_submission(path: Path, *, resume: bool) -> None:
    if resume and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["initial_state_id", "path"])
        writer.writeheader()
        f.flush()


def append_submission_row(path: Path, sample_id: int, solution_path: str) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["initial_state_id", "path"])
        writer.writerow({"initial_state_id": str(sample_id), "path": solution_path})
        f.flush()


def existing_submission_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {int(row["initial_state_id"]) for row in reader if row.get("initial_state_id", "").strip()}


def main() -> None:
    os.environ.setdefault("USE_CUDA_GRAPHS", "1")
    os.environ.setdefault("INFERENCE_BACKEND", "fullbeamnice_static")
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
    os.environ.setdefault("PREPASS_DEPTH", "60")
    os.environ.setdefault("PREPASS_DYNAMIC_WIDTH", "1")
    os.environ.setdefault("PREPASS_STOP_AT_WIDTH", "1")
    os.environ.setdefault("PREPASS_STOP_WIDTH_FRAC", "0.98")
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
    sample_log_every = int(os.environ.get("SAMPLE_LOG_EVERY", os.environ.get("LOG_EVERY", "1")))
    append_each = os.environ.get("SUBMISSION_APPEND_EACH", "1") != "0"
    resume_submission = os.environ.get("RESUME_SUBMISSION", "0") != "0"
    if cfg["rank"] == 0 and resume_submission:
        done_ids = existing_submission_ids(output_path)
    else:
        done_ids = set()
    done_ids_list = [done_ids]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(done_ids_list, src=0)
    done_ids = set(done_ids_list[0])
    if resume_submission and done_ids:
        rows = [(sample_id, state) for sample_id, state in rows if sample_id not in done_ids]

    solved_rows: list[dict[str, str]] = []
    t0 = time.time()
    graph_expected = os.environ.get("USE_CUDA_GRAPHS", "0") != "0"
    if cfg["rank"] == 0 and append_each:
        initialize_submission(output_path, resume=resume_submission)
    for pos, (sample_id, state) in enumerate(rows):
        result = solve_one(engine, cfg, buffers, sample_id, state, max_depth, device)
        if graph_expected and result["depth"] >= 2 and result["cuda_graph_captured_sum"] < cfg["world_size"]:
            raise AssertionError(
                "CUDA graph was not captured on all ranks; "
                + json.dumps({"cuda_graph_captured_sum": result["cuda_graph_captured_sum"], "world_size": cfg["world_size"]}, ensure_ascii=False)
            )
        if cfg["rank"] == 0:
            row = {"initial_state_id": str(sample_id), "path": result["path"]}
            solved_rows.append(row)
            if append_each:
                append_submission_row(output_path, sample_id, result["path"])
            if sample_log_every > 0 and (pos % sample_log_every == 0 or pos + 1 == len(rows)):
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
        if not append_each:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["initial_state_id", "path"])
                writer.writeheader()
                writer.writerows(solved_rows)
        print(
            "SUBMISSION_WRITTEN "
            + json.dumps(
                {
                    "path": str(output_path),
                    "rows": len(solved_rows),
                    "append_each": append_each,
                    "elapsed_sec": round(time.time() - t0, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    sys.stdout.flush()
    sys.stderr.flush()
    # COMMENTED OUT: os._exit(0)  # Kaggle submission


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"FATAL_EXIT type={type(exc).__name__}; message={exc}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
