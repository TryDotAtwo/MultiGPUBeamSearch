#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import beam_engine
import data_loader


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


def broadcast_entry_from_owner(owner_rank: int, local_entry):
    box = [local_entry if dist.get_rank() == owner_rank else None]
    dist.broadcast_object_list(box, src=owner_rank)
    return box[0]


def make_fullbeamnice_random_walk_state(device: torch.device):
    root = PROJECT_DIR / "FullBeamNice"
    spec = json.loads((root / "generators" / "p900.json").read_text(encoding="utf-8"))
    moves = torch.tensor(spec["actions"], dtype=torch.int64, device=device)
    names = list(spec["names"])
    inverse = [names.index(m.replace("'", "")) if "'" in m else names.index(m + "'") for m in names]
    state = torch.load(root / "targets" / "p900-t000.pt", map_location=device, weights_only=True).to(torch.int64).unsqueeze(0)
    prev = torch.full((1,), -1, dtype=torch.int64)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)
    inv_cpu = torch.tensor(inverse, dtype=torch.int64)
    path = []
    for _ in range(20):
        nxt = torch.randint(moves.size(0), (1,), generator=rng, dtype=torch.int64)
        invalid = (prev >= 0) & (nxt == inv_cpu[prev.clamp_min(0)])
        while bool(invalid.any()):
            nxt[invalid] = torch.randint(moves.size(0), (int(invalid.sum().item()),), generator=rng, dtype=torch.int64)
            invalid = (prev >= 0) & (nxt == inv_cpu[prev.clamp_min(0)])
        state = torch.gather(state, 1, moves[nxt.to(device)])
        path.append(names[int(nxt.item())])
        prev = nxt
    return state.squeeze(0).to(torch.uint8).cpu().numpy(), path


def fullbeamnice_name_from_current(name: str) -> str:
    return name[1:] + "'" if name.startswith("-") else name


def inverse_current_path(names: list[str]) -> list[str]:
    return [data_loader.inverse_action_name(name) for name in reversed(names)]


def choose_valid_solution_path(initial_state: np.ndarray, central: np.ndarray, path_current: list[str]):
    variants = {
        "direct": path_current,
        "inverse_reversed": inverse_current_path(path_current),
        "reversed": list(reversed(path_current)),
        "inverse_each": [data_loader.inverse_action_name(name) for name in path_current],
    }
    best_name = ""
    best_path: list[str] = []
    best_distance = 10**9
    best_restored = None
    for name, candidate in variants.items():
        restored = data_loader.apply_actions_cpu(initial_state, candidate)
        distance = int(np.count_nonzero(restored != central))
        if distance < best_distance:
            best_name = name
            best_path = candidate
            best_distance = distance
            best_restored = restored
        if distance == 0:
            return name, candidate, True, 0, restored
    return best_name, best_path, False, best_distance, best_restored


def reconstruct_path(engine, cfg: dict, found_depth: int, found_rank: int, found_local_index: int):
    actions: list[int] = []
    owner_rank = int(found_rank)
    local_index = int(found_local_index)
    for history_depth in range(found_depth - 1, -1, -1):
        local_entry = None
        if cfg["rank"] == owner_rank:
            local_entry = dict(engine.history_entry(history_depth, local_index))
        entry = broadcast_entry_from_owner(owner_rank, local_entry)
        if int(entry["valid"]) != 1:
            raise RuntimeError(f"invalid history entry at depth={history_depth}; rank={owner_rank}; index={local_index}")
        actions.append(int(entry["action"]))
        local_index = int(entry["parent_idx"])
        owner_rank = int(entry["parent_rank"])
    actions.reverse()
    current_names = [data_loader.ACTION_NAMES[a] for a in actions]
    fullbeamnice_names = [fullbeamnice_name_from_current(x) for x in current_names]
    return actions, current_names, fullbeamnice_names


def main() -> None:
    os.environ.setdefault("USE_CUDA_GRAPHS", "0")
    os.environ.setdefault("INFERENCE_BACKEND", "torchscript_ensemble")
    os.environ.setdefault("INFERENCE_PARALLELISM", "1")
    os.environ.setdefault("K_EXPAND_TILE", "32768")
    os.environ.setdefault("GLOBAL_BEAM_WIDTH", str(2**23))
    os.environ.setdefault("B_MICRO", "32768")
    os.environ.setdefault("SCORE_RING_DEPTH", "8")
    os.environ.setdefault("NET_RING_DEPTH", "2")
    os.environ.setdefault("BUCKET_CAP_PER_PEER", "65536")
    os.environ.setdefault("BETA", "1.20")
    os.environ.setdefault("HASH_LOAD_FACTOR", "0.45")
    os.environ.setdefault("PROBE_LIMIT", "256")
    os.environ.setdefault("MAX_DEPTH", "80")
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

    export_cmd = [
        sys.executable,
        str(PROJECT_DIR / "scripts" / "export_fullbeamnice_scorer.py"),
        "--copies",
        str(cfg["inference_parallelism"]),
        "--out-dir",
        str(PROJECT_DIR / "runtime" / "fullbeamnice_scorers" / f"rank{cfg['rank']}"),
    ]
    out = subprocess.check_output(export_cmd, cwd=str(PROJECT_DIR), text=True)
    if cfg["rank"] == 0:
        print("SCORER_EXPORT_OUTPUT")
        print(out, flush=True)
    cfg["torchscript_scorer_paths"] = [line for line in out.splitlines() if line.startswith("TORCHSCRIPT_SCORER_PATHS=")][0].split("=", 1)[1]
    cfg["inference_backend"] = "torchscript_ensemble"

    data_loader.validate_inverse_pairs()
    central = data_loader.get_central_state_u8()
    full_target = torch.load(PROJECT_DIR / "FullBeamNice" / "targets" / "p900-t000.pt", map_location="cpu", weights_only=True).numpy().astype("uint8")
    if not bool((central == full_target).all()):
        raise AssertionError("central state differs from FullBeamNice target")

    test_state, scramble_path = make_fullbeamnice_random_walk_state(device)
    owner = data_loader.owner_rank_for_state(test_state, cfg["world_size"])

    ext = beam_engine.build_extension(verbose=os.environ.get("BUILD_VERBOSE", "0") == "1")
    buffers = beam_engine.allocate_buffers(ext, cfg)
    engine = beam_engine.configure_engine(ext, cfg, buffers)
    engine.reset_search(test_state.tobytes(), cfg["rank"] == owner)

    reports = []
    found_depth = -1
    for depth in range(0, int(os.environ["MAX_DEPTH"]) + 1):
        if depth > 0:
            engine.step(histogram_period_micro=cfg["histogram_period_micro"])
        st = dict(engine.status())
        counters = [int(x) for x in st["counters"]]
        sums = allreduce_i64(
            [
                int(st["found"]),
                int(st["current_size"]),
                int(st["compacted_size"]),
                counters[0],
                counters[1],
                counters[2],
                counters[3],
                counters[4],
                counters[5],
                counters[6],
                counters[7],
                int(st["cuda_graph_captured"]),
                int(st["threshold_valid"]),
                int(st["threshold_q"]),
            ],
            device,
        )
        row = {
            "depth": depth,
            "global_found": sums[0],
            "global_current_size": sums[1],
            "global_compacted_size": sums[2],
            "global_next_pool_counter": sums[3],
            "global_local_inserted": sums[4],
            "global_local_updated": sums[10],
            "global_remote_packed": sums[6],
            "global_bucket_overflow": sums[7],
            "global_hash_overflow": sums[8],
            "global_pruned": sums[9],
            "cuda_graph_captured_sum": sums[11],
            "threshold_valid_sum": sums[12],
            "threshold_q_sum": sums[13],
            "rank": cfg["rank"],
            "local_status": st,
        }
        reports.append(row)
        if cfg["rank"] == 0:
            print("STEP_SUMMARY", json.dumps(row, ensure_ascii=False), flush=True)
        if sums[7] != 0 or sums[8] != 0:
            raise AssertionError(f"overflow at depth={depth}: bucket={sums[7]} hash={sums[8]}")
        if sums[0] > 0:
            found_depth = depth
            break

    local_found = dict(engine.status())
    found_reports = gather_objects(
        {
            "rank": cfg["rank"],
            "found": int(local_found.get("local_found", local_found["found"])),
            "found_local_index": int(local_found["found_local_index"]),
        }
    )
    found_rank = -1
    found_local_index = -1
    for item in found_reports:
        if int(item["found"]) != 0:
            found_rank = int(item["rank"])
            found_local_index = int(item["found_local_index"])
            break

    path_actions: list[int] = []
    path_current: list[str] = []
    path_fullbeamnice: list[str] = []
    restore_ok = False
    if found_depth > 0 and found_rank >= 0:
        path_actions, path_current, path_fullbeamnice = reconstruct_path(engine, cfg, found_depth, found_rank, found_local_index)
        if cfg["rank"] == 0:
            path_variant, selected_path_current, restore_ok, restore_distance, restored = choose_valid_solution_path(test_state, central, path_current)
            selected_path_fullbeamnice = [fullbeamnice_name_from_current(x) for x in selected_path_current]
            if not restore_ok:
                found_state_bytes = bytes(engine.current_state_bytes(found_local_index)) if found_rank == cfg["rank"] else b""
                found_state_is_central = bool(found_state_bytes and np.array_equal(np.frombuffer(found_state_bytes, dtype=np.uint8), central))
                diagnostic = {
                    "path_variant": path_variant,
                    "restore_distance": restore_distance,
                    "found_state_is_central_on_rank0": found_state_is_central,
                    "raw_path_current_solver_order": path_current,
                    "selected_path_current_solver_order": selected_path_current,
                }
                raise AssertionError("reconstructed path does not restore central state; diagnostic=" + json.dumps(diagnostic, ensure_ascii=False))
            path_current = selected_path_current
            path_fullbeamnice = selected_path_fullbeamnice

    if cfg["rank"] == 0:
        result = {
            "test_name": "current_solver_with_fullbeamnice_model_and_static_history",
            "beam_width": int(os.environ["GLOBAL_BEAM_WIDTH"]),
            "max_depth": int(os.environ["MAX_DEPTH"]),
            "use_cuda_graphs": os.environ.get("USE_CUDA_GRAPHS", "0"),
            "fullbeamnice_reference": {"rnd_depth": 20, "rnd_seed": 42, "beam_width": 4096, "num_steps": 80, "num_attempts": 1},
            "scramble_path_fullbeamnice_order": scramble_path,
            "owner": owner,
            "final": reports[-1],
            "found": bool(reports[-1]["global_found"] > 0),
            "depth": reports[-1]["depth"],
            "found_rank": found_rank,
            "found_local_index": found_local_index,
            "path_action_indices_current_order": path_actions,
            "path_current_solver_order": path_current,
            "path_fullbeamnice_order": path_fullbeamnice,
            "path_restore_ok": restore_ok,
        }
        print("FULLBEAMNICE_CURRENT_SOLVER_RESULT", json.dumps(result, ensure_ascii=False), flush=True)

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
