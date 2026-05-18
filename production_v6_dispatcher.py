from __future__ import annotations

import csv
import json
import os
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

import beam_engine
import data_loader


STATE_LEN = 120
STATE_STORAGE_LEN = 128
STATE_VALUE_PAD = 128
MOVE_COUNT = 24
SCORE_BIN_COUNT = 76801
UINT32_MAX = 0xFFFFFFFF


@dataclass
class ProductionV6Result:
    task_id: int
    status: str
    path: str
    depth_rows: list[dict[str, Any]]
    solved_depth: int = -1
    solved_rank: int = -1
    solved_parent_idx: int = -1
    solved_move: int = -1
    raw_solved_record_exists: bool = False
    path_replay_valid: bool = False
    path_failure_reason: str = ""


def require_world2_t4_runtime() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("architecture_v6 production dispatcher requires CUDA")
    device_count = torch.cuda.device_count()
    print(f"cuda_device_count={device_count}")
    try:
        subprocess.run(["nvidia-smi", "-L"], check=False)
    except FileNotFoundError:
        print("nvidia-smi_not_found=true")
    names = [torch.cuda.get_device_name(i) for i in range(device_count)]
    for idx, name in enumerate(names):
        print(f"cuda_device_{idx}={name}")
    if device_count != 2 or not all("T4" in name for name in names):
        raise RuntimeError(f"Kaggle 2xT4 runtime required, got device_count={device_count}, names={names}")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"WORLD_SIZE=2 required, got {world_size}")
    torch.cuda.set_device(rank)
    return rank, world_size, torch.device("cuda", rank)


def make_real_generators_padded() -> np.ndarray:
    return np.frombuffer(data_loader.get_action_table128_u8(), dtype=np.uint8).reshape((MOVE_COUNT, STATE_STORAGE_LEN)).copy()


def make_central_padded() -> np.ndarray:
    return data_loader.get_central_state128_u8()


def make_zobrist() -> np.ndarray:
    rng = np.random.default_rng(0xC0DEC0DE)
    zobrist = rng.integers(0, np.iinfo(np.uint64).max, size=(STATE_STORAGE_LEN, STATE_VALUE_PAD, 2), dtype=np.uint64)
    zobrist[STATE_LEN:, :, :] = 0
    return zobrist


def owner_from_hash128(hi: int, lo: int, world_size: int) -> int:
    if world_size <= 1:
        return 0
    return int((hi ^ ((lo * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)) % world_size)


def unpack_meta(raw: bytes, idx: int) -> dict[str, int]:
    lo, hi, parent_idx, score_key, route = struct.unpack_from("<QQQII", raw, idx * 32)
    return {
        "lo": int(lo),
        "hi": int(hi),
        "parent_idx": int(parent_idx),
        "score_key": int(score_key),
        "route": int(route),
        "source_rank": int(route >> 16),
        "owner": int((route >> 8) & 0xFF),
        "move": int(route & 0xFF),
    }


def pack_final_request(parent_idx: int, target_local_idx: int, return_rank: int, move: int) -> bytes:
    return struct.pack("<QIHBB", int(parent_idx), int(target_local_idx), int(return_rank), int(move), 0)


def append_move_to_path(path: str, move: int) -> str:
    move_name = data_loader.ACTION_NAMES[int(move)]
    return move_name if not path else f"{path}.{move_name}"


def histogram_threshold(scores: list[int], keep_count: int) -> int:
    if not scores:
        return UINT32_MAX
    hist = np.bincount(np.asarray(scores, dtype=np.int64), minlength=max(max(scores) + 1, SCORE_BIN_COUNT))
    remaining = int(keep_count)
    for score_key, count in enumerate(hist.tolist()):
        remaining -= int(count)
        if remaining <= 0:
            return int(score_key)
    return UINT32_MAX


def allreduce_score_threshold(
    local_scores: list[int],
    current_threshold: int,
    threshold_initialized: bool,
    global_beam_width_effective: int,
    device: torch.device,
) -> tuple[int, bool, int]:
    gathered_scores: list[list[int] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_scores, [int(x) for x in local_scores])
    global_scores = [score for scores in gathered_scores for score in (scores or [])]
    total_survivors = len(global_scores)
    if (not threshold_initialized) and total_survivors < global_beam_width_effective:
        return UINT32_MAX, False, total_survivors
    if total_survivors >= global_beam_width_effective:
        new_threshold = histogram_threshold(global_scores, global_beam_width_effective)
        return min(int(current_threshold), int(new_threshold)), True, total_survivors
    return int(current_threshold), bool(threshold_initialized), total_survivors


class ProductionV6Dispatcher:
    def __init__(self, rank: int, world_size: int, device: torch.device, *, beam_width: int, b_micro: int) -> None:
        os.environ["INFERENCE_BACKEND"] = "fullbeamnice_static"
        os.environ["USE_CUDA_GRAPHS"] = "0"
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = device
        self.beam_width = int(beam_width)
        self.b_micro = int(b_micro)
        self.ext = beam_engine.build_extension(verbose=False)
        cfg = beam_engine.make_default_config()
        cfg.update(
            {
                "world_size": self.world_size,
                "rank": self.rank,
                "global_beam_width": self.beam_width,
                "b_micro": self.b_micro,
                "score_ring_depth": 1,
                "net_ring_depth": 1,
                "bucket_cap_per_peer": max(4096, self.b_micro * MOVE_COUNT),
                "k_expand_tile": self.b_micro * MOVE_COUNT,
                "inference_parallelism": 1,
                "stream3_batch_candidates": self.b_micro * MOVE_COUNT,
                "stream4_batch_candidates": max(2, min(self.b_micro * MOVE_COUNT, 256)),
                "stream4_batch_candidates_per_shard_unit": 2,
                "shard_count": 1,
                "max_depth": int(os.environ.get("MAX_DEPTH", "20")),
                "inference_backend": "fullbeamnice_static",
            }
        )
        self.cfg = cfg
        self.buffers = beam_engine.allocate_buffers(self.ext, cfg)
        self.engine = beam_engine.configure_engine(self.ext, cfg, self.buffers)
        self.generators_np = make_real_generators_padded()
        self.central_np = make_central_padded()
        self.zobrist_np = make_zobrist()
        self.generators_t = torch.tensor(self.generators_np.reshape(-1), dtype=torch.uint8, device=device)
        self.central_t = torch.tensor(self.central_np, dtype=torch.uint8, device=device)
        self.zobrist_t = torch.tensor(self.zobrist_np.reshape(-1).view(np.uint8), dtype=torch.uint8, device=device)

    def _run_ring_streams(self, current_frontier: np.ndarray, depth: int) -> dict[str, Any]:
        frontier_count = int(min(len(current_frontier), self.b_micro))
        if frontier_count <= 0:
            return {"frontier_count": 0, "solved_count": 0, "stop_flag": 0, "candidates": []}
        states_storage = np.zeros((self.b_micro, STATE_STORAGE_LEN), dtype=np.uint8)
        states_storage[:frontier_count] = current_frontier[:frontier_count]
        states_storage[:, STATE_LEN:] = 0
        self.buffers["beam_current"][: self.b_micro, :STATE_STORAGE_LEN].copy_(torch.tensor(states_storage, dtype=torch.uint8, device=self.device))
        self.buffers["current_active_flags"][: self.b_micro].zero_()
        self.buffers["current_active_flags"][:frontier_count].fill_(1)
        self.buffers["score_ring"][: self.b_micro * MOVE_COUNT].fill_(-1)
        self.engine.warmup_inference(1)
        torch.cuda.synchronize()

        current_frontier_states = torch.tensor(states_storage.reshape(-1), dtype=torch.uint8, device=self.device)
        parent_base = torch.tensor([0], dtype=torch.int64, device=self.device)
        count = torch.tensor([frontier_count], dtype=torch.int32, device=self.device)
        hash_ring = torch.zeros((self.b_micro * MOVE_COUNT * 16,), dtype=torch.uint8, device=self.device)
        solved_capacity = 16
        solved_flag = torch.zeros((1,), dtype=torch.int32, device=self.device)
        stop_flag = torch.zeros((1,), dtype=torch.int32, device=self.device)
        solved_count = torch.zeros((1,), dtype=torch.int32, device=self.device)
        solved_overflow = torch.zeros((1,), dtype=torch.int32, device=self.device)
        solved_meta_list = torch.zeros((solved_capacity * 32,), dtype=torch.uint8, device=self.device)
        solved_depth_list = torch.zeros((solved_capacity,), dtype=torch.int32, device=self.device)
        score_ring = self.buffers["score_ring"][: self.b_micro * MOVE_COUNT]
        self.ext.v6_stream2_hash_goal(
            current_frontier_states,
            parent_base,
            count,
            score_ring,
            hash_ring,
            self.generators_t,
            self.central_t,
            self.zobrist_t,
            solved_flag,
            stop_flag,
            solved_count,
            solved_overflow,
            solved_meta_list,
            solved_depth_list,
            solved_capacity,
            int(depth),
            self.rank,
            0,
            0,
            1,
            self.b_micro,
        )
        torch.cuda.synchronize()
        solved_meta = None
        if int(solved_count.cpu()[0]) > 0:
            solved_meta = unpack_meta(solved_meta_list.cpu().numpy().tobytes(), 0)
        return {
            "frontier_count": frontier_count,
            "states_storage": states_storage,
            "current_frontier_states": current_frontier_states,
            "parent_base": parent_base,
            "count": count,
            "score_ring": score_ring,
            "hash_ring": hash_ring,
            "solved_flag": int(solved_flag.cpu()[0]),
            "stop_flag": int(stop_flag.cpu()[0]),
            "solved_count": int(solved_count.cpu()[0]),
            "solved_overflow": int(solved_overflow.cpu()[0]),
            "solved_meta": solved_meta,
        }

    def _run_stream3(self, stream12: dict[str, Any], current_threshold: int) -> dict[str, Any]:
        batch_candidates = self.b_micro * MOVE_COUNT
        key_a = torch.zeros((batch_candidates * 16,), dtype=torch.uint8, device=self.device)
        key_b = torch.zeros_like(key_a)
        val_a = torch.zeros((batch_candidates,), dtype=torch.int64, device=self.device)
        val_b = torch.zeros_like(val_a)
        compact_count = torch.zeros((1,), dtype=torch.int32, device=self.device)
        self.ext.v6_stream3_pack_threshold_compact(
            stream12["score_ring"],
            stream12["hash_ring"],
            stream12["parent_base"],
            stream12["count"],
            key_a,
            val_a,
            compact_count,
            beam_engine._v6_validate_u32("current_threshold", current_threshold),
            0,
            1,
            self.b_micro,
            batch_candidates,
        )
        torch.cuda.synchronize()
        compact_n = int(compact_count.cpu()[0])
        if compact_n == 0:
            return {"unique_count": 0, "local_count": 0, "remote_count": 0}
        temp = torch.empty((int(self.ext.v6_stream3_sort_temp_bytes(compact_n)),), dtype=torch.uint8, device=self.device)
        self.ext.v6_stream3_sort_pairs(temp, key_a, key_b, val_a, val_b, compact_n)
        unique_key = torch.zeros((batch_candidates * 16,), dtype=torch.uint8, device=self.device)
        unique_val = torch.zeros((batch_candidates,), dtype=torch.int64, device=self.device)
        unique_count = torch.zeros((1,), dtype=torch.int32, device=self.device)
        self.ext.v6_stream3_dedup_sorted(key_b, val_b, unique_key, unique_val, unique_count, compact_n)
        torch.cuda.synchronize()
        unique_n = int(unique_count.cpu()[0])
        local_pending = torch.zeros((max(unique_n, 1) * 32,), dtype=torch.uint8, device=self.device)
        remote_send = torch.zeros((max(unique_n, 1) * 32,), dtype=torch.uint8, device=self.device)
        local_count = torch.zeros((1,), dtype=torch.int32, device=self.device)
        send_count = torch.zeros((self.world_size,), dtype=torch.int32, device=self.device)
        if self.rank == 0:
            send_offset_values = [0, 0, unique_n]
        else:
            send_offset_values = [0, unique_n, unique_n]
        send_offset = torch.tensor(send_offset_values, dtype=torch.int32, device=self.device)
        self.ext.v6_stream3_restore_split(
            unique_key,
            unique_val,
            stream12["parent_base"],
            local_pending,
            remote_send,
            local_count,
            send_count,
            send_offset,
            unique_n,
            self.rank,
            self.world_size,
            0,
            1,
            self.b_micro,
        )
        torch.cuda.synchronize()
        return {
            "unique_count": unique_n,
            "local_pending": local_pending,
            "remote_send": remote_send,
            "local_count": int(local_count.cpu()[0]),
            "send_count": send_count,
            "send_offset": send_offset,
        }

    def _run_stream5(self, stream3: dict[str, Any]) -> dict[str, Any]:
        remote_capacity = max(int(stream3.get("unique_count", 0)), 1)
        remote_recv = torch.zeros((remote_capacity * 32,), dtype=torch.uint8, device=self.device)
        recv_count = torch.zeros((self.world_size,), dtype=torch.int32, device=self.device)
        recv_offset = torch.zeros((self.world_size + 1,), dtype=torch.int32, device=self.device)
        self.engine.v6_stream5_exchange_candidate_meta(
            stream3["remote_send"],
            remote_recv,
            stream3["send_count"],
            stream3["send_offset"],
            recv_count,
            recv_offset,
        )
        torch.cuda.synchronize()
        return {
            "remote_recv": remote_recv,
            "recv_count": recv_count,
            "recv_offset": recv_offset,
            "remote_recv_count": int(recv_count.cpu().numpy().sum()),
        }

    def _collector_to_stream4(self, stream3: dict[str, Any], stream5: dict[str, Any], current_threshold: int) -> dict[str, Any]:
        local_n = int(stream3.get("local_count", 0))
        remote_n = int(stream5.get("remote_recv_count", 0))
        input_n = local_n + remote_n
        if input_n == 0:
            return {"clean": [], "clean_count": 0, "dirty_count": 0}
        survivor = torch.zeros((max(input_n, 1) * 32,), dtype=torch.uint8, device=self.device)
        if local_n:
            survivor[: local_n * 32].copy_(stream3["local_pending"][: local_n * 32])
        if remote_n:
            survivor[local_n * 32 : (local_n + remote_n) * 32].copy_(stream5["remote_recv"][: remote_n * 32])
        key_a = torch.zeros((input_n * 16,), dtype=torch.uint8, device=self.device)
        key_b = torch.zeros_like(key_a)
        val_a = torch.zeros((input_n * 32,), dtype=torch.uint8, device=self.device)
        val_b = torch.zeros_like(val_a)
        compact_count = torch.zeros((1,), dtype=torch.int32, device=self.device)
        self.ext.v6_stream4_threshold_compact(survivor, key_a, val_a, compact_count, input_n, current_threshold)
        torch.cuda.synchronize()
        compact_n = int(compact_count.cpu()[0])
        if compact_n:
            temp = torch.empty((int(self.ext.v6_stream4_sort_temp_bytes(compact_n)),), dtype=torch.uint8, device=self.device)
            self.ext.v6_stream4_sort_pairs(temp, key_a, key_b, val_a, val_b, compact_n)
            clean_tmp = torch.zeros((input_n * 32,), dtype=torch.uint8, device=self.device)
            new_clean_count = torch.zeros((1,), dtype=torch.int32, device=self.device)
            self.ext.v6_stream4_dedup_sorted(key_b, val_b, clean_tmp, new_clean_count, compact_n)
            torch.cuda.synchronize()
            clean_n = int(new_clean_count.cpu()[0])
        else:
            clean_tmp = torch.zeros((input_n * 32,), dtype=torch.uint8, device=self.device)
            new_clean_count = torch.zeros((1,), dtype=torch.int32, device=self.device)
            clean_n = 0
        clean_count = torch.tensor([0], dtype=torch.int32, device=self.device)
        dirty_count = torch.tensor([input_n], dtype=torch.int32, device=self.device)
        processing_flag = torch.tensor([1], dtype=torch.uint8, device=self.device)
        self.ext.v6_stream4_write_clean(survivor, clean_tmp, clean_count, dirty_count, processing_flag, clean_n)
        torch.cuda.synchronize()
        raw = survivor.cpu().numpy().tobytes()
        clean = [unpack_meta(raw, i) for i in range(clean_n)]
        return {
            "survivor": survivor,
            "clean": clean,
            "clean_count": clean_n,
            "dirty_count": int(dirty_count.cpu()[0]),
            "processing_flag": int(processing_flag.cpu()[0]),
        }

    def _final_materialize(
        self,
        current_frontier_states: torch.Tensor,
        clean: list[dict[str, int]],
        current_threshold: int,
        current_paths: list[str] | None = None,
    ) -> np.ndarray | tuple[np.ndarray, list[str]]:
        keep = [c for c in clean if int(c["score_key"]) <= int(current_threshold)]
        keep_counts = [None for _ in range(self.world_size)]
        dist.all_gather_object(keep_counts, len(keep))
        counts = [int(x or 0) for x in keep_counts]
        prefix = [0, counts[0]]
        global_keep = sum(counts)
        if global_keep == 0:
            empty = np.zeros((0, STATE_STORAGE_LEN), dtype=np.uint8)
            return (empty, []) if current_paths is not None else empty
        gathered_paths: list[list[str] | None] | None = None
        if current_paths is not None:
            gathered_paths = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered_paths, list(current_paths))
        request_by_peer = [[] for _ in range(self.world_size)]
        path_by_target_local_idx: dict[int, str] = {}
        expected_local = 0
        for local_idx, candidate in enumerate(keep):
            global_idx = prefix[self.rank] + local_idx
            target_rank = min(global_idx * self.world_size // max(global_keep, 1), self.world_size - 1)
            target_start = (global_keep * target_rank + self.world_size - 1) // self.world_size
            target_local_idx = int(global_idx - target_start)
            expected_local += 1 if target_rank == self.rank else 0
            source_rank = int(candidate["source_rank"])
            move = int(candidate["move"])
            request_by_peer[source_rank].append(pack_final_request(candidate["parent_idx"], target_local_idx, target_rank, move))
            if gathered_paths is not None and target_rank == self.rank:
                source_paths = gathered_paths[source_rank] or []
                parent_idx = int(candidate["parent_idx"])
                parent_path = source_paths[parent_idx] if 0 <= parent_idx < len(source_paths) else ""
                path_by_target_local_idx[target_local_idx] = append_move_to_path(parent_path, move)
        send_request_counts = [len(x) for x in request_by_peer]
        recv_request_counts = [0 for _ in range(self.world_size)]
        send_count_t = torch.tensor(send_request_counts, dtype=torch.int64, device=self.device)
        recv_count_t = torch.empty_like(send_count_t)
        dist.all_to_all_single(recv_count_t, send_count_t)
        recv_request_counts = [int(x) for x in recv_count_t.cpu().tolist()]
        send_request_bytes = b"".join(b"".join(items) for items in request_by_peer)
        recv_request_total = sum(recv_request_counts)
        send_request_t = torch.tensor(np.frombuffer(send_request_bytes, dtype=np.uint8).copy(), dtype=torch.uint8, device=self.device)
        if send_request_t.numel() == 0:
            send_request_t = torch.empty((0,), dtype=torch.uint8, device=self.device)
        recv_request_t = torch.empty((recv_request_total * 16,), dtype=torch.uint8, device=self.device)
        dist.all_to_all_single(
            recv_request_t,
            send_request_t,
            output_split_sizes=[c * 16 for c in recv_request_counts],
            input_split_sizes=[c * 16 for c in send_request_counts],
        )
        final_response_t = torch.zeros((max(recv_request_total, 1) * STATE_STORAGE_LEN,), dtype=torch.uint8, device=self.device)
        if recv_request_total:
            self.ext.v6_final_materialize(current_frontier_states, recv_request_t, self.generators_t, final_response_t, recv_request_total)
            torch.cuda.synchronize()
        recv_request_raw = recv_request_t.cpu().numpy().tobytes()
        responses = final_response_t.cpu().numpy().reshape((-1, STATE_STORAGE_LEN))[:recv_request_total].copy()
        response_by_peer = [[] for _ in range(self.world_size)]
        for idx in range(recv_request_total):
            _parent_idx, _target_local_idx, return_rank, _move, _pad = struct.unpack_from("<QIHBB", recv_request_raw, idx * 16)
            response_by_peer[int(return_rank)].append(responses[idx].tobytes())
        send_response_counts = [len(x) for x in response_by_peer]
        send_response_count_t = torch.tensor(send_response_counts, dtype=torch.int64, device=self.device)
        recv_response_count_t = torch.empty_like(send_response_count_t)
        dist.all_to_all_single(recv_response_count_t, send_response_count_t)
        recv_response_counts = [int(x) for x in recv_response_count_t.cpu().tolist()]
        send_response_bytes = b"".join(b"".join(items) for items in response_by_peer)
        send_response_t = torch.tensor(np.frombuffer(send_response_bytes, dtype=np.uint8).copy(), dtype=torch.uint8, device=self.device)
        if send_response_t.numel() == 0:
            send_response_t = torch.empty((0,), dtype=torch.uint8, device=self.device)
        recv_response_total = sum(recv_response_counts)
        recv_response_t = torch.empty((recv_response_total * STATE_STORAGE_LEN,), dtype=torch.uint8, device=self.device)
        dist.all_to_all_single(
            recv_response_t,
            send_response_t,
            output_split_sizes=[c * STATE_STORAGE_LEN for c in recv_response_counts],
            input_split_sizes=[c * STATE_STORAGE_LEN for c in send_response_counts],
        )
        next_frontier = torch.zeros((max(recv_response_total, 1) * STATE_STORAGE_LEN,), dtype=torch.uint8, device=self.device)
        if recv_response_total:
            self.ext.v6_final_scatter_responses(recv_response_t, next_frontier, recv_response_total)
            torch.cuda.synchronize()
        next_states = next_frontier.cpu().numpy().reshape((-1, STATE_STORAGE_LEN))[:recv_response_total].copy()
        if current_paths is None:
            return next_states
        next_paths = [path_by_target_local_idx.get(i, "") for i in range(recv_response_total)]
        return next_states, next_paths

    def run_task(self, task_id: int, state: np.ndarray, max_depth: int, global_beam_width_effective: int) -> ProductionV6Result:
        current = np.zeros((1, STATE_STORAGE_LEN), dtype=np.uint8)
        current[0] = data_loader.pad_state128_u8(state)
        current_paths = [""]
        current_threshold = UINT32_MAX
        threshold_initialized = False
        depth_rows: list[dict[str, Any]] = []
        status = "max_depth_reached"
        solved_path = ""
        solved_depth = -1
        solved_parent_idx = -1
        solved_move = -1
        raw_solved_record_exists = False
        path_replay_valid = False
        path_failure_reason = "no_solved_state"
        for depth in range(int(max_depth)):
            start = time.time()
            stream12 = self._run_ring_streams(current, depth)
            if int(stream12["solved_count"]) > 0:
                status = "solved"
                solved_meta = stream12.get("solved_meta")
                solved_depth = int(depth)
                raw_solved_record_exists = solved_meta is not None
                if solved_meta is not None:
                    parent_idx = int(solved_meta["parent_idx"])
                    solved_parent_idx = parent_idx
                    solved_move = int(solved_meta["move"])
                    parent_path = current_paths[parent_idx] if 0 <= parent_idx < len(current_paths) else ""
                    if 0 <= parent_idx < len(current_paths):
                        solved_path = append_move_to_path(parent_path, int(solved_meta["move"]))
                        path_replay_valid, _path_len, path_failure_reason = replay_path_to_central(state, solved_path)
                    else:
                        path_failure_reason = "parent_chain_broken"
                else:
                    path_failure_reason = "solved_state_but_no_parent_chain"
                depth_rows.append(self._depth_row(task_id, depth, len(current), threshold_initialized, current_threshold, 0, 0, 0, int(stream12["solved_count"]), 1, start))
                break
            stream3 = self._run_stream3(stream12, current_threshold)
            stream5 = self._run_stream5(stream3) if int(stream3.get("unique_count", 0)) else {"remote_recv_count": 0, "remote_recv": torch.empty((0,), dtype=torch.uint8, device=self.device)}
            stream4 = self._collector_to_stream4(stream3, stream5, current_threshold) if int(stream3.get("unique_count", 0)) else {"clean": [], "clean_count": 0, "dirty_count": 0}
            local_scores = [int(c["score_key"]) for c in stream4["clean"]]
            current_threshold, threshold_initialized, total_survivors = allreduce_score_threshold(
                local_scores,
                current_threshold,
                threshold_initialized,
                global_beam_width_effective,
                self.device,
            )
            next_current, next_paths = self._final_materialize(
                stream12["current_frontier_states"],
                list(stream4["clean"]),
                current_threshold,
                current_paths,
            )
            global_keep_tensor = torch.tensor([len(next_current)], dtype=torch.int64, device=self.device)
            dist.all_reduce(global_keep_tensor, op=dist.ReduceOp.SUM)
            global_keep = int(global_keep_tensor.cpu()[0])
            depth_rows.append(
                self._depth_row(
                    task_id,
                    depth,
                    len(current),
                    threshold_initialized,
                    current_threshold,
                    int(stream4["clean_count"]),
                    int(stream4["dirty_count"]),
                    global_keep,
                    int(stream12["solved_count"]),
                    int(stream12["stop_flag"]),
                    start,
                )
            )
            current = next_current
            current_paths = next_paths
            if global_keep == 0 or len(current) == 0:
                status = "unsolved"
                break
        return ProductionV6Result(
            task_id=int(task_id),
            status=status,
            path=solved_path,
            depth_rows=depth_rows,
            solved_depth=solved_depth,
            solved_rank=self.rank if raw_solved_record_exists else -1,
            solved_parent_idx=solved_parent_idx,
            solved_move=solved_move,
            raw_solved_record_exists=raw_solved_record_exists,
            path_replay_valid=path_replay_valid,
            path_failure_reason=path_failure_reason,
        )

    def _depth_row(
        self,
        task_id: int,
        depth: int,
        frontier_size: int,
        threshold_initialized: bool,
        current_threshold: int,
        clean_count_total: int,
        dirty_count_total: int,
        global_keep_count: int,
        solved_count: int,
        stop_flag: int,
        start: float,
    ) -> dict[str, Any]:
        return {
            "task_id": int(task_id),
            "depth": int(depth),
            "frontier_size": int(frontier_size),
            "threshold_initialized": bool(threshold_initialized),
            "current_threshold": int(current_threshold),
            "clean_count_total": int(clean_count_total),
            "dirty_count_total": int(dirty_count_total),
            "global_keep_count": int(global_keep_count),
            "solved_count": int(solved_count),
            "stop_flag": int(stop_flag),
            "elapsed_sec": float(time.time() - start),
            "gpu_memory_allocated": int(torch.cuda.memory_allocated(self.device)),
            "gpu_memory_reserved": int(torch.cuda.memory_reserved(self.device)),
        }


def run_real_data_production_v6_world2(
    *,
    task_count: int,
    max_depth: int,
    beam_width: int,
    output_path: Path,
    stats_path: Path,
    b_micro: int = 4,
) -> dict[str, Any]:
    rank, world_size, device = require_world2_t4_runtime()
    dispatcher = ProductionV6Dispatcher(rank, world_size, device, beam_width=beam_width, b_micro=b_micro)
    puzzles = data_loader.load_test_puzzles(max_puzzles=task_count)
    if len(puzzles) != int(task_count):
        raise RuntimeError(f"expected task_count={task_count}, got {len(puzzles)}")
    output_rows: list[dict[str, str]] = []
    status_counts = {"solved": 0, "unsolved": 0, "max_depth_reached": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["initial_state_id", "path", "status"])
            writer.writeheader()
        stats_path.write_text("", encoding="utf-8")
    dist.barrier()
    global_beam_width_effective = int(beam_width)
    for index, (task_id, state) in enumerate(puzzles):
        result = dispatcher.run_task(int(task_id), state, max_depth, global_beam_width_effective)
        status_counts[result.status] += 1
        output_rows.append({"initial_state_id": str(result.task_id), "path": result.path, "status": result.status})
        if rank == 0:
            with output_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["initial_state_id", "path", "status"])
                writer.writerow(output_rows[-1])
        with stats_path.open("a", encoding="utf-8") as f:
            for row in result.depth_rows:
                record = dict(row)
                record["rank"] = rank
                record["world_size"] = world_size
                f.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            "PRODUCTION_V6_REAL_DATA_TASK_STATUS "
            f"rank={rank} index={index} task_id={task_id} status={result.status} "
            f"depth_rows={len(result.depth_rows)} beam={beam_width} max_depth={max_depth}"
        )
    local_counts = torch.tensor(
        [len(output_rows), status_counts["solved"], status_counts["unsolved"], status_counts["max_depth_reached"]],
        dtype=torch.int64,
        device=device,
    )
    gathered = [torch.zeros_like(local_counts) for _ in range(world_size)]
    dist.all_gather(gathered, local_counts)
    dist.barrier()
    if rank == 0:
        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise AssertionError("production v6 output CSV missing")
        if not stats_path.exists() or stats_path.stat().st_size <= 0:
            raise AssertionError("production v6 stats JSONL missing")
    return {
        "rank": rank,
        "world_size": world_size,
        "task_count": len(output_rows),
        "status_counts": status_counts,
        "gathered_counts": [[int(x) for x in t.cpu().tolist()] for t in gathered],
        "output_path": str(output_path),
        "stats_path": str(stats_path),
        "production_v6_dispatcher_path": True,
        "legacy_next_state_pool_path": False,
        "prefilled_score_ring_fake_path": False,
        "runtime_120_slice": False,
        "fallback_backend": False,
    }


def run_real_data_production_v6_world2_detailed(
    *,
    task_count: int,
    max_depth: int,
    beam_width: int,
    output_path: Path,
    stats_path: Path,
    b_micro: int = 4,
) -> dict[str, Any]:
    rank, world_size, device = require_world2_t4_runtime()
    dispatcher = ProductionV6Dispatcher(rank, world_size, device, beam_width=beam_width, b_micro=b_micro)
    puzzles = data_loader.load_test_puzzles(max_puzzles=task_count)
    if len(puzzles) != int(task_count):
        raise RuntimeError(f"expected task_count={task_count}, got {len(puzzles)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_id", "initial_state_id", "status", "found", "depth_reached", "solution_len", "path", "rank", "error_or_note"]
    if rank == 0:
        with output_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        stats_path.write_text("", encoding="utf-8")
    dist.barrier()
    status_counts = {"solved": 0, "unsolved": 0, "max_depth_reached": 0, "error": 0}
    local_rows: list[dict[str, Any]] = []
    for row_id, (task_id, state) in enumerate(puzzles):
        task_start = time.time()
        try:
            result = dispatcher.run_task(int(task_id), state, max_depth, int(beam_width))
            depth_reached = len(result.depth_rows)
            found = result.status == "solved"
            solution_len = len(result.path.split(".")) if result.path else 0
            status = result.status
            note = ""
            status_counts[status] += 1
            for depth_row in result.depth_rows:
                record = dict(depth_row)
                record["rank"] = rank
                record["world_size"] = world_size
                record["task_idx"] = row_id
                record["found"] = found
                record["status"] = status
                with stats_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as exc:  # noqa: BLE001 - stress runner must record per-task failure.
            depth_reached = 0
            found = False
            solution_len = 0
            status = "error"
            note = f"{type(exc).__name__}: {exc}"
            status_counts["error"] += 1
        row = {
            "row_id": int(row_id),
            "initial_state_id": int(task_id),
            "status": status,
            "found": int(found),
            "depth_reached": int(depth_reached),
            "solution_len": int(solution_len),
            "path": result.path if status != "error" else "",
            "rank": int(rank),
            "error_or_note": note,
        }
        local_rows.append(row)
        if rank == 0:
            with output_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        print(
            "REAL_DATA_100SAMPLES_PROGRESS "
            f"task_idx={row_id} rank={rank} status={status} depth_reached={depth_reached} "
            f"found={int(found)} elapsed_sec={time.time() - task_start:.6f} "
            f"gpu_memory_allocated={torch.cuda.memory_allocated(device)} gpu_memory_reserved={torch.cuda.memory_reserved(device)}"
        )
    local_counts = torch.tensor(
        [
            len(local_rows),
            status_counts["solved"],
            status_counts["unsolved"],
            status_counts["max_depth_reached"],
            status_counts["error"],
        ],
        dtype=torch.int64,
        device=device,
    )
    gathered = [torch.zeros_like(local_counts) for _ in range(world_size)]
    dist.all_gather(gathered, local_counts)
    dist.barrier()
    if rank == 0:
        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise AssertionError("real-data 100 output CSV missing")
        with output_path.open("r", encoding="utf-8") as f:
            output_rows = max(sum(1 for _ in f) - 1, 0)
        if output_rows != int(task_count):
            raise AssertionError(f"output row count mismatch: {output_rows} != {task_count}")
        if not stats_path.exists() or stats_path.stat().st_size <= 0:
            raise AssertionError("real-data 100 stats JSONL missing")
    return {
        "rank": rank,
        "world_size": world_size,
        "task_count": len(local_rows),
        "status_counts": status_counts,
        "gathered_counts": [[int(x) for x in item.cpu().tolist()] for item in gathered],
        "output_path": str(output_path),
        "stats_path": str(stats_path),
        "production_v6_dispatcher_path": True,
        "legacy_next_state_pool_path": False,
        "prefilled_score_ring_fake_path": False,
        "runtime_120_slice": False,
        "fallback_backend": False,
    }


def run_real_data_path_audit_world2(
    *,
    task_count: int,
    max_depth: int,
    beam_width: int,
    output_path: Path,
    audit_path: Path,
    b_micro: int = 4,
) -> dict[str, Any]:
    rank, world_size, device = require_world2_t4_runtime()
    dispatcher = ProductionV6Dispatcher(rank, world_size, device, beam_width=beam_width, b_micro=b_micro)
    puzzles = data_loader.load_test_puzzles(max_puzzles=task_count)
    if len(puzzles) != int(task_count):
        raise RuntimeError(f"expected task_count={task_count}, got {len(puzzles)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_id",
        "initial_state_id",
        "status",
        "found_flag",
        "solved_depth",
        "solved_rank",
        "solved_parent_idx",
        "solved_move",
        "raw_solved_record_exists",
        "reconstructed_path_exists",
        "path_replay_valid",
        "solution_len",
        "path",
        "failure_reason",
        "rank",
    ]
    if rank == 0:
        with output_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        audit_path.write_text("", encoding="utf-8")
    dist.barrier()
    status_counts = {"solved": 0, "unsolved": 0, "max_depth_reached": 0, "error": 0}
    failure_counts: dict[str, int] = {}
    local_rows: list[dict[str, Any]] = []
    for row_id, (task_id, state) in enumerate(puzzles):
        task_start = time.time()
        try:
            result = dispatcher.run_task(int(task_id), state, max_depth, int(beam_width))
            status = result.status
            found = status == "solved"
            solution_len = len(result.path.split(".")) if result.path else 0
            failure_reason = ""
            if found and not result.raw_solved_record_exists:
                failure_reason = "solved_state_but_no_parent_chain"
            elif found and not result.path:
                failure_reason = "output_writer_empty_path"
            elif found and not result.path_replay_valid:
                failure_reason = result.path_failure_reason or "replay_failed"
            elif not found:
                failure_reason = "no_solved_state"
            else:
                ok, replay_len, note = replay_path_to_central(state, result.path)
                if replay_len != solution_len:
                    failure_reason = "solution_len_mismatch"
                elif not ok:
                    failure_reason = note or "replay_failed"
                else:
                    failure_reason = "ok"
            status_counts[status] += 1
        except Exception as exc:  # noqa: BLE001 - diagnostic runner records per-task failure.
            result = ProductionV6Result(
                task_id=int(task_id),
                status="error",
                path="",
                depth_rows=[],
                path_failure_reason=f"{type(exc).__name__}: {exc}",
            )
            status = "error"
            found = False
            solution_len = 0
            failure_reason = result.path_failure_reason
            status_counts["error"] += 1
        failure_counts[failure_reason] = failure_counts.get(failure_reason, 0) + 1
        row = {
            "row_id": int(row_id),
            "initial_state_id": int(task_id),
            "status": status,
            "found_flag": int(found),
            "solved_depth": int(result.solved_depth),
            "solved_rank": int(result.solved_rank),
            "solved_parent_idx": int(result.solved_parent_idx),
            "solved_move": int(result.solved_move),
            "raw_solved_record_exists": int(result.raw_solved_record_exists),
            "reconstructed_path_exists": int(bool(result.path)),
            "path_replay_valid": int(bool(result.path_replay_valid)),
            "solution_len": int(solution_len),
            "path": result.path,
            "failure_reason": failure_reason,
            "rank": int(rank),
        }
        local_rows.append(row)
        if rank == 0:
            with output_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        with audit_path.open("a", encoding="utf-8") as f:
            audit_record = dict(row)
            audit_record["elapsed_sec"] = float(time.time() - task_start)
            audit_record["gpu_memory_allocated"] = int(torch.cuda.memory_allocated(device))
            audit_record["gpu_memory_reserved"] = int(torch.cuda.memory_reserved(device))
            f.write(json.dumps(audit_record, sort_keys=True) + "\n")
        print(
            "REAL_DATA_PATH_AUDIT_PROGRESS "
            f"task_idx={row_id} rank={rank} status={status} found={int(found)} "
            f"raw_solved_record_exists={int(result.raw_solved_record_exists)} "
            f"path_replay_valid={int(result.path_replay_valid)} failure_reason={failure_reason} "
            f"elapsed_sec={time.time() - task_start:.6f}"
        )
    local_counts = torch.tensor(
        [
            len(local_rows),
            status_counts["solved"],
            status_counts["unsolved"],
            status_counts["max_depth_reached"],
            status_counts["error"],
        ],
        dtype=torch.int64,
        device=device,
    )
    gathered = [torch.zeros_like(local_counts) for _ in range(world_size)]
    dist.all_gather(gathered, local_counts)
    dist.barrier()
    if rank == 0:
        with output_path.open("r", encoding="utf-8") as f:
            output_rows = max(sum(1 for _ in f) - 1, 0)
        if output_rows != int(task_count):
            raise AssertionError(f"path audit output row count mismatch: {output_rows} != {task_count}")
        if not audit_path.exists() or audit_path.stat().st_size <= 0:
            raise AssertionError("path audit JSONL missing")
    return {
        "rank": rank,
        "world_size": world_size,
        "task_count": len(local_rows),
        "status_counts": status_counts,
        "failure_counts": failure_counts,
        "gathered_counts": [[int(x) for x in item.cpu().tolist()] for item in gathered],
        "output_path": str(output_path),
        "audit_path": str(audit_path),
        "production_v6_dispatcher_path": True,
        "legacy_next_state_pool_path": False,
        "prefilled_score_ring_fake_path": False,
        "runtime_120_slice": False,
        "fallback_backend": False,
    }


def replay_path_to_central(state: np.ndarray, path: str) -> tuple[bool, int, str]:
    if not path:
        return False, 0, "empty_path"
    moves = path.split(".")
    action_set = set(data_loader.ACTION_NAMES)
    invalid = [move for move in moves if move not in action_set]
    if invalid:
        return False, len(moves), f"invalid_moves={invalid[:4]}"
    replayed = data_loader.apply_actions_cpu(np.asarray(state, dtype=np.uint8), moves)
    central = data_loader.get_central_state_u8()
    if not np.array_equal(replayed, central):
        return False, len(moves), "replay_not_central"
    return True, len(moves), ""


def validate_output_paths(
    *,
    output_path: Path,
    task_count: int,
) -> dict[str, Any]:
    puzzles = dict(data_loader.load_test_puzzles(max_puzzles=task_count))
    solved_rows = 0
    validated_rows = 0
    errors: list[str] = []
    with output_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "solved":
                continue
            solved_rows += 1
            task_id = int(row["initial_state_id"])
            path = row.get("path", "")
            solution_len = int(row.get("solution_len", "0") or 0)
            if task_id not in puzzles:
                errors.append(f"task_id_missing={task_id}")
                continue
            ok, replay_len, note = replay_path_to_central(puzzles[task_id], path)
            if solution_len != replay_len:
                errors.append(f"task_id={task_id}:solution_len_mismatch:{solution_len}!={replay_len}")
                continue
            if not ok:
                errors.append(f"task_id={task_id}:{note}")
                continue
            validated_rows += 1
    return {
        "solved_rows": solved_rows,
        "validated_rows": validated_rows,
        "errors": errors,
        "path_replay_valid": solved_rows == validated_rows and not errors,
    }
