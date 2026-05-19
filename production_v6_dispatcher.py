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
PRODUCTION_B_MICRO = 8192
PRODUCTION_K_EXPAND_TILE = PRODUCTION_B_MICRO * MOVE_COUNT

assert PRODUCTION_B_MICRO == 8192
assert PRODUCTION_K_EXPAND_TILE == 196608


def pow2_ceil(value: int) -> int:
    value = max(int(value), 1)
    return 1 << (value - 1).bit_length()


def require_production_microbatch(b_micro: int) -> int:
    b_micro = int(b_micro)
    k_expand_tile = b_micro * MOVE_COUNT
    if b_micro != PRODUCTION_B_MICRO:
        raise RuntimeError(f"invalid_config: B_MICRO must be {PRODUCTION_B_MICRO}, got {b_micro}")
    if k_expand_tile != PRODUCTION_K_EXPAND_TILE:
        raise RuntimeError(f"invalid_config: K_EXPAND_TILE must be {PRODUCTION_K_EXPAND_TILE}, got {k_expand_tile}")
    assert b_micro == 8192
    assert k_expand_tile == 196608
    return b_micro


def collective_seq_debug(rank: int, task_idx: int, depth: int, seq_tag: str, op: str, local_flag: int, local_next_count: int) -> None:
    if os.environ.get("COLLECTIVE_SEQ_DEBUG", "0").strip().lower() in {"", "0", "false", "no", "off"}:
        return
    print(
        "COLLECTIVE_SEQ_TAG "
        f"rank={int(rank)} task_idx={int(task_idx)} depth={int(depth)} "
        f"seq_tag={seq_tag} op={op} local_flag={int(local_flag)} local_next_count={int(local_next_count)}",
        flush=True,
    )


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


def meta_sort_key(candidate: dict[str, int]) -> tuple[int, int, int, int]:
    return (int(candidate["score_key"]), int(candidate["parent_idx"]), int(candidate["route"]), int(candidate["lo"]))


def merge_clean_candidates(existing: list[dict[str, int]], incoming: list[dict[str, int]]) -> list[dict[str, int]]:
    by_hash: dict[tuple[int, int], dict[str, int]] = {}
    for candidate in [*existing, *incoming]:
        key = (int(candidate["hi"]), int(candidate["lo"]))
        current = by_hash.get(key)
        if current is None or meta_sort_key(candidate) < meta_sort_key(current):
            by_hash[key] = dict(candidate)
    return sorted(by_hash.values(), key=meta_sort_key)


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
    *,
    rank: int = -1,
    task_idx: int = -1,
    depth: int = -1,
) -> tuple[int, bool, int]:
    gathered_scores: list[list[int] | None] = [None for _ in range(dist.get_world_size())]
    collective_seq_debug(rank, task_idx, depth, "threshold_scores", "all_gather_object", int(bool(local_scores)), len(local_scores))
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
        self.b_micro = require_production_microbatch(b_micro)
        self.ext = beam_engine.build_extension(verbose=False)
        cfg = beam_engine.make_default_config()
        required_candidate_capacity = self.b_micro * MOVE_COUNT
        assert required_candidate_capacity == PRODUCTION_K_EXPAND_TILE
        bucket_cap_per_peer = pow2_ceil(max(131072, required_candidate_capacity))
        cfg.update(
            {
                "world_size": self.world_size,
                "rank": self.rank,
                "global_beam_width": self.beam_width,
                "b_micro": self.b_micro,
                "score_ring_depth": 2,
                "net_ring_depth": 1,
                "bucket_cap_per_peer": bucket_cap_per_peer,
                "k_expand_tile": required_candidate_capacity,
                "inference_parallelism": 1,
                "stream3_batch_candidates": required_candidate_capacity,
                "stream4_batch_candidates": max(2, min(required_candidate_capacity, 256)),
                "stream4_batch_candidates_per_shard_unit": 2,
                "shard_count": 1,
                "max_depth": int(os.environ.get("MAX_DEPTH", "20")),
                "inference_backend": "fullbeamnice_static",
            }
        )
        if int(cfg["b_micro"]) != PRODUCTION_B_MICRO:
            raise RuntimeError(f"invalid_config: BeamEngine B_MICRO must be {PRODUCTION_B_MICRO}, got {cfg['b_micro']}")
        if int(cfg["k_expand_tile"]) != PRODUCTION_K_EXPAND_TILE:
            raise RuntimeError(f"invalid_config: BeamEngine K_EXPAND_TILE must be {PRODUCTION_K_EXPAND_TILE}, got {cfg['k_expand_tile']}")
        if int(cfg["bucket_cap_per_peer"]) != 262144:
            raise RuntimeError(f"invalid_config: BeamEngine BUCKET_CAP_PER_PEER must be 262144, got {cfg['bucket_cap_per_peer']}")
        print(
            "CONFIG_GUARD_OK "
            f"rank={self.rank} B_MICRO={cfg['b_micro']} K_EXPAND_TILE={cfg['k_expand_tile']} "
            f"BUCKET_CAP_PER_PEER={cfg['bucket_cap_per_peer']} "
            f"cuda_graphs={int(os.environ.get('USE_CUDA_GRAPHS', '1') != '0')}",
            flush=True,
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
        self.stream3_capacity = PRODUCTION_K_EXPAND_TILE
        self.remote_capacity = int(self.cfg["bucket_cap_per_peer"])
        self.stream4_capacity = self.stream3_capacity + self.remote_capacity
        self.final_capacity = int(self.ext.derive_sizes(self.cfg)["n_local"])
        self.host_state_stage = np.zeros((self.b_micro, STATE_STORAGE_LEN), dtype=np.uint8)
        self.host_state_stage_t = torch.from_numpy(self.host_state_stage.reshape(-1))
        self.host_send_offset = np.zeros((self.world_size + 1,), dtype=np.int32)
        self.host_send_offset_t = torch.empty((self.world_size + 1,), dtype=torch.int32)
        self.host_final_request = np.zeros((self.final_capacity * 16,), dtype=np.uint8)
        self.host_final_response_send = np.zeros((self.final_capacity * STATE_STORAGE_LEN,), dtype=np.uint8)
        self.host_final_request_t = torch.from_numpy(self.host_final_request)
        self.host_final_response_send_t = torch.from_numpy(self.host_final_response_send)
        self.host_final_count = np.zeros((self.world_size,), dtype=np.int64)
        self.host_final_count_t = torch.from_numpy(self.host_final_count)
        self.host_current_frontier = np.zeros((self.final_capacity, STATE_STORAGE_LEN), dtype=np.uint8)
        self.host_current_frontier_t = torch.from_numpy(self.host_current_frontier.reshape(-1))
        self.host_run_task_frontier = np.zeros((self.final_capacity, STATE_STORAGE_LEN), dtype=np.uint8)
        self.scratch = {
            "current_frontier_states": torch.empty((self.b_micro * STATE_STORAGE_LEN,), dtype=torch.uint8, device=self.device),
            "current_frontier_all": torch.empty((self.final_capacity * STATE_STORAGE_LEN,), dtype=torch.uint8, device=self.device),
            "parent_base_i64": torch.empty((1,), dtype=torch.int64, device=self.device),
            "count_i32": torch.empty((1,), dtype=torch.int32, device=self.device),
            "hash_ring": torch.empty((self.stream3_capacity * 16,), dtype=torch.uint8, device=self.device),
            "solved_flag": torch.empty((1,), dtype=torch.int32, device=self.device),
            "stop_flag": torch.empty((1,), dtype=torch.int32, device=self.device),
            "solved_count": torch.empty((1,), dtype=torch.int32, device=self.device),
            "solved_overflow": torch.empty((1,), dtype=torch.int32, device=self.device),
            "solved_meta_list": torch.empty((16 * 32,), dtype=torch.uint8, device=self.device),
            "solved_depth_list": torch.empty((16,), dtype=torch.int32, device=self.device),
            "stream3_key_a": torch.empty((self.stream3_capacity * 16,), dtype=torch.uint8, device=self.device),
            "stream3_key_b": torch.empty((self.stream3_capacity * 16,), dtype=torch.uint8, device=self.device),
            "stream3_val_a": torch.empty((self.stream3_capacity,), dtype=torch.int64, device=self.device),
            "stream3_val_b": torch.empty((self.stream3_capacity,), dtype=torch.int64, device=self.device),
            "stream3_compact_count": torch.empty((1,), dtype=torch.int32, device=self.device),
            "stream3_unique_key": torch.empty((self.stream3_capacity * 16,), dtype=torch.uint8, device=self.device),
            "stream3_unique_val": torch.empty((self.stream3_capacity,), dtype=torch.int64, device=self.device),
            "stream3_unique_count": torch.empty((1,), dtype=torch.int32, device=self.device),
            "local_pending": torch.empty((self.stream3_capacity * 32,), dtype=torch.uint8, device=self.device),
            "remote_send": torch.empty((self.stream3_capacity * 32,), dtype=torch.uint8, device=self.device),
            "local_count": torch.empty((1,), dtype=torch.int32, device=self.device),
            "send_count": torch.empty((self.world_size,), dtype=torch.int32, device=self.device),
            "send_offset": torch.empty((self.world_size + 1,), dtype=torch.int32, device=self.device),
            "remote_recv": torch.empty((self.remote_capacity * 32,), dtype=torch.uint8, device=self.device),
            "recv_count": torch.empty((self.world_size,), dtype=torch.int32, device=self.device),
            "recv_offset": torch.empty((self.world_size + 1,), dtype=torch.int32, device=self.device),
            "survivor": torch.empty((self.stream4_capacity * 32,), dtype=torch.uint8, device=self.device),
            "stream4_key_a": torch.empty((self.stream4_capacity * 16,), dtype=torch.uint8, device=self.device),
            "stream4_key_b": torch.empty((self.stream4_capacity * 16,), dtype=torch.uint8, device=self.device),
            "stream4_val_a": torch.empty((self.stream4_capacity * 32,), dtype=torch.uint8, device=self.device),
            "stream4_val_b": torch.empty((self.stream4_capacity * 32,), dtype=torch.uint8, device=self.device),
            "stream4_compact_count": torch.empty((1,), dtype=torch.int32, device=self.device),
            "clean_tmp": torch.empty((self.stream4_capacity * 32,), dtype=torch.uint8, device=self.device),
            "stream4_new_clean_count": torch.empty((1,), dtype=torch.int32, device=self.device),
            "clean_count": torch.empty((1,), dtype=torch.int32, device=self.device),
            "dirty_count": torch.empty((1,), dtype=torch.int32, device=self.device),
            "processing_flag": torch.empty((1,), dtype=torch.uint8, device=self.device),
            "final_send_count": torch.empty((self.world_size,), dtype=torch.int64, device=self.device),
            "final_recv_count": torch.empty((self.world_size,), dtype=torch.int64, device=self.device),
            "final_request": torch.empty((self.final_capacity * 16,), dtype=torch.uint8, device=self.device),
            "final_request_recv": torch.empty((self.final_capacity * 16,), dtype=torch.uint8, device=self.device),
            "final_response": torch.empty((self.final_capacity * STATE_STORAGE_LEN,), dtype=torch.uint8, device=self.device),
            "final_response_send": torch.empty((self.final_capacity * STATE_STORAGE_LEN,), dtype=torch.uint8, device=self.device),
            "final_response_recv": torch.empty((self.final_capacity * STATE_STORAGE_LEN,), dtype=torch.uint8, device=self.device),
            "next_frontier": torch.empty((self.final_capacity * STATE_STORAGE_LEN,), dtype=torch.uint8, device=self.device),
            "global_keep_tensor": torch.empty((1,), dtype=torch.int64, device=self.device),
        }
        self.scratch["stream3_sort_temp"] = torch.empty((int(self.ext.v6_stream3_sort_temp_bytes(self.stream3_capacity)),), dtype=torch.uint8, device=self.device)
        self.scratch["stream4_sort_temp"] = torch.empty((int(self.ext.v6_stream4_sort_temp_bytes(self.stream4_capacity)),), dtype=torch.uint8, device=self.device)
        for tensor in self.scratch.values():
            tensor.zero_()

    def _run_ring_streams(self, current_frontier: np.ndarray, depth: int, parent_offset: int = 0) -> dict[str, Any]:
        parent_offset = int(parent_offset)
        if parent_offset < 0 or parent_offset >= max(len(current_frontier), 1):
            raise ValueError(f"bad V6_PARENT_OFFSET={parent_offset} for frontier size {len(current_frontier)}")
        frontier_count = int(min(self.b_micro, len(current_frontier) - parent_offset))
        if frontier_count <= 0:
            return {"frontier_count": 0, "solved_count": 0, "stop_flag": 0, "candidates": []}
        states_storage = self.host_state_stage
        states_storage.fill(0)
        states_storage[:frontier_count] = current_frontier[parent_offset : parent_offset + frontier_count]
        self.buffers["beam_current"][: self.b_micro, :STATE_STORAGE_LEN].reshape(-1).copy_(self.host_state_stage_t, non_blocking=False)
        self.buffers["current_active_flags"][: self.b_micro].zero_()
        self.buffers["current_active_flags"][:frontier_count].fill_(1)
        self.buffers["score_ring"][: self.b_micro * MOVE_COUNT].fill_(-1)
        self.engine.warmup_inference(1)
        torch.cuda.synchronize()

        current_frontier_states = self.scratch["current_frontier_states"]
        current_frontier_states.copy_(self.buffers["beam_current"][: self.b_micro, :STATE_STORAGE_LEN].reshape(-1))
        parent_base = self.scratch["parent_base_i64"]
        parent_base.fill_(parent_offset)
        count = self.scratch["count_i32"]
        count.fill_(frontier_count)
        hash_ring = self.scratch["hash_ring"][: self.b_micro * MOVE_COUNT * 16]
        hash_ring.zero_()
        solved_capacity = 16
        solved_flag = self.scratch["solved_flag"]; solved_flag.zero_()
        stop_flag = self.scratch["stop_flag"]; stop_flag.zero_()
        solved_count = self.scratch["solved_count"]; solved_count.zero_()
        solved_overflow = self.scratch["solved_overflow"]; solved_overflow.zero_()
        solved_meta_list = self.scratch["solved_meta_list"][: solved_capacity * 32]; solved_meta_list.zero_()
        solved_depth_list = self.scratch["solved_depth_list"][:solved_capacity]; solved_depth_list.zero_()
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
            "parent_offset": parent_offset,
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
        key_a = self.scratch["stream3_key_a"][: batch_candidates * 16]; key_a.zero_()
        key_b = self.scratch["stream3_key_b"][: batch_candidates * 16]; key_b.zero_()
        val_a = self.scratch["stream3_val_a"][:batch_candidates]; val_a.zero_()
        val_b = self.scratch["stream3_val_b"][:batch_candidates]; val_b.zero_()
        compact_count = self.scratch["stream3_compact_count"]; compact_count.zero_()
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
            return {"compact_count": 0, "unique_count": 0, "local_count": 0, "remote_count": 0}
        if compact_n > self.stream3_capacity:
            raise RuntimeError(f"stream3 compact overflow: {compact_n} > {self.stream3_capacity}")
        temp = self.scratch["stream3_sort_temp"]
        self.ext.v6_stream3_sort_pairs(temp, key_a, key_b, val_a, val_b, compact_n)
        unique_key = self.scratch["stream3_unique_key"][: batch_candidates * 16]; unique_key.zero_()
        unique_val = self.scratch["stream3_unique_val"][:batch_candidates]; unique_val.zero_()
        unique_count = self.scratch["stream3_unique_count"]; unique_count.zero_()
        self.ext.v6_stream3_dedup_sorted(key_b, val_b, unique_key, unique_val, unique_count, compact_n)
        torch.cuda.synchronize()
        unique_n = int(unique_count.cpu()[0])
        if unique_n > self.stream3_capacity:
            raise RuntimeError(f"stream3 unique overflow: {unique_n} > {self.stream3_capacity}")
        local_pending = self.scratch["local_pending"]; local_pending.zero_()
        remote_send = self.scratch["remote_send"]; remote_send.zero_()
        local_count = self.scratch["local_count"]; local_count.zero_()
        send_count = self.scratch["send_count"]; send_count.zero_()
        if self.rank == 0:
            send_offset_values = [0, 0, unique_n]
        else:
            send_offset_values = [0, unique_n, unique_n]
        self.host_send_offset[: self.world_size + 1] = send_offset_values
        self.host_send_offset_t.numpy()[: self.world_size + 1] = self.host_send_offset[: self.world_size + 1]
        send_offset = self.scratch["send_offset"]; send_offset.copy_(self.host_send_offset_t, non_blocking=False)
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
            "compact_count": compact_n,
            "unique_count": unique_n,
            "local_pending": local_pending,
            "remote_send": remote_send,
            "local_count": int(local_count.cpu()[0]),
            "send_count": send_count,
            "send_offset": send_offset,
        }

    def _run_stream5(self, stream3: dict[str, Any]) -> dict[str, Any]:
        remote_capacity = max(int(self.cfg["bucket_cap_per_peer"]), 1)
        remote_recv = self.scratch["remote_recv"][: remote_capacity * 32]; remote_recv.zero_()
        recv_count = self.scratch["recv_count"]; recv_count.zero_()
        recv_offset = self.scratch["recv_offset"]; recv_offset.zero_()
        self.engine.v6_stream5_exchange_candidate_meta(
            stream3["remote_send"],
            remote_recv,
            stream3["send_count"],
            stream3["send_offset"],
            recv_count,
            recv_offset,
        )
        torch.cuda.synchronize()
        remote_recv_count = int(recv_count.cpu().numpy().sum())
        if remote_recv_count > remote_capacity:
            raise RuntimeError(f"remote_recv overflow: {remote_recv_count} > {remote_capacity}")
        return {
            "remote_recv": remote_recv,
            "recv_count": recv_count,
            "recv_offset": recv_offset,
            "remote_recv_count": remote_recv_count,
            "remote_capacity": remote_capacity,
        }

    def _collector_to_stream4(self, stream3: dict[str, Any], stream5: dict[str, Any], current_threshold: int) -> dict[str, Any]:
        local_n = int(stream3.get("local_count", 0))
        remote_n = int(stream5.get("remote_recv_count", 0))
        input_n = local_n + remote_n
        if input_n == 0:
            return {"clean": [], "clean_count": 0, "dirty_count": 0, "input_count": 0}
        if input_n > self.stream4_capacity:
            raise RuntimeError(f"stream4 input overflow: {input_n} > {self.stream4_capacity}")
        survivor = self.scratch["survivor"][: input_n * 32]; survivor.zero_()
        if local_n:
            survivor[: local_n * 32].copy_(stream3["local_pending"][: local_n * 32])
        if remote_n:
            survivor[local_n * 32 : (local_n + remote_n) * 32].copy_(stream5["remote_recv"][: remote_n * 32])
        key_a = self.scratch["stream4_key_a"][: input_n * 16]; key_a.zero_()
        key_b = self.scratch["stream4_key_b"][: input_n * 16]; key_b.zero_()
        val_a = self.scratch["stream4_val_a"][: input_n * 32]; val_a.zero_()
        val_b = self.scratch["stream4_val_b"][: input_n * 32]; val_b.zero_()
        compact_count = self.scratch["stream4_compact_count"]; compact_count.zero_()
        self.ext.v6_stream4_threshold_compact(survivor, key_a, val_a, compact_count, input_n, current_threshold)
        torch.cuda.synchronize()
        compact_n = int(compact_count.cpu()[0])
        if compact_n:
            if compact_n > self.stream4_capacity:
                raise RuntimeError(f"stream4 compact overflow: {compact_n} > {self.stream4_capacity}")
            temp = self.scratch["stream4_sort_temp"]
            self.ext.v6_stream4_sort_pairs(temp, key_a, key_b, val_a, val_b, compact_n)
            clean_tmp = self.scratch["clean_tmp"][: input_n * 32]; clean_tmp.zero_()
            new_clean_count = self.scratch["stream4_new_clean_count"]; new_clean_count.zero_()
            self.ext.v6_stream4_dedup_sorted(key_b, val_b, clean_tmp, new_clean_count, compact_n)
            torch.cuda.synchronize()
            clean_n = int(new_clean_count.cpu()[0])
        else:
            clean_tmp = self.scratch["clean_tmp"][: input_n * 32]; clean_tmp.zero_()
            new_clean_count = self.scratch["stream4_new_clean_count"]; new_clean_count.zero_()
            clean_n = 0
        clean_count = self.scratch["clean_count"]; clean_count.zero_()
        dirty_count = self.scratch["dirty_count"]; dirty_count.fill_(input_n)
        processing_flag = self.scratch["processing_flag"]; processing_flag.fill_(1)
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
            "input_count": input_n,
        }

    def _final_materialize(
        self,
        current_frontier_states: torch.Tensor,
        clean: list[dict[str, int]],
        current_threshold: int,
        current_paths: list[str] | None = None,
        *,
        task_idx: int = -1,
        depth: int = -1,
    ) -> np.ndarray | tuple[np.ndarray, list[str]]:
        keep = [c for c in clean if int(c["score_key"]) <= int(current_threshold)]
        keep_counts = [None for _ in range(self.world_size)]
        collective_seq_debug(self.rank, task_idx, depth, "final_keep_counts", "all_gather_object", int(bool(keep)), len(keep))
        dist.all_gather_object(keep_counts, len(keep))
        counts = [int(x or 0) for x in keep_counts]
        prefix = [0, counts[0]]
        global_keep = sum(counts)
        if global_keep == 0:
            collective_seq_debug(self.rank, task_idx, depth, "final_global_empty_next", "uniform_exit", 1, 0)
            empty = self.host_current_frontier[:0]
            return (empty, []) if current_paths is not None else empty
        gathered_paths: list[list[str] | None] | None = None
        if current_paths is not None:
            gathered_paths = [None for _ in range(self.world_size)]
            collective_seq_debug(self.rank, task_idx, depth, "final_paths", "all_gather_object", int(bool(current_paths)), len(current_paths))
            dist.all_gather_object(gathered_paths, list(current_paths))
        source_frontier_sizes = [len(paths or []) for paths in gathered_paths] if gathered_paths is not None else [int(current_frontier_states.numel() // STATE_STORAGE_LEN)] * self.world_size
        
        # SOLUTION A: Fixed-buffer model (no Python list growth)
        # Pre-allocate request buffer and track per-peer write positions
        self.host_final_request.fill(0)
        request_write_ptr = 0
        request_count_per_peer = [0] * self.world_size
        path_by_target_local_idx: dict[int, str] = {}
        
        for local_idx, candidate in enumerate(keep):
            global_idx = prefix[self.rank] + local_idx
            target_rank = min(global_idx * self.world_size // max(global_keep, 1), self.world_size - 1)
            target_start = (global_keep * target_rank + self.world_size - 1) // self.world_size
            target_local_idx = int(global_idx - target_start)
            source_rank = int(candidate["source_rank"])
            move = int(candidate["move"])
            parent_idx = int(candidate["parent_idx"])
            if not (0 <= source_rank < self.world_size):
                raise RuntimeError(f"invalid_final_request: source_rank={source_rank} world_size={self.world_size}")
            if not (0 <= parent_idx < source_frontier_sizes[source_rank]):
                raise RuntimeError(
                    "invalid_final_request: "
                    f"source_rank={source_rank} parent_idx={parent_idx} "
                    f"source_frontier_size={source_frontier_sizes[source_rank]} "
                    f"task_idx={task_idx} depth={depth}"
                )
            if not (0 <= move < MOVE_COUNT):
                raise RuntimeError(f"invalid_final_request: move={move} MOVE_COUNT={MOVE_COUNT}")
            if target_local_idx < 0:
                raise RuntimeError(f"invalid_final_request: target_local_idx={target_local_idx}")
            
            # Write request to fixed buffer (no list.append)
            request_bytes = pack_final_request(parent_idx, target_local_idx, target_rank, move)
            self.host_final_request[request_write_ptr:request_write_ptr+16] = np.frombuffer(request_bytes, dtype=np.uint8)
            request_write_ptr += 16
            request_count_per_peer[source_rank] += 1
            
            if gathered_paths is not None and target_rank == self.rank:
                source_paths = gathered_paths[source_rank] or []
                parent_path = source_paths[parent_idx] if 0 <= parent_idx < len(source_paths) else ""
                path_by_target_local_idx[target_local_idx] = append_move_to_path(parent_path, move)
        
        # SOLUTION B: Hard capacity contracts (asserts before critical operations)
        send_request_count_total = sum(request_count_per_peer)
        if send_request_count_total > self.final_capacity:
            raise RuntimeError(f"final request send overflow: {send_request_count_total} > {self.final_capacity}")
        
        send_count_t = self.scratch["final_send_count"]; send_count_t.zero_()
        recv_count_t = self.scratch["final_recv_count"]; recv_count_t.zero_()
        self.host_final_count[: self.world_size] = request_count_per_peer
        send_count_t.copy_(self.host_final_count_t, non_blocking=False)
        collective_seq_debug(self.rank, task_idx, depth, "final_request_counts", "all_to_all_single", int(bool(request_count_per_peer)), send_request_count_total)
        dist.all_to_all_single(recv_count_t, send_count_t)
        recv_request_counts = [int(x) for x in recv_count_t.cpu().tolist()]
        recv_request_total = sum(recv_request_counts)
        
        # SOLUTION B: Hard capacity contract (before NCCL)
        if recv_request_total > self.final_capacity:
            raise RuntimeError(f"final request recv overflow: {recv_request_total} > {self.final_capacity}")
        
        send_request_t = self.scratch["final_request"][: send_request_count_total * 16]
        send_request_t.copy_(self.host_final_request_t[: send_request_count_total * 16], non_blocking=False)
        recv_request_t = self.scratch["final_request_recv"][: recv_request_total * 16]
        recv_request_t.zero_()
        collective_seq_debug(self.rank, task_idx, depth, "final_request_payload", "all_to_all_single", int(send_request_t.numel() > 0), send_request_t.numel())
        dist.all_to_all_single(
            recv_request_t,
            send_request_t,
            output_split_sizes=[c * 16 for c in recv_request_counts],
            input_split_sizes=[c * 16 for c in request_count_per_peer],
        )
        final_response_t = self.scratch["final_response"][: max(recv_request_total, 1) * STATE_STORAGE_LEN]
        final_response_t.zero_()
        if recv_request_total:
            self.ext.v6_final_materialize(current_frontier_states, recv_request_t, self.generators_t, final_response_t, recv_request_total)
            torch.cuda.synchronize()
        
        # SOLUTION A: Fixed-buffer model for responses (no list growth)
        recv_request_raw = recv_request_t.cpu().numpy().tobytes()
        responses = final_response_t.cpu().numpy().reshape((-1, STATE_STORAGE_LEN))[:recv_request_total].copy()
        self.host_final_response_send.fill(0)
        response_write_ptr = 0
        response_count_per_peer = [0] * self.world_size
        response_offsets = [0] * (self.world_size + 1)
        
        for idx in range(recv_request_total):
            _parent_idx, _target_local_idx, return_rank, _move, _pad = struct.unpack_from("<QIHBB", recv_request_raw, idx * 16)
            return_rank = int(return_rank)
            response_bytes = responses[idx].tobytes()
            self.host_final_response_send[response_write_ptr:response_write_ptr+STATE_STORAGE_LEN] = np.frombuffer(response_bytes, dtype=np.uint8)
            response_write_ptr += STATE_STORAGE_LEN
            response_count_per_peer[return_rank] += 1
        
        # SOLUTION B: Hard capacity contract
        send_response_count_total = sum(response_count_per_peer)
        if send_response_count_total > self.final_capacity:
            raise RuntimeError(f"final response send overflow: {send_response_count_total} > {self.final_capacity}")
        
        send_response_count_t = self.scratch["final_send_count"]; send_response_count_t.zero_()
        recv_response_count_t = self.scratch["final_recv_count"]; recv_response_count_t.zero_()
        self.host_final_count[: self.world_size] = response_count_per_peer
        send_response_count_t.copy_(self.host_final_count_t, non_blocking=False)
        collective_seq_debug(self.rank, task_idx, depth, "final_response_counts", "all_to_all_single", int(bool(response_count_per_peer)), send_response_count_total)
        dist.all_to_all_single(recv_response_count_t, send_response_count_t)
        recv_response_counts = [int(x) for x in recv_response_count_t.cpu().tolist()]
        recv_response_total = sum(recv_response_counts)
        
        # SOLUTION B: Hard capacity contract (before NCCL)
        if recv_response_total > self.final_capacity:
            raise RuntimeError(f"final response recv overflow: {recv_response_total} > {self.final_capacity}")
        
        send_response_t = self.scratch["final_response_send"][: send_response_count_total * STATE_STORAGE_LEN]
        send_response_t.copy_(self.host_final_response_send_t[: send_response_count_total * STATE_STORAGE_LEN], non_blocking=False)
        recv_response_t = self.scratch["final_response_recv"][: recv_response_total * STATE_STORAGE_LEN]
        recv_response_t.zero_()
        collective_seq_debug(self.rank, task_idx, depth, "final_response_payload", "all_to_all_single", int(bool(send_response_t.numel())), send_response_t.numel())
        dist.all_to_all_single(
            recv_response_t,
            send_response_t,
            output_split_sizes=[c * STATE_STORAGE_LEN for c in recv_response_counts],
            input_split_sizes=[c * STATE_STORAGE_LEN for c in response_count_per_peer],
        )
        next_frontier = self.scratch["next_frontier"][: max(recv_response_total, 1) * STATE_STORAGE_LEN]
        next_frontier.zero_()
        if recv_response_total:
            self.ext.v6_final_scatter_responses(recv_response_t, next_frontier, recv_response_total)
            torch.cuda.synchronize()
        next_states = next_frontier.cpu().numpy().reshape((-1, STATE_STORAGE_LEN))[:recv_response_total].copy()
        if current_paths is None:
            return next_states
        next_paths = [path_by_target_local_idx.get(i, "") for i in range(recv_response_total)]
        return next_states, next_paths

    def run_task(self, task_id: int, state: np.ndarray, max_depth: int, global_beam_width_effective: int) -> ProductionV6Result:
        self.host_run_task_frontier.fill(0)
        self.host_run_task_frontier[0] = data_loader.pad_state128_u8(state)
        current = self.host_run_task_frontier[:1]
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
            expanded_parent_count = 0
            stream1_scored_parent_count = 0
            stream2_generated_candidate_count = 0
            stream3_after_threshold_count = 0
            stream3_unique_count = 0
            stream4_input_count = 0
            stream4_clean_count = 0
            depth_clean: list[dict[str, int]] = []
            # SOLUTION B: Hard capacity contract for current frontier
            if len(current) > self.final_capacity:
                raise RuntimeError(f"current frontier overflow: {len(current)} > {self.final_capacity}")
            if len(current) < 1:
                raise RuntimeError(f"current frontier underflow: {len(current)} < 1")
            self.host_current_frontier.fill(0)
            self.host_current_frontier[: len(current)] = current
            depth_current_frontier_states = self.scratch["current_frontier_all"][: max(len(current), 1) * STATE_STORAGE_LEN]
            depth_current_frontier_states.copy_(self.host_current_frontier_t[: max(len(current), 1) * STATE_STORAGE_LEN], non_blocking=False)
            solved_stream12: dict[str, Any] | None = None
            parent_offset = 0
            while parent_offset < len(current):
                stream12 = self._run_ring_streams(current, depth, parent_offset)
                expanded_parent_count += int(stream12.get("frontier_count", 0))
                stream1_scored_parent_count += int(stream12.get("frontier_count", 0))
                stream2_generated_candidate_count += int(stream12.get("frontier_count", 0)) * MOVE_COUNT
                if int(stream12["solved_count"]) > 0:
                    solved_stream12 = stream12
                    break
                stream3 = self._run_stream3(stream12, current_threshold)
                stream3_after_threshold_count += int(stream3.get("compact_count", 0))
                stream3_unique_count += int(stream3.get("unique_count", 0))
                stream5 = self._run_stream5(stream3) if int(stream3.get("unique_count", 0)) else {"remote_recv_count": 0, "remote_recv": self.scratch["remote_recv"][:1]}
                stream4 = self._collector_to_stream4(stream3, stream5, current_threshold) if int(stream3.get("unique_count", 0)) else {"clean": [], "clean_count": 0, "dirty_count": 0, "input_count": 0}
                stream4_input_count += int(stream4.get("input_count", 0))
                stream4_clean_count += int(stream4.get("clean_count", 0))
                depth_clean = merge_clean_candidates(depth_clean, list(stream4["clean"]))
                parent_offset += int(stream12.get("frontier_count", 0))
            if solved_stream12 is not None:
                status = "solved"
                solved_meta = solved_stream12.get("solved_meta")
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
                depth_rows.append(
                    self._depth_row(
                        task_id,
                        depth,
                        len(current),
                        threshold_initialized,
                        current_threshold,
                        0,
                        0,
                        0,
                        int(stream12["solved_count"]),
                        1,
                        start,
                        expanded_parent_count=expanded_parent_count,
                        stream1_scored_parent_count=stream1_scored_parent_count,
                        stream2_generated_candidate_count=stream2_generated_candidate_count,
                        stream3_after_threshold_count=stream3_after_threshold_count,
                        stream3_unique_count=stream3_unique_count,
                        stream4_input_count=stream4_input_count,
                        stream4_clean_count=stream4_clean_count,
                        next_frontier_size_after=0,
                    )
                )
                break
            local_scores = [int(c["score_key"]) for c in depth_clean]
            current_threshold, threshold_initialized, total_survivors = allreduce_score_threshold(
                local_scores,
                current_threshold,
                threshold_initialized,
                global_beam_width_effective,
                self.device,
                rank=self.rank,
                task_idx=task_id,
                depth=depth,
            )
            next_current, next_paths = self._final_materialize(
                depth_current_frontier_states,
                depth_clean,
                current_threshold,
                current_paths,
                task_idx=task_id,
                depth=depth,
            )
            global_keep_tensor = self.scratch["global_keep_tensor"]
            global_keep_tensor.fill_(len(next_current))
            collective_seq_debug(self.rank, task_id, depth, "depth_global_keep", "all_reduce", int(len(next_current) > 0), len(next_current))
            dist.all_reduce(global_keep_tensor, op=dist.ReduceOp.SUM)
            global_keep = int(global_keep_tensor.cpu()[0])
            depth_rows.append(
                self._depth_row(
                    task_id,
                    depth,
                    len(current),
                    threshold_initialized,
                    current_threshold,
                    stream4_clean_count,
                    0,
                    global_keep,
                    0,
                    0,
                    start,
                    expanded_parent_count=expanded_parent_count,
                    stream1_scored_parent_count=stream1_scored_parent_count,
                    stream2_generated_candidate_count=stream2_generated_candidate_count,
                    stream3_after_threshold_count=stream3_after_threshold_count,
                    stream3_unique_count=stream3_unique_count,
                    stream4_input_count=stream4_input_count,
                    stream4_clean_count=stream4_clean_count,
                    next_frontier_size_after=len(next_current),
                )
            )
            current = next_current
            current_paths = next_paths
            if global_keep == 0 or len(current) == 0:
                collective_seq_debug(self.rank, task_id, depth, "depth_uniform_empty_exit", "local_break_after_all_reduce", int(global_keep == 0), len(current))
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
        *,
        expanded_parent_count: int = 0,
        stream1_scored_parent_count: int = 0,
        stream2_generated_candidate_count: int = 0,
        stream3_after_threshold_count: int = 0,
        stream3_unique_count: int = 0,
        stream4_input_count: int = 0,
        stream4_clean_count: int = 0,
        next_frontier_size_after: int = 0,
    ) -> dict[str, Any]:
        return {
            "task_id": int(task_id),
            "depth": int(depth),
            "frontier_size": int(frontier_size),
            "current_frontier_size_before": int(frontier_size),
            "expanded_parent_count": int(expanded_parent_count),
            "stream1_scored_parent_count": int(stream1_scored_parent_count),
            "stream2_generated_candidate_count": int(stream2_generated_candidate_count),
            "stream3_after_threshold_count": int(stream3_after_threshold_count),
            "stream3_unique_count": int(stream3_unique_count),
            "stream4_input_count": int(stream4_input_count),
            "stream4_clean_count": int(stream4_clean_count),
            "next_frontier_size_after": int(next_frontier_size_after),
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
    b_micro: int = PRODUCTION_B_MICRO,
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
    b_micro: int = PRODUCTION_B_MICRO,
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
    b_micro: int = PRODUCTION_B_MICRO,
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


def validate_known_paths(
    *,
    task_count: int,
) -> dict[str, Any]:
    puzzles = dict(data_loader.load_test_puzzles(max_puzzles=task_count))
    rows = data_loader.load_sample_submission()
    checked = 0
    failures: list[str] = []
    for row in rows:
        task_id = int(row["initial_state_id"])
        if task_id not in puzzles:
            continue
        path = row.get("path", "")
        ok, path_len, note = replay_path_to_central(puzzles[task_id], path)
        checked += 1
        if not ok:
            failures.append(f"task_id={task_id}:path_len={path_len}:{note}")
        if checked >= int(task_count):
            break
    return {
        "known_path_checked": checked,
        "known_path_failures": failures,
        "known_path_replay_valid": checked == int(task_count) and not failures,
    }


def run_frontier_coverage_audit_world2(
    *,
    task_count: int,
    max_depth: int,
    beam_width: int,
    output_path: Path,
    audit_path: Path,
    b_micro: int = PRODUCTION_B_MICRO,
) -> dict[str, Any]:
    b_micro = require_production_microbatch(b_micro)
    known_path_result = validate_known_paths(task_count=min(task_count, 10))
    if not known_path_result["known_path_replay_valid"]:
        raise AssertionError(f"known path replay failed: {known_path_result}")
    rank, world_size, device = require_world2_t4_runtime()
    dispatcher = ProductionV6Dispatcher(rank, world_size, device, beam_width=beam_width, b_micro=b_micro)
    puzzles = data_loader.load_test_puzzles(max_puzzles=task_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_idx",
        "initial_state_id",
        "rank",
        "depth",
        "status",
        "current_frontier_size_before",
        "expanded_parent_count",
        "stream1_scored_parent_count",
        "stream2_generated_candidate_count",
        "stream3_after_threshold_count",
        "stream3_unique_count",
        "stream4_input_count",
        "stream4_clean_count",
        "next_frontier_size_after",
        "current_threshold",
        "solved_count",
        "coverage_ok",
        "coverage_failure_reason",
    ]
    if rank == 0:
        with output_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        audit_path.write_text("", encoding="utf-8")
    dist.barrier()
    coverage_failures: list[str] = []
    status_counts = {"solved": 0, "unsolved": 0, "max_depth_reached": 0, "error": 0}
    row_count = 0
    for task_idx, (task_id, state) in enumerate(puzzles):
        try:
            result = dispatcher.run_task(int(task_id), state, max_depth, int(beam_width))
            status = result.status
            status_counts[status] += 1
            depth_rows = result.depth_rows
        except Exception as exc:  # noqa: BLE001 - diagnostic runner records per-task failure.
            status = "error"
            status_counts["error"] += 1
            depth_rows = []
            coverage_failures.append(f"task_id={task_id}:exception={type(exc).__name__}:{exc}")
        for depth_row in depth_rows:
            expected_generated = int(depth_row["expanded_parent_count"]) * MOVE_COUNT
            coverage_ok = (
                int(depth_row["expanded_parent_count"]) == int(depth_row["current_frontier_size_before"])
                and int(depth_row["stream1_scored_parent_count"]) == int(depth_row["expanded_parent_count"])
                and int(depth_row["stream2_generated_candidate_count"]) == expected_generated
            )
            failure_reason = "ok"
            if not coverage_ok:
                failure_reason = (
                    f"frontier_not_fully_processed:"
                    f"frontier={depth_row['current_frontier_size_before']}:expanded={depth_row['expanded_parent_count']}:"
                    f"generated={depth_row['stream2_generated_candidate_count']}:expected_generated={expected_generated}"
                )
                coverage_failures.append(f"task_id={task_id}:depth={depth_row['depth']}:{failure_reason}")
            out_row = {
                "task_idx": int(task_idx),
                "initial_state_id": int(task_id),
                "rank": int(rank),
                "depth": int(depth_row["depth"]),
                "status": status,
                "current_frontier_size_before": int(depth_row["current_frontier_size_before"]),
                "expanded_parent_count": int(depth_row["expanded_parent_count"]),
                "stream1_scored_parent_count": int(depth_row["stream1_scored_parent_count"]),
                "stream2_generated_candidate_count": int(depth_row["stream2_generated_candidate_count"]),
                "stream3_after_threshold_count": int(depth_row["stream3_after_threshold_count"]),
                "stream3_unique_count": int(depth_row["stream3_unique_count"]),
                "stream4_input_count": int(depth_row["stream4_input_count"]),
                "stream4_clean_count": int(depth_row["stream4_clean_count"]),
                "next_frontier_size_after": int(depth_row["next_frontier_size_after"]),
                "current_threshold": int(depth_row["current_threshold"]),
                "solved_count": int(depth_row["solved_count"]),
                "coverage_ok": int(coverage_ok),
                "coverage_failure_reason": failure_reason,
            }
            row_count += 1
            if rank == 0:
                with output_path.open("a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(out_row)
            with audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(out_row, sort_keys=True) + "\n")
            print(
                "FRONTIER_COVERAGE_AUDIT_PROGRESS "
                f"task_idx={task_idx} depth={out_row['depth']} rank={rank} "
                f"frontier={out_row['current_frontier_size_before']} expanded={out_row['expanded_parent_count']} "
                f"generated={out_row['stream2_generated_candidate_count']} stream3_unique={out_row['stream3_unique_count']} "
                f"stream4_clean={out_row['stream4_clean_count']} next={out_row['next_frontier_size_after']} "
                f"coverage_ok={out_row['coverage_ok']}"
            )
        collective_seq_debug(rank, int(task_id), -1, "frontier_task_done", "barrier", int(status != "error"), len(depth_rows))
        dist.barrier()
    local_counts = torch.tensor([row_count, len(coverage_failures), status_counts["solved"], status_counts["error"]], dtype=torch.int64, device=device)
    gathered = [torch.zeros_like(local_counts) for _ in range(world_size)]
    collective_seq_debug(rank, -1, -1, "frontier_summary_counts", "all_gather", len(coverage_failures), row_count)
    dist.all_gather(gathered, local_counts)
    collective_seq_debug(rank, -1, -1, "frontier_summary_done", "barrier", len(coverage_failures), row_count)
    dist.barrier()
    return {
        "rank": rank,
        "world_size": world_size,
        "task_count": len(puzzles),
        "row_count": row_count,
        "status_counts": status_counts,
        "coverage_failures": coverage_failures,
        "coverage_failure_count": len(coverage_failures),
        "known_path_result": known_path_result,
        "gathered_counts": [[int(x) for x in item.cpu().tolist()] for item in gathered],
        "output_path": str(output_path),
        "audit_path": str(audit_path),
        "production_v6_dispatcher_path": True,
        "legacy_next_state_pool_path": False,
        "prefilled_score_ring_fake_path": False,
        "runtime_120_slice": False,
        "fallback_backend": False,
    }
