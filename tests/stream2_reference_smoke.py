from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import beam_engine


STATE_STORAGE_LEN = 128
STATE_LEN = 120
STATE_VALUE_PAD = 128
MOVE_COUNT = 24
GOAL_SCORE_KEY = 0


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


def make_zobrist() -> np.ndarray:
    z = np.zeros((STATE_STORAGE_LEN, STATE_VALUE_PAD, 2), dtype=np.uint64)
    for p in range(STATE_LEN):
        for v in range(STATE_VALUE_PAD):
            seed = p * 257 + v * 17 + 11
            z[p, v, 0] = splitmix64(seed)
            z[p, v, 1] = splitmix64(seed + 0xABCDEF)
    return z


def hash_state_cpu(state: np.ndarray, zobrist: np.ndarray) -> tuple[int, int]:
    lo = 0
    hi = 0
    for p in range(STATE_STORAGE_LEN):
        v = int(state[p])
        lo ^= int(zobrist[p, v, 0])
        hi ^= int(zobrist[p, v, 1])
    return lo, hi


def pack_route(source_rank: int, owner: int, move: int) -> int:
    return (source_rank << 16) | (owner << 8) | move


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Stream2 reference smoke")

    os.environ.setdefault("INFERENCE_BACKEND", "fullbeamnice_static")
    ext = beam_engine.build_extension(verbose=False)
    device = torch.device("cuda", 0)

    b_micro = 3
    ring_count = 1
    ring_slot_count = 1
    depth = 7
    local_rank = 0
    solved_capacity = 1

    states = np.zeros((b_micro, STATE_STORAGE_LEN), dtype=np.uint8)
    for i in range(b_micro):
        states[i, :STATE_LEN] = (np.arange(STATE_LEN, dtype=np.uint16) + i * 3).astype(np.uint8) % STATE_VALUE_PAD
    states[:, STATE_LEN:] = 0

    generators = np.zeros((MOVE_COUNT, STATE_STORAGE_LEN), dtype=np.uint8)
    for move in range(MOVE_COUNT):
        generators[move] = np.arange(STATE_STORAGE_LEN, dtype=np.uint8)
        if move != 0:
            generators[move, 0] = move % STATE_LEN
            generators[move, move % STATE_LEN] = 0
        generators[move, STATE_LEN:] = np.arange(STATE_LEN, STATE_STORAGE_LEN, dtype=np.uint8)

    central = np.zeros((STATE_STORAGE_LEN,), dtype=np.uint8)
    central[:] = states[0]
    central[STATE_LEN:] = 0

    zobrist = make_zobrist()
    assert np.all(zobrist[STATE_LEN:, :, :] == 0)

    current_frontier_states = torch.tensor(states.reshape(-1), dtype=torch.uint8, device=device)
    parent_base = torch.tensor([0], dtype=torch.int64, device=device)
    count = torch.tensor([b_micro], dtype=torch.int32, device=device)
    score_ring = torch.zeros((ring_count * ring_slot_count * b_micro * MOVE_COUNT,), dtype=torch.int32, device=device)
    hash_ring = torch.zeros((ring_count * ring_slot_count * b_micro * MOVE_COUNT * 16,), dtype=torch.uint8, device=device)
    generators_t = torch.tensor(generators.reshape(-1), dtype=torch.uint8, device=device)
    central_t = torch.tensor(central, dtype=torch.uint8, device=device)
    zobrist_t = torch.tensor(zobrist.reshape(-1).view(np.uint8), dtype=torch.uint8, device=device)
    solved_flag = torch.zeros((1,), dtype=torch.int32, device=device)
    stop_flag = torch.zeros((1,), dtype=torch.int32, device=device)
    solved_count = torch.zeros((1,), dtype=torch.int32, device=device)
    solved_overflow = torch.zeros((1,), dtype=torch.int32, device=device)
    solved_meta_list = torch.zeros((solved_capacity * 32,), dtype=torch.uint8, device=device)
    solved_depth_list = torch.zeros((solved_capacity,), dtype=torch.int32, device=device)

    ext.v6_stream2_hash_goal(
        current_frontier_states,
        parent_base,
        count,
        score_ring,
        hash_ring,
        generators_t,
        central_t,
        zobrist_t,
        solved_flag,
        stop_flag,
        solved_count,
        solved_overflow,
        solved_meta_list,
        solved_depth_list,
        solved_capacity,
        depth,
        local_rank,
        0,
        0,
        ring_slot_count,
        b_micro,
    )
    torch.cuda.synchronize()

    hash_bytes = hash_ring.cpu().numpy().tobytes()
    got_hashes = [struct.unpack_from("<QQ", hash_bytes, i * 16) for i in range(b_micro * MOVE_COUNT)]
    expected_hashes: list[tuple[int, int]] = []
    for parent_local in range(b_micro):
        for move in range(MOVE_COUNT):
            child = states[parent_local][generators[move]]
            expected_hashes.append(hash_state_cpu(child, zobrist))
    assert got_hashes == expected_hashes

    assert int(solved_flag.cpu()[0]) == 1
    assert int(stop_flag.cpu()[0]) == 1
    assert int(solved_count.cpu()[0]) >= 1
    assert int(solved_depth_list.cpu()[0]) == depth

    meta = solved_meta_list.cpu().numpy().tobytes()
    meta_lo, meta_hi, parent_idx, score_key, route_packed = struct.unpack_from("<QQQII", meta, 0)
    expected_goal_hash = expected_hashes[0]
    assert (meta_lo, meta_hi) == expected_goal_hash
    assert parent_idx == 0
    assert score_key == GOAL_SCORE_KEY
    assert route_packed == pack_route(local_rank, local_rank, 0)

    print("STREAM2_REFERENCE_SMOKE_OK")


if __name__ == "__main__":
    main()
