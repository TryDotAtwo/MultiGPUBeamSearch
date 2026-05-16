from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import beam_engine


STATE_LEN = 120
STATE_STORAGE_LEN = 128
MOVE_COUNT = 24


def make_generators() -> np.ndarray:
    generators = np.zeros((MOVE_COUNT, STATE_STORAGE_LEN), dtype=np.uint8)
    for move in range(MOVE_COUNT):
        generators[move] = np.arange(STATE_STORAGE_LEN, dtype=np.uint8)
        if move != 0:
            a = move % STATE_LEN
            b = (move * 7) % STATE_LEN
            generators[move, a] = b
            generators[move, b] = a
        generators[move, STATE_LEN:] = np.arange(STATE_LEN, STATE_STORAGE_LEN, dtype=np.uint8)
    return generators


def pack_final_request(parent_idx: int, target_local_idx: int, return_rank: int, move: int) -> bytes:
    return struct.pack("<QI HBB", parent_idx, target_local_idx, return_rank, move, 0)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for final materialization smoke")

    os.environ.setdefault("INFERENCE_BACKEND", "fullbeamnice_static")
    ext = beam_engine.build_extension(verbose=False)
    device = torch.device("cuda", 0)

    state_count = 4
    request_count = 3
    states = np.zeros((state_count, STATE_STORAGE_LEN), dtype=np.uint8)
    for i in range(state_count):
        states[i, :STATE_LEN] = ((np.arange(STATE_LEN, dtype=np.uint16) * 3 + i * 11) % 128).astype(np.uint8)
        states[i, STATE_LEN:] = 0
    states[2, STATE_LEN:] = np.arange(1, 9, dtype=np.uint8)

    generators = make_generators()
    requests = [
        (2, 1, 0, 5),
        (0, 0, 0, 0),
        (3, 2, 0, 11),
    ]
    request_bytes = b"".join(pack_final_request(*r) for r in requests)

    current_frontier_states = torch.tensor(states.reshape(-1), dtype=torch.uint8, device=device)
    final_request_buffer = torch.tensor(np.frombuffer(request_bytes, dtype=np.uint8).copy(), dtype=torch.uint8, device=device)
    generators_t = torch.tensor(generators.reshape(-1), dtype=torch.uint8, device=device)
    final_response_buffer = torch.zeros((request_count * STATE_STORAGE_LEN,), dtype=torch.uint8, device=device)
    next_frontier_states_tmp = torch.full((state_count * STATE_STORAGE_LEN,), 255, dtype=torch.uint8, device=device)

    ext.v6_final_materialize(
        current_frontier_states,
        final_request_buffer,
        generators_t,
        final_response_buffer,
        request_count,
    )
    torch.cuda.synchronize()

    responses = final_response_buffer.cpu().numpy().reshape(request_count, STATE_STORAGE_LEN)
    for i, (parent_idx, target_local_idx, _return_rank, move) in enumerate(requests):
        expected_child = states[parent_idx][generators[move]]
        assert np.array_equal(responses[i, :STATE_LEN], expected_child[:STATE_LEN])
        packed_target = int.from_bytes(bytes(responses[i, 120:124]), "little")
        assert packed_target == target_local_idx
        assert np.array_equal(responses[i, 124:128], np.zeros((4,), dtype=np.uint8))

    ext.v6_final_scatter_responses(
        final_response_buffer,
        next_frontier_states_tmp,
        request_count,
    )
    torch.cuda.synchronize()

    next_states = next_frontier_states_tmp.cpu().numpy().reshape(state_count, STATE_STORAGE_LEN)
    for parent_idx, target_local_idx, _return_rank, move in requests:
        expected_child = states[parent_idx][generators[move]]
        assert np.array_equal(next_states[target_local_idx, :STATE_LEN], expected_child[:STATE_LEN])
        assert np.array_equal(next_states[target_local_idx, STATE_LEN:], np.zeros((8,), dtype=np.uint8))

    assert np.array_equal(next_states[3], np.full((STATE_STORAGE_LEN,), 255, dtype=np.uint8))
    print("FINAL_MATERIALIZATION_SMOKE_OK")


if __name__ == "__main__":
    main()
