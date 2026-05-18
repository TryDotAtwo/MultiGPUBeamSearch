"""
data_loader.py

Strict loader and CPU reference helpers for puzzle_info.json and test.csv.
The CUDA engine consumes action tables as uint8 permutations with fixed order.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

ACTION_NAMES: List[str] = [
    "-B", "-BL", "-BR", "-D", "-DL", "-DR",
    "-F", "-FL", "-FR", "-L", "-R", "-U",
    "B", "BL", "BR", "D", "DL", "DR",
    "F", "FL", "FR", "L", "R", "U",
]
STATE_SIZE = 120
STATE_STORAGE_SIZE = 128
FANOUT = 24
HASH_EMPTY = 0
HASH_BUSY = 1
HASH_TOMBSTONE = 2


def get_data_dir() -> Path:
    data_dir = Path(__file__).resolve().parent / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"data directory not found: {data_dir}")
    return data_dir


def load_puzzle_info() -> Dict[str, Any]:
    path = get_data_dir() / "puzzle_info.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    validate_puzzle_info(data)
    return data


def validate_permutation(perm: Iterable[int], *, name: str) -> np.ndarray:
    arr = np.asarray(list(perm), dtype=np.int32)
    if arr.shape != (STATE_SIZE,):
        raise ValueError(f"{name}: expected length {STATE_SIZE}, got {arr.shape}")
    if sorted(arr.tolist()) != list(range(STATE_SIZE)):
        raise ValueError(f"{name}: not a permutation of 0..{STATE_SIZE-1}")
    return arr


def validate_puzzle_info(data: Dict[str, Any]) -> None:
    if "central_state" not in data or "generators" not in data:
        raise ValueError("puzzle_info.json must contain central_state and generators")
    validate_state(data["central_state"], name="central_state")
    generators = data["generators"]
    missing = [name for name in ACTION_NAMES if name not in generators]
    extra = sorted(set(generators) - set(ACTION_NAMES))
    if missing:
        raise ValueError(f"missing generators: {missing}")
    if extra:
        raise ValueError(f"unexpected generators: {extra}")
    for name in ACTION_NAMES:
        validate_permutation(generators[name], name=f"generator[{name}]")


def validate_state(state: Iterable[int], *, name: str = "state") -> np.ndarray:
    arr = np.asarray(list(state), dtype=np.int32)
    if arr.shape != (STATE_SIZE,):
        raise ValueError(f"{name}: expected length {STATE_SIZE}, got {arr.shape}")
    if arr.min(initial=0) < 0 or arr.max(initial=0) > 255:
        raise ValueError(f"{name}: values must fit uint8")
    return arr


def get_central_state() -> np.ndarray:
    return validate_state(load_puzzle_info()["central_state"], name="central_state")


def get_central_state_u8() -> np.ndarray:
    return get_central_state().astype(np.uint8)


def pad_state128_u8(state120: np.ndarray) -> np.ndarray:
    state = np.asarray(state120, dtype=np.uint8)
    if state.shape != (STATE_SIZE,):
        raise ValueError(f"state120 shape must be ({STATE_SIZE},), got {state.shape}")
    out = np.zeros((STATE_STORAGE_SIZE,), dtype=np.uint8)
    out[:STATE_SIZE] = state
    return out


def pad_states128_u8(states120: np.ndarray) -> np.ndarray:
    states = np.asarray(states120, dtype=np.uint8)
    if states.ndim != 2 or states.shape[1] != STATE_SIZE:
        raise ValueError(f"states120 shape must be (N,{STATE_SIZE}), got {states.shape}")
    out = np.zeros((states.shape[0], STATE_STORAGE_SIZE), dtype=np.uint8)
    out[:, :STATE_SIZE] = states
    return out


def get_central_state128_u8() -> np.ndarray:
    return pad_state128_u8(get_central_state_u8())


def get_generators() -> Dict[str, np.ndarray]:
    info = load_puzzle_info()
    return {name: validate_permutation(info["generators"][name], name=name) for name in ACTION_NAMES}


def get_action_table_u8() -> bytes:
    generators = get_generators()
    table = np.stack([generators[name] for name in ACTION_NAMES], axis=0).astype(np.uint8)
    if table.shape != (FANOUT, STATE_SIZE):
        raise AssertionError(f"bad action table shape: {table.shape}")
    return table.tobytes(order="C")


def get_action_table128_u8() -> bytes:
    generators = get_generators()
    table120 = np.stack([generators[name] for name in ACTION_NAMES], axis=0).astype(np.uint8)
    if table120.shape != (FANOUT, STATE_SIZE):
        raise AssertionError(f"bad action table shape: {table120.shape}")
    table128 = np.empty((FANOUT, STATE_STORAGE_SIZE), dtype=np.uint8)
    table128[:, :STATE_SIZE] = table120
    table128[:, STATE_SIZE:STATE_STORAGE_SIZE] = np.arange(STATE_SIZE, STATE_STORAGE_SIZE, dtype=np.uint8)
    return table128.tobytes(order="C")


def action_index(name: str) -> int:
    return ACTION_NAMES.index(name)


def inverse_action_name(name: str) -> str:
    return name[1:] if name.startswith("-") else f"-{name}"


def apply_action_cpu(state: np.ndarray, action: str | int) -> np.ndarray:
    state_u8 = np.asarray(state, dtype=np.uint8)
    if state_u8.shape != (STATE_SIZE,):
        raise ValueError(f"state shape must be ({STATE_SIZE},), got {state_u8.shape}")
    generators = get_generators()
    name = ACTION_NAMES[int(action)] if isinstance(action, int) else action
    perm = generators[name]
    return state_u8[perm].astype(np.uint8)


def apply_actions_cpu(state: np.ndarray, actions: Iterable[str | int]) -> np.ndarray:
    out = np.asarray(state, dtype=np.uint8)
    for action in actions:
        out = apply_action_cpu(out, action)
    return out


def validate_inverse_pairs() -> None:
    central = get_central_state_u8()
    for name in ACTION_NAMES:
        inv = inverse_action_name(name)
        if inv not in ACTION_NAMES:
            raise ValueError(f"inverse action absent for {name}: {inv}")
        restored = apply_actions_cpu(central, [name, inv])
        if not np.array_equal(restored, central):
            raise ValueError(f"inverse pair failed on central state: {name}, {inv}")


def load_test_puzzles(max_puzzles: int | None = None) -> List[Tuple[int, np.ndarray]]:
    path = get_data_dir() / "test.csv"
    puzzles: List[Tuple[int, np.ndarray]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"initial_state_id", "initial_state"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"test.csv fields must be {sorted(expected)}, got {reader.fieldnames}")
        for i, row in enumerate(reader):
            if max_puzzles is not None and i >= max_puzzles:
                break
            puzzle_id = int(row["initial_state_id"])
            state = validate_state((int(x) for x in row["initial_state"].split(",")), name=f"test[{puzzle_id}]")
            puzzles.append((puzzle_id, state.astype(np.uint8)))
    return puzzles


def load_sample_submission() -> List[Dict[str, Any]]:
    path = get_data_dir() / "sample_submission.csv"
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mix64(x: int) -> int:
    x &= (1 << 64) - 1
    x ^= x >> 33
    x = (x * 0xff51afd7ed558ccd) & ((1 << 64) - 1)
    x ^= x >> 33
    x = (x * 0xc4ceb9fe1a85ec53) & ((1 << 64) - 1)
    x ^= x >> 33
    return x & ((1 << 64) - 1)


def hash_state_cpu(state: np.ndarray) -> int:
    s = np.asarray(state, dtype=np.uint8)
    if s.shape != (STATE_SIZE,):
        raise ValueError(f"state shape must be ({STATE_SIZE},), got {s.shape}")
    h = 1469598103934665603
    for v in s.tolist():
        h ^= int(v)
        h = (h * 1099511628211) & ((1 << 64) - 1)
    h = _mix64(h)
    if h in (HASH_EMPTY, HASH_BUSY, HASH_TOMBSTONE):
        h += 4
    return h


def owner_rank_for_state(state: np.ndarray, world_size: int) -> int:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    return int(hash_state_cpu(state) % world_size)


def print_puzzle_info_summary() -> None:
    validate_inverse_pairs()
    central_state = get_central_state()
    generators = get_generators()
    test_puzzles = load_test_puzzles(max_puzzles=5)
    print("=" * 60)
    print("Puzzle Information Summary")
    print("=" * 60)
    print(f"Central state size: {len(central_state)}")
    print(f"Central state first 10: {central_state[:10].tolist()}")
    print(f"Number of generators: {len(generators)}")
    print(f"Generator names: {ACTION_NAMES}")
    print(f"Loaded test puzzle sample: {len(test_puzzles)}")
    if test_puzzles:
        puzzle_id, state = test_puzzles[0]
        print(f"First test puzzle ID: {puzzle_id}")
        print(f"First test puzzle state first 10: {state[:10].tolist()}")
    print("=" * 60)


if __name__ == "__main__":
    print_puzzle_info_summary()
