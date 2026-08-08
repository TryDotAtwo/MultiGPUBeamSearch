import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PuzzleContract:
    central_state: tuple[int, ...]
    generators: dict[str, tuple[int, ...]]
    initial_states: dict[int, tuple[int, ...]]
    sample_submission: pd.DataFrame
    state_len: int
    num_classes: int

    @property
    def move_names(self) -> tuple[str, ...]:
        return tuple(self.generators)

    @property
    def move_count(self) -> int:
        return len(self.generators)


def _state_from_cell(value: object) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise ValueError("initial_state must be a comma-separated string")
    try:
        return tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise ValueError("initial_state must contain comma-separated integers") from error


def _strict_json_integer_state(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(f"{name} must contain JSON integers")
    return tuple(value)


def _validate_state_classes(state: tuple[int, ...], num_classes: int, name: str) -> None:
    if any(value < 0 or value >= num_classes for value in state):
        raise ValueError(
            f"{name} values must be integers in [0, num_classes) for the supported permutation contract"
        )


def load_puzzle_contract(
    puzzle_info_path: Path,
    test_csv: Path,
    sample_submission_csv: Path,
    start: int,
    end: int,
) -> PuzzleContract:
    if start > end:
        raise ValueError("selected puzzle range must be non-empty")
    info = json.loads(puzzle_info_path.read_text(encoding="utf-8"))
    central_state = _strict_json_integer_state(info["central_state"], "central_state")
    state_len = len(central_state)
    if not 1 <= state_len <= 120:
        raise ValueError(
            "public runner requires 1 <= state_len <= 120 for the State128 logical payload"
        )
    _validate_state_classes(central_state, state_len, "central_state")

    generators = {
        name: _strict_json_integer_state(permutation, f"generator {name}")
        for name, permutation in info["generators"].items()
    }
    expected_permutation = set(range(state_len))
    for name, permutation in generators.items():
        if len(permutation) != state_len or set(permutation) != expected_permutation:
            raise ValueError(f"generator {name} must be a permutation of range(state_len)")

    selected_ids = tuple(range(start, end + 1))
    test_frame = pd.read_csv(test_csv)
    if "initial_state_id" not in test_frame or "initial_state" not in test_frame:
        raise ValueError("test CSV must contain initial_state_id and initial_state columns")
    initial_states: dict[int, tuple[int, ...]] = {}
    for puzzle_id in selected_ids:
        rows = test_frame.loc[test_frame["initial_state_id"] == puzzle_id]
        if len(rows) == 0:
            raise ValueError(f"missing selected test id {puzzle_id}")
        if len(rows) > 1:
            raise ValueError(f"duplicate selected test id {puzzle_id}")
        state = _state_from_cell(rows.iloc[0]["initial_state"])
        if len(state) != state_len:
            raise ValueError(f"state for id {puzzle_id} must have state_len {state_len}")
        _validate_state_classes(state, state_len, f"state for id {puzzle_id}")
        initial_states[puzzle_id] = state

    sample_submission = pd.read_csv(sample_submission_csv)
    if "initial_state_id" not in sample_submission:
        raise ValueError("sample submission must contain initial_state_id column")
    submission_ids = set(sample_submission["initial_state_id"])
    for puzzle_id in selected_ids:
        if puzzle_id not in submission_ids:
            raise ValueError(f"sample submission missing selected id {puzzle_id}")

    return PuzzleContract(
        central_state=central_state,
        generators=generators,
        initial_states=initial_states,
        sample_submission=sample_submission,
        state_len=state_len,
        num_classes=state_len,
    )