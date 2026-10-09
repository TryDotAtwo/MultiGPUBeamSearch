import json

import pandas as pd
import pytest

from tools.cayleypy_public.data import load_puzzle_contract


def write_fixtures(tmp_path, ids=(7, 8, 9)):
    puzzle_info = tmp_path / "puzzle_info.json"
    puzzle_info.write_text(json.dumps({
        "central_state": [0, 1, 2],
        "generators": {"swap01": [1, 0, 2], "rotate": [1, 2, 0]},
    }), encoding="utf-8")
    test_csv = tmp_path / "test.csv"
    pd.DataFrame({"initial_state_id": ids, "initial_state": ["0,1,2"] * len(ids)}).to_csv(test_csv, index=False)
    submission_csv = tmp_path / "sample_submission.csv"
    pd.DataFrame({"initial_state_id": [7, 8, 9], "path": ["", "", ""]}).to_csv(submission_csv, index=False)
    return puzzle_info, test_csv, submission_csv


def test_load_contract_derives_standard_dimensions(tmp_path):
    puzzle_info, test_csv, submission_csv = write_fixtures(tmp_path)
    contract = load_puzzle_contract(puzzle_info, test_csv, submission_csv, 7, 9)
    assert contract.central_state == (0, 1, 2)
    assert contract.move_names == ("swap01", "rotate")
    assert contract.move_count == 2
    assert contract.state_len == 3
    assert contract.num_classes == 3
    assert contract.initial_states == {7: (0, 1, 2), 8: (0, 1, 2), 9: (0, 1, 2)}
    assert contract.sample_submission["initial_state_id"].tolist() == [7, 8, 9]


def test_load_contract_derives_value_alphabet_independently_of_state_length(tmp_path):
    puzzle_info, test_csv, submission_csv = write_fixtures(tmp_path, ids=(7,))
    central = [value % 6 for value in range(96)]
    puzzle_info.write_text(json.dumps({
        "central_state": central,
        "generators": {"identity": list(range(96))},
    }), encoding="utf-8")
    pd.DataFrame({
        "initial_state_id": [7],
        "initial_state": [",".join(str(value) for value in central)],
    }).to_csv(test_csv, index=False)
    contract = load_puzzle_contract(puzzle_info, test_csv, submission_csv, 7, 7)
    assert contract.state_len == 96
    assert contract.num_classes == 6


def test_load_contract_rejects_state_len_above_state128_logical_capacity(tmp_path):
    puzzle_info, test_csv, submission_csv = write_fixtures(tmp_path, ids=(7,))
    oversized_state = list(range(121))
    puzzle_info.write_text(json.dumps({
        "central_state": oversized_state,
        "generators": {"identity": oversized_state},
    }), encoding="utf-8")
    pd.DataFrame({
        "initial_state_id": [7],
        "initial_state": [",".join(str(value) for value in oversized_state)],
    }).to_csv(test_csv, index=False)

    with pytest.raises(ValueError, match=r"1 <= state_len <= 120.*State128"):
        load_puzzle_contract(puzzle_info, test_csv, submission_csv, 7, 7)


def test_load_contract_rejects_missing_selected_state_id(tmp_path):
    puzzle_info, test_csv, submission_csv = write_fixtures(tmp_path, ids=(7, 9))
    with pytest.raises(ValueError, match="missing selected test id 8"):
        load_puzzle_contract(puzzle_info, test_csv, submission_csv, 7, 9)


def test_load_contract_rejects_duplicate_selected_state_id(tmp_path):
    puzzle_info, test_csv, submission_csv = write_fixtures(tmp_path, ids=(7, 8, 8, 9))
    with pytest.raises(ValueError, match="duplicate selected test id 8"):
        load_puzzle_contract(puzzle_info, test_csv, submission_csv, 7, 9)


def test_load_contract_rejects_non_permutation_generator(tmp_path):
    puzzle_info, test_csv, submission_csv = write_fixtures(tmp_path)
    puzzle_info.write_text(json.dumps({
        "central_state": [0, 1, 2], "generators": {"bad": [0, 0, 2]},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="permutation"):
        load_puzzle_contract(puzzle_info, test_csv, submission_csv, 7, 9)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([-1, 1, 2], r"central_state values must fit the State128 uint8 payload"),
        ([0, 1, 3], r"central_state labels must form the contiguous range"),
    ],
)
def test_load_contract_rejects_negative_or_overflow_central_state(tmp_path, value, message):
    puzzle_info, test_csv, submission_csv = write_fixtures(tmp_path)
    puzzle_info.write_text(json.dumps({
        "central_state": value,
        "generators": {"swap01": [1, 0, 2], "rotate": [1, 2, 0]},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_puzzle_contract(puzzle_info, test_csv, submission_csv, 7, 9)


@pytest.mark.parametrize("state", ["-1,1,2", "0,1,3"])
def test_load_contract_rejects_negative_or_overflow_initial_state(tmp_path, state):
    puzzle_info, test_csv, submission_csv = write_fixtures(tmp_path)
    pd.DataFrame({
        "initial_state_id": [7, 8, 9],
        "initial_state": [state, "0,1,2", "0,1,2"],
    }).to_csv(test_csv, index=False)

    with pytest.raises(ValueError, match=r"state for id 7 values must be integers in \[0, num_classes\)"):
        load_puzzle_contract(puzzle_info, test_csv, submission_csv, 7, 9)

@pytest.mark.parametrize(
    "value",
    [
        [True, 1, 2],
        [0, 1, 2.0],
    ],
)
def test_load_contract_rejects_non_integer_json_central_state(tmp_path, value):
    puzzle_info, test_csv, submission_csv = write_fixtures(tmp_path)
    puzzle_info.write_text(json.dumps({
        "central_state": value,
        "generators": {"swap01": [1, 0, 2], "rotate": [1, 2, 0]},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="central_state must contain JSON integers"):
        load_puzzle_contract(puzzle_info, test_csv, submission_csv, 7, 9)


def test_load_contract_rejects_float_generator_entry(tmp_path):
    puzzle_info, test_csv, submission_csv = write_fixtures(tmp_path)
    puzzle_info.write_text(json.dumps({
        "central_state": [0, 1, 2],
        "generators": {"swap01": [1, 0, 2.0]},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="generator swap01 must contain JSON integers"):
        load_puzzle_contract(puzzle_info, test_csv, submission_csv, 7, 9)
