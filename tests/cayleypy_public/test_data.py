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