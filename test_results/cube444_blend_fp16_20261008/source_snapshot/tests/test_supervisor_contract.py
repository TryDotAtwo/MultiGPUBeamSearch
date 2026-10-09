"""Pure supervisor input validation, runnable without POSIX/GPU access."""
import json
import pytest
from tools.cube4_run_supervisor import load_contract


def test_unsolved_verification_cannot_finish_after_deadline(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools import cube4_run_supervisor as supervisor
    ticks = iter([0., 2.])
    monkeypatch.setattr(supervisor, 'time', SimpleNamespace(monotonic=lambda: next(ticks)))
    (tmp_path / 'rank-status').mkdir()
    (tmp_path / 'rank-status/rank-0.json').write_text(json.dumps(
        {'rank': 0, 'puzzle_id': 7, 'exit_code': 0, 'status': 'unsolved', 'completed_depths': 1}))
    with pytest.raises(TimeoutError):
        supervisor.verify_rank_result(tmp_path, 1, {'puzzle_id': 7, 'requested_depth': 1}, deadline=1.)


@pytest.mark.parametrize('row', ['7,"unterminated\n', '7,' + 's' * 140000 + '\n'],
                         ids=['unterminated_quote', 'oversized_field'])
def test_csv_parser_errors_are_typed_verification_failures(tmp_path, row):
    from tools.cube4_run_supervisor import verify_rank_result
    (tmp_path / 'rank-status').mkdir()
    (tmp_path / 'rank-status/rank-0.json').write_text(json.dumps(
        {'rank': 0, 'puzzle_id': 7, 'exit_code': 0, 'status': 'solved'}))
    (tmp_path / 'submit.csv').write_text('initial_state_id,path\n' + row)
    contract = {'puzzle_id': 7, 'initial_state': [1, 0], 'central_state': [0, 1],
                'generators': {'swap': [1, 0]}}
    with pytest.raises(ValueError):
        verify_rank_result(tmp_path, 1, contract)


def test_extra_csv_fields_cannot_be_promoted_to_solved(tmp_path):
    from tools.cube4_run_supervisor import verify_rank_result
    (tmp_path / 'rank-status').mkdir()
    (tmp_path / 'rank-status/rank-0.json').write_text(json.dumps(
        {'rank': 0, 'puzzle_id': 7, 'exit_code': 0, 'status': 'solved'}))
    (tmp_path / 'submit.csv').write_text('initial_state_id,path\n7,swap,unexpected\n')
    contract = {'puzzle_id': 7, 'initial_state': [1, 0], 'central_state': [0, 1],
                'generators': {'swap': [1, 0]}}
    with pytest.raises(ValueError, match='submission schema'):
        verify_rank_result(tmp_path, 1, contract)


@pytest.mark.parametrize('generators', [[], {'swap': [True, False]},
    {'swap': [1.0, 0.0]}, {'a.b': [1, 0]}, {'a b': [1, 0]}])
def test_invalid_move_table_rejected_before_process_launch(tmp_path, generators):
    path = tmp_path / 'contract.json'
    path.write_text(json.dumps({'puzzle_id': 7, 'initial_state': [1, 0],
        'central_state': [0, 1], 'generators': generators}))
    with pytest.raises(ValueError):
        load_contract(path)


def test_unreachable_symbol_multiset_rejected_before_process_launch(tmp_path):
    path = tmp_path / 'contract.json'
    path.write_text(json.dumps({'puzzle_id': 7, 'initial_state': [1, 1],
        'central_state': [0, 1], 'generators': {'swap': [1, 0]}}))
    with pytest.raises(ValueError):
        load_contract(path)


def test_valid_contract_retains_raw_identity_and_move_orientation(tmp_path):
    path = tmp_path / 'contract.json'
    raw = b'{"puzzle_id":7,"initial_state":[1,0],"central_state":[0,1],"generators":{"swap":[1,0]}}'
    path.write_bytes(raw)
    contract, digest = load_contract(path)
    assert contract['generators'] == {'swap': [1, 0]}
    import hashlib
    assert digest == hashlib.sha256(raw).hexdigest()
