import pytest
from tools.beam_smoke_fixture import replay, verify_submission

GEN = {'move_names': ['r', '-r'], 'moves': [[1, 2, 0], [2, 0, 1]]}


def test_replay_checks_actual_final_state():
    assert replay([1, 2, 0], [0, 1, 2], GEN, '-r') == 1
    with pytest.raises(ValueError):
        replay([1, 2, 0], [0, 1, 2], GEN, 'r')


@pytest.mark.parametrize('path', ['unknown', '-r.', '.-r', '-r..r'])
def test_invalid_move_sequence_rejected(path):
    with pytest.raises(ValueError):
        replay([1, 2, 0], [0, 1, 2], GEN, path)


def test_submission_id_and_length_are_checked(tmp_path):
    file = tmp_path / 'submit.csv'
    case = {'puzzle_id': 7, 'initial': [1, 2, 0], 'central': [0, 1, 2], 'minimum_depth': 1}
    file.write_text('initial_state_id,path\n7,-r\n')
    assert verify_submission(file, case, GEN, 3)['replay_valid']
    file.write_text('initial_state_id,path\n8,-r\n')
    with pytest.raises(ValueError):
        verify_submission(file, case, GEN, 3)


def test_submission_allows_explicit_bfs_suffix(tmp_path):
    file = tmp_path / 'submit.csv'
    case = {'puzzle_id': 7, 'initial': [1, 2, 0], 'central': [0, 1, 2], 'minimum_depth': 1}
    file.write_text('initial_state_id,path\n7,r.r\n')
    assert verify_submission(file, case, GEN, 1, bfs_radius=1)['length'] == 2
    with pytest.raises(ValueError, match='outside fixture bounds'):
        verify_submission(file, case, GEN, 1)


@pytest.mark.parametrize('depth,radius', [(True, 0), (-1, 0), (1, True), (1, -1), (1, 0.5)])
def test_submission_rejects_invalid_depth_budget(tmp_path, depth, radius):
    file = tmp_path / 'submit.csv'
    case = {'puzzle_id': 7, 'initial': [1, 2, 0], 'central': [0, 1, 2], 'minimum_depth': 1}
    file.write_text('initial_state_id,path\n7,-r\n')
    with pytest.raises(ValueError, match='nonnegative integer'):
        verify_submission(file, case, GEN, depth, bfs_radius=radius)
