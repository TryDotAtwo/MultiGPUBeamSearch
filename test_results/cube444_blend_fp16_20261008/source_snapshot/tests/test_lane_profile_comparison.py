"""Independent fixtures for requested graph-lane diagnostic acceptance."""
import pytest

from tools import full_score_comparison


def fixture():
    reference = [[100 * row + move for move in range(24)] for row in range(8)]
    # Literal micro=2, lanes=3: bases 0,1,0; counts 2,1,2; rows 0,1,1,0,1.
    actual = dict(microbatch=2, lane_count=3, parent_bases=[0, 1, 0],
                  active_counts=[2, 1, 2], scores=[reference[i][:] for i in [0, 1, 1, 0, 1]])
    return reference, actual


def test_requested_profile_compares_every_active_score():
    reference, actual = fixture()
    result = full_score_comparison.compare_graph_lane_scores(reference, actual, microbatch=2, lanes=3)
    assert result['status'] == 'pass'
    assert result['elements_compared'] == 120
    assert result['production_quality_accepted'] is False


@pytest.mark.parametrize('field,value', [('microbatch', 4), ('lane_count', 2),
    ('parent_bases', [0, 0, 0]), ('active_counts', [2, 2, 2]),
    ('microbatch', True), ('parent_bases', [False, 1, 0])])
def test_artifact_cannot_override_requested_profile(field, value):
    reference, actual = fixture()
    actual[field] = value
    with pytest.raises(ValueError):
        full_score_comparison.compare_graph_lane_scores(reference, actual, microbatch=2, lanes=3)


def test_last_lane_last_move_is_not_omitted():
    reference, actual = fixture()
    actual['scores'][-1][-1] += 1
    result = full_score_comparison.compare_graph_lane_scores(reference, actual, microbatch=2, lanes=3)
    assert result['status'] == 'failed'
    assert result['numeric_violations'] == 1


def test_missing_raw_row_cannot_pass():
    reference, actual = fixture()
    actual['scores'].pop()
    result = full_score_comparison.compare_graph_lane_scores(reference, actual, microbatch=2, lanes=3)
    assert result['status'] == 'failed'


def test_missing_rows_rejected_before_expanding_large_expected_matrix(monkeypatch):
    reference, _ = fixture()
    actual = dict(microbatch=65536, lane_count=64, parent_bases=list(range(64)),
                  active_counts=[65536 - lane for lane in range(64)], scores=[])
    real_matrix = full_score_comparison._matrix

    def bounded_matrix(value):
        # Instrument actual conversion to catch a resource bug without allocating
        # millions of copied rows. Ordinary reference conversion still runs.
        if len(value) > 8:
            pytest.fail('expanded missing score rows before validating artifact shape')
        return real_matrix(value)

    monkeypatch.setattr(full_score_comparison, '_matrix', bounded_matrix)
    result = full_score_comparison.compare_graph_lane_scores(reference, actual,
                                                            microbatch=65536, lanes=64)
    assert result['status'] == 'failed'
    assert result['production_quality_accepted'] is False
