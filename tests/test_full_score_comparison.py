import math
import pytest
from tools.full_score_comparison import compare_scores


def compare(reference, actual, **kwargs):
    return compare_scores(reference, actual, atol=kwargs.get('atol', 0.01),
                          rtol=kwargs.get('rtol', 0.0), top_k=kwargs.get('top_k', 1),
                          min_topk_overlap=kwargs.get('min_topk_overlap', 1.0))


def test_late_row_error_cannot_hide_behind_identical_first_row():
    result = compare([[1., 2.], [3., 4.]], [[1., 2.], [3., 8.]])
    assert result['status'] == 'failed'
    assert result['numeric_violations'] == 1
    assert result['worst_index'] == [1, 1]


def test_within_tolerance_is_not_required_to_be_bit_exact():
    result = compare([[1., 2.], [3., 4.]], [[1.001, 2.], [3., 4.001]])
    assert result['status'] == 'pass'
    assert result['elements_compared'] == 4
    assert result['max_abs_error'] == pytest.approx(.001)


@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf])
def test_nonfinite_values_rejected_on_either_side(bad):
    for a, b in [([[1., bad]], [[1., 2.]]), ([[1., 2.]], [[1., bad]])]:
        assert compare(a, b)['status'] == 'failed'


@pytest.mark.parametrize('actual', [[], [[1.]], [[1., 2.], [3.]], [[True, 2.]], [['1', 2.]]])
def test_invalid_tensor_rejected(actual):
    assert compare([[1., 2.]], actual)['status'] == 'failed'


def test_rank_inversion_can_fail_even_inside_numeric_tolerance():
    result = compare([[1., 1.001]], [[1.002, 1.001]])
    assert result['numeric_violations'] == 0
    assert result['min_topk_overlap_observed'] == 0
    assert result['status'] == 'failed'


def test_topk_uses_smallest_scores_and_explicit_overlap_requirement():
    result = compare([[1., 2., 3.]], [[1., 3., 2.]], atol=2., top_k=2, min_topk_overlap=.5)
    assert result['status'] == 'pass'
    assert result['min_topk_overlap_observed'] == .5


@pytest.mark.parametrize('kwargs', [{'atol': -1}, {'rtol': math.nan}, {'top_k': 0},
                                  {'top_k': 3}, {'min_topk_overlap': 1.1}])
def test_invalid_comparison_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        compare([[1., 2.]], [[1., 2.]], **kwargs)


def test_intermediate_overflow_does_not_turn_inf_comparison_into_pass():
    result = compare([[1e308]], [[-1e308]], rtol=1e308)
    assert result['status'] == 'failed'


def test_report_metrics_stay_finite_when_error_sum_overflows():
    result = compare([[1e308, 1e308]], [[0., 0.]], atol=1e308)
    assert result['status'] == 'pass'
    assert math.isfinite(result['mean_abs_error'])
    assert result['mean_abs_error'] == 1e308
