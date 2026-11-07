"""Next production admission step: all actual runner jobs, not first-row metrics.

Test-first source; remote RED has not run yet. No production acceptance here.
"""
from copy import deepcopy
import math

import pytest
from tools import full_score_comparison


def fixture():
    reference = [[float(move + row * 30) for move in range(24)] for row in range(3)]
    jobs = []
    for lane in range(2):
        for name, base, count, mapping in [
                ('full', 0, 4, [0, 1, 2, 0]),
                ('partial', 1, 3, [1, 2, 0]),
                ('singleton', 4, 1, [1]), ('zero', 0, 0, [])]:
            jobs.append(dict(lane=lane, case=name, parent_base=base,
                             active_rows=count, reference_rows=mapping,
                             raw_scores=[list(reference[index]) for index in mapping]))
    return reference, dict(schema_version=1, jobs=jobs)


def compare(reference, artifact):
    function = getattr(full_score_comparison, 'compare_production_raw_jobs', None)
    assert callable(function), 'missing independent actual-runner job comparison'
    return function(reference, artifact, outer_microbatch=4, lanes=2,
                    atol=.05, rtol=.001, top_k=4, min_topk_overlap=.75)


def test_all_jobs_compared_without_promoting_candidate_to_admission():
    reference, artifact = fixture()
    result = compare(reference, artifact)
    assert result['status'] == 'pass'
    assert result['elements_compared'] == 384
    assert result['jobs_compared'] == 8
    assert result['production_quality_accepted'] is False


def test_error_in_last_lane_singleton_last_move_is_not_hidden():
    reference, artifact = fixture()
    artifact['jobs'][6]['raw_scores'][0][23] += 5
    result = compare(reference, artifact)
    assert result['status'] == 'failed'
    assert result['numeric_violations'] == 1


def test_ranking_failure_is_not_hidden_by_numeric_tolerance():
    reference, artifact = fixture()
    reference[0] = [0., .001, .002, .003, .004, .005, .006, .007] + [float(i) for i in range(8, 24)]
    for job in artifact['jobs']:
        job['raw_scores'] = [list(reference[index]) for index in job['reference_rows']]
    artifact['jobs'][0]['raw_scores'][0][:4] = [.02, .02, .02, .02]
    result = compare(reference, artifact)
    assert result['numeric_violations'] == 0
    assert result['status'] == 'failed'


@pytest.mark.parametrize('mutation', [
    'missing_lane', 'duplicate_job', 'extra_job', 'unknown_case', 'wrong_base',
    'wrong_count', 'wrong_mapping', 'bool_lane', 'bool_count', 'bool_mapping',
    'extra_field', 'extra_root_field', 'wrong_schema', 'zero_has_scores',
    'short_scores', 'short_row', 'nonfinite', 'bool_score',
])
def test_malformed_or_incomplete_job_coverage_rejected(mutation):
    reference, artifact = fixture()
    job = artifact['jobs'][0]
    if mutation == 'missing_lane': artifact['jobs'] = artifact['jobs'][:4]
    elif mutation == 'duplicate_job': artifact['jobs'][1] = deepcopy(job)
    elif mutation == 'extra_job': artifact['jobs'].append(deepcopy(job))
    elif mutation == 'unknown_case': job['case'] = 'warmup'
    elif mutation == 'wrong_base': job['parent_base'] = 1
    elif mutation == 'wrong_count': job['active_rows'] = 3
    elif mutation == 'wrong_mapping': job['reference_rows'][0] = 1
    elif mutation == 'bool_lane': job['lane'] = False
    elif mutation == 'bool_count': artifact['jobs'][2]['active_rows'] = True
    elif mutation == 'bool_mapping': job['reference_rows'][0] = False
    elif mutation == 'extra_field': job['accepted'] = True
    elif mutation == 'extra_root_field': artifact['accepted'] = True
    elif mutation == 'wrong_schema': artifact['schema_version'] = True
    elif mutation == 'zero_has_scores': artifact['jobs'][3]['raw_scores'] = [reference[0]]
    elif mutation == 'short_scores': job['raw_scores'].pop()
    elif mutation == 'short_row': job['raw_scores'][0].pop()
    elif mutation == 'nonfinite': job['raw_scores'][0][0] = math.nan
    elif mutation == 'bool_score': job['raw_scores'][0][0] = False
    with pytest.raises(ValueError):
        compare(reference, artifact)
