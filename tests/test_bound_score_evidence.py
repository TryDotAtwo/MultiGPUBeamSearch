"""Recompute all scores and reject stale binary/model/config identities."""
import pytest


@pytest.mark.parametrize('evidence', [None, [], 'pass', 1, True])
def test_bound_gate_rejects_non_object_receipts(evidence):
    from tools.full_score_comparison import compare_bound_evidence
    with pytest.raises(ValueError, match='expected numerical evidence object'):
        compare_bound_evidence(evidence, identity(),
            dict(atol=0., rtol=0., top_k=1, min_topk_overlap=1.))


def identity():
    return {name: str(index) * 64 for index, name in enumerate(
        ('runner_sha256', 'model_manifest_sha256', 'reference_fixture_sha256',
         'resolved_config_sha256', 'hardware_sha256'), 1)}


@pytest.mark.parametrize('changed', range(5))
def test_observed_files_invalidate_old_evidence(tmp_path, changed):
    from tools.full_score_comparison import observe_score_identity, compare_bound_evidence
    files = [tmp_path / str(index) for index in range(5)]
    for path in files:
        path.write_bytes(b'original')
    observed = observe_score_identity(*files)
    evidence = {'identity': observed, 'status': 'pass',
        'reference_scores': [[1., 2.]], 'actual_scores': [[1., 2.]]}
    files[changed].write_bytes(b'replaced')
    with pytest.raises(ValueError, match='stale'):
        compare_bound_evidence(evidence, observe_score_identity(*files),
            dict(atol=0., rtol=0., top_k=1, min_topk_overlap=1.))


@pytest.mark.parametrize('kind', ['valid', 'last_row', 'stale', 'missing', 'skip'])
def test_bound_numeric_gate_recomputes_not_trusts_status(kind):
    from tools.full_score_comparison import compare_bound_evidence
    current = identity()
    evidence = {'identity': dict(current), 'status': 'pass',
        'reference_scores': [[1., 2.], [3., 4.]],
        'actual_scores': [[1., 2.], [3., 4.]]}
    if kind == 'last_row':
        evidence['actual_scores'][1][1] = 99.
    if kind == 'stale':
        evidence['identity']['runner_sha256'] = 'a' * 64
    if kind == 'missing':
        del evidence['identity']['model_manifest_sha256']
    if kind == 'skip':
        evidence['status'] = 'skip'
    policy = dict(atol=0., rtol=0., top_k=1, min_topk_overlap=1.)
    if kind in ('stale', 'missing', 'skip'):
        with pytest.raises(ValueError):
            compare_bound_evidence(evidence, current, policy)
    else:
        result = compare_bound_evidence(evidence, current, policy)
        assert result['numerical_accepted'] is (kind == 'valid')
        assert result['production_quality_accepted'] is False
        assert result['elements_compared'] == 4
