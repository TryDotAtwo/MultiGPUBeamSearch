"""Full row-major score comparison. Lower scores rank first; no first-row shortcut."""
from __future__ import annotations
import math
import re
from hashlib import sha256
from pathlib import Path


def observe_score_identity(runner, model_manifest, reference_fixture, resolved_config, hardware):
    """Hash actual immutable inputs; callers must prevent mutation during use.

    Does not validate file schemas or attest that a supplied scorer ran them.
    """
    result = {}
    keys = ('runner_sha256', 'model_manifest_sha256', 'reference_fixture_sha256',
            'resolved_config_sha256', 'hardware_sha256')
    for key, path in zip(keys, (runner, model_manifest, reference_fixture, resolved_config, hardware)):
        digest = sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1048576), b''):
                digest.update(block)
        result[key] = digest.hexdigest()
    return result


def compare_bound_evidence(evidence, current_identity, policy):
    """Validate caller-observed identity and recompute numerical evidence.

    Caller must independently measure identity/policy; receipt claims are not
    observations. This does not certify reference provenance or solve quality.
    """
    if not isinstance(evidence, dict):
        raise ValueError('expected numerical evidence object')
    keys = {'runner_sha256', 'model_manifest_sha256', 'reference_fixture_sha256',
            'resolved_config_sha256', 'hardware_sha256'}
    for identity in (current_identity, evidence.get('identity')):
        if not isinstance(identity, dict) or set(identity) != keys:
            raise ValueError('incomplete numerical evidence identity')
        if any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)
               for value in identity.values()):
            raise ValueError('invalid numerical evidence identity hash')
    if evidence['identity'] != current_identity:
        raise ValueError('stale numerical evidence identity')
    if evidence.get('status') != 'pass':
        raise ValueError('numerical evidence did not pass')
    result = compare_scores(evidence.get('reference_scores'), evidence.get('actual_scores'),
                            **policy)
    result.update(identity=dict(current_identity), numerical_accepted=result['status'] == 'pass')
    return result


def _matrix(value):
    if not isinstance(value, list) or not value or not isinstance(value[0], list) or not value[0]:
        raise ValueError('expected nonempty rectangular score matrix')
    width = len(value[0])
    result = []
    for row in value:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError('ragged score matrix')
        if any(type(x) not in (int, float) for x in row):
            raise ValueError('score must be a real number, not bool/string')
        row = [float(x) for x in row]
        if not all(math.isfinite(x) for x in row):
            raise ValueError('nonfinite score')
        result.append(row)
    return result


def compare_graph_lane_scores(reference, artifact, *, microbatch, lanes):
    """Compare the requested native lane diagnostic, never a production gate.

    The probe repeats eight fixture states. Row order is lane then active row;
    derive it from the requested profile, not untrusted artifact claims.
    """
    if (type(microbatch) is not int or not 1 <= microbatch <= 65536 or
        type(lanes) is not int or not 1 <= lanes <= 64):
        raise ValueError('invalid requested graph lane profile')
    if (not isinstance(artifact, dict) or
        set(artifact) != {'microbatch', 'lane_count', 'parent_bases', 'active_counts', 'scores'} or
        type(artifact['microbatch']) is not int or artifact['microbatch'] != microbatch or
        type(artifact['lane_count']) is not int or artifact['lane_count'] != lanes):
        raise ValueError('graph lane artifact differs from requested profile')
    bases = [lane % microbatch for lane in range(lanes)]
    counts = [microbatch - base for base in bases]
    for field, expected in (('parent_bases', bases), ('active_counts', counts)):
        values = artifact[field]
        if (not isinstance(values, list) or len(values) != lanes or
            any(type(value) is not int for value in values) or values != expected):
            raise ValueError('invalid graph lane row layout: ' + field)
    reference = _matrix(reference)
    if len(reference) != 8 or len(reference[0]) != 24:
        raise ValueError('graph lane reference requires eight 24-score rows')
    actual = artifact['scores']
    if (not isinstance(actual, list) or len(actual) != sum(counts) or
        any(not isinstance(row, list) or len(row) != 24 for row in actual)):
        return {'status': 'failed', 'reason': 'score shape mismatch',
                'production_quality_accepted': False}
    expected = [reference[(base + row) % 8]
                for base, count in zip(bases, counts) for row in range(count)]
    return compare_scores(expected, actual, atol=.05, rtol=.001,
                          top_k=4, min_topk_overlap=.75)


def compare_scores(reference, actual, *, atol, rtol, top_k, min_topk_overlap):
    """All elements must satisfy |actual-ref| <= atol+rtol*|ref|.

    Each row must also meet the declared top-k overlap threshold. Ties break by
    move index, explicitly and deterministically. A numerical pass is not a
    solved-puzzle/replay quality claim. Policies must be chosen before inspection.
    """
    for name, value in [('atol', atol), ('rtol', rtol), ('min_topk_overlap', min_topk_overlap)]:
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid ' + name)
    if min_topk_overlap > 1 or type(top_k) is not int or top_k < 1:
        raise ValueError('invalid ranking policy')
    try:
        reference = _matrix(reference)
        actual = _matrix(actual)
    except (ValueError, OverflowError) as error:
        return {'status': 'failed', 'reason': str(error), 'production_quality_accepted': False}
    width = len(reference[0])
    if top_k > width:
        raise ValueError('top_k exceeds row width')
    if len(reference) != len(actual) or width != len(actual[0]):
        return {'status': 'failed', 'reason': 'score shape mismatch', 'production_quality_accepted': False}
    worst, max_error, violations, mean_error, seen = [0, 0], 0., 0, 0., 0
    overlaps, top1 = [], 0
    for row_index, (expected, observed) in enumerate(zip(reference, actual)):
        for column, (a, b) in enumerate(zip(expected, observed)):
            error = abs(a - b)
            limit = atol + rtol * abs(a)
            if not math.isfinite(error) or not math.isfinite(limit):
                return {'status': 'failed', 'reason': 'comparison arithmetic overflow',
                        'worst_index': [row_index, column], 'production_quality_accepted': False}
            if error > max_error:
                max_error, worst = error, [row_index, column]
            seen += 1
            mean_error += (error - mean_error) / seen
            violations += error > limit
        left = sorted(range(width), key=lambda i: (expected[i], i))
        right = sorted(range(width), key=lambda i: (observed[i], i))
        overlaps.append(len(set(left[:top_k]) & set(right[:top_k])) / top_k)
        top1 += left[0] == right[0]
    count = len(reference) * width
    return {
        'status': 'pass' if violations == 0 and min(overlaps) >= min_topk_overlap else 'failed',
        'scope': 'full score tensor numeric and per-row ranking comparison',
        'production_quality_accepted': False,
        'rows': len(reference), 'columns': width, 'elements_compared': count,
        'numeric_violations': violations, 'max_abs_error': max_error,
        'mean_abs_error': mean_error, 'worst_index': worst,
        'min_topk_overlap_observed': min(overlaps),
        'mean_topk_overlap': sum(overlaps) / len(overlaps),
        'top1_agreement': top1 / len(reference),
        'policy': {'atol': atol, 'rtol': rtol, 'top_k': top_k,
                   'min_topk_overlap': min_topk_overlap, 'tie_break': 'move_index'},
    }


def compare_production_raw_jobs(reference, artifact, *, outer_microbatch, lanes,
                                atol, rtol, top_k, min_topk_overlap):
    """Independent complete actual-runner job comparison, NOT admission.

    Reject malformed/missing coverage; return numerical/ranking failure metrics
    for well-formed data. Derive expected mappings from requested profile rather
    than trusting producer row labels. Execution/content identity is a separate
    mandatory preflight responsibility.
    """
    if (type(outer_microbatch) is not int or not 1 <= outer_microbatch <= 65536 or
            type(lanes) is not int or not 1 <= lanes <= 64 or
            2 * outer_microbatch * lanes * 24 > 4194304):
        raise ValueError('invalid production raw-score coverage budget')
    reference = _matrix(reference)
    if len(reference) > 1024 or len(reference[0]) != 24:
        raise ValueError('invalid production raw-score reference shape')
    if (not isinstance(artifact, dict) or set(artifact) != {'schema_version', 'jobs'} or
            type(artifact['schema_version']) is not int or artifact['schema_version'] != 1 or
            not isinstance(artifact['jobs'], list) or len(artifact['jobs']) != 4 * lanes):
        raise ValueError('invalid production raw-score artifact schema')
    cases = {'full': (0, outer_microbatch), 'partial': (1, outer_microbatch - 1),
             'singleton': (outer_microbatch, 1), 'zero': (0, 0)}
    seen, expected, actual = set(), [], []
    fields = {'lane', 'case', 'parent_base', 'active_rows', 'reference_rows', 'raw_scores'}
    for job in artifact['jobs']:
        if (not isinstance(job, dict) or set(job) != fields or
                type(job['lane']) is not int or not 0 <= job['lane'] < lanes or
                not isinstance(job['case'], str) or job['case'] not in cases):
            raise ValueError('invalid production raw-score job schema')
        key = (job['lane'], job['case'])
        if key in seen:
            raise ValueError('duplicate production raw-score job')
        seen.add(key)
        base, count = cases[job['case']]
        if (type(job['parent_base']) is not int or job['parent_base'] != base or
                type(job['active_rows']) is not int or job['active_rows'] != count):
            raise ValueError('production raw-score job differs from requested coverage')
        mapping = [(base + row) % len(reference) for row in range(count)]
        supplied = job['reference_rows']
        if (not isinstance(supplied, list) or any(type(x) is not int for x in supplied) or
                supplied != mapping):
            raise ValueError('production raw-score row mapping mismatch')
        rows = job['raw_scores']
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError('production raw-score active row count mismatch')
        if count:
            rows = _matrix(rows)
            if len(rows[0]) != 24:
                raise ValueError('production raw-score move count mismatch')
            expected.extend(reference[index] for index in mapping)
            actual.extend(rows)
    if len(seen) != 4 * lanes:
        raise ValueError('incomplete production raw-score job coverage')
    result = compare_scores(expected, actual, atol=atol, rtol=rtol, top_k=top_k,
                            min_topk_overlap=min_topk_overlap)
    result.update(jobs_compared=len(seen),
                  scope='actual runner full raw-job numeric and ranking comparison; not admission',
                  production_quality_accepted=False)
    return result
