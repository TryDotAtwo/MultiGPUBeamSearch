"""Recompute bounded full numerical comparison; never grant campaign promotion."""
import argparse
from hashlib import sha256
import json
import re
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cube4_profile import unique
from tools.full_score_comparison import compare_bound_evidence, compare_scores, observe_score_identity
from tools.reference_pair_cohort import validate_reference_pair_evidence
from tools.cube4_bundle_contract import validate_cube4_weights

POLICY = dict(atol=0.05, rtol=0.001, top_k=4, min_topk_overlap=0.75)


def validate_checkpoint_binding(reference, manifest):
    digest = manifest.get('source_weights_sha256')
    metadata = reference.get('metadata')
    if (not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None or
        not isinstance(metadata, dict) or metadata.get('source_weights_sha256') != digest):
        raise ValueError('reference checkpoint identity missing or mismatched')


def validate_reference(reference):
    validate_reference_pair_evidence(reference, state_len=96, num_classes=6)
    states = reference.get('states')
    expected = reference.get('scores_fp32')
    if (not isinstance(states, list) or not states or
        any(not isinstance(row, list) or len(row) != 96 or
            any(type(value) is not int or not 0 <= value < 6 for value in row) for row in states) or
        not isinstance(expected, list) or len(expected) != len(states) or
        any(not isinstance(row, list) or len(row) != 24 for row in expected)):
        raise ValueError('Cube4 numerical reference shape mismatch')
    checked = compare_scores(expected, expected, atol=0., rtol=0., top_k=1, min_topk_overlap=1.)
    if checked['status'] != 'pass':
        raise ValueError('invalid reference scores: ' + checked['reason'])
    return expected


def read_json(path):
    with path.open('rb') as stream:
        raw = stream.read(8388609)
    if len(raw) > 8388608:
        raise ValueError('numerical artifact exceeds 8MiB')
    value = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError('expected numerical artifact object')
    return value, sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runner', 'producer', 'manifest', 'reference', 'actual', 'config', 'hardware'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        identity = observe_score_identity(args.runner, args.manifest, args.reference,
                                          args.config, args.hardware)
        reference, reference_hash = read_json(args.reference)
        actual, actual_hash = read_json(args.actual)
        if set(actual) != {'scores'}:
            raise ValueError('native score dump schema mismatch')
        expected = validate_reference(reference)
        manifest, manifest_hash = read_json(args.manifest)
        validate_checkpoint_binding(reference, manifest)
        if manifest_hash != identity['model_manifest_sha256']:
            raise ValueError('manifest changed during observation')
        if reference_hash != identity['reference_fixture_sha256']:
            raise ValueError('reference changed during observation')
        prepared = validate_cube4_weights(args.manifest.parent)
        report = compare_bound_evidence({'status': 'pass', 'identity': identity,
            'reference_scores': expected, 'actual_scores': actual['scores']}, identity, POLICY)
        producer = observe_score_identity(args.producer, args.manifest, args.reference,
                                          args.config, args.hardware)
        if any(producer[key] != identity[key] for key in identity if key != 'runner_sha256'):
            raise ValueError('inputs changed during comparison')
        if observe_score_identity(args.runner, args.manifest, args.reference,
                                  args.config, args.hardware) != identity:
            raise ValueError('inputs changed during comparison')
        if validate_cube4_weights(args.manifest.parent) != prepared:
            raise ValueError('exported tensors changed during comparison')
        report.update(producer_sha256=producer['runner_sha256'], actual_scores_sha256=actual_hash,
            tensor_sha256=prepared['tensor_sha256'],
            provenance_attested=False, policy_scope='predeclared FP16 diagnostic policy; not solve quality')
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report['numerical_accepted'] else 2
    except (ValueError, OSError, KeyError, TypeError) as error:
        print('numerical gate rejected: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
