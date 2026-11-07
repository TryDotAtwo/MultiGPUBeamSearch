"""Resolve a historical Cube4 parameter template; never grant GPU promotion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

PARAMETERS = frozenset(('H200_WORLD_SIZE', 'BEAM_WIDTH', 'BEAM_B_MICRO',
    'BEAM_STREAM1_TRANSFORMER_MICRO', 'BEAM_STREAM1_CONCURRENCY',
    'BEAM_STREAM3_RING_SLOTS', 'BEAM_SHARD_COUNT', 'BEAM_STREAM4_BATCH_ALIGNMENT',
    'BEAM_STREAM4_BATCH_CANDIDATES', 'BEAM_STREAM4_TRIGGER_CANDIDATES',
    'BEAM_STREAM4_ACTIVE_SORT_SLOTS'))
U64 = (1 << 64) - 1
U32 = (1 << 32) - 1


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate profile key: ' + key)
        result[key] = value
    return result


def bounded(value, maximum, label):
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError('invalid positive integer: ' + label)
    return value


def resolve_profile(path: Path, overrides: dict[str, str]) -> dict:
    raw = path.read_bytes()
    profile = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(profile, dict) or set(profile) != {
            'schema_version', 'profile_id', 'status', 'parameters', 'historical_evidence'}:
        raise ValueError('profile root schema mismatch')
    if type(profile['schema_version']) is not int or profile['schema_version'] != 1:
        raise ValueError('unsupported profile schema')
    if profile['status'] != 'historically_measured':
        raise ValueError('profile cannot assert current production acceptance')
    if not isinstance(profile['profile_id'], str) or not re.fullmatch(r'[a-z0-9_]+', profile['profile_id']):
        raise ValueError('invalid profile identity')
    parameters = profile['parameters']
    if not isinstance(parameters, dict) or set(parameters) != PARAMETERS:
        raise ValueError('missing/unknown profile parameter')
    parameters = dict(parameters)
    changed = {}
    for key, value in overrides.items():
        if key not in PARAMETERS or not isinstance(value, str) or not re.fullmatch(r'[0-9]+', value):
            raise ValueError('invalid override: ' + key)
        parsed = int(value)
        if parsed != parameters[key]:
            changed[key] = {'old': parameters[key], 'new': parsed}
        parameters[key] = parsed
    for key, value in parameters.items():
        bounded(value, U64 if key == 'BEAM_WIDTH' else U32, key)
    world = parameters['H200_WORLD_SIZE']
    if world > 8:
        raise ValueError('profile supports at most eight local GPUs')
    outer = parameters['BEAM_B_MICRO']
    if parameters['BEAM_STREAM1_TRANSFORMER_MICRO'] > outer:
        raise ValueError('inner micro exceeds outer batch')
    slots = parameters['BEAM_STREAM3_RING_SLOTS']
    if parameters['BEAM_STREAM1_CONCURRENCY'] > slots:
        raise ValueError('concurrency exceeds ring slots')
    batch = bounded(outer * 24 * slots, U32, 'Stream3 batch')
    alignment = bounded(world * parameters['BEAM_SHARD_COUNT'] *
                        parameters['BEAM_STREAM4_BATCH_ALIGNMENT'], U64, 'beam alignment')
    requested = parameters['BEAM_WIDTH']
    effective = bounded(((requested + alignment - 1) // alignment) * alignment, U64, 'effective beam')
    capacity = bounded(effective // world // parameters['BEAM_SHARD_COUNT'], U32, 'shard capacity')
    evidence = profile['historical_evidence']
    if not isinstance(evidence, dict) or evidence.get('numerical_quality_accepted') is not False:
        raise ValueError('historical evidence must retain unaccepted numerical quality')
    for key in ('archive_sha256', 'runner_sha256', 'model_manifest_sha256', 'puzzle_info_sha256', 'test_csv_sha256'):
        if not isinstance(evidence.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', evidence[key]):
            raise ValueError('invalid historical hash: ' + key)
    return {'profile_id': profile['profile_id'], 'profile_sha256': hashlib.sha256(raw).hexdigest(),
            'status': 'unvalidated_override' if changed else 'historically_measured',
            'promotion_allowed': False, 'parameters': parameters, 'overrides': changed,
            'derived': {'effective_beam': effective, 'shard_capacity': capacity,
                        'stream3_batch_candidates': batch, 'beam_alignment': alignment},
            'historical_evidence': evidence,
            'remaining_gates': ['current_build_identity', 'current_bundle_identity',
                                'hardware_identity', 'full_numerical_quality', 'bounded_pipeline']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--override', action='append', default=[])
    parser.add_argument('--environment', action='store_true',
                        help='Require every profile parameter in the actual launch environment')
    args = parser.parse_args()
    try:
        overrides = {}
        if args.environment:
            missing = PARAMETERS - os.environ.keys()
            if missing:
                raise ValueError('missing launch environment: ' + ','.join(sorted(missing)))
            overrides = {key: os.environ[key] for key in PARAMETERS}
        explicit = set()
        for item in args.override:
            key, separator, value = item.partition('=')
            if not separator or key in explicit:
                raise ValueError('malformed/duplicate override')
            explicit.add(key)
            overrides[key] = value
        print(json.dumps(resolve_profile(args.profile, overrides), indent=2))
        return 0
    except (ValueError, OSError, TypeError) as error:
        print('profile rejected: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
