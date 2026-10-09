"""Replay-bound ordinary/reflected inputs; not numerical/model acceptance."""
from hashlib import sha256
import json

try:
    from tools.cayleypy_public.paths import apply_path, make_reflected_state
except ModuleNotFoundError:
    from cayleypy_public.paths import apply_path, make_reflected_state


def validate_pair_generator_identity(contract, generator, move_names):
    """Bind pair replay permutations to the selected export generator file."""
    pairs = contract.get('generators') if isinstance(contract, dict) else None
    if not isinstance(pairs, dict) or list(pairs) != move_names:
        raise ValueError('reference pair generator order differs from exported model')
    moves = generator.get('moves', generator.get('actions'))
    if moves is None and isinstance(generator.get('generators'), dict):
        source = generator['generators']
        if set(source) != set(move_names):
            raise ValueError('export generator names do not match permutations')
        moves = [source[name] for name in move_names]
    if not isinstance(moves, list) or len(moves) != len(move_names):
        raise ValueError('export generator requires explicit ordered permutations')
    for name, move in zip(move_names, moves):
        expected = pairs[name]
        if (not isinstance(move, list) or not isinstance(expected, list) or
                any(type(x) is not int for x in move + expected) or
                move != expected or sorted(move) != list(range(len(move)))):
            raise ValueError('reference pair generator permutation differs from export')


def prepare_reference_pairs(contract, *, state_len, num_classes):
    if not isinstance(contract, dict) or set(contract) != {'central_state', 'generators', 'cases'}:
        raise ValueError('reference pair contract schema mismatch')
    def state(value):
        if (not isinstance(value, list) or len(value) != state_len or
            any(type(x) is not int or not 0 <= x < num_classes for x in value)):
            raise ValueError('reference pair state domain mismatch')
        return value
    central = state(contract['central_state'])
    generators = contract['generators']
    apply_path(central, '', generators)
    cases = contract['cases']
    if not isinstance(cases, list) or not cases:
        raise ValueError('reference pairs require nonempty cases')
    states, rows, seen = [], [], set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {'puzzle_id', 'initial_state', 'solution'}:
            raise ValueError('reference pair case schema mismatch')
        identity = case['puzzle_id']
        if type(identity) is not int or not 0 <= identity < 2**64 or identity in seen:
            raise ValueError('invalid or duplicate reference puzzle identity')
        seen.add(identity)
        initial = state(case['initial_state'])
        path = case['solution']
        if not isinstance(path, str) or apply_path(initial, path, generators) != tuple(central):
            raise ValueError('reference source solution does not solve original')
        reflected = list(make_reflected_state(central, path, generators))
        digest = sha256(path.encode('utf-8')).hexdigest()
        for variant, value in [('original', initial), ('reflected', reflected)]:
            states.append(list(value))
            rows.append({'puzzle_id': identity, 'variant': variant,
                         'source_solution': path, 'source_solution_sha256': digest})
    contract_hash = sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode('utf-8')).hexdigest()
    return states, {'scope': 'replay-bound input pairs; not model quality',
                    'contract_sha256': contract_hash, 'replay_validated': True, 'rows': rows,
                    'source_contract': contract}


def validate_reference_pair_evidence(reference, *, state_len, num_classes):
    """Recompute paired inputs and provenance, never trust a replay boolean."""
    metadata = reference.get('metadata', {})
    if not isinstance(metadata, dict):
        raise ValueError('reference pair metadata must be an object')
    provenance = metadata.get('pair_provenance')
    paired = metadata.get('state_generation') == 'replay_bound_original_reflected_pairs'
    if provenance is None and not paired:
        return
    if not paired or not isinstance(provenance, dict):
        raise ValueError('reference pair provenance/generation mismatch')
    states, expected = prepare_reference_pairs(provenance.get('source_contract'),
        state_len=state_len, num_classes=num_classes)
    if provenance != expected or reference.get('states') != states:
        raise ValueError('reference pair evidence differs from independent replay')
