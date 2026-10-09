"""Independent CPU permutation replay for bounded real-run smoke fixtures."""
from __future__ import annotations
import csv
from pathlib import Path


def apply_move(state, permutation):
    if len(permutation) != len(state) or sorted(permutation) != list(range(len(state))):
        raise ValueError('invalid permutation')
    return tuple(state[index] for index in permutation)


def prepare_three_move_case(generator, central):
    moves, names = generator['moves'], generator['move_names']
    if len(moves) != len(names) or len(set(names)) != len(names):
        raise ValueError('invalid move names')
    central = tuple(central)
    for move in moves:
        apply_move(central, move)
        inverse = [0] * len(central)
        for dst, src in enumerate(move):
            inverse[src] = dst
        if inverse not in moves:
            raise ValueError('minimum-depth proof requires inverse-closed move set')
    # Prove no solution shorter than three by exact CPU expansion from goal.
    nearby = {central}
    frontier = {central}
    for _ in range(2):
        frontier = {apply_move(state, move) for state in frontier for move in moves} - nearby
        nearby.update(frontier)
    for sequence in ((0, 8, 16), (0, 10, 18), (2, 8, 20)):
        state = central
        for index in sequence:
            state = apply_move(state, moves[index])
        if state in nearby:
            continue
        inverse_indices = []
        for index in reversed(sequence):
            inverse = [0] * len(central)
            for dst, src in enumerate(moves[index]):
                inverse[src] = dst
            inverse_indices.append(moves.index(inverse))
        result = {'puzzle_id': 910003, 'central': list(central), 'initial': list(state),
                  'scramble': [names[i] for i in sequence],
                  'known_solution': '.'.join(names[i] for i in inverse_indices),
                  'minimum_depth': 3, 'cpu_states_within_two': len(nearby)}
        replay(result['initial'], result['central'], generator, result['known_solution'])
        return result
    raise ValueError('could not construct a proven three-move fixture')


def replay(initial, target, generator, path):
    names, moves = generator['move_names'], generator['moves']
    if len(names) != len(moves) or len(set(names)) != len(names):
        raise ValueError('ambiguous move table')
    indices = {name: i for i, name in enumerate(names)}
    tokens = path.split('.') if path else []
    state = tuple(initial)
    for token in tokens:
        if token not in indices:
            raise ValueError('unknown/empty move token: ' + token)
        state = apply_move(state, moves[indices[token]])
    if state != tuple(target):
        raise ValueError('solution does not reach target')
    return len(tokens)


def verify_submission(path: Path, case, generator, depth_limit, *, bfs_radius=0):
    # Expansion iterations exclude the independently replayed BFS touch suffix.
    for value in (depth_limit, bfs_radius):
        if type(value) is not int or value < 0:
            raise ValueError('depth limit and BFS radius must be nonnegative integer values')
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['initial_state_id', 'path']:
            raise ValueError('submission schema mismatch')
        rows = list(reader)
    if len(rows) != 1 or rows[0]['initial_state_id'] != str(case['puzzle_id']):
        raise ValueError('submission puzzle/count mismatch')
    length = replay(case['initial'], case['central'], generator, rows[0]['path'])
    if not case['minimum_depth'] <= length <= depth_limit + bfs_radius:
        raise ValueError('solution length outside fixture bounds')
    return {'replay_valid': True, 'length': length, 'path': rows[0]['path']}
