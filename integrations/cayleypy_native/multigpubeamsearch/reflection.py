"""Solution-based reflection, always checked in the caller's original frame."""
from dataclasses import replace
from .contracts import GraphContract


def reflected_search(search, graph, kwargs, backend):
    kwargs = dict(kwargs)
    mode = kwargs.pop('reflect_mode', 'off')
    source = kwargs.pop('reflection_source', None)
    if mode not in ('off', 'after_original', 'only'):
        raise ValueError('reflect_mode must be off, after_original or only')
    if mode == 'off':
        return search(kwargs)
    if backend != 'native':
        raise ValueError('reflection requires backend=native')
    if kwargs.get('return_path', False) is not True:
        raise ValueError('reflection requires return_path=True')
    contract = GraphContract.from_graph(graph, kwargs['start_state'])
    inverses = []
    for perm in contract.generators:
        inverse = [0]*len(perm)
        for i, value in enumerate(perm): inverse[value] = i
        try: inverses.append(contract.generators.index(tuple(inverse)))
        except ValueError: raise ValueError('reflection requires inverse-closed generators') from None
    original = search(kwargs) if mode == 'after_original' else None
    if source is None and original is not None and original.path_found:
        source = original.path
    if source is None:
        if original is not None: return original
        raise ValueError('reflect_mode=only requires a verified reflection_source path')
    source = tuple(source)
    if any(type(move) is not int or not 0 <= move < contract.move_count for move in source) or not contract.replay(source):
        raise ValueError('reflection_source does not solve the original puzzle')
    state = list(contract.center)
    for move in source: state = [state[i] for i in contract.generators[move]]
    reflected = search(dict(kwargs, start_state=state))
    rows = [dict(row, final_orientation='original') for row in
            (original.native_metadata.get('collected_solutions', []) if original else [])]
    if original and original.path_found and not rows:
        rows = [dict(path=list(original.path), final_orientation='original')]
    for path in reflected.solutions:
        converted = tuple(inverses[move] for move in reversed(path))
        # Repeated-color graphs can have a nontrivial stabilizer. Never publish
        # a reflected path merely because it solved the transformed state.
        if not contract.replay(converted):
            raise ValueError('reflected solution does not replay in the original frame')
        rows.append(dict(path=list(converted), final_orientation='reflected',
                         searched_path=list(path), reflected_source_path=list(source)))
    unique = {tuple(row['path']): row for row in rows}
    rows = list(unique.values())
    best = min(unique, key=len) if unique else None
    metadata = dict(reflected.native_metadata, reflect_mode=mode, collected_solutions=rows,
                    reflection_runs=[r.native_metadata for r in (original, reflected) if r is not None],
                    replay_valid=True if best is not None else None)
    return replace(reflected, path_found=best is not None, path=list(best) if best is not None else None,
                   path_length=len(best) if best is not None else 0,
                   solutions=tuple(unique), native_metadata=metadata)
