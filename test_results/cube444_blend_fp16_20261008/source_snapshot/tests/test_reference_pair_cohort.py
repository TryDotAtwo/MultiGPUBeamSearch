import pytest


def contract():
    return {'central_state': [0, 1, 2], 'generators': {'r': [1, 2, 0], 'l': [2, 0, 1]},
            'cases': [{'puzzle_id': 7, 'initial_state': [1, 2, 0], 'solution': 'r.r'}]}


def test_pairs_use_algebraic_reflection_not_sticker_mirror():
    from tools.reference_pair_cohort import prepare_reference_pairs
    states, provenance = prepare_reference_pairs(contract(), state_len=3, num_classes=3)
    assert states == [[1, 2, 0], [2, 0, 1]]
    assert [row['variant'] for row in provenance['rows']] == ['original', 'reflected']
    assert provenance['rows'][1]['source_solution'] == 'r.r'
    assert provenance['replay_validated'] is True


def test_invalid_original_solution_cannot_create_reflected_reference():
    from tools.reference_pair_cohort import prepare_reference_pairs
    value = contract()
    value['cases'][0]['solution'] = 'r'
    with pytest.raises(ValueError, match='source solution'):
        prepare_reference_pairs(value, state_len=3, num_classes=3)


@pytest.mark.parametrize('mutation', ['state', 'variant', 'path', 'digest', 'missing-contract'])
def test_pair_evidence_rejects_tampering_despite_replay_boolean(mutation):
    from tools.reference_pair_cohort import prepare_reference_pairs, validate_reference_pair_evidence
    states, provenance = prepare_reference_pairs(contract(), state_len=3, num_classes=3)
    reference = {'states': states, 'metadata': {
        'state_generation': 'replay_bound_original_reflected_pairs', 'pair_provenance': provenance}}
    validate_reference_pair_evidence(reference, state_len=3, num_classes=3)
    if mutation == 'state':
        states[1] = states[0][:]
    elif mutation == 'variant':
        provenance['rows'][1]['variant'] = 'original'
    elif mutation == 'path':
        provenance['rows'][1]['source_solution'] = 'l'
    elif mutation == 'digest':
        provenance['contract_sha256'] = '0' * 64
    else:
        del provenance['source_contract']
    with pytest.raises(ValueError, match='reference pair'):
        validate_reference_pair_evidence(reference, state_len=3, num_classes=3)


def test_cube4_numeric_gate_replays_paired_reference():
    from tools.reference_pair_cohort import prepare_reference_pairs
    from tools.cube4_numeric_gate import validate_reference
    value = contract()
    value['central_state'] += [0] * 93
    value['cases'][0]['initial_state'] += [0] * 93
    for move in value['generators'].values():
        move += list(range(3, 96))
    states, provenance = prepare_reference_pairs(value, state_len=96, num_classes=6)
    reference = {'states': states, 'scores_fp32': [[0.] * 24, [1.] * 24],
        'metadata': {'state_generation': 'replay_bound_original_reflected_pairs',
                     'pair_provenance': provenance}}
    assert validate_reference(reference) == [[0.] * 24, [1.] * 24]
    reference['states'][1] = reference['states'][0][:]
    with pytest.raises(ValueError, match='independent replay'):
        validate_reference(reference)


@pytest.mark.parametrize('moves', [[[2, 0, 1], [1, 2, 0]], None])
def test_matching_generator_names_do_not_admit_wrong_or_missing_permutations(moves):
    from tools.reference_pair_cohort import validate_pair_generator_identity
    with pytest.raises(ValueError, match='generator'):
        validate_pair_generator_identity(contract(), {'names': ['r', 'l'], 'moves': moves}, ['r', 'l'])


def test_generator_identity_accepts_exact_ordered_permutations():
    from tools.reference_pair_cohort import validate_pair_generator_identity
    validate_pair_generator_identity(contract(),
        {'names': ['r', 'l'], 'moves': [[1, 2, 0], [2, 0, 1]]}, ['r', 'l'])


def test_generator_identity_accepts_named_source_permutations():
    from tools.reference_pair_cohort import validate_pair_generator_identity
    validate_pair_generator_identity(contract(),
        {'generators': {'l': [2, 0, 1], 'r': [1, 2, 0]}}, ['r', 'l'])


@pytest.mark.parametrize('names,moves', [
    (['l', 'r'], [[2, 0, 1], [1, 2, 0]]),
    (['r', 'l'], [[True, 2, 0], [2, 0, 1]]),
])
def test_generator_identity_rejects_order_and_bool_aliases(names, moves):
    from tools.reference_pair_cohort import validate_pair_generator_identity
    with pytest.raises(ValueError, match='generator'):
        validate_pair_generator_identity(contract(), {'moves': moves}, names)


def test_pair_writer_evaluates_distinct_inputs_and_preserves_row_binding(tmp_path, monkeypatch):
    import json
    import torch
    from tools import export_stream1_transformer as exporter

    observed = []

    class ReferenceModel(torch.nn.Module):
        def forward(self, states):
            observed.append(states.tolist())
            return torch.stack((states[:, 0] * 10 + states[:, 1],
                                states[:, 2] * 10 + states[:, 0]), dim=1).float()

    monkeypatch.setattr(exporter, 'instantiate_reference_model', lambda *args: ReferenceModel())
    weights = tmp_path / 'weights.pt'
    weights.write_bytes(b'fixture weights only')
    output = tmp_path / 'pairs.json'
    exporter.write_reference(output, {}, {}, tmp_path,
        dict(state_len=3, num_classes=3, output_dim=2), 2, 17, weights,
        reference_pair_contract=contract())
    payload = json.loads(output.read_text())
    assert observed == [[[1, 2, 0], [2, 0, 1]]]
    assert payload['states'] == [[1, 2, 0], [2, 0, 1]]
    assert payload['scores_fp32'] == [[12., 1.], [20., 12.]]
    assert payload['metadata']['state_generation'] == 'replay_bound_original_reflected_pairs'
    rows = payload['metadata']['pair_provenance']['rows']
    assert [(r['puzzle_id'], r['variant']) for r in rows] == [(7, 'original'), (7, 'reflected')]


@pytest.mark.parametrize('extra,count', [('count', 1), ('explicit', 2)])
def test_pair_writer_rejects_conflicting_inputs_before_model_creation(tmp_path, monkeypatch, extra, count):
    import torch
    from tools import export_stream1_transformer as exporter

    def unexpected_model(*args):
        pytest.fail('invalid pair request reached model creation')

    monkeypatch.setattr(exporter, 'instantiate_reference_model', unexpected_model)
    output = tmp_path / 'pairs.json'
    with pytest.raises(ValueError, match='shape/count|mutually exclusive'):
        exporter.write_reference(output, {}, {}, tmp_path,
            dict(state_len=3, num_classes=3, output_dim=2), count, 17, tmp_path / 'missing.pt',
            reference_pair_contract=contract(),
            reference_states=torch.zeros(2, 3, dtype=torch.int64) if extra == 'explicit' else None)
    assert not output.exists()
