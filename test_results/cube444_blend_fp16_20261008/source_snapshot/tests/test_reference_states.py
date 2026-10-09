import pytest
import json
import torch
from tools.export_stream1_transformer import make_reference_states


def test_checkpoint_identity_uses_deserialized_snapshot(tmp_path, monkeypatch):
    import hashlib
    from tools import export_stream1_transformer as exporter

    checkpoint = tmp_path / 'weights.pt'
    torch.save({'state_dict': {'module.weight': torch.tensor([3.])}}, checkpoint)
    original = checkpoint.read_bytes()
    real_load = torch.load

    def replace_during_load(source, **kwargs):
        torch.save({'weight': torch.tensor([9.])}, checkpoint)
        return real_load(source, **kwargs)

    monkeypatch.setattr(exporter.torch, 'load', replace_during_load)
    state, digest = exporter.load_checkpoint_snapshot(checkpoint)
    assert state['weight'].tolist() == [3.]
    assert digest == hashlib.sha256(original).hexdigest()
    assert digest != hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_reference_preserves_loaded_identity_after_path_replacement(tmp_path, monkeypatch):
    from tools import export_stream1_transformer as exporter

    class ReferenceModel(torch.nn.Module):
        def forward(self, states):
            checkpoint.write_bytes(b'replacement')
            return torch.zeros(2, 24)

    monkeypatch.setattr(exporter, 'instantiate_reference_model', lambda *args: ReferenceModel())
    checkpoint = tmp_path / 'weights.pt'
    checkpoint.write_bytes(b'original')
    output = tmp_path / 'reference.json'
    digest = 'a' * 64
    exporter.write_reference(output, {}, {}, tmp_path,
        dict(output_dim=24, num_classes=2, state_len=4), 2, 123, checkpoint,
        reference_states=torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]]),
        source_weights_sha256=digest)
    assert json.loads(output.read_text())['metadata']['source_weights_sha256'] == digest


def test_explicit_reference_count_mismatch_is_rejected(tmp_path, monkeypatch):
    from tools import export_stream1_transformer as exporter

    class ReferenceModel(torch.nn.Module):
        def forward(self, states):
            return states[:, :1].float().repeat(1, 24)

    monkeypatch.setattr(exporter, 'instantiate_reference_model', lambda *args: ReferenceModel())
    checkpoint = tmp_path / 'weights.pt'
    checkpoint.write_bytes(b'identity')
    output = tmp_path / 'reference.json'
    with pytest.raises(ValueError, match='reference.*count'):
        exporter.write_reference(output, {}, {}, tmp_path,
            dict(output_dim=24, num_classes=2, state_len=4), 8, 123, checkpoint,
            reference_states=torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]]))
    assert not output.exists()


def test_explicit_reference_does_not_claim_generator_walk(tmp_path, monkeypatch):
    from tools import export_stream1_transformer as exporter

    class ReferenceModel(torch.nn.Module):
        def forward(self, states):
            return states[:, :1].float().repeat(1, 24)

    monkeypatch.setattr(exporter, 'instantiate_reference_model', lambda *args: ReferenceModel())
    checkpoint = tmp_path / 'weights.pt'
    checkpoint.write_bytes(b'identity')
    output = tmp_path / 'reference.json'
    exporter.write_reference(output, {}, {}, tmp_path,
        dict(output_dim=24, num_classes=2, state_len=4), 2, 123, checkpoint,
        reference_states=torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]]))
    payload = json.loads(output.read_text())
    assert payload['metadata']['state_generation'] == 'explicit_states'
    assert payload['metadata']['reference_count'] == 2
    assert payload['scores_fp32'] == [[0.] * 24, [1.] * 24]


@pytest.mark.parametrize('kind', ['missing-row', 'missing-move', 'nan', 'inf'])
def test_reference_rejects_invalid_model_outputs_before_writing(tmp_path, monkeypatch, kind):
    from tools import export_stream1_transformer as exporter

    class ReferenceModel(torch.nn.Module):
        def forward(self, states):
            if kind == 'missing-row':
                return torch.zeros(1, 24)
            if kind == 'missing-move':
                return torch.zeros(2, 23)
            scores = torch.zeros(2, 24)
            scores[-1, -1] = float(kind)
            return scores

    monkeypatch.setattr(exporter, 'instantiate_reference_model', lambda *args: ReferenceModel())
    checkpoint = tmp_path / 'weights.pt'
    checkpoint.write_bytes(b'identity')
    output = tmp_path / 'reference.json'
    with pytest.raises(ValueError, match='reference.*scores'):
        exporter.write_reference(output, {}, {}, tmp_path,
            dict(output_dim=24, num_classes=2, state_len=4), 2, 123, checkpoint,
            reference_states=torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]]))
    assert not output.exists()


def test_cube4_without_alphabet_and_moves_is_not_a_reference():
    with pytest.raises(ValueError):
        make_reference_states(96, 8, 123)


def test_reference_walk_preserves_classes_and_is_replayable():
    central = [0, 0, 1, 1]
    moves = [[1, 2, 3, 0]]
    states = make_reference_states(4, 8, 123, num_classes=2, central_state=central, moves=moves)
    reachable = {tuple(central[i:] + central[:i]) for i in range(4)}
    assert all(tuple(row) in reachable for row in states.tolist())
    assert torch.equal(states, make_reference_states(4, 8, 123, num_classes=2, central_state=central, moves=moves))
    assert states.dtype == torch.int64
    assert states.shape == (8, 4)


@pytest.mark.parametrize('moves', [[[0, 0, 2, 3]], [[0, 1, 2, 4]], [[0, 1, 2]], []])
def test_reference_walk_rejects_invalid_generator(moves):
    with pytest.raises(ValueError):
        make_reference_states(4, 8, 123, num_classes=2, central_state=[0, 0, 1, 1], moves=moves)


def test_generic_permutation_reference_remains_supported():
    states = make_reference_states(120, 2, 7)
    assert all(sorted(row) == list(range(120)) for row in states.tolist())
