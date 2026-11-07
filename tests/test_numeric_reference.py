"""Reject invalid reference scores before native producer execution."""
import pytest
from tools.cube4_numeric_gate import validate_reference


@pytest.mark.parametrize('digest', [None, '', 'b' * 64, True, 'A' * 64])
def test_checkpoint_binding_rejects_missing_invalid_or_different_identity(digest):
    from tools.cube4_numeric_gate import validate_checkpoint_binding
    with pytest.raises(ValueError, match='checkpoint identity'):
        validate_checkpoint_binding({'metadata': {'source_weights_sha256': digest}},
                                    {'source_weights_sha256': 'a' * 64})


def test_checkpoint_binding_accepts_equal_observed_hashes():
    from tools.cube4_numeric_gate import validate_checkpoint_binding
    validate_checkpoint_binding({'metadata': {'source_weights_sha256': 'a' * 64}},
                                {'source_weights_sha256': 'a' * 64})


@pytest.mark.parametrize('value', [float('nan'), float('inf'), True, '1', 10**400])
def test_reference_rejects_invalid_last_score(value):
    reference = {'states': [[0] * 96], 'scores_fp32': [list(range(24))]}
    reference['scores_fp32'][0][-1] = value
    with pytest.raises(ValueError, match='invalid reference scores'):
        validate_reference(reference)
