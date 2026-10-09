from pathlib import Path
import re

import pytest

from tools.stream1_execution_environment import observe_environment


def test_actual_source_inventory_preserves_unset_and_explicit_selectors():
    root = Path(__file__).resolve().parents[1]
    report = observe_environment(root, {'BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM': '1',
                                       'SECRET_TOKEN': 'must-not-appear'})
    assert report['selectors']['BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM'] == '1'
    assert report['selectors']['BEAM_STREAM1_TRANSFORMER_QKV_POLICY'] is None
    assert 'must-not-appear' not in str(report)
    assert report['production_quality_accepted'] is False
    assert 'cuda/stream1_transformer.cu' in report['selector_sources_sha256']
    launcher = (root / 'hpc/h200_dual_large_beam.sh').read_text()
    assert set(re.findall(r'BEAM_STREAM1_[A-Z0-9_]+', launcher)) <= report['selectors'].keys()
    assert 'tools/stream1_weight_io.hpp' in report['selector_sources_sha256']


def test_unknown_selector_rejected():
    with pytest.raises(ValueError, match='unknown native Stream1 selector'):
        observe_environment(Path(__file__).resolve().parents[1],
                            {'BEAM_STREAM1_TRANSFORMER_NOT_A_REAL_SELECTOR': '1'})


def test_source_change_changes_inventory_identity(tmp_path):
    directory = tmp_path / 'cuda'
    directory.mkdir()
    source = directory / 'example.cu'
    source.write_text('getenv("BEAM_STREAM1_EXECUTOR");')
    first = observe_environment(tmp_path, {})
    source.write_text('getenv("BEAM_STREAM1_EXECUTOR"); // changed')
    assert observe_environment(tmp_path, {})['selector_sources_sha256'] != first['selector_sources_sha256']
