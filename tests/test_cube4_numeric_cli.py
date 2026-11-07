"""Actual CLI consumes native score-dump and exported reference schemas."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
from test_cube4_bundle_contract import bundle


@pytest.mark.parametrize('kind', ['valid', 'last_row', 'skip', 'corrupt_tensor'])
def test_numeric_cli_recomputes_exported_fixture(tmp_path, bundle, kind):
    root = Path(__file__).resolve().parents[1]
    paths = {}
    for name in ('runner', 'producer', 'manifest', 'config', 'hardware'):
        paths[name] = tmp_path / name
        paths[name].write_bytes(b'fixture identity')
    paths['manifest'] = bundle[1] / 'manifest.json'
    model = json.loads(paths['manifest'].read_text())
    model['source_weights_sha256'] = 'a' * 64
    paths['manifest'].write_text(json.dumps(model))
    if kind == 'corrupt_tensor':
        (bundle[1] / 'output_bias.fp16').write_bytes(b'\x00\x3c' + bytes(46))
    scores = [list(range(24)), list(range(24))]
    reference = tmp_path / 'reference.json'
    reference.write_text(json.dumps({'states': [[0] * 96] * 2, 'scores_fp32': scores,
        'metadata': {'source_weights_sha256': 'a' * 64}}))
    observed = json.loads(json.dumps({'scores': scores}))
    if kind == 'last_row':
        observed['scores'][1][23] = 99
    if kind == 'skip':
        observed['status'] = 'skip'
    actual = tmp_path / 'scores.json'
    actual.write_text(json.dumps(observed))
    command = [sys.executable, str(root / 'tools/cube4_numeric_gate.py'),
        '--reference', str(reference), '--actual', str(actual)]
    for name, path in paths.items():
        command.extend(['--' + name, str(path)])
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert (result.returncode == 0) is (kind == 'valid'), result.stderr
    if kind == 'valid':
        report = json.loads(result.stdout)
        assert report['elements_compared'] == 48
        assert report['numerical_accepted'] is True
        assert report['production_quality_accepted'] is False
        assert len(report['producer_sha256']) == 64
        assert len(report['tensor_sha256']) == 60
    elif kind == 'last_row':
        assert json.loads(result.stdout)['numerical_accepted'] is False
    elif kind == 'skip':
        assert 'native score dump schema' in result.stderr
    else:
        assert 'hash' in result.stderr
