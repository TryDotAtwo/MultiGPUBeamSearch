"""Hardware identity is compatibility evidence, never numerical quality."""
import pytest
import json
import subprocess
import sys
from pathlib import Path


@pytest.mark.parametrize('name,accepted', [('NVIDIA H200', True), ('NVIDIA B200', False)])
def test_build_cli_consumes_hardware_before_acceptance(tmp_path, name, accepted):
    root = Path(__file__).resolve().parents[1]
    runner = tmp_path / 'runner'
    runner.write_bytes(b'fixture binary identity')
    info = tmp_path / 'build.json'
    info.write_text(json.dumps(dict(schema_version=1, state_len=96,
        state_storage_len=112, move_count=24, candidate_meta_bytes=32,
        cuda_architectures='90a')))
    report = tmp_path / 'hardware.csv'
    report.write_text(f'{name}, 143771, 9.0\n')
    result = subprocess.run([sys.executable, str(root / 'tools/cube4_build_contract.py'),
        str(runner), str(info), '--hardware-report', str(report), '--world-size', '1'],
        capture_output=True, text=True, timeout=15)
    if accepted:
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload['hardware']['gpu_count'] == 1
        assert payload['quality_accepted'] is False
    else:
        assert result.returncode != 0
        assert 'expected H200 hardware' in result.stderr


def test_h200_identity_keeps_variant_and_quality_unaccepted():
    from tools.cube4_build_contract import validate_h200_hardware
    result = validate_h200_hardware('NVIDIA H200, 143771, 9.0\n' * 2, 2)
    assert result['gpu_count'] == 2
    assert result['variant'] == 'H200'
    assert result['quality_accepted'] is False


@pytest.mark.parametrize('raw,world', [
    ('NVIDIA H200, 143771, 9.0\n', 2),
    ('NVIDIA B200, 183000, 10.0\n', 1),
    ('NVIDIA H200, 10000, 9.0\n', 1),
    ('NVIDIA H200, 143771, 8.6\n', 1),
    ('NVIDIA H200, 143771, 9.0, extra\n', 1),
    ('NVIDIA H200, 143771, 9.0\nNVIDIA H200 NVL, 143771, 9.0\n', 2)])
def test_incompatible_or_ambiguous_hardware_rejected(raw, world):
    from tools.cube4_build_contract import validate_h200_hardware
    with pytest.raises(ValueError):
        validate_h200_hardware(raw, world)
