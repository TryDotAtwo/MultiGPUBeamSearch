import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
INFO = dict(schema_version=1, state_len=96, state_storage_len=112, move_count=24,
            candidate_meta_bytes=32, cuda_architectures='90a')


def invoke(tmp_path, info):
    metadata = tmp_path / 'build.json'
    metadata.write_text(json.dumps(info))
    binary = tmp_path / 'runner'
    binary.write_bytes(b'abc')  # Hash-receipt fixture, not an executable claim.
    return subprocess.run([sys.executable, str(ROOT / 'tools/cube4_build_contract.py'),
                           str(binary), str(metadata)], capture_output=True, text=True)


def test_expected_build_records_binary_hash_but_not_quality(tmp_path):
    result = invoke(tmp_path, INFO)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['runner_sha256'] == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    assert report['quality_accepted'] is False
    assert report['build_info'] == INFO


@pytest.mark.parametrize('key,value', [('state_len', 120), ('state_storage_len', 128),
    ('move_count', 18), ('candidate_meta_bytes', 16), ('schema_version', True),
    ('cuda_architectures', '75,86'), ('cuda_architectures', 'unknown'),
    ('cuda_architectures', 'x90a'), ('move_count', '24')])
def test_incompatible_build_rejected(tmp_path, key, value):
    result = invoke(tmp_path, {**INFO, key: value})
    assert result.returncode == 2


def test_missing_build_field_rejected(tmp_path):
    info = dict(INFO)
    del info['cuda_architectures']
    assert invoke(tmp_path, info).returncode == 2
