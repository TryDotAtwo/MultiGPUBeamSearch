"""Profile resolution, not GPU acceptance or authentication of evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'hpc/profiles/h200_8gpu_cube4_256m.json'


def invoke(path=PROFILE, *overrides):
    command = [sys.executable, str(ROOT / 'tools/cube4_profile.py'), '--profile', str(path)]
    for item in overrides:
        command += ['--override', item]
    return subprocess.run(command, capture_output=True, text=True)


def test_historical_profile_does_not_promote_new_binary():
    result = invoke()
    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout)
    assert resolved['derived']['effective_beam'] == 256376832
    assert resolved['derived']['shard_capacity'] == 500736
    assert resolved['derived']['stream3_batch_candidates'] == 393216
    assert resolved['promotion_allowed'] is False
    assert resolved['status'] == 'historically_measured'


def test_override_is_explicitly_unvalidated():
    result = invoke(PROFILE, 'BEAM_B_MICRO=1024')
    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout)
    assert resolved['status'] == 'unvalidated_override'
    assert resolved['parameters']['BEAM_B_MICRO'] == 1024
    assert resolved['derived']['stream3_batch_candidates'] == 196608
    assert resolved['promotion_allowed'] is False


@pytest.mark.parametrize('override', ['BEAM_B_MICRO=-1', 'BEAM_B_MICRO=0',
    'BEAM_WIDTH=18446744073709551615', 'BEAM_STREAM3_RING_SLOTS=4294967295',
    'BEAM_STREAM1_TRANSFORMER_MICRO=4096', 'UNKNOWN=1', 'BEAM_B_MICRO=1.2'])
def test_invalid_override_rejected(override):
    result = invoke(PROFILE, override)
    assert result.returncode == 2


@pytest.mark.parametrize('mutation', ['missing', 'boolean', 'unknown', 'approved', 'duplicate'])
def test_invalid_profile_rejected(tmp_path, mutation):
    profile = json.loads(PROFILE.read_text())
    if mutation == 'missing': del profile['parameters']['BEAM_STREAM1_TRANSFORMER_MICRO']
    if mutation == 'boolean': profile['parameters']['BEAM_B_MICRO'] = True
    if mutation == 'unknown': profile['parameters']['TYPO'] = 1
    if mutation == 'approved': profile['status'] = 'approved'
    text = json.dumps(profile)
    if mutation == 'duplicate': text = text.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
    path = tmp_path / 'profile.json'
    path.write_text(text)
    result = invoke(path)
    assert result.returncode == 2


def test_environment_must_be_complete_and_actual_values_are_reported():
    profile = json.loads(PROFILE.read_text())
    environment = {key: value for key, value in os.environ.items() if key not in profile['parameters']}
    command = [sys.executable, str(ROOT / 'tools/cube4_profile.py'), '--profile', str(PROFILE), '--environment']
    missing = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert missing.returncode == 2
    assert 'missing launch environment' in missing.stderr
    environment.update({key: str(value) for key, value in profile['parameters'].items()})
    environment['BEAM_STREAM1_CONCURRENCY'] = '2'
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout)
    assert resolved['status'] == 'unvalidated_override'
    assert resolved['parameters']['BEAM_STREAM1_CONCURRENCY'] == 2
    assert resolved['promotion_allowed'] is False
