"""Exercise real shell rejection before any external preflight or rank launch."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
GIT_BASH = Path('C:/Program Files/Git/bin/bash.exe')
BASH = str(GIT_BASH) if os.name == 'nt' and GIT_BASH.exists() else shutil.which('bash')


@pytest.mark.skipif(not BASH, reason='bash required')
@pytest.mark.parametrize('inner', [None, '128'])
def test_profile_gate_receives_actual_runtime_defaults(tmp_path, inner):
    marker = tmp_path / 'profile-environment'
    verifier = tmp_path / 'profile_verifier.sh'
    verifier.write_text('#!/bin/bash\n'
        'printf "%s\\n" "$H200_WORLD_SIZE" "$BEAM_WIDTH" '
        '"$BEAM_B_MICRO" "$BEAM_STREAM1_TRANSFORMER_MICRO" > "$PROFILE_TEST_MARKER"\n'
        'exit 23\n', encoding='utf-8')
    verifier.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if key not in ('H200_WORLD_SIZE', 'BEAM_STREAM1_TRANSFORMER_MICRO')}
    env.update(H200_REPO_DIR=ROOT.as_posix(), H200_RUN_ROOT=tmp_path.as_posix(),
        H200_PYTHON=verifier.as_posix(), H200_PROFILE_PATH='unused-profile.json',
        PROFILE_TEST_MARKER=marker.as_posix(), BEAM_WIDTH='4096', BEAM_B_MICRO='1024')
    if inner is not None:
        env['BEAM_STREAM1_TRANSFORMER_MICRO'] = inner
    result = subprocess.run([BASH, (ROOT / 'hpc/h200_dual_large_beam.sh').as_posix()],
        env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 23, result.stdout + result.stderr
    assert marker.read_text().splitlines() == ['2', '4096', '1024', inner or '1024']
    assert not list(tmp_path.glob('h200_run.*/rank*.log'))


@pytest.mark.skipif(not BASH, reason='bash required')
@pytest.mark.parametrize('depth', ['0', '-1', '1.5', 'abc', '4294967296', '999999999999999999999'])
def test_invalid_depth_rejected_before_external_preflight(tmp_path, depth):
    # If validation is absent, the first external verifier creates this marker.
    marker = tmp_path / 'external-called'
    verifier = tmp_path / 'verifier.sh'
    verifier.write_text('#!/bin/bash\ntouch "$DEPTH_TEST_MARKER"\nexit 0\n', encoding='utf-8')
    verifier.chmod(0o755)
    env = {**os.environ, 'H200_REPO_DIR': ROOT.as_posix(),
           'H200_RUN_ROOT': tmp_path.as_posix(), 'H200_PYTHON': verifier.as_posix(),
           'DEPTH_TEST_MARKER': marker.as_posix(), 'DEPTH_LIMIT': depth,
           'BEAM_WIDTH': '4096'}
    result = subprocess.run([BASH, (ROOT / 'hpc/h200_dual_large_beam.sh').as_posix()],
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'invalid DEPTH_LIMIT' in result.stderr
    assert not marker.exists()
    assert not list(tmp_path.glob('h200_run.*/rank*.log'))


@pytest.mark.skipif(not BASH, reason='bash required')
@pytest.mark.parametrize('depth', ['1', '50', '4294967295'])
def test_valid_depth_reaches_bundle_gate_without_starting_ranks(tmp_path, depth):
    marker = tmp_path / 'external-called'
    verifier = tmp_path / 'verifier.sh'
    # Deliberately fail the first bundle gate: no model, GPU or rank is needed
    # to establish that a valid depth was admitted by the real shell parser.
    verifier.write_text('#!/bin/bash\ntouch "$DEPTH_TEST_MARKER"\nexit 23\n', encoding='utf-8')
    verifier.chmod(0o755)
    result = subprocess.run([BASH, (ROOT / 'hpc/h200_dual_large_beam.sh').as_posix()],
        env={**os.environ, 'H200_REPO_DIR': ROOT.as_posix(),
             'H200_RUN_ROOT': tmp_path.as_posix(), 'H200_PYTHON': verifier.as_posix(),
             'DEPTH_TEST_MARKER': marker.as_posix(), 'DEPTH_LIMIT': depth,
             'BEAM_WIDTH': '4096'}, capture_output=True, text=True, timeout=15)
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'invalid DEPTH_LIMIT' not in result.stderr
    assert marker.exists()
    assert not list(tmp_path.glob('h200_run.*/rank*.log'))


@pytest.mark.skipif(not BASH, reason='bash required')
def test_impossible_history_disk_budget_stops_before_ranks(tmp_path):
    runner = tmp_path / 'production_runner'
    runner.write_text('#!/bin/bash\necho "{}"\n', encoding='utf-8')
    runner.chmod(0o755)
    verifier = tmp_path / 'verifier.sh'
    verifier.write_text('#!/bin/bash\n'
        'if [[ "$1" == *host_disk_budget.py ]]; then exec "' + Path(sys.executable).as_posix() + '" "$@"; fi\n'
        'exit 0\n', encoding='utf-8')
    verifier.chmod(0o755)
    result = subprocess.run([BASH, (ROOT / 'hpc/h200_dual_large_beam.sh').as_posix()],
        env={**os.environ, 'H200_REPO_DIR': ROOT.as_posix(),
             'H200_RUN_ROOT': tmp_path.as_posix(), 'H200_BUILD_DIR': tmp_path.as_posix(),
             'H200_PYTHON': verifier.as_posix(), 'H200_SMI': verifier.as_posix(),
             'DEPTH_LIMIT': '50', 'BEAM_WIDTH': '4096',
             'BEAM_HISTORY_DISK_BYTES': '18446744073709551615',
             'BEAM_HOST_DISK_HEADROOM_BYTES': '0'}, capture_output=True, text=True, timeout=15)
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'insufficient filesystem space' in result.stderr
    assert not list(tmp_path.glob('h200_run.*/rank*.log'))


@pytest.mark.skipif(not BASH, reason='bash required')
def test_nccl_observation_failure_stops_before_ranks(tmp_path):
    runner = tmp_path / 'production_runner'
    runner.write_text('#!/bin/bash\necho "{}"\n', encoding='utf-8')
    runner.chmod(0o755)
    verifier = tmp_path / 'verifier.sh'
    verifier.write_text('#!/bin/bash\n'
        'if [[ "$1" == *nccl_library_observation.py ]]; then exit 23; fi\n'
        'exit 0\n', encoding='utf-8')
    verifier.chmod(0o755)
    result = subprocess.run([BASH, (ROOT / 'hpc/h200_dual_large_beam.sh').as_posix()],
        env={**os.environ, 'H200_REPO_DIR': ROOT.as_posix(),
             'H200_RUN_ROOT': tmp_path.as_posix(), 'H200_BUILD_DIR': tmp_path.as_posix(),
             'H200_PYTHON': verifier.as_posix(), 'H200_SMI': verifier.as_posix(),
             'BEAM_WIDTH': '4096'}, capture_output=True, text=True, timeout=15)
    assert result.returncode == 23, result.stdout + result.stderr
    assert not list(tmp_path.glob('h200_run.*/rank*.log'))
