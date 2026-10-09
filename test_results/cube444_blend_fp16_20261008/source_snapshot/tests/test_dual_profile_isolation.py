"""Execute actual helper profile preparation; stale status paths must not escape."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('check', ['micro', 'status_path'])
@pytest.mark.parametrize('mode', ['neural', 'reflected'])
def test_neural_helper_preserves_profile_and_isolates_status(check, mode, tmp_path):
    filename = 'run_vast_dual_neural_20261002.py' if mode == 'neural' else 'run_vast_reflected_20261002.py'
    source = ROOT / 'evidence' / filename
    if not source.exists():
        source = ROOT / 'test_results/audit_fixes_2026-09-28' / filename
    text = source.read_text()
    start = text.index('profile_document=')
    end = text.index('admission = validate_cube4_bundle' if mode == 'neural' else 'env = {', start)
    actual = text[start:end]
    request = tmp_path / 'old-request.json'
    stale = tmp_path / 'prior-case' / 'rank-status'
    request.write_text(json.dumps({'profile': {'BEAM_STREAM1_TRANSFORMER_MICRO': '256',
                                             'BEAM_RANK_STATUS_DIR': str(stale)}}))
    run = tmp_path / 'fresh-case'
    scope = dict(json=json, args=SimpleNamespace(profile_source=request,
                 history_disk_bytes=None, final_split=None), evidence=tmp_path,
                 weights=tmp_path/'weights', data=tmp_path/'inputs', run=run)
    scope.update(a=scope['args'], out=run, reflected=[0, 1])
    exec(compile(actual, 'actual_helper_profile', 'exec'), scope)
    profile = scope['profile']
    if check == 'micro':
        assert profile['BEAM_STREAM1_TRANSFORMER_MICRO'] == '256'
    else:
        assert profile['BEAM_RANK_STATUS_DIR'] == str(run/'rank-status')
