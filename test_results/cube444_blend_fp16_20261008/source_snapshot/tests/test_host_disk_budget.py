"""Real filesystem admission at the CLI boundary; no cloud or GPU involved."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_budget(path, budget, headroom):
    return subprocess.run([sys.executable, str(ROOT / 'tools/host_disk_budget.py'),
        '--path', str(path), '--history-disk-bytes', str(budget),
        '--headroom-bytes', str(headroom)], capture_output=True, text=True, timeout=10)


def test_impossible_disk_reservation_rejected_before_allocating(tmp_path):
    result = run_budget(tmp_path, 18446744073709551615, 0)
    assert result.returncode == 2
    assert 'insufficient filesystem space' in result.stderr
    assert not list(tmp_path.iterdir())


def test_zero_disk_budget_observes_actual_filesystem(tmp_path):
    result = run_budget(tmp_path, 0, 0)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['required_bytes'] == 0
    assert report['available_bytes'] >= 0
    assert report['physical_reservation_confirmed'] is False
