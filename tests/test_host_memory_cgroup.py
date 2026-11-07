"""Public RAM probe must not spend host RAM unavailable to its container."""
from pathlib import Path
import subprocess
import sys
import os
import pytest
from tools import run_cayleypy_public as cli
from tools import host_memory_budget as memory


def fixture_paths(tmp_path, monkeypatch):
    meminfo = tmp_path / 'meminfo'
    meminfo.write_text('MemAvailable: 1000000 kB\n')
    root = tmp_path / 'cgroup'
    root.mkdir()
    actual_path = Path
    mapping = {'/proc/meminfo': meminfo, '/sys/fs/cgroup': root}
    monkeypatch.setattr(cli, 'Path', lambda value: mapping.get(str(value), actual_path(value)))
    return root


def membership_file(tmp_path, monkeypatch, text):
    membership = tmp_path / 'self-cgroup'
    membership.write_text(text)
    actual_path = Path
    monkeypatch.setattr(memory, 'Path', lambda value: membership if str(value) == '/proc/self/cgroup' else actual_path(value))


def test_v2_parent_limit_is_stricter_than_child_limit(tmp_path, monkeypatch):
    root = fixture_paths(tmp_path, monkeypatch)
    membership_file(tmp_path, monkeypatch, '0::/parent/child\n')
    for path, limit, usage in ((root, 1000000000, 0),
                               (root / 'parent', 200000000, 50000000),
                               (root / 'parent/child', 500000000, 100000000)):
        path.mkdir(parents=True, exist_ok=True)
        (path / 'memory.max').write_text(str(limit))
        (path / 'memory.current').write_text(str(usage))
    assert cli._available_ram_bytes() == 150000000


def test_v1_parent_limit_is_stricter_than_child_limit(tmp_path, monkeypatch):
    root = fixture_paths(tmp_path, monkeypatch) / 'memory'
    membership_file(tmp_path, monkeypatch, '7:memory:/parent/child\n')
    for path, limit, usage in ((root, 1000000000, 0),
                               (root / 'parent', 200000000, 50000000),
                               (root / 'parent/child', 500000000, 100000000)):
        path.mkdir(parents=True, exist_ok=True)
        (path / 'memory.limit_in_bytes').write_text(str(limit))
        (path / 'memory.usage_in_bytes').write_text(str(usage))
    assert cli._available_ram_bytes() == 150000000


def test_public_probe_clamps_v2_memory_to_limit_minus_current(tmp_path, monkeypatch):
    root = fixture_paths(tmp_path, monkeypatch)
    (root / 'memory.max').write_text('100000000\n')
    (root / 'memory.current').write_text('30000000\n')
    assert cli._available_ram_bytes() == 70000000


def test_public_probe_clamps_v1_memory_to_limit_minus_usage(tmp_path, monkeypatch):
    root = fixture_paths(tmp_path, monkeypatch) / 'memory'
    root.mkdir()
    (root / 'memory.limit_in_bytes').write_text('200000000\n')
    (root / 'memory.usage_in_bytes').write_text('50000000\n')
    assert cli._available_ram_bytes() == 150000000


def test_public_probe_does_not_ignore_malformed_detected_controller(tmp_path, monkeypatch):
    root = fixture_paths(tmp_path, monkeypatch)
    (root / 'memory.max').write_text('bad\n')
    (root / 'memory.current').write_text('0\n')
    with pytest.raises(RuntimeError):
        cli._available_ram_bytes()


def test_host_preflight_cli_rejects_history_plus_headroom_above_cgroup(tmp_path):
    meminfo = tmp_path / 'meminfo'
    meminfo.write_text('MemAvailable: 1000000 kB\n')
    root = tmp_path / 'cgroup'
    root.mkdir()
    (root / 'memory.max').write_text('100000000')
    (root / 'memory.current').write_text('30000000')
    script = Path(__file__).resolve().parents[1] / 'tools/host_memory_budget.py'
    result = subprocess.run([sys.executable, str(script), '--meminfo', str(meminfo),
        '--cgroup-root', str(root), '--history-ram-bytes', '60000000',
        '--headroom-bytes', '20000000'], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'insufficient container RAM' in result.stderr


def test_host_preflight_cli_accepts_history_and_preserves_headroom(tmp_path):
    meminfo = tmp_path / 'meminfo'
    meminfo.write_text('MemAvailable: 1000000 kB\n')
    root = tmp_path / 'cgroup'
    root.mkdir()
    (root / 'memory.max').write_text('100000000')
    (root / 'memory.current').write_text('30000000')
    script = Path(__file__).resolve().parents[1] / 'tools/host_memory_budget.py'
    result = subprocess.run([sys.executable, str(script), '--meminfo', str(meminfo),
        '--cgroup-root', str(root), '--history-ram-bytes', '40000000',
        '--headroom-bytes', '20000000'], capture_output=True, text=True)
    assert result.returncode == 0
    import json
    record = json.loads(result.stdout)
    assert record['available_bytes'] == 70000000
    assert record['remaining_after_history_bytes'] == 30000000


def test_explicit_rental_ram_cap_rejects_budget_below_cgroup(tmp_path):
    meminfo = tmp_path / 'meminfo'
    meminfo.write_text('MemAvailable: 1000000 kB\n')
    root = tmp_path / 'cgroup'
    root.mkdir()
    environment = dict(os.environ, BEAM_HOST_RAM_CAP_BYTES='50000000')
    script = Path(__file__).resolve().parents[1] / 'tools/host_memory_budget.py'
    result = subprocess.run([sys.executable, str(script), '--meminfo', str(meminfo),
        '--cgroup-root', str(root), '--history-ram-bytes', '40000000',
        '--headroom-bytes', '20000000'], env=environment, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'insufficient container RAM' in result.stderr


def test_public_automatic_budget_obeys_explicit_rental_cap(tmp_path, monkeypatch):
    fixture_paths(tmp_path, monkeypatch)
    monkeypatch.setenv('BEAM_HOST_RAM_CAP_BYTES', '50000000')
    assert cli._available_ram_bytes() == 50000000


@pytest.mark.parametrize('cap', ['', '0', '-1', '1.5', '18446744073709551616'])
def test_public_probe_rejects_invalid_explicit_cap(tmp_path, monkeypatch, cap):
    fixture_paths(tmp_path, monkeypatch)
    monkeypatch.setenv('BEAM_HOST_RAM_CAP_BYTES', cap)
    with pytest.raises(ValueError):
        cli._available_ram_bytes()
