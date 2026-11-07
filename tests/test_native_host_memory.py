"""Exercise the actual C++ admission helper, including namespace-root cgroups."""
from pathlib import Path
import shutil
import subprocess
import pytest


@pytest.fixture(scope='module')
def probe(tmp_path_factory):
    compiler = shutil.which('g++') or shutil.which('clang++')
    if not compiler:
        pytest.skip('C++ compiler required for native memory helper')
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path_factory.mktemp('native_memory') / 'probe'
    subprocess.run([compiler, '-std=c++20', '-I', str(root),
                    str(root / 'tests/host_memory_budget_probe.cpp'), '-o', str(binary)],
                   check=True, capture_output=True, text=True)
    return binary


@pytest.mark.parametrize('version', [1, 2])
def test_namespace_root_membership_accepts_valid_budget(probe, tmp_path, version):
    # Removing root-path handling must not reject a valid namespace-root container.
    meminfo = tmp_path / 'meminfo'
    meminfo.write_text('MemAvailable: 1000000 kB\n')
    root = tmp_path / 'cgroup'
    controller = root if version == 2 else root / 'memory'
    controller.mkdir(parents=True)
    (controller / ('memory.max' if version == 2 else 'memory.limit_in_bytes')).write_text('100000000')
    (controller / ('memory.current' if version == 2 else 'memory.usage_in_bytes')).write_text('30000000')
    membership = tmp_path / 'membership'
    membership.write_text('0::/\n' if version == 2 else '7:memory:/\n')
    result = subprocess.run([str(probe), str(meminfo), str(root), str(membership)],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '70000000'


@pytest.mark.parametrize('version', [1, 2])
def test_parent_budget_wins_over_child(probe, tmp_path, version):
    meminfo = tmp_path / 'meminfo'
    meminfo.write_text('MemAvailable: 1000000 kB\n')
    root = tmp_path / 'cgroup'
    base = root if version == 2 else root / 'memory'
    for folder, limit, usage in ((base, 900000000, 0),
                                  (base / 'parent', 200000000, 50000000),
                                  (base / 'parent/child', 500000000, 100000000)):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / ('memory.max' if version == 2 else 'memory.limit_in_bytes')).write_text(str(limit))
        (folder / ('memory.current' if version == 2 else 'memory.usage_in_bytes')).write_text(str(usage))
    membership = tmp_path / 'membership'
    membership.write_text('0::/parent/child\n' if version == 2 else '7:memory:/parent/child\n')
    result = subprocess.run([str(probe), str(meminfo), str(root), str(membership)],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '150000000'


@pytest.mark.parametrize('mem_text,limit,usage,member', [
    ('MemAvailable: 18446744073709551615 kB\n', '100', '0', '/'),
    ('MemAvailable: -1 kB\n', '100', '0', '/'),
    ('MemAvailable: 1000 kB\n', 'bad', '0', '/'),
    ('MemAvailable: 1000 kB\n', '100', 'bad', '/'),
    ('MemAvailable: 1000 kB\n', '100', '0', '/../escape'),
])
def test_invalid_resource_snapshot_rejected(probe, tmp_path, mem_text, limit, usage, member):
    meminfo = tmp_path / 'meminfo'
    meminfo.write_text(mem_text)
    root = tmp_path / 'cgroup'
    root.mkdir()
    (root / 'memory.max').write_text(limit)
    (root / 'memory.current').write_text(usage)
    membership = tmp_path / 'membership'
    membership.write_text('0::' + member + '\n')
    result = subprocess.run([str(probe), str(meminfo), str(root), str(membership)],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    assert result.stdout == ''
