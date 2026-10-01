from pathlib import Path
import pytest
from tools.cube555.host_budget import available_ram_bytes


def fixture_controller(tmp_path, kind='cgroup'):
    mount = tmp_path / 'controller'
    leaf = mount / 'sandbox'
    leaf.mkdir(parents=True)
    proc = tmp_path / 'proc'
    proc.mkdir()
    (proc / 'cgroup').write_text('0::/sandbox\n' if kind == 'cgroup2' else '6:memory:/sandbox\n')
    (proc / 'mountinfo').write_text(
        f'1 0 0:1 / {mount.as_posix()} rw - {kind} controller rw,memory\n')
    names = ('memory.current', 'memory.max') if kind == 'cgroup2' else (
        'memory.usage_in_bytes', 'memory.limit_in_bytes')
    def counters(path, usage, limit):
        (path / names[0]).write_text(str(usage))
        (path / names[1]).write_text(str(limit))
    return proc, mount, leaf, counters


def test_child_limit_not_host_memory(tmp_path):
    proc, parent, leaf, counters = fixture_controller(tmp_path)
    counters(parent, 90, 160)
    counters(leaf, 3, 32)
    assert available_ram_bytes(500, proc) == 29


def test_parent_usage_includes_siblings(tmp_path):
    proc, parent, leaf, counters = fixture_controller(tmp_path)
    counters(parent, 10, 16)
    counters(leaf, 2, 32)
    assert available_ram_bytes(500, proc) == 6


def test_v2_unlimited_leaf_still_has_limited_parent(tmp_path):
    proc, parent, leaf, counters = fixture_controller(tmp_path, 'cgroup2')
    counters(parent, 11, 12)
    counters(leaf, 2, 'max')
    assert available_ram_bytes(500, proc) == 1


def test_host_available_can_be_tighter(tmp_path):
    proc, parent, leaf, counters = fixture_controller(tmp_path)
    counters(parent, 0, 160)
    counters(leaf, 3, 32)
    assert available_ram_bytes(7, proc) == 7


def test_unknown_membership_does_not_size_from_host(tmp_path):
    proc, parent, leaf, counters = fixture_controller(tmp_path)
    counters(parent, 0, 160)
    (proc / 'cgroup').write_text('6:memory:/missing\n')
    with pytest.raises(RuntimeError, match='Cannot resolve'):
        available_ram_bytes(500, proc)


def test_exhausted_parent_has_no_history_budget(tmp_path):
    proc, parent, leaf, counters = fixture_controller(tmp_path)
    counters(parent, 17, 16)
    counters(leaf, 2, 32)
    assert available_ram_bytes(500, proc) == 0


def test_missing_proc_is_explicit_error(tmp_path):
    with pytest.raises(RuntimeError, match='Cannot resolve'):
        available_ram_bytes(500, tmp_path)
