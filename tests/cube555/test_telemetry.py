import tempfile
import unittest
from pathlib import Path

from tools.cube555.telemetry import host_memory_sample


class HostMemoryTests(unittest.TestCase):
    def test_container_counters_and_missing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(host_memory_sample(root), {})
            (root / 'memory.current').write_text('4096\n')
            (root / 'memory.max').write_text('max\n')
            (root / 'memory.events').write_text('oom 2\noom_kill 1\n')
            sample = host_memory_sample(root, root / 'absent-proc')
            self.assertEqual(sample.pop('memory.scope'), 'controller_root_unverified')
            sample.pop('memory.note')
            self.assertEqual(sample, {
                'memory.current': 4096, 'memory.max': 'unlimited',
                'memory.events': {'oom': 2, 'oom_kill': 1},
            })

    def test_cgroup_v1_limit_is_not_host_meminfo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = root / 'memory'
            memory.mkdir()
            (memory / 'memory.usage_in_bytes').write_text('8192')
            (memory / 'memory.limit_in_bytes').write_text('32768')
            (memory / 'memory.failcnt').write_text('3')
            (memory / 'memory.oom_control').write_text('under_oom 0\noom_kill 1\n')
            sample = host_memory_sample(root, root / 'absent-proc')
            self.assertEqual(sample.pop('memory.scope'), 'controller_root_unverified')
            sample.pop('memory.note')
            self.assertEqual(sample, {
                'memory.current': 8192, 'memory.max': 32768, 'memory.failcnt': 3,
                'memory.oom_control': {'under_oom': 0, 'oom_kill': 1},
            })

    def test_process_membership_selects_child_not_parent_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = root / 'memory'
            child = controller / 'sandbox'
            child.mkdir(parents=True)
            (controller / 'memory.usage_in_bytes').write_text('9000')
            (controller / 'memory.limit_in_bytes').write_text('160000')
            (child / 'memory.usage_in_bytes').write_text('3000')
            (child / 'memory.limit_in_bytes').write_text('32000')
            proc = root / 'proc'
            proc.mkdir()
            (proc / 'cgroup').write_text('6:memory:/sandbox\n')
            (proc / 'mountinfo').write_text(
                f'1 0 0:1 / {controller.as_posix()} rw - cgroup cgroup rw,memory\n')
            sample = host_memory_sample(root, proc)
            self.assertEqual(sample['memory.current'], 3000)
            self.assertEqual(sample['memory.max'], 32000)
            self.assertEqual(sample['memory.scope'], 'process_cgroup')


if __name__ == '__main__':
    unittest.main()
