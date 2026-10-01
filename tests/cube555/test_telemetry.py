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
            self.assertEqual(host_memory_sample(root), {
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
            self.assertEqual(host_memory_sample(root), {
                'memory.current': 8192, 'memory.max': 32768, 'memory.failcnt': 3,
                'memory.oom_control': {'under_oom': 0, 'oom_kill': 1},
            })


if __name__ == '__main__':
    unittest.main()
