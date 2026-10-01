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


if __name__ == '__main__':
    unittest.main()
