import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from tools.cuda_device_observation import query

@unittest.skipUnless(os.name=='posix','POSIX file limit required')
class ProbeOutputBound(unittest.TestCase):
    def test_oversized_probe_cannot_fill_temporary_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            probe=Path(directory)/'probe'
            probe.write_text('#!/usr/bin/env python3\nimport os\nwhile True: os.write(1,b"x"*65536)\n')
            probe.chmod(0o755)
            with self.assertRaises(subprocess.CalledProcessError): query(probe)
    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            probe=Path(directory)/'probe'
            probe.write_text('#!/usr/bin/env python3\nprint(\'{"schema_version":1,"schema_version":1}\')\n')
            probe.chmod(0o755)
            with self.assertRaises(ValueError): query(probe)
