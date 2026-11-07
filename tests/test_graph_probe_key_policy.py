"""Execute actual test-probe key guard; no CUDA/runtime acceptance claim."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'posix' and shutil.which('g++'), 'POSIX g++ required')
class GraphProbeKeyPolicyTests(unittest.TestCase):
    def test_cross_shape_tolerance_and_same_shape_exactness(self):
        source = (ROOT / 'tests/stream1_graph_lane_probe.hpp').read_text()
        guard = source[source.index('inline bool stream1_graph_probe_score_matches('):
                       source.index('// Test-only helper:')]
        program = '#include <cstdint>\n#include <cassert>\n' + guard + '''
int main(){
assert(stream1_graph_probe_score_matches(10000,10000,false));
assert(!stream1_graph_probe_score_matches(10000,10001,false));
assert(stream1_graph_probe_score_matches(10000,13072,true));
assert(stream1_graph_probe_score_matches(13072,10000,true));
assert(!stream1_graph_probe_score_matches(10000,13073,true));
assert(!stream1_graph_probe_score_matches(13073,10000,true));
assert(!stream1_graph_probe_score_matches(0,UINT32_MAX,true));
assert(!stream1_graph_probe_score_matches(UINT32_MAX,0,true));
}
'''
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / 'probe.cpp'
            binary = Path(directory) / 'probe'
            fixture.write_text(program)
            subprocess.run(['g++', '-std=c++17', '-fsanitize=undefined',
                            str(fixture), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)
