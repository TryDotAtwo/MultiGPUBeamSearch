"""Execute actual dispatcher reserve arithmetic with bounded extreme inputs."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'posix' and shutil.which('g++'), 'POSIX g++ required')
class Stream3WriteReserveTests(unittest.TestCase):
    def test_extreme_batch_ceil_and_margin_do_not_wrap(self):
        source = (ROOT / 'cuda/dispatcher.cu').read_text()
        start = source.index('        const std::uint32_t average_shard_write =')
        end = source.index('        const std::uint32_t write_reserve =', start)
        program = '''#include <cstdint>
#include <cassert>
struct Config { std::uint32_t shard_count, stream3_batch_candidates; };
struct Plan { Config config; };
int main() {
for (unsigned variant=0;variant<2;++variant) {
Plan plan{{variant==0 ? 64U : 0U, UINT32_MAX}};
''' + source[start:end] + '''
assert(average_shard_write == (variant==0 ? 67108864U : UINT32_MAX));
assert(write_margin == (variant==0 ? 16777216U : 1073741824U));
assert(unclamped_write_reserve == (variant==0 ? 83886080U : UINT32_MAX));
}}
'''
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / 'reserve.cpp'
            binary = Path(directory) / 'reserve'
            fixture.write_text(program)
            build = subprocess.run(['g++', '-std=c++20', '-fsanitize=undefined',
                '-fno-sanitize-recover=undefined', str(fixture), '-o', str(binary)],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
