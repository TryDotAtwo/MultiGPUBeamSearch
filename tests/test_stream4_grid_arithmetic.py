"""Execute actual Stream4 block_count expressions, not CUDA kernels."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'posix' and shutil.which('g++'), 'POSIX g++ required')
class Stream4GridArithmeticTests(unittest.TestCase):
    def test_actual_block_count_expressions_do_not_wrap(self):
        source = (ROOT / 'cuda/stream4.cu').read_text()
        expressions = re.findall(r'const std::uint32_t block_count = ([^;]+);', source)
        self.assertEqual(len(expressions), 3)
        checks = '\n'.join('assert((' + expression + ') == expected);' for expression in expressions)
        program = '''#include <cstdint>
#include <cassert>
int main(){
constexpr std::uint32_t block_size=256;
struct Case{std::uint32_t capacity, expected;};
for(auto item:{Case{0,0},Case{1,1},Case{256,1},Case{257,2},
              Case{4294967040U,16777215U},Case{4294967295U,16777216U}}){
const auto capacity=item.capacity,expected=item.expected;
''' + checks + '\n}}'
        program = '#include <initializer_list>\n' + program
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'grid.cpp'
            file.write_text(program)
            binary = Path(directory) / 'grid'
            subprocess.run(['g++', '-std=c++20', '-fsanitize=undefined',
                            '-fno-sanitize-recover=undefined', str(file), '-o', str(binary)], check=True)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
