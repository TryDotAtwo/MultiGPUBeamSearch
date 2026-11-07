"""Bounded layout-risk reproduction, NOT a repaired-kernel acceptance test."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'posix' and shutil.which('g++'), 'POSIX g++ required')
class HopperPackedFallbackReproduction(unittest.TestCase):
    def test_actual_packer_is_not_legacy_row_major(self):
        loader = (ROOT / 'tools/stream1_weight_io.hpp').read_text()
        start = loader.index('inline std::vector<std::byte> pack_kxn_column_major_2byte(')
        end = loader.index('\ninline void alloc_managed_pointer_table', start)
        native = (ROOT / 'cuda/stream1_transformer.cu').read_text()
        linear = native[native.index('void stream1_transformer_linear_bias_typed('):
                        native.index('void stream1_transformer_linear_bias_cuda(')]
        self.assertIn('cutlass::layout::RowMajor', linear)
        self.assertIn('static_cast<int>(output_cols)', linear)
        program = '''#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <cassert>
#include <iostream>
#include "stream1_transformer_hopper_policy.hpp"
''' + loader[start:end] + '''
int main() {
const auto mode=beam::select_stream1_transformer_hopper_mode("off","fp16_tma");
assert(beam::stream1_transformer_hopper_uses_packed_full_token_weight(mode,0,4));
assert(!beam::stream1_transformer_hopper_large_gemm_allowed(mode,90,57));
const std::uint16_t original[6]={1,2,3,4,5,6};
std::vector<std::byte> bytes(sizeof(original));
std::memcpy(bytes.data(),original,sizeof(original));
auto packed=pack_kxn_column_major_2byte(bytes,3,2);
std::uint16_t recovered[6];
std::memcpy(recovered,packed.data(),sizeof(recovered));
const unsigned x[3]={1,10,100};
unsigned wrong[2]={0,0},correct[2]={0,0};
for(unsigned k=0;k<3;++k)for(unsigned n=0;n<2;++n){
wrong[n]+=x[k]*recovered[k*2+n];
correct[n]+=x[k]*original[k*2+n];
}
assert(correct[0]==531 && correct[1]==642);
assert(wrong[0]==451 && wrong[1]==623);
std::cout<<"packed-as-legacy:451,623; original:531,642; mismatch reproduced\\n";
}
'''
        with tempfile.TemporaryDirectory() as directory:
            fixture, binary = Path(directory)/'layout.cpp', Path(directory)/'layout'
            fixture.write_text(program)
            built = subprocess.run(['g++','-std=c++17','-fsanitize=undefined',
                '-fno-sanitize-recover=all','-I'+str(ROOT/'cuda'),str(fixture),'-o',str(binary)],
                capture_output=True,text=True,timeout=30)
            self.assertEqual(built.returncode,0,built.stdout+built.stderr)
            run = subprocess.run([str(binary)],capture_output=True,text=True,timeout=10)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            print(run.stdout,end='')
