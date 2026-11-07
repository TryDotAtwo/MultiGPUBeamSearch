"""Execute actual QKV dispatch prefix; kernel execution is substituted."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'posix' and shutil.which('g++'), 'POSIX g++ required')
class HopperLayoutDispatchTests(unittest.TestCase):
    def test_loaded_layout_controls_relu_and_silu_ff1(self):
        source = (ROOT/'cuda/stream1_transformer.cu').read_text()
        start = source.index('void stream1_transformer_ff1_linear_bias_activation_impl(')
        end = source.index('    if (dtype == STREAM1_DTYPE_BF16)', start)
        prefix = '''#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <cassert>
using half=std::uint16_t; using cudaStream_t=int;
constexpr unsigned STREAM1_DTYPE_FP16=1;
int sm=90,hopper=0,legacy=0;
int stream1_transformer_current_device_sm(){return sm;}
namespace cute {struct _128{};struct _64{};template<class... T>struct Shape{};}
namespace cutlass {namespace epilogue {namespace thread {
template<class T>struct ReLu{};template<class T>struct SiLu{};
}}}
template<template<class>class A,class Tile=void>
void stream1_transformer_hopper_fp16_bias_activation(const half*,const half*,
 const half*,half*,unsigned,unsigned,unsigned,int){++hopper;}
template<typename Activation>
'''
        prefix = ('#include "' + (ROOT/'cuda/stream1_transformer_hopper_policy.hpp').as_posix() + '"\n'
                  '#include "' + (ROOT/'cuda/stream1_policy_snapshot.hpp').as_posix() + '"\n'
                  '#include "' + (ROOT/'cuda/stream1_resolved_launch.hpp').as_posix() + '"\n'
                  'using beam::stream1_policy_value;\n'
                  'using beam::stream1_resolved_hopper_ff1_epilogue;\n' + prefix)
        program=prefix+source[start:end]+'''++legacy;}
template<typename A>void check(){
for(unsigned rows:{57U,4096U}){
hopper=legacy=0;sm=90;
stream1_transformer_ff1_linear_bias_activation_impl<A>(nullptr,nullptr,nullptr,nullptr,rows,256,1024,1,0,false);
assert(legacy==1 && hopper==0);
bool rejected=false;
try{stream1_transformer_ff1_linear_bias_activation_impl<A>(nullptr,nullptr,nullptr,nullptr,rows,256,1024,1,0,true);}
catch(const std::invalid_argument&){rejected=true;}
#if BEAM_HAS_CUTLASS
assert(!rejected && hopper==1 && legacy==1);
#else
assert(rejected && hopper==0 && legacy==1);
#endif
for(unsigned dtype:{1U,2U}){
sm=dtype==1 ? 86 : 90;rejected=false;
const int before=hopper;
try{stream1_transformer_ff1_linear_bias_activation_impl<A>(nullptr,nullptr,nullptr,nullptr,rows,256,1024,dtype,0,true);}
catch(const std::invalid_argument&){rejected=true;}
assert(rejected && legacy==1 && hopper==before);
}
}}
int main(){check<cutlass::epilogue::thread::ReLu<float>>();
check<cutlass::epilogue::thread::SiLu<float>>();}
'''
        with tempfile.TemporaryDirectory() as directory:
            fixture,binary=Path(directory)/'ff1.cpp',Path(directory)/'ff1'
            fixture.write_text(program)
            for enabled in (0,1):
                build=subprocess.run(['g++','-std=c++17','-fsanitize=undefined',
                    '-fno-sanitize-recover=all',f'-DBEAM_HAS_CUTLASS={enabled}',
                    '-DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1',str(fixture),'-o',str(binary)],
                    capture_output=True,text=True,timeout=30)
                self.assertEqual(build.returncode,0,build.stdout+build.stderr)
                run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=10)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)

    def test_loaded_layout_controls_small_and_full_qkv(self):
        source = (ROOT/'cuda/stream1_transformer.cu').read_text()
        start = source.index('void stream1_transformer_linear_bias_cuda(')
        end = source.index('    if (dtype == STREAM1_DTYPE_BF16)', start)
        prefix = '''#include <cstdint>
#include <stdexcept>
#include <cassert>
using half=std::uint16_t; using cudaStream_t=int;
constexpr unsigned STREAM1_DTYPE_FP16=1;
int sm=90,hopper=0,legacy=0;
int stream1_transformer_current_device_sm(){return sm;}
namespace cutlass { namespace epilogue { namespace thread {
template<class T>struct Identity{};
}}}
template<template<class>class A>
void stream1_transformer_hopper_fp16_bias_activation(const half*,const half*,
 const half*,half*,unsigned,unsigned,unsigned,int){++hopper;}
'''
        prefix = '#include "' + (ROOT/'cuda/stream1_transformer_hopper_policy.hpp').as_posix() + '"\n' + prefix
        program = prefix + source[start:end] + '''++legacy;}
int main(){
for(unsigned rows:{57U,4096U}){
sm=90;hopper=legacy=0;
stream1_transformer_linear_bias_cuda(nullptr,nullptr,nullptr,nullptr,rows,256,768,1,0,false);
assert(legacy==1 && hopper==0);
bool rejected=false;
try{stream1_transformer_linear_bias_cuda(nullptr,nullptr,nullptr,nullptr,rows,256,768,1,0,true);}
catch(const std::invalid_argument&){rejected=true;}
#if BEAM_HAS_CUTLASS
assert(!rejected && hopper==1 && legacy==1);
#else
assert(rejected && hopper==0 && legacy==1);
#endif
sm=86;rejected=false;
try{stream1_transformer_linear_bias_cuda(nullptr,nullptr,nullptr,nullptr,rows,256,768,1,0,true);}
catch(const std::invalid_argument&){rejected=true;}
assert(rejected && legacy==1);
sm=90;rejected=false;const int before=hopper;
try{stream1_transformer_linear_bias_cuda(nullptr,nullptr,nullptr,nullptr,rows,256,768,2,0,true);}
catch(const std::invalid_argument&){rejected=true;}
assert(rejected && legacy==1 && hopper==before);
}}
'''
        with tempfile.TemporaryDirectory() as directory:
            fixture,binary=Path(directory)/'dispatch.cpp',Path(directory)/'dispatch'
            fixture.write_text(program)
            for enabled in (0,1):
                build=subprocess.run(['g++','-std=c++17','-fsanitize=undefined',
                    '-fno-sanitize-recover=all',f'-DBEAM_HAS_CUTLASS={enabled}',
                    '-DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1',str(fixture),'-o',str(binary)],
                    capture_output=True,text=True,timeout=30)
                self.assertEqual(build.returncode,0,build.stdout+build.stderr)
                run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=10)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
