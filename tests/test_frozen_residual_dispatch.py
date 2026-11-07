"""Execute actual residual dispatch prefix with substituted GPU launches."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import pytest
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.skipif(os.name!='posix' or not shutil.which('g++'),reason='remote POSIX compiler')
def test_actual_residual_dispatch_consumes_frozen_epilogue_and_swizzle():
    source=(ROOT/'cuda/stream1_transformer.cu').read_text()
    start=source.index('void stream1_transformer_residual_bias_round_layernorm_cuda(')
    end=source.index('    if (stream1_transformer_current_device_sm80_or_newer())',start)
    prefix=f'''#include "{(ROOT/'cuda/stream1_resolved_gemm.hpp').as_posix()}"
#include "{(ROOT/'cuda/stream1_policy_snapshot.hpp').as_posix()}"
#include <cassert>
#include <cstdint>
using namespace beam;
using half=std::uint16_t; using cudaStream_t=int;
constexpr unsigned STREAM1_DTYPE_FP16=1;
int separate=0,normalized=0,fused=0;
int stream1_transformer_current_device_sm(){{return 86;}}
template<class... A>void stream1_transformer_linear_residual_cuda(A...){{++separate;}}
template<class... A>void stream1_transformer_bias_round_layernorm_copy_launch(A...){{++normalized;}}
'''
    program=prefix+source[start:end]+'''++fused;}
int main(){
for(const char* key:{"BEAM_STREAM1_TRANSFORMER_FF2_POLICY","BEAM_STREAM1_TRANSFORMER_FF2_EPILOGUE",
"BEAM_STREAM1_TRANSFORMER_FF2_SWIZZLE","BEAM_STREAM1_TRANSFORMER_ATTN_OUT_POLICY",
"BEAM_STREAM1_TRANSFORMER_ATTN_OUT_EPILOGUE","BEAM_STREAM1_TRANSFORMER_ATTN_OUT_SWIZZLE"})
setenv(key,"changed-invalid",1);
for(bool ff2:{false,true})for(bool fused_policy:{false,true}){
Stream1ResolvedGemmSet choices;choices.sm=86;choices.fp16=true;
auto& choice=choices.families[static_cast<unsigned>(ff2?Stream1TransformerGemmFamily::Ff2:Stream1TransformerGemmFamily::AttentionOut)];
choice.policy=Stream1TransformerGemmPolicy::M128N128;
choice.epilogue=fused_policy?Stream1TransformerResidualEpiloguePolicy::FusedBiasRound:Stream1TransformerResidualEpiloguePolicy::Separate;
choice.swizzle=fused_policy?Stream1TransformerGemmSwizzlePolicy::Identity2:Stream1TransformerGemmSwizzlePolicy::Identity1;
Stream1ResolvedGemmScope scope(choices);
separate=normalized=fused=0;
stream1_transformer_residual_bias_round_layernorm_cuda(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,13,ff2?1024:256,256,1,0);
assert(separate==(!fused_policy) && normalized==(!fused_policy) && fused==fused_policy);
}
}
'''
    with tempfile.TemporaryDirectory() as directory:
        fixture,binary=Path(directory)/'residual.cpp',Path(directory)/'residual'
        fixture.write_text(program)
        build=subprocess.run(['g++','-std=c++17','-fsanitize=undefined',
            '-fno-sanitize-recover=all',str(fixture),'-o',str(binary)],
            capture_output=True,text=True,timeout=30)
        assert build.returncode==0,build.stdout+build.stderr
        run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=10)
        assert run.returncode==0,run.stdout+run.stderr
