"""Actual prediction emitter: shared POSIX pipe and error handling, no CUDA."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import pytest

ROOT = Path(__file__).resolve().parents[1]
PREAMBLE = r'''
#include <unistd.h>
#include <sys/wait.h>
#include <cerrno>
#include <iostream>
#include <string>
#include <stdexcept>
'''

def emitter():
    source=(ROOT/'tools/production_runner.cu').read_text()
    return source[source.index('void emit_predict_record('):source.index('void log_predict_stats(')]

def run(body, flags=()):
    with tempfile.TemporaryDirectory() as directory:
        base=Path(directory)
        fixture,binary=base/'pipe.cpp',base/'pipe'
        fixture.write_text(PREAMBLE+emitter()+body)
        built=subprocess.run(['g++','-std=c++17',str(fixture),*flags,'-o',str(binary)],capture_output=True,text=True,timeout=30)
        assert built.returncode==0,built.stderr
        result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=15)
        assert result.returncode==0,result.stdout+result.stderr
        return result

pytestmark=pytest.mark.skipif(os.name!='posix' or not shutil.which('g++'),reason='remote POSIX compiler')

def test_actual_emitter_two_processes_shared_pipe():
    result=run(r'''
int main(){
int channel[2];if(pipe(channel))return 10;
pid_t children[2];
for(int rank=0;rank<2;++rank){
children[rank]=fork();if(children[rank]<0)return 11;
if(children[rank]==0){
close(channel[0]);if(dup2(channel[1],STDOUT_FILENO)<0)_exit(12);close(channel[1]);
for(int depth=0;depth<200;++depth)
emit_predict_record("rank="+std::to_string(rank)+" depth="+std::to_string(depth)+"\n");
_exit(0);
}}
close(channel[1]);std::string records;char buffer[97];ssize_t count;
while((count=read(channel[0],buffer,sizeof(buffer)))>0)records.append(buffer,count);
close(channel[0]);if(count<0)return 13;
for(auto child:children){int status;if(waitpid(child,&status,0)<0||!WIFEXITED(status)||WEXITSTATUS(status))return 14;}
std::cout<<records;
}
''')
    lines=result.stdout.splitlines()
    assert len(lines)==400
    assert set(lines)=={f'rank={rank} depth={depth}' for rank in (0,1) for depth in range(200)}

def test_actual_emitter_retries_eintr_and_rejects_short_error_oversize():
    result=run(r'''
int mode=0,calls=0;
extern "C" ssize_t __wrap_write(int,const void*,size_t n){
++calls;
if(mode==0&&calls==1){errno=EINTR;return -1;}
if(mode==1)return 1;
if(mode==2){errno=EIO;return -1;}
return n;
}
int main(){
emit_predict_record("test\n");if(calls!=2)return 20;
for(mode=1;mode<=2;++mode){bool threw=false;try{emit_predict_record("test\n");}catch(const std::runtime_error&){threw=true;}if(!threw)return 21;}
for(auto record:{std::string(),std::string("no-newline"),std::string(100000,'x')+"\n"}){
bool threw=false;try{emit_predict_record(record);}catch(const std::runtime_error&){threw=true;}if(!threw)return 22;
}
std::cout<<"ERROR_CONTRACT_OK\n";
}
''',('-Wl,--wrap=write',))
    assert result.stdout=='ERROR_CONTRACT_OK\n'
