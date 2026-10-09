"""Catch shared JSONL ownership and missing rank/device attribution.

Compile the actual config builder and writer; no CUDA summaries are executed.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.skipif(os.name != 'posix' or not shutil.which('g++'), reason='remote POSIX compiler')
def test_two_processes_write_disjoint_attributed_jsonl():
    source = (ROOT / 'tools/production_runner.cu').read_text()
    structs = source[source.index('struct PredictStatsConfig {'):source.index('PredictStatsConfig predict_stats_config_from_env(')]
    builder = source[source.index('PredictStatsConfig predict_stats_config_from_env('):source.index('double score_key_to_score(', source.index('PredictStatsConfig predict_stats_config_from_env('))]
    writer_end = source.index('void emit_predict_record(') if 'void emit_predict_record(' in source else source.index('void log_predict_stats(')
    writer = source[source.index('void write_predict_stats_jsonl('):writer_end]
    program = r'''
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <stdexcept>
unsigned env_u32(const char* name,unsigned fallback){auto p=std::getenv(name);return p?std::stoul(p):fallback;}
unsigned env_or_default_u32(const char* name,unsigned fallback){return env_u32(name,fallback);}
std::filesystem::path env_path(const char* name,const char* fallback){auto p=std::getenv(name);return p?p:fallback;}
double score_key_to_score(double value){return value;}
''' + structs + builder + writer + r'''
int main(){
auto config=predict_stats_config_from_env();
PredictStatsSummary score;score.count=23;
for(unsigned depth=0;depth<32;++depth)
write_predict_stats_jsonl(config.jsonl_path,depth,score,score,7,2,23,config.rank,config.device_local_rank);
}
'''
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        fixture, binary = base/'writer.cpp', base/'writer'
        fixture.write_text(program)
        build = subprocess.run(['g++','-std=c++17',str(fixture),'-o',str(binary)], capture_output=True,text=True,timeout=30)
        assert build.returncode == 0, build.stderr
        processes = []
        for rank in (0,1):
            env = dict(os.environ, RANK=str(rank), LOCAL_RANK=str(rank), BEAM_PREDICT_STATS_VERBOSE='1', BEAM_PREDICT_STATS_PATH=str(base/'stats.jsonl'))
            processes.append(subprocess.Popen([str(binary)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
        for process in processes:
            stdout,stderr=process.communicate(timeout=10)
            assert process.returncode == 0, stdout+stderr
        assert not (base/'stats.jsonl').exists(), 'ranks still share the requested path'
        for rank in (0,1):
            records = [json.loads(line) for line in (base/f'stats.jsonl.rank{rank}.jsonl').read_text().splitlines()]
            assert len(records) == 32
            assert [record['depth'] for record in records] == list(range(32))
            assert all(record['rank']==rank and record['device_local_rank']==rank for record in records)
