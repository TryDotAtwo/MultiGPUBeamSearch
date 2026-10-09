"""Exercise actual stdout renderer against an interleaving shared sink.

GPU summaries/JSON writer are substituted; this checks record framing only.
The sink forces two old prefix writes to meet. A process-local logging mutex
cannot satisfy this test: it deadlocks at that rendezvous instead of protecting
records emitted by separate processes. Complete-record output bypasses it.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import pytest

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.skipif(os.name!='posix' or not shutil.which('g++'),reason='remote POSIX compiler')
@pytest.mark.parametrize('verbose', [0,1,10])
def test_actual_prediction_stdout_preserves_complete_records(verbose):
    source=(ROOT/'tools/production_runner.cu').read_text()
    start=source.index('void log_predict_stats(')
    end=source.index('std::uint32_t parse_next_u32(',start)
    program=r'''
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <string_view>
#include <sstream>
#include <unistd.h>
#include <cerrno>
#include <limits.h>
#include <stdexcept>
#include <atomic>
std::atomic<unsigned> writes{0};
extern "C" ssize_t __wrap_write(int,const void* p,size_t n){++writes;std::cout.write(static_cast<const char*>(p),n);return n;}
#include <mutex>
#include <condition_variable>
#include <thread>
struct CandidateMeta{};
struct FinalizeDepthState{std::uint32_t final_candidate_count=2,final_threshold=7;};
struct PredictStatsConfig{unsigned verbose=1;std::filesystem::path jsonl_path;unsigned rank=0,device_local_rank=0;};
struct PredictStatsSummary{
std::uint64_t count=23;
std::uint32_t min_key=1,p01_key=1,p05_key=1,p50_key=1,p95_key=1,p99_key=1,max_key=1;
double mean_key=1;
};
double score_key_to_score(double x){return x;}
PredictStatsSummary summarize_score_hist(const std::vector<std::uint64_t>&){return {};}
PredictStatsSummary summarize_frontier_scores(const CandidateMeta*,unsigned){return {};}
template<class... T>void write_predict_stats_jsonl(T...){}
class SharedSink:public std::streambuf{
std::mutex mutex;std::condition_variable ready;unsigned prefixes=0;
std::string data;
protected:
std::streamsize xsputn(const char* p,std::streamsize n)override{
std::unique_lock<std::mutex> lock(mutex);
data.append(p,static_cast<std::size_t>(n));
if(std::string_view(p,static_cast<std::size_t>(n))=="predict_stats"){
++prefixes;ready.notify_all();ready.wait(lock,[&]{return prefixes==2;});
}
lock.unlock();std::this_thread::yield();return n;
}
int overflow(int value)override{
if(value!=traits_type::eof()){
std::lock_guard<std::mutex> lock(mutex);data.push_back(static_cast<char>(value));
}
return traits_type::not_eof(value);
}
public:
std::string result(){std::lock_guard<std::mutex> lock(mutex);return data;}
};
'''+(source[source.index('void emit_predict_record('):start] if 'void emit_predict_record(' in source else '')+source[start:end]+r'''
int main(){
SharedSink sink;auto* original=std::cout.rdbuf(&sink);
const PredictStatsConfig config;const FinalizeDepthState final_state;
const std::vector<std::uint64_t> histogram{23};
std::thread a([&]{log_predict_stats(config,0,histogram,nullptr,final_state,23);});
std::thread b([&]{log_predict_stats(config,1,histogram,nullptr,final_state,23);});
a.join();b.join();std::cout.rdbuf(original);std::cout<<sink.result();
std::cerr<<"atomic_writes="<<writes.load()<<"\n";
}
'''
    with tempfile.TemporaryDirectory() as directory:
        fixture,binary=Path(directory)/'records.cpp',Path(directory)/'records'
        fixture.write_text(program.replace('const PredictStatsConfig config;',f'PredictStatsConfig config;config.verbose={verbose};'))
        build=subprocess.run(['g++','-std=c++17','-pthread','-fsanitize=undefined',
            '-fno-sanitize-recover=all','-Wl,--wrap=write',str(fixture),'-o',str(binary)],
            capture_output=True,text=True,timeout=30)
        assert build.returncode==0,build.stdout+build.stderr
        result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=10)
        assert result.returncode==0,result.stdout+result.stderr
        expected_writes = 0 if verbose==0 else 4 if verbose>=10 else 2
        assert f'atomic_writes={expected_writes}\n' == result.stderr, result.stderr
    lines=result.stdout.splitlines()
    if verbose==0:
        assert lines==[]
        return
    if verbose>=10:
        details=[line for line in lines if line.startswith('predict_stats_detail ')]
        assert len(details)==2,result.stdout
        expected_detail={'depth','frontier_p01','frontier_p05','frontier_p50','frontier_p95','frontier_p99','frontier_max','selected_ratio'}
        for line in details:
            fields=[token.split('=',1)[0] for token in line.split()[1:]]
            assert len(fields)==len(expected_detail) and set(fields)==expected_detail,line
        lines=[line for line in lines if line.startswith('predict_stats ')]
    assert len(lines)==2,result.stdout
    seen=set()
    expected={'depth','generated_count','hist_count','score_min','score_p01','score_p05',
        'score_mean','score_p50','score_p95','score_p99','score_max','best_frontier_min',
        'best_frontier_mean','threshold'}
    for line in lines:
        assert line.startswith('predict_stats '),line
        tokens=line.split()[1:]
        assert all(re.fullmatch(r'[a-z0-9_]+=[0-9.]+',token) for token in tokens),line
        fields=dict(token.split('=',1) for token in tokens)
        assert len(tokens)==len(expected) and set(fields)==expected,line
        assert fields['generated_count']==fields['hist_count']=='23',line
        assert fields['threshold']=='7',line
        seen.add(fields['depth'])
    assert seen=={'0','1'}
