"""Actual wait/acquire source with deterministic CUDA API status substitution."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.name != 'posix' or not shutil.which('g++'), reason='POSIX g++ required')
@pytest.mark.parametrize('mode', ['empty_wait', 'empty_acquire', 'single_error', 'later_ready'])
def test_blocking_slot_preserves_queue_ownership_and_error_contract(mode, tmp_path):
    source = (ROOT / 'cuda/dispatcher.cu').read_text()
    start = source.index('    const auto release_completed_stream4_slots_nonblocking =')
    end = source.index('    const auto read_current_threshold_host =', start)
    actual = source[start:end]
    waiting = [name for name in re.findall(r'const auto (wait_\w+) =', actual)
               if name != 'wait_all_stream4_slots']
    assert len(waiting) == 1
    program = r'''
#include <cassert>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using cudaError_t = int;
struct NvtxRange { explicit NvtxRange(const char*) {} };
constexpr int cudaSuccess=0, cudaErrorNotReady=1;
std::string mode;
unsigned later_queries=0;
int cudaEventQuery(int event) {
    if (mode == "later_ready" && event == 1 && ++later_queries >= 2) return cudaSuccess;
    return cudaErrorNotReady;
}
int cudaEventSynchronize(int) {
    if (mode == "single_error") return 7;
    throw std::runtime_error("must not synchronize unready oldest while later can finish");
}
void check_cuda(int code, const char*) {
    if (code != cudaSuccess) throw std::runtime_error("CUDA error");
}
int main(int argc, char** argv) {
    mode=argv[1];
    struct { int stream4_slot_done[2]{0,1}; } streams;
    struct { struct { unsigned stream4_active_sort_slots=2; } config; } plan;
    std::deque<unsigned> stream4_busy_slots, stream4_free_slots;
    std::vector<unsigned> released;
    auto mark_stream4_slot_complete = [&](unsigned slot) {
        released.push_back(slot); stream4_free_slots.push_back(slot);
    };
''' + actual + f'''
    if (mode == "empty_wait") {{
        assert(!{waiting[0]}());
        assert(released.empty());
    }} else if (mode == "empty_acquire") {{
        bool failed=false;
        try {{ acquire_stream4_slot_blocking(); }} catch(const std::runtime_error&) {{ failed=true; }}
        assert(failed && released.empty() && stream4_free_slots.empty());
    }} else if (mode == "single_error") {{
        stream4_busy_slots={{0}};
        bool failed=false;
        try {{ acquire_stream4_slot_blocking(); }} catch(const std::runtime_error&) {{ failed=true; }}
        assert(failed && released.empty() && stream4_free_slots.empty());
        assert((stream4_busy_slots == std::deque<unsigned>{{0}}));
    }} else {{
        stream4_busy_slots={{0,1}};
        assert(acquire_stream4_slot_blocking()==1);
        assert((released == std::vector<unsigned>{{1}}));
        assert((stream4_busy_slots == std::deque<unsigned>{{0}}));
        assert(stream4_free_slots.empty());
    }}
}}
'''
    fixture = tmp_path / 'blocking_host.cpp'
    binary = tmp_path / 'blocking_host'
    fixture.write_text(program)
    build = subprocess.run(['g++', '-std=c++20', '-fsanitize=undefined',
                            '-fno-sanitize-recover=undefined', str(fixture), '-o', str(binary)],
                           capture_output=True, text=True, timeout=30)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(binary), mode], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
