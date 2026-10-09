"""Execute the actual dispatcher queue lambda with deterministic event results."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'posix' and shutil.which('g++'), 'POSIX g++ required')
class Stream4SlotReleaseTests(unittest.TestCase):
    def test_finished_later_slots_are_released_without_waiting_for_oldest(self):
        source = (ROOT / 'cuda/dispatcher.cu').read_text()
        start = source.index('    const auto release_completed_stream4_slots_nonblocking =')
        end = source.index('    const auto wait_all_stream4_slots =', start)
        # Execute the real source body, not a copied queue algorithm. CUDA event
        # readiness is the only substituted dependency; no GPU claim is made.
        program = '''#include <deque>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <cassert>
using cudaError_t = int;
constexpr int cudaSuccess=0, cudaErrorNotReady=1;
std::vector<int> readiness{cudaErrorNotReady,cudaSuccess,cudaErrorNotReady,cudaSuccess};
int cudaEventQuery(int event) { return readiness.at(event); }
void check_cuda(int, const char*) { throw std::runtime_error("CUDA query error"); }
int main() {
std::deque<unsigned> stream4_busy_slots{0,1,2,3};
struct { int stream4_slot_done[4]{0,1,2,3}; } streams;
std::vector<unsigned> released;
auto mark_stream4_slot_complete = [&](unsigned slot) { released.push_back(slot); };
''' + source[start:end] + '''
assert(release_completed_stream4_slots_nonblocking());
assert((released == std::vector<unsigned>{1,3}));
assert((stream4_busy_slots == std::deque<unsigned>{0,2}));
assert(!release_completed_stream4_slots_nonblocking());
assert((released == std::vector<unsigned>{1,3}));
// Empty, completely blocked and completely finished bounded queues.
stream4_busy_slots.clear(); released.clear();
assert(!release_completed_stream4_slots_nonblocking());
readiness = {1,1,1,1}; stream4_busy_slots={0,1,2,3};
assert(!release_completed_stream4_slots_nonblocking());
assert((stream4_busy_slots == std::deque<unsigned>{0,1,2,3}));
readiness = {0,0,0,0};
assert(release_completed_stream4_slots_nonblocking());
assert(stream4_busy_slots.empty());
assert((released == std::vector<unsigned>{0,1,2,3}));
// A query error propagates: never mark the errored slot complete or lose it.
released.clear(); stream4_busy_slots={0,1,2,3}; readiness={1,0,7,0};
bool threw=false;
try { release_completed_stream4_slots_nonblocking(); }
catch(const std::runtime_error&) { threw=true; }
assert(threw);
assert((released == std::vector<unsigned>{1}));
assert((stream4_busy_slots == std::deque<unsigned>{0,2,3}));
}
'''
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / 'probe.cpp'
            binary = Path(directory) / 'probe'
            fixture.write_text(program)
            build = subprocess.run(['g++', '-std=c++20', '-fsanitize=undefined',
                '-fno-sanitize-recover=undefined', str(fixture), '-o', str(binary)],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
