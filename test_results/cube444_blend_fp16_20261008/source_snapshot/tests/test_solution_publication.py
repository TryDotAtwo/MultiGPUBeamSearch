"""Compile the production writer with inert formatting/history dependencies."""
from pathlib import Path
import shutil
import subprocess
import pytest


@pytest.mark.parametrize('valid', [False, True])
def test_solution_publication_from_fresh_directory(tmp_path, valid):
    compiler = shutil.which('g++') or shutil.which('clang++')
    if not compiler:
        pytest.skip('C++ compiler required')
    root = Path(__file__).resolve().parents[1]
    source = (root / 'tools/production_runner.cu').read_text()
    start = source.index('void write_solution_artifacts(')
    end = source.index('\n#if BEAM_DEBUG_INFERENCE_TRACE', start)
    writer = source[start:end]
    # The writer itself is unchanged real production code. Dependencies only
    # format inert values; expected no-file behavior does not depend on them.
    harness = r'''
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#define BEAM_ENABLE_DEBUG_LOGS 0
struct State128 {};
struct ReconstructedSolution { std::vector<unsigned> moves; };
struct CpuCandidateHistory {
 int mode=0; std::filesystem::path dir; std::vector<unsigned> depth_counts;
 unsigned bytes_received=0, bytes_stored=0, bytes_stored_ram=0,
 bytes_stored_disk=0, bytes_pruned=0;
};
struct SolvedSnapshot { unsigned count=1, overflow=0; };
std::string moves_to_path_text(const std::vector<unsigned>&, const std::vector<std::string>&) { return "f0"; }
std::string state_to_text(const State128&) { return "0"; }
std::string timestamp_id() { return "fixture"; }
std::string history_mode_name(int) { return "ram"; }
'''
    harness += writer
    harness += r'''
int main() {
 try { write_solution_artifacts(1, 5, 4096, {}, {}, {}, VALID, {}, {}); }
 catch (const std::exception&) {
   return VALID || std::filesystem::exists("submit.csv") ||
          std::filesystem::exists("test_results") ? 3 : 0;
 }
 return VALID && std::filesystem::exists("submit.csv") &&
        std::filesystem::exists("test_results") ? 0 : 4;
}
'''
    harness = harness.replace('VALID', 'true' if valid else 'false')
    cpp = tmp_path / 'writer.cpp'
    cpp.write_text(harness)
    binary = tmp_path / 'writer'
    subprocess.run([compiler, '-std=c++20', str(cpp), '-o', str(binary)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, 'invalid solution published or was not rejected'
