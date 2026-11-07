"""Execute actual runner build-info branch, without CUDA or fabricated receipts.

Regression target: a timing-disabled binary cannot identify its diagnostic
mode, so an external benchmark can unknowingly compare unlike builds.
Remote-only verification policy applies; this test has not been run locally.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.name != 'posix' or not shutil.which('g++'), reason='remote POSIX g++ required')
@pytest.mark.parametrize('timing,depth_logs', [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_actual_build_info_exposes_effective_timing_and_depth_flags(timing, depth_logs, tmp_path):
    text = (ROOT / 'tools/production_runner.cu').read_text()
    start = text.index('    if (argc == 2 && std::string(argv[1]) == "--build-info") {')
    end = text.index('    if (argc != 4 && argc != 6)', start)
    actual = text[start:end]
    program = '''#include <iostream>
#include <string>
constexpr unsigned STATE_LEN=96, STATE_STORAGE_LEN=112, MOVE_COUNT=24;
struct alignas(32) CandidateMeta { unsigned char bytes[32]; };
int main(int argc, char** argv) {
''' + actual + '\nreturn 2;\n}\n'
    source = tmp_path / 'actual_build_info.cpp'
    source.write_text(program)
    binary = tmp_path / 'actual_build_info'
    subprocess.run(['g++', '-std=c++17', '-fsanitize=undefined',
                    f'-DBEAM_DEBUG_STREAM_TIMING={timing}',
                    f'-DBEAM_ENABLE_DEPTH_LOGS={depth_logs}',
                    '-DBEAM_ENABLE_DEBUG=1', str(source), '-o', str(binary)],
                   check=True, capture_output=True, text=True, timeout=60)
    result = subprocess.run([str(binary), '--build-info'], check=True,
                            capture_output=True, text=True, timeout=10)
    report = json.loads(result.stdout)
    assert report.get('debug_stream_timing') is bool(timing), report
    assert report.get('depth_logs_enabled') is bool(depth_logs), report
    assert report.get('debug_enabled') is True, report
    assert (report['state_len'], report['state_storage_len'],
            report['move_count'], report['candidate_meta_bytes']) == (96, 112, 24, 32)
