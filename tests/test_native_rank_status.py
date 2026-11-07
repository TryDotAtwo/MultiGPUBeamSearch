"""Compile and execute the production rank-status writer on a real filesystem."""
import json
from pathlib import Path
import shutil
import subprocess
import pytest


def test_terminal_rank_report_is_exclusive_and_complete(tmp_path):
    compiler = shutil.which('g++') or shutil.which('clang++')
    if not compiler:
        pytest.skip('C++ compiler required')
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / 'rank.cpp'
    source.write_text('''
#include "tools/native_rank_status.hpp"
int main() {
 beam::rank_status::publish("rank-status", 1, 7, true, 3);
 try { beam::rank_status::publish("rank-status", 1, 8, false, 5); }
 catch (const std::exception&) { return 0; }
 return 9;
}
''')
    binary = tmp_path / 'rank'
    build = subprocess.run([compiler, '-std=c++20', '-I', str(root), str(source),
                            '-o', str(binary)], capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    result = subprocess.run([str(binary)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / 'rank-status/rank-1.json').read_text())
    assert report == {'rank': 1, 'puzzle_id': 7, 'exit_code': 0,
                      'status': 'solved', 'completed_depths': 3}
    assert not list((tmp_path / 'rank-status').glob('*.tmp'))
