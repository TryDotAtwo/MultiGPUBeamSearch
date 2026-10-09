"""Opt-in target-GPU integration: production-format Cube4 inputs, no local GPU use."""
import os
import subprocess
from pathlib import Path
import pytest


def test_benchmark_accepts_verified_production_format_inputs(tmp_path):
    binary = os.environ.get('BEAM_TEST_STREAM_BENCH_BINARY')
    data = os.environ.get('BEAM_TEST_CUBE4_DATA')
    weights = os.environ.get('BEAM_TEST_CUBE4_WEIGHTS')
    if not all((binary, data, weights)):
        pytest.skip('requires explicit target-GPU binary, Cube4 data and weights')
    env = dict(os.environ,
        BEAM_GENERATOR_PATH=str(Path(data) / 'puzzle_info.json'),
        BEAM_PUZZLE_INFO_JSON=str(Path(data) / 'puzzle_info.json'),
        BEAM_TEST_CSV=str(Path(data) / 'test.csv'),
        BEAM_WEIGHT_DIR=weights,
        BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH='1',
        BEAM_STREAM1_TRANSFORMER_B_MICRO='256',
        BEAM_STREAM1_TRANSFORMER_CONCURRENCY='2',
        BEAM_STREAM1_TRANSFORMER_BENCH_ITERS='20',
        BEAM_STREAM_BENCH_REPORT=str(tmp_path / 'report.md'))
    result = subprocess.run([binary, '1000'], cwd=tmp_path, env=env,
        text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'stream1_transformer_benchmark_done=1' in result.stdout
    assert 'parents_per_sec=' in result.stdout


def test_native_graph_runner_accepts_fixed_addressing(tmp_path):
    binary = os.environ.get('BEAM_TEST_PRODUCTION_BINARY')
    data = os.environ.get('BEAM_TEST_CUBE4_DATA')
    weights = os.environ.get('BEAM_TEST_CUBE4_WEIGHTS')
    if not all((binary, data, weights)):
        pytest.skip('requires explicit target-GPU production runner and inputs')
    env = dict(os.environ,
        BEAM_GENERATOR_PATH=str(Path(data) / 'puzzle_info.json'),
        BEAM_PUZZLE_INFO_JSON=str(Path(data) / 'puzzle_info.json'),
        BEAM_TEST_CSV=str(Path(data) / 'test.csv'), BEAM_WEIGHT_DIR=weights,
        BEAM_B_MICRO='256', BEAM_STREAM1_CONCURRENCY='2',
        BEAM_STREAM1_EXECUTOR='native_cuda_graph', BEAM_HISTORY_MODE='ram',
        BEAM_HISTORY_RAM_BYTES='33554432')
    env.pop('BEAM_RING_GRAPH_EXECS_PER_LANE', None)
    result = subprocess.run([binary, '1000', '1', '262144', '1', '0'],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'runtime_ring_slot_graph_windowed=0' in result.stdout
    assert 'depth_done=0' in result.stdout
