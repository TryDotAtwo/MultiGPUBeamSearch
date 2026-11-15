"""Pair source validators in one Rust-owned, two-GPU public session."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from multigpubeamsearch import NativeModel, NativeOptions, NativeSession, models
from multigpubeamsearch.errors import NativeBackendError
from public_model_snapshot_gpu import graph_fixture, large_hamming_artifact, start_state


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'cutlass', 'cache', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    graph, contract, moves = graph_fixture()
    weights = large_hamming_artifact(contract, root)
    model = NativeModel.for_graph(graph, weights)
    size = sum(path.stat().st_size for path in weights.iterdir() if path.is_file())
    tensor_bytes = size - (weights / 'manifest.json').stat().st_size
    start = start_state(contract.center, moves, 12)
    options = NativeOptions(source_dir=args.source, cutlass_dir=args.cutlass, cache_dir=args.cache,
        num_gpus=2, inference_backend='cutlass', autotune=False, report_calibration=False,
        build_jobs=2, build_timeout_seconds=1800, timeout_seconds=180)
    session = NativeSession(engine='rust', experimental_reuse_model_snapshot=True,
                            experimental_parallel_nccl_identity=True, experimental_rust_mlp_finite=True)
    cache = session.model_snapshot_cache
    rust_validator = cache.finite_validator()
    assert callable(rust_validator)
    rows = []
    mode = None
    calls = []
    validations = []
    original_validate = models._validate_artifact

    def validate_chunk(data, dtype):
        calls.append(len(data))
        return rust_validator(data, dtype)

    def measured_validate(path, *args, **kwargs):
        started = time.perf_counter()
        try:
            return original_validate(path, *args, **kwargs)
        finally:
            if Path(path).resolve() == weights.resolve():
                validations.append(time.perf_counter() - started)

    def run(selected, phase, pair=None):
        nonlocal mode
        mode = selected
        calls.clear()
        validations.clear()
        before = time.perf_counter()
        result = session.beam_search(graph, backend='native', native_options=options,
            start_state=start, predictor=model, beam_width=4096, max_steps=6, return_path=True)
        elapsed = time.perf_counter() - before
        meta = result.native_metadata
        assert not result.path_found and result.path is None
        assert meta['effective_beam_width'] == 4096
        assert meta['persistent_session']['reused'] == (phase != 'cold')
        assert meta['model_snapshot']['owner'] == 'rust'
        assert meta['model_snapshot']['reused'] == (phase != 'cold')
        assert meta['model_snapshot']['copied_bytes'] == (size if phase == 'cold' else 0)
        assert meta['mlp_host_binding'] == {'artifact_hash': meta['model_hash'], 'ranks': 2}
        assert len(validations) == 1
        assert bool(calls) == (selected == 'rust')
        assert sum(calls) == (tensor_bytes if selected == 'rust' else 0)
        rows.append(dict(mode=selected, phase=phase, pair=pair, wall_seconds=elapsed,
            validation_seconds=validations[0], rust_checked_bytes=sum(calls),
            worker_seconds=meta['elapsed_seconds'], model_hash=meta['model_hash'],
            pids=meta['persistent_session']['worker_pids'], run_dir=meta['run_dir'],
            execution_parameters=meta['execution_parameters']))
        print(f'FINITE_PAIR phase={phase} pair={pair} mode={selected} wall={elapsed:.6f} '
              f'validation={validations[0]:.6f}', flush=True)

    cache.finite_validator = lambda: validate_chunk if mode == 'rust' else None
    models._validate_artifact = measured_validate
    try:
        run('rust', 'cold')
        run('torch', 'warmup')
        run('rust', 'warmup')
        for pair in range(5):
            for selected in (('torch', 'rust') if pair % 2 == 0 else ('rust', 'torch')):
                run(selected, 'repeated', pair)
        # A non-finite source must be rejected before a resident worker is submitted.
        bias = weights / 'output_bias.fp16'
        saved = bias.read_bytes()
        bias.write_bytes(b'\x00\x7c')
        mode = 'rust'
        calls.clear()
        try:
            session.beam_search(graph, backend='native', native_options=options,
                start_state=start, predictor=model, beam_width=4096, max_steps=6, return_path=True)
        except NativeBackendError as error:
            assert 'non-finite' in str(error) and calls
        else:
            raise AssertionError('non-finite source was accepted')
        finally:
            bias.write_bytes(saved)
        run('rust', 'recovery')
        gpu_memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
            '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    finally:
        models._validate_artifact = original_validate
        session.close()
    assert len({tuple(row['pids']) for row in rows}) == 1
    assert all(not Path(f'/proc/{pid}').exists() for pid in rows[0]['pids'])
    assert len({row['model_hash'] for row in rows}) == 1
    assert len({json.dumps(row['execution_parameters'], sort_keys=True) for row in rows}) == 1
    repeats = [row for row in rows if row['phase'] == 'repeated']
    medians = {mode: {metric: statistics.median(row[metric] for row in repeats if row['mode'] == mode)
                     for metric in ('wall_seconds', 'worker_seconds', 'validation_seconds')}
               for mode in ('torch', 'rust')}
    changes = []
    for pair in range(5):
        old = next(row for row in repeats if row['pair'] == pair and row['mode'] == 'torch')
        new = next(row for row in repeats if row['pair'] == pair and row['mode'] == 'rust')
        changes.append(new['wall_seconds'] / old['wall_seconds'] - 1)
    (root / 'results.json').write_text(json.dumps(dict(
        scope='five alternating pairs of complete public calls in the same resident Rust session; '
              'CPU source-weight validation changes, GPU execution remains the same',
        model='synthetic Hamming FP16 width 512, 8 residual blocks', model_bytes=size,
        checked_tensor_bytes=tensor_bytes, nonfinite_rejected_and_session_reused=True,
        width=4096, max_steps=6, num_gpus=2, gpu_memory_mib=gpu_memory,
        median_seconds=medians, paired_wall_changes=changes,
        median_paired_wall_change=statistics.median(changes), rows=rows), indent=2) + '\n')
    print('PUBLIC_RUST_MLP_FINITE_GPU_OK', flush=True)


if __name__ == '__main__':
    main()
