"""Compare Python and Rust launch verification with the same Rust snapshot owner."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

import torch

from multigpubeamsearch import NativeModel, NativeOptions, NativeSession
from multigpubeamsearch import backend, models
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
    model_bytes = sum(path.stat().st_size for path in weights.iterdir() if path.is_file())
    options = NativeOptions(source_dir=args.source, cutlass_dir=args.cutlass, cache_dir=args.cache,
        num_gpus=2, inference_backend='cutlass', autotune=False, report_calibration=False,
        build_jobs=2, build_timeout_seconds=1800, timeout_seconds=180)
    rows = []
    modes = ('python_verify', 'rust_verify')
    for pair in range(5):
        for mode in (modes if pair % 2 == 0 else tuple(reversed(modes))):
            free_before = [torch.cuda.mem_get_info(device)[0] for device in (0, 1)]
            counts = {'launch_checks': 0, 'python_checks': 0, 'rust_checks': 0, 'backend_checks': 0}
            with NativeSession(engine='rust', experimental_parallel_nccl_identity=True,
                               experimental_reuse_model_snapshot=True) as session:
                cache = session.model_snapshot_cache
                rust_verify = cache.verify_for_launch
                python_verify = models.verify_prepared_model
                backend_verify = backend.verify_prepared_model
                def validate_python(*values):
                    counts['python_checks'] += 1
                    return python_verify(*values)
                def validate_launch(*values):
                    counts['launch_checks'] += 1
                    if mode == 'python_verify':
                        return False
                    result = rust_verify(*values)
                    assert result, 'Rust fast verification was not used'
                    counts['rust_checks'] += 1
                    return result
                def validate_backend(*values):
                    counts['backend_checks'] += 1
                    return backend_verify(*values)
                cache.verify_for_launch = validate_launch
                models.verify_prepared_model = validate_python
                backend.verify_prepared_model = validate_backend
                try:
                    phases = [('cold', 12), ('warmup', 12), ('repeat', 12)]
                    if pair == 0:
                        phases.append(('solved', 3))
                    for phase, move_count in phases:
                        state = start_state(contract.center, moves, move_count)
                        begin = time.monotonic()
                        result = session.beam_search(graph, native_options=options, start_state=state,
                            predictor=model, beam_width=4096, max_steps=6, return_path=True)
                        wall = time.monotonic() - begin
                        meta = result.native_metadata
                        if phase == 'cold':
                            assert counts['backend_checks'] >= 1
                            cold_backend_checks = counts['backend_checks']
                        else:
                            assert counts['backend_checks'] == cold_backend_checks
                        snapshot, resident = meta['model_snapshot'], meta['persistent_session']
                        assert snapshot['owner'] == 'rust'
                        assert snapshot['reused'] == (phase != 'cold')
                        assert snapshot['copied_bytes'] == (model_bytes if phase == 'cold' else 0)
                        assert resident['reused'] == (phase != 'cold')
                        assert meta['mlp_host_binding'] == {'artifact_hash': meta['model_hash'], 'ranks': 2}
                        assert meta['effective_beam_width'] == 4096
                        if phase == 'solved':
                            assert result.path_found and replace(contract, start=state).replay(result.path)
                        else:
                            assert not result.path_found and result.path is None
                        rows.append(dict(pair=pair, mode=mode, phase=phase, wall_seconds=wall,
                            worker_seconds=meta['elapsed_seconds'], path=result.path, run_dir=meta['run_dir'],
                            model_hash=meta['model_hash'], pids=resident['worker_pids'],
                            parameters=meta['execution_parameters'], free_before=free_before,
                            checks=dict(counts)))
                        print(f'MLP_VERIFY pair={pair} mode={mode} phase={phase} wall={wall:.6f}', flush=True)
                    group = [r for r in rows if r['mode'] == mode and r['pair'] == pair]
                    assert all(r['pids'] == group[0]['pids'] for r in group)
                    assert counts['launch_checks'] == len(phases)
                    if mode == 'python_verify':
                        assert counts['python_checks'] >= len(phases) and counts['rust_checks'] == 0
                    else:
                        assert counts['rust_checks'] == len(phases) and counts['python_checks'] == 0
                finally:
                    models.verify_prepared_model = python_verify
                    backend.verify_prepared_model = backend_verify
            assert all(not Path(f'/proc/{pid}').exists() for pid in group[0]['pids'])
            (root / 'partial.json').write_text(json.dumps(rows, indent=2) + '\n')
    assert len({row['model_hash'] for row in rows}) == 1
    assert len({tuple(row['path']) for row in rows if row['phase'] == 'solved'}) == 1
    assert len({json.dumps(row['parameters'], sort_keys=True) for row in rows
                if row['phase'] == 'repeat'}) == 1
    medians = {mode: statistics.median(row['wall_seconds'] for row in rows
        if row['mode'] == mode and row['phase'] == 'repeat') for mode in modes}
    changes = []
    for pair in range(5):
        old = next(row for row in rows if row['pair'] == pair and row['mode'] == 'python_verify'
                   and row['phase'] == 'repeat')
        new = next(row for row in rows if row['pair'] == pair and row['mode'] == 'rust_verify'
                   and row['phase'] == 'repeat')
        changes.append(new['wall_seconds'] / old['wall_seconds'] - 1)
    (root / 'results.json').write_text(json.dumps(dict(
        scope='five isolated paired repeated full public calls; same Rust snapshot owner',
        model_bytes=model_bytes, width=4096, max_steps=6, num_gpus=2,
        median_seconds=medians, paired_changes=changes,
        median_paired_change=statistics.median(changes), rows=rows), indent=2) + '\n')
    print('PUBLIC_MLP_LAUNCH_VERIFY_GPU_OK', flush=True)


if __name__ == '__main__':
    main()
