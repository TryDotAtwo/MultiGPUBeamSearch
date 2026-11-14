"""Validate session model snapshots and compare complete paired public calls."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import statistics
import struct
import time
from types import SimpleNamespace

import numpy as np
import torch

from multigpubeamsearch import NativeModel, NativeOptions, NativeSession
from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.hamming import write_hamming_artifact


def graph_fixture():
    center = tuple(i // 16 for i in range(96))
    moves = []
    for i in range(24):
        move = list(range(96))
        move[i], move[i + 24] = move[i + 24], move[i]
        moves.append(tuple(move))
    graph = SimpleNamespace(device='cpu', definition=SimpleNamespace(
        state_size=96, central_state=center, generators_permutations=moves,
        generator_names=[f'm{i}' for i in range(24)], is_permutation_group=lambda: True))
    return graph, GraphContract.from_graph(graph, center), moves


def start_state(center, moves, count):
    state = center
    for move in moves[:count]:
        state = tuple(state[i] for i in move)
    return state


def large_hamming_artifact(contract, root):
    """A larger finite FP16 MLP with exactly the synthetic Hamming score."""
    target = root / 'large-weights'
    target.mkdir()
    width, blocks, size = 512, 8, contract.state_len
    classes = max(size, contract.num_classes)

    def blob(name, shape, update=None):
        values = np.zeros(shape, dtype='<f2')
        if update is not None:
            update(values)
        values.tofile(target / f'{name}.fp16')

    def input_weights(values):
        for position, label in enumerate(contract.center):
            values[position, label, 0] = -1

    blob('input_weight_hxk', (size, classes, width), input_weights)
    blob('input_bias', (width,), lambda value: value.__setitem__(0, size))
    blob('hidden_weight_hxk', (width, width), lambda value: value.__setitem__((0, 0), 1))
    blob('hidden_bias', (width,))
    for block in range(blocks):
        for fc in (1, 2):
            blob(f'residual{block}_fc{fc}_weight_hxk', (width, width))
            blob(f'residual{block}_fc{fc}_bias', (width,))
    blob('output_weight_hxk', (width,), lambda value: value.__setitem__(0, 1))
    blob('output_bias', (1,))
    manifest = dict(state_len=size, num_classes=classes, hidden1=width, hidden2=width,
                    residual_count=blocks, output_dim=1, dtype='fp16', normalization='none',
                    graph_hash=contract.graph_hash, heuristic='hamming', trained=False)
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return target


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'cutlass', 'cache', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    graph, contract, moves = graph_fixture()
    weights = write_hamming_artifact(contract, root)
    alternate = root / 'alternate'
    shutil.copytree(weights, alternate)
    (alternate / 'output_bias.fp16').write_bytes(struct.pack('<e', 1.0))
    models = [NativeModel.for_graph(graph, path) for path in (weights, alternate)]
    starts = [start_state(contract.center, moves, count) for count in (3, 4, 5)]
    records = []

    for world in (1, 2):
        options = NativeOptions(source_dir=args.source, cutlass_dir=args.cutlass, cache_dir=args.cache,
            num_gpus=world, inference_backend='cutlass', autotune=False, report_calibration=False,
            build_jobs=2, build_timeout_seconds=1800, timeout_seconds=180)
        for engine in ('python', 'rust'):
            old_pids = []
            cases = [('first', starts[0], models[0], 64), ('repeat', starts[1], models[0], 64),
                     ('unsolved', starts[2], models[0], 64), ('model-change', starts[0], models[1], 64),
                     ('beam-change', starts[0], models[1], 65)]
            with NativeSession(engine=engine, experimental_reuse_model_snapshot=True,
                               experimental_parallel_nccl_identity=True) as session:
                for name, state, model, width in cases:
                    result = session.beam_search(graph, native_options=options, start_state=state,
                        predictor=model, beam_width=width, max_steps=4, return_path=True)
                    meta = result.native_metadata
                    persistent, snapshot = meta['persistent_session'], meta['model_snapshot']
                    reused = name in ('repeat', 'unsolved')
                    assert persistent['reused'] == reused
                    assert snapshot['reused'] == reused
                    assert (snapshot['copied_bytes'] == 0) == reused
                    assert meta['effective_beam_width'] == width
                    pids = persistent['worker_pids']
                    assert len(pids) == world
                    if old_pids:
                        assert (pids == old_pids) == reused
                        if not reused:
                            assert all(not Path(f'/proc/{pid}').exists() for pid in old_pids)
                    old_pids = pids
                    if name == 'unsolved':
                        assert not result.path_found and result.path is None
                    else:
                        assert result.path_found and replace(contract, start=state).replay(result.path)
                    if not reused:
                        assert Path(snapshot['snapshot_dir']) == Path(meta['run_dir']) / 'weights'
                    records.append(dict(world=world, engine=engine, case=name, path=result.path,
                        model_hash=meta['model_hash'], snapshot=snapshot, persistent=persistent,
                        run_dir=meta['run_dir'], build_key=meta['build']['build_key']))
                    print(f'SNAPSHOT_CASE world={world} engine={engine} {name}', flush=True)
            assert all(not Path(f'/proc/{pid}').exists() for pid in old_pids)
        for name, *_ in cases:
            pair = [row for row in records if row['world'] == world and row['case'] == name]
            assert len(pair) == 2 and pair[0]['path'] == pair[1]['path']
            assert pair[0]['model_hash'] == pair[1]['model_hash']

    large = large_hamming_artifact(contract, root)
    model = NativeModel.for_graph(graph, large)
    model_bytes = sum(path.stat().st_size for path in large.iterdir() if path.is_file())
    options = NativeOptions(source_dir=args.source, cutlass_dir=args.cutlass, cache_dir=args.cache,
        num_gpus=2, inference_backend='cutlass', autotune=False, report_calibration=False,
        build_jobs=2, build_timeout_seconds=1800, timeout_seconds=180)
    sessions = {mode: NativeSession(engine='rust', experimental_parallel_nccl_identity=True,
                    experimental_reuse_model_snapshot=(mode == 'reuse')) for mode in ('baseline', 'reuse')}
    rows = []

    def run(mode, phase, pair, state):
        before = time.monotonic()
        result = sessions[mode].beam_search(graph, native_options=options, start_state=state,
            predictor=model, beam_width=4096, max_steps=6, return_path=True)
        wall = time.monotonic() - before
        meta = result.native_metadata
        assert meta['effective_beam_width'] == 4096
        assert meta['persistent_session']['reused'] == (phase != 'cold')
        if mode == 'reuse':
            snapshot = meta['model_snapshot']
            assert snapshot['reused'] == (phase != 'cold')
            assert snapshot['copied_bytes'] == (model_bytes if phase == 'cold' else 0)
        else:
            assert 'model_snapshot' not in meta
            assert (Path(meta['run_dir']) / 'weights').is_dir()
        if phase == 'solved':
            assert result.path_found and replace(contract, start=state).replay(result.path)
        else:
            assert not result.path_found and result.path is None
        rows.append(dict(mode=mode, phase=phase, pair=pair, wall_seconds=wall,
            worker_seconds=meta['elapsed_seconds'], path=result.path, run_dir=meta['run_dir'],
            model_hash=meta['model_hash'], pids=meta['persistent_session']['worker_pids']))
        print(f'SNAPSHOT_PAIR phase={phase} pair={pair} mode={mode} wall={wall:.6f}', flush=True)

    try:
        for mode in sessions:
            run(mode, 'cold', None, start_state(contract.center, moves, 12))
        for mode in sessions:
            run(mode, 'warmup', None, start_state(contract.center, moves, 12))
        for pair in range(5):
            order = ('baseline', 'reuse') if pair % 2 == 0 else ('reuse', 'baseline')
            for mode in order:
                run(mode, 'repeat', pair, start_state(contract.center, moves, 12))
        for mode in sessions:
            run(mode, 'solved', None, start_state(contract.center, moves, 3))
    finally:
        for session in sessions.values():
            session.close()
    for mode in sessions:
        group = [row for row in rows if row['mode'] == mode]
        assert all(row['pids'] == group[0]['pids'] for row in group)
        assert all(not Path(f'/proc/{pid}').exists() for pid in group[0]['pids'])
    assert len({row['model_hash'] for row in rows}) == 1
    assert rows[-1]['path'] == rows[-2]['path']
    medians = {mode: {metric: statistics.median(row[metric] for row in rows
                if row['mode'] == mode and row['phase'] == 'repeat')
                for metric in ('wall_seconds', 'worker_seconds')} for mode in sessions}
    changes = []
    for pair in range(5):
        baseline = next(row for row in rows if row['phase'] == 'repeat' and row['pair'] == pair and row['mode'] == 'baseline')
        reuse = next(row for row in rows if row['phase'] == 'repeat' and row['pair'] == pair and row['mode'] == 'reuse')
        changes.append(reuse['wall_seconds'] / baseline['wall_seconds'] - 1)
    (root / 'results.json').write_text(json.dumps(dict(
        scope='synthetic public session correctness and five paired full-call timings; not saturated-step measurements',
        correctness=records, model_bytes=model_bytes, model='synthetic Hamming FP16 width 512, 8 residual blocks',
        width=4096, max_steps=6, num_gpus=2, median_seconds=medians,
        paired_wall_changes=changes, median_paired_wall_change=statistics.median(changes), rows=rows), indent=2) + '\n')
    print('PUBLIC_MODEL_SNAPSHOT_GPU_OK', flush=True)


if __name__ == '__main__':
    main()
