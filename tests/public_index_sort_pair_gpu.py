"""Compare complete resident calls and steps with public Stream 4 index sorting."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import statistics
import subprocess
import time
from types import SimpleNamespace

from multigpubeamsearch import NativeModel, NativeOptions, NativeSession
from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.hamming import write_hamming_artifact


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'cutlass', 'cache', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    center = tuple(i // 16 for i in range(96))
    moves = []
    for i in range(24):
        move = list(range(96))
        move[i], move[i + 24] = move[i + 24], move[i]
        moves.append(tuple(move))
    graph = SimpleNamespace(device='cpu', definition=SimpleNamespace(
        state_size=96, central_state=center, generators_permutations=moves,
        generator_names=[f'm{i}' for i in range(24)], is_permutation_group=lambda: True))
    contract = GraphContract.from_graph(graph, center)
    model = NativeModel.for_graph(graph, write_hamming_artifact(contract, root))
    start, solvable = center, center
    for move in moves[:12]:
        start = tuple(start[i] for i in move)
    for move in moves[:3]:
        solvable = tuple(solvable[i] for i in move)
    base = NativeOptions(source_dir=args.source, cutlass_dir=args.cutlass, cache_dir=args.cache,
        num_gpus=2, inference_backend='cutlass', autotune=False, report_calibration=False,
        build_jobs=2, build_timeout_seconds=1800, timeout_seconds=180)
    options = {'baseline': base, 'index': replace(base, experimental_index_sort=True)}
    sessions = {mode: NativeSession(engine='rust', experimental_parallel_nccl_identity=True)
                for mode in options}
    rows = []

    def run(mode, phase, pair, state):
        before = time.monotonic()
        result = sessions[mode].beam_search(graph, native_options=options[mode], start_state=state,
            predictor=model, beam_width=4096, max_steps=6, return_path=True)
        wall = time.monotonic() - before
        meta = result.native_metadata
        assert meta['effective_beam_width'] == 4096
        assert meta['persistent_session']['reused'] == (phase != 'cold')
        assert meta['build'].get('experimental_index_sort', False) == (mode == 'index')
        if phase == 'solved':
            assert result.path_found and replace(contract, start=state).replay(result.path)
            rank_steps = []
        else:
            assert not result.path_found and result.path is None
            log = (Path(meta['run_dir']) / 'native.log').read_text()
            rank_steps = [float(x) for x in re.findall(r'^avg_depth_sec=([0-9.eE+-]+)$', log, re.M)]
            depths = [int(x) for x in re.findall(r'^completed_depths=(\d+)$', log, re.M)]
            assert len(rank_steps) == len(depths) == 2 and depths == [6, 6]
            assert all(value > 0 for value in rank_steps)
        row = dict(mode=mode, phase=phase, pair=pair, wall_seconds=wall,
            worker_seconds=meta['elapsed_seconds'], rank_reported_avg_depth_seconds=rank_steps,
            slow_rank_reported_avg_depth_seconds=max(rank_steps) if rank_steps else None,
            run_dir=meta['run_dir'], path=result.path, build_key=meta['build']['build_key'],
            pids=meta['persistent_session']['worker_pids'])
        rows.append(row)
        print(f'PUBLIC_INDEX {phase} pair={pair} mode={mode} wall={wall:.6f}', flush=True)

    try:
        for mode in options:
            run(mode, 'cold', None, start)
        for mode in options:
            run(mode, 'warmup', None, start)
        for pair in range(5):
            order = ('baseline', 'index') if pair % 2 == 0 else ('index', 'baseline')
            for mode in order:
                run(mode, 'repeat', pair, start)
        for mode in options:
            run(mode, 'solved', None, solvable)
        memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
            '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    finally:
        for session in sessions.values():
            session.close()
    for mode in options:
        group = [row for row in rows if row['mode'] == mode]
        assert all(row['pids'] == group[0]['pids'] for row in group)
        assert len(group[0]['pids']) == 2
        assert all(not Path(f'/proc/{pid}').exists() for pid in group[0]['pids'])
        assert all(row['build_key'] == group[0]['build_key'] for row in group)
    assert rows[0]['build_key'] != rows[1]['build_key']
    assert [row['path'] for row in rows if row['phase'] == 'solved'][0] == \
           [row['path'] for row in rows if row['phase'] == 'solved'][1]
    medians = {mode: {metric: statistics.median(row[metric] for row in rows
                if row['mode'] == mode and row['phase'] == 'repeat')
                for metric in ('wall_seconds', 'worker_seconds', 'slow_rank_reported_avg_depth_seconds')}
               for mode in options}
    (root / 'results.json').write_text(json.dumps(dict(
        scope=('five alternating public calls; reported average depth includes startup and frontier growth, '
               'and is not an isolated saturated-step measurement'),
        width=4096, max_steps=6, num_gpus=2, model='synthetic Hamming FP16',
        median_seconds=medians, gpu_memory_mib_with_both_sessions=memory, rows=rows), indent=2))
    print('PUBLIC_INDEX_SORT_PAIR_GPU_OK', flush=True)


if __name__ == '__main__':
    main()
