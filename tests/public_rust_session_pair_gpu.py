"""Paired end-to-end resident search timings; not an optimization promotion."""
import argparse
import json
from pathlib import Path
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
    start = center
    for move in moves[:12]:
        start = tuple(start[i] for i in move)
    options = NativeOptions(source_dir=args.source, cutlass_dir=args.cutlass,
        cache_dir=args.cache, num_gpus=2, inference_backend='cutlass', autotune=False,
        report_calibration=False, build_jobs=2, build_timeout_seconds=1200, timeout_seconds=180)
    sessions = {engine: NativeSession(engine=engine) for engine in ('python', 'rust')}
    rows = []

    def run(engine, phase, pair):
        before = time.monotonic()
        result = sessions[engine].beam_search(graph, native_options=options, start_state=start,
            predictor=model, beam_width=4096, max_steps=6, return_path=True)
        wall = time.monotonic() - before
        metadata = result.native_metadata
        assert not result.path_found and result.path is None
        assert metadata['effective_beam_width'] == 4096
        assert metadata['persistent_session']['reused'] == (phase != 'cold')
        rows.append(dict(engine=engine, phase=phase, pair=pair, wall_seconds=wall,
            worker_seconds=metadata['elapsed_seconds'], run_dir=metadata['run_dir'],
            pids=metadata['persistent_session']['worker_pids']))
        print(f'PUBLIC_PAIR {phase} pair={pair} engine={engine} wall={wall:.6f}', flush=True)

    try:
        for engine in ('python', 'rust'):
            run(engine, 'cold', None)
        for pair in range(5):
            order = ('python', 'rust') if pair % 2 == 0 else ('rust', 'python')
            for engine in order:
                run(engine, 'repeated', pair)
        memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
            '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    finally:
        for session in sessions.values():
            session.close()
    for engine in sessions:
        pids = [row['pids'] for row in rows if row['engine'] == engine]
        assert all(ids == pids[0] for ids in pids)
        assert all(not Path(f'/proc/{pid}').exists() for pid in pids[0])
    median = {engine: statistics.median(row['wall_seconds'] for row in rows
                                       if row['engine'] == engine and row['phase'] == 'repeated')
              for engine in sessions}
    (root / 'results.json').write_text(json.dumps(dict(
        scope='five alternating pairs; full six-depth public calls with two resident groups on the same GPUs',
        width=4096, max_steps=6, num_gpus=2, model='synthetic Hamming FP16',
        median_wall_seconds=median, gpu_memory_mib_with_both_sessions=memory, rows=rows), indent=2))
    print('PUBLIC_RUST_PAIR_GPU_OK', flush=True)


if __name__ == '__main__':
    main()
