"""Compare the opt-in Rust NativeSession to the public Python session on GPUs."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import struct
from types import SimpleNamespace

from multigpubeamsearch import NativeModel, NativeOptions, NativeSession
from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.hamming import write_hamming_artifact


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'cutlass', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cache', type=Path, help='reuse a previously verified managed build cache')
    parser.add_argument('--backend', choices=('cutlass', 'libtorch'), required=True)
    parser.add_argument('--experimental-parallel-nccl-identity', action='store_true')
    parser.add_argument('--experimental-index-sort', action='store_true')
    parser.add_argument('--exercise-index-change', action='store_true')
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    center = tuple(i // 16 for i in range(96))
    moves = []
    for i in range(24):
        move = list(range(96))
        move[i], move[i + 24] = move[i + 24], move[i]
        moves.append(tuple(move))
    definition = SimpleNamespace(state_size=96, central_state=center,
        generators_permutations=moves, generator_names=[f'm{i}' for i in range(24)],
        is_permutation_group=lambda: True)
    graph = SimpleNamespace(definition=definition, device='cpu')
    contract = GraphContract.from_graph(graph, center)
    weights = write_hamming_artifact(contract, root)
    alternate = root / 'alternate'
    import shutil
    shutil.copytree(weights, alternate)
    (alternate / 'output_bias.fp16').write_bytes(struct.pack('<e', 1.0))
    models = [NativeModel.for_graph(graph, path) for path in (weights, alternate)]
    starts = []
    for count in (3, 4, 5):
        state = center
        for move in moves[:count]:
            state = tuple(state[index] for index in move)
        starts.append(state)
    records = []
    cases = [
        ('first', starts[0], models[0], 64),
        ('repeat', starts[1], models[0], 64),
        ('unsolved', starts[2], models[0], 64),
        ('model-change', starts[0], models[1], 64),
    ]
    if args.exercise_index_change:
        cases.append(('index-change', starts[0], models[1], 64))
    cases.extend((('beam-change', starts[0], models[1], 65),
                  ('collect', starts[0], models[1], 65)))
    for world in (1, 2):
        options = NativeOptions(source_dir=args.source, cutlass_dir=args.cutlass,
            cache_dir=args.cache or root / 'cache', num_gpus=world, inference_backend=args.backend,
            autotune=False, report_calibration=False, build_jobs=2,
            build_timeout_seconds=1200, timeout_seconds=120,
            experimental_index_sort=args.experimental_index_sort)
        for engine in ('python', 'rust'):
            old_pids = []
            with NativeSession(engine=engine, experimental_parallel_nccl_identity=
                               args.experimental_parallel_nccl_identity) as session:
                for name, start, model, width in cases:
                    selected = (replace(options, solution_mode='collect', collect_extra_depths=1,
                                        max_collected_solutions=7) if name == 'collect' else options)
                    if name == 'index-change':
                        selected = replace(selected, experimental_index_sort=not args.experimental_index_sort)
                    result = session.beam_search(graph, native_options=selected,
                        start_state=start, predictor=model, beam_width=width,
                        max_steps=4, return_path=True)
                    meta = result.native_metadata
                    persistent = meta['persistent_session']
                    pids = persistent['worker_pids']
                    reused = name in ('repeat', 'unsolved')
                    assert persistent['reused'] == reused
                    assert meta['effective_beam_width'] == width
                    assert meta['build'].get('experimental_index_sort', False) is selected.experimental_index_sort
                    if old_pids:
                        assert (pids == old_pids) == reused
                        if not reused:
                            assert all(not Path(f'/proc/{pid}').exists() for pid in old_pids)
                    old_pids = pids
                    if name == 'unsolved':
                        assert not result.path_found and result.path is None
                    else:
                        assert result.path_found and replace(contract, start=start).replay(result.path)
                    if name == 'collect':
                        assert 1 <= len(result.solutions) <= 7
                        assert all(replace(contract, start=start).replay(path) for path in result.solutions)
                    records.append(dict(world=world, engine=engine, case=name,
                        path=result.path, width=meta['effective_beam_width'],
                        experimental_index_sort=selected.experimental_index_sort,
                        build_key=meta['build']['build_key'], persistent=persistent,
                        run_dir=meta['run_dir'], solutions=result.solutions))
                    print(f'PUBLIC_RUST_CASE {args.backend} world={world} engine={engine} {name}', flush=True)
            assert all(not Path(f'/proc/{pid}').exists() for pid in old_pids)
        for case in (entry[0] for entry in cases):
            pair = [row for row in records if row['world'] == world and row['case'] == case]
            assert len(pair) == 2 and pair[0]['path'] == pair[1]['path'], pair
    (root / 'results.json').write_text(json.dumps(dict(backend=args.backend,
        experimental_parallel_nccl_identity=args.experimental_parallel_nccl_identity,
        experimental_index_sort=args.experimental_index_sort,
        scope='public bounded deterministic session correctness; no performance claim', records=records), indent=2))
    print('PUBLIC_RUST_SESSION_GPU_OK', flush=True)


if __name__ == '__main__':
    main()
