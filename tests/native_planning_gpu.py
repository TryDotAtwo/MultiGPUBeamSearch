"""Compare retained Rust planning and component protocol to the Python reference.

Run explicitly on a CUDA host with two devices. Timings are diagnostics only.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.hamming import write_hamming_artifact
from multigpubeamsearch.plan_session import NativePlanSession
from multigpubeamsearch.pipeline_profiles import validate_rank_plans, rank_frontier_counts


def main():
    parser = argparse.ArgumentParser()
    for key in ('runner', 'extension', 'out'):
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('_beam_runtime', args.extension)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    center = tuple(i // 16 for i in range(96))
    moves = []
    for i in range(24):
        row = list(range(96))
        row[i], row[i+24] = row[i+24], row[i]
        moves.append(tuple(row))
    graph_hash = hashlib.sha256(json.dumps(moves).encode()).hexdigest()
    contract = GraphContract(96, 6, tuple(moves), tuple('m'+str(i) for i in range(24)), center, center, graph_hash)
    weights = write_hamming_artifact(contract, root)
    puzzle = root / 'puzzle.json'
    puzzle.write_text(json.dumps(contract.to_puzzle_info()))
    csv = root / 'test.csv'
    csv.write_text('initial_state_id,initial_state\n0,"'+','.join(map(str, center))+'"\n')
    base = dict(os.environ, BEAM_GENERATOR_PATH=str(puzzle), BEAM_PUZZLE_INFO_JSON=str(puzzle),
        BEAM_TEST_CSV=str(csv), BEAM_WEIGHT_DIR=str(weights), BEAM_RUNTIME_CONFIG_MODE='auto',
        BEAM_B_MICRO='768', BEAM_SHARD_COUNT='1', BEAM_STREAM4_BATCH_CANDIDATES='1024',
        BEAM_ENSEMBLE_INFERENCE_MICRO='32', BEAM_SOLVED_NEIGHBORHOOD_RADIUS='0', BEAM_STREAM2_SUFFIX_RADIUS='0')
    results = []
    for world in (1, 2):
        env = dict(base, CUDA_VISIBLE_DEVICES=','.join(map(str, range(world))))
        reference = {}
        with NativePlanSession(args.runner.resolve(), env, world, root/f'python-{world}', deadline=time.monotonic()+120) as owner:
            for beam in (5, 64, 65):
                reference[beam] = owner.admit(beam, {})
        owner = native.PlanningSession([str(args.runner.resolve()), '0', '1', '1024'], env,
            world, Path.cwd(), root/f'rust-{world}', 120.)
        pids = owner.pids
        try:
            assert owner.supports_components
            for beam in (5, 64, 65):
                plans = json.loads(owner.admit(beam, {}))
                validate_rank_plans(plans)
                assert sum(rank_frontier_counts(plans)) == beam
                # Budget snapshots can differ by CUDA context bookkeeping; exact
                # layouts, capacities and logical counts must be identical.
                scrub = lambda rows: [{k:v for k,v in r.items() if k != 'gpu_budget_bytes'} for r in rows]
                assert scrub(plans) == scrub(reference[beam]), (plans, reference[beam])
                assert owner.pids == pids
                results.append(dict(world=world, beam=beam, plans=plans, pids=pids))
            for _ in range(2):
                components = json.loads(owner.measure_components(64, {}))
                assert len(components) == world
                assert all(r['rank'] == rank and r['correctness_passed'] is True for rank,r in enumerate(components))
                results.append(dict(world=world, components=components))
            owner.set_deadline(60.)
            assert json.loads(owner.admit(5, {'BEAM_ENSEMBLE_INFERENCE_MICRO':'32'}))
            frozen=dict(BEAM_B_MICRO='768',BEAM_STREAM3_RING_SLOTS='2',BEAM_SHARD_COUNT='1',
                BEAM_STREAM4_BATCH_CANDIDATES='1024',BEAM_STREAM4_ACTIVE_SORT_SLOTS='1')
            capacity=json.loads(owner.find_capacity(json.dumps([frozen]),68,4))
            assert capacity['requested_beam']==68 and capacity['search_complete'] and not capacity['full_step_verified']
            results.append(dict(world=world,capacity=capacity))
            assert owner.pids == pids
        finally:
            owner.close()
        assert all(code is not None for code in owner.exit_codes())
    (root/'results.json').write_text(json.dumps(dict(scope='planner/protocol correctness; no performance promotion', results=results), indent=2))
    print('NATIVE_PLANNING_GPU_OK', flush=True)


if __name__ == '__main__':
    main()
