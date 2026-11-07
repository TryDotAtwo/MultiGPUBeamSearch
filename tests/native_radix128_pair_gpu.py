"""Alternating complete-step comparison for an experimental hash-sort build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from multigpubeamsearch.pipeline_probe import NativePipelineProbe, parse_depths
from multigpubeamsearch.pipeline_profiles import rank_frontier_counts
from multigpubeamsearch.plan_session import NativePlanSession
from multigpubeamsearch.calibration_telemetry import verified_telemetry


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_files(directory):
    return {str(path.relative_to(directory)): sha256(path)
            for path in sorted(directory.rglob('*')) if path.is_file()}


def main():
    parser = argparse.ArgumentParser()
    for name in ('control', 'radix', 'fixture', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--world', type=int, default=2, choices=(1, 2))
    parser.add_argument('--beam', type=int, default=4096)
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    fixture = args.fixture.resolve()
    puzzle = fixture / 'puzzle.json'
    weights = fixture / 'weights'
    input_hash = sha256(puzzle)
    weight_hashes = model_files(weights)
    manifest = json.loads((weights / 'manifest.json').read_text())
    graph = json.loads(puzzle.read_text())
    start = graph['central_state']
    moves = list(graph['generators'].values())
    for move in range(20):
        start = [start[i] for i in moves[move]]
    csv = root / 'test.csv'
    csv.write_text('initial_state_id,initial_state\n0,"' + ','.join(map(str, start)) + '"\n')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, range(args.world))),
        BEAM_GENERATOR_PATH=str(puzzle), BEAM_PUZZLE_INFO_JSON=str(puzzle),
        BEAM_TEST_CSV=str(csv), BEAM_WEIGHT_DIR=str(weights), BEAM_RUNTIME_CONFIG_MODE='auto',
        BEAM_SHARD_COUNT='1', BEAM_STREAM3_RING_SLOTS='2', BEAM_STREAM4_ACTIVE_SORT_SLOTS='1',
        BEAM_STREAM4_BATCH_CANDIDATES='131072', BEAM_STREAM1_EXECUTOR='native_cuda_graph',
        BEAM_ENSEMBLE_INFERENCE_MICRO='32',
        BEAM_SOLVED_NEIGHBORHOOD_RADIUS='0', BEAM_STREAM2_SUFFIX_RADIUS='0',
        BEAM_SOLVE_BUCKET_MODE='0')
    profile = dict(BEAM_B_MICRO='768', BEAM_ENSEMBLE_INFERENCE_MICRO='32')
    runners = {'control': args.control.resolve(), 'radix': args.radix.resolve()}
    runner_hashes = {name: sha256(path) for name, path in runners.items()}
    probes, plans = {}, {}
    for name, runner in runners.items():
        def verify(texts, rank_plans):
            counts = rank_frontier_counts(rank_plans)
            return all(len(parse_depths(text, count, 6)) == 6 for text, count in zip(texts, counts))
        probes[name] = NativePipelineProbe(runner, environment, args.beam, args.world,
            root / name, [puzzle] * args.world, 112, deadline=time.monotonic() + 1200,
            verify=verify, repeats=6, grow=True)
        with NativePlanSession(runner, environment, args.world, root / name / 'plans',
                               deadline=time.monotonic() + 120) as planner:
            plans[name] = planner.admit(args.beam, profile)
    geometry = ('GLOBAL_BEAM_WIDTH_EFFECTIVE', 'BEAM_WIDTH_ALIGNMENT', 'WORLD_SIZE', 'LOCAL_RANK',
                'B_MICRO', 'RING_COUNT', 'SHARD_COUNT', 'SHARD_CAPACITY_CANDIDATES',
                'frontier_state_capacity', 'STREAM3_RING_SLOTS', 'STREAM4_ACTIVE_SORT_SLOTS',
                'STREAM4_BATCH_ALIGNMENT', 'STREAM4_BATCH_CANDIDATES')
    for old, new in zip(plans['control'], plans['radix']):
        for key in geometry:
            if old[key] != new[key]:
                raise ValueError(f'profile geometry changed: {key}')
    pairs = []
    for index in range(5):
        observed = {}
        for name in (('control', 'radix') if index % 2 == 0 else ('radix', 'control')):
            started = time.monotonic()
            rows = probes[name].measure(profile, plans[name])
            elapsed = time.monotonic() - started
            if len(rows) != 5 or any(row.throttled for row in rows):
                raise ValueError('missing measurements or throttled GPU')
            run = probes[name].directory / str(probes[name].counter)
            telemetry = json.loads((run / 'gpu-telemetry.json').read_text())
            if not verified_telemetry(telemetry, dict(world_size=args.world,
                    device_indices=list(range(args.world)), gpu_uuids=[])):
                raise ValueError('GPU telemetry missing, incomplete or throttled')
            for rank in range(args.world):
                if 'stream1_executor=native_cuda_graph' not in (run / f'rank-{rank}.log').read_text():
                    raise ValueError('unexpected inference executor')
            seconds = [max(row.seconds_by_rank) for row in rows]
            observed[name] = dict(seconds=seconds, median=statistics.median(seconds),
                                  wall_seconds=elapsed, run=str(run))
        observed['speedup_percent'] = 100 * (observed['control']['median'] / observed['radix']['median'] - 1)
        pairs.append(observed)
        print(f"PAIR {index + 1} speedup_percent={observed['speedup_percent']:.3f}", flush=True)
    metadata = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
        '--format=csv,noheader'], text=True)
    if (sha256(puzzle) != input_hash or model_files(weights) != weight_hashes or
            any(sha256(path) != runner_hashes[name] for name, path in runners.items())):
        raise ValueError('runner, model or input changed during the paired series')
    result = dict(world=args.world, beam=args.beam, precision=manifest['dtype'], profile=profile,
        runners={name: dict(path=str(path), sha256=runner_hashes[name]) for name, path in runners.items()},
        input_sha256=input_hash, model_files_sha256=weight_hashes,
        plans=plans, gpu=metadata, pairs=pairs,
        median_speedup_percent=statistics.median(p['speedup_percent'] for p in pairs),
        scope='five alternating independent process pairs; each uses one warmup and five full steps')
    (root / 'paired-results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('RADIX_PAIRED_FULL_STEP_OK', flush=True)


if __name__ == '__main__':
    main()
