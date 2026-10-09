"""Remote-only fresh preregistered 2/8-rank correctness/quality transaction.

Training-set membership is unknown. This is a new validation cohort, not a
claim of training disjointness or solving the competition's 1000 puzzles.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cayleypy_public.paths import apply_path
from tools.reference_pair_cohort import prepare_reference_pairs
from tools.cube4_bundle_contract import validate_cube4_bundle
from tools.cube4_run_supervisor import stop_group
from tools.beam_smoke_fixture import verify_submission
from tools.cube4_numeric_gate import validate_reference, validate_checkpoint_binding


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def prepare(root, cohort):
    cohort.mkdir()
    puzzle = json.loads(Path('/workspace/cube4_fixed_heldout_54295544/data/puzzle_info.json').read_text())
    central, generators = puzzle['central_state'], puzzle['generators']
    names = list(generators)
    inverses = {}
    for name, move in generators.items():
        inverse = [0] * len(move)
        for target, source in enumerate(move): inverse[source] = target
        inverses[name] = next(k for k, v in generators.items() if v == inverse)
    rng = random.Random(2026100502)
    cases, seen = [], set()
    while len(cases) < 32:
        walk = []
        for _ in range(3 + len(cases) % 4):
            options = [name for name in names if not walk or name != inverses[walk[-1]]]
            walk.append(rng.choice(options))
        state = apply_path(central, '.'.join(walk), generators)
        if tuple(state) in seen or state == central: continue
        seen.add(tuple(state))
        cases.append(dict(puzzle_id=952000 + len(cases), initial_state=list(state),
                          solution='.'.join(inverses[name] for name in reversed(walk))))
    pair_contract = dict(central_state=central, generators=generators, cases=cases)
    states, provenance = prepare_reference_pairs(pair_contract, state_len=96, num_classes=6)
    write(cohort / 'pairs.json', pair_contract)
    policy = dict(schema_version=1, scope='fresh_validation_cohort_not_training_disjointness',
        seed=2026100502, reference_count=64, pair_contract_sha256=digest(cohort/'pairs.json'),
        numeric=dict(atol=.05, rtol=.001, top_k=4, min_topk_overlap=.75),
        solve=dict(beam=4096, depth_limit=12, minimum_solved=64, per_case_timeout_seconds=180),
        strict_rank_memcheck_required=True, immutable_checkpoint_required=True,
        baseline_and_optimized_required=True, training_disjointness_proven=False,
        generalization_to_1000_puzzles_proven=False)
    write(cohort / 'policy.json', policy)  # Before either candidate is executed.
    data = cohort / 'data'; data.mkdir()
    write(data / 'puzzle_info.json', puzzle)
    with (data/'test.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream); writer.writerow(['initial_state_id', 'initial_state'])
        for row, state in enumerate(states): writer.writerow([954000+row, ','.join(map(str,state))])
    source = Path('/workspace/cube4_heldout_source_54295544')
    command = [sys.executable, str(root/'tools/export_stream1_transformer.py'),
        '--weights', str(source/'model/model.pth'), '--metadata', str(source/'model/model.json'),
        '--generators', str(source/'generators/p002.json'), '--source-root', str(source),
        '--out', str(cohort/'weights_fp16'), '--dtype', 'fp16', '--num-classes', '6',
        '--reference-out', str(cohort/'reference.json'), '--reference-count', '64',
        '--reference-pairs', str(cohort/'pairs.json')]
    with (cohort/'prepare.log').open('xb') as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    reference = json.loads((cohort/'reference.json').read_text())
    if reference['states'] != states: raise ValueError('exported cohort order mismatch')
    seal(cohort)


def seal(cohort):
    reference=json.loads((cohort/'reference.json').read_text());validate_reference(reference)
    pairs=json.loads((cohort/'pairs.json').read_text())
    states,_=prepare_reference_pairs(pairs,state_len=96,num_classes=6)
    manifest=json.loads((cohort/'weights_fp16/manifest.json').read_text())
    validate_checkpoint_binding(reference,manifest)
    if manifest['source_weights_sha256']!=digest('/workspace/cube4_heldout_source_54295544/model/model.pth'):
        raise ValueError('original source checkpoint changed')
    if states!=reference['states'] or len(states)!=64:raise ValueError('fresh cohort differs from pair replay')
    with (cohort/'data/test.csv').open(newline='') as stream:
        reader=csv.DictReader(stream);rows=list(reader)
    if len(rows)!=64 or any(row['initial_state_id']!=str(954000+i) or
            [int(x) for x in row['initial_state'].split(',')]!=states[i] for i,row in enumerate(rows)):
        raise ValueError('fresh CSV order/state mismatch')
    write(cohort/'immutable.json', {name:digest(cohort/name) for name in ('pairs.json','policy.json','reference.json','data/test.csv','data/puzzle_info.json','weights_fp16/manifest.json')})


def run(root, cohort, build, world, deadline, output):
    output.mkdir()
    policy = json.loads((cohort/'policy.json').read_text())
    reference = json.loads((cohort/'reference.json').read_text())
    immutable = json.loads((cohort/'immutable.json').read_text())
    runner = build/'production_runner'; runner_sha = digest(runner)
    base_profile = json.loads((root/'evidence/dual_base_profile.json').read_text())
    base_profile.update(BEAM_B_MICRO='2048', BEAM_STREAM1_TRANSFORMER_MICRO='1024',
        BEAM_SHARD_CAPACITY_CANDIDATES='131072', BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM=str(world*1000000))
    clean_env = {k:v for k,v in os.environ.items() if not k.startswith(('BEAM_','CUDA_'))}
    modes = []
    for mode in ('baseline','optimized'):
        optimized = mode == 'optimized'
        profile = dict(base_profile)
        profile.update(BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM=str(int(optimized)),
            BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=str(int(optimized)),
            BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=str(int(optimized)),
            BEAM_STREAM4_BOUNDED_SORT=str(int(optimized)),
            BEAM_STREAM5_PREFILL_BEFORE_WAIT=str(int(optimized)))
        env = dict(clean_env, **profile)
        gate = output/f'{mode}-numeric'
        if time.time()+900 >= deadline: raise RuntimeError('archive reserve approaching')
        command = [sys.executable,str(root/'tools/cube4_production_preflight.py'),
            '--runner',str(runner),'--probe',str(build/'cuda_visible_device_probe'),
            '--reference-dir',str(cohort),'--source-root',str(root),'--world-size',str(world),
            '--output-root',str(gate),'--timeout','600',
            '--cupti-library','/workspace/cupti_replay_54295544/libcupti_trace_injection.so']
        with (output/f'{mode}-numeric.log').open('xb') as log:
            code = subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200).returncode
        summary = json.loads((gate/'summary.json').read_text())
        if code != 2 or summary['status'] != 'numeric_pass_unadmitted' or len(summary['devices']) != world:
            raise RuntimeError('fresh numeric transaction failed')
        reports = []
        with (output/f'{mode}-search.jsonl').open('x') as ledger:
            for row, state in enumerate(reference['states']):
                if time.time()+300 >= deadline: raise RuntimeError('archive reserve reached')
                pid = 954000+row; directory = output/f'{mode}-p{pid}'; directory.mkdir()
                before = validate_cube4_bundle(cohort/'data',cohort/'weights_fp16',str(pid))
                if before['initial_state'] != state: raise ValueError('search state order mismatch')
                run_env = dict(env,BEAM_WEIGHT_DIR=str(cohort/'weights_fp16'),
                    BEAM_GENERATOR_PATH=str(cohort/'data/puzzle_info.json'),
                    BEAM_PUZZLE_INFO_JSON=str(cohort/'data/puzzle_info.json'),BEAM_TEST_CSV=str(cohort/'data/test.csv'),
                    BEAM_HISTORY_DIR=str(directory/'history'),BEAM_HISTORY_DISK_PATH=str(directory/'history-disk'),
                    BEAM_PREDICT_STATS_PATH=str(directory/'predict_stats'),BEAM_NCCL_ID_FILE=str(directory/'nccl-id'),
                    BEAM_RANK_STATUS_DIR=str(directory/'rank-status'),BEAM_NCCL_RUN_ID=str(directory))
                write(directory/'request.json',dict(profile=profile,world=world,puzzle_id=pid,
                    runner_sha256=runner_sha,policy_sha256=digest(cohort/'policy.json'),
                    reference_sha256=digest(cohort/'reference.json'),initial_state=state))
                processes, logs = [], []; started=time.monotonic()
                try:
                    for rank in range(world):
                        log=(directory/f'rank{rank}.log').open('xb');logs.append(log)
                        command=['compute-sanitizer','--tool','memcheck','--leak-check','full','--error-exitcode','99',
                                 str(runner),str(pid),'12','4096']
                        processes.append(subprocess.Popen(command,cwd=directory,
                            env=dict(run_env,RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE=str(world)),
                            stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
                    for process in processes:
                        remaining=180-(time.monotonic()-started)
                        if remaining<=0 or process.wait(timeout=remaining)!=0:raise RuntimeError('rank failed or timed out')
                finally:
                    for process in processes:stop_group(process)
                    for log in logs:log.close()
                for rank in range(world):
                    raw=(directory/f'rank{rank}.log').read_text()
                    status=json.loads((directory/'rank-status'/f'rank-{rank}.json').read_text())
                    if raw.count('ERROR SUMMARY: 0 errors')!=1 or '0 bytes leaked in 0 allocations' not in raw:
                        raise RuntimeError('missing clean strict sanitizer result')
                    if status['rank']!=rank or status['puzzle_id']!=pid or status['exit_code']!=0:
                        raise RuntimeError('terminal rank identity/failure')
                    if status['status']!='solved':raise RuntimeError('preregistered solve threshold failed')
                submissions=list((directory/'test_results').glob(f'submit_p{pid}_*.csv'))
                if len(submissions)!=1:raise RuntimeError('ambiguous/missing solution')
                puzzle=json.loads((cohort/'data/puzzle_info.json').read_text())
                replay=verify_submission(submissions[0],dict(puzzle_id=pid,initial=state,
                    central=puzzle['central_state'],minimum_depth=0),
                    dict(move_names=list(puzzle['generators']),moves=list(puzzle['generators'].values())),12)
                if digest(runner)!=runner_sha or any(digest(cohort/name)!=sha for name,sha in immutable.items()):
                    raise RuntimeError('immutable input/build changed during search')
                result=dict(puzzle_id=pid,row=row,replay=replay,elapsed_seconds=time.monotonic()-started,
                    directory=str(directory),runner_sha256=runner_sha,strict_rank_memchecks=world,status='solved')
                write(directory/'receipt.json',result);reports.append(result)
                ledger.write(json.dumps(result)+'\n');ledger.flush()
                print(json.dumps(dict(mode=mode,row=row,status='solved')),flush=True)
        modes.append(dict(mode=mode,profile=profile,numeric_dir=str(gate),cases=reports))
    write(output/'transaction.json',dict(schema_version=1,scope='fresh_bounded_profile_validation',
        runner_sha256=runner_sha,world=world,cohort=str(cohort),modes=modes,
        policy_sha256=digest(cohort/'policy.json'),reference_sha256=digest(cohort/'reference.json'),
        production_admitted=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','seal','run'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--build',type=Path)
    parser.add_argument('--world',type=int,choices=(2,8))
    parser.add_argument('--deadline',type=int)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if os.name!='posix':raise SystemExit('Remote POSIX only; do not execute locally')
    if args.action=='prepare':prepare(args.root,args.cohort)
    elif args.action=='seal':seal(args.cohort)
    else:run(args.root,args.cohort,args.build,args.world,args.deadline,args.output)

if __name__=='__main__':main()
