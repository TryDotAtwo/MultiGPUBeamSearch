"""Bounded remote before/after beam timing; diagnostics, never admission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from tools.cube4_bundle_contract import validate_cube4_bundle
from tools.cube4_production_preflight import run_scorer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--nccl-dir', required=True)
p.add_argument('--tag', required=True)
p.add_argument('--beam', type=int, choices=(16384,65536,262144), default=65536)
p.add_argument('--depth', type=int, choices=range(8,11), default=8)
p.add_argument('--pairs', type=int, choices=range(1,4), default=3)
p.add_argument('--deadline', type=int, required=True)
a = p.parse_args()
if not a.tag.replace('-', '').replace('_', '').isalnum():
    raise ValueError('invalid fresh experiment tag')
root = a.root.resolve(strict=True)
out = root / 'evidence' / ('pipeline_ab_' + a.tag)
out.mkdir(exist_ok=False)
profile = json.loads((root / 'evidence/dual_base_profile.json').read_text())
data = root / 'evidence/recovered_campaign_inputs_20261001'
weights = root / 'evidence/cube4_paired_reference_20261001/weights_fp16'
inputs = validate_cube4_bundle(data, weights, '1000')
hashes = {mode: hashlib.sha256((root / folder / 'production_runner').read_bytes()).hexdigest()
          for mode, folder in (('before','build-before'), ('after','build'))}
results = []
for index in range(a.pairs):
    # Alternate order, including bookends; no concurrent competing GPU workload.
    for mode in (('before','after') if index % 2 == 0 else ('after','before')):
        if time.time() + 240 >= a.deadline - 600:
            raise RuntimeError('stop deadline headroom; preserve partial measurements')
        case = out / f'{index}_{mode}'
        case.mkdir()
        binary = root / ('build-before' if mode == 'before' else 'build') / 'production_runner'
        selected = {k:v for k,v in profile.items() if k not in ('RANK','LOCAL_RANK','WORLD_SIZE')}
        selected.update(BEAM_WEIGHT_DIR=str(weights), BEAM_GENERATOR_PATH=str(data/'puzzle_info.json'),
            BEAM_PUZZLE_INFO_JSON=str(data/'puzzle_info.json'), BEAM_TEST_CSV=str(data/'test.csv'),
            BEAM_HISTORY_DIR=str(case/'history'), BEAM_HISTORY_DISK_PATH=str(case/'history-disk'),
            BEAM_PREDICT_STATS_PATH=str(case/'predict_stats'), BEAM_PREDICT_STATS_VERBOSE='1',
            BEAM_RANK_STATUS_DIR=str(case/'rank-status'), BEAM_NCCL_ID_FILE=str(case/'nccl-id'))
        env = {k:v for k,v in os.environ.items() if not k.startswith('BEAM_')}
        env.update(selected, LD_LIBRARY_PATH=a.nccl_dir)
        command = ['/venv/main/bin/torchrun','--standalone','--nproc_per_node=2',
                   '--no-python','--log-dir',str(case/'ranklogs'),'--redirects','3',
                   str(binary),'1000',str(a.depth),str(a.beam)]
        (case/'request.json').write_text(json.dumps(dict(command=command,profile=selected,
            runner_sha256=hashes[mode],production_admitted=False,sanitizer=False),indent=2))
        print(f'START pair={index} mode={mode}',flush=True)
        code = run_scorer(command,case/'launcher.log',180,cwd=case,env=env)
        if code or hashlib.sha256(binary.read_bytes()).hexdigest() != hashes[mode]:
            raise RuntimeError('runner failed or changed')
        if validate_cube4_bundle(data,weights,'1000') != inputs:
            raise RuntimeError('model/input identity changed')
        rank_rows=[]
        for rank in (0,1):
            status=json.loads((case/f'rank-status/rank-{rank}.json').read_text())
            if status != dict(rank=rank,puzzle_id=1000,exit_code=0,status='unsolved',completed_depths=a.depth):
                raise RuntimeError('benchmark requires identical full unsolved depth range')
            logs=[path for path in (case/'ranklogs').rglob('stdout.log') if path.parent.name==str(rank)]
            if len(logs)!=1:raise RuntimeError('missing unique rank stdout')
            rows=[]
            for line in logs[0].read_text().splitlines():
                if not line.startswith('depth_done='):continue
                values=dict(word.split('=',1) for word in line.split() if '=' in word)
                rows.append(dict(depth=int(values['depth_done']),seconds=float(values['depth_sec']),
                    frontier=int(values['next_frontier_size']),stream4_jobs=int(values['stream4_jobs']),
                    stream4_ms=float(values['stream4_ms']) if 'stream4_ms' in values else None,
                    stream12_ms=float(values['stream12_ms']) if 'stream12_ms' in values else None,
                    stream4_busy_max=int(values['stream4_busy_max']) if 'stream4_busy_max' in values else None,
                    stream4_pending_max=int(values['stream4_pending_max']) if 'stream4_pending_max' in values else None))
            if [row['depth'] for row in rows]!=list(range(a.depth)):
                raise RuntimeError('incomplete physical rank depth rows')
            stats=[json.loads(line) for line in (case/f'predict_stats.rank{rank}.jsonl').read_text().splitlines()]
            if ([row['depth'] for row in stats]!=list(range(a.depth)) or
                any(row['rank']!=rank or row['device_local_rank']!=rank for row in stats)):
                raise RuntimeError('bad physical device/depth mapping')
            rank_rows.append(rows)
        depths=[dict(depth=depth,wall_seconds=max(rank_rows[rank][depth]['seconds'] for rank in (0,1)),
                     global_frontier=sum(rank_rows[rank][depth]['frontier'] for rank in (0,1)),
                     ranks=[rank_rows[rank][depth] for rank in (0,1)]) for depth in range(a.depth)]
        receipt=dict(pair=index,mode=mode,depths=depths,production_admitted=False,
                     scope='two RTX3060, timing-instrumented debug build, no sanitizer/H200 inference')
        (case/'receipt.json').write_text(json.dumps(receipt,indent=2))
        results.append(receipt)
        (out/'partial.json').write_text(json.dumps(results,indent=2))
        print(f'DONE pair={index} mode={mode} last_depth_seconds={depths[-1]["wall_seconds"]}',flush=True)

# Compare only matched saturated workloads; otherwise do not manufacture a gain.
last = [entry['depths'][-1] for entry in results]
frontiers = [row['global_frontier'] for row in last]
comparable = len(set(frontiers))==1 and min(frontiers)>=a.beam*0.9
before=[entry['depths'][-1]['wall_seconds'] for entry in results if entry['mode']=='before']
after=[entry['depths'][-1]['wall_seconds'] for entry in results if entry['mode']=='after']
summary=dict(scope='bounded paired end-to-end debug timing, not H200 throughput',
    production_admitted=False,runner_sha256=hashes,beam=a.beam,depth=a.depth,
    comparable_saturated_frontier=comparable,last_global_frontiers=frontiers,
    before_seconds=before,after_seconds=after,
    median_before_seconds=statistics.median(before),median_after_seconds=statistics.median(after),
    relative_after_over_before=(statistics.median(after)/statistics.median(before)) if comparable else None)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
