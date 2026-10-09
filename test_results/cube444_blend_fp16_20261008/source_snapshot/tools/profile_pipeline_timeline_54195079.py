"""One bounded real two-rank CUDA graph/selection timeline, not a timing row."""
import hashlib
import json
import os
from pathlib import Path
from tools.cube4_bundle_contract import validate_cube4_bundle
from tools.cube4_production_preflight import run_scorer

root=Path('/workspace/pipeline_audit_54195079')
out=root/'evidence/nsys_pipeline_54195079'
out.mkdir(exist_ok=False)
previous=json.loads((root/'evidence/pipeline_ab_hol54195079/2_after/request.json').read_text())
profile=dict(previous['profile'])
profile.update(BEAM_HISTORY_DIR=str(out/'history'),BEAM_HISTORY_DISK_PATH=str(out/'history-disk'),
    BEAM_RANK_STATUS_DIR=str(out/'rank-status'),BEAM_NCCL_ID_FILE=str(out/'nccl-id'),
    BEAM_PREDICT_STATS_PATH=str(out/'predict_stats'))
data=root/'evidence/recovered_campaign_inputs_20261001'
weights=root/'evidence/cube4_paired_reference_20261001/weights_fp16'
identity=validate_cube4_bundle(data,weights,'1000')
binary=root/'build/production_runner'
digest=hashlib.sha256(binary.read_bytes()).hexdigest()
if digest!=previous['runner_sha256']:raise RuntimeError('not the paired after binary')
command=['/venv/main/bin/torchrun','--standalone','--nproc_per_node=2','--no-python',
    '--log-dir',str(out/'ranklogs'),'--redirects','3','nsys','profile',
    '--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace=graph',
    '--force-overwrite=false','-o',str(out/'rank_%q{RANK}'),str(binary),'1000','8','65536']
env={k:v for k,v in os.environ.items() if not k.startswith('BEAM_')}
env.update(profile,LD_LIBRARY_PATH='/workspace/nccl225_54195079/nvidia/nccl/lib')
(out/'request.json').write_text(json.dumps(dict(command=command,profile=profile,
    runner_sha256=digest,production_admitted=False,not_a_performance_row=True),indent=2))
code=run_scorer(command,out/'launcher.log',180,cwd=out,env=env)
if code:raise RuntimeError('profiled runner failed; preserve profiler/rank logs')
for rank in (0,1):
    status=json.loads((out/f'rank-status/rank-{rank}.json').read_text())
    if status!=dict(rank=rank,puzzle_id=1000,exit_code=0,status='unsolved',completed_depths=8):
        raise RuntimeError('profile did not cover matching full depth range')
    report=out/f'rank_{rank}.nsys-rep'
    if not report.is_file() or report.stat().st_size==0:raise RuntimeError('missing profiler report')
if hashlib.sha256(binary.read_bytes()).hexdigest()!=digest or validate_cube4_bundle(data,weights,'1000')!=identity:
    raise RuntimeError('profiled inputs changed')
(out/'receipt.json').write_text(json.dumps(dict(status='pass',physical_ranks=[0,1],
    scope='CUDA graph-level trace, no CPU sampling/context-switch tracing, no profiler-free speed claim',
    production_admitted=False),indent=2))
print('PASS profiled two-rank depth8 beam65536')
