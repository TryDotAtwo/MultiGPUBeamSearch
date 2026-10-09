"""Remote real two-rank dispatch/union/empty-rank strict CUDA verification."""
import hashlib
import json
import os
from pathlib import Path
from tools.cube4_production_preflight import run_scorer

root=Path('/workspace/pipeline_audit_54195079')
out=root/'evidence/primitive_acceptance_54195079'
out.mkdir(exist_ok=False)
binary=root/'build/final_union_dispatcher_cuda_tests'
identity=hashlib.sha256(binary.read_bytes()).hexdigest()
env={k:v for k,v in os.environ.items() if not k.startswith('BEAM_')}
env['LD_LIBRARY_PATH']='/workspace/nccl225_54195079/nvidia/nccl/lib'
results=[]
for mode in ('prefinal','collect','pipeline','pipeline_empty_rank'):
    case=out/mode
    case.mkdir()
    command=['/venv/main/bin/torchrun','--standalone','--nproc_per_node=2','--no-python',
             'compute-sanitizer','--tool','memcheck','--leak-check','full','--error-exitcode','99',
             str(binary),str(case/'nccl-id')]
    if mode!='prefinal':command.append(mode)
    (case/'request.json').write_text(json.dumps(dict(command=command,binary_sha256=identity),indent=2))
    code=run_scorer(command,case/'runner.log',90,cwd=case,env=env)
    raw=(case/'runner.log').read_text()
    if (code or raw.count('ERROR SUMMARY: 0 errors')!=2 or
        raw.count('LEAK SUMMARY: 0 bytes leaked in 0 allocations')!=2 or
        raw.count('dispatcher_union=PASS')!=2 or 'dispatcher_union=FAIL' in raw):
        raise RuntimeError(f'{mode} failed strict native verification')
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=identity:
        raise RuntimeError('binary mutated')
    results.append(dict(mode=mode,pass_verified=True,physical_ranks=[0,1]))
    print('PASS '+mode,flush=True)
(out/'receipt.json').write_text(json.dumps(dict(results=results,binary_sha256=identity,
    scope='four bounded real two-rank uniform/synthetic primitive controls',
    production_admitted=False,full_goal_complete=False),indent=2))
