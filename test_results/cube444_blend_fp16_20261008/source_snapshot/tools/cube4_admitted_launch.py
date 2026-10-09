"""Consume the independently replayed exact-profile admission before ranks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if not __package__:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.cube4_production_admission import read,digest,validate,raw
from tools.cuda_device_observation import observe
from tools.cube4_run_supervisor import stop_group,atomic_summary,verify_rank_result


def validate_native_publication(native,copy):
    # The exclusive run directory belongs to this invocation. Rank0 publishes
    # both CSVs; the coordinator must consume them rather than recreate them.
    submitted=raw(native,1024*1024)
    if submitted!=raw(copy,1024*1024):
        raise ValueError('native submission differs from its unique path copy')
    return hashlib.sha256(submitted).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','runner','build-receipt','transaction','admission','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--puzzle-id',type=int,required=True)
    parser.add_argument('--mode',choices=('baseline','optimized'),default='optimized')
    args=parser.parse_args()
    if os.name!='posix':raise SystemExit('Remote POSIX only')
    processes,logs=[],[]
    try:
        decision=validate(args.source_root,args.runner,args.build_receipt,args.transaction)
        if read(args.admission)!=decision:raise ValueError('admission receipt stale or altered')
        tx=read(args.transaction);world=tx['world'];cohort=Path(tx['cohort'])
        if type(args.puzzle_id) is not int or not 954000<=args.puzzle_id<954064:
            raise ValueError('puzzle is outside the admitted validation cohort')
        row=args.puzzle_id-954000
        mapping=observe(args.runner.parent/'cuda_visible_device_probe')
        observed=mapping['visible_devices']['devices'];hardware=read(args.build_receipt)['hardware']
        if len(observed)!=world or any(device['device']!=rank or device['sm']!=86 or
            device['uuid_hex']!=hardware[rank]['uuid'].removeprefix('GPU-').replace('-','').lower()
            for rank,device in enumerate(observed)):raise ValueError('different current hardware')
        profile=decision['profiles'][args.mode]
        reference=read(cohort/'reference.json');puzzle=read(cohort/'data/puzzle_info.json')
        contract=dict(puzzle_id=args.puzzle_id,initial_state=reference['states'][row],
                      central_state=puzzle['central_state'],generators=puzzle['generators'],requested_depth=12)
        args.output.mkdir(exist_ok=False)
        request=dict(schema_version=1,admission_sha256=digest(args.admission),
            runner_sha256=digest(args.runner),profile=profile,world=world,contract=contract)
        with (args.output/'request.json').open('x') as stream:json.dump(request,stream,indent=2)
        env={k:v for k,v in os.environ.items() if not k.startswith(('BEAM_','CUDA_'))}
        env.update(profile,BEAM_WEIGHT_DIR=str(cohort/'weights_fp16'),
            BEAM_GENERATOR_PATH=str(cohort/'data/puzzle_info.json'),BEAM_PUZZLE_INFO_JSON=str(cohort/'data/puzzle_info.json'),
            BEAM_TEST_CSV=str(cohort/'data/test.csv'),BEAM_NCCL_ID_FILE=str(args.output/'nccl-id'),
            BEAM_NCCL_RUN_ID=str(args.output),BEAM_RANK_STATUS_DIR=str(args.output/'rank-status'),
            BEAM_HISTORY_DIR=str(args.output/'history'),BEAM_HISTORY_DISK_PATH=str(args.output/'history-disk'),
            BEAM_PREDICT_STATS_PATH=str(args.output/'predict_stats'))
        started=time.monotonic()
        for rank in range(world):
            log=(args.output/f'rank{rank}.log').open('xb');logs.append(log)
            processes.append(subprocess.Popen([str(args.runner),str(args.puzzle_id),'12','4096'],
                cwd=args.output,env=dict(env,RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE=str(world)),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        for process in processes:
            remaining=180-(time.monotonic()-started)
            if remaining<=0 or process.wait(timeout=remaining):raise RuntimeError('admitted rank failed or timed out')
        for log in logs:log.close()
        submissions=list((args.output/'test_results').glob(f'submit_p{args.puzzle_id}_*.csv'))
        if len(submissions)!=1:raise ValueError('admitted solve missing/ambiguous')
        submission_sha=validate_native_publication(args.output/'submit.csv',submissions[0])
        result=verify_rank_result(args.output,world,contract,deadline=time.monotonic()+30)
        if digest(args.runner)!=request['runner_sha256']:raise ValueError('runner changed during admitted execution')
        result.update(production_admitted=True,admission_sha256=request['admission_sha256'],
            elapsed_seconds=time.monotonic()-started,scope=decision['scope'],
            native_submission_sha256=submission_sha)
        atomic_summary(args.output,result);print(json.dumps(result));return 0
    except (ValueError,OSError,KeyError,TypeError,RuntimeError,subprocess.SubprocessError) as error:
        print('admitted launch rejected: '+str(error),file=sys.stderr);return 2
    finally:
        for process in processes:stop_group(process)
        for log in logs:
            if not log.closed:log.close()

if __name__=='__main__':raise SystemExit(main())
