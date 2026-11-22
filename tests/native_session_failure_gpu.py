"""Cancellation and one-rank initialization failure on the real CUDA bridge."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time


def main():
    parser=argparse.ArgumentParser()
    for name in ('fixture','runner','library','extension','out'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('_beam_runtime',args.extension)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=False)
    fixture=args.fixture.resolve()
    puzzle=json.loads((fixture/'puzzle.json').read_text())
    start=puzzle['central_state']
    for move in puzzle['generators'].values():start=[start[i] for i in move]
    records=[]
    for mode in ('cancel','failed_rank'):
        directory=root/mode;directory.mkdir()
        env=dict(os.environ,BEAM_GENERATOR_PATH=str(fixture/'puzzle.json'),
            BEAM_PUZZLE_INFO_JSON=str(fixture/'puzzle.json'),BEAM_TEST_CSV=str(fixture/'test.csv'),
            BEAM_WEIGHT_DIR=str(fixture/'weights'),BEAM_RUNTIME_CONFIG_MODE='auto',
            BEAM_SOLVED_NEIGHBORHOOD_RADIUS='0',BEAM_STREAM2_SUFFIX_RADIUS='0',
            BEAM_SOLVE_BUCKET_MODE='0',BEAM_HISTORY_MODE='disk',BEAM_B_MICRO='768',
            BEAM_SHARD_COUNT='1',BEAM_STREAM4_BATCH_CANDIDATES='1024',
            BEAM_HISTORY_DIR=str(directory/'history'),BEAM_HISTORY_DISK_PATH=str(directory/'history'),
            BEAM_NCCL_ID_FILE=str(directory/'nccl-id.bin'),
            BEAM_NATIVE_LIBRARY=str(args.library.resolve()),
            CUDA_VISIBLE_DEVICES='0,1')
        for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','BEAM_SESSION_DIR'):
            env.pop(key,None)
        command=[str(args.runner.resolve()),'0','32','64']
        if mode=='failed_rank':
            wrapper=directory/'fail-one-rank.py'
            wrapper.write_text("import os,sys\n"
                "if os.environ['RANK']=='1':os.environ['BEAM_NATIVE_LIBRARY']='/beam-no-such-library.so'\n"
                "os.execv(sys.argv[1],sys.argv[1:])\n")
            command=[sys.executable,str(wrapper),*command]
        group=module.ResidentWorkers(command,2,
            directory/'session',directory,env)
        errors=[]
        def submit():
            try:group.submit(start,directory,20.0)
            except RuntimeError as error:errors.append(str(error))
        began=time.monotonic()
        worker=threading.Thread(target=submit);worker.start()
        if mode=='cancel':
            deadline=time.monotonic()+10
            while not (directory/'resident-rank-0.log').exists() and worker.is_alive():
                if time.monotonic()>deadline:break
                time.sleep(.005)
            group.cancel()
        worker.join(timeout=25)
        assert not worker.is_alive(), 'supervisor failed to terminate'
        group.close()
        assert errors, (mode,'missing terminal error')
        assert ('cancelled' if mode=='cancel' else 'exited') in errors[0],errors
        codes=group.exit_codes()
        assert all(code is not None for code in codes),codes
        records.append(dict(mode=mode,error=errors[0],exit_codes=codes,seconds=time.monotonic()-began))
    (root/'results.json').write_text(json.dumps(records,indent=2))
    print('NATIVE_SESSION_FAILURE_GPU_OK',flush=True)


if __name__=='__main__':main()
