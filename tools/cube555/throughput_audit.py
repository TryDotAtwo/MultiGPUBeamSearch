"""Bounded native Stream1 sweep followed by matched saturated pipeline probes."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace
import numpy as np
import torch
from tools.cube555.export import export_blend
from tools.cube555.run import configuration, runtime_plan
from tools.cube555.telemetry import Telemetry
from tools.run_cayleypy_public import locate_or_build_runner, validate_t4_hardware


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--competition', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + 3300
    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError('55 minute audit budget exhausted')
        return seconds
    gpus = validate_t4_hardware()
    manifest = export_blend(a.assets/'q555_f1_bell2k.pt', a.assets/'q555_2k_BEST.pt',
        a.assets/'piece_layout_555.json', a.assets/'puzzle_info.json', a.output/'export')
    info = json.loads((a.assets/'puzzle_info.json').read_text())
    moves = np.asarray(list(info['generators'].values()))
    rng = np.random.default_rng(555)
    state = np.arange(150, dtype=np.uint8)
    corpus = np.zeros((8192,160),dtype=np.uint8)
    for i in range(8192):
        state = state[moves[rng.integers(len(moves))]]
        corpus[i,:150] = state
    corpus_path = a.output/'parents.u8'
    corpus.tofile(corpus_path)
    cfg = SimpleNamespace(assets=a.assets,competition=a.competition,beam=1048576,depth=140,
        touch_radius=5,solution_mode='collect',collect_until_depth=8,max_collected_solutions=2000)
    runner = locate_or_build_runner(a.output, a.assets/'puzzle_info.json', backend='piece_transformer',
        config=configuration(cfg,1020,a.assets/'puzzle_info.json'))
    subprocess.run(['cmake','--build',str(runner.parent),'--target','cube555_stream1_benchmark','-j','2'],check=True,timeout=remaining())
    binary = runner.parent/'cube555_stream1_benchmark'
    report = dict(gpus=gpus,torch=torch.__version__,manifest=manifest,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        corpus_sha256=hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        outer=8192,concurrency_sweep=[1,2,4],rows=[],pipeline=[],
        metric='1 - pipeline_parents_per_second / isolated_Stream1_parents_per_second',
        excluded='isolated: model load/input H2D; included: production forward, fp32 blend, quantizer, score-ring copy. Pipeline includes all search stages; state populations differ but shapes/work counts match.')
    def save():
        report['audit_wall_seconds'] = time.monotonic()-started
        (a.output/'throughput_report.json').write_text(json.dumps(report,indent=2))
    for micro, concurrency in itertools.product((64,128,256,384,512,768,1024,1536,2048),(1,2,4)):
        directory = a.output/f'isolated-{micro}-c{concurrency}'
        directory.mkdir()
        monitor = Telemetry(directory)
        monitor.start()
        processes=[]
        try:
            for gpu in range(2):
                out=(directory/f'gpu-{gpu}.log').open('w')
                err=(directory/f'gpu-{gpu}.err').open('w')
                proc=subprocess.Popen([str(binary),str(a.output/'export'),str(corpus_path),str(micro),str(gpu),'5',str(concurrency)],stdout=out,stderr=err,
                    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
                processes.append((gpu,proc,out,err))
            for gpu,proc,out,err in processes:
                code=proc.wait(timeout=min(180,remaining()))
                out.close();err.close()
                row=dict(gpu=gpu,micro=micro,concurrency=concurrency,return_code=code,status='failed')
                if code==0:
                    row.update(json.loads((directory/f'gpu-{gpu}.log').read_text()))
                    row.update(status='complete',median_seconds=statistics.median(row['seconds']))
                    row['parents_per_second']=8192*concurrency/row['median_seconds']
                report['rows'].append(row)
                print(json.dumps(row),flush=True)
        finally:
            for _,proc,out,err in processes:
                if proc.poll() is None: proc.kill();proc.wait()
                out.close();err.close()
            report.setdefault('isolated_telemetry',{})[f'{micro}-c{concurrency}']=monitor.finish()
            save()
    successful={}
    for row in report['rows']:
        if row['status']=='complete': successful.setdefault((row['micro'],row['concurrency']),[]).append(row)
    ranked=sorted((1/min(r['parents_per_second'] for r in rows),case) for case,rows in successful.items() if len(rows)==2)
    if not ranked: raise RuntimeError('no both-GPU passing microbatch')
    winner=ranked[0][1]
    safe=sorted(case[0] for _,case in ranked if case[1]==winner[1])
    index=safe.index(winner[0])
    neighbors=[(m,winner[1]) for m in safe[max(0,index-1):min(len(safe),index+2)]]
    for _,case in ranked:
        if len(neighbors)>=4: break
        if case[1] not in {chosen[1] for chosen in neighbors}: neighbors.append(case)
    report['isolated_winner']=dict(micro=winner[0],concurrency=winner[1])
    report['pipeline_candidates']=neighbors
    save()
    for micro, concurrency in neighbors:
        directory=a.output/f'pipeline-{micro}-c{concurrency}'
        log=a.output/f'pipeline-{micro}-c{concurrency}.launcher.log'
        print(f'pipeline_start micro={micro} concurrency={concurrency}',flush=True)
        with log.open('w') as stream:
            command=[os.sys.executable,'-u','-m','tools.cube555.run',
                '--assets',str(a.assets),'--competition',str(a.competition),'--output',str(directory),
                '--pids','1020','--beam','1048576','--depth','140','--collect-until-depth','8',
                '--touch-radius','5','--model-micro',str(micro),'--inference-concurrency',str(concurrency)]
            proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in proc.stdout:
                print(line,end='',flush=True);stream.write(line);stream.flush()
                remaining()
            if proc.wait(timeout=remaining())!=0: raise RuntimeError(f'pipeline failed {micro}/{concurrency}')
        report['pipeline'].append(dict(micro=micro,concurrency=concurrency,summary=json.loads((directory/'run_summary.json').read_text())))
        save()
    report['status']='complete'
    save()


if __name__=='__main__': main()
