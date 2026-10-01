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
    p.add_argument('--microbatches', type=int, nargs='+',
                   default=[64, 128, 256, 384, 512, 768, 1024, 1536, 2048])
    p.add_argument('--concurrencies', type=int, choices=(1, 2, 4), nargs='+', default=[1, 2, 4])
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--parent-groups', type=int, default=0,
                   help='fixed timed work: total8192-parent groups per case, divided across lanes')
    p.add_argument('--isolated-only', action='store_true')
    p.add_argument('--runtime-target', choices=['kaggle-2xt4', 'molab-single-gpu'],
                   default='kaggle-2xt4')
    p.add_argument('--time-budget-seconds', type=int, default=3300)
    p.add_argument('--case-sequence', nargs='+',
                   help='explicit repeated micro:concurrency order for isolated paired controls')
    a = p.parse_args()
    if not 300 <= a.time_budget_seconds <= 7200:
        p.error('--time-budget-seconds must be within [300,7200]')
    if not 3 <= a.repeats <= 100:
        p.error('--repeats must be within [3,100]')
    if any(not 1 <= micro <= 8192 for micro in a.microbatches):
        p.error('--microbatches must be within [1,8192]')
    if len(set(a.microbatches)) != len(a.microbatches) or len(set(a.concurrencies)) != len(a.concurrencies):
        p.error('sweep cases must not contain duplicates')
    if a.parent_groups < 0 or (a.parent_groups and any(
            a.parent_groups % c or a.parent_groups//c < 3 for c in a.concurrencies)):
        p.error('--parent-groups must divide across all lane counts with >=3 repeats each')
    cases = list(itertools.product(a.microbatches, a.concurrencies))
    if a.case_sequence:
        if not a.isolated_only or not a.parent_groups:
            p.error('--case-sequence requires --isolated-only and fixed --parent-groups')
        try:
            cases = [tuple(map(int, value.split(':'))) for value in a.case_sequence]
            if any(len(case) != 2 or not 1 <= case[0] <= 8192 or case[1] not in (1, 2, 4)
                   or a.parent_groups % case[1] or a.parent_groups//case[1] < 3
                   for case in cases):
                raise ValueError('invalid case')
        except ValueError:
            p.error('each case must be supported micro:concurrency with equal timed parent work')
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + a.time_budget_seconds
    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError('bounded audit time budget exhausted')
        return seconds
    if a.runtime_target == 'molab-single-gpu':
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Molab benchmark requires one real GPU')
        gpus = [torch.cuda.get_device_name(0)]
        major, minor = torch.cuda.get_device_capability(0)
        cuda_arch = 10 * major + minor
    else:
        gpus = validate_t4_hardware()
        cuda_arch = 75
    preparation_monitor = Telemetry(a.output)
    preparation_monitor.start()
    print(f'audit_preparing target={a.runtime_target} cuda_arch={cuda_arch}', flush=True)
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
        config=configuration(cfg,1020,a.assets/'puzzle_info.json'), cuda_arch=cuda_arch)
    subprocess.run(['cmake','--build',str(runner.parent),'--target','cube555_stream1_benchmark','-j','2'],check=True,timeout=remaining())
    preparation_performance = preparation_monitor.finish()
    binary = runner.parent/'cube555_stream1_benchmark'
    report = dict(gpus=gpus,torch=torch.__version__,manifest=manifest,
        runtime_target=a.runtime_target, cuda_arch=cuda_arch,
        preparation_performance=preparation_performance,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        corpus_sha256=hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        outer=8192,concurrency_sweep=a.concurrencies,microbatch_sweep=a.microbatches,
        repeats=({str(c): a.parent_groups//c for c in a.concurrencies}
                 if a.parent_groups else a.repeats),
        warmup_parent_groups=8,warmup_parent_count=65536,
        isolated_only=a.isolated_only,rows=[],pipeline=[],
        fixed_total_parent_groups=a.parent_groups,
        metric='1 - pipeline_parents_per_second / isolated_Stream1_parents_per_second',
        excluded='isolated: model load/input H2D; included: production forward, fp32 blend, quantizer, score-ring copy. Pipeline includes all search stages; state populations differ but shapes/work counts match.')
    def save():
        report['audit_wall_seconds'] = time.monotonic()-started
        (a.output/'throughput_report.json').write_text(json.dumps(report,indent=2))
    report['case_sequence'] = cases
    for case_index, (micro, concurrency) in enumerate(cases):
        print(f'isolated_case_start index={case_index} micro={micro} concurrency={concurrency}',
              flush=True)
        repeats = a.parent_groups//concurrency if a.parent_groups else a.repeats
        label = f'{micro}-c{concurrency}'
        if a.case_sequence:
            label = f'block-{case_index:02d}-'+label
        directory = a.output/f'isolated-{label}'
        directory.mkdir()
        monitor = Telemetry(directory)
        monitor.start()
        processes=[]
        try:
            for gpu in range(len(gpus)):
                out=(directory/f'gpu-{gpu}.log').open('w')
                err=(directory/f'gpu-{gpu}.err').open('w')
                proc=subprocess.Popen([str(binary),str(a.output/'export'),str(corpus_path),str(micro),str(gpu),str(repeats),str(concurrency)],stdout=out,stderr=err,
                    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
                processes.append((gpu,proc,out,err))
            for gpu,proc,out,err in processes:
                code=proc.wait(timeout=min(300 if a.parent_groups else 180, remaining()))
                out.close();err.close()
                row=dict(gpu=gpu,micro=micro,concurrency=concurrency,case_index=case_index,
                         return_code=code,status='failed')
                if code==0:
                    row.update(json.loads((directory/f'gpu-{gpu}.log').read_text()))
                    row.update(status='complete',median_seconds=statistics.median(row['seconds']))
                    row['repeat_groups'] = repeats
                    row['timed_parents'] = 8192*concurrency*repeats
                    if a.parent_groups:
                        steady = row['seconds'][len(row['seconds'])//2:]
                        row['early_median_seconds'] = statistics.median(row['seconds'][:len(row['seconds'])//2])
                        row['all_repeat_median_seconds'] = row['median_seconds']
                        row['median_seconds'] = statistics.median(steady)
                        row['steady_spread_fraction'] = (max(steady)-min(steady))/row['median_seconds']
                        if row['steady_spread_fraction'] > 0.10:
                            row['status'] = 'unstable'
                    row['parents_per_second']=8192*concurrency/row['median_seconds']
                report['rows'].append(row)
                print(json.dumps(row),flush=True)
        finally:
            for _,proc,out,err in processes:
                if proc.poll() is None: proc.kill();proc.wait()
                out.close();err.close()
            report.setdefault('isolated_telemetry',{})[label]=monitor.finish()
            save()
    if a.case_sequence:
        report.update(status='complete', isolated_winner=None, pipeline_candidates=[],
                      comparison_note='Repeated same-work controls retained; no chronological winner selected.')
        save()
        return
    successful={}
    for row in report['rows']:
        if row['status']=='complete': successful.setdefault((row['micro'],row['concurrency']),[]).append(row)
    ranked=sorted((1/min(r['parents_per_second'] for r in rows),case) for case,rows in successful.items() if len(rows)==len(gpus))
    if not ranked: raise RuntimeError('no both-GPU passing microbatch')
    winner=ranked[0][1]
    safe=sorted(case[0] for _,case in ranked if case[1]==winner[1])
    index=safe.index(winner[0])
    neighbors=[(m,winner[1]) for m in safe[max(0,index-1):min(len(safe),index+2)]]
    for _,case in ranked:
        if len(neighbors)>=4: break
        if case[1] not in {chosen[1] for chosen in neighbors}: neighbors.append(case)
    report['isolated_winner']=dict(micro=winner[0],concurrency=winner[1])
    if a.isolated_only:
        report.update(status='complete', pipeline_candidates=[])
        save()
        return
    report['pipeline_candidates']=neighbors
    save()
    for micro, concurrency in neighbors:
        directory=a.output/f'pipeline-{micro}-c{concurrency}'
        log=a.output/f'pipeline-{micro}-c{concurrency}.launcher.log'
        print(f'pipeline_start micro={micro} concurrency={concurrency}',flush=True)
        with log.open('w') as stream:
            command=[os.sys.executable,'-u','-m','tools.cube555.run',
                '--runtime-target',a.runtime_target,
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
