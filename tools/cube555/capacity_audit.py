"""Native near-limit allocation probes with a separately tuned model microbatch."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from tools.cube555.inference_benchmark import benchmark
from tools.cube555.run import configuration, runtime_plan, DEFAULT_BEAM, BEAM_PROFILES, history_budgets
from tools.cube555.export import export_blend
from tools.cube555.smoke import make_fixture
from tools.cube555.telemetry import Telemetry
from tools.cayleypy_public.data import load_puzzle_contract
from tools.cayleypy_public.model import ExportedModel
from tools.cayleypy_public.runner import maximum_history_depth, PublicSearchRunError
from tools.run_cayleypy_public import validate_t4_hardware, locate_or_build_runner, _run_with_history_budgets, _materialize_run_artifacts, _available_ram_bytes, _derive_history_budgets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--competition',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--full-depth',type=int,default=9)
    p.add_argument('--model-micro',type=int,default=512)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    gpus=validate_t4_hardware()
    micro=a.model_micro
    if not 1 <= micro <= 8192: p.error('--model-micro must be in [1,8192]')
    info=a.assets/'puzzle_info.json'
    manifest=export_blend(a.assets/'q555_f1_bell2k.pt',a.assets/'q555_2k_BEST.pt',a.assets/'piece_layout_555.json',info,a.output/'export')
    model=ExportedModel('cube555-q-blend','fp16',manifest['script_sha256'],manifest,'piece_transformer')
    cfg=SimpleNamespace(assets=a.assets,competition=a.competition,beam=DEFAULT_BEAM,depth=140,touch_radius=5,solution_mode="collect",collect_until_depth=a.full_depth,max_collected_solutions=2000)
    runner=locate_or_build_runner(a.output,info,backend='piece_transformer',config=configuration(cfg,1020,info))
    fixture=Path('/tmp/cube555_macro_micro_fixture');make_fixture(a.assets,fixture)
    jobs=[dict(name='smoke',beam=8192,depth=6,pid=3,competition=fixture,radius=0)]
    jobs += [dict(name=f'capacity-{beam}',beam=beam,depth=2,pid=1020,competition=a.competition,radius=5) for beam in BEAM_PROFILES]
    jobs += [dict(name=f'saturated-{beam}',beam=beam,depth=a.full_depth,pid=1020,competition=a.competition,radius=5) for beam in BEAM_PROFILES]
    report=dict(gpus=gpus,outer_parent_batch=8192,model_microbatch=micro,rows=[],note='Depth2 probes validate allocation only. Depth9 loops require rank-log proof of a full local frontier to establish saturated throughput. All supported widths must pass allocation.')
    for job in jobs:
        out=a.output/job['name'];out.mkdir();row={k:v for k,v in job.items() if k!='competition'}
        cfg.competition=job['competition'];cfg.beam=job['beam'];cfg.depth=140;cfg.collect_until_depth=job['depth'];cfg.touch_radius=job['radius']
        plan=runtime_plan(cfg.beam,model_micro=micro)
        ram,disk=history_budgets(_available_ram_bytes(),shutil.disk_usage('/tmp').free)
        row.update(plan=asdict(plan),history_ram_bytes=ram,history_disk_bytes=disk,budget_max_depth=maximum_history_depth(plan,30,cfg.touch_radius,ram,disk))
        contract=load_puzzle_contract(info,cfg.competition/'test.csv',cfg.competition/'sample_submission.csv',job['pid'],job['pid'])
        monitor=Telemetry(out);monitor.start();start=time.monotonic()
        try:
            artifacts=_run_with_history_budgets(ram,disk,configuration(cfg,job['pid'],info),contract,model,plan,a.output/'export',out/'logs',runner_path=str(runner))
            result=_materialize_run_artifacts(artifacts,out)
            row.update(status='complete',result=result)
            if job['name']=='smoke' and not artifacts.solution_records:
                raise RuntimeError('native microbatched smoke did not solve')
        except Exception as e:
            if isinstance(e,PublicSearchRunError): _materialize_run_artifacts(e.partial_artifacts,out)
            row.update(status='failed',error=str(e))
        finally:
            row.update(wall_seconds=time.monotonic()-start,performance=monitor.finish())
            report['rows'].append(row)
            (a.output/'capacity_report.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(row),flush=True)
        if job['name']=='smoke' and row['status']!='complete':
            raise RuntimeError('smoke failed; capacity tests aborted')
    if any(r['status']!='complete' for r in report['rows']):
        raise RuntimeError('requested Cube555 configuration failed')


if __name__=='__main__':main()
