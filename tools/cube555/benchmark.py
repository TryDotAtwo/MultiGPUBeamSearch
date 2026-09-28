"""Compare complete depth-8 loops on identical Cube555 input and beam."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import shutil
import time

from tools.cube555.run import configuration, runtime_plan
from tools.cube555.export import export_blend
from tools.cube555.telemetry import Telemetry
from tools.cayleypy_public.data import load_puzzle_contract
from tools.cayleypy_public.model import ExportedModel
from tools.run_cayleypy_public import validate_t4_hardware, locate_or_build_runner, _run_with_history_budgets, _materialize_run_artifacts, _available_ram_bytes

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--competition',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--beam',type=int,default=65536)
    p.add_argument('--pid',type=int,default=1020)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    names=validate_t4_hardware()
    info=a.assets/'puzzle_info.json'
    cfgargs=SimpleNamespace(assets=a.assets,competition=a.competition,beam=a.beam,depth=9,touch_radius=2)
    config=configuration(cfgargs,a.pid,info)
    contract=load_puzzle_contract(info,a.competition/'test.csv',a.competition/'sample_submission.csv',a.pid,a.pid)
    manifest=export_blend(a.assets/'q555_f1_bell2k.pt',a.assets/'q555_2k_BEST.pt',a.assets/'piece_layout_555.json',info,a.output/'export')
    model=ExportedModel('cube555-q-blend','fp16',manifest['script_sha256'],manifest,'piece_transformer')
    runner=locate_or_build_runner(a.output,info,backend='piece_transformer',config=config)
    rows=[]
    for profile in ['safe','balanced','throughput']:
        out=a.output/profile;out.mkdir();monitor=Telemetry(out);monitor.start();start=time.monotonic()
        row=dict(profile=profile,plan=asdict(runtime_plan(a.beam,profile)))
        try:
            artifacts=_run_with_history_budgets(min(8*1024**3,_available_ram_bytes()-4*1024**3),min(16*1024**3,shutil.disk_usage('/tmp').free-4*1024**3),config,contract,model,runtime_plan(a.beam,profile),a.output/'export',out/'logs',runner_path=str(runner))
            _materialize_run_artifacts(artifacts,out);row['status']='complete'
        except Exception as e:
            row.update(status='failed',error=str(e))
        finally:
            row['wall_seconds']=time.monotonic()-start;row['performance']=monitor.finish()
            times=[x['seconds'] for x in row['performance']['depths'] if x['depth']==8]
            row['depth8_seconds']=times[0] if len(times)==1 else None
            rows.append(row)
            (a.output/'profile_comparison.json').write_text(json.dumps(dict(gpus=names,beam=a.beam,pid=a.pid,rows=rows),indent=2))
            print(json.dumps(row),flush=True)
    if any(r['status']!='complete' or r['depth8_seconds'] is None for r in rows):
        raise RuntimeError('At least one profile failed the complete depth-8 comparison')

if __name__=='__main__':main()
