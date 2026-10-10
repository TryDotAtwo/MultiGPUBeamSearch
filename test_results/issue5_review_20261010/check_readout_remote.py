import os,json,subprocess,shutil
from pathlib import Path
root=Path('/root/gnn-beam-result')
result=json.loads((root/'result.json').read_text())
assert len(result['records'])==4 and result['status']=='PASS'
checks=[]
for record in result['records']:
 if record['model']!='blend3':continue
 run=Path(record['metadata']['run_dir'])
 candidates=list(run.rglob('ensemble.json'))
 assert len(candidates)==1,candidates
 destination=root/('oracle-'+record['backend'])
 shutil.copytree(candidates[0].parent,destination,dirs_exist_ok=True)
 manifest=json.loads((destination/'ensemble.json').read_text())
 manifest['generators']=[list(range(k-1,-1,-1))+list(range(k,4)) for k in range(2,5)]
 manifest['calibration_states']=[[3,1,0,2],[1,0,3,2],[2,1,3,0],[0,3,2,1]]
 (destination/'ensemble.json').write_text(json.dumps(manifest))
 binary=root/'cache/builds'/record['metadata']['build']['build_key']/'stream1_ensemble_benchmark'
 env=dict(os.environ,BEAM_ENSEMBLE_BACKEND=record['backend'])
 job=subprocess.run([str(binary),str(destination),'4','4','0'],env=env,text=True,capture_output=True,timeout=90)
 (root/('readout-'+record['backend']+'.log')).write_text(job.stdout+job.stderr)
 assert job.returncode==0,job.stdout+job.stderr
 report=json.loads(job.stdout.strip().splitlines()[-1])
 assert report['correctness_passed'] and report['numeric_error']==0
 checks.append(dict(backend=record['backend'],report=report))
(root/'readout.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks))
