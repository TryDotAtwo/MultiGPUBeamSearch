"""Remote-only Stream1 microbatch sweep; full-pipeline acceptance is separate."""
import pathlib, subprocess, json, sys, urllib.request, time
root=pathlib.Path('/workspace/source')
base='https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBeamSearch/37c379884aba9cf538618e60aa6279d0b51b79e8/'
(root/'tools/inference_first_calibration.py').write_bytes(urllib.request.urlopen(base+'tools/inference_first_calibration.py').read())
sys.path.insert(0,str(root))
from tools.inference_first_calibration import Measurement,select
out=pathlib.Path('/workspace/results/ensemble-calibration');out.mkdir(exist_ok=True)
directory=pathlib.Path('/workspace/ensemble');directory.mkdir(exist_ok=True)
manifest={'schema_version':1,'state_len':96,'output_dim':24,'score_semantics':'distance_q','models':[{'family':'cube444_transformer','weights_dir':'/workspace/inputs/bundle','coefficient':0.6},{'family':'cube444_mlp','weights_dir':'/workspace/inputs/bundle','coefficient':0.2},{'family':'cube444_mlp','weights_dir':'/workspace/inputs/bundle','coefficient':0.2}]}
(directory/'ensemble.json').write_text(json.dumps(manifest))
measurements=[];rejected={}
for batch in (256,32,64,128,512,1024):
 procs=[]
 for device in (0,1):
  log=open(out/f'batch{batch}-gpu{device}.log','w')
  proc=subprocess.Popen(['/workspace/ensemble-benchmark',str(directory),str(batch),'8192',str(device)],stdout=log,stderr=subprocess.STDOUT)
  procs.append((device,proc,log))
 rows=[]
 for device,proc,log in procs:
  try: code=proc.wait(timeout=600)
  except subprocess.TimeoutExpired:proc.kill();proc.wait();code=-1
  log.close()
  text=(out/f'batch{batch}-gpu{device}.log').read_text()
  records=[]
  for line in text.splitlines():
   try:
    x=json.loads(line)
    if 'seconds' in x:records.append(x)
   except ValueError:pass
  if code!=0 or len(records)!=1 or records[0]['numeric_error']!=0:rejected[str(batch)]=f'gpu{device} failed exit{code}';continue
  rows.append(records[0])
 if len(rows)==2:
  for repeat in range(7):measurements.append(Measurement(str(batch),8192,tuple(x['seconds'][repeat] for x in rows),True,True))
 (out/'progress.json').write_text(json.dumps({'finished_batch':batch,'rejected':rejected,'records':[x.__dict__ for x in measurements]},indent=2))
 print(json.dumps({'finished_batch':batch,'accepted':len(rows)==2}),flush=True)
chosen,stat_rejected=select(measurements,baseline='256')
(out/'selection.json').write_text(json.dumps({'selected':chosen.__dict__,'stat_rejected':stat_rejected,'runtime_rejected':rejected,'phase':'STREAM1_ONLY','input':'synthetic_zero_labels','full_pipeline_acceptance':False,'model_count':3},indent=2))
print(json.dumps({'selection':chosen.__dict__}),flush=True)
