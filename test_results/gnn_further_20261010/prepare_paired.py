from pathlib import Path
p=Path(__file__).resolve().parent
b=(p.parent/'gnn_bias_20261010/paired.py').read_text().replace('bias-','further-').replace('beam_gnn_bias_','beam_gnn_further_')
b=b.replace("Path('/root/further-result/paired-'", "torch.save(actual.cpu(),'/root/further-output-'+variant+'.pt')\nPath('/root/further-result/paired-'")
(p/'paired.py').write_text(b)
driver=r'''
import json,time,subprocess,sys,torch
from pathlib import Path
out=Path('/root/further-result')
while not Path('/root/further-softmax-status/exit.json').exists():time.sleep(2)
assert json.loads(Path('/root/further-softmax-status/exit.json').read_text())['returncode']==0
try:
 for rid,order in [('1',('baseline','candidate','softmax')),('2',('softmax','candidate','baseline'))]:
  for variant in order:subprocess.run([sys.executable,'/root/further-paired.py',variant,rid],check=True)
 base=torch.load('/root/further-output-baseline.pt',weights_only=True)
 candidate=torch.load('/root/further-output-candidate.pt',weights_only=True)
 softmax=torch.load('/root/further-output-softmax.pt',weights_only=True)
 torch.testing.assert_close(candidate,base,rtol=.03,atol=.02)
 assert torch.equal(softmax,candidate),'softmax changed ordered FP16 results'
 (out/'paired-parity.json').write_text(json.dumps({'norm_vs_baseline_max_error':(candidate-base).abs().max().item(),'softmax_bitwise_equal_norm':True}))
 (out/'paired-exit.json').write_text(json.dumps({'returncode':0}))
except Exception as e:
 (out/'paired-exit.json').write_text(json.dumps({'returncode':1,'error':repr(e)}));raise
'''
payload="from pathlib import Path\nimport subprocess,sys\nPath('/root/further-paired.py').write_text("+repr(b)+")\nPath('/root/further-paired-job.py').write_text("+repr(driver)+")\nwith Path('/root/further-paired.log').open('ab') as log:\n p=subprocess.Popen([sys.executable,'/root/further-paired-job.py'],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)\nprint(p.pid)\n"
(p/'start_paired.py').write_text(payload)
