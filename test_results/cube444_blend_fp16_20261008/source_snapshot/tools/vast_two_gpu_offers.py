"""Read-only bounded rental inventory; no rental mutations."""
import json
import sys
import requests
sys.path.insert(0,r'C:/Users/Иван Литвак/Documents/ChatGPT/MultiGPUBFS/.worktrees/bfs-tail-archive/scripts')
from windows_vast_key import load_key
query={'verified':{'eq':True},'rentable':{'eq':True},'rented':{'eq':False},
    'gpu_name':{'eq':'RTX 3060'},'reliability':{'gte':.99},'cpu_ram':{'gte':32000},
    'num_gpus':{'eq':2},'gpu_ram':{'gte':12000},'disk_space':{'gte':40},
    'cuda_max_good':{'gte':12.8},'type':'on-demand','limit':5,
    'order':[['dph_total','asc']],'allocated_storage':40}
response=requests.post('https://console.vast.ai/api/v0/bundles/',
    headers={'Authorization':'Bearer '+load_key()},json=query,timeout=20,allow_redirects=False)
response.raise_for_status()
items=response.json().get('offers',[])
items=[item for item in items if item.get('num_gpus')==2 and item.get('cuda_max_good',0)>=12.8 and item.get('disk_space',0)>=40]
keys=('id','machine_id','gpu_name','num_gpus','gpu_ram','cpu_ram','disk_space','dph_total','storage_cost','inet_up_cost','inet_down_cost','cuda_max_good','reliability','duration')
print(json.dumps([{k:item.get(k) for k in keys} for item in items],ensure_ascii=False))
