"""Lossless export of the pinned Kaggle NPZ heads; no inference or BN re-folding."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

EXPECTED = {
    's3.npz': '1239e036f9fc792ffcf177c867aca758a48098966f54c1b4c4427127a68b4dca',
    'mlp_x16.npz': 'bddae95ba86de5a007daf284d6d9f66585a7792a59e4fc4ad26288862590933a',
}

def export(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    rows, files, inputs = [], {}, {}
    for prefix, name in [('s3','s3.npz'), ('mlp','mlp_x16.npz')]:
        path=source/name
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=EXPECTED[name]:
            raise ValueError(f'Unadmitted model identity: {name}')
        inputs[name]=digest
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                a=np.ascontiguousarray(data[key])
                if a.dtype not in (np.dtype('float32'), np.dtype('int32')) or not np.isfinite(a).all():
                    raise ValueError(f'Invalid dtype/nonfinite tensor: {name}/{key}')
                full=prefix+'/'+key
                filename=hashlib.sha256(full.encode()).hexdigest()+'.bin'
                (destination/filename).write_bytes(a.tobytes())
                files[filename]=hashlib.sha256(a.tobytes()).hexdigest()
                rows.append('\t'.join([full,filename,str(a.dtype),','.join(map(str,a.shape))]))
    table='\n'.join(rows)+'\n'
    (destination/'tensors.tsv').write_text(table,encoding='utf-8')
    files['tensors.tsv']=hashlib.sha256(table.encode()).hexdigest()
    manifest={'backend':'cube444_q_blend','dtype':'fp32','state_len':96,'num_classes':6,
              'output_dim':24,'transformer_weight':0.6,'mlp_weight':0.4,
              'inputs':inputs,'files':files}
    (destination/'blend.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    export(args.source,args.out)
