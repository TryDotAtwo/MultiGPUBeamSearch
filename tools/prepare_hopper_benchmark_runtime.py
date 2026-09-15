"""Create a non-overwriting Cube4 runtime and verify offline artifact lineage."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(source,bundle,data,output):
    manifest=json.loads((source/'manifest.json').read_text())
    packed=json.loads((bundle/'manifest.json').read_text())
    assert packed['schema']=='hopper_native_e4m3_experimental_v1'
    assert packed['source_manifest_sha256']==sha(source/'manifest.json')
    assert packed['model']==manifest
    for name,record in packed['files'].items():
        assert Path(name).name==name and Path(record['file']).name==record['file']
        assert sha(source/name)==record['source_sha256'],name
        target=bundle/record['file']
        assert target.stat().st_size==record['bytes'] and sha(target)==record['sha256'],name
    assert len(packed['files'])==60
    puzzle=json.loads((data/'puzzle_info.json').read_text())
    names=manifest['move_names']
    assert names==list(puzzle['generators']), 'Move ordering differs from checkpoint'
    assert len(puzzle['central_state'])==96 and len(names)==24
    output.mkdir(parents=True,exist_ok=False)
    (output/'data').mkdir()
    for name in ('puzzle_info.json','test.csv'):shutil.copyfile(data/name,output/'data'/name)
    target=output/'FullBeamNice'/'generators';target.mkdir(parents=True)
    (target/'p900.json').write_text(json.dumps({'actions':[puzzle['generators'][n] for n in names]}))
    proof=dict(source_manifest_sha256=sha(source/'manifest.json'),bundle_manifest_sha256=sha(bundle/'manifest.json'),
        files_verified=len(packed['files']),puzzle_info_sha256=sha(data/'puzzle_info.json'),test_csv_sha256=sha(data/'test.csv'),move_names=names)
    (output/'lineage.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','bundle','data','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.source,a.bundle,a.data,a.output)
