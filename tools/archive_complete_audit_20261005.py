"""Remote-only source/evidence/binary archive; excludes per-case history arenas."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.name!='posix':raise SystemExit('Remote archive only')
    if args.output.exists():raise ValueError('archive destination already exists')
    paths=set()
    for root in (Path('/workspace/complete_audit_20261005'),Path('/workspace/complete_audit_v6_staged')):
        manifest=root/'source_manifest.json';paths.add(manifest)
        paths.update(root/name for name in json.loads(manifest.read_text()))
        for path in (root/'evidence/complete_audit').rglob('*'):
            if path.is_file() and not any(part=='history' or part.startswith('history-disk') for part in path.relative_to(root).parts):
                paths.add(path)
        for build in ('build-complete','build-diagnostics','build-timing'):
            directory=root/build
            if not directory.exists():continue
            for path in directory.iterdir():
                if path.is_file() and (os.access(path,os.X_OK) or path.name in ('CMakeCache.txt','build.ninja','.ninja_deps')):
                    paths.add(path)
    paths.add(Path('/workspace/complete_audit_20261005/evidence/dual_base_profile.json'))
    for directory in ('/workspace/complete_audit_cohort_20261005','/workspace/cube4_heldout_source_54295544',
                      '/workspace/cupti_replay_54295544','/workspace/nccl225_complete/nvidia/nccl'):
        paths.update(path for path in Path(directory).rglob('*') if path.is_file() and
                     not any(part in ('__pycache__','.git') for part in path.parts))
    paths.update(Path('/root').glob('complete_audit_source*_20261005.tar.gz'))
    paths.update(Path('/root').glob('*complete*20261005.sh'))
    manifest={}
    with tarfile.open(args.output,'w:gz',dereference=True) as archive:
        for path in sorted(paths):
            if not path.exists():raise ValueError('missing archive artifact: '+str(path))
            name=str(path.relative_to('/'));sha=digest(path)
            archive.add(path,arcname=name,recursive=False)
            if digest(path)!=sha:raise ValueError('artifact changed while archiving: '+str(path))
            manifest[name]=dict(sha256=sha,bytes=path.stat().st_size)
        raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
        info=tarfile.TarInfo('ARTIFACT_MANIFEST.json');info.size=len(raw)
        archive.addfile(info,io.BytesIO(raw))
    receipt=dict(status='pass',archive=str(args.output),bytes=args.output.stat().st_size,
        sha256=digest(args.output),files=len(manifest),
        scope='source,models,raw evidence,traces,native binaries; per-case history backing arenas excluded')
    with args.output.with_suffix(args.output.suffix+'.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt))

if __name__=='__main__':main()
