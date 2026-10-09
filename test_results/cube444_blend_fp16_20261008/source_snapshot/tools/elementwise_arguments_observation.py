"""Pinned source and separately compiled canonical empty activation exclusion."""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from tools.cube4_numeric_gate import unique
from tools.storage_probe_bundle import validate_probe_build_binding

HEADERS={
    'linear_combination_bias_elementwise.h':'e5524c2ad7a9d4794a0e223e1fea4db9f25ce3d15515247afb3b339b91397838',
    'activation.h':'4574c38e89158b4741aa2e698e7c99fd65f23a023ac777ee5becba298daf199e'}

def validate_empty_arguments_table(value):
    wanted=dict(schema_version=1,scope='compiled_canonical_empty_elementwise_arguments_not_admission',
        entries=[dict(activation=name,empty=True,pinned_empty_type=True,storage_bytes=1,alignment=1)
                 for name in ('Identity','ReLu')])
    if json.dumps(value,sort_keys=True,allow_nan=False)!=json.dumps(wanted,sort_keys=True):
        raise ValueError('canonical elementwise argument exclusion mismatch')
    return value

def observe_headers(build):
    cache=Path(build)/'CMakeCache.txt'
    if cache.stat().st_size>1024*1024:raise ValueError('CMake cache exceeds bound')
    paths=[line.split('=',1)[1] for line in cache.read_text().splitlines()
           if line.startswith('CUTLASS_DIR:PATH=')]
    if len(paths)!=1:raise ValueError('missing unique CUTLASS binding')
    folder=Path(paths[0]).resolve(strict=True)/'include/cutlass/epilogue/thread'
    hashes={}
    for name,wanted in HEADERS.items():
        path=folder/name
        if not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size>1024*1024:
            raise ValueError('invalid elementwise authority header')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=wanted:raise ValueError('elementwise authority header drift: '+name)
        hashes[name]=digest
    return hashes

def build_empty_arguments_probe(source,build,output,timeout):
    validate_probe_build_binding(source,build)
    observe_headers(build)
    with (Path(output)/'elementwise_arguments_probe_build.log').open('xb') as log:
        subprocess.run(['cmake','--build',str(build),'--target','stream1_elementwise_arguments_probe','-j2'],
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)

def observe_empty_arguments_probe(source,build):
    if os.name!='posix':raise ValueError('bounded elementwise observation requires POSIX')
    import resource
    source=Path(source);binary=Path(build)/'stream1_elementwise_arguments_probe'
    if not stat.S_ISREG(binary.lstat().st_mode):raise ValueError('invalid elementwise probe')
    hashes=observe_headers(build)
    digest=hashlib.sha256(binary.read_bytes()).hexdigest()
    def cap():resource.setrlimit(resource.RLIMIT_FSIZE,(65536,65536))
    with tempfile.TemporaryFile() as output:
        subprocess.run([str(binary)],stdout=output,stderr=subprocess.DEVNULL,
                       check=True,timeout=15,preexec_fn=cap)
        if output.tell()>65536:raise ValueError('elementwise probe output exceeds bound')
        output.seek(0);value=json.loads(output.read(),object_pairs_hook=unique)
    validate_empty_arguments_table(value)
    if hashes!=observe_headers(build) or digest!=hashlib.sha256(binary.read_bytes()).hexdigest():
        raise ValueError('elementwise probe/authority changed')
    return dict(table=value,binary_sha256=digest,cutlass_header_sha256=hashes,
                source_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in
                    ('tools/stream1_elementwise_arguments_probe.cu','tools/elementwise_arguments_observation.py')})
