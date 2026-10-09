"""Remote build transaction with before/after source and binary provenance."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
if not __package__:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.nccl_library_observation import observe as observe_nccl


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for name in ('source','build','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if os.name!='posix':raise SystemExit('Remote POSIX only')
    source=args.source.resolve(strict=True);build=args.build.resolve(strict=True)
    args.output.mkdir(exist_ok=False)
    manifest=json.loads((source/'source_manifest.json').read_text())
    def observe():
        current={name:digest(source/name) for name in manifest}
        if current!=manifest:raise ValueError('source differs from uploaded immutable manifest')
        return current
    before=observe()
    command=['cmake','--build',str(build),'--target','production_runner',
             'cuda_visible_device_probe','stream4_bounded_graph_cuda_tests','-j8']
    with (args.output/'build.log').open('xb') as log:
        subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
    if observe()!=before:raise ValueError('source changed during build')
    compiler=Path('/usr/local/cuda/bin/nvcc').resolve(strict=True)
    version=subprocess.check_output([str(compiler),'--version'],text=True,timeout=10)
    hardware_raw=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total',
        '--format=csv,noheader,nounits'],text=True,timeout=15)
    hardware=[dict(name=row[0].strip(),uuid=row[1].strip(),memory_mib=int(row[2].strip()))
              for row in csv.reader(hardware_raw.splitlines())]
    if not hardware or any('RTX 3060' not in row['name'] or row['memory_mib']!=12288 for row in hardware):
        raise ValueError('requested RTX3060 12GiB hardware mismatch')
    cache=build/'CMakeCache.txt'
    values={line.split('=',1)[0]:line.split('=',1)[1] for line in cache.read_text().splitlines()
            if '=' in line and not line.startswith(('#','//'))}
    for key,value in {'BEAM_STATE_LOGICAL_BYTES:STRING':'96','BEAM_MOVE_COUNT:STRING':'24',
                      'BEAM_CUDA_ARCHITECTURES:STRING':'86'}.items():
        if values.get(key)!=value:raise ValueError('wrong Cube4/SM86 build: '+key)
    if Path(values['CMAKE_HOME_DIRECTORY:INTERNAL']).resolve(strict=True)!=source:
        raise ValueError('wrong CMake source root')
    cutlass=Path(values['CUTLASS_DIR:PATH']).resolve(strict=True)
    host_compiler=Path(values['CMAKE_CXX_COMPILER:FILEPATH']).resolve(strict=True)
    host_version=subprocess.check_output([str(host_compiler),'--version'],text=True,timeout=10)
    cutlass_commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True,timeout=10).strip()
    cutlass_changes=subprocess.check_output(['git','-C',str(cutlass),'status','--porcelain'],text=True,timeout=20)
    if cutlass_changes:raise ValueError('CUTLASS checkout has unrecorded changes')
    cutlass_hashes={str(path.relative_to(cutlass)):digest(path)
        for directory in ('include','tools/util/include') for path in sorted((cutlass/directory).rglob('*')) if path.is_file()}
    if not cutlass_hashes:raise ValueError('missing CUTLASS headers')
    runtime_library=observe_nccl(build/'production_runner')
    nccl_header=Path(values['NCCL_INCLUDE_DIR:PATH'])/'nccl.h'
    if Path(runtime_library['loaded_library']['nccl_library_path']).resolve(strict=True)!=Path(values['NCCL_LIBRARY:FILEPATH']).resolve(strict=True):
        raise ValueError('loaded NCCL differs from configured build library')
    shutil.copyfile(cache,args.output/'CMakeCache.snapshot.txt')
    shutil.copyfile(build/'.ninja_deps',args.output/'ninja-deps.snapshot.bin')
    result=dict(schema_version=1,status='pass',scope='observed_build_transaction_not_cryptographic_compiler_attestation',
        source_manifest_sha256=digest(source/'source_manifest.json'),source_hashes=before,
        command=command,build_log_sha256=digest(args.output/'build.log'),
        cmake_cache_sha256=digest(cache),ninja_deps_sha256=digest(build/'.ninja_deps'),
        compiler_path=str(compiler),compiler_sha256=digest(compiler),compiler_version=version,hardware=hardware,
        host_compiler_path=str(host_compiler),host_compiler_sha256=digest(host_compiler),host_compiler_version=host_version,
        runtime_library=runtime_library,nccl_header_path=str(nccl_header),nccl_header_sha256=digest(nccl_header),
        cutlass_path=str(cutlass),cutlass_commit=cutlass_commit,cutlass_header_hashes=cutlass_hashes,
        runner_sha256=digest(build/'production_runner'),
        device_probe_sha256=digest(build/'cuda_visible_device_probe'),
        bounded_sort_test_sha256=digest(build/'stream4_bounded_graph_cuda_tests'))
    with (args.output/'receipt.json').open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('source_hashes','cutlass_header_hashes')}))

if __name__=='__main__':main()
