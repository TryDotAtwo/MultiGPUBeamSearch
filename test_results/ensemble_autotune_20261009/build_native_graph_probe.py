"""Compile the changed probe outside the source tree of a live public build."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

directory=Path('/workspace/graph-probe');directory.mkdir(exist_ok=True)
build=Path('/workspace/results/public-ensemble-api/cache/builds/3c758e5bfedf49fe190e89ce7e71fe0102b692c03149bf6f425525e56ef9f3a9')
target=build/'CMakeFiles/stream1_native_mlp_benchmark.dir'
fields={}
for line in (target/'flags.make').read_text().splitlines():
    if ' = ' in line:
        key,value=line.split(' = ',1);fields[key]=shlex.split(value)
source=directory/'stream1_native_mlp_benchmark.cpp'
obj=directory/'native-graph.o';binary=directory/'stream1_native_mlp_benchmark'
command=['/usr/bin/x86_64-linux-gnu-g++-13',*fields['CXX_DEFINES'],
    *fields['CXX_INCLUDES'],*fields['CXX_FLAGS'],'-I/workspace/source/tools',
    '-c',str(source),'-o',str(obj)]
subprocess.run(command,check=True,cwd=directory)
link=shlex.split((target/'link.txt').read_text())
object_index=next(i for i,value in enumerate(link) if value.endswith('stream1_native_mlp_benchmark.cpp.o'))
link[object_index]=str(obj);link[link.index('-o')+1]=str(binary)
subprocess.run(link,check=True,cwd=build)
hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (source,binary,build/'libbeam_cuda.a',build/'libbeam_core.a')}
(directory/'build-receipt.json').write_text(json.dumps(dict(
    compile=command,link=link,hashes=hashes),indent=2))
print(json.dumps(dict(binary=str(binary),hashes=hashes)),flush=True)

