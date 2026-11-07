from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import time

from multigpubeamsearch.build import file_sha256
from multigpubeamsearch.native_library import ensure_native_library
from multigpubeamsearch.options import NativeOptions
from multigpubeamsearch.rust_control import worker_executable

base = Path('/workspace/beam-results/native-library')
out = base / 'reconfigure-real'
out.mkdir(exist_ok=False)
old = json.loads((base / 'actual/build/result.json').read_text())
runner = Path(old['runner'])
build_dir = runner.parent
source = base / 'distribution3/check/checkout/installed/multigpubeamsearch/_native_source'
metadata = json.loads((build_dir / 'native-build.json').read_text())
sidecar = build_dir / 'native-library-build.json'
sidecar.replace(out / 'previous-receipt.json')
with (out / 'poison-configure.log').open('w') as log:
    subprocess.run(['cmake', '-S', str(source), '-B', str(build_dir),
                    '-DBEAM_EXPERIMENTAL_INDEX_SORT=ON'], check=True, stdout=log, stderr=subprocess.STDOUT)
cache = build_dir / 'CMakeCache.txt'
assert 'BEAM_EXPERIMENTAL_INDEX_SORT:BOOL=ON' in cache.read_text()
contract = SimpleNamespace(state_len=96, move_count=24)
model = SimpleNamespace(backend='mlp')
options = NativeOptions(source_dir=source, cutlass_dir=Path('/workspace/cutlass'),
                        cache_dir=base / 'actual/build/cache', inference_backend='cutlass',
                        autotune=False, build_jobs=2, build_timeout_seconds=1200)
assert 'libbeam_native.so' not in Path('/proc/self/maps').read_text()
start = time.monotonic()
library, receipt = ensure_native_library(contract, model, options, (86,), out, runner, metadata)
elapsed = time.monotonic() - start
assert 'BEAM_EXPERIMENTAL_INDEX_SORT:BOOL=OFF' in cache.read_text()
assert (out / 'native-library-configure.log').is_file()
assert (out / 'native-library-build.log').is_file()
assert ensure_native_library(contract, model, options, (86,), out, runner, metadata) == (library, receipt)
assert 'libbeam_native.so' not in Path('/proc/self/maps').read_text()
result = dict(runner=str(runner), runner_sha256=file_sha256(runner), library=str(library),
              library_sha256=file_sha256(library), worker=str(worker_executable()),
              worker_sha256=file_sha256(worker_executable()), receipt=receipt,
              reconfigure_seconds=elapsed, cache_flag_before='ON', cache_flag_after='OFF')
(out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print('NATIVE_LIBRARY_RECONFIGURE_OK', json.dumps(result, sort_keys=True), flush=True)
