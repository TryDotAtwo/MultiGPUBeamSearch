from pathlib import Path
from types import SimpleNamespace
import json
from multigpubeamsearch import native_library
from multigpubeamsearch.build import file_sha256
from multigpubeamsearch.options import NativeOptions

base=Path('/workspace/beam-results/native-library')
old=json.loads((base/'reconfigure-real/result.json').read_text())
runner=Path(old['runner'])
metadata=json.loads((runner.parent/'native-build.json').read_text())
source=base/'distribution3/check/checkout/installed/multigpubeamsearch/_native_source'
worker=base/'distribution4/check/checkout/installed/multigpubeamsearch/_bin/beam-runner'
native_library.worker_executable=lambda:worker
out=Path('/workspace/beam-results/native-auto/real')
out.mkdir(exist_ok=False)
options=NativeOptions(source_dir=source,cutlass_dir=Path('/workspace/cutlass'),
                      cache_dir=base/'actual/build/cache',inference_backend='auto',autotune=False,
                      build_jobs=2,build_timeout_seconds=1200)
contract=SimpleNamespace(state_len=96,move_count=24)
model=SimpleNamespace(backend='mlp')
library,receipt=native_library.ensure_native_library(contract,model,options,(86,),out,runner,metadata)
assert receipt['specification']['inference_backend']=='cutlass'
assert receipt['library_sha256']==file_sha256(library)
assert native_library.ensure_native_library(contract,model,options,(86,),out,runner,metadata)==(library,receipt)
assert 'libbeam_native.so' not in Path('/proc/self/maps').read_text()
result=dict(runner=str(runner),library=str(library),worker=str(worker),receipt=receipt,
            runner_sha256=file_sha256(runner),library_sha256=file_sha256(library),worker_sha256=file_sha256(worker))
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print('NATIVE_AUTO_REAL_CACHE_OK',flush=True)
