from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
from multigpubeamsearch import native_library
from multigpubeamsearch.build import ensure_runner, file_sha256
from multigpubeamsearch.options import NativeOptions

base=Path('/workspace/beam-results/native-library')
source=base/'distribution4/check/checkout/installed/multigpubeamsearch/_native_source'
worker=base/'distribution4/check/checkout/installed/multigpubeamsearch/_bin/beam-runner'
native_library.worker_executable=lambda:worker
out=Path('/workspace/beam-results/native-auto/libtorch')
out.mkdir(exist_ok=False)
logs=out/'logs';logs.mkdir()
options=NativeOptions(source_dir=source,cutlass_dir=Path('/workspace/cutlass'),cache_dir=out/'cache',
                      inference_backend='libtorch',autotune=False,build_jobs=2,build_timeout_seconds=1200)
contract=SimpleNamespace(state_len=96,move_count=24)
model=SimpleNamespace(backend='mlp')
runner,metadata=ensure_runner(contract,model,options,(86,),logs)
auto=replace(options,inference_backend='auto')
library,receipt=native_library.ensure_native_library(contract,model,auto,(86,),logs,runner,metadata)
assert receipt['specification']['inference_backend']=='libtorch'
assert library.name=='libbeam_native_libtorch.so' and receipt['library_sha256']==file_sha256(library)
assert native_library.ensure_native_library(contract,model,auto,(86,),logs,runner,metadata)==(library,receipt)
assert 'libbeam_native_libtorch.so' not in Path('/proc/self/maps').read_text()
result=dict(runner=str(runner),library=str(library),worker=str(worker),receipt=receipt,
            runner_sha256=file_sha256(runner),library_sha256=file_sha256(library),worker_sha256=file_sha256(worker))
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print('NATIVE_AUTO_LIBTORCH_BUILD_OK',flush=True)
