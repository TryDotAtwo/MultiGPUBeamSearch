from pathlib import Path
from types import SimpleNamespace
import json
from multigpubeamsearch.native_library import ensure_native_library
from multigpubeamsearch.build import file_sha256
from multigpubeamsearch.options import NativeOptions
from multigpubeamsearch.rust_control import worker_executable

base=Path('/workspace/beam-results/native-library');auto_base=Path('/workspace/beam-results/native-auto')
worker=worker_executable()
assert 'native-torch-identity/distribution/check/checkout/installed' in str(worker)
out=Path('/workspace/beam-results/native-torch-identity/wheel')
out.mkdir(exist_ok=False)
contract=SimpleNamespace(state_len=96,move_count=24);model=SimpleNamespace(backend='mlp')
rows={}
for kind,result,source,cache in (
    ('cutlass',base/'reconfigure-real/result.json',base/'distribution3/check/checkout/installed/multigpubeamsearch/_native_source',base/'actual/build/cache'),
    ('libtorch',auto_base/'libtorch/result.json',base/'distribution4/check/checkout/installed/multigpubeamsearch/_native_source',auto_base/'libtorch/cache'),
):
    old=json.loads(result.read_text());runner=Path(old['runner'])
    metadata=json.loads((runner.parent/'native-build.json').read_text())
    options=NativeOptions(source_dir=source,cutlass_dir=Path('/workspace/cutlass'),cache_dir=cache,
                          inference_backend='auto',autotune=False,build_jobs=2,build_timeout_seconds=1200)
    directory=out/kind;directory.mkdir()
    library,receipt=ensure_native_library(contract,model,options,(86,),directory,runner,metadata)
    assert receipt['specification']['inference_backend']==kind
    assert (metadata.get('torch') is not None)==(kind=='libtorch')
    assert receipt['library_sha256']==file_sha256(library)
    assert ensure_native_library(contract,model,options,(86,),directory,runner,metadata)==(library,receipt)
    rows[kind]=dict(runner=str(runner),library=str(library),worker=str(worker),receipt=receipt,
                    runner_sha256=file_sha256(runner),library_sha256=file_sha256(library),worker_sha256=file_sha256(worker))
assert 'libbeam_native.so' not in Path('/proc/self/maps').read_text()
assert 'libbeam_native_libtorch.so' not in Path('/proc/self/maps').read_text()
(out/'result.json').write_text(json.dumps(rows,indent=2)+'\n')
print('NATIVE_TORCH_IDENTITY_INSTALLED_WHEEL_OK',flush=True)
