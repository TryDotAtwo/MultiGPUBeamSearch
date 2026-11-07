"""Build, validate and reuse a real shape-specific native library on a GPU host."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

from multigpubeamsearch.build import ensure_runner, file_sha256
from multigpubeamsearch.native_library import ensure_native_library
from multigpubeamsearch.options import NativeOptions
from multigpubeamsearch.rust_control import worker_executable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cutlass', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    logs = out / 'logs'
    logs.mkdir()
    contract = SimpleNamespace(state_len=96, move_count=24)
    model = SimpleNamespace(backend='mlp')
    options = NativeOptions(source_dir=args.source, cutlass_dir=args.cutlass, cache_dir=out / 'cache',
                            inference_backend='cutlass', autotune=False, build_jobs=2,
                            build_timeout_seconds=1200)
    maps = Path('/proc/self/maps')
    if maps.exists():
        assert 'libbeam_native.so' not in maps.read_text()
    began = time.monotonic()
    runner, metadata = ensure_runner(contract, model, options, (86,), logs)
    library, receipt = ensure_native_library(contract, model, options, (86,), logs, runner, metadata)
    cold_seconds = time.monotonic() - began
    began = time.monotonic()
    cached_runner, cached_metadata = ensure_runner(contract, model, options, (86,), logs)
    cached_library, cached_receipt = ensure_native_library(
        contract, model, options, (86,), logs, cached_runner, cached_metadata)
    cached_seconds = time.monotonic() - began
    assert (cached_runner, cached_metadata, cached_library, cached_receipt) == (runner, metadata, library, receipt)
    assert receipt['library_sha256'] == file_sha256(library)
    if maps.exists():
        assert 'libbeam_native.so' not in maps.read_text()
    result = dict(runner=str(runner), runner_sha256=file_sha256(runner), library=str(library),
                  library_sha256=file_sha256(library), worker=str(worker_executable()),
                  worker_sha256=file_sha256(worker_executable()), receipt=receipt,
                  cold_seconds=cold_seconds, cached_seconds=cached_seconds,
                  scope='build and cache lifetime; not a search-step benchmark')
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('NATIVE_LIBRARY_GPU_BUILD_OK', json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
