"""Build and verify the native C ABI library for an existing shape-specific build.

The library is inspected only in a short-lived Rust worker process. Importing
this module or resolving its path never loads CUDA into the Python process.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from .backend import child_environment, run_process
from .build import (_validate_index_sort, configure_command, discover_nccl, file_sha256, prerequisites, source_digest,
                    toolchain_identity, validate_runner)
from .errors import NativeBackendError, NativeUnavailable
from .rust_control import worker_executable


def _check_inputs(options, metadata, programs, worker):
    if options.source_dir is None or options.cutlass_dir is None:
        raise NativeUnavailable('native library build requires matching source and CUTLASS directories')
    if source_digest(options.source_dir) != metadata.get('source_digest'):
        raise NativeBackendError('native source changed since the runner was built')
    if source_digest(options.cutlass_dir, cutlass=True) != metadata.get('cutlass_digest'):
        raise NativeBackendError('CUTLASS source changed since the runner was built')
    if toolchain_identity(programs) != metadata.get('toolchain'):
        raise NativeBackendError('native toolchain changed since the runner was built')
    if discover_nccl() != metadata.get('nccl'):
        raise NativeBackendError('NCCL changed since the runner was built')
    if metadata.get('torch') is not None:
        import torch
        observed = dict(version=torch.__version__, cuda=torch.version.cuda,
                        cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                        cmake_prefix=str(torch.utils.cmake_prefix_path))
        if observed != metadata['torch']:
            raise NativeBackendError('PyTorch changed since the runner was built')
    if not worker.is_file() or not os.access(worker, os.X_OK):
        raise NativeUnavailable('matching Rust worker is missing or not executable')


def _inspect(worker, library, shape, run_dir):
    env = child_environment((), run_dir)
    env['BEAM_NATIVE_LIBRARY'] = str(library)
    try:
        result = subprocess.run([str(worker), '--native-abi-info'], cwd=run_dir, env=env,
                                capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise NativeBackendError(f'cannot inspect native ABI in worker: {error}') from error
    if result.returncode:
        raise NativeBackendError(f'native ABI worker failed rc={result.returncode}: {result.stderr[-4000:]}')
    try:
        observed = json.loads(result.stdout)
    except (TypeError, ValueError) as error:
        raise NativeBackendError('native ABI worker returned invalid JSON') from error
    expected = dict(abi_version=1, state_len=shape['state_len'], storage_len=shape['storage_len'],
                    state_alignment=shape['alignment'], move_count=shape['move_count'],
                    candidate_bytes=32, candidate_alignment=32, hash_bytes=16, hash_alignment=16)
    if observed != expected:
        raise NativeBackendError(f'native ABI layout mismatch: required={expected} available={observed}')
    return observed


def ensure_native_library(contract, model, options, architectures, run_dir, runner, runner_metadata):
    """Return a verified shared library for the already verified runner build.

    This explicit preparatory operation does not switch the public search path.
    The returned library is loaded later by a dedicated rank process.
    """
    if sys.platform != 'linux':
        raise NativeUnavailable('native CUDA shared-library builds require Linux')
    if options.runner_path is not None:
        raise NativeUnavailable('native shared-library build requires a managed source build, not an external runner')
    runner, run_dir = Path(runner).resolve(), Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = validate_runner(runner, contract, model.backend, tuple(architectures))
    _validate_index_sort(metadata, options.experimental_index_sort)
    resolved_backend = metadata.get('inference_backend')
    if (metadata != runner_metadata or resolved_backend not in ('cutlass', 'libtorch')
            or options.inference_backend not in ('auto', resolved_backend)):
        raise NativeBackendError('runner build metadata does not match the native library request')
    needs_torch = (resolved_backend == 'libtorch' or model.backend in ('piece_transformer', 'ensemble')
                   or bool(metadata.get('calibration_binary_name')))
    if needs_torch and not isinstance(metadata.get('torch'), dict):
        raise NativeBackendError('PyTorch build identity is required for this runner')
    build_dir = runner.parent
    if not (build_dir / 'CMakeCache.txt').is_file():
        raise NativeUnavailable('native runner has no managed CMake build directory')
    programs = prerequisites(options)
    worker = worker_executable().resolve()
    _check_inputs(options, metadata, programs, worker)
    target = 'beam_native_libtorch' if resolved_backend == 'libtorch' or model.backend != 'mlp' else 'beam_native'
    library = build_dir / f'lib{target}.so'
    sidecar = build_dir / 'native-library-build.json'
    specification = dict(abi_version=1, target=target, runner_build_key=metadata.get('build_key'),
                         runner_sha256=metadata['binary_sha256'], worker_sha256=file_sha256(worker),
                         shape=metadata['shape'], backend=model.backend,
                         inference_backend=resolved_backend,
                         cuda_architectures=metadata['cuda_architectures'], source_digest=metadata['source_digest'],
                         cutlass_digest=metadata['cutlass_digest'], toolchain=metadata['toolchain'], nccl=metadata['nccl'],
                         torch=metadata.get('torch'))
    if options.experimental_index_sort:
        specification['experimental_index_sort'] = True

    def validated():
        try:
            receipt = json.loads(sidecar.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise NativeBackendError(f'native library metadata is missing or invalid: {error}') from error
        if not isinstance(receipt, dict) or receipt.get('schema_version') != 1 or receipt.get('specification') != specification:
            raise NativeBackendError('native library metadata does not match the requested build')
        if not library.is_file() or receipt.get('library_sha256') != file_sha256(library):
            raise NativeBackendError('native library SHA256 differs from its build metadata')
        if _inspect(worker, library, metadata['shape'], run_dir) != receipt.get('layout'):
            raise NativeBackendError('native library layout differs from its build metadata')
        if file_sha256(worker) != specification['worker_sha256'] or file_sha256(library) != receipt['library_sha256']:
            raise NativeBackendError('native library or worker changed during ABI inspection')
        return library, receipt

    if sidecar.exists():
        return validated()
    lock = build_dir / '.native-library.lock'
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise NativeBackendError(f'native library build is already active or has a retained lock: {lock}') from error
    try:
        if sidecar.exists():
            return validated()
        env = child_environment((), run_dir)
        env['CXX'], env['CUDACXX'] = programs['cxx'], programs['nvcc']
        env['TORCH_CUDA_ARCH_LIST'] = ';'.join(f'{sm // 10}.{sm % 10}' for sm in metadata['cuda_architectures'])
        run_process(configure_command(options.source_dir, build_dir, options.cutlass_dir,
                                      contract, model.backend, tuple(metadata['cuda_architectures']), programs,
                                      nccl=metadata['nccl'], inference_backend=resolved_backend,
                                      calibration=bool(metadata.get('calibration_binary_name')),
                                      experimental_index_sort=options.experimental_index_sort),
                    cwd=run_dir, env=env, timeout=options.build_timeout_seconds,
                    log_path=run_dir / 'native-library-configure.log')
        run_process([programs['cmake'], '--build', str(build_dir), '--target', target,
                     '--parallel', str(options.build_jobs)], cwd=run_dir, env=env,
                    timeout=options.build_timeout_seconds, log_path=run_dir / 'native-library-build.log')
        _check_inputs(options, metadata, programs, worker)
        if file_sha256(worker) != specification['worker_sha256'] or file_sha256(runner) != specification['runner_sha256']:
            raise NativeBackendError('worker or runner changed during native library build')
        if not library.is_file():
            raise NativeBackendError('native build did not produce the requested shared library')
        layout = _inspect(worker, library, metadata['shape'], run_dir)
        receipt = dict(schema_version=1, specification=specification, library_sha256=file_sha256(library), layout=layout)
        temporary = build_dir / 'native-library-build.json.tmp'
        temporary.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        temporary.replace(sidecar)
        return validated()
    finally:
        lock.rmdir()
