"""Experimental bridge from the public session to Rust-owned rank processes."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .errors import NativeBackendError, NativeUnavailable


class RustResidentWorkers:
    """Submit whole searches; CUDA and the rank group live outside Python."""

    def __init__(self, command, *, contract, model, options, max_steps, devices,
                 runtime, run_dir, env, world):
        if runtime is None or len(command) != 4 or world != len(devices):
            raise NativeBackendError("Rust resident launch has incomplete prepared inputs")
        from .native_library import ensure_native_library
        from .rust_control import extension, options_descriptor, worker_executable

        library, receipt = ensure_native_library(contract, model, options, runtime.architectures,
            run_dir, runtime.runner, runtime.build_metadata)
        worker = worker_executable()
        module = extension()
        if not hasattr(module, 'SearchSession'):
            raise NativeUnavailable('the matching Rust runtime lacks SearchSession')
        self._owner = module.SearchSession()
        self.sequence = 0
        build = dict(runtime.build_metadata, native_library_sha256=receipt['library_sha256'],
                     rust_worker_sha256=receipt['specification']['worker_sha256'], abi_version=1)
        identity = dict(graph=dict(center=contract.center, generators=contract.generators),
            graph_hash=contract.graph_hash, model_artifact_hash=model.artifact_hash,
            model_backend=model.backend, devices=list(devices), beam_width=int(command[3]),
            max_steps=max_steps, runtime_build=build, options=options_descriptor(options),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            cuda_device_order=os.environ.get('CUDA_DEVICE_ORDER'))
        launch = dict(command=[str(worker), *map(str, command[1:])],
            protocol_command=[str(worker), '--resident-protocol-info'],
            environment=dict(env, BEAM_NATIVE_LIBRARY=str(library)), cwd=str(Path(run_dir).resolve()))
        self._identity = json.dumps(identity)
        self._launch = json.dumps(launch)

    @property
    def pids(self):
        return self._owner.pids

    def submit(self, start, directory, *, timeout):
        try:
            result = json.loads(self._owner.run(self._identity, self._launch, list(start),
                                                Path(directory), timeout, '{}'))
            self.sequence = result['metadata']['persistent_session']['request_count']
            return result['elapsed_seconds']
        except (RuntimeError, ValueError, KeyError, TypeError) as error:
            raise NativeBackendError(f'Rust resident search failed: {error}') from error

    def close(self):
        self._owner.close()
