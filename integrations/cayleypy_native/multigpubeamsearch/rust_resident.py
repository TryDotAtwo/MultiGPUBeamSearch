"""Bridge a public session to Rust-owned rank processes."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from .errors import NativeBackendError, NativeUnavailable


# The remaining Rust metadata describes its session rather than a terminal
# record; Python assembles that part of the public result itself.
TERMINAL_FIELDS = (
    'native_seconds', 'terminal_records', 'collected_solutions', 'collection_status',
    'observed_parent_batch', 'observed_parent_batch_by_rank_values',
    'observed_inference_rows', 'observed_inference_rows_by_rank_values',
    'observed_stream3_batch_candidates', 'observed_stream3_batch_candidates_by_rank_values',
)


class RustResidentWorkers:
    """Submit whole searches; CUDA and the rank group live outside Python."""

    def __init__(self, command, *, contract, model, options, max_steps, devices,
                 runtime, run_dir, env, world):
        if runtime is None or len(command) != 4 or world != len(devices):
            raise NativeBackendError("Rust resident launch has incomplete prepared inputs")
        from .native_library import ensure_native_library
        from .rust_control import extension, options_descriptor, worker_executable

        module = extension()
        if (not hasattr(module, 'SearchSession')
                or getattr(module, 'SEARCH_TERMINAL_RESULT_VERSION', None) != 1):
            raise NativeUnavailable('the matching Rust runtime lacks terminal-result v1; rebuild the PyO3 module')
        library, receipt = ensure_native_library(contract, model, options, runtime.architectures,
            run_dir, runtime.runner, runtime.build_metadata)
        worker = worker_executable()
        self._owner = module.SearchSession()
        self.sequence = 0
        self._terminal = None
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
        self._terminal = None
        try:
            result = json.loads(self._owner.run(self._identity, self._launch, list(start),
                                                Path(directory), timeout, '{}'))
            metadata = result['metadata']
            sequence = metadata['persistent_session']['request_count']
            elapsed = result['elapsed_seconds']
            native_seconds = metadata['native_seconds']
            records = metadata['terminal_records']
            width = result['effective_beam_width']
            path = result['path']
            valid_time = lambda n: type(n) in (int, float) and math.isfinite(n) and n >= 0
            if (type(metadata) is not dict or type(sequence) is not int
                    or sequence != self.sequence + 1 or not valid_time(elapsed)
                    or not valid_time(native_seconds) or type(records) is not int or records <= 0
                    or type(width) is not int or width <= 0
                    or (path is not None and (type(path) is not list
                        or any(type(move) is not int or not 0 <= move <= 0xffffffff for move in path)))):
                raise ValueError('invalid terminal result or request sequence')
            terminal = {name: metadata[name] for name in TERMINAL_FIELDS if name in metadata}
            self._terminal = (Path(directory).resolve(),
                              (None if path is None else tuple(path), width, terminal))
            self.sequence = sequence
            return elapsed
        except (RuntimeError, ValueError, KeyError, TypeError, OverflowError) as error:
            raise NativeBackendError(f'Rust resident search failed: {error}') from error

    def take_terminal(self, directory):
        if self._terminal is None:
            raise NativeBackendError('Rust resident result is unavailable or already consumed')
        expected, terminal = self._terminal
        if Path(directory).resolve() != expected:
            raise NativeBackendError('Rust resident result belongs to a different directory')
        self._terminal = None
        return terminal

    def close(self):
        self._terminal = None
        self._owner.close()
