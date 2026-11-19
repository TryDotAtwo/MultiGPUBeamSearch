"""Explicit lifetime for sequential searches on one native GPU worker group."""
from contextvars import ContextVar
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import threading

from .errors import NativeBackendError, NativeUnavailable
from .model_snapshot import requires_mlp_host_binding
from .options import NativeOutcome

_ACTIVE = ContextVar('multigpubeamsearch_native_session', default=None)


def current_session():
    return _ACTIVE.get()


def _rust_resident_available():
    """Check for the complete Rust session surface without loading CUDA."""
    from .rust_control import extension, worker_executable

    try:
        module = extension()
        if (not hasattr(module, 'SearchSession')
                or getattr(module, 'SEARCH_TERMINAL_RESULT_VERSION', None) != 1):
            return False
        worker_executable()
        return True
    except NativeUnavailable:
        return False


class NativeSession:
    """Keep compatible native workers alive inside ``with NativeSession():``.

    Graph/model/runner/topology/beam/depth/options changes release the old
    worker group before recalibration. Concurrent or nested use is rejected.
    The default selects the Rust owner when its matching runtime and worker
    are installed; otherwise it retains the Python owner. Pass
    ``engine="python"`` to force the compatibility owner or
    ``engine="rust"`` to require Rust.
    Exiting the context, including on exceptions, frees all owned workers.
    The experimental NCCL identity option retains block digests per session;
    every repeated check rereads the complete library.
    Experimental model-snapshot reuse covers explicit MLP NativeModel inputs
    in strict native mode; source and retained bytes are revalidated on reuse.
    Experimental Rust MLP finite validation is separately opt-in and only
    applies to the Rust-owned model snapshot path.
    """
    def __init__(self, *, engine="auto", experimental_parallel_nccl_identity=False,
                 experimental_reuse_model_snapshot=False, experimental_rust_mlp_finite=False):
        if engine not in ("auto", "python", "rust"):
            raise ValueError("engine must be 'auto', 'python', or 'rust'")
        if engine == 'auto':
            engine = 'rust' if _rust_resident_available() else 'python'
        if type(experimental_parallel_nccl_identity) is not bool:
            raise ValueError("experimental_parallel_nccl_identity must be a boolean")
        if type(experimental_reuse_model_snapshot) is not bool:
            raise ValueError("experimental_reuse_model_snapshot must be a boolean")
        if type(experimental_rust_mlp_finite) is not bool:
            raise ValueError("experimental_rust_mlp_finite must be a boolean")
        if experimental_rust_mlp_finite and (engine != 'rust' or not experimental_reuse_model_snapshot):
            raise ValueError("experimental_rust_mlp_finite requires the Rust snapshot session")
        self.nccl_digest_cache = None
        if experimental_parallel_nccl_identity:
            from .nccl_digest import ChunkDigestCache
            self.nccl_digest_cache = ChunkDigestCache()
        self.model_snapshot_cache = None
        if experimental_reuse_model_snapshot:
            from .model_snapshot import ModelSnapshotCache
            self.model_snapshot_cache = ModelSnapshotCache(prefer_rust=engine == 'rust',
                                                          enable_rust_finite=experimental_rust_mlp_finite)
        self._owner_pid = os.getpid()
        self._engine = engine
        self._pending = None
        self.closed = False
        self._workers = None
        self._identity = None
        self._metadata = None
        self._rust_metadata_owned = False
        self._token = None
        self._lock = threading.Lock()
        self._operation_lock = threading.Lock()
        self._operation_owner = None
        self._run_owner = None

    def _check_owner(self):
        if os.getpid()!=self._owner_pid:
            raise RuntimeError('native session cannot be shared across Python processes')

    def _check_operation_owner(self):
        # A contender may have acquired _operation_lock but still be unable
        # to reserve the session because an existing run owns _lock. Only a
        # published owner may reject that existing run's internal launch.
        if self._operation_owner is not None and self._operation_owner != threading.get_ident():
            raise RuntimeError('concurrent native session searches are unsupported')

    def __enter__(self):
        self._check_owner()
        if self.closed:raise RuntimeError('native session is closed')
        if current_session() is not None:raise RuntimeError('nested native sessions are unsupported')
        if self._token is not None:raise RuntimeError('native session context is already active')
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, *exc):
        try:self.close()
        finally:
            _ACTIVE.reset(self._token)
            self._token = None

    @contextmanager
    def activate(self):
        """Route CayleyPy graph calls through this session without closing it.

        Suitable for separate notebook cells. Explicit close remains required;
        ``with NativeSession()`` still owns and closes the complete lifetime.
        """
        self._check_owner()
        if self.closed:raise RuntimeError('native session is closed')
        active=current_session()
        if active is not None and active is not self:
            raise RuntimeError('nested native sessions are unsupported')
        token=_ACTIVE.set(self)
        try:yield self
        finally:_ACTIVE.reset(token)

    def beam_search(self, graph, **kwargs):
        """Explicit session call, also usable across separate notebook cells."""
        from .dispatch import beam_search
        with self.activate():
            return beam_search(graph,**kwargs)

    @contextmanager
    def search_scope(self):
        """Reserve the session from public dispatch through result assembly."""
        self._check_owner()
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError('concurrent native session searches are unsupported')
        try:
            if not self._lock.acquire(blocking=False):
                raise RuntimeError('concurrent native session searches are unsupported')
            try:
                if self.closed:raise RuntimeError('native session is closed')
                self._operation_owner = threading.get_ident()
            finally:self._lock.release()
            try:yield
            finally:self._operation_owner = None
        finally:self._operation_lock.release()

    @contextmanager
    def prepare_nccl_identity(self):
        """Hold session ownership while a cache-backed identity is prepared."""
        self._check_owner()
        if self.closed:raise RuntimeError('native session is closed')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('concurrent native session searches are unsupported')
        try:
            if self.closed:raise RuntimeError('native session is closed')
            self._check_operation_owner()
            yield self.nccl_digest_cache
        finally:self._lock.release()

    @contextmanager
    def prepare_model_snapshot(self):
        """Hold session ownership while validating or storing its snapshot."""
        self._check_owner()
        if self.closed:raise RuntimeError('native session is closed')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('concurrent native session searches are unsupported')
        try:
            if self.closed:raise RuntimeError('native session is closed')
            self._check_operation_owner()
            yield self.model_snapshot_cache
        finally:self._lock.release()

    def _release(self):
        if self._workers is not None:self._workers.close()
        self._workers = None
        self._identity = None
        self._metadata = None
        self._rust_metadata_owned = False

    def close(self):
        self._check_owner()
        if self.closed:return
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError('cannot close a native session during a search')
        try:
            if not self._lock.acquire(blocking=False):
                raise RuntimeError('cannot close a native session during a search')
            try:
                self._release()
                if self.nccl_digest_cache is not None:self.nccl_digest_cache.clear()
                if self.model_snapshot_cache is not None:self.model_snapshot_cache.clear()
                self.closed = True
            finally:self._lock.release()
        finally:self._operation_lock.release()

    def launch(self, command, *, contract, run_dir, env, world, timeout, runtime=None):
        self._check_owner()
        if self._run_owner == threading.get_ident():
            if self.closed:raise RuntimeError('native session is closed')
            self._check_operation_owner()
            return self._launch(command, contract=contract, run_dir=run_dir, env=env,
                                world=world, timeout=timeout, runtime=runtime)
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError('concurrent native session searches are unsupported')
        try:
            if not self._lock.acquire(blocking=False):
                raise RuntimeError('concurrent native session searches are unsupported')
            try:
                if self.closed:raise RuntimeError('native session is closed')
                previous_workers = self._workers
                try:
                    return self._launch(command, contract=contract, run_dir=run_dir, env=env,
                                        world=world, timeout=timeout, runtime=runtime)
                except BaseException as error:
                    if previous_workers is None and self._workers is not None:
                        try:self._release()
                        except Exception as cleanup_error:
                            if hasattr(error, 'add_note'):
                                error.add_note(f'worker cleanup also failed: {cleanup_error}')
                    raise
            finally:self._lock.release()
        finally:self._operation_lock.release()

    def _launch(self, command, *, contract, run_dir, env, world, timeout, runtime=None):
        from .resident_transport import ResidentWorkers
        import subprocess
        if self._engine == 'rust':
            if self._pending is None:
                raise NativeBackendError('Rust resident launch requires a prepared session')
            from .rust_resident import RustResidentWorkers
            prepared = self._pending
            model = prepared['model']
            options = prepared['options']
            launch_runtime = runtime or prepared['runtime']
            launch_inputs = dict(contract=contract, model=model, options=options,
                max_steps=prepared['max_steps'], devices=prepared['devices'],
                runtime=launch_runtime, run_dir=run_dir, env=env, world=world,
                session_identity=prepared['session_identity'])
            if self._workers is None:
                self._workers = RustResidentWorkers(command, **launch_inputs)
            else:
                configure = getattr(self._workers, 'configure', None)
                if not callable(configure):
                    raise NativeBackendError('resident workers cannot be reconfigured')
                configure(command, **launch_inputs)
            elapsed = self._workers.submit(contract.start, run_dir, timeout=timeout)
            prepared['launch_started'] = True
            return elapsed
        if self._workers is not None:raise NativeBackendError('resident workers already exist')
        probe=subprocess.run([command[0],'--resident-protocol-info'],env=env,cwd=run_dir,
            capture_output=True,text=True,timeout=min(timeout,15),check=False)
        if probe.returncode:
            detail=(probe.stderr or probe.stdout).strip()[-4000:]
            raise NativeBackendError(
                f'resident runner probe failed (exit {probe.returncode}): {detail}')
        if probe.stdout.strip()!='{"protocol":"resident-search-v1"}':
            raise NativeBackendError('runner lacks resident-search-v1; rebuild native sources')
        self._workers = ResidentWorkers(command, world=world,
            directory=Path(run_dir)/'resident-session',cwd=run_dir,env=env)
        return self._workers.submit(contract.start, run_dir, timeout=timeout)

    def terminal_result(self, log_path, contract, *, max_steps=None):
        if self._engine == 'rust':
            if self._workers is None:
                raise NativeBackendError('Rust resident workers are unavailable')
            return self._workers.take_terminal(Path(log_path).parent)
        from .backend import parse_terminal
        return parse_terminal(Path(log_path), contract, max_steps=max_steps)

    def run(self, execute, contract, model, options, beam_width, max_steps, run_dir, devices, *, runtime=None):
        from .models import verify_prepared_model
        self._check_owner()
        if self.closed:raise RuntimeError('native session is closed')
        if not self._lock.acquire(blocking=False):raise RuntimeError('concurrent native session searches are unsupported')
        if self.closed:
            self._lock.release()
            raise RuntimeError('native session is closed')
        try:
            self._check_operation_owner()
        except BaseException:
            self._lock.release()
            raise
        self._run_owner = threading.get_ident()
        try:
            # Runtime preparation already verifies executable bytes. Start state
            # is intentionally absent; all other search geometry stays fixed.
            binding_scope = requires_mlp_host_binding(self.model_snapshot_cache, model, contract,
                {} if runtime is None else runtime.build_metadata)
            geometry=tuple(getattr(contract,name,None) for name in ('state_len','num_classes','move_count'))
            center=getattr(contract,'center',None)
            generators=getattr(contract,'generators',None)
            graph_content=(None if center is None else tuple(center),
                           None if generators is None else tuple(tuple(move) for move in generators))
            artifact_sizes=getattr(model,'artifact_sizes',())
            device_order=tuple(devices)
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')
            cuda_device_order=os.environ.get('CUDA_DEVICE_ORDER')
            options_repr=repr(options)
            runtime_build=runtime.build_metadata if runtime is not None else None
            runtime_build_key=json.dumps(runtime_build,sort_keys=True) if runtime is not None else None
            identity=(contract.graph_hash,geometry,graph_content,model.artifact_hash,artifact_sizes,
                model.backend,device_order,binding_scope,
                beam_width,max_steps,options_repr,visible_devices,cuda_device_order,
                runtime_build_key)
            preparation_identity = {
                'graph_hash': contract.graph_hash,
                'geometry': [None if value is None else int(value) for value in geometry],
                'center': None if center is None else [int(value) for value in center],
                'generators': None if generators is None else [
                    [int(value) for value in move] for move in generators],
                'model_artifact_hash': model.artifact_hash,
                'artifact_sizes': None if artifact_sizes is None else [
                    int(size) for size in artifact_sizes],
                'model_backend': model.backend,
                'devices': [int(device) for device in device_order],
                'binding_scope': bool(binding_scope),
                'beam_width': int(beam_width),
                'max_steps': int(max_steps),
                'options_repr': options_repr,
                'cuda_visible_devices': visible_devices,
                'cuda_device_order': cuda_device_order,
                'runtime_build': runtime_build,
            }
            identity_key = json.dumps(preparation_identity, sort_keys=True)
            if (self.model_snapshot_cache is None
                    or not self.model_snapshot_cache.verify_for_launch(model, contract)):
                verify_prepared_model(model,contract)
            rust_identity_match = None
            if self._engine == 'rust' and self._workers is not None:
                preparer = getattr(self._workers, 'begin_preparation', None)
                if callable(preparer):
                    rust_identity_match = preparer(identity_key)
                if rust_identity_match is None:
                    matcher = getattr(self._workers, 'preparation_matches', None)
                    if callable(matcher):
                        rust_identity_match = matcher(identity_key)
            identity_changed = (identity != self._identity if rust_identity_match is None
                                else not rust_identity_match)
            preserve_rust_owner = (identity_changed and self._engine == 'rust'
                and self._workers is not None
                and callable(getattr(self._workers, 'configure', None)))
            if identity_changed and not preserve_rust_owner:self._release()
            rust_outcome_owned = False
            remember_preparation = False
            if self._workers is None or identity_changed:
                if (self.model_snapshot_cache is not None
                        and self.model_snapshot_cache.needs_fresh_worker_snapshot(model, contract)):
                    model = self.model_snapshot_cache.refresh_for_worker(model, contract, run_dir)
                artifact_sizes = getattr(model, 'artifact_sizes', ())
                session_identity = {
                    'geometry': [getattr(contract, name, None)
                                 for name in ('state_len', 'num_classes', 'move_count')],
                    'artifact_sizes': None if artifact_sizes is None else list(artifact_sizes),
                    'binding_scope': binding_scope,
                    'options_repr': repr(options),
                }
                self._pending = {
                    'model': model,
                    'options': options,
                    'max_steps': max_steps,
                    'devices': devices,
                    'runtime': runtime,
                    'session_identity': session_identity,
                    'launch_started': False,
                }
                try:
                    outcome=execute(contract,model,options,beam_width,max_steps,run_dir,devices,runtime=runtime)
                finally:
                    launch_started = self._pending['launch_started']
                    self._pending=None
                if self._workers is not None:
                    if preserve_rust_owner and not launch_started:
                        self._release()
                    else:
                        remember_preparation = True
                reused=bool(self._workers is not None
                    and getattr(self._workers, '_last_reused', False))
            else:
                directory=Path(run_dir)
                elapsed=self._workers.submit(contract.start,directory,timeout=options.timeout_seconds)
                rust_outcome_owned = self._engine == 'rust' and self._rust_metadata_owned
                if rust_outcome_owned:
                    take_outcome = getattr(self._workers, 'take_outcome', None)
                    if not callable(take_outcome):
                        raise NativeBackendError('Rust-owned result metadata is unavailable')
                    result = take_outcome(directory)
                    outcome = NativeOutcome(result['path'], result['elapsed_seconds'],
                        result['effective_beam_width'], directory, result['metadata'])
                    if outcome.effective_beam_width != beam_width:
                        raise NativeBackendError('resident effective beam changed')
                else:
                    path,effective,terminal=self.terminal_result(directory/'native.log',contract,max_steps=max_steps)
                    if effective != self._metadata['requested_beam_width']:
                        raise NativeBackendError('resident effective beam changed')
                    metadata=deepcopy(self._metadata)
                    from .rust_resident import TERMINAL_FIELDS
                    for name in TERMINAL_FIELDS:
                        metadata.pop(name, None)
                    metadata.update(terminal,log_path=str(directory/'native.log'),replay_valid=path is not None)
                    metadata['profile'].pop('observed_saturated_depths',None)
                    metadata['profile']['resident_profile_reused']=True
                    metadata['profile']['original_calibration_wall_seconds']=metadata['profile'].get('calibration_total_wall_seconds')
                    metadata['profile']['calibration_total_wall_seconds']=0.0
                    outcome=NativeOutcome(path,elapsed,effective,directory,metadata)
                reused=True
            # Distributed reconstruction prints the solved path only on its
            # controller. Unsolved searches print one record on every rank.
            # ResidentWorkers separately requires a completion receipt from all ranks.
            expected_records = 1 if outcome.path is not None else len(devices)
            if self._workers is not None and outcome.metadata.get('terminal_records') != expected_records:
                raise NativeBackendError('resident search has unexpected terminal record count')
            persistent_metadata = dict(reused=reused,
                worker_pids=[] if self._workers is None else self._workers.pids,
                request_count=0 if self._workers is None else self._workers.sequence)
            if not rust_outcome_owned:
                outcome.metadata['persistent_session'] = persistent_metadata
            if remember_preparation and self._workers is not None:
                result_recorder = getattr(self._workers, 'remember_preparation_result', None)
                rust_owns_metadata = (self._engine == 'rust'
                    and callable(result_recorder)
                    and result_recorder(identity_key, outcome.metadata))
                self._rust_metadata_owned = rust_owns_metadata
                recorder = getattr(self._workers, 'remember_preparation', None)
                rust_owns_identity = rust_owns_metadata or (
                    self._engine == 'rust' and callable(recorder)
                    and recorder(identity_key))
                self._identity = None if rust_owns_identity else identity
                self._metadata = (None if rust_owns_metadata
                                  else deepcopy(outcome.metadata))
                rust_outcome_owned = rust_owns_metadata
            if rust_outcome_owned:
                owner_metadata = outcome.metadata.get('persistent_session')
                if (type(owner_metadata) is not dict
                        or owner_metadata.get('reused') is not reused
                        or owner_metadata.get('request_count') != persistent_metadata['request_count']):
                    raise NativeBackendError('Rust resident session metadata is inconsistent')
            else:
                Path(run_dir,'native-outcome.json').write_text(json.dumps(dict(path=outcome.path,
                    elapsed_seconds=outcome.elapsed_seconds,effective_beam_width=outcome.effective_beam_width,
                    metadata=outcome.metadata),indent=2),encoding='utf-8')
            return outcome
        except BaseException:
            self._release()
            raise
        finally:
            self._run_owner = None
            self._lock.release()

