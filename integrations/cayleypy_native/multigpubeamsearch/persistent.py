"""Explicit lifetime for sequential searches on one native GPU worker group."""
from contextvars import ContextVar
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import threading

from .errors import NativeBackendError
from .model_snapshot import requires_mlp_host_binding
from .options import NativeOutcome

_ACTIVE = ContextVar('multigpubeamsearch_native_session', default=None)


def current_session():
    return _ACTIVE.get()


class NativeSession:
    """Keep compatible native workers alive inside ``with NativeSession():``.

    Graph/model/runner/topology/beam/depth/options changes release the old
    worker group before recalibration. Concurrent or nested use is rejected.
    Exiting the context, including on exceptions, frees all owned workers.
    The experimental NCCL identity option retains block digests per session;
    every repeated check rereads the complete library.
    Experimental model-snapshot reuse covers explicit MLP NativeModel inputs
    in strict native mode; source and retained bytes are revalidated on reuse.
    """
    def __init__(self, *, engine="python", experimental_parallel_nccl_identity=False,
                 experimental_reuse_model_snapshot=False):
        if engine not in ("python", "rust"):
            raise ValueError("engine must be 'python' or 'rust'")
        if type(experimental_parallel_nccl_identity) is not bool:
            raise ValueError("experimental_parallel_nccl_identity must be a boolean")
        if type(experimental_reuse_model_snapshot) is not bool:
            raise ValueError("experimental_reuse_model_snapshot must be a boolean")
        self.nccl_digest_cache = None
        if experimental_parallel_nccl_identity:
            from .nccl_digest import ChunkDigestCache
            self.nccl_digest_cache = ChunkDigestCache()
        self.model_snapshot_cache = None
        if experimental_reuse_model_snapshot:
            from .model_snapshot import ModelSnapshotCache
            self.model_snapshot_cache = ModelSnapshotCache()
        self._owner_pid = os.getpid()
        self._engine = engine
        self._pending = None
        self.closed = False
        self._workers = None
        self._identity = None
        self._metadata = None
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
        if self._workers is not None:raise NativeBackendError('resident workers already exist')
        if self._engine == 'rust':
            if self._pending is None:
                raise NativeBackendError('Rust resident launch requires a prepared session')
            from .rust_resident import RustResidentWorkers
            model, options, max_steps, devices, prepared = self._pending
            self._workers = RustResidentWorkers(command, contract=contract, model=model,
                options=options, max_steps=max_steps, devices=devices, runtime=runtime or prepared,
                run_dir=run_dir, env=env, world=world)
            return self._workers.submit(contract.start, run_dir, timeout=timeout)
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

    def run(self, execute, contract, model, options, beam_width, max_steps, run_dir, devices, *, runtime=None):
        from .backend import parse_terminal
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
            identity=(contract.graph_hash,model.artifact_hash,model.backend,tuple(devices),binding_scope,
                beam_width,max_steps,repr(options),
                os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('CUDA_DEVICE_ORDER'),
                json.dumps(runtime.build_metadata,sort_keys=True) if runtime is not None else None)
            verify_prepared_model(model,contract)
            if identity != self._identity:self._release()
            if self._workers is None:
                if (self.model_snapshot_cache is not None
                        and self.model_snapshot_cache.needs_fresh_worker_snapshot(model, contract)):
                    model = self.model_snapshot_cache.refresh_for_worker(model, contract, run_dir)
                self._pending=(model,options,max_steps,devices,runtime)
                try:
                    outcome=execute(contract,model,options,beam_width,max_steps,run_dir,devices,runtime=runtime)
                finally:
                    self._pending=None
                if self._workers is not None:
                    self._identity=identity
                    self._metadata=deepcopy(outcome.metadata)
                reused=False
            else:
                directory=Path(run_dir)
                elapsed=self._workers.submit(contract.start,directory,timeout=options.timeout_seconds)
                path,effective,terminal=parse_terminal(directory/'native.log',contract,max_steps=max_steps)
                if effective != self._metadata['requested_beam_width']:
                    raise NativeBackendError('resident effective beam changed')
                metadata=deepcopy(self._metadata)
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
            outcome.metadata['persistent_session']=dict(reused=reused,
                worker_pids=[] if self._workers is None else self._workers.pids,
                request_count=0 if self._workers is None else self._workers.sequence)
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
