"""Explicit lifetime for sequential searches on one native GPU worker group."""
from contextvars import ContextVar
from copy import deepcopy
import json
import os
from pathlib import Path
import threading

from .errors import NativeBackendError
from .options import NativeOutcome

_ACTIVE = ContextVar('multigpubeamsearch_native_session', default=None)


def current_session():
    return _ACTIVE.get()


class NativeSession:
    """Keep compatible native workers alive inside ``with NativeSession():``.

    Graph/model/runner/topology/beam/depth/options changes release the old
    worker group before recalibration. Concurrent or nested use is rejected.
    Exiting the context, including on exceptions, frees all owned workers.
    """
    def __init__(self):
        self._owner_pid = os.getpid()
        self.closed = False
        self._workers = None
        self._identity = None
        self._metadata = None
        self._token = None
        self._lock = threading.Lock()

    def _check_owner(self):
        if os.getpid()!=self._owner_pid:
            raise RuntimeError('native session cannot be shared across Python processes')

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

    def beam_search(self, graph, **kwargs):
        """Explicit session call, also usable across separate notebook cells."""
        from .dispatch import beam_search
        self._check_owner()
        if self.closed:raise RuntimeError('native session is closed')
        active=current_session()
        if active is not None and active is not self:
            raise RuntimeError('nested native sessions are unsupported')
        token=_ACTIVE.set(self)
        try:return beam_search(graph,**kwargs)
        finally:_ACTIVE.reset(token)

    def _release(self):
        if self._workers is not None:self._workers.close()
        self._workers = None
        self._identity = None
        self._metadata = None

    def close(self):
        self._check_owner()
        if self.closed:return
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('cannot close a native session during a search')
        try:
            self._release()
            self.closed = True
        finally:self._lock.release()

    def launch(self, command, *, contract, run_dir, env, world, timeout):
        from .resident_transport import ResidentWorkers
        import subprocess
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

    def run(self, execute, contract, model, options, beam_width, max_steps, run_dir, devices, *, runtime=None):
        from .backend import parse_terminal
        from .models import verify_prepared_model
        self._check_owner()
        if self.closed:raise RuntimeError('native session is closed')
        if not self._lock.acquire(blocking=False):raise RuntimeError('concurrent native session searches are unsupported')
        try:
            # Runtime preparation already verifies executable bytes. Start state
            # is intentionally absent; all other search geometry stays fixed.
            identity=(contract.graph_hash,model.artifact_hash,model.backend,tuple(devices),
                beam_width,max_steps,repr(options),
                os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('CUDA_DEVICE_ORDER'),
                json.dumps(runtime.build_metadata,sort_keys=True) if runtime is not None else None)
            verify_prepared_model(model,contract)
            if identity != self._identity:self._release()
            if self._workers is None:
                outcome=execute(contract,model,options,beam_width,max_steps,run_dir,devices,runtime=runtime)
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
        finally:self._lock.release()
