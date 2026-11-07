"""Private atomic request mailbox for a single local native worker group."""
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from .errors import NativeBackendError


class ResidentWorkers:
    def __init__(self, command, *, world, directory, cwd, env):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.processes = []
        self.handles = []
        self.sequence = 0
        self.closed = False
        rendezvous = 'resident-' + uuid.uuid4().hex
        try:
            for rank in range(world):
                handle = (self.directory / f'bootstrap-{rank}.log').open('wb')
                self.handles.append(handle)
                rank_env = dict(env, BEAM_SESSION_DIR=str(self.directory),
                    WORLD_SIZE=str(world), RANK=str(rank), LOCAL_RANK=str(rank),
                    BEAM_NCCL_RUN_ID=rendezvous)
                kwargs = {'start_new_session': True} if os.name == 'posix' else {
                    'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
                self.processes.append(subprocess.Popen(command, cwd=cwd, env=rank_env,
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=handle, shell=False, **kwargs))
        except BaseException:
            self.close()
            raise

    @property
    def pids(self):
        return [p.pid for p in self.processes]

    def submit(self, start, run_dir, *, timeout):
        if self.closed:
            raise NativeBackendError('resident worker group is closed')
        directory = Path(run_dir).resolve()
        request = dict(sequence=self.sequence, start=list(start), run_dir=str(directory))
        path = self.directory / f'request-{self.sequence}.json'
        temporary = path.with_suffix('.tmp')
        began = time.monotonic()
        try:
            temporary.write_text(json.dumps(request), encoding='utf-8')
            temporary.replace(path)
            while True:
                codes = [p.poll() for p in self.processes]
                if any(code is not None for code in codes):
                    raise NativeBackendError(f'resident worker exited: {codes}; logs={self.directory}')
                receipts = [self.directory/f'done-{self.sequence}-{rank}.json'
                            for rank in range(len(self.processes))]
                if all(p.is_file() for p in receipts):
                    for rank, receipt in enumerate(receipts):
                        row = json.loads(receipt.read_text())
                        if row != dict(sequence=self.sequence, rank=rank):
                            raise NativeBackendError('invalid resident completion receipt')
                    break
                if time.monotonic()-began >= timeout:
                    raise NativeBackendError(f'resident search timed out; logs={self.directory}')
                time.sleep(.02)
            with (directory/'native.log').open('w', encoding='utf-8') as log:
                for rank in range(len(self.processes)):
                    source = directory/f'resident-rank-{rank}.log'
                    log.write(f'[rank {rank}]\n' + source.read_text(encoding='utf-8') + '\n')
            self.sequence += 1
            return time.monotonic()-began
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.directory.exists():
            (self.directory/'stop').touch()
        deadline = time.monotonic()+2
        for process in self.processes:
            try:process.wait(timeout=max(.01,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:pass
        from .backend import _stop_process_tree
        errors=[]
        for process in self.processes:
            try:_stop_process_tree(process)
            except Exception as error:errors.append(error)
        for handle in self.handles:
            handle.close()
        if errors:
            raise NativeBackendError(f'failed to reap {len(errors)} resident workers') from errors[0]
