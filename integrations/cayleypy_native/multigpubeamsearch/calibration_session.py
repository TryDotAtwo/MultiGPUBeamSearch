"""Explicit single-owner persistent inference probes, one process per GPU."""
from __future__ import annotations
import json
from pathlib import Path
import queue
import subprocess
import threading
import time


class InferenceSession:
    def __init__(self, command, environment, log_path, *, deadline):
        self.deadline = deadline
        self.rows = queue.Queue()
        self.log = Path(log_path).open('w', encoding='utf-8')
        try:
            self.process = subprocess.Popen(command, env=environment, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                bufsize=1, start_new_session=True)
        except BaseException:
            self.log.close()
            raise
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.log.write(line); self.log.flush()
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        self.rows.put(row)
                except ValueError:
                    pass
        finally:
            self.rows.put(None)

    def receive(self, deadline=None):
        until = self.deadline if deadline is None else min(self.deadline, deadline)
        try:
            row = self.rows.get(timeout=max(.001, until-time.monotonic()))
        except queue.Empty as error:
            raise TimeoutError('persistent inference probe timed out') from error
        if row is None:
            raise RuntimeError('persistent inference probe exited; inspect its log')
        return row

    def send(self, batch, parents):
        self.send_request({'batch':batch,'parents':parents})

    def send_request(self, request):
        if self.process.poll() is not None:
            raise RuntimeError('inference session is no longer running')
        self.process.stdin.write(json.dumps(request)+'\n')
        self.process.stdin.flush()

    def close(self):
        from .backend import _stop_process_tree
        if self.process.poll() is None:
            _stop_process_tree(self.process)
        self.process.wait()
        self.reader.join(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()


class EnsembleProbePool:
    """Lazy rank sessions; restart only a failed process, not every batch."""
    def __init__(self, helper, inputs, capacity, parents, world, environment, directory, deadline):
        self.helper=helper;self.inputs=inputs;self.capacity=capacity;self.parents=parents
        self.world=world;self.environment=environment;self.directory=Path(directory)
        self.deadline=deadline;self.sessions={};self.starts=0

    def measure(self, batch, parents, deadline):
        for rank in range(self.world):
            if rank not in self.sessions:
                self.starts+=1
                session=InferenceSession([str(self.helper),str(self.inputs),str(self.capacity),
                    str(self.parents),str(rank),'--session'],self.environment,
                    self.directory/f'session-{self.starts}-rank-{rank}.log',deadline=self.deadline)
                self.sessions[rank]=session
                ready=session.receive(deadline)
                if ready.get('ready') is not True or ready.get('device')!=rank:
                    raise RuntimeError('inference rank did not acknowledge readiness')
        for session in self.sessions.values():session.send(batch,parents)
        rows=[];failures=[]
        for rank in range(self.world):
            session=self.sessions[rank]
            try:
                row=session.receive(deadline)
                if (row.get('batch')!=batch or row.get('parents')!=parents or row.get('device')!=rank
                    or row.get('correctness_passed') is not True or row.get('numeric_error')!=0
                    or len(row.get('seconds',[]))<5):
                    raise ValueError('invalid inference session receipt')
                rows.append(row)
            except (TimeoutError,RuntimeError,ValueError) as error:
                failures.append(f'rank {rank}: {error}')
                session.close();del self.sessions[rank]
        return rows,failures

    def close(self):
        for session in self.sessions.values():session.close()
        self.sessions={}
