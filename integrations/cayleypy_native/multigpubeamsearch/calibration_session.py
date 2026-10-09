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
        if self.process.poll() is not None:
            raise RuntimeError('inference session is no longer running')
        self.process.stdin.write(json.dumps({'batch':batch,'parents':parents})+'\n')
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
