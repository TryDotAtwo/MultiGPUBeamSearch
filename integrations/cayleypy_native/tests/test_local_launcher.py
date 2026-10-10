import os
import sys

import pytest

from multigpubeamsearch.backend import collect_worker_logs
from multigpubeamsearch.errors import NativeBackendError
from multigpubeamsearch.local_launcher import run_local_ranks


def launch(tmp_path, script, timeout=5):
    worker_logs = tmp_path / 'worker-logs'
    worker_logs.mkdir()
    log = tmp_path / 'launcher.log'
    seconds = run_local_ranks([sys.executable, '-c', script], world_size=2,
        cwd=tmp_path, env=dict(os.environ), timeout=timeout, log_path=log,
        worker_logs=worker_logs)
    metadata = collect_worker_logs(worker_logs, log, tmp_path/'native.log', 2, strict=True)
    return seconds, metadata


def test_native_cohort_identity_and_complete_logs(tmp_path):
    seconds, metadata = launch(tmp_path,
        "import os,sys; assert sys.argv[-2:] == ['2',os.environ['RANK']]; "
        "assert os.environ['LOCAL_RANK']==os.environ['RANK']; "
        "assert os.environ['OMP_NUM_THREADS']=='1'; print(os.environ['RANK']); print('err',file=sys.stderr)")
    assert seconds > 0 and metadata['worker_logs_complete']
    assert len(metadata['worker_streams']) == 4


def test_failure_cancels_other_rank(tmp_path):
    with pytest.raises(NativeBackendError, match='rank failed'):
        launch(tmp_path, "import os,time,sys; sys.exit(3) if os.environ['RANK']=='0' else time.sleep(30)")


def test_shared_timeout(tmp_path):
    with pytest.raises(NativeBackendError, match='timed out'):
        launch(tmp_path, 'import time; time.sleep(30)', timeout=.1)
