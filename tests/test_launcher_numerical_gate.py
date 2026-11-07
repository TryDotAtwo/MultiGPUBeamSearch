"""Real shell rank-start boundary; synthetic prerequisites, not GPU acceptance."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]
BASH=shutil.which('bash')


@pytest.mark.skipif(os.name!='posix' or not BASH, reason='remote POSIX shell regression')
def test_failed_numerical_gate_starts_zero_ranks(tmp_path):
    called=tmp_path/'numerical-gate-called'
    started=tmp_path/'rank-started'
    runner=tmp_path/'production_runner'
    runner.write_text('#!/bin/bash\nset -euo pipefail\n'
        'if [[ ${1:-} == --build-info ]]; then echo "{}"; exit 0; fi\n'
        'printf "%s\\n" "$RANK" >> "$NUMERIC_TEST_RANK_MARKER"\n',encoding='utf-8')
    runner.chmod(0o755)
    verifier=tmp_path/'verifier.sh'
    verifier.write_text('#!/bin/bash\nset -euo pipefail\n'
        'if [[ ${1:-} == tools/cube4_production_preflight.py ]]; then\n'
        '  touch "$NUMERIC_TEST_GATE_MARKER"; exit 23\n'
        'fi\n'
        '# Other prerequisites are synthetic here; this is shell protocol only.\n'
        'exit 0\n',encoding='utf-8')
    verifier.chmod(0o755)
    env={key:value for key,value in os.environ.items()
         if not key.startswith(('BEAM_','H200_')) and key not in ('WORLD_SIZE','RANK','LOCAL_RANK')}
    env.update(H200_REPO_DIR=ROOT.as_posix(),H200_RUN_ROOT=tmp_path.as_posix(),
        H200_BUILD_DIR=tmp_path.as_posix(),H200_PYTHON=verifier.as_posix(),
        H200_SMI=verifier.as_posix(),H200_WORLD_SIZE='2',BEAM_WIDTH='4096',
        NUMERIC_TEST_GATE_MARKER=called.as_posix(),
        NUMERIC_TEST_RANK_MARKER=started.as_posix())
    completed=subprocess.run([BASH,str(ROOT/'hpc/h200_dual_large_beam.sh')],
        env=env,capture_output=True,text=True,timeout=15)
    assert called.exists(), 'launcher never invoked mandatory numerical gate'
    assert completed.returncode!=0, completed.stdout+completed.stderr
    assert not started.exists(), 'beam ranks started despite failed numerical gate'
    assert not list(tmp_path.glob('h200_run.*/rank*.log'))
