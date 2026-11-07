"""CLI must reject missing inputs, not silently return success on import."""
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.skipif(os.name!='posix',reason='remote POSIX CLI')
def test_missing_numerical_reference_is_not_success(tmp_path):
    completed=subprocess.run([sys.executable,str(ROOT/'tools/cube4_production_preflight.py'),
        '--runner',str(tmp_path/'absent-runner'),'--probe',str(tmp_path/'absent-probe'),
        '--reference-dir',str(tmp_path/'absent-reference'),'--output-root',str(tmp_path/'output'),
        '--source-root',str(ROOT),'--world-size','2'],capture_output=True,text=True,timeout=10)
    assert completed.returncode!=0
    assert not (tmp_path/'output').exists()
