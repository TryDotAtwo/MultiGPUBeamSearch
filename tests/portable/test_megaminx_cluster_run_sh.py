from pathlib import Path
import os
import subprocess

import pytest


def test_run_sh_is_thin_python_wrapper():
    script = Path("portable/megaminx_cluster/run.sh")
    result = subprocess.run(["bash", "-n", str(script)], text=True, capture_output=True)
    if result.returncode != 0 and ("CreateProcessCommon" in result.stderr or os.name == "nt"):
        pytest.skip("Linux bash is unavailable on this Windows host")
    assert result.returncode == 0, result.stderr
    text = script.read_text()
    assert "scripts/submit.py" not in text
    assert "-m portable.megaminx_cluster.submit" in text
    assert '"$@"' in text
