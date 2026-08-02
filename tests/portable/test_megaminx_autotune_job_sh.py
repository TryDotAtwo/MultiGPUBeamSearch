from pathlib import Path
import os
import subprocess

import pytest


def test_autotune_wrapper_is_thin_and_fail_closed():
    path = Path("portable/megaminx_cluster/autotune.sh")
    result = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True)
    if result.returncode != 0 and ("CreateProcessCommon" in result.stderr or os.name == "nt"):
        pytest.skip("Linux bash is unavailable on this Windows host")
    assert result.returncode == 0, result.stderr
    text = path.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "-m portable.megaminx_cluster.autotune_submit" in text
    assert '"$@"' in text


def test_autotune_job_verifies_payload_and_runs_one_controller():
    text = Path("portable/megaminx_cluster/scripts/autotune_job.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "verify_archive_payloads" in text
    assert "preflight_cli" in text
    assert "--hardware-only" in text
    assert "MEGAMINX_AUTOTUNE_GPU_IDS//:/," in text
    assert 'export PYTHONPATH="${MEGAMINX_ARCHIVE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"' in text
    assert 'export LD_LIBRARY_PATH="${MEGAMINX_ARCHIVE_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"' in text
    assert text.count("-m portable.megaminx_cluster.autotune.controller") == 1
    assert "cmake" not in text
    assert "nvcc" not in text
    assert "git clone" not in text
