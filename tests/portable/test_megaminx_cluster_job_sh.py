from pathlib import Path


def test_job_payload_is_fail_closed_and_uses_one_rank_per_gpu():
    text = Path("portable/megaminx_cluster/scripts/job.sh").read_text()
    assert "set -euo pipefail" in text
    assert "MEGAMINX_GPU_IDS//:/," in text
    assert '--nproc-per-node="${GPU_COUNT}"' in text
    assert "--no-python" in text
    assert 'source "${MEGAMINX_RUN_DIR}/selected_profile.env"' in text
    assert '"${MEGAMINX_PUZZLE}" "${MEGAMINX_DEPTH}" "${MEGAMINX_BEAM}"' in text
    assert "cmake" not in text
    assert "nvcc" not in text
    assert "git clone" not in text
    assert "CUDA_FORCE_PTX_JIT" not in text
