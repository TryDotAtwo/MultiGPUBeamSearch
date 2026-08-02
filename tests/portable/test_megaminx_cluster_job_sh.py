from pathlib import Path


def test_job_payload_is_fail_closed_and_uses_one_rank_per_gpu():
    text = Path("portable/megaminx_cluster/scripts/job.sh").read_text()
    assert "set -euo pipefail" in text
    assert "MEGAMINX_GPU_IDS//:/," in text
    workflow = Path("portable/megaminx_cluster/orchestrate.py").read_text()
    assert 'f"--nproc-per-node={world_size}"' in workflow
    assert "-m portable.megaminx_cluster.workflow" in text
    assert 'source "${MEGAMINX_RUN_DIR}/selected_profile.env"' in text
    assert 'str(puzzle_id), str(depth), str(beam)' in workflow
    assert "cmake" not in text
    assert "nvcc" not in text
    assert "git clone" not in text
    assert "CUDA_FORCE_PTX_JIT" not in text


def test_job_detects_then_autotunes_unknown_hardware_before_profile_selection():
    text = Path("portable/megaminx_cluster/scripts/job.sh").read_text()
    hardware = text.index("--hardware-only")
    autotune = text.index("portable.megaminx_cluster.auto_profile")
    selected = text.index("selected_profile.env")
    assert hardware < autotune < selected


def test_auto_profile_does_not_cap_autotune_at_requested_solve_beam():
    text = Path("portable/megaminx_cluster/auto_profile.py").read_text()
    assert "MEGAMINX_AUTOTUNE_MAX_BEAM" not in text
