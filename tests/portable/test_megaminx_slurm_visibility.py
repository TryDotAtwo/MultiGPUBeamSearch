from pathlib import Path


def test_job_preserves_slurm_cuda_visibility_and_preflight_queries_only_it():
    job = Path("portable/megaminx_cluster/scripts/job.sh").read_text()
    preflight = Path("portable/megaminx_cluster/scripts/preflight_cli.py").read_text()
    assert 'export CUDA_VISIBLE_DEVICES="${GPU_CSV}"' not in job
    assert 'CUDA_VISIBLE_DEVICES:?SLURM did not provide GPU visibility' in job
    assert 'f"--id={visible}"' in preflight
