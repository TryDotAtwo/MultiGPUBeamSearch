import subprocess

import pytest

from portable.megaminx_cluster.submit import build_sbatch_command, load_cluster_env, main, parse_args


class Recorder:
    def __init__(self, stdout="Submitted batch job 4321\n"):
        self.calls = []
        self.stdout = stdout

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, self.stdout, "")


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--beam", "1000", "--puzzle", "900"], "missing required --gpus"),
        (["--gpus", "0,1", "--puzzle", "900"], "missing required --beam"),
        (["--gpus", "0,1", "--beam", "1000"], "missing required --puzzle; specify one Megaminx puzzle id"),
    ],
)
def test_missing_required_argument_never_calls_sbatch(tmp_path, capsys, argv, message):
    recorder = Recorder()
    assert main(argv, run_sbatch=recorder, archive_root=tmp_path) == 2
    assert recorder.calls == []
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("gpus", ["", "0,", "0,0", "-1", "a", "0 1"])
def test_rejects_invalid_gpu_lists(gpus):
    with pytest.raises(ValueError, match="gpus"):
        parse_args(["--gpus", gpus, "--beam", "1000", "--puzzle", "900"])


def test_reflect_only_requires_original_solution():
    with pytest.raises(ValueError, match="original-solution"):
        parse_args(["--gpus", "0", "--beam", "1000", "--puzzle", "900", "--reflect", "only"])


def test_builds_one_node_sbatch_command(tmp_path):
    config = parse_args([
        "--gpus", "3,1", "--beam", "1000000000", "--puzzle", "900",
        "--reflect", "after", "--depth", "120",
    ])
    command = build_sbatch_command(config, tmp_path, tmp_path / "runs" / "r1")
    assert command[:5] == ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--gres=gpu:2"]
    assert command[-1] == str(tmp_path / "scripts" / "job.sh")
    exports = next(item for item in command if item.startswith("--export="))
    assert "MEGAMINX_GPU_IDS=3:1" in exports
    assert "MEGAMINX_BEAM=1000000000" in exports
    assert "MEGAMINX_PUZZLE=900" in exports
    assert "MEGAMINX_REFLECT=after" in exports


def test_submit_prints_parsable_job_id_and_run_dir(tmp_path, capsys):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "job.sh").write_text("#!/bin/bash\n")
    recorder = Recorder(stdout="4321;cluster\n")
    rc = main(
        ["--gpus", "0,1", "--beam", "1000", "--puzzle", "900"],
        run_sbatch=recorder,
        archive_root=tmp_path,
    )
    assert rc == 0
    assert len(recorder.calls) == 1
    output = capsys.readouterr().out
    assert "submitted_job_id=4321" in output
    assert "run_dir=" in output


def test_dry_run_makes_no_sbatch_call(tmp_path, capsys):
    recorder = Recorder()
    rc = main(
        ["--gpus", "0", "--beam", "1000", "--puzzle", "1", "--dry-run"],
        run_sbatch=recorder,
        archive_root=tmp_path,
    )
    assert rc == 0
    assert recorder.calls == []
    assert "sbatch --parsable" in capsys.readouterr().out


def test_loads_cluster_env_and_adds_only_configured_sbatch_options(tmp_path):
    env_file = tmp_path / "cluster.env"
    env_file.write_text(
        "SLURM_PARTITION=gpu\nSLURM_ACCOUNT=research\nSLURM_QOS=normal\n"
        "SLURM_TIME=12:00:00\nSLURM_CPUS_PER_TASK=16\nSLURM_MEM=240G\n"
        "MEGAMINX_SCRATCH_ROOT=/scratch/beam\n"
    )
    cluster = load_cluster_env(env_file)
    config = parse_args(["--gpus", "0", "--beam", "1000", "--puzzle", "7"])
    command = build_sbatch_command(config, tmp_path, tmp_path / "runs" / "r1", cluster)
    assert "--partition=gpu" in command
    assert "--account=research" in command
    assert "--qos=normal" in command
    assert "--time=12:00:00" in command
    assert "--cpus-per-task=16" in command
    assert "--mem=240G" in command
    assert "MEGAMINX_SCRATCH_ROOT=/scratch/beam" in next(x for x in command if x.startswith("--export="))