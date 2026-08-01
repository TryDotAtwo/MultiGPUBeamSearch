from __future__ import annotations

import subprocess

import pytest

from portable.megaminx_cluster.autotune_submit import (
    build_autotune_sbatch_command,
    main,
    parse_args,
)


class Recorder:
    def __init__(self, stdout="8123;cluster\n"):
        self.calls = []
        self.stdout = stdout

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, self.stdout, "")


def test_defaults_match_public_six_hour_contract():
    config = parse_args(["--gpus", "0,1,2,3,4,5,6,7"])
    assert config.gpu_ids == tuple(range(8))
    assert config.min_beam == 30_000_000
    assert config.time_budget_seconds == 6 * 3600
    assert config.bfs_hash_budget_mib == 256
    assert config.puzzle_ids == (900, 950, 1000)


def test_explicit_puzzles_and_budget_are_accepted():
    config = parse_args([
        "--gpus", "7,5", "--puzzles", "3,4,9", "--min-beam", "40000000",
        "--time-budget", "90m", "--bfs-hash-budget-mib", "128",
    ])
    assert config.gpu_ids == (7, 5)
    assert config.puzzle_ids == (3, 4, 9)
    assert config.time_budget_seconds == 5400
    assert config.bfs_hash_budget_mib == 128


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--gpus", "0,0"],
        ["--gpus", "0", "--puzzles", "1,2"],
        ["--gpus", "0", "--puzzles", "1,1,2"],
        ["--gpus", "0", "--time-budget", "6hours"],
        ["--gpus", "0", "--bfs-hash-budget-mib", "0"],
    ],
)
def test_invalid_inputs_fail_closed(argv):
    with pytest.raises(ValueError):
        parse_args(argv)


def test_builds_one_job_with_safe_exports(tmp_path):
    config = parse_args(["--gpus", "3,1", "--dry-run"])
    command = build_autotune_sbatch_command(config, tmp_path, tmp_path / "autotune-runs" / "r1")
    assert command[:5] == ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--gres=gpu:2"]
    assert command[-1] == str(tmp_path / "scripts" / "autotune_job.sh")
    exports = next(item for item in command if item.startswith("--export="))
    assert "MEGAMINX_AUTOTUNE_GPU_IDS=3:1" in exports
    assert "MEGAMINX_AUTOTUNE_MIN_BEAM=30000000" in exports
    assert "MEGAMINX_AUTOTUNE_PUZZLES=900:950:1000" in exports
    assert "MEGAMINX_AUTOTUNE_BFS_HASH_BUDGET_MIB=256" in exports


def test_dry_run_never_calls_sbatch(tmp_path, capsys):
    recorder = Recorder()
    assert main(["--gpus", "0", "--dry-run"], run_sbatch=recorder, archive_root=tmp_path) == 0
    assert recorder.calls == []
    assert "sbatch --parsable" in capsys.readouterr().out


def test_submit_creates_run_and_prints_job_id(tmp_path, capsys):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "autotune_job.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    recorder = Recorder()
    assert main(["--gpus", "0"], run_sbatch=recorder, archive_root=tmp_path) == 0
    assert len(recorder.calls) == 1
    assert "submitted_job_id=8123" in capsys.readouterr().out
    assert (tmp_path / "autotune-runs").is_dir()
