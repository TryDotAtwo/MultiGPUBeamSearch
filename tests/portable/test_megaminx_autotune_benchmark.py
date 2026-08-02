from pathlib import Path
import subprocess

import pytest

from portable.megaminx_cluster.workflow import _run_benchmark


def test_depth8_benchmark_scores_only_full_frontier(monkeypatch, tmp_path):
    root = tmp_path / "archive"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (root / "portable/megaminx_cluster").mkdir(parents=True)
    (root / "bin").mkdir()
    (root / "data").mkdir()
    (root / "weights").mkdir()
    rows = "\n".join(
        f"autotune_depth_done=7 depth_sec={1.0 + rank / 100} next_frontier_size=3751936"
        for rank in range(8)
    )
    monkeypatch.setattr(
        "portable.megaminx_cluster.workflow.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, rows, ""),
    )
    metrics = _run_benchmark(root, run_dir, 8, "job", 900, 8, 30_000_000, root / "data/test.csv")
    assert metrics["benchmark_depth"] == 8
    assert metrics["frontier_full"] is True
    assert metrics["depth_sec"] == pytest.approx(1.07)
    assert metrics["rank_samples"] == 8


def test_depth8_benchmark_rejects_unfilled_frontier(monkeypatch, tmp_path):
    root = tmp_path / "archive"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (root / "portable/megaminx_cluster").mkdir(parents=True)
    (root / "bin").mkdir()
    (root / "data").mkdir()
    (root / "weights").mkdir()
    rows = "\n".join(
        "autotune_depth_done=7 depth_sec=1 next_frontier_size=3749999" for _ in range(8)
    )
    monkeypatch.setattr(
        "portable.megaminx_cluster.workflow.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, rows, ""),
    )
    with pytest.raises(RuntimeError, match="frontier is not full"):
        _run_benchmark(root, run_dir, 8, "job", 900, 8, 30_000_000, root / "data/test.csv")