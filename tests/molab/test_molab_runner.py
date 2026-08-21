from __future__ import annotations

from pathlib import Path

from tools.run_cayleypy_molab import DEFAULT_MOLAB_CACHE_ROOT, _toolchain_workspace


def test_toolchain_workspace_is_stable_across_output_directories(monkeypatch) -> None:
    monkeypatch.delenv("CAYLEYPY_MOLAB_CACHE_ROOT", raising=False)

    assert _toolchain_workspace() == DEFAULT_MOLAB_CACHE_ROOT.resolve()


def test_toolchain_workspace_accepts_explicit_cache_root(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "shared-cache"
    monkeypatch.setenv("CAYLEYPY_MOLAB_CACHE_ROOT", str(cache))

    assert _toolchain_workspace() == cache.resolve()
