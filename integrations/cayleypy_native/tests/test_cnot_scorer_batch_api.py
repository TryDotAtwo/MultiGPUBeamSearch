"""Fail closed when the explicitly requested scorer ABI is unavailable."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from multigpubeamsearch import cnot_scorer


def test_experimental_native_batch_rejects_an_old_scorer_before_open(monkeypatch, tmp_path):
    opened = []
    old_api = SimpleNamespace(mgbs_tqp_open=lambda *_: opened.append(True))
    monkeypatch.setattr(cnot_scorer, "_load_api", lambda _: old_api)
    monkeypatch.setattr(torch, "__version__", "2.9.0")
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="batch scorer ABI"):
        cnot_scorer._PythonTQPScorer("old-library", manifest, n=4, device="cuda:0",
                                     experimental_native_batch=True)
    assert not opened
