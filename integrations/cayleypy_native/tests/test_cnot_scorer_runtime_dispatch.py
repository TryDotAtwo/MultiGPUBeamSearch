"""The packaged Rust extension owns the CNOT scorer when it is available."""

from contextlib import nullcontext
import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from multigpubeamsearch import cnot_scorer


def _prepare_python_scorer(monkeypatch, tmp_path):
    api = SimpleNamespace(
        mgbs_tqp_open=lambda *_: 1,
        mgbs_tqp_metadata=lambda *_: json.dumps({
            "feature_model_inputs": 102,
            "feature_ids": [f"F{i:02d}" for i in range(102)],
            "max_packet_rows": 8,
        }).encode("utf-8"),
        mgbs_tqp_close=lambda *_: None,
    )
    monkeypatch.setattr(cnot_scorer, "_load_api", lambda _: api)
    monkeypatch.setattr(torch, "__version__", "2.9.0")
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    return manifest


def test_public_tqp_scorer_uses_rust_extension_when_it_is_available(monkeypatch, tmp_path):
    class NativeScorer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    manifest = _prepare_python_scorer(monkeypatch, tmp_path)
    monkeypatch.setattr(cnot_scorer, "_rust_tqp_scorer", lambda: NativeScorer, raising=False)

    scorer = cnot_scorer.TQPScorer(
        "field.so", manifest, n=4, device="cuda:0", inference_batch=3
    )

    assert type(scorer) is NativeScorer
    assert scorer.args == ("field.so", manifest)
    assert scorer.kwargs == {"n": 4, "device": "cuda:0", "inference_batch": 3}


def test_public_tqp_scorer_keeps_python_fallback_when_extension_is_missing(monkeypatch, tmp_path):
    manifest = _prepare_python_scorer(monkeypatch, tmp_path)
    monkeypatch.setattr(cnot_scorer, "_rust_tqp_scorer", lambda: None, raising=False)

    scorer = cnot_scorer.TQPScorer("field.so", manifest, n=4, device="cuda:0")

    assert type(scorer) is cnot_scorer.TQPScorer
    assert scorer.metadata["feature_model_inputs"] == 102


def test_installed_pyo3_scorer_is_selected_and_preserves_argument_contract():
    native = importlib.import_module("multigpubeamsearch._beam_runtime")

    assert cnot_scorer._rust_tqp_scorer() is native.TQPScorer
    with pytest.raises(ValueError, match="valid CNOT dimension"):
        cnot_scorer.TQPScorer("unused", "unused", n=1, device="cuda:0")
    with pytest.raises((TypeError, ValueError)):
        cnot_scorer.TQPScorer(
            "unused",
            "unused",
            n=4,
            device="cuda:0",
            experimental_native_batch=1,
        )
