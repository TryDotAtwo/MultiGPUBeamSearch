import json
from hashlib import sha256
from pathlib import Path

from tools.render_megaminx_release_metadata import render_metadata


def test_renderer_adds_dynamic_release_and_model_provenance(tmp_path: Path):
    weights = tmp_path / "weights"; weights.mkdir()
    model = weights / "manifest.json"; model.write_text('{"state_len":120,"num_classes":120,"output_dim":24,"dtype":"fp16"}')
    (weights / "x.fp16").write_bytes(b"x")
    result = render_metadata({"backend": "mlp"}, "v1", "megaminx-sm90-linux-x86_64.tar.zst", "b" * 40, model, weights)
    assert result["release_tag"] == "v1" and result["solver_commit"] == "b" * 40
    assert result["model_manifest"]["state_len"] == 120
    assert result["model_sha256"] == sha256(b"x").hexdigest()
