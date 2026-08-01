import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_release_uses_public_model_manifest_without_source_path():
    manifest = json.loads((ROOT / "portable/megaminx_cluster/model_manifest.json").read_text())
    assert "source_weights" not in manifest
    workflow = (ROOT / ".github/workflows/megaminx-native-release.yml").read_text()
    assert "model_manifest.json" in workflow
