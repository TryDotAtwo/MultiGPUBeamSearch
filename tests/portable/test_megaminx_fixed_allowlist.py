from pathlib import Path
import pytest
from test_megaminx_native_release import cuobjdump_for, stage
from tools.build_megaminx_native_release import build_release


def test_builder_rejects_benign_named_file_outside_fixed_contract(tmp_path: Path):
    root = stage(tmp_path)
    (root / "notes.txt").write_text("not secret, but not part of the release")
    with pytest.raises(ValueError, match="allowlist"):
        build_release(root, tmp_path / "out", 90, cuobjdump_for(90))
