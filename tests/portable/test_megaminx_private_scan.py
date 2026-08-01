from pathlib import Path
import pytest
from tools.build_megaminx_native_release import build_release


def test_builder_rejects_private_absolute_paths_in_text_payload(tmp_path: Path):
    root = tmp_path / "stage"
    for directory in ("bin", "lib", "data", "weights", "profiles", "scripts", "portable/megaminx_cluster", "portable/megaminx_cluster/autotune"):
        (root / directory).mkdir(parents=True)
    (root / "bin/production_runner").write_bytes(b"ELF")
    (root / "lib/libtorch.so").write_bytes(b"runtime")
    (root / "data/test.csv").write_text("initial_state_id,initial_state\n900,1,0\n")
    (root / "data/puzzle_info.json").write_text("{}")
    (root / "profiles/registry.json").write_text('{"schema_version":1,"profiles":[]}')
    (root / "scripts/job.sh").write_text("#!/usr/bin/env bash\n")
    (root / "scripts/preflight.sh").write_text("#!/usr/bin/env bash\n")
    (root / "scripts/autotune_job.sh").write_text("#!/usr/bin/env bash\n")
    (root / "run.sh").write_text("#!/usr/bin/env bash\n")
    (root / "autotune.sh").write_text("#!/usr/bin/env bash\n")
    (root / "portable/megaminx_cluster/autotune/calibration.json").write_text("{}")
    (root / "README.md").write_text("Megaminx release\n")
    (root / "weights/manifest.json").write_text('{"source_weights":"C:\\\\Users\\\\person\\\\secret.pth"}')
    with pytest.raises(ValueError, match="private"):
        build_release(root, tmp_path / "out", 90, lambda _: "arch = sm_90\n")
