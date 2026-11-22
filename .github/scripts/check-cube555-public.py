"""Exercise CayleyPy's real public registry without launching a CUDA runner."""

import json
import runpy
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from multigpubeamsearch import dispatch
from multigpubeamsearch.options import NativeOutcome


def main(root):
    application = runpy.run_path(str(root / "solve_cube555.py"))["main"]
    calls = []

    def native_search(contract, model, options, width, steps, run_dir, devices, *, runtime):
        assert contract.state_len == 150 and model.backend == "ensemble"
        members = model.manifest["ensemble"]["models"]
        assert [member["family"] for member in members] == ["cube555_transformer", "cube555_mlp"]
        assert [member["coefficient"] for member in members] == [0.8, 1 - 0.8]
        assert devices == (0, 1) and steps == 2 and options.touch_bfs_radius == 0
        calls.append(width)
        profile = {
            "inference_calibration": {"parent_batch": 128},
            "pipeline_calibration": {
                "phase": "component_calibrated",
                "selection": "fixture",
                "environment": {"BEAM_B_MICRO": "128"},
            },
        }
        return NativeOutcome(None, 0.0, 65536 if width == "max" else width,
                             run_dir, {"profile": profile})

    manifest = root / "SHA256SUMS"
    original = manifest.read_bytes()
    with tempfile.TemporaryDirectory() as work, ExitStack() as entered:
        for fixture in (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
            patch.object(torch.cuda, "get_device_name", return_value="Fixture GPU"),
            patch.object(torch.cuda, "get_device_properties", side_effect=lambda i: f"fixture-{i}"),
            patch.object(dispatch, "prepare_runtime", return_value=SimpleNamespace(build_metadata={})),
            patch.object(dispatch, "run_native", side_effect=native_search),
        ):
            entered.enter_context(fixture)
        args = ["--assets", str(root / "assets"), "--work-dir", work,
                "--beam-width", "max", "--max-steps", "2", "--bfs-radius", "0"]
        try:
            application(args)
            assert calls == ["max", 65536], calls
            docs = original.splitlines(keepends=True)
            docs = [b"f" * 64 + b"  README.md\n"
                    if line.endswith(b"  README.md\n") else line for line in docs]
            manifest.write_bytes(b"".join(docs))
            application(args)
            assert calls == ["max", 65536, 65536], calls
            runtime = original.splitlines(keepends=True)
            index = next(i for i, line in enumerate(runtime) if b"  wheels/" in line)
            runtime[index] = b"f" * 64 + runtime[index][64:]
            manifest.write_bytes(b"".join(runtime))
            application(args)
            assert calls == ["max", 65536, 65536, "max", 65536], calls
            files = sorted((Path(work) / "cache" / "profile-grids").glob("*.json"))
            assert len(files) == 2
            assert {json.loads(path.read_text())["grid_reused"] for path in files} == {False, True}
        finally:
            manifest.write_bytes(original)
    print("CayleyPy public dispatch passed: real blend, cold max, warm width after README change, cold max after wheel input change. No CUDA launched.")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
