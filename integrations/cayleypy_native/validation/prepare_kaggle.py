"""Build a reviewable private kernel containing only explicitly allowed adapter files."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


def prepare(output: Path):
    package = Path(__file__).resolve().parents[1]
    selected = sorted((package / "cayleypy_native").glob("*.py")) + sorted((package / "tests").glob("*.py"))
    selected += [package / "validation" / name for name in (
        "__init__.py", "smoke_cases.py", "test_smoke_cases.py", "real_cases.py",
        "test_real_cases.py", "tetraminx_fixture.json", "prepared_acceptance.py")]
    files = {path.relative_to(package).as_posix(): path.read_bytes() for path in selected}
    for name, contents in files.items():
        if name.endswith(".py"):
            ast.parse(contents.decode("utf-8"), filename=name)
        else:
            json.loads(contents)
    manifest = {"scope": "new optional adapter and validation only; native source is fetched by pinned public commit",
                "files": {name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()}}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for name, contents in files.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 31, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            stream.writestr(info, contents)
        info = zipfile.ZipInfo("payload_manifest.json", date_time=(2026, 8, 31, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        stream.writestr(info, json.dumps(manifest, indent=2))
    payload = archive.getvalue()
    template = (package / "validation" / "kaggle_bootstrap.py").read_text(encoding="utf-8")
    source = template.replace("@@PAYLOAD_B64@@", base64.b64encode(payload).decode("ascii")).replace(
        "@@PAYLOAD_SHA256@@", hashlib.sha256(payload).hexdigest())
    ast.parse(source)
    output.mkdir(parents=True, exist_ok=True)
    (output / "validate_adapter.py").write_text(source, encoding="utf-8", newline="\n")
    metadata = {"id": "trydotatwo/cayleypy-native-adapter-2xt4-validation",
                "title": "CayleyPy Native Adapter 2xT4 Validation", "code_file": "validate_adapter.py",
                "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
                "machine_shape": "NvidiaTeslaT4", "enable_internet": True,
                "dataset_sources": [],
                "competition_sources": ["cayley-py-professor-tetraminx-solve-optimally"],
                "kernel_sources": ["trydotatwo/cayleypy-2xt4-tetraminx-example"], "model_sources": []}
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "payload_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result = {"directory": str(output), "files": len(files), "payload_bytes": len(payload),
              "script_bytes": len(source.encode()), "script_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "private": metadata["is_private"], "native_uncommitted_source_included": False}
    (output / "package-verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output)
