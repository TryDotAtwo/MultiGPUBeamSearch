"""Ship the matching native source with a wheel built from this repository."""
from pathlib import Path
import hashlib
import json
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithNativeSource(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parents[2]
        if not (root / "tools/stream1_mlp_libtorch_backend.hpp").is_file():
            raise RuntimeError("Build this wheel from the matching complete native repository checkout")
        target = Path(self.build_lib) / "multigpubeamsearch/_native_source"
        extensions = {".py", ".cu", ".cuh", ".cpp", ".hpp", ".h", ".cmake", ".sh", ".json", ".txt"}
        files = [root / "CMakeLists.txt", Path(__file__).parent / "LICENSE"]
        for folder in ("cuda", "src", "tools", "tests", "cmake", "configs", "schemas", "third_party"):
            files.extend(path for path in (root / folder).rglob("*") if path.is_file()
                         and (path.suffix in extensions or path.name.startswith("LICENSE"))
                         and not any(part in (".git", "__pycache__", ".pytest_cache") for part in path.parts))
        manifest = {}
        for path in sorted(set(files)):
            relative = Path("LICENSE") if path.name == "LICENSE" and path.parent == Path(__file__).parent else path.relative_to(root)
            raw = path.read_bytes().replace(b"\r\n", b"\n")
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            manifest[relative.as_posix()] = hashlib.sha256(raw).hexdigest()
        (target / "source_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")


setup(cmdclass={"build_py": BuildWithNativeSource})
