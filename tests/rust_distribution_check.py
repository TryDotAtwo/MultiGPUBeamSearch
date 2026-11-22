"""Exercise checkout and source-distribution builds on a test host.

Outputs are written only under --out; native CUDA is not loaded on import.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(command: list[str], *, log: Path, cwd: Path, env: dict[str, str]) -> None:
    with log.open("w") as output:
        subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT,
                       check=True, timeout=900)


def one(directory: Path, pattern: str) -> Path:
    found = list(directory.glob(pattern))
    if len(found) != 1:
        raise AssertionError(f"expected one {pattern} in {directory}, got {found}")
    return found[0]


def check_wheel(wheel: Path, source: Path, out: Path, env: dict[str, str]) -> dict:
    prefix = "multigpubeamsearch/_native_source/"
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)):
            raise AssertionError("wheel contains duplicate archive paths")
        names = set(members)
        worker_name = "beam-runner.exe" if sys.platform == "win32" else "beam-runner"
        required = {"multigpubeamsearch/__init__.py", "multigpubeamsearch/rust_resident.py",
                    "cayleypy_native/__init__.py",
                    "multigpubeamsearch/_bin/" + worker_name}
        native_required = {prefix + name for name in (
            "source_manifest.json", "CMakeLists.txt", "cuda/cnot_domain.cu",
            "tools/production_runner.cu", "rust/beam-cli/Cargo.toml")}
        missing = (required | native_required) - names
        if missing:
            raise AssertionError(f"wheel missing package files: {sorted(missing)}")
        manifest = json.loads(archive.read(prefix + "source_manifest.json"))
        for name, expected in manifest.items():
            if sha256(archive.read(prefix + name)) != expected:
                raise AssertionError(f"bundled native source checksum mismatch: {name}")
        for name in ("CMakeLists.txt", "cuda/cnot_domain.cu", "tools/production_runner.cu",
                     "rust/beam-cli/Cargo.toml"):
            expected = sha256((source / name).read_bytes().replace(b"\r\n", b"\n"))
            if manifest.get(name) != expected:
                raise AssertionError(f"bundled native source differs from checkout: {name}")
        extensions = sorted(n for n in names if n.startswith("multigpubeamsearch/_beam_runtime.")
                            and n.endswith((".so", ".pyd")))
        if len(extensions) != 1:
            raise AssertionError(f"expected exactly one Rust extension, got {extensions}")
    target = out / "installed"
    run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
         "--target", str(target), str(wheel)], log=out / "install.log", cwd=out, env=env)
    code = r'''
import importlib, json, os, pathlib, subprocess, sys
sys.path.insert(0, sys.argv[1])
import multigpubeamsearch as primary
import cayleypy_native as compat
runtime = importlib.import_module("multigpubeamsearch._beam_runtime")
assert runtime.ABI_VERSION == 1
assert all(hasattr(runtime, n) for n in ("SearchSession", "ResidentWorkers", "FrontierCalibration"))
assert primary.NativeOptions is compat.NativeOptions
assert primary.NativeSession is compat.NativeSession
session = primary.NativeSession(engine="rust")
session.close()
from multigpubeamsearch.rust_control import extension, worker_executable
assert extension() is runtime
worker = pathlib.Path(primary.__file__).parent / "_bin" / ("beam-runner.exe" if sys.platform == "win32" else "beam-runner")
assert worker.is_file() and os.access(worker, os.X_OK)
assert worker_executable() == worker
probe = subprocess.run([str(worker), "--resident-protocol-info"], check=True, capture_output=True, text=True)
assert json.loads(probe.stdout) == {"protocol": "resident-search-v1"}
from multigpubeamsearch.bootstrap import prepare_sources
options = prepare_sources(primary.NativeOptions(cutlass_dir=pathlib.Path(sys.argv[1])))
assert options.source_dir == pathlib.Path(primary.__file__).parent / "_native_source"
if sys.platform.startswith("linux"):
    maps = open("/proc/self/maps", encoding="utf8").read()
    assert "libbeam_native.so" not in maps and "libbeam_native_libtorch.so" not in maps
print(json.dumps({"abi": runtime.ABI_VERSION, "module": runtime.__file__, "worker": str(worker),
                  "bundled_source": str(options.source_dir), "both_package_names": True,
                  "native_cuda_library_loaded": False}))
'''
    run([sys.executable, "-I", "-c", code, str(target)], log=out / "import.log", cwd=out, env=env)
    return dict(wheel=wheel.name, sha256=sha256(wheel.read_bytes()), extension=extensions[0],
                bundled_files=len(manifest), bundled_manifest_sha256=sha256(
                    json.dumps(manifest, sort_keys=True).encode()),
                both_package_names=True, native_cuda_library_loaded=False)


def build_wheel(project: Path, out: Path, env: dict[str, str]) -> Path:
    out.mkdir()
    wheels = out / "wheel"
    wheels.mkdir()
    run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "--no-index", "--wheel-dir", str(wheels), str(project)],
        log=out / "build.log", cwd=out, env=env)
    return one(wheels, "*.whl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    project = source / "integrations/cayleypy_native"
    env = dict(os.environ, PYO3_PYTHON=sys.executable, CARGO_NET_OFFLINE="true")

    checkout = out / "checkout"
    checkout_wheel = build_wheel(project, checkout, env)
    checkout_result = check_wheel(checkout_wheel, source, checkout, env)

    sdist_dir = out / "sdist"
    sdist_dir.mkdir()
    run([sys.executable, "-m", "build", "--sdist", "--no-isolation", "--outdir", str(sdist_dir), str(project)],
        log=out / "sdist.log", cwd=out, env=env)
    sdist = one(sdist_dir, "*.tar.gz")
    rebuilt = out / "rebuilt"
    rebuilt_wheel = build_wheel(sdist, rebuilt, env)
    rebuilt_result = check_wheel(rebuilt_wheel, source, rebuilt, env)
    if checkout_result["bundled_manifest_sha256"] != rebuilt_result["bundled_manifest_sha256"]:
        raise AssertionError("checkout and sdist wheels have different native source manifests")

    result = dict(checkout=checkout_result, sdist=dict(name=sdist.name, sha256=sha256(sdist.read_bytes())),
                  rebuilt=rebuilt_result)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("RUST_DISTRIBUTION_CHECK_OK", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
