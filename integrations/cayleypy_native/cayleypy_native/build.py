"""Shape-specific builds and verified native-build.json binary sidecars.

An existing runner is opt-in: its sibling ``native-build.json`` must use schema
version 1 and match shape, backend, CUDA architectures and SHA256 of the binary.
Build metadata records working-tree source bytes, CUTLASS bytes and toolchain;
neither Git HEAD nor a successful old CMake cache is sufficient provenance.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from typing import Any

from .errors import NativeBackendError, NativeUnavailable


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest(source: Path, *, cutlass: bool = False) -> str:
    """Hash source content, including dirty files; ignore build/output trees."""
    roots = [source / "include"] if cutlass else [source / p for p in ("src", "cuda", "tools", "cmake")]
    if cutlass:
        roots.append(source / "tools" / "util" / "include")
    files = [source / "CMakeLists.txt"] if not cutlass else []
    extensions = {".h", ".hpp", ".cuh", ".c", ".cpp", ".cu", ".cmake", ".py", ".in"}
    for root in roots:
        if root.is_dir():
            files.extend(p for p in root.rglob("*") if p.is_file() and p.suffix in extensions)
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        if not path.is_file():
            raise NativeUnavailable(f"native source file is missing: {path}")
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def shape_contract(contract) -> dict[str, int]:
    alignment = 16
    storage = ((contract.state_len + 4 + alignment - 1) // alignment) * alignment
    return {"state_len": contract.state_len, "storage_len": storage,
            "move_count": contract.move_count, "alignment": alignment}


def build_key(specification: dict[str, Any]) -> str:
    encoded = json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def prerequisites(options) -> dict[str, str]:
    if options.runner_path is not None:
        runner = options.runner_path
        if not runner.is_file() or not os.access(runner, os.X_OK):
            raise NativeUnavailable(f"native runner is not executable: {runner}")
        if not (runner.parent / "native-build.json").is_file():
            raise NativeUnavailable("existing native runner requires sibling native-build.json")
        return {}
    source, cutlass = options.source_dir, options.cutlass_dir
    if source is None or not (source / "CMakeLists.txt").is_file() or not (source / "tools/production_runner.cu").is_file():
        raise NativeUnavailable("building native requires an explicit valid source_dir")
    if cutlass is None or not (cutlass / "include/cutlass/cutlass.h").is_file():
        raise NativeUnavailable("building native requires an explicit valid cutlass_dir")
    result = {}
    for name, executable in (("cmake", "cmake"), ("nvcc", os.environ.get("CUDACXX", "nvcc")),
                             ("cxx", os.environ.get("CXX", "c++"))):
        resolved = shutil.which(executable)
        if resolved is None:
            raise NativeUnavailable(f"native build prerequisite is missing: {name}")
        result[name] = str(Path(resolved).resolve())
    return result


def toolchain_identity(programs: dict[str, str]) -> dict[str, str]:
    result = {}
    for name, executable in programs.items():
        try:
            completed = subprocess.run([executable, "--version"], capture_output=True, text=True,
                                       timeout=15, check=False, shell=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise NativeUnavailable(f"cannot inspect native toolchain {name}: {exc}") from exc
        if completed.returncode:
            raise NativeUnavailable(f"cannot inspect native toolchain {name}: rc={completed.returncode}")
        result[name] = executable + "\n" + completed.stdout + completed.stderr
    return result


def _nccl_locations() -> list[tuple[Path, Path]]:
    """Installed package and conventional Linux header/library directories."""
    locations = []
    for name in ("NCCL_ROOT", "NCCL_HOME"):
        if os.environ.get(name):
            root = Path(os.environ[name]).expanduser()
            locations.extend((root / "include", root / subdir) for subdir in ("lib", "lib64"))
    # PyTorch's nvidia-nccl-cu12 wheel is beside torch in site-packages. It may
    # provide libnccl.so.2 without an unversioned linker symlink.
    python_roots = [Path(path) for path in sys.path if path]
    python_roots.extend(Path(path) for key, path in sysconfig.get_paths().items() if key in ("purelib", "platlib"))
    for parent in dict.fromkeys(python_roots):
        root = parent / "nvidia" / "nccl"
        locations.append((root / "include", root / "lib"))
    for prefix in (Path("/usr/local"), Path("/usr"), Path("/usr/local/cuda")):
        locations.extend((prefix / "include", prefix / subdir) for subdir in
                         ("lib", "lib64", "lib/x86_64-linux-gnu", "lib/aarch64-linux-gnu"))
    return list(dict.fromkeys(locations))


def _nccl_identity(include: Path, library: Path) -> dict[str, str]:
    try:
        include, library = include.resolve(), library.resolve()
        return {"include_dir": str(include), "library": str(library),
                "header_sha256": file_sha256(include / "nccl.h"),
                "library_sha256": file_sha256(library)}
    except OSError as exc:
        raise NativeUnavailable(f"cannot read installed NCCL dependency: {exc}") from exc


def discover_nccl() -> dict[str, str]:
    """Select an existing header/library pair; never install or invoke a library."""
    for include, libdir in _nccl_locations():
        if not (include / "nccl.h").is_file():
            continue
        choices = [libdir / "libnccl.so", libdir / "libnccl.so.2"]
        choices.extend(sorted(libdir.glob("libnccl.so.*")))
        for library in dict.fromkeys(choices):
            if library.is_file():
                return _nccl_identity(include, library)
    raise NativeUnavailable("native build requires installed NCCL headers and libnccl.so (system or nvidia-nccl Python package)")


def validate_runner(runner: Path, contract, backend: str, architectures: tuple[int, ...]) -> dict[str, Any]:
    try:
        metadata = json.loads((runner.parent / "native-build.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise NativeUnavailable(f"native runner build metadata is missing or invalid: {exc}") from exc
    expected = {"schema_version": 1, "shape": shape_contract(contract), "backend": backend}
    if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in expected.items()):
        raise NativeUnavailable("native runner metadata does not match requested shape/backend")
    arch = metadata.get("cuda_architectures")
    if not isinstance(arch, list) or any(type(v) is not int for v in arch) or not set(architectures).issubset(arch):
        raise NativeUnavailable("native runner does not contain the requested CUDA architectures")
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise NativeUnavailable(f"native runner is not executable: {runner}")
    if metadata.get("binary_sha256") != file_sha256(runner):
        raise NativeBackendError("native runner SHA256 differs from native-build.json")
    return metadata


def configure_command(source: Path, build_dir: Path, cutlass: Path, contract,
                      backend: str, architectures: tuple[int, ...], programs: dict[str, str],
                      *, nccl: dict[str, str] | None = None) -> list[str]:
    shape = shape_contract(contract)
    command = [programs["cmake"], "-S", str(source), "-B", str(build_dir),
               "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_CXX_COMPILER={programs['cxx']}",
               f"-DCMAKE_CUDA_COMPILER={programs['nvcc']}", f"-DCUTLASS_DIR={cutlass}",
               f"-DBEAM_STATE_LOGICAL_BYTES={shape['state_len']}",
               f"-DBEAM_STATE_ALIGNMENT={shape['alignment']}",
               f"-DBEAM_MOVE_COUNT={shape['move_count']}",
               "-DBEAM_CUDA_ARCHITECTURES=" + ";".join(map(str, architectures)),
               "-DBEAM_ENABLE_DEBUG=ON", "-DBEAM_ENABLE_DEBUG_LOGS=ON",
               "-DBEAM_ENABLE_DEPTH_LOGS=OFF", "-DBEAM_DEBUG_STREAM_TIMING=OFF",
               "-DBEAM_DEBUG_INFERENCE_TRACE=OFF", "-DBEAM_DEBUG_PATH_TRACE=OFF"]
    if nccl is not None:
        command += [f"-DNCCL_INCLUDE_DIR={nccl['include_dir']}", f"-DNCCL_LIBRARY={nccl['library']}",
                    f"-DCMAKE_BUILD_RPATH={Path(nccl['library']).parent}"]
    if backend == "piece_transformer":
        import torch
        command += ["-DBEAM_ENABLE_LIBTORCH_STREAM1=ON", f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}"]
    elif backend != "mlp":
        raise NativeUnavailable(f"unsupported native build backend: {backend}")
    return command


def ensure_runner(contract, model, options, architectures: tuple[int, ...], run_dir: Path) -> tuple[Path, dict[str, Any]]:
    if options.runner_path is not None:
        return options.runner_path, validate_runner(options.runner_path, contract, model.backend, architectures)
    programs = prerequisites(options)
    source, cutlass = options.source_dir, options.cutlass_dir
    cmake_text = (source / "CMakeLists.txt").read_text(encoding="utf-8")
    target = "production_runner" if model.backend == "mlp" else "production_runner_libtorch_stream1"
    if target not in cmake_text or (model.backend == "piece_transformer" and "BEAM_ENABLE_LIBTORCH_STREAM1" not in cmake_text):
        raise NativeUnavailable(f"configured native source does not expose compatible target {target}")
    nccl = discover_nccl()
    specification = {"schema_version": 1, "shape": shape_contract(contract), "backend": model.backend,
                     "cuda_architectures": list(architectures), "source_digest": source_digest(source),
                     "cutlass_digest": source_digest(cutlass, cutlass=True),
                     "toolchain": toolchain_identity(programs), "target": target, "nccl": nccl,
                     "build_mode": "release-debug-logs-v1"}
    if model.backend == "piece_transformer":
        import torch
        specification["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                                  "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                                  "cmake_prefix": str(torch.utils.cmake_prefix_path)}
    key = build_key(specification)
    build_dir = options.cache_dir / "builds" / key
    runner = build_dir / target
    if (build_dir / "native-build.json").is_file():
        return runner, validate_runner(runner, contract, model.backend, architectures)
    build_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = build_dir.parent / (key + ".lock")
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise NativeBackendError(f"native build is already active or has a retained lock: {lock}") from exc
    try:
        from .backend import child_environment, run_process
        build_dir.mkdir(parents=True, exist_ok=True)
        env = child_environment((), run_dir)
        env["CXX"], env["CUDACXX"] = programs["cxx"], programs["nvcc"]
        run_process(configure_command(source, build_dir, cutlass, contract, model.backend, architectures, programs, nccl=nccl),
                    cwd=run_dir, env=env, timeout=options.build_timeout_seconds, log_path=run_dir / "cmake-configure.log")
        run_process([programs["cmake"], "--build", str(build_dir), "--target", target,
                     "--parallel", str(options.build_jobs)], cwd=run_dir, env=env,
                    timeout=options.build_timeout_seconds, log_path=run_dir / "cmake-build.log")
        if specification["source_digest"] != source_digest(source) or specification["cutlass_digest"] != source_digest(cutlass, cutlass=True):
            raise NativeBackendError("source changed during native build; refusing ambiguous binary")
        try:
            current_nccl = _nccl_identity(Path(nccl["include_dir"]), Path(nccl["library"]))
        except NativeUnavailable as exc:
            raise NativeBackendError("NCCL became unavailable during native build") from exc
        if nccl != current_nccl:
            raise NativeBackendError("NCCL changed during native build; refusing ambiguous binary")
        if not runner.is_file():
            raise NativeBackendError("native build completed without producing the requested executable")
        metadata = dict(specification, build_key=key, binary_sha256=file_sha256(runner))
        temporary = build_dir / "native-build.json.tmp"
        temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temporary.replace(build_dir / "native-build.json")
        return runner, validate_runner(runner, contract, model.backend, architectures)
    finally:
        lock.rmdir()
