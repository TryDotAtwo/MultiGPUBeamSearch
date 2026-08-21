"""Molab CUDA build-tool bootstrap helpers."""
from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
from typing import Callable, Mapping


_CUDA_PACKAGES = {
    "13.0": (
        "nvidia-cuda-nvcc==13.0.88",
        "nvidia-cuda-crt==13.0.88",
        "nvidia-nvvm==13.0.88",
        "nvidia-cuda-runtime==13.0.96",
        "nvidia-cuda-cccl==13.0.85",
        "nvidia-nvtx==13.0.85",
        "nvidia-curand==10.4.0.35",
    ),
}


def cuda_packages_for(cuda_version: str) -> tuple[str, ...]:
    try:
        return _CUDA_PACKAGES[cuda_version]
    except KeyError as error:
        raise ValueError(f"unsupported Molab CUDA toolkit: {cuda_version}") from error


def materialize_cuda_overlay(package_root: Path, overlay: Path) -> Path:
    if overlay.exists():
        shutil.rmtree(overlay)
    (overlay / "bin").mkdir(parents=True)
    (overlay / "lib").mkdir(parents=True)
    (overlay / "lib64").mkdir(parents=True)
    shutil.copytree(package_root / "bin", overlay / "bin", dirs_exist_ok=True, symlinks=True)
    for name in ("include", "nvvm"):
        shutil.copytree(package_root / name, overlay / name, symlinks=True)
    for source in (package_root / "lib").iterdir():
        if source.is_file():
            shutil.copy2(source, overlay / "lib" / source.name)
            shutil.copy2(source, overlay / "lib64" / source.name)
    for lib_dir in (overlay / "lib", overlay / "lib64"):
        versioned = lib_dir / "libcudart.so.13"
        unversioned = lib_dir / "libcudart.so"
        if versioned.exists() and not unversioned.exists():
            shutil.copy2(versioned, unversioned)
    return overlay


def _run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _package_root_is_complete(package_root: Path) -> bool:
    return all(
        path.exists()
        for path in (
            package_root / "bin" / "nvcc",
            package_root / "bin" / "crt" / "link.stub",
            package_root / "include" / "nv" / "target",
            package_root / "include" / "nvtx3" / "nvToolsExt.h",
            package_root / "include" / "curand_kernel.h",
            package_root / "nvvm" / "libdevice",
            package_root / "lib" / "libcudart.so.13",
            package_root / "lib" / "libcurand.so.10",
        )
    )


def prepare_molab_build_environment(
    workspace: Path,
    cuda_version: str,
    python_executable: str,
    *,
    base_environment: Mapping[str, str] | None = None,
    command_runner: Callable[[list[str]], None] = _run_checked,
    executable_finder: Callable[[str], str | None] = shutil.which,
) -> dict[str, str]:
    environment = dict(os.environ if base_environment is None else base_environment)
    user_bin = Path.home() / ".local" / "bin"
    if executable_finder("cmake") is None or executable_finder("ninja") is None:
        command_runner([
            python_executable,
            "-m",
            "pip",
            "install",
            "--user",
            "cmake==4.4.2",
            "ninja==1.13.0",
        ])

    toolchain = workspace / ".molab_toolchain"
    packages = toolchain / "packages"
    package_root = packages / "nvidia" / "cu13"
    if not _package_root_is_complete(package_root):
        if packages.exists():
            shutil.rmtree(packages)
        command_runner([
            python_executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(packages),
            "--no-deps",
            *cuda_packages_for(cuda_version),
        ])
    if not _package_root_is_complete(package_root):
        raise RuntimeError("Molab CUDA wheel install did not materialize a complete toolkit")

    overlay = materialize_cuda_overlay(package_root, toolchain / "cuda")
    build = toolchain / "build"
    path = environment.get("PATH", "")
    libraries = environment.get("LD_LIBRARY_PATH", "")
    environment.update({
        "PATH": f"{user_bin}:{overlay / 'bin'}:{path}",
        "LD_LIBRARY_PATH": f"{overlay / 'lib64'}:{libraries}",
        "CUDACXX": str(overlay / "bin" / "nvcc"),
        "CUDA_HOME": str(overlay),
        "CUDAToolkit_ROOT": str(overlay),
        "CAYLEYPY_BUILD_DIR": str(build),
        "CAYLEYPY_BUILD_JOBS": "1",
    })
    return environment
