from __future__ import annotations

from pathlib import Path

import pytest

from tools.molab_toolchain import (
    cuda_packages_for,
    materialize_cuda_overlay,
    prepare_molab_build_environment,
)


def test_cuda_13_0_uses_one_matching_wheel_family() -> None:
    assert cuda_packages_for("13.0") == (
        "nvidia-cuda-nvcc==13.0.88",
        "nvidia-cuda-crt==13.0.88",
        "nvidia-nvvm==13.0.88",
        "nvidia-cuda-runtime==13.0.96",
        "nvidia-cuda-cccl==13.0.85",
    )


def test_unknown_cuda_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported Molab CUDA toolkit"):
        cuda_packages_for("12.1")


def test_materialize_cuda_overlay_adds_unversioned_cudart(tmp_path: Path) -> None:
    package_root = tmp_path / "packages" / "nvidia" / "cu13"
    (package_root / "bin" / "crt").mkdir(parents=True)
    (package_root / "include" / "nv").mkdir(parents=True)
    (package_root / "nvvm" / "libdevice").mkdir(parents=True)
    (package_root / "lib").mkdir(parents=True)
    (package_root / "bin" / "nvcc").write_text("nvcc", encoding="utf-8")
    (package_root / "bin" / "crt" / "link.stub").write_text("stub", encoding="utf-8")
    (package_root / "include" / "nv" / "target").write_text("target", encoding="utf-8")
    (package_root / "lib" / "libcudart.so.13").write_text("runtime", encoding="utf-8")

    overlay = materialize_cuda_overlay(package_root, tmp_path / "overlay")

    assert (overlay / "bin" / "crt" / "link.stub").is_file()
    assert (overlay / "include" / "nv" / "target").is_file()
    assert (overlay / "lib" / "libcudart.so").read_text() == "runtime"
    assert (overlay / "lib64" / "libcudart.so").read_text() == "runtime"


def test_prepare_environment_reuses_complete_cached_toolkit(tmp_path: Path) -> None:
    package_root = tmp_path / ".molab_toolchain" / "packages" / "nvidia" / "cu13"
    (package_root / "bin" / "crt").mkdir(parents=True)
    (package_root / "include" / "nv").mkdir(parents=True)
    (package_root / "nvvm" / "libdevice").mkdir(parents=True)
    (package_root / "lib").mkdir(parents=True)
    (package_root / "bin" / "nvcc").write_text("nvcc", encoding="utf-8")
    (package_root / "bin" / "crt" / "link.stub").write_text("stub", encoding="utf-8")
    (package_root / "include" / "nv" / "target").write_text("target", encoding="utf-8")
    (package_root / "lib" / "libcudart.so.13").write_text("runtime", encoding="utf-8")
    calls: list[list[str]] = []

    environment = prepare_molab_build_environment(
        tmp_path,
        "13.0",
        "/usr/bin/python",
        base_environment={"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/base/lib"},
        command_runner=lambda command: calls.append(command),
        executable_finder=lambda name: f"/usr/bin/{name}",
    )

    assert calls == []
    assert Path(environment["CUDACXX"]) == tmp_path / ".molab_toolchain" / "cuda" / "bin" / "nvcc"
    assert Path(environment["CUDAToolkit_ROOT"]) == tmp_path / ".molab_toolchain" / "cuda"
    assert Path(environment["CAYLEYPY_BUILD_DIR"]) == tmp_path / ".molab_toolchain" / "build"
    assert environment["LD_LIBRARY_PATH"].endswith(":/base/lib")


def test_prepare_environment_installs_missing_tools_and_cuda_atomically(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        calls.append(command)
        if "--target" in command:
            target = Path(command[command.index("--target") + 1]) / "nvidia" / "cu13"
            (target / "bin" / "crt").mkdir(parents=True)
            (target / "include" / "nv").mkdir(parents=True)
            (target / "nvvm" / "libdevice").mkdir(parents=True)
            (target / "lib").mkdir(parents=True)
            (target / "bin" / "nvcc").write_text("nvcc", encoding="utf-8")
            (target / "bin" / "crt" / "link.stub").write_text("stub", encoding="utf-8")
            (target / "include" / "nv" / "target").write_text("target", encoding="utf-8")
            (target / "lib" / "libcudart.so.13").write_text("runtime", encoding="utf-8")

    environment = prepare_molab_build_environment(
        tmp_path,
        "13.0",
        "/usr/bin/python",
        base_environment={"PATH": "/usr/bin"},
        command_runner=fake_run,
        executable_finder=lambda name: None,
    )

    assert calls[0][:5] == [
        "/usr/bin/python", "-m", "pip", "install", "--user",
    ]
    assert calls[1][:7] == [
        "/usr/bin/python", "-m", "pip", "install", "--target",
        str(tmp_path / ".molab_toolchain" / "packages"), "--no-deps",
    ]
    assert calls[1][7:] == list(cuda_packages_for("13.0"))
    assert environment["PATH"].startswith(str(Path.home() / ".local" / "bin"))
