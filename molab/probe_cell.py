from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path


REPO_URL = os.environ.get(
    "MOLAB_BEAM_REPO_URL",
    "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git",
)
REPO_BRANCH = os.environ.get("MOLAB_BEAM_REPO_BRANCH", "main")
WORK_ROOT = Path(os.environ.get("MOLAB_BEAM_WORK_ROOT", "/tmp/molab_beam"))
REPO_DIR = WORK_ROOT / "repo"
CUTLASS_DIR = WORK_ROOT / "cutlass"
BUILD_DIR = WORK_ROOT / "build"
RUN_PROJECT_BUILD = os.environ.get("MOLAB_RUN_PROJECT_BUILD", "1") == "1"
RUN_CPU_TESTS = os.environ.get("MOLAB_RUN_CPU_TESTS", "1") == "1"


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 120) -> dict:
    started = time.time()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        output = completed.stdout
        code = completed.returncode
    except Exception as exc:
        output = f"{type(exc).__name__}: {exc}"
        code = -999
    return {
        "cmd": cmd,
        "cwd": str(cwd) if cwd else None,
        "returncode": code,
        "seconds": round(time.time() - started, 3),
        "output": output[-12000:],
    }


def print_section(title: str, body) -> None:
    print(f"\n## {title}")
    if isinstance(body, str):
        print(body)
    else:
        print(json.dumps(body, indent=2, sort_keys=True))


def read_text_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def cgroup_quota() -> dict:
    return {
        "cpu_max": read_text_file("/sys/fs/cgroup/cpu.max"),
        "cpuset_cpus": read_text_file("/sys/fs/cgroup/cpuset.cpus"),
        "cpuset_cpus_effective": read_text_file(
            "/sys/fs/cgroup/cpuset.cpus.effective"
        ),
        "memory_max": read_text_file("/sys/fs/cgroup/memory.max"),
        "memory_current": read_text_file("/sys/fs/cgroup/memory.current"),
    }


def detect_cuda_arch_from_nvidia_smi() -> str | None:
    result = run([
        "nvidia-smi",
        "--query-gpu=compute_cap",
        "--format=csv,noheader",
    ])
    if result["returncode"] != 0:
        return None
    caps = []
    for line in result["output"].splitlines():
        cap = line.strip().replace(".", "")
        if cap.isdigit():
            caps.append(cap)
    return ";".join(sorted(set(caps))) if caps else None


def torch_gpu_task() -> dict:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    info = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    devices = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        devices.append({
            "index": idx,
            "name": props.name,
            "total_memory": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
            "major": props.major,
            "minor": props.minor,
        })
    info["devices"] = devices
    if not info["available"]:
        return info

    torch.cuda.set_device(0)
    n = int(os.environ.get("MOLAB_TORCH_MATMUL_N", "4096"))
    repeats = int(os.environ.get("MOLAB_TORCH_MATMUL_REPEATS", "5"))
    a = torch.randn((n, n), device="cuda", dtype=torch.float16)
    b = torch.randn((n, n), device="cuda", dtype=torch.float16)
    c = a @ b
    torch.cuda.synchronize()
    started = time.time()
    for _ in range(repeats):
        c = a @ b
    torch.cuda.synchronize()
    seconds = time.time() - started
    flops = 2.0 * n * n * n * repeats
    info["matmul"] = {
        "n": n,
        "repeats": repeats,
        "seconds": seconds,
        "tflops": flops / seconds / 1e12,
        "checksum": float(c[0, 0].detach().cpu()),
    }
    return info


def clone_cutlass_if_needed() -> dict:
    if (CUTLASS_DIR / "include" / "cutlass").exists():
        return {"status": "already_present", "path": str(CUTLASS_DIR)}
    CUTLASS_DIR.parent.mkdir(parents=True, exist_ok=True)
    return run([
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/NVIDIA/cutlass.git",
        str(CUTLASS_DIR),
    ], timeout=600)


def clone_repo_if_needed() -> dict:
    if (REPO_DIR / ".git").exists():
        return {"status": "already_present", "path": str(REPO_DIR)}
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    return run([
        "git",
        "clone",
        "--branch",
        REPO_BRANCH,
        "--depth",
        "1",
        REPO_URL,
        str(REPO_DIR),
    ], timeout=600)


def patch_cmake_arch(arch: str | None) -> dict:
    if not arch:
        return {"status": "skipped", "reason": "cuda arch unknown"}
    path = REPO_DIR / "CMakeLists.txt"
    text = path.read_text(encoding="utf-8")
    old = "set(BEAM_CUDA_ARCHITECTURES 75 86)"
    new = f"set(BEAM_CUDA_ARCHITECTURES {arch.replace(';', ' ')})"
    if old not in text:
        return {"status": "skipped", "reason": "expected CMake arch line not found"}
    path.write_text(text.replace(old, new), encoding="utf-8")
    return {"status": "patched", "arch": arch, "line": new}


def project_sanity_run() -> list[dict]:
    results: list[dict] = []
    if not RUN_PROJECT_BUILD:
        return [{"status": "skipped", "reason": "MOLAB_RUN_PROJECT_BUILD != 1"}]
    if shutil.which("git") is None:
        return [{"status": "skipped", "reason": "git missing"}]
    if shutil.which("cmake") is None:
        return [{"status": "skipped", "reason": "cmake missing"}]
    if shutil.which("nvcc") is None:
        return [{"status": "skipped", "reason": "nvcc missing"}]

    results.append({"clone_repo": clone_repo_if_needed()})
    results.append({"clone_cutlass": clone_cutlass_if_needed()})
    arch = detect_cuda_arch_from_nvidia_smi()
    results.append({"cmake_arch": patch_cmake_arch(arch)})
    generator = "Ninja" if shutil.which("ninja") else "Unix Makefiles"
    configure = [
        "cmake",
        "-S",
        str(REPO_DIR),
        "-B",
        str(BUILD_DIR),
        "-G",
        generator,
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCUTLASS_DIR={CUTLASS_DIR}",
    ]
    results.append({"configure": run(configure, timeout=300)})
    if results[-1]["configure"]["returncode"] != 0:
        return results

    build_target = os.environ.get("MOLAB_BUILD_TARGET", "stream1_cuda_tests")
    results.append({"build": run([
        "cmake",
        "--build",
        str(BUILD_DIR),
        "--target",
        build_target,
        "-j",
        str(os.cpu_count() or 2),
    ], timeout=900)})
    if results[-1]["build"]["returncode"] != 0:
        return results

    exe = BUILD_DIR / build_target
    if not exe.exists():
        exe = BUILD_DIR / (build_target + ".exe")
    results.append({"run_target": run([str(exe)], cwd=REPO_DIR, timeout=300)})
    return results


def project_cpu_contract_run() -> list[dict]:
    results: list[dict] = []
    if not RUN_CPU_TESTS:
        return [{"status": "skipped", "reason": "MOLAB_RUN_CPU_TESTS != 1"}]
    if shutil.which("git") is None:
        return [{"status": "skipped", "reason": "git missing"}]
    if shutil.which("g++") is None:
        return [{"status": "skipped", "reason": "g++ missing"}]

    results.append({"clone_repo": clone_repo_if_needed()})
    sources = [
        "src/config.cpp",
        "src/state.cpp",
        "src/hash.cpp",
        "src/history.cpp",
        "src/stream3.cpp",
        "src/stream4.cpp",
        "src/frontier_cpu.cpp",
    ]
    source_args = [str(REPO_DIR / source) for source in sources]
    tests = [
        ("contract_tests", "tests/contract_tests.cpp"),
        ("history_tests", "tests/history_tests.cpp"),
    ]
    for name, test_source in tests:
        exe = WORK_ROOT / name
        compile_cmd = [
            "g++",
            "-std=c++20",
            "-O2",
            "-I",
            str(REPO_DIR / "src"),
            str(REPO_DIR / test_source),
            *source_args,
            "-o",
            str(exe),
        ]
        compile_result = run(compile_cmd, cwd=REPO_DIR, timeout=300)
        results.append({f"{name}_compile": compile_result})
        if compile_result["returncode"] != 0:
            continue
        run_result = run([str(exe)], cwd=REPO_DIR, timeout=120)
        results.append({f"{name}_run": run_result})
        report_files = sorted((REPO_DIR / "test_results").glob(f"{name}_*.md"))
        if report_files:
            results.append({
                f"{name}_report": report_files[-1].read_text(
                    encoding="utf-8"
                )
            })
    results.append({"git_head": run([
        "bash",
        "-lc",
        "git rev-parse --short HEAD && git log -1 --oneline",
    ], cwd=REPO_DIR)})
    return results


print_section("Python", {
    "executable": sys.executable,
    "version": sys.version,
    "platform": platform.platform(),
    "cwd": os.getcwd(),
})

for name, cmd in [
    ("uname", ["uname", "-a"]),
    ("cpu", ["bash", "-lc", "lscpu | sed -n '1,20p'"]),
    ("memory", ["bash", "-lc", "free -h"]),
    ("cgroup_quota", ["bash", "-lc", "cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max 2>/dev/null || true"]),
    ("disk", ["bash", "-lc", "df -h . /tmp || true"]),
    ("nvidia_smi_L", ["nvidia-smi", "-L"]),
    ("nvidia_smi_query", [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version",
        "--format=csv,noheader",
    ]),
    ("nvcc", ["nvcc", "--version"]),
    ("cmake", ["cmake", "--version"]),
    ("ninja", ["ninja", "--version"]),
    ("git", ["git", "--version"]),
    ("nccl", ["bash", "-lc", "ldconfig -p | grep -i nccl || true"]),
]:
    print_section(name, run(cmd))

print_section("Cgroup Quota", cgroup_quota())
print_section("Torch GPU Task", torch_gpu_task())
print_section("Project CPU Contract Run", project_cpu_contract_run())
print_section("Project Sanity Run", project_sanity_run())
