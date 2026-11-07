"""Private Kaggle validation entrypoint. Payload is filled by prepare_kaggle.py."""
from __future__ import annotations

# Exact public source snapshots; only adapter/validation files are embedded below.
NATIVE_REVISION = "a1db0e6d9bb5458c8a842b37dfa99572d3025667"
CAYLEYPY_REVISION = "28f3841b34009ea8d51bb36eece3dd0be757a145"
CUTLASS_REVISION = "afa1772203677c5118fcd82537a9c8fefbcc7008"
DEVICES = (0, 1)
PAYLOAD_B64 = "@@PAYLOAD_B64@@"
PAYLOAD_SHA256 = "@@PAYLOAD_SHA256@@"

import base64
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile

OUT = Path("/kaggle/working/cayleypy_native_validation")
TEMP = Path(tempfile.mkdtemp(prefix="cayleypy-native-validation-", dir="/tmp"))
OUT.mkdir(parents=True, exist_ok=True)
PHASE = "initializing"
STOP = threading.Event()
SUMMARY = {"native_revision": NATIVE_REVISION, "cayleypy_revision": CAYLEYPY_REVISION,
           "cutlass_revision": CUTLASS_REVISION, "scope": "real two-T4 adapter validation; no publication",
           "temporary_directory": str(TEMP), "commands": []}


def save_summary():
    (OUT / "job_summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str) + "\n")


def command(name, args, *, cwd=None, timeout=600, env=None):
    global PHASE
    PHASE = name
    print("PHASE", name, flush=True)
    begin = time.perf_counter()
    logfile = OUT / f"{name}.log"
    with logfile.open("w") as log:
        completed = subprocess.run([str(x) for x in args], cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, timeout=timeout, check=False, shell=False)
    SUMMARY["commands"].append({"name": name, "command": [str(x) for x in args],
        "return_code": completed.returncode, "wall_seconds": time.perf_counter() - begin, "log": str(logfile)})
    save_summary()
    print(logfile.read_text(errors="replace")[-4000:], flush=True)
    if completed.returncode:
        raise RuntimeError(f"{name} failed rc={completed.returncode}; see {logfile}")


def checkout(name, url, revision):
    directory = TEMP / name
    directory.mkdir()
    command(name + "-init", ["git", "init", str(directory)])
    command(name + "-fetch", ["git", "-C", str(directory), "fetch", "--depth", "1", url, revision])
    command(name + "-checkout", ["git", "-C", str(directory), "checkout", "--detach", "FETCH_HEAD"])
    actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise RuntimeError(f"{name} revision mismatch: {actual}")
    return directory


def unpack_payload():
    blob = base64.b64decode(PAYLOAD_B64, validate=True)
    if hashlib.sha256(blob).hexdigest() != PAYLOAD_SHA256:
        raise RuntimeError("adapter payload checksum mismatch")
    destination = TEMP / "adapter"
    destination.mkdir()
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for member in archive.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts or member.is_dir():
                raise RuntimeError(f"unsafe payload member: {member.filename}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
    manifest = json.loads((destination / "payload_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"payload member checksum mismatch: {name}")
    SUMMARY["payload"] = manifest
    return destination


def heartbeat():
    while not STOP.wait(20):
        print("HEARTBEAT phase=" + PHASE, flush=True)
        logs = [path for name in ("acceptance", "real-acceptance") for path in (TEMP / name).rglob("*.log")]
        if logs:
            latest = max(logs, key=lambda p: p.stat().st_mtime)
            with latest.open("rb") as stream:
                stream.seek(max(0, latest.stat().st_size - 1200))
                print("LATEST_LOG", str(latest), stream.read().decode(errors="replace"), flush=True)


def preserve_evidence():
    # Preserve small diagnostic artifacts, not native build trees or history arenas.
    for name in ("acceptance", "real-acceptance"):
        source = TEMP / name
        for path in source.rglob("*"):
            if not path.is_file() or path.suffix not in (".json", ".jsonl", ".log", ".csv", ".txt"):
                continue
            if "builds" in path.relative_to(source).parts or path.stat().st_size > 32 * 1024**2:
                continue
            destination = OUT / name / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def main():
    global PHASE
    start = time.perf_counter()
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        import torch
        gpu = [{"index": i, "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
                "total_memory": torch.cuda.get_device_properties(i).total_memory}
               for i in range(torch.cuda.device_count())]
        SUMMARY.update(python=sys.version, torch=torch.__version__, torch_cuda=torch.version.cuda, gpus=gpu)
        print("HARDWARE", json.dumps(gpu), flush=True)
        if len(gpu) != 2 or any("T4" not in item["name"] or tuple(item["capability"]) != (7, 5) for item in gpu):
            raise RuntimeError("this acceptance requires exactly two real Tesla T4 GPUs")
        command("nvidia-smi", ["nvidia-smi"])
        for i in DEVICES:
            value = torch.ones(1, device=f"cuda:{i}") + 1
            assert value.item() == 2
        del value
        for i in DEVICES:
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        save_summary()
        adapter = unpack_payload()
        native = checkout("native", "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git", NATIVE_REVISION)
        upstream = checkout("cayleypy", "https://github.com/cayleypy/cayleypy.git", CAYLEYPY_REVISION)
        cutlass = checkout("cutlass", "https://github.com/NVIDIA/cutlass.git", CUTLASS_REVISION)
        # Use existing CUDA/compiler environment, including Kaggle's known supported GCC when present.
        if Path("/usr/bin/g++-11").is_file():
            os.environ["CXX"] = "/usr/bin/g++-11"
            os.environ["CUDAHOSTCXX"] = "/usr/bin/g++-11"
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        command("install-cayleypy", [sys.executable, "-m", "pip", "install", str(upstream)], timeout=600)
        sys.path[:0] = [str(adapter), str(upstream)]
        os.environ["PYTHONPATH"] = os.pathsep.join((str(adapter), str(upstream)))
        os.environ["CAYLEYPY_NATIVE_TEST_SOURCE_DIR"] = str(native)
        SUMMARY["packages"] = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "cayleypy", "pytest")}
        command("cpu-contract-tests", [sys.executable, "-B", "-m", "pytest", str(adapter / "tests"),
                                      str(adapter / "validation" / "test_smoke_cases.py"),
                                      str(adapter / "validation" / "test_real_cases.py"), "-q", "-p", "no:cacheprovider"],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"), timeout=180)
        from validation.smoke_cases import run_acceptance
        PHASE = "native-acceptance"
        report = run_acceptance(source_dir=native, cutlass_dir=cutlass, output_dir=TEMP / "acceptance",
                                devices=DEVICES, timeout_seconds=120, build_timeout_seconds=900)
        SUMMARY["acceptance"] = report
        SUMMARY["status"] = "passed" if report.get("status") == "passed" else "failed"
        if SUMMARY["status"] != "passed":
            raise RuntimeError("native acceptance did not pass; inspect report")
        from validation.prepared_acceptance import run_prepared_acceptance
        PHASE = "prepared-model-acceptance"
        prepared_report = run_prepared_acceptance(source_dir=native, cutlass_dir=cutlass,
            output_dir=TEMP / "acceptance" / "prepared-api", cache_dir=TEMP / "acceptance" / "cache",
            devices=DEVICES)
        SUMMARY["prepared_acceptance"] = prepared_report
        if prepared_report.get("status") != "passed":
            raise RuntimeError("prepared API acceptance did not pass; inspect report")
        from validation.real_cases import run_real_acceptance
        PHASE = "real-tetraminx-acceptance"
        real_report = run_real_acceptance(source_dir=native, cutlass_dir=cutlass,
            input_root=Path("/kaggle/input"), output_dir=TEMP / "real-acceptance",
            devices=DEVICES, timeout_seconds=120, build_timeout_seconds=900)
        SUMMARY["real_acceptance"] = real_report
        if real_report.get("status") != "passed":
            raise RuntimeError("real Tetraminx acceptance did not pass; inspect report")
        for name, directory in (("native", native), ("cayleypy", upstream)):
            dirty = subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain"], text=True).strip()
            SUMMARY[name + "_dirty"] = dirty
            if dirty:
                raise RuntimeError(f"{name} source changed during validation")
        return 0
    except BaseException as error:
        SUMMARY.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(SUMMARY["traceback"], flush=True)
        return 1
    finally:
        STOP.set()
        preserve_evidence()
        SUMMARY["wall_seconds"] = time.perf_counter() - start
        save_summary()
        print("FINAL_SUMMARY", json.dumps(SUMMARY, default=str), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
